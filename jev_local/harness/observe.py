"""Read-only Accessibility snapshots of the frontmost app (docs/research/SPEC.md §4.5).

One walk = one `AXUIElementCopyMultipleAttributeValues` call per node over the focused window,
plus the top-level menu bar when the walk is cheap. The walk turns actionable nodes into
`Element`s (humanized role, short label, " (k of n)" for duplicates) and keeps an eid -> AX ref
registry per snapshot so the executor can resolve a target in O(1).

Nothing here changes the UI. The only AX write is `AXManualAccessibility` on Chromium/Electron
apps (it asks the app to build its accessibility tree), and `manual_ax=False` turns even that off.
We never set `AXEnhancedUserInterface`: it makes some apps animate window moves and break layout.

`Element.path` is the child-index path from the walked window. A negative first index marks a
different root: MENU_ROOT (the app's menu bar) or FOCUS_ROOT (the focused element, when the walk
did not reach it).
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

import ApplicationServices as AX
import CoreFoundation as CF
import objc
from AppKit import NSRunningApplication, NSWorkspace
from Foundation import NSArray, NSAttributedString, NSURL

from jev_local.harness.types import Element, Snapshot

# ---------------------------------------------------------------- AX errors

AX_ERRORS: dict[int, str] = {
    0: "success",
    -25200: "failure",
    -25201: "illegal argument",
    -25202: "invalid element (stale)",
    -25203: "invalid observer",
    -25204: "cannot complete (app did not answer in time)",
    -25205: "attribute unsupported",
    -25206: "action unsupported",
    -25207: "notification unsupported",
    -25208: "not implemented",
    -25209: "notification already registered",
    -25210: "notification not registered",
    -25211: "accessibility API disabled (permission missing)",
    -25212: "no value",
    -25213: "parameterized attribute unsupported",
    -25214: "not enough precision",
}
AX_OK = 0
AX_STALE = -25202
AX_TIMEOUT = -25204
AX_DISABLED = -25211
AX_NO_VALUE = -25212


def ax_error_name(code: int) -> str:
    return f"{AX_ERRORS.get(code, 'unknown AX error')} ({code})"


class PermissionMissing(RuntimeError):
    """The host process is not trusted for Accessibility (AXIsProcessTrusted() is False / -25211)."""

    def __init__(self, what: str = "accessibility"):
        super().__init__(
            f"{what} permission missing: add the app that runs jev-local (your terminal or launcher) "
            f"under System Settings > Privacy & Security > Accessibility, then restart it"
        )
        self.what = what


# ---------------------------------------------------------------- low-level reads

def _py(v: Any) -> Any:
    """Convert one AX attribute value to plain Python. AX error placeholders become None."""
    if v is None:
        return None
    if isinstance(v, AX.AXValueRef):
        t = AX.AXValueGetType(v)
        if t in (AX.kAXValueTypeCGPoint, AX.kAXValueTypeCGSize, AX.kAXValueTypeCGRect, AX.kAXValueTypeCFRange):
            ok, s = AX.AXValueGetValue(v, t, None)
            if not ok:
                return None
            if t == AX.kAXValueTypeCGPoint:
                return (float(s.x), float(s.y))
            if t == AX.kAXValueTypeCGSize:
                return (float(s.width), float(s.height))
            if t == AX.kAXValueTypeCGRect:
                return (float(s.origin.x), float(s.origin.y), float(s.size.width), float(s.size.height))
            if isinstance(s, tuple):  # pyobjc returns CFRange as a plain (location, length) tuple
                return (int(s[0]), int(s[1]))
            return (int(s.location), int(s.length))
        return None  # kAXValueTypeAXError: the attribute is missing on this element
    if isinstance(v, (bool, int, float, AX.AXUIElementRef)):
        return v
    if isinstance(v, str):
        return str(v)
    if isinstance(v, (NSArray, list, tuple)):
        return list(v)
    if isinstance(v, NSURL):
        return str(v.absoluteString())
    if isinstance(v, NSAttributedString):
        return str(v.string())
    return v


def ax_get(el: Any, attr: str) -> tuple[int, Any]:
    """(error, python value) for one attribute."""
    try:
        err, v = AX.AXUIElementCopyAttributeValue(el, attr, None)
    except Exception:  # pyobjc raises on a few exotic value types; treat as missing
        return (-25200, None)
    return (int(err), _py(v) if err == AX_OK else None)


def ax_value(el: Any, attr: str, default: Any = None) -> Any:
    err, v = ax_get(el, attr)
    return v if err == AX_OK and v is not None else default


def ax_get_many(el: Any, attrs: tuple[str, ...] | list[str]) -> tuple[int, list[Any]]:
    err, vals = AX.AXUIElementCopyMultipleAttributeValues(el, list(attrs), 0, None)
    if err != AX_OK or vals is None:
        return (int(err), [None] * len(attrs))
    return (AX_OK, [_py(v) for v in vals])


def ax_actions(el: Any) -> tuple[str, ...]:
    err, names = AX.AXUIElementCopyActionNames(el, None)
    return tuple(str(n) for n in names) if err == AX_OK and names else ()


def ax_settable(el: Any, attr: str) -> bool:
    err, ok = AX.AXUIElementIsAttributeSettable(el, attr, None)
    return err == AX_OK and bool(ok)


def ax_frame(el: Any) -> tuple[float, float, float, float] | None:
    err, vals = ax_get_many(el, ("AXPosition", "AXSize"))
    return _frame(vals[0], vals[1]) if err == AX_OK else None


def ax_pid(el: Any) -> int | None:
    err, pid = AX.AXUIElementGetPid(el, None)
    return int(pid) if err == AX_OK else None


def screen_locked() -> bool:
    """True while the login window covers the session (screen locked or at the login screen)."""
    import Quartz

    d = Quartz.CGSessionCopyCurrentDictionary()
    return bool(d is not None and d.get("CGSSessionScreenIsLocked", False))


def _frame(pos: Any, size: Any) -> tuple[float, float, float, float] | None:
    if not pos or not size:
        return None
    return (pos[0], pos[1], size[0], size[1])


def _intersects(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def _intersect(a: tuple[float, ...] | None, b: tuple[float, ...]) -> tuple[float, float, float, float]:
    if a is None:
        return (b[0], b[1], b[2], b[3])
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    return (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))


def _expand(r: tuple[float, ...], m: float) -> tuple[float, float, float, float]:
    return (r[0] - m, r[1] - m, r[2] + 2 * m, r[3] + 2 * m)


# ---------------------------------------------------------------- roles and labels

ROLE_NAMES: dict[str, str] = {
    "AXButton": "button",
    "AXCheckBox": "checkbox",
    "AXRadioButton": "radio button",
    "AXPopUpButton": "pop-up button",
    "AXMenuButton": "menu button",
    "AXTextField": "text field",
    "AXTextArea": "text area",
    "AXSecureTextField": "password field",
    "AXComboBox": "combo box",
    "AXLink": "link",
    "AXMenuItem": "menu item",
    "AXMenuBarItem": "menu",
    "AXSlider": "slider",
    "AXIncrementor": "stepper",
    "AXDisclosureTriangle": "disclosure triangle",
    "AXColorWell": "color well",
    "AXDateField": "date field",
    "AXTimeField": "time field",
    "AXRow": "row",
    "AXCell": "cell",
    "AXImage": "image",
    "AXStaticText": "text",
    "AXGroup": "group",
    "AXDockItem": "dock item",
    "AXHeading": "heading",
}
SUBROLE_NAMES: dict[str, str] = {
    "AXSearchField": "search field",
    "AXSecureTextField": "password field",
    "AXTabButton": "tab",
    "AXSwitch": "switch",
    "AXToggle": "toggle button",
    "AXCloseButton": "close button",
    "AXMinimizeButton": "minimize button",
    "AXZoomButton": "zoom button",
    "AXFullScreenButton": "full screen button",
    "AXSortButton": "sort button",
    "AXOutlineRow": "row",
    "AXTableRow": "row",
}
# Subroles whose humanized name is enough to refer to an unlabeled element ("close button").
SELF_NAMING_SUBROLES = frozenset({
    "AXCloseButton", "AXMinimizeButton", "AXZoomButton", "AXFullScreenButton", "AXSearchField",
    "AXSecureTextField",
})
# Always offered as targets (when visible).
ACTIONABLE = frozenset({
    "AXButton", "AXCheckBox", "AXRadioButton", "AXPopUpButton", "AXMenuButton", "AXTextField",
    "AXTextArea", "AXSecureTextField", "AXComboBox", "AXLink", "AXMenuItem", "AXMenuBarItem",
    "AXSlider", "AXIncrementor", "AXDisclosureTriangle", "AXColorWell", "AXDateField", "AXTimeField",
    "AXDockItem",
})
TEXT_INPUTS = frozenset({"AXTextField", "AXTextArea", "AXSecureTextField", "AXComboBox"})
# Offered only when they answer to a press/open action (web "clickable div", desktop icons, cells).
PRESSABLE_IF_ACTION = frozenset({"AXCell", "AXImage", "AXGroup", "AXStaticText", "AXHeading"})
PRESS_ACTIONS = frozenset({"AXPress", "AXOpen", "AXPick", "AXConfirm"})
CONTEXT_OF_ROLE = {
    "AXToolbar": "toolbar",
    "AXTabGroup": "tab bar",
    "AXMenuBar": "menu bar",
    "AXMenu": "menu",
    "AXSheet": "dialog",
    "AXPopover": "popover",
    "AXTable": "list",
    "AXOutline": "list",
    "AXList": "list",
}
MODAL_SUBROLES = frozenset({"AXDialog", "AXSystemDialog", "AXFloatingWindow"})

MAX_LABEL = 60
SCROLL_MARGIN = 200.0  # keep elements this far outside a scroll area's visible rect (SPEC §4.5)
BIG_NODE = 50  # above this many children, walk only the visible ones
MAX_DEPTH = 60
MENU_ROOT = -1
FOCUS_ROOT = -2
FOCUSED_WEIGHT = 1000.0
FRONT_CHECK_TIMEOUT_S = 0.15
LOGINWINDOW = "com.apple.loginwindow"

CHROMIUM_BUNDLES = frozenset({
    "com.google.Chrome", "com.google.Chrome.beta", "com.google.Chrome.dev", "com.google.Chrome.canary",
    "org.chromium.Chromium", "com.microsoft.edgemac", "com.microsoft.edgemac.Beta", "com.brave.Browser",
    "company.thebrowser.Browser", "com.vivaldi.Vivaldi", "com.operasoftware.Opera",
    "com.operasoftware.OperaGX", "com.naver.Whale",
})
_ELECTRON_MARKERS = (
    "Contents/Frameworks/Electron Framework.framework",
    "Contents/Frameworks/Chromium Embedded Framework.framework",
)


def humanize_role(role: str, subrole: str = "") -> str:
    if subrole in SUBROLE_NAMES:
        return SUBROLE_NAMES[subrole]
    if role in ROLE_NAMES:
        return ROLE_NAMES[role]
    r = role.removeprefix("AX")
    out = "".join(" " + c.lower() if c.isupper() else c for c in r).strip()
    return out or "element"


def clean_text(v: Any, max_len: int = MAX_LABEL) -> str:
    """Single-line, trimmed, <= max_len characters (ellipsis when cut)."""
    if v is None or isinstance(v, bool):
        return ""
    s = " ".join(str(v).split())
    s = "".join(ch for ch in s if ch.isprintable())
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "…"
    return s


def format_value(role: str, subrole: str, v: Any) -> str | None:
    """What the model sees after `=`: text, on/off, a number. None when there is nothing useful."""
    if v is None:
        return None
    if role in ("AXCheckBox",) or subrole in ("AXSwitch", "AXToggle"):
        if isinstance(v, bool):
            return "on" if v else "off"
        if isinstance(v, (int, float)):
            return {0: "off", 1: "on", 2: "mixed"}.get(int(v))
    if role == "AXRadioButton":
        return "selected" if (v is True or v == 1) else None
    if isinstance(v, bool):
        return None
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return clean_text(v) or None
    return None


def is_chromium_like(bundle_id: str, bundle_path: str | None) -> bool:
    if bundle_id in CHROMIUM_BUNDLES:
        return True
    if bundle_path:
        return any(os.path.isdir(os.path.join(bundle_path, m)) for m in _ELECTRON_MARKERS)
    return False


def dedupe_labels(pairs: list[tuple[str, str]]) -> list[str]:
    """[(role, label)] -> labels with " (k of n)" on repeated (role, label) pairs, in order."""
    counts: dict[tuple[str, str], int] = {}
    for role, label in pairs:
        if label:
            k = (role, label.casefold())
            counts[k] = counts.get(k, 0) + 1
    seen: dict[tuple[str, str], int] = {}
    out = []
    for role, label in pairs:
        k = (role, label.casefold())
        n = counts.get(k, 0)
        if label and n > 1:
            seen[k] = seen.get(k, 0) + 1
            out.append(f"{label} ({seen[k]} of {n})")
        else:
            out.append(label)
    return out


# ---------------------------------------------------------------- walk records

ATTRS = (
    "AXRole", "AXSubrole", "AXTitle", "AXDescription", "AXValue", "AXPosition", "AXSize",
    "AXChildren", "AXEnabled", "AXFocused", "AXPlaceholderValue", "AXHelp", "AXHidden", "AXSelected",
)
(_ROLE, _SUB, _TITLE, _DESC, _VALUE, _POS, _SIZE, _KIDS, _ENABLED, _FOCUSED, _PLACEHOLDER, _HELP,
 _HIDDEN, _SELECTED) = range(len(ATTRS))


@dataclass
class _Cand:
    ref: Any
    order: int
    role_ax: str
    subrole: str
    role: str
    label: str
    value: str | None
    frame: tuple[float, float, float, float]
    path: tuple[int, ...]
    secure: bool
    focused: bool
    enabled: bool
    context: str | None
    modal: bool
    menu_bar: bool = False
    actions: tuple[str, ...] = ()
    visible: bool = True  # False when the frame is known and has zero area


@dataclass(frozen=True)
class _Ctx:
    """What a node inherits from its ancestors."""

    context: str | None = None  # nearest named container: toolbar, sidebar, dialog, ...
    clip: tuple[float, float, float, float] | None = None  # visible rect (+ margin) of enclosing scroll areas
    modal: bool = False  # inside a sheet/dialog/open menu
    in_row: bool = False  # inside a table/outline row, which already stands for its text


@dataclass
class _Walk:
    deadline: float
    max_nodes: int
    max_children: int
    n: int = 0
    partial: bool = False
    errors: int = 0
    timeouts: int = 0
    action_queries: int = 0
    seen: set = field(default_factory=set)
    cands: list[_Cand] = field(default_factory=list)
    scroll_areas: list[tuple[Any, tuple[float, float, float, float], tuple[int, ...]]] = field(default_factory=list)
    window_area: float = 0.0


@dataclass
class _Registry:
    app: Any
    window: Any
    refs: dict[str, Any]
    scroll_areas: list[tuple[Any, tuple[float, float, float, float], tuple[int, ...]]]


@dataclass(frozen=True)
class FrontApp:
    pid: int
    name: str
    bundle_id: str
    path: str | None


@dataclass
class _Cached:
    snap: Snapshot
    window: Any
    events: int
    gen: int


# ---------------------------------------------------------------- change watcher

WATCH_NOTIFICATIONS = (
    "AXFocusedUIElementChanged", "AXFocusedWindowChanged", "AXMainWindowChanged", "AXWindowCreated",
    "AXUIElementDestroyed", "AXValueChanged", "AXLayoutChanged", "AXTitleChanged", "AXMenuOpened",
    "AXMenuClosed", "AXSheetCreated",
)


class _Watcher:
    """AXObserver on one app, run on a private CFRunLoop thread; calls on_change(notification).

    Retargeting happens on the watcher thread (AXObserver sources must be added to the run loop
    that services them), so `watch(pid)` only records the wish and pokes the loop.
    """

    def __init__(self, on_change: Callable[[str], None]):
        self._on_change = on_change
        self._want: int | None = None
        self._cur: int | None = None
        self._obs: Any = None
        self._rl: Any = None
        self._stop = False
        self._thread: threading.Thread | None = None
        self.failed: dict[int, int] = {}  # pid -> AX error from AXObserverCreate

        @objc.callbackFor(AX.AXObserverCreate)
        def _cb(observer: Any, element: Any, notification: Any, refcon: Any) -> None:
            try:
                self._on_change(str(notification))
            except Exception:
                pass

        self._cb = _cb  # keep the closure alive as long as the watcher

    def watch(self, pid: int) -> None:
        if pid == self._want:
            return
        self._want = pid
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="jev-ax-watch", daemon=True)
            self._thread.start()
        elif self._rl is not None:
            CF.CFRunLoopStop(self._rl)

    @property
    def active_pid(self) -> int | None:
        return self._cur if self._obs is not None else None

    def close(self) -> None:
        self._stop = True
        if self._rl is not None:
            CF.CFRunLoopStop(self._rl)

    def _run(self) -> None:
        self._rl = CF.CFRunLoopGetCurrent()
        while not self._stop:
            if self._want != self._cur:
                self._retarget(self._want)
            CF.CFRunLoopRunInMode(CF.kCFRunLoopDefaultMode, 0.5, False)
        self._retarget(None)

    def _retarget(self, pid: int | None) -> None:
        if self._obs is not None:
            CF.CFRunLoopRemoveSource(self._rl, AX.AXObserverGetRunLoopSource(self._obs), CF.kCFRunLoopDefaultMode)
            self._obs = None
        self._cur = pid
        if pid is None:
            return
        err, obs = AX.AXObserverCreate(pid, self._cb, None)
        if err != AX_OK or obs is None:
            self.failed[pid] = int(err)
            return
        app = AX.AXUIElementCreateApplication(pid)
        added = 0
        for n in WATCH_NOTIFICATIONS:
            if AX.AXObserverAddNotification(obs, app, n, None) == AX_OK:
                added += 1
        if not added:
            self.failed[pid] = -25207
            return
        CF.CFRunLoopAddSource(self._rl, AX.AXObserverGetRunLoopSource(obs), CF.kCFRunLoopDefaultMode)
        self._obs = obs


# ---------------------------------------------------------------- observer

WATCHED_MAX_AGE_S = 10.0  # cache lifetime while the AX change watcher reports no change


class Observer:
    """Snapshots of the frontmost app's focused window, cached and prefetchable.

    Thread-safe: walks are serialized by a lock, and a caller that arrives while a prefetch walk is
    running waits for it and reuses its result instead of walking twice.
    """

    def __init__(
        self,
        max_elements: int = 60,
        deadline_s: float = 1.5,
        messaging_timeout_s: float = 0.4,
        *,
        max_nodes: int = 600,
        max_children: int = 200,
        include_menu_bar: bool = True,
        manual_ax: bool = True,
        watch: bool = True,
        max_action_queries: int = 150,
    ):
        self.max_elements = max_elements
        self.deadline_s = deadline_s
        self.messaging_timeout_s = messaging_timeout_s
        self.max_nodes = max_nodes
        self.max_children = max_children
        self.include_menu_bar = include_menu_bar
        self.manual_ax = manual_ax
        self.max_action_queries = max_action_queries
        self._sys = AX.AXUIElementCreateSystemWide()
        # The system-wide timeout is the global default for every element (Apple's default is 6 s,
        # which lets one hung app stall a decision).
        AX.AXUIElementSetMessagingTimeout(self._sys, messaging_timeout_s)
        self._walk_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._cache: _Cached | None = None
        self._gen = 0
        self._events = 0
        self._registry: OrderedDict[tuple[int, float], _Registry] = OrderedDict()
        self._manual_done: set[int] = set()
        self._chromium: dict[str, bool] = {}
        self._contacted: set[int] = set()
        self._prefetching = False
        self.last_stats: dict[str, Any] = {}
        self._watcher = _Watcher(self._on_notification) if watch else None

    # ------------------------------------------------------------ public API

    def snapshot(
        self,
        allow_cached: bool = True,
        max_age_s: float = 1.5,
        *,
        deadline_s: float | None = None,
        pid: int | None = None,
    ) -> Snapshot:
        """The frontmost app's screen as a Snapshot (or app `pid`'s, which need not be frontmost).

        Raises PermissionMissing if the process is not trusted for Accessibility.
        """
        if not AX.AXIsProcessTrusted():
            raise PermissionMissing()
        front = self.frontmost() if pid is None else self.app_info(pid)
        if allow_cached:
            hit = self._cached_for(front, max_age_s)
            if hit is not None:
                return hit
        if screen_locked() or (front is not None and front.bundle_id == LOGINWINDOW):
            # Nothing on screen belongs to the user's apps, and we don't read the lock screen.
            # (Right after unlocking, loginwindow can stay "active" with the lock flag already gone.)
            return Snapshot("loginwindow", LOGINWINDOW, 0, None, (), time.monotonic(), partial=True)
        with self._walk_lock:
            if allow_cached:  # a prefetch may have finished while we waited for the lock
                hit = self._cached_for(front, max_age_s)
                if hit is not None:
                    return hit
            return self._walk_and_store(front, deadline_s or self.deadline_s)

    def prefetch(self) -> None:
        """Warm the cache on a background thread (called on speech_start). Never raises."""
        if self._prefetching:
            return
        self._prefetching = True

        def run() -> None:
            try:
                self.snapshot(allow_cached=True)
            except Exception:
                pass
            finally:
                self._prefetching = False

        threading.Thread(target=run, name="jev-ax-prefetch", daemon=True).start()

    def invalidate(self) -> None:
        """Drop the cache; a walk already in flight will not repopulate it."""
        with self._state_lock:
            self._gen += 1
            self._cache = None

    def resolve(self, snap: Snapshot, eid: str) -> Any | None:
        """AX element for `eid` of `snap`: the registry ref if still alive, else re-found by path."""
        el = next((e for e in snap.elements if e.eid == eid), None)
        if el is None:
            return None
        reg = self._registry.get((snap.pid, snap.taken_at))
        if reg is not None:
            ref = reg.refs.get(eid)
            if ref is not None:
                err, _ = ax_get(ref, "AXRole")
                if err != AX_STALE:
                    return ref
        return self._resolve_path(snap, el)

    def focused_element(self) -> Any | None:
        """The live keyboard-focus element (system-wide, else the frontmost app's)."""
        err, el = ax_get(self._sys, "AXFocusedUIElement")
        if err == AX_OK and el is not None:
            return el
        front = self.frontmost()
        if front is None:
            return None
        err, el = ax_get(AX.AXUIElementCreateApplication(front.pid), "AXFocusedUIElement")
        return el if err == AX_OK else None

    def frontmost(self) -> FrontApp | None:
        """The active app. NSWorkspace can be stale in a process without a main run loop, so its
        answer is checked against AXFrontmost and, if wrong, the window list is searched."""
        pid: int | None = None
        err, app_el = ax_get(self._sys, "AXFocusedApplication")
        if err == AX_OK and app_el is not None:
            pid = ax_pid(app_el)
        if pid is None:
            ws_app = NSWorkspace.sharedWorkspace().frontmostApplication()
            cand = int(ws_app.processIdentifier()) if ws_app is not None else None
            if cand is not None and self._is_frontmost(cand) is not False:
                pid = cand
            else:
                pid = self._frontmost_from_windows() or cand
        if pid is None:
            return None
        info = self.app_info(pid) or FrontApp(pid, "", "", None)
        if info.bundle_id == LOGINWINDOW and not screen_locked():
            # Right after an unlock, loginwindow keeps activation until the user clicks, while the
            # app on screen (the one whose menus fill the menu bar) is the one the user means.
            owner = NSWorkspace.sharedWorkspace().menuBarOwningApplication()
            if owner is not None and owner.bundleIdentifier() != LOGINWINDOW:
                info = self.app_info(int(owner.processIdentifier())) or info
        return info

    @staticmethod
    def app_info(pid: int) -> FrontApp | None:
        ra = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        if ra is None:
            return None
        url = ra.bundleURL()
        return FrontApp(pid, str(ra.localizedName() or ""), str(ra.bundleIdentifier() or ""),
                        str(url.path()) if url is not None else None)

    def scroll_areas(self, snap: Snapshot) -> list[tuple[Any, tuple[float, float, float, float], tuple[int, ...]]]:
        """(ref, frame, path) of the scroll areas seen by the walk that produced `snap`."""
        reg = self._registry.get((snap.pid, snap.taken_at))
        return list(reg.scroll_areas) if reg else []

    def close(self) -> None:
        if self._watcher is not None:
            self._watcher.close()

    # ------------------------------------------------------------ cache

    def _on_notification(self, name: str) -> None:
        with self._state_lock:
            self._events += 1

    def _cached_for(self, front: FrontApp | None, max_age_s: float) -> Snapshot | None:
        with self._state_lock:
            c = self._cache
            if c is None or front is None or c.snap.pid != front.pid or c.gen != self._gen:
                return None
            age = time.monotonic() - c.snap.taken_at
            watching = self._watcher is not None and self._watcher.active_pid == front.pid
            if watching:
                # The app's AX notifications say nothing changed since the walk, so the snapshot can be
                # kept much longer than an unwatched one: the first decision of the next command then
                # needs no new walk (250-1070 ms on real apps) on the critical path.
                if age > max(max_age_s, WATCHED_MAX_AGE_S):
                    return None
                return c.snap if self._events == c.events else None
            if age > max_age_s:
                return None
        # Without a working watcher, at least check that the focused window is the same one.
        if c.window is not None:
            app = AX.AXUIElementCreateApplication(front.pid)
            AX.AXUIElementSetMessagingTimeout(app, FRONT_CHECK_TIMEOUT_S)
            err, win = ax_get(app, "AXFocusedWindow")
            if err == AX_OK and win is not None and win != c.window:
                return None
        return c.snap

    def _walk_and_store(self, front: FrontApp | None, deadline_s: float) -> Snapshot:
        with self._state_lock:
            gen = self._gen
        snap, reg = self._walk(front, deadline_s)
        with self._state_lock:
            events = self._events  # notifications our own walk triggered don't count as changes
            self._registry[(snap.pid, snap.taken_at)] = reg
            while len(self._registry) > 8:
                self._registry.popitem(last=False)
            if gen == self._gen:
                self._cache = _Cached(snap, reg.window, events, gen)
        if self._watcher is not None and front is not None:
            self._watcher.watch(front.pid)
        return snap

    # ------------------------------------------------------------ frontmost helpers

    def _is_frontmost(self, pid: int) -> bool | None:
        app = AX.AXUIElementCreateApplication(pid)
        # Short timeout: a busy app must not stall the check; None means "can't tell".
        AX.AXUIElementSetMessagingTimeout(app, FRONT_CHECK_TIMEOUT_S)
        err, v = ax_get(app, "AXFrontmost")
        return bool(v) if err == AX_OK else None

    def _frontmost_from_windows(self) -> int | None:
        import Quartz

        opts = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
        info = Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID) or []
        tried: set[int] = set()
        for w in info:
            if w.get("kCGWindowLayer") != 0:
                continue
            pid = int(w.get("kCGWindowOwnerPID", 0))
            if pid in tried:
                continue
            tried.add(pid)
            if self._is_frontmost(pid):
                return pid
            if len(tried) >= 6:
                break
        return None

    # ------------------------------------------------------------ the walk

    def _walk(self, front: FrontApp | None, deadline_s: float) -> tuple[Snapshot, _Registry]:
        t0 = time.monotonic()
        if front is None:
            snap = Snapshot("", "", 0, None, (), t0, partial=True, walk_ms=0.0)
            return snap, _Registry(None, None, {}, [])
        app = AX.AXUIElementCreateApplication(front.pid)
        newly_manual = self._maybe_manual_ax(app, front)
        # The first message to an app that is napping can take ~0.5 s; give it longer than the
        # per-node timeout so the snapshot doesn't come back empty.
        if front.pid not in self._contacted:
            AX.AXUIElementSetMessagingTimeout(app, max(1.5, self.messaging_timeout_s))
        window = self._window_for(app)
        if window is None and front.pid not in self._contacted:
            window = self._window_for(app)
        AX.AXUIElementSetMessagingTimeout(app, self.messaging_timeout_s)
        self._contacted.add(front.pid)

        snap, reg = self._walk_window(front, app, window, t0, deadline_s)
        if newly_manual and len(snap.elements) < 3:
            time.sleep(0.25)  # Chromium builds its tree asynchronously after AXManualAccessibility
            t0 = time.monotonic()
            snap, reg = self._walk_window(front, app, self._window_for(app), t0, deadline_s)
        return snap, reg

    def _window_for(self, app: Any) -> Any | None:
        for attr in ("AXFocusedWindow", "AXMainWindow"):
            err, w = ax_get(app, attr)
            if err == AX_DISABLED:
                raise PermissionMissing()
            if err == AX_OK and w is not None:
                return w
        err, wins = ax_get(app, "AXWindows")
        if err == AX_DISABLED:
            raise PermissionMissing()
        return wins[0] if err == AX_OK and wins else None

    def _walk_window(self, front: FrontApp, app: Any, window: Any, t0: float, deadline_s: float) -> tuple[Snapshot, _Registry]:
        w = _Walk(deadline=t0 + deadline_s, max_nodes=self.max_nodes, max_children=self.max_children)
        title: str | None = None
        if window is not None:
            keys = ("AXTitle", "AXDescription", "AXRole", "AXSubrole", "AXPosition", "AXSize")
            err, vals = ax_get_many(window, keys)
            if err == AX_TIMEOUT:  # one retry: a napping app answers the second message
                err, vals = ax_get_many(window, keys)
            if err == AX_DISABLED:
                raise PermissionMissing()
            modal = False
            if err == AX_OK:
                # Finder's desktop "window" is a scroll area described as "desktop".
                title = clean_text(vals[0] or (vals[1] if vals[2] != "AXWindow" else ""), 120) or None
                modal = vals[3] in MODAL_SUBROLES
                wf = _frame(vals[4], vals[5])
                w.window_area = (wf[2] * wf[3]) if wf else 0.0
            self._visit(w, window, (), 0, _Ctx(context="dialog" if modal else None, modal=modal))

        # The deadline is soft (one in-flight IPC can overrun it by up to the messaging timeout), so
        # don't start optional queries once it has passed.
        err, fel = ax_get(app, "AXFocusedUIElement") if time.monotonic() < w.deadline else (AX_TIMEOUT, None)
        if err == AX_OK and fel is not None:
            hit = next((c for c in w.cands if c.ref == fel), None)
            if hit is not None:
                hit.focused = True
            elif fel not in w.seen and time.monotonic() < w.deadline:
                # Focus is outside the walked window (a popover, another window) or past the budget.
                before = len(w.cands)
                self._visit(w, fel, (FOCUS_ROOT,), 0, _Ctx(modal=True), force=True)
                if len(w.cands) > before:
                    w.cands[before].focused = True

        menu_ms = 0.0
        if self.include_menu_bar and time.monotonic() - t0 < deadline_s * 0.5:
            tm = time.monotonic()
            self._visit_menu_bar(w, app)
            menu_ms = (time.monotonic() - tm) * 1000

        kept = self._select(w)
        for c in kept:  # action names only for what we keep (one cheap IPC each)
            if not c.actions and time.monotonic() < w.deadline:
                c.actions = ax_actions(c.ref)
        labels = dedupe_labels([(c.role, c.label) for c in kept])
        elements: list[Element] = []
        refs: dict[str, Any] = {}
        focused_eid = None
        for i, (c, label) in enumerate(zip(kept, labels), start=1):
            eid = f"e{i:02d}"
            refs[eid] = c.ref
            if c.focused and focused_eid is None:
                focused_eid = eid
            elements.append(
                Element(
                    eid=eid,
                    role=c.role,
                    label=label,
                    value=None if c.secure else c.value,
                    frame=c.frame,
                    actions=c.actions,
                    path=c.path,
                    secure=c.secure,
                    focused=c.focused,
                    enabled=c.enabled,
                    context=c.context,
                )
            )
        walk_ms = (time.monotonic() - t0) * 1000
        self.last_stats = {
            "nodes": w.n,
            "candidates": len(w.cands),
            "elements": len(elements),
            "action_queries": w.action_queries,
            "errors": w.errors,
            "timeouts": w.timeouts,
            "menu_ms": round(menu_ms, 1),
            "walk_ms": round(walk_ms, 1),
            "partial": w.partial,
        }
        snap = Snapshot(
            app_name=front.name,
            bundle_id=front.bundle_id,
            pid=front.pid,
            window_title=title,
            elements=tuple(elements),
            taken_at=t0,
            partial=w.partial,
            focused_eid=focused_eid,
            walk_ms=walk_ms,
        )
        return snap, _Registry(app, window, refs, w.scroll_areas)

    def _visit(self, w: _Walk, el: Any, path: tuple[int, ...], depth: int, ctx: _Ctx, force: bool = False) -> str:
        """Depth-first visit. Returns a short text summary of the subtree (labels unlabeled rows)."""
        if w.n >= w.max_nodes or depth > MAX_DEPTH or time.monotonic() > w.deadline:
            w.partial = True
            return ""
        if el in w.seen:  # Chromium lists some subtrees under two parents
            return ""
        w.seen.add(el)
        w.n += 1
        err, a = ax_get_many(el, ATTRS)
        if err == AX_DISABLED:
            raise PermissionMissing()
        if err != AX_OK:
            w.errors += 1
            if err == AX_TIMEOUT:
                w.timeouts += 1
            return ""
        if a[_HIDDEN] is True:
            return ""
        role = a[_ROLE] if isinstance(a[_ROLE], str) else ""
        sub = a[_SUB] if isinstance(a[_SUB], str) else ""
        frame = _frame(a[_POS], a[_SIZE])
        has_area = frame is not None and frame[2] > 0 and frame[3] > 0
        if not force and has_area and ctx.clip is not None and not _intersects(frame, ctx.clip):
            return ""  # scrolled out of view (plus margin)

        secure = role == "AXSecureTextField" or sub == "AXSecureTextField"
        title, desc = clean_text(a[_TITLE]), clean_text(a[_DESC])
        raw_value = None if secure else a[_VALUE]
        own_text = title or desc
        if not own_text and role in ("AXStaticText", "AXTextField") and isinstance(raw_value, str):
            own_text = clean_text(raw_value)

        cand = None
        ok, acts = self._candidate(w, el, role, own_text, frame, ctx, force, a[_FOCUSED] is True)
        if ok:
            cand = _Cand(
                ref=el,
                order=len(w.cands),
                role_ax=role,
                subrole=sub,
                role="icon" if role == "AXImage" and "AXOpen" in acts else humanize_role(role, sub),
                label=self._label(role, title, desc, a, secure),
                value=None if secure or role == "AXStaticText" else format_value(role, sub, raw_value),
                frame=frame or (0.0, 0.0, 0.0, 0.0),
                path=path,
                secure=secure,
                focused=a[_FOCUSED] is True,
                enabled=a[_ENABLED] is not False,
                context=ctx.context,
                modal=ctx.modal,
                actions=acts,
                visible=frame is None or has_area,
            )
            if role in TEXT_INPUTS and not cand.label:
                cand.label = self._title_element_text(el)
            w.cands.append(cand)

        clip = ctx.clip
        if role in ("AXScrollArea", "AXWindow") and has_area:
            if role == "AXScrollArea":
                w.scroll_areas.append((el, frame, path))
            clip = _intersect(clip, _expand(frame, SCROLL_MARGIN))
        context = self._context(role, sub, desc, ctx.context)
        if own_text and (sub == "AXTabButton" or role in ("AXRow", "AXCell")):
            # "button "Close" in tab "Inbox"": names which tab/row a small inner control belongs to
            context = f'{humanize_role(role, sub)} "{clean_text(own_text, 30)}"'
        child_ctx = _Ctx(
            context=context,
            clip=clip,
            modal=ctx.modal or role == "AXSheet" or sub in MODAL_SUBROLES,
            in_row=ctx.in_row or role == "AXRow",
        )

        kids = a[_KIDS] if isinstance(a[_KIDS], list) else []
        indices: list[int] = list(range(len(kids)))
        if len(kids) > BIG_NODE:
            indices = self._visible_indices(el, kids) or indices
        texts: list[str] = []
        for i in indices[: w.max_children]:
            t = self._visit(w, kids[i], path + (i,), depth + 1, child_ctx)
            if t and len(texts) < 3:
                texts.append(t)
        child_text = clean_text(" ".join(texts), 80)
        if cand is not None and not cand.label and not secure:
            cand.label = clean_text(child_text)
        return own_text or child_text

    def _candidate(
        self, w: _Walk, el: Any, role: str, own_text: str, frame: Any, ctx: _Ctx, force: bool, focused: bool
    ) -> tuple[bool, tuple[str, ...]]:
        """Is this node a target the user could name? Returns (keep, action names if queried)."""
        if force or focused:
            return True, ()
        if ctx.in_row and role in ("AXTextField", "AXStaticText", "AXImage", "AXCell"):
            return False, ()  # the row already stands for its name cell and icon
        if role in ACTIONABLE or role == "AXRow":
            return True, ()
        if role not in PRESSABLE_IF_ACTION or w.action_queries >= self.max_action_queries:
            return False, ()
        if frame is None or frame[2] <= 0 or frame[3] <= 0:
            return False, ()
        # A "clickable" group covering a quarter of the window is layout, not a target.
        if role == "AXGroup" and w.window_area and frame[2] * frame[3] > 0.25 * w.window_area:
            return False, ()
        if role in ("AXStaticText", "AXHeading", "AXImage") and not own_text:
            return False, ()
        w.action_queries += 1
        acts = ax_actions(el)
        return bool(PRESS_ACTIONS.intersection(acts)), acts

    @staticmethod
    def _context(role: str, sub: str, desc: str, inherited: str | None) -> str | None:
        if role in ("AXOutline", "AXTable", "AXList") and (sub == "AXSourceList" or "sidebar" in desc.lower()):
            return "sidebar"
        if desc.lower() == "desktop" and role in ("AXScrollArea", "AXGroup"):
            return "desktop"
        if role in ("AXSplitGroup", "AXGroup", "AXScrollArea") and "sidebar" in desc.lower():
            return "sidebar"
        return CONTEXT_OF_ROLE.get(role, inherited)

    @staticmethod
    def _label(role: str, title: str, desc: str, a: list[Any], secure: bool) -> str:
        if role == "AXStaticText" and not secure:
            return clean_text(a[_VALUE]) or title or desc
        label = title or desc
        if not label and role in TEXT_INPUTS:
            label = clean_text(a[_PLACEHOLDER])
        return label or clean_text(a[_HELP])

    @staticmethod
    def _title_element_text(el: Any) -> str:
        """Label of a text field that is named by a separate static text ("To:" in Mail)."""
        err, t = ax_get(el, "AXTitleUIElement")
        if err != AX_OK or t is None:
            return ""
        err, vals = ax_get_many(t, ("AXValue", "AXTitle", "AXDescription"))
        if err != AX_OK:
            return ""
        return clean_text(next((v for v in vals if isinstance(v, str) and v.strip()), ""))

    @staticmethod
    def _visible_indices(el: Any, kids: list[Any]) -> list[int]:
        """Indices (into AXChildren) of the visible children of a big table/list, so paths stay
        valid child indices for re-resolving."""
        for attr in ("AXVisibleChildren", "AXVisibleRows"):
            err, vis = ax_get(el, attr)
            if err == AX_OK and vis:
                pos = {k: i for i, k in enumerate(kids)}  # AXUIElementRef hashes via CFHash
                return [pos[v] for v in vis if v in pos]
        return []

    def _visit_menu_bar(self, w: _Walk, app: Any) -> None:
        err, bar = ax_get(app, "AXMenuBar")
        if err != AX_OK or bar is None:
            return
        err, items = ax_get(bar, "AXChildren")
        if err != AX_OK or not items:
            return
        keys = ("AXRole", "AXTitle", "AXEnabled", "AXSelected", "AXPosition", "AXSize", "AXChildren")
        for i, item in enumerate(items):
            if time.monotonic() > w.deadline:
                w.partial = True
                return
            err, a = ax_get_many(item, keys)
            if err != AX_OK or a[0] != "AXMenuBarItem":
                continue
            title = clean_text(a[1])
            if not title:
                continue
            w.n += 1
            w.cands.append(
                _Cand(
                    ref=item, order=len(w.cands), role_ax="AXMenuBarItem", subrole="", role="menu",
                    label=title, value=None, frame=_frame(a[4], a[5]) or (0.0, 0.0, 0.0, 0.0),
                    path=(MENU_ROOT, i), secure=False, focused=False, enabled=a[2] is not False,
                    context="menu bar", modal=False, menu_bar=True,
                )
            )
            # An open menu (its bar item is selected) is on screen; its items are what the user sees.
            if a[3] is True and a[6]:
                self._visit(w, a[6][0], (MENU_ROOT, i, 0), 1, _Ctx(context="menu", modal=True))

    def _select(self, w: _Walk) -> list[_Cand]:
        """Drop unnameable candidates, rank, keep max_elements, and restore walk order."""
        cands = []
        for c in w.cands:
            if not c.focused:
                if not c.visible:
                    continue
                if not c.label and c.role_ax not in TEXT_INPUTS and c.subrole not in SELF_NAMING_SUBROLES:
                    continue  # an unlabeled button/row is not something a user can refer to

            cands.append(c)

        def score(c: _Cand) -> float:
            s = FOCUSED_WEIGHT if c.focused else 0.0
            s += 50 if c.modal else 0
            s += 20 if c.label else 0
            s += 10 if c.enabled else 0
            if c.role_ax in TEXT_INPUTS:
                s += 15
            elif c.role_ax in ACTIONABLE:
                s += 12
            elif c.role_ax == "AXRow":
                s += 6
            else:
                s += 2
            if c.menu_bar:
                s -= 40  # always present; the window is what the user is looking at
            return s - c.order * 1e-4  # earlier in the walk wins ties

        ranked = sorted(cands, key=score, reverse=True)[: self.max_elements]
        return sorted(ranked, key=lambda c: c.order)

    # ------------------------------------------------------------ resolve helpers

    def _resolve_path(self, snap: Snapshot, el: Element) -> Any | None:
        path = el.path
        app = AX.AXUIElementCreateApplication(snap.pid)
        if path[:1] == (MENU_ROOT,):
            node = ax_value(app, "AXMenuBar")
            rest = path[1:]
        elif path[:1] == (FOCUS_ROOT,):
            node = ax_value(app, "AXFocusedUIElement")
            rest = path[1:]
        else:
            node = self._window_for(app)
            rest = path
        for i in rest:
            if node is None:
                return None
            kids = ax_value(node, "AXChildren") or []
            if i >= len(kids):
                return None
            node = kids[i]
        if node is None:
            return None
        err, a = ax_get_many(node, ("AXRole", "AXSubrole", "AXTitle", "AXDescription"))
        if err != AX_OK:
            return None
        if humanize_role(a[0] or "", a[1] or "") != el.role:
            return None
        direct = clean_text(a[2]) or clean_text(a[3])
        base = el.label.rsplit(" (", 1)[0] if el.label.endswith(")") and " of " in el.label else el.label
        if direct and base and direct.casefold() != base.casefold():
            return None
        return node

    def _maybe_manual_ax(self, app: Any, front: FrontApp) -> bool:
        if not self.manual_ax or front.pid in self._manual_done:
            return False
        key = front.path or front.bundle_id
        if key not in self._chromium:
            self._chromium[key] = is_chromium_like(front.bundle_id, front.path)
        self._manual_done.add(front.pid)
        if not self._chromium[key]:
            return False
        return AX.AXUIElementSetAttributeValue(app, "AXManualAccessibility", True) == AX_OK


# ---------------------------------------------------------------- CLI (read-only)

def snapshot_lines(snap: Snapshot) -> list[str]:
    """The snapshot as the model sees it: one `eNN element_line` per element."""
    from jev_local.harness.questions import element_line

    head = f"{snap.app_name} ({snap.bundle_id}, pid {snap.pid})"
    if snap.window_title:
        head += f' window "{snap.window_title}"'
    head += f" - {len(snap.elements)} elements, {snap.walk_ms:.0f} ms" + (" (partial)" if snap.partial else "")
    return [head] + [f"{e.eid} {element_line(e)}" for e in snap.elements]


def main(argv: list[str] | None = None) -> int:
    """`python -m jev_local.harness.observe [--app NAME] [--runs N] [--quiet]`: print a snapshot and
    walk timings. Read-only: never sets AXManualAccessibility."""
    import argparse
    import statistics

    from jev_local.harness import apps as apps_mod

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--app", help="a running app's name (default: the frontmost app)")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--max-elements", type=int, default=60)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    obs = Observer(max_elements=args.max_elements, manual_ax=False, watch=False)
    pid = None
    if args.app:
        r = apps_mod.find_running(args.app)
        if r is None:
            print(f"{args.app!r} is not running")
            return 1
        pid = r.pid
    times = []
    snap = None
    for _ in range(max(1, args.runs)):
        snap = obs.snapshot(allow_cached=False, pid=pid)
        times.append(snap.walk_ms)
    assert snap is not None
    if not args.quiet:
        print("\n".join(snapshot_lines(snap)))
    if len(times) > 1:
        print(f"walk_ms over {len(times)} runs: p50 {statistics.median(times):.1f}  min {min(times):.1f}  "
              f"max {max(times):.1f}  first {times[0]:.1f}")
    print("stats:", obs.last_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
