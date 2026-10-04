"""Carry out harness actions on macOS (docs/research/SPEC.md §4.6).

`cfg.dry_run=True` (the default) only validates and describes the action: it never imports Quartz,
never posts an event and never talks to another app. Live execution goes through `MacUI`, the only
class in the harness that changes the screen, so tests can swap in a recorder and exercise every
live code path without touching the UI.

Methods, in order of preference (the most semantic, least pointer-dependent first):
- apps, URLs, folders: NSWorkspace (no AppleScript, so no Automation prompt);
- click: AXPress, AXPick, AXSelected=True, then an occlusion-checked CGEvent click that restores
  the pointer;
- type: set AXSelectedText (read back to verify), else CGEvent unicode keystrokes;
- keys and shortcuts: CGEvent key events posted to the frontmost app's pid;
- scroll: the scroll bar's AXValue for "all the way", else scroll-wheel events located over the
  scroll area.

`ExecResult.changed` is True when the action was delivered and, where AX lets us check, had a
visible effect (value, focus, selection or frontmost app changed). Keys and wheel events that
cannot be checked count as changed once posted.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import quote_plus, urlsplit, urlunsplit

from jev_local.harness import apps
from jev_local.harness.catalog import DENY_APP_NAMES, DENY_APP_PATTERNS, FOLDER_PATHS, KEYS, SCROLL_LEVELS, SECURE_FIELD_RE
from jev_local.harness.types import Action, ActionKind, ActionRecord, Element, ExecResult, HarnessConfig, Snapshot

# ---------------------------------------------------------------- keys

# CGEventFlags masks, spelled out so this module imports without Quartz (dry-run needs no pyobjc).
FLAG_SHIFT = 1 << 17
FLAG_CTRL = 1 << 18
FLAG_ALT = 1 << 19
FLAG_CMD = 1 << 20
FLAG_NUMPAD = 1 << 21
FLAG_FN = 1 << 23

MODIFIERS: dict[str, int] = {
    "cmd": FLAG_CMD, "command": FLAG_CMD, "shift": FLAG_SHIFT, "ctrl": FLAG_CTRL, "control": FLAG_CTRL,
    "alt": FLAG_ALT, "option": FLAG_ALT, "opt": FLAG_ALT, "fn": FLAG_FN,
}

# macOS virtual key codes (kVK_*, Events.h). Letters/digits/punctuation are ANSI *positions*: on a
# non-QWERTY layout "cmd+z" posts the key where Z sits on QWERTY (see report: known gap).
KEYCODES: dict[str, int] = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9, "b": 11, "q": 12,
    "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23,
    "=": 24, "9": 25, "7": 26, "-": 27, "8": 28, "0": 29, "]": 30, "o": 31, "u": 32, "[": 33, "i": 34,
    "p": 35, "l": 37, "j": 38, "'": 39, "k": 40, ";": 41, "\\": 42, ",": 43, "/": 44, "n": 45, "m": 46,
    ".": 47, "`": 50,
    "return": 36, "tab": 48, "space": 49, "delete": 51, "escape": 53, "forward_delete": 117,
    "home": 115, "end": 119, "page_up": 116, "page_down": 121,
    "left": 123, "right": 124, "down": 125, "up": 126,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98, "f8": 100, "f9": 101,
    "f10": 109, "f11": 103, "f12": 111,
}
KEY_ALIASES: dict[str, str] = {
    "enter": "return", "esc": "escape", "backspace": "delete", "del": "delete", "spacebar": "space",
    "pagedown": "page_down", "pageup": "page_up", "page down": "page_down", "page up": "page_up",
    "arrow_up": "up", "arrow_down": "down", "arrow_left": "left", "arrow_right": "right",
    "up_arrow": "up", "down_arrow": "down", "left_arrow": "left", "right_arrow": "right",
}
# Hardware sets SecondaryFn on these keys (and NumericPad on arrows); some apps look at the flags.
_FN_KEYS = frozenset({"up", "down", "left", "right", "page_up", "page_down", "home", "end", "forward_delete"} | {f"f{i}" for i in range(1, 13)})
_ARROWS = frozenset({"up", "down", "left", "right"})


@dataclass(frozen=True)
class KeyStroke:
    label: str
    keycode: int
    flags: int


def parse_key(label: str) -> KeyStroke:
    """'cmd+shift+z' -> KeyStroke(keycode=6, flags=cmd|shift). Raises ValueError if unknown."""
    raw = label.strip().lower()
    if not raw or raw == "none":
        raise ValueError(f"no key: {label!r}")
    parts = [p.strip() for p in raw.split("+")]
    if parts[-1] == "" and len(parts) >= 2:  # "cmd++" means the "=" key's shifted glyph; not supported
        raise ValueError(f"unsupported key: {label!r}")
    *mods, key = parts
    flags = 0
    for m in mods:
        if m not in MODIFIERS:
            raise ValueError(f"unknown modifier {m!r} in {label!r}")
        flags |= MODIFIERS[m]
    key = KEY_ALIASES.get(key, key)
    if key not in KEYCODES:
        raise ValueError(f"unknown key {key!r} in {label!r}")
    if key in _FN_KEYS:
        flags |= FLAG_FN
    if key in _ARROWS:
        flags |= FLAG_NUMPAD
    return KeyStroke(label, KEYCODES[key], flags)


# Shortcuts the executor sends for fixed intents and undo.
SHORTCUTS: dict[str, str] = {
    "go_back": "cmd+[",
    "new_tab": "cmd+t",
    "close_tab": "cmd+w",
    "undo": "cmd+z",
    "reopen_tab": "cmd+shift+t",
}
# Every catalog.KEYS label (except "none") plus the shortcuts above. press_key only accepts these.
KEYMAP: dict[str, KeyStroke] = {k: parse_key(k) for k in KEYS if k != "none"} | {
    v: parse_key(v) for v in SHORTCUTS.values()
}

BROWSER_BUNDLES = frozenset({
    "com.apple.Safari", "com.apple.SafariTechnologyPreview", "com.google.Chrome", "com.google.Chrome.canary",
    "org.chromium.Chromium", "com.microsoft.edgemac", "com.brave.Browser", "company.thebrowser.Browser",
    "org.mozilla.firefox", "com.vivaldi.Vivaldi", "com.operasoftware.Opera", "com.operasoftware.OperaGX",
})

FINDER_BUNDLE = "com.apple.finder"


def _is_browser(front: Any) -> bool:
    return getattr(front, "bundle_id", None) in BROWSER_BUNDLES


def _is_finder(front: Any) -> bool:
    return getattr(front, "bundle_id", None) == FINDER_BUNDLE


# ---------------------------------------------------------------- URLs, folders, text

_SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*):(.*)$", re.S)
_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "[::1]"})


def build_url(raw: str | None) -> str | None:
    """A safe http(s) URL for what the user said, or None.

    "github.com/x" -> "https://github.com/x"; "localhost:3000" -> "http://localhost:3000".
    Anything with another scheme (file:, javascript:, mailto:, x-apple...:), credentials in the
    host part, spaces or a malformed host is refused rather than repaired.
    """
    if not raw:
        return None
    s = raw.strip()
    if not s or any(c.isspace() for c in s):
        return None
    m = _SCHEME_RE.match(s)
    if m:
        scheme, rest = m.group(1).lower(), m.group(2)
        if scheme in ("http", "https"):
            if not rest.startswith("//"):
                return None
        elif re.match(r"^\d{1,5}(?:[/?#]|$)", rest):  # "localhost:3000" parses as scheme "localhost"
            s = ("http://" if scheme in _LOCAL_HOSTS else "https://") + s
        else:
            return None
    else:
        s = "https://" + s
    try:
        parts = urlsplit(s)
        port = parts.port  # raises ValueError on a bad port
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or "@" in parts.netloc:
        return None
    host = (parts.hostname or "").lower()
    if host in _LOCAL_HOSTS or host == "::1":
        pass
    else:
        try:
            ascii_host = host.encode("idna").decode("ascii")
        except UnicodeError:
            return None
        if "." not in ascii_host or not _HOST_RE.match(ascii_host):
            return None
    if port is not None and not 0 < port < 65536:
        return None
    return urlunsplit((parts.scheme.lower(), parts.netloc, parts.path, parts.query, parts.fragment))


def build_search_url(template: str, query: str | None) -> str | None:
    """cfg.search_url with {q} replaced by the URL-encoded query; None if empty or unsafe."""
    q = (query or "").strip()
    if not q or "{q}" not in template:
        return None
    return build_url(template.replace("{q}", quote_plus(q)))


def folder_path(label: str | None) -> str | None:
    """Absolute path for a catalog.FOLDERS label (never an arbitrary path from the transcript)."""
    p = FOLDER_PATHS.get((label or "").strip().lower())
    return os.path.expanduser(p) if p else None


def utf16_len(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2


def chunk_utf16(text: str, max_units: int = 20) -> list[str]:
    """Split text into chunks of <= max_units UTF-16 code units without splitting a surrogate pair
    (CGEventKeyboardSetUnicodeString drops characters beyond ~20 units per event)."""
    chunks: list[str] = []
    cur: list[str] = []
    n = 0
    for ch in text:
        u = 2 if ord(ch) > 0xFFFF else 1
        if cur and n + u > max_units:
            chunks.append("".join(cur))
            cur, n = [], 0
        cur.append(ch)
        n += u
    if cur:
        chunks.append("".join(cur))
    return chunks


def text_digest(text: str) -> str:
    """Short digest stored in undo tokens instead of the typed text (tokens may end up in logs)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def is_secure_element(el: Element | None) -> bool:
    return el is not None and (el.secure or bool(SECURE_FIELD_RE.search(el.label or "")))


def denied_app(name: str | None, bundle_id: str | None = None, allow: tuple[str, ...] = ()) -> str | None:
    """The DENY_APP_PATTERNS entry an app matches (unless allow-listed), else None.

    The safety layer denies these first; the executor re-checks so a bug upstream can't type into
    Terminal or click around System Settings."""
    hay = f"{name or ''} {bundle_id or ''}".casefold()
    if not hay.strip() or any(a.casefold() in hay for a in allow if a):
        return None
    for whole in ((name or "").casefold().strip(), (bundle_id or "").casefold().strip()):
        if whole and whole in DENY_APP_NAMES:
            return whole
    return next((p for p in DENY_APP_PATTERNS if p in hay), None)


def _element(snap: Snapshot | None, eid: str | None) -> Element | None:
    if snap is None or not eid:
        return None
    return next((e for e in snap.elements if e.eid == eid), None)


def _focused(snap: Snapshot | None) -> Element | None:
    if snap is None:
        return None
    return _element(snap, snap.focused_eid) or next((e for e in snap.elements if e.focused), None)


def _el_text(el: Element) -> str:
    return f'{el.role} "{el.label}"' if el.label else el.role


def _clip(s: str, n: int = 40) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def describe(action: Action, snap: Snapshot | None, search_url: str = "https://www.google.com/search?q={q}") -> str:
    """Human description of what the action does, used for dry-run results and the HUD."""
    k = action.kind
    amount = action.amount if action.amount in (0, 1, 2) else 1
    if k == ActionKind.OPEN_APP:
        return f"open {action.app!r}"
    if k == ActionKind.QUIT_APP:
        return f"quit {action.app!r}"
    if k == ActionKind.CLICK:
        el = _element(snap, action.target_eid)
        what = _el_text(el) if el else (action.target_label or action.target_eid or "?")
        return f"click {what} ({action.target_eid})"
    if k == ActionKind.TYPE_TEXT:
        el = _element(snap, action.target_eid) or _focused(snap)
        where = f" into {_el_text(el)}" if el else ""
        return f"type {_clip(action.text or '')!r} ({len(action.text or '')} chars){where}"
    if k == ActionKind.SEARCH_WEB:
        return f"search the web for {_clip(action.text or '')!r}: {build_search_url(search_url, action.text)}"
    if k == ActionKind.OPEN_URL:
        return f"open {build_url(action.url)}"
    if k == ActionKind.PRESS_KEY:
        return f"press {action.key}"
    if k in (ActionKind.SCROLL_DOWN, ActionKind.SCROLL_UP):
        d = "down" if k == ActionKind.SCROLL_DOWN else "up"
        return f"scroll {d} {SCROLL_LEVELS[amount]}"
    if k == ActionKind.OPEN_FOLDER:
        return f"open folder {FOLDER_PATHS.get(action.folder or '', action.folder)}"
    if k == ActionKind.GO_BACK:
        return f"go back ({SHORTCUTS['go_back']})"
    if k == ActionKind.NEW_TAB:
        return f"open a new tab ({SHORTCUTS['new_tab']})"
    if k == ActionKind.CLOSE_TAB:
        return f"close the tab ({SHORTCUTS['close_tab']})"
    if k == ActionKind.UNDO:
        return f"undo ({SHORTCUTS['undo']})"
    return f"{k.value}: nothing to execute (handled by the controller)"


# ---------------------------------------------------------------- the only side-effecting layer

AX_OK = 0
AX_TIMEOUT = -25204


class MacUI:
    """Everything that reads from or writes to other apps. Imports pyobjc lazily.

    Reads go through here too (not straight to the observe helpers) so a test double can stand in
    for a whole app.
    """

    def __init__(self, post_to_pid: bool = True):
        import ApplicationServices as AXmod
        import Quartz
        from AppKit import NSRunningApplication, NSURL, NSWorkspace, NSWorkspaceOpenConfiguration

        from jev_local.harness import observe

        self.AX, self.Q, self.obs = AXmod, Quartz, observe
        self._NSRunningApplication = NSRunningApplication
        self._NSURL = NSURL
        self._ws = NSWorkspace.sharedWorkspace()
        self._OpenCfg = NSWorkspaceOpenConfiguration
        self.post_to_pid = post_to_pid
        # HID-state source + explicit flags: our events never inherit a modifier the user is
        # physically holding (e.g. a push-to-talk key).
        self._src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)

    # ---- AX reads
    def get(self, el: Any, attr: str) -> Any:
        return self.obs.ax_value(el, attr)

    def actions(self, el: Any) -> tuple[str, ...]:
        return self.obs.ax_actions(el)

    def settable(self, el: Any, attr: str) -> bool:
        return self.obs.ax_settable(el, attr)

    def frame(self, el: Any) -> tuple[float, float, float, float] | None:
        return self.obs.ax_frame(el)

    def is_alive(self, el: Any) -> bool:
        err, _ = self.obs.ax_get(el, "AXRole")
        return err != self.obs.AX_STALE

    def element_at(self, pid: int, x: float, y: float) -> Any | None:
        """Topmost AX element at a screen point (system-wide hit test, which sees occlusion)."""
        err, el = self.AX.AXUIElementCopyElementAtPosition(self.AX.AXUIElementCreateSystemWide(), x, y, None)
        if err == AX_OK:
            return el
        # System-wide hit testing can fail (-25204) in some sessions; fall back to checking that
        # the topmost window under the point belongs to the app, then hit-test inside the app.
        if self.window_owner_at(x, y) != pid:
            return None
        err, el = self.AX.AXUIElementCopyElementAtPosition(self.AX.AXUIElementCreateApplication(pid), x, y, None)
        return el if err == AX_OK else None

    def window_owner_at(self, x: float, y: float) -> int | None:
        Q = self.Q
        info = Q.CGWindowListCopyWindowInfo(
            Q.kCGWindowListOptionOnScreenOnly | Q.kCGWindowListExcludeDesktopElements, Q.kCGNullWindowID
        ) or []
        for w in info:  # front to back
            if w.get("kCGWindowLayer") != 0 or float(w.get("kCGWindowAlpha", 1.0)) == 0.0:
                continue
            b = w.get("kCGWindowBounds") or {}
            if b.get("X", 0) <= x < b.get("X", 0) + b.get("Width", 0) and b.get("Y", 0) <= y < b.get("Y", 0) + b.get("Height", 0):
                return int(w.get("kCGWindowOwnerPID", 0))
        return None

    def parent(self, el: Any) -> Any | None:
        return self.obs.ax_value(el, "AXParent")

    def screen_locked(self) -> bool:
        return self.obs.screen_locked()

    def is_running(self, pid: int) -> bool:
        ra = self._NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        return ra is not None and not ra.isTerminated()

    # ---- AX writes
    def perform(self, el: Any, action: str) -> int:
        return int(self.AX.AXUIElementPerformAction(el, action))

    def set_attr(self, el: Any, attr: str, value: Any) -> int:
        return int(self.AX.AXUIElementSetAttributeValue(el, attr, value))

    def set_text_range(self, el: Any, location: int, length: int) -> int:
        rng = self.AX.AXValueCreate(self.AX.kAXValueTypeCFRange, self.Q.CFRangeMake(location, length))
        return int(self.AX.AXUIElementSetAttributeValue(el, "AXSelectedTextRange", rng))

    # ---- events
    def _post(self, ev: Any, pid: int | None) -> None:
        if pid and self.post_to_pid:
            self.Q.CGEventPostToPid(pid, ev)
        else:
            self.Q.CGEventPost(self.Q.kCGHIDEventTap, ev)

    def post_key(self, stroke: KeyStroke, pid: int | None) -> None:
        Q = self.Q
        for down in (True, False):
            ev = Q.CGEventCreateKeyboardEvent(self._src, stroke.keycode, down)
            Q.CGEventSetFlags(ev, stroke.flags)
            self._post(ev, pid)
            if down:
                time.sleep(0.008)

    def type_unicode(self, text: str, pid: int | None, cancel: threading.Event | None = None) -> int:
        """Type text as unicode key events; returns the number of UTF-16 units posted."""
        Q = self.Q
        sent = 0
        lines = text.split("\n")
        for li, line in enumerate(lines):
            for chunk in chunk_utf16(line):
                if cancel is not None and cancel.is_set():
                    return sent
                n = utf16_len(chunk)
                for down in (True, False):
                    ev = Q.CGEventCreateKeyboardEvent(self._src, 0, down)
                    Q.CGEventSetFlags(ev, 0)
                    Q.CGEventKeyboardSetUnicodeString(ev, n, chunk)
                    self._post(ev, pid)
                sent += n
                time.sleep(0.004)
            if li < len(lines) - 1:
                self.post_key(KEYMAP["return"], pid)
                sent += 1
        return sent

    def scroll(self, dy: int, unit: str, point: tuple[float, float] | None, pid: int | None) -> None:
        """dy > 0 scrolls up (content moves down), as CGEvent wheel deltas do."""
        Q = self.Q
        u = Q.kCGScrollEventUnitLine if unit == "line" else Q.kCGScrollEventUnitPixel
        ev = Q.CGEventCreateScrollWheelEvent(self._src, u, 1, int(dy))
        if point is not None:
            Q.CGEventSetLocation(ev, Q.CGPointMake(point[0], point[1]))
        self._post(ev, pid)

    def click_at(self, x: float, y: float) -> None:
        """Left click at a screen point, then put the pointer back where the user left it."""
        Q = self.Q
        saved = Q.CGEventGetLocation(Q.CGEventCreate(None))
        pt = Q.CGPointMake(x, y)
        for t in (Q.kCGEventMouseMoved, Q.kCGEventLeftMouseDown, Q.kCGEventLeftMouseUp):
            ev = Q.CGEventCreateMouseEvent(self._src, t, pt, Q.kCGMouseButtonLeft)
            if t != Q.kCGEventMouseMoved:
                Q.CGEventSetIntegerValueField(ev, Q.kCGMouseEventClickState, 1)
            Q.CGEventPost(Q.kCGHIDEventTap, ev)  # mouse events route by location, not pid
            time.sleep(0.01)
        Q.CGWarpMouseCursorPosition(saved)
        Q.CGAssociateMouseAndMouseCursorPosition(True)

    # ---- apps, URLs, folders
    def open_app(self, path: str, timeout_s: float = 5.0) -> tuple[bool, str]:
        done = threading.Event()
        box: dict[str, Any] = {}

        def handler(app: Any, error: Any) -> None:
            box["app"], box["error"] = app, error
            done.set()

        cfg = self._OpenCfg.configuration()
        cfg.setActivates_(True)
        self._ws.openApplicationAtURL_configuration_completionHandler_(self._NSURL.fileURLWithPath_(path), cfg, handler)
        if not done.wait(timeout_s):
            return True, "launch requested (no completion yet)"
        if box.get("error") is not None:
            return False, str(box["error"].localizedDescription())
        return True, "opened"

    def open_url(self, url: str) -> bool:
        u = self._NSURL.URLWithString_(url)
        return bool(u is not None and self._ws.openURL_(u))

    def open_folder(self, path: str) -> bool:
        return bool(self._ws.openURL_(self._NSURL.fileURLWithPath_isDirectory_(path, True)))

    def terminate(self, pid: int) -> bool:
        ra = self._NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        return bool(ra is not None and ra.terminate())  # never forceTerminate: the app may ask to save

    def hide(self, pid: int) -> bool:
        ra = self._NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        return bool(ra is not None and ra.hide())


# ---------------------------------------------------------------- executor

_UI_ACTIONS = frozenset({
    ActionKind.CLICK, ActionKind.TYPE_TEXT, ActionKind.PRESS_KEY, ActionKind.SCROLL_DOWN, ActionKind.SCROLL_UP,
    ActionKind.GO_BACK, ActionKind.NEW_TAB, ActionKind.CLOSE_TAB, ActionKind.UNDO,
})


class _Fail(Exception):
    """Internal: abort the current action with a readable reason."""


class Executor:
    def __init__(self, observer: Any, cfg: HarnessConfig, ui: MacUI | None = None, *, settle_s: float = 0.3):
        self.observer = observer
        self.cfg = cfg
        self._ui = ui
        self.settle_s = settle_s  # how long to poll for a visible effect after an action

    @property
    def ui(self) -> MacUI:
        if self._ui is None:  # created on first live use, so dry-run never loads Quartz
            self._ui = MacUI()
        return self._ui

    # ------------------------------------------------------------ public API

    def run(self, action: Action, snap: Snapshot | None, cancel: threading.Event) -> ExecResult:
        """Execute (or, in dry-run, describe) one action. Never raises."""
        t0 = time.monotonic()
        dry = self.cfg.dry_run
        try:
            if cancel.is_set():
                return self._res(False, False, "cancelled before start", t0, dry=dry)
            problem = self.validate(action, snap)
            if dry:
                if problem:
                    return self._res(False, False, f"would fail: {problem}", t0, dry=True)
                return self._res(True, False, "would " + describe(action, snap, self.cfg.search_url), t0, dry=True)
            if problem:
                return self._res(False, False, problem, t0)
            handler = self._handlers().get(action.kind)
            if handler is None:
                return self._res(True, False, describe(action, snap), t0)
            if self._locked():
                return self._res(False, False, "refused: the screen is locked", t0)
            try:
                if action.kind in _UI_ACTIONS:
                    self._front_guard(snap)
                ok, changed, detail, token = handler(action, snap, cancel)
            finally:
                self._invalidate()
        except _Fail as e:
            return self._res(False, False, str(e), t0)
        except Exception as e:  # an executor bug must never take the harness down
            return self._res(False, False, f"error: {type(e).__name__}: {e}", t0, dry=dry)
        return self._res(ok, changed, detail, t0, token=token)

    def undo(self, record: ActionRecord) -> ExecResult:
        """Revert a previous action using its undo token (see _undo_plan). Never raises."""
        t0 = time.monotonic()
        try:
            tok = record.undo_token or {}
            if self.cfg.dry_run:
                plan = self._undo_plan(record.action, tok)
                return self._res(True, False, f"would undo {record.action.describe()}: {plan}", t0, dry=True)
            if self._locked():
                return self._res(False, False, "refused: the screen is locked", t0)
            try:
                ok, changed, detail = self._undo_live(record.action, tok)
            finally:
                self._invalidate()
        except _Fail as e:
            return self._res(False, False, str(e), t0)
        except Exception as e:
            return self._res(False, False, f"error: {type(e).__name__}: {e}", t0, dry=self.cfg.dry_run)
        return self._res(ok, changed, detail, t0)

    def validate(self, action: Action, snap: Snapshot | None) -> str | None:
        """Checks that need no UI access. Returns a reason to refuse, or None."""
        k = action.kind
        if k in _UI_ACTIONS and snap is not None:
            hit = denied_app(snap.app_name, snap.bundle_id, self.cfg.allow_apps)
            if hit:
                return f"refused: {snap.app_name!r} is on the deny list ({hit})"
        if k in (ActionKind.OPEN_APP, ActionKind.QUIT_APP):
            hit = denied_app(action.app, None, self.cfg.allow_apps)
            if hit:
                return f"refused: {action.app!r} is on the deny list ({hit})"
        if k == ActionKind.OPEN_APP:
            if not action.app or action.app == "none":
                return "no app given"
            if apps.app_path(action.app) is None:
                return f"no installed app named {action.app!r}"
        elif k == ActionKind.QUIT_APP:
            if not action.app or action.app == "none":
                return "no app given"
        elif k == ActionKind.CLICK:
            if not action.target_eid:
                return "no target element"
            if snap is not None:
                el = _element(snap, action.target_eid)
                if el is None:
                    return f"target {action.target_eid} is not in the snapshot"
                if not el.enabled:
                    return f"target {_el_text(el)} is disabled"
        elif k == ActionKind.TYPE_TEXT:
            if not action.text:
                return "nothing to type"
            if action.target_eid and snap is not None and _element(snap, action.target_eid) is None:
                return f"target {action.target_eid} is not in the snapshot"
            el = _element(snap, action.target_eid) or _focused(snap)
            if is_secure_element(el):
                return "refused: the field is a password/secure field"
        elif k == ActionKind.SEARCH_WEB:
            if not action.text:
                return "nothing to search for"
            if build_search_url(self.cfg.search_url, action.text) is None:
                return f"search_url {self.cfg.search_url!r} is not a valid http(s) template"
        elif k == ActionKind.OPEN_URL:
            if build_url(action.url) is None:
                return f"refused url {action.url!r} (only http/https web addresses)"
        elif k == ActionKind.PRESS_KEY:
            if not action.key or action.key not in KEYMAP:
                return f"key {action.key!r} is not in the key map"
        elif k in (ActionKind.SCROLL_DOWN, ActionKind.SCROLL_UP):
            if action.amount is not None and action.amount not in (0, 1, 2):
                return f"bad scroll amount {action.amount}"
        elif k == ActionKind.OPEN_FOLDER:
            if folder_path(action.folder) is None:
                return f"unknown folder {action.folder!r}"
        return None

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _res(ok: bool, changed: bool, detail: str, t0: float, *, dry: bool = False, token: dict | None = None) -> ExecResult:
        return ExecResult(
            ok=ok, changed=changed, detail=detail, elapsed_ms=(time.monotonic() - t0) * 1000,
            undo_token=token, dry_run=dry,
        )

    def _invalidate(self) -> None:
        inv = getattr(self.observer, "invalidate", None)
        if callable(inv):
            inv()

    def _locked(self) -> bool:
        return self.ui.screen_locked()

    def _front(self) -> Any | None:
        fm = getattr(self.observer, "frontmost", None)
        return fm() if callable(fm) else None

    def _front_guard(self, snap: Snapshot | None) -> None:
        """Keys, clicks and typing go to the app the user was looking at when they spoke, or nowhere."""
        if snap is None or not snap.pid:
            return
        front = self._front()
        if front is not None and front.pid != snap.pid:
            raise _Fail(f"frontmost app changed since the snapshot ({snap.app_name!r} -> {front.name!r})")

    def _target_pid(self, snap: Snapshot | None) -> int | None:
        if snap is not None and snap.pid:
            return snap.pid
        front = self._front()
        return front.pid if front is not None else None

    def _wait(self, cond: Callable[[], bool], timeout_s: float, cancel: threading.Event | None = None) -> bool:
        end = time.monotonic() + timeout_s
        while True:
            if cond():
                return True
            if time.monotonic() >= end or (cancel is not None and cancel.is_set()):
                return False
            time.sleep(0.05)

    def _handlers(self) -> dict[ActionKind, Callable[..., tuple[bool, bool, str, dict | None]]]:
        return {
            ActionKind.OPEN_APP: self._open_app,
            ActionKind.QUIT_APP: self._quit_app,
            ActionKind.CLICK: self._click,
            ActionKind.TYPE_TEXT: self._type,
            ActionKind.SEARCH_WEB: self._search_web,
            ActionKind.OPEN_URL: self._open_url,
            ActionKind.PRESS_KEY: self._press_key,
            ActionKind.SCROLL_DOWN: self._scroll,
            ActionKind.SCROLL_UP: self._scroll,
            ActionKind.OPEN_FOLDER: self._open_folder,
            ActionKind.GO_BACK: self._shortcut("go_back"),
            ActionKind.NEW_TAB: self._shortcut("new_tab"),
            ActionKind.CLOSE_TAB: self._shortcut("close_tab"),
            ActionKind.UNDO: self._shortcut("undo"),
        }

    # ------------------------------------------------------------ apps, URLs, folders

    def _open_app(self, a: Action, snap: Snapshot | None, cancel: threading.Event):
        path = apps.app_path(a.app or "")
        if path is None:
            raise _Fail(f"no installed app named {a.app!r}")
        bid = apps.bundle_id(path) or ""
        front = self._front()
        if front is not None and (front.path == path or (bid and front.bundle_id == bid)):
            return True, False, f"{a.app} is already frontmost", None
        was_running = any(r.bundle_id == bid or r.path == path for r in apps.running(regular_only=False))
        ok, detail = self.ui.open_app(path)
        if not ok:
            raise _Fail(f"could not open {a.app}: {detail}")

        def is_front() -> bool:
            f = self._front()
            return f is not None and (f.path == path or (bool(bid) and f.bundle_id == bid))

        changed = self._wait(is_front, 2.0, cancel)
        f = self._front()
        token = {"kind": "open_app", "bundle_id": bid, "path": path, "launched": not was_running,
                 "pid": f.pid if (f and changed) else None, "t": time.time()}
        return True, changed, f"opened {a.app}" + ("" if changed else " (not frontmost after 2 s)"), token

    def _quit_app(self, a: Action, snap: Snapshot | None, cancel: threading.Event):
        r = apps.find_running(a.app or "")
        if r is None:
            raise _Fail(f"{a.app!r} is not running")
        if r.pid == os.getpid():
            raise _Fail("refused: that is the harness itself")
        if not self.ui.terminate(r.pid):
            raise _Fail(f"{r.name} refused to quit")
        gone = self._wait(lambda: not self.ui.is_running(r.pid), 2.0, cancel)
        return True, gone, f"asked {r.name} to quit" + ("" if gone else " (still running: it may be asking to save)"), None

    def _opened_in(self, kind: str, ok: Callable[[Any], bool], cancel: threading.Event) -> dict:
        """Undo token for something opened in another app (a browser tab, a Finder window): which app
        received it, so "undo" closes a window there and never in whatever app is frontmost later."""
        self._wait(lambda: (f := self._front()) is not None and ok(f), 1.5, cancel)
        f = self._front()
        if f is None or not ok(f):
            return {"kind": kind, "pid": None, "bundle_id": None}
        return {"kind": kind, "pid": f.pid, "bundle_id": f.bundle_id}

    def _open_url(self, a: Action, snap: Snapshot | None, cancel: threading.Event):
        url = build_url(a.url)
        if url is None:
            raise _Fail(f"refused url {a.url!r}")
        if not self.ui.open_url(url):
            raise _Fail(f"could not open {url}")
        return True, True, f"opened {url}", self._opened_in("open_url", _is_browser, cancel)

    def _search_web(self, a: Action, snap: Snapshot | None, cancel: threading.Event):
        url = build_search_url(self.cfg.search_url, a.text)
        if url is None:
            raise _Fail("bad search")
        if not self.ui.open_url(url):
            raise _Fail(f"could not open {url}")
        return True, True, f"searched: {url}", self._opened_in("open_url", _is_browser, cancel)

    def _open_folder(self, a: Action, snap: Snapshot | None, cancel: threading.Event):
        path = folder_path(a.folder)
        if path is None or not os.path.isdir(path):
            raise _Fail(f"folder {a.folder!r} does not exist")
        if not self.ui.open_folder(path):
            raise _Fail(f"could not open {path}")
        return True, True, f"opened {path}", self._opened_in("open_folder", _is_finder, cancel)

    # ------------------------------------------------------------ click

    def _resolve(self, snap: Snapshot | None, eid: str | None) -> Any:
        if snap is None or not eid:
            raise _Fail("no target")
        ref = self.observer.resolve(snap, eid)
        if ref is None:
            raise _Fail(f"target {eid} is no longer on screen")
        return ref

    def _probe(self, ref: Any) -> tuple[Any, ...]:
        ui = self.ui
        if not ui.is_alive(ref):
            return ("gone",)
        return (ui.get(ref, "AXValue"), ui.get(ref, "AXSelected"), ui.get(ref, "AXFocused"), ui.get(ref, "AXExpanded"))

    def _settled_change(self, ref: Any, before: tuple[Any, ...], cancel: threading.Event) -> bool:
        return self._wait(lambda: self._probe(ref) != before, self.settle_s, cancel)

    def _inside(self, hit: Any, target: Any, max_up: int = 12) -> bool:
        node = hit
        for _ in range(max_up):
            if node is None:
                return False
            if node == target:
                return True
            node = self.ui.parent(node)
        return False

    def _click(self, a: Action, snap: Snapshot | None, cancel: threading.Event):
        ref = self._resolve(snap, a.target_eid)
        el = _element(snap, a.target_eid)
        name = _el_text(el) if el else a.target_eid
        ui = self.ui
        if ui.get(ref, "AXEnabled") is False:
            raise _Fail(f"{name} is disabled")
        before = self._probe(ref)
        acts = ui.actions(ref)
        method = None
        errors: list[str] = []
        for act in ("AXPress", "AXPick"):
            if act in acts:
                err = ui.perform(ref, act)
                if err == AX_OK:
                    method = act
                    break
                if err == AX_TIMEOUT:
                    # Pressing a menu or a button that opens a modal blocks the app's AX reply;
                    # the press itself was delivered.
                    method = f"{act} (no reply)"
                    break
                errors.append(f"{act} failed ({err})")
        if method is None and ui.settable(ref, "AXSelected"):
            if ui.set_attr(ref, "AXSelected", True) == AX_OK:
                method = "AXSelected"
        if method is None:
            if cancel.is_set():
                raise _Fail("cancelled")
            frame = ui.frame(ref)
            if not frame or frame[2] <= 0 or frame[3] <= 0:
                raise _Fail(f"cannot click {name}: no press action and no on-screen frame")
            x, y = frame[0] + frame[2] / 2, frame[1] + frame[3] / 2
            hit = ui.element_at(snap.pid if snap else 0, x, y)
            if not self._inside(hit, ref):
                raise _Fail(f"cannot click {name}: something else is on top of it")
            ui.click_at(x, y)
            method = "pointer click"
        changed = self._settled_change(ref, before, cancel)
        extra = f"; {', '.join(errors)}" if errors else ""
        return True, changed, f"clicked {name} via {method}{extra}", None

    # ------------------------------------------------------------ type

    def _type(self, a: Action, snap: Snapshot | None, cancel: threading.Event):
        text = a.text or ""
        ui = self.ui
        if a.target_eid:
            ref = self._resolve(snap, a.target_eid)
            if ui.get(ref, "AXFocused") is not True and ui.settable(ref, "AXFocused"):
                ui.set_attr(ref, "AXFocused", True)
        else:
            ref = self.observer.focused_element()
        if ref is None:
            raise _Fail("no focused text field to type into")
        # Re-check live: the snapshot may predate focus moving into a password field.
        role, sub = ui.get(ref, "AXRole"), ui.get(ref, "AXSubrole")
        names = " ".join(str(ui.get(ref, k) or "") for k in ("AXTitle", "AXDescription", "AXPlaceholderValue"))
        if "AXSecureTextField" in (role, sub) or SECURE_FIELD_RE.search(names):
            raise _Fail("refused: the focused field is a password/secure field")
        pid = self._target_pid(snap)
        before = ui.get(ref, "AXValue")
        before = before if isinstance(before, str) else None
        method = None
        verified = False
        if ui.settable(ref, "AXSelectedText") and ui.set_attr(ref, "AXSelectedText", text) == AX_OK:
            if before is None:
                method = "AXSelectedText"
            elif self._wait(lambda: ui.get(ref, "AXValue") != before, max(self.settle_s, 0.2), cancel):
                method, verified = "AXSelectedText", True
            # else: the app accepted the write but ignored it (common in web fields); type instead
        if method is None:
            if cancel.is_set():
                raise _Fail("cancelled")
            sent = ui.type_unicode(text, pid, cancel)
            if sent < utf16_len(text):
                raise _Fail(f"cancelled after typing {sent} of {utf16_len(text)} characters")
            method = "keystrokes"
        after = ui.get(ref, "AXValue")
        if isinstance(after, str) and before is not None:
            verified = verified or (after != before and text in after)
        token = {"kind": "type_text", "method": method, "n16": utf16_len(text), "digest": text_digest(text), "pid": pid}
        note = "" if verified else " (unverified)"
        return True, True, f"typed {len(text)} chars via {method}{note}", token

    # ------------------------------------------------------------ keys and scroll

    def _press_key(self, a: Action, snap: Snapshot | None, cancel: threading.Event):
        stroke = KEYMAP[a.key or ""]
        self.ui.post_key(stroke, self._target_pid(snap))
        return True, True, f"pressed {stroke.label}", {"kind": "press_key", "key": stroke.label}

    def _shortcut(self, name: str) -> Callable[..., tuple[bool, bool, str, dict | None]]:
        def run(a: Action, snap: Snapshot | None, cancel: threading.Event):
            label = SHORTCUTS[name]
            pid = self._target_pid(snap)
            self.ui.post_key(KEYMAP[label], pid)
            token = {"kind": name, "pid": pid, "bundle_id": snap.bundle_id if snap else ""}
            return True, True, f"{name.replace('_', ' ')} ({label})", token

        return run

    def _scroll_area(self, snap: Snapshot | None, eid: str | None) -> tuple[Any, tuple[float, float, float, float]] | None:
        getter = getattr(self.observer, "scroll_areas", None)
        areas = getter(snap) if (callable(getter) and snap is not None) else []
        if not areas:
            return None
        anchor = _element(snap, eid) or _focused(snap)
        if anchor is not None and anchor.frame[2] > 0:
            cx, cy = anchor.frame[0] + anchor.frame[2] / 2, anchor.frame[1] + anchor.frame[3] / 2
            inside = [(r, f) for r, f, _ in areas if f[0] <= cx <= f[0] + f[2] and f[1] <= cy <= f[1] + f[3]]
            if inside:
                return min(inside, key=lambda rf: rf[1][2] * rf[1][3])  # innermost
        r, f, _ = max(areas, key=lambda t: t[1][2] * t[1][3])  # the main content area
        return r, f

    def _scroll(self, a: Action, snap: Snapshot | None, cancel: threading.Event):
        up = a.kind == ActionKind.SCROLL_UP
        amount = a.amount if a.amount in (0, 1, 2) else 1
        ui = self.ui
        pid = self._target_pid(snap)
        area = self._scroll_area(snap, a.target_eid)
        bar = ui.get(area[0], "AXVerticalScrollBar") if area else None
        before = ui.get(bar, "AXValue") if bar is not None else None
        method = None
        if amount == 2 and bar is not None and ui.settable(bar, "AXValue"):
            if ui.set_attr(bar, "AXValue", 0.0 if up else 1.0) == AX_OK:
                method = "scroll bar"
        if method is None:
            point = (area[1][0] + area[1][2] / 2, area[1][1] + area[1][3] / 2) if area else None
            sign = 1 if up else -1
            if amount == 0:
                ui.scroll(3 * sign, "line", point, pid)
            elif amount == 1:
                page = int(area[1][3] * 0.85) if area else 600
                ui.scroll(max(page, 100) * sign, "pixel", point, pid)
            else:
                for _ in range(4):
                    ui.scroll(20000 * sign, "pixel", point, pid)
            method = "wheel"
        changed = True
        if bar is not None and isinstance(before, (int, float)):
            changed = self._wait(lambda: ui.get(bar, "AXValue") != before, self.settle_s, cancel)
        d = "up" if up else "down"
        token = {"kind": "scroll", "dir": d, "amount": amount, "pid": pid}
        return True, changed, f"scrolled {d} {SCROLL_LEVELS[amount]} via {method}", token

    # ------------------------------------------------------------ undo

    @staticmethod
    def _undo_plan(action: Action, tok: dict) -> str:
        kind = tok.get("kind")
        if kind == "type_text":
            return f"select the {tok.get('n16')} typed characters and delete them (else {SHORTCUTS['undo']})"
        if kind == "open_app":
            return "quit it (launched < 10 s ago)" if tok.get("launched") else "hide it"
        if kind in ("new_tab", "open_url", "open_folder"):
            return f"close it ({SHORTCUTS['close_tab']})"
        if kind == "close_tab":
            return f"reopen it ({SHORTCUTS['reopen_tab']})"
        if kind == "scroll":
            return f"scroll {'down' if tok.get('dir') == 'up' else 'up'} by the same amount"
        return f"press {SHORTCUTS['undo']}"

    def _undo_live(self, action: Action, tok: dict) -> tuple[bool, bool, str]:
        kind = tok.get("kind")
        ui = self.ui
        front = self._front()
        front_pid = front.pid if front is not None else None
        if front is not None and kind != "open_app":
            # Undo posts keys to the frontmost app: same deny list as any other in-app action.
            hit = denied_app(front.name, front.bundle_id, self.cfg.allow_apps)
            if hit:
                raise _Fail(f"refused: {front.name!r} is on the deny list ({hit})")
        if kind == "type_text":
            done = self._undo_typing(tok)
            if done:
                return True, True, done
        elif kind == "open_app":
            r = next((r for r in apps.running(regular_only=False) if r.bundle_id == tok.get("bundle_id")), None)
            if r is None:
                return True, False, "the app is no longer running"
            if tok.get("launched") and time.time() - float(tok.get("t", 0)) < 10 and r.pid != os.getpid():
                return ui.terminate(r.pid), True, f"quit {r.name} (we had just launched it)"
            return ui.hide(r.pid), True, f"hid {r.name}"
        elif kind in ("new_tab", "open_url", "open_folder"):
            if kind == "new_tab" and tok.get("pid") and front_pid != tok.get("pid"):
                raise _Fail("the app that got the new tab is no longer frontmost")
            if kind in ("open_url", "open_folder"):
                # cmd+w goes to the frontmost app: only close if that is still the browser / Finder
                # window that received it (never the Mail message being written).
                want = _is_browser if kind == "open_url" else _is_finder
                if not tok.get("pid") or front is None or front_pid != tok.get("pid") or not want(front):
                    where = "browser" if kind == "open_url" else "Finder window"
                    raise _Fail(f"not closing anything: the {where} it opened in is no longer frontmost")
            ui.post_key(KEYMAP[SHORTCUTS["close_tab"]], front_pid)
            return True, True, f"closed it ({SHORTCUTS['close_tab']})"
        elif kind == "close_tab":
            if tok.get("bundle_id") not in BROWSER_BUNDLES:
                raise _Fail("can only reopen closed tabs in a browser")
            ui.post_key(KEYMAP[SHORTCUTS["reopen_tab"]], front_pid)
            return True, True, f"reopened the tab ({SHORTCUTS['reopen_tab']})"
        elif kind == "scroll":
            back = ActionKind.SCROLL_DOWN if tok.get("dir") == "up" else ActionKind.SCROLL_UP
            inv = Action(kind=back, source_vid="undo", confidence=1.0, amount=tok.get("amount"))
            ok, changed, detail, _ = self._scroll(inv, None, threading.Event())
            return ok, changed, f"undo: {detail}"
        ui.post_key(KEYMAP[SHORTCUTS["undo"]], front_pid)
        return True, True, f"pressed {SHORTCUTS['undo']}"

    def _undo_typing(self, tok: dict) -> str | None:
        """Select exactly the text we typed (verified by digest) and delete it. None if unsure."""
        ui = self.ui
        ref = self.observer.focused_element()
        n = int(tok.get("n16") or 0)
        if ref is None or n <= 0:
            return None
        rng = ui.get(ref, "AXSelectedTextRange")
        if not (isinstance(rng, tuple) and len(rng) == 2 and rng[1] == 0 and rng[0] >= n):
            return None
        caret = rng[0]
        if ui.set_text_range(ref, caret - n, n) != AX_OK:
            return None
        sel = ui.get(ref, "AXSelectedText")
        if isinstance(sel, str) and text_digest(sel) == tok.get("digest"):
            if ui.set_attr(ref, "AXSelectedText", "") == AX_OK:
                return f"deleted the {n} typed characters"
        ui.set_text_range(ref, caret, 0)  # not our text: put the caret back and fall back to cmd+z
        return None
