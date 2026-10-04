"""A simulated screen for dry-run demos: the frontmost "app" follows the dry-run actions.

In dry-run nothing really opens, so with the real Observer "open textedit and type hello" would
judge the typing against whatever terminal is in front (and the safety layer rightly refuses to
type into a terminal). `SimWorld` stands in for the screen instead: `open_app` / `switch_app`
make a plausible window of that app frontmost, `quit_app` closes it, and every snapshot is built
from that state. `SimExecutor` wraps the real (dry-run) `Executor`, so validation and the "would
..." descriptions are the real ones, and then updates the world. Nothing touches the UI.
"""

from __future__ import annotations

import threading
from typing import Any

from jev_local.harness.fakes import make_snapshot
from jev_local.harness.types import Action, ActionKind, ActionRecord, ExecResult, Snapshot

_MENU = [("menu bar item", "File"), ("menu bar item", "Edit"), ("menu bar item", "View"), ("menu bar item", "Window")]

SCREENS: dict[str, tuple[str | None, list[tuple[str, str]], str | None]] = {
    # app: (window title, elements, focused label)
    "Finder": ("Recents", [("button", "Back"), ("button", "Forward"), ("search field", "Search"),
                           ("row", "Desktop"), ("row", "Documents"), ("row", "Downloads"), ("row", "Applications")],
               None),
    "TextEdit": ("Untitled", [("text area", "Untitled document"), ("pop-up button", "Font"),
                              ("button", "Bold"), ("button", "Italic")], "Untitled document"),
    "Notes": ("Notes", [("button", "New Note"), ("search field", "Search"), ("row", "Groceries"),
                        ("row", "Ideas"), ("text area", "Note body")], "Note body"),
    "Safari": ("Start Page", [("button", "Back"), ("button", "Forward"), ("text field", "Search or enter website name"),
                              ("button", "New Tab"), ("link", "Favorites"), ("link", "Wikipedia"),
                              ("link", "Apple"), ("button", "Reader")], None),
    "Google Chrome": ("New Tab", [("button", "Back"), ("text field", "Address and search bar"),
                                  ("button", "New Tab"), ("link", "Gmail")], "Address and search bar"),
    "Mail": ("Inbox", [("button", "New Message"), ("button", "Reply"), ("button", "Archive"), ("button", "Delete"),
                       ("search field", "Search"), ("row", "Weekly update")], None),
}
_GENERIC: tuple[str | None, list[tuple[str, str]], str | None] = (None, [("button", "OK"), ("button", "Cancel")], None)


class SimWorld:
    """Observer stand-in whose frontmost app changes with executed (dry-run) actions."""

    def __init__(self, front: str = "Finder"):
        self._lock = threading.Lock()
        self.running: list[str] = ["Finder"] if front == "Finder" else [front, "Finder"]
        self.typed: dict[str, str] = {}
        self.url: dict[str, str] = {}
        self.scroll: dict[str, int] = {}
        self.events: list[str] = []
        self.prefetch_calls = 0
        self._snap: Snapshot | None = None

    # ------------------------------------------------------------ observer protocol

    @property
    def front(self) -> str:
        return self.running[0] if self.running else "Finder"

    def snapshot(self, allow_cached: bool = True, max_age_s: float = 1.5, **_: Any) -> Snapshot:
        with self._lock:
            if self._snap is None or not allow_cached:
                self._snap = self._build()
            return self._snap

    def prefetch(self) -> None:
        self.prefetch_calls += 1

    def invalidate(self) -> None:
        with self._lock:
            self._snap = None

    def resolve(self, snap: Snapshot, eid: str) -> Any:
        return next((e for e in snap.elements if e.eid == eid), None)

    def focused_element(self) -> Any:
        s = self.snapshot()
        return next((e for e in s.elements if e.focused), None)

    def running_apps(self) -> list[str]:
        return list(self.running)

    # ------------------------------------------------------------ world updates

    def apply(self, a: Action) -> None:
        with self._lock:
            k = a.kind
            if k == ActionKind.OPEN_APP and a.app:
                if a.app in self.running:
                    self.running.remove(a.app)
                self.running.insert(0, a.app)
            elif k == ActionKind.QUIT_APP and a.app and a.app in self.running and a.app != "Finder":
                self.running.remove(a.app)
            elif k == ActionKind.TYPE_TEXT and a.text:
                self.typed[self.front] = (self.typed.get(self.front, "") + a.text)[-60:]
            elif k in (ActionKind.OPEN_URL, ActionKind.SEARCH_WEB):
                browser = "Safari" if "Safari" in self.running or "Google Chrome" not in self.running else "Google Chrome"
                if browser in self.running:
                    self.running.remove(browser)
                self.running.insert(0, browser)
                self.url[browser] = a.url or a.text or ""
            elif k in (ActionKind.SCROLL_DOWN, ActionKind.SCROLL_UP):
                self.scroll[self.front] = self.scroll.get(self.front, 0) + (1 if k == ActionKind.SCROLL_DOWN else -1)
            self.events.append(a.describe())
            self._snap = None

    def _build(self) -> Snapshot:
        app = self.front
        title, els, focused = SCREENS.get(app, _GENERIC)
        els = list(els)
        typed = self.typed.get(app)
        if typed and focused:
            els = [(r, l, {"value": typed}) if l == focused else (r, l) for r, l in els]  # type: ignore[misc]
        if app in self.url:
            title = self.url[app][:60]
        return make_snapshot(app, els + _MENU, window_title=title or app, focused=focused)


class SimExecutor:
    """The real dry-run Executor, then the simulated world follows the action."""

    def __init__(self, inner: Any, world: SimWorld):
        self.inner = inner
        self.world = world

    def run(self, action: Action, snap: Snapshot | None, cancel: threading.Event) -> ExecResult:
        r: ExecResult = self.inner.run(action, snap, cancel)
        if r.ok:
            self.world.apply(action)
        return r

    def undo(self, record: ActionRecord) -> ExecResult:
        return self.inner.undo(record)

