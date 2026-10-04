"""Executor tests: dry-run, pure helpers (key map, URLs), and the live dispatch logic against a
recording fake UI. Nothing here posts an event, opens an app or writes an AX attribute."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Any

import pytest

from jev_local.harness import execute as ex
from jev_local.harness.catalog import KEYS
from jev_local.harness.execute import (
    FLAG_ALT,
    FLAG_CMD,
    FLAG_CTRL,
    FLAG_FN,
    FLAG_NUMPAD,
    FLAG_SHIFT,
    KEYMAP,
    Executor,
    build_search_url,
    build_url,
    chunk_utf16,
    folder_path,
    parse_key,
    text_digest,
    utf16_len,
)
from jev_local.harness.types import Action, ActionKind, ActionRecord, Element, HarnessConfig, Snapshot

# ---------------------------------------------------------------- key map


def test_keymap_covers_every_catalog_key():
    for label in KEYS:
        if label == "none":
            assert label not in KEYMAP
        else:
            assert label in KEYMAP, label
            assert KEYMAP[label].label == label


@pytest.mark.parametrize(
    "label,code,flags",
    [
        ("return", 36, 0),
        ("escape", 53, 0),
        ("tab", 48, 0),
        ("space", 49, 0),
        ("delete", 51, 0),
        ("up", 126, FLAG_FN | FLAG_NUMPAD),
        ("down", 125, FLAG_FN | FLAG_NUMPAD),
        ("left", 123, FLAG_FN | FLAG_NUMPAD),
        ("right", 124, FLAG_FN | FLAG_NUMPAD),
        ("page_down", 121, FLAG_FN),
        ("page_up", 116, FLAG_FN),
        ("cmd+a", 0, FLAG_CMD),
        ("cmd+c", 8, FLAG_CMD),
        ("cmd+v", 9, FLAG_CMD),
        ("cmd+x", 7, FLAG_CMD),
        ("cmd+z", 6, FLAG_CMD),
        ("cmd+shift+z", 6, FLAG_CMD | FLAG_SHIFT),
        ("cmd+s", 1, FLAG_CMD),
        ("cmd+f", 3, FLAG_CMD),
        ("cmd+n", 45, FLAG_CMD),
        ("cmd+r", 15, FLAG_CMD),
        ("cmd+[", 33, FLAG_CMD),
        ("cmd+t", 17, FLAG_CMD),
        ("cmd+w", 13, FLAG_CMD),
        ("cmd+shift+t", 17, FLAG_CMD | FLAG_SHIFT),
    ],
)
def test_keymap_codes(label, code, flags):
    k = KEYMAP[label]
    assert (k.keycode, k.flags) == (code, flags)


def test_parse_key_aliases_and_modifiers():
    assert parse_key("Enter").keycode == 36
    assert parse_key("esc").keycode == 53
    assert parse_key("backspace").keycode == 51
    k = parse_key("ctrl+alt+shift+cmd+k")
    assert k.keycode == 40 and k.flags == FLAG_CTRL | FLAG_ALT | FLAG_SHIFT | FLAG_CMD
    assert parse_key("option+left").flags == FLAG_ALT | FLAG_FN | FLAG_NUMPAD
    assert parse_key("f5").keycode == 96


@pytest.mark.parametrize("bad", ["", "none", "hyper+a", "cmd+banana", "cmd++", "cmd+"])
def test_parse_key_rejects(bad):
    with pytest.raises(ValueError):
        parse_key(bad)


# ---------------------------------------------------------------- URLs


@pytest.mark.parametrize(
    "raw,want",
    [
        ("github.com", "https://github.com"),
        ("github.com/anthropics", "https://github.com/anthropics"),
        ("news.ycombinator.com", "https://news.ycombinator.com"),
        ("https://GitHub.com/a?b=1#c", "https://GitHub.com/a?b=1#c"),
        ("HTTP://example.org", "http://example.org"),
        ("http://example.org:8080/x", "http://example.org:8080/x"),
        ("localhost:3000", "http://localhost:3000"),
        ("localhost:3000/api", "http://localhost:3000/api"),
        ("http://127.0.0.1:8765/healthz", "http://127.0.0.1:8765/healthz"),
        ("münchen.de", "https://münchen.de"),
        ("  docs.python.org/3/  ", "https://docs.python.org/3/"),
    ],
)
def test_build_url_accepts(raw, want):
    assert build_url(raw) == want


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "   ",
        "javascript:alert(1)",
        "JavaScript:alert(1)",
        "file:///etc/passwd",
        "file://localhost/Users",
        "mailto:bob@example.com",
        "data:text/html,<b>x</b>",
        "x-apple.systempreferences:com.apple.preference.security",
        "ftp://example.com",
        "http:example.com",
        "https://user:pw@evil.com",
        "google.com@evil.com",
        "not a url",
        "example",
        "https://ex ample.com",
        "https://-bad-.com",
        "https://example.com:99999",
        "vnc://host.local",
        "ssh://host.example.com",
    ],
)
def test_build_url_refuses(raw):
    assert build_url(raw) is None


def test_build_search_url():
    t = "https://www.google.com/search?q={q}"
    assert build_search_url(t, "salt & pepper") == "https://www.google.com/search?q=salt+%26+pepper"
    assert build_search_url(t, "  alan turing ") == "https://www.google.com/search?q=alan+turing"
    assert build_search_url("https://duckduckgo.com/?q={q}", "a/b?c") == "https://duckduckgo.com/?q=a%2Fb%3Fc"
    assert build_search_url(t, "") is None
    assert build_search_url(t, None) is None
    assert build_search_url("https://example.com/search", "x") is None  # no {q}
    assert build_search_url("javascript:{q}", "x") is None


def test_folder_path():
    assert folder_path("downloads") == os.path.expanduser("~/Downloads")
    assert folder_path("Documents") == os.path.expanduser("~/Documents")
    assert folder_path("home") == os.path.expanduser("~")
    assert folder_path("applications") == "/Applications"
    for bad in (None, "", "none", "/etc", "../..", "~/.ssh"):
        assert folder_path(bad) is None


def test_chunk_utf16():
    text = "x" * 45
    chunks = chunk_utf16(text)
    assert [len(c) for c in chunks] == [20, 20, 5]
    emoji = "a" * 19 + "😀" + "b"
    chunks = chunk_utf16(emoji)
    assert chunks == ["a" * 19, "😀b"]  # the surrogate pair is never split
    for s in ["héllo wörld " * 5, "日本語のテキスト" * 4, "👍🏽" * 12, ""]:
        cs = chunk_utf16(s)
        assert "".join(cs) == s
        assert all(utf16_len(c) <= 20 for c in cs)
    assert utf16_len("😀") == 2


# ---------------------------------------------------------------- fixtures


def el(eid: str, role: str = "button", label: str = "", **kw: Any) -> Element:
    kw.setdefault("frame", (100.0, 200.0, 80.0, 24.0))
    return Element(eid=eid, role=role, label=label, **kw)


def snap_of(*elements: Element, pid: int = 4242, focused: str | None = None) -> Snapshot:
    return Snapshot(
        app_name="Mail", bundle_id="com.apple.mail", pid=pid, window_title="Inbox",
        elements=tuple(elements), taken_at=1.0, focused_eid=focused,
    )


def act(kind: ActionKind, **kw: Any) -> Action:
    return Action(kind=kind, source_vid="u1+0", confidence=0.9, **kw)


class Exploding:
    """Any attribute access fails the test: proves a code path never touched it."""

    def __init__(self, what: str):
        object.__setattr__(self, "_what", what)

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"dry-run touched {self._what}.{name}")


@dataclass(frozen=True)
class Front:
    pid: int
    name: str = "Mail"
    bundle_id: str = "com.apple.mail"
    path: str | None = "/System/Applications/Mail.app"


class Ref:
    """A fake AX element."""

    def __init__(self, name: str, actions: tuple[str, ...] = (), settable: tuple[str, ...] = (), **attrs: Any):
        self.name = name
        self.acts = actions
        self.settable = set(settable)
        self.attrs = dict(attrs)
        self.alive = True
        self.parent: Ref | None = None
        self.rc: dict[str, int] = {}  # action/attr -> forced AX error
        self.ignore_writes = False

    def __repr__(self) -> str:
        return f"Ref({self.name})"


@dataclass
class FakeUI:
    calls: list[tuple] = field(default_factory=list)
    locked: bool = False
    hit: Any = None
    running_pids: set[int] = field(default_factory=set)

    # reads
    def get(self, el, attr):
        return None if el is None else el.attrs.get(attr)

    def actions(self, el):
        return el.acts

    def settable(self, el, attr):
        return attr in el.settable

    def frame(self, el):
        return el.attrs.get("frame")

    def is_alive(self, el):
        return el.alive

    def element_at(self, pid, x, y):
        self.calls.append(("element_at", round(x), round(y)))
        return self.hit

    def parent(self, el):
        return el.parent

    def is_running(self, pid):
        return pid in self.running_pids

    def screen_locked(self):
        return self.locked

    # writes
    def perform(self, el, action):
        self.calls.append(("perform", el.name, action))
        rc = el.rc.get(action, 0)
        if rc == 0 and not el.ignore_writes:
            el.attrs["AXValue"] = f"{action} done"
        return rc

    def set_attr(self, el, attr, value):
        self.calls.append(("set", el.name, attr, value))
        if attr not in el.settable:
            return -25205
        if el.rc.get(attr):
            return el.rc[attr]
        if el.ignore_writes:
            return 0
        if attr == "AXSelectedText":
            el.attrs["AXValue"] = (el.attrs.get("AXValue") or "") + value
            el.attrs["AXSelectedText"] = value
        else:
            el.attrs[attr] = value
        return 0

    def set_text_range(self, el, location, length):
        self.calls.append(("range", el.name, location, length))
        el.attrs["AXSelectedTextRange"] = (location, length)
        v = el.attrs.get("AXValue") or ""
        el.attrs["AXSelectedText"] = v[location:location + length]
        return 0

    def post_key(self, stroke, pid):
        self.calls.append(("key", stroke.label, pid))

    def type_unicode(self, text, pid, cancel=None):
        self.calls.append(("type", text, pid))
        return utf16_len(text)

    def scroll(self, dy, unit, point, pid):
        self.calls.append(("scroll", dy, unit, point, pid))

    def click_at(self, x, y):
        self.calls.append(("click_at", round(x), round(y)))

    def open_app(self, path):
        self.calls.append(("open_app", path))
        return True, "opened"

    def open_url(self, url):
        self.calls.append(("open_url", url))
        return True

    def open_folder(self, path):
        self.calls.append(("open_folder", path))
        return True

    def terminate(self, pid):
        self.calls.append(("terminate", pid))
        return True

    def hide(self, pid):
        self.calls.append(("hide", pid))
        return True

    def writes(self) -> list[tuple]:
        return [c for c in self.calls if c[0] != "element_at"]


class FakeObserver:
    def __init__(self, refs: dict[str, Ref] | None = None, front: Front | None = None, focused: Ref | None = None,
                 areas: list | None = None):
        self.refs = refs or {}
        self.front = front
        self.focused = focused
        self.areas = areas or []
        self.invalidated = 0

    def resolve(self, snap, eid):
        return self.refs.get(eid)

    def frontmost(self):
        return self.front

    def focused_element(self):
        return self.focused

    def scroll_areas(self, snap):
        return list(self.areas)

    def invalidate(self):
        self.invalidated += 1


def live(obs: FakeObserver, ui: FakeUI) -> Executor:
    return Executor(obs, HarnessConfig(dry_run=False), ui=ui, settle_s=0.05)


# ---------------------------------------------------------------- dry-run


ALL_VALID = [
    act(ActionKind.OPEN_APP, app="Notes"),
    act(ActionKind.QUIT_APP, app="Mail"),
    act(ActionKind.CLICK, target_eid="e01", target_label='button "Send"'),
    act(ActionKind.TYPE_TEXT, text="hello world"),
    act(ActionKind.SEARCH_WEB, text="alan turing"),
    act(ActionKind.OPEN_URL, url="github.com"),
    act(ActionKind.PRESS_KEY, key="cmd+c"),
    act(ActionKind.SCROLL_DOWN, amount=0),
    act(ActionKind.SCROLL_UP, amount=2),
    act(ActionKind.OPEN_FOLDER, folder="downloads"),
    act(ActionKind.GO_BACK),
    act(ActionKind.NEW_TAB),
    act(ActionKind.CLOSE_TAB),
    act(ActionKind.UNDO),
    act(ActionKind.CONFIRM),
    act(ActionKind.CANCEL),
]


def test_dry_run_is_default():
    assert HarnessConfig().dry_run is True


@pytest.mark.parametrize("action", ALL_VALID, ids=lambda a: a.kind.value)
def test_dry_run_never_touches_ui(action):
    if action.kind == ActionKind.OPEN_APP and sys.platform != "darwin":
        pytest.skip("open_app dry-run consults the installed macOS apps (LaunchServices)")
    snap = snap_of(el("e01", "button", "Send"), el("e02", "text field", "To", focused=True), focused="e02")
    x = Executor(Exploding("observer"), HarnessConfig(), ui=Exploding("ui"))
    r = x.run(action, snap, threading.Event())
    assert r.dry_run and not r.changed
    assert r.ok, r.detail
    assert r.detail.startswith("would ")
    assert r.undo_token is None


def test_dry_run_covers_every_action_kind():
    assert {a.kind for a in ALL_VALID} == set(ActionKind)


def test_dry_run_descriptions():
    snap = snap_of(el("e01", "button", "Send"), el("e02", "text field", "To", focused=True), focused="e02")
    x = Executor(None, HarnessConfig())
    run = lambda a: x.run(a, snap, threading.Event()).detail  # noqa: E731
    assert run(act(ActionKind.CLICK, target_eid="e01")) == 'would click button "Send" (e01)'
    assert run(act(ActionKind.TYPE_TEXT, text="hi bob")) == 'would type \'hi bob\' (6 chars) into text field "To"'
    assert "q=alan+turing" in run(act(ActionKind.SEARCH_WEB, text="alan turing"))
    assert run(act(ActionKind.OPEN_URL, url="github.com")) == "would open https://github.com"
    assert run(act(ActionKind.PRESS_KEY, key="escape")) == "would press escape"
    assert run(act(ActionKind.SCROLL_DOWN, amount=2)) == "would scroll down all the way to the top or bottom"
    assert run(act(ActionKind.OPEN_FOLDER, folder="downloads")) == "would open folder ~/Downloads"
    assert x._ui is None  # no MacUI was created


def test_dry_run_search_uses_config_template():
    x = Executor(None, HarnessConfig(search_url="https://duckduckgo.com/?q={q}"))
    r = x.run(act(ActionKind.SEARCH_WEB, text="m1 macbook"), None, threading.Event())
    assert r.ok and "https://duckduckgo.com/?q=m1+macbook" in r.detail


@pytest.mark.parametrize(
    "action,why",
    [
        (act(ActionKind.OPEN_URL, url="javascript:alert(1)"), "refused url"),
        (act(ActionKind.OPEN_URL, url="file:///etc/passwd"), "refused url"),
        (act(ActionKind.PRESS_KEY, key="cmd+q"), "not in the key map"),
        (act(ActionKind.PRESS_KEY, key="none"), "not in the key map"),
        (act(ActionKind.CLICK, target_eid="e09"), "not in the snapshot"),
        (act(ActionKind.CLICK, target_eid="e03"), "disabled"),
        (act(ActionKind.CLICK), "no target"),
        (act(ActionKind.TYPE_TEXT, text=""), "nothing to type"),
        (act(ActionKind.TYPE_TEXT, text="hunter2", target_eid="e04"), "password"),
        (act(ActionKind.TYPE_TEXT, text="1234", target_eid="e05"), "password"),
        (act(ActionKind.OPEN_APP, app="NoSuchApp-Jev-1234"), "no installed app"),
        (act(ActionKind.OPEN_APP, app="none"), "no app"),
        (act(ActionKind.OPEN_FOLDER, folder="/etc"), "unknown folder"),
        (act(ActionKind.SCROLL_DOWN, amount=7), "bad scroll amount"),
        (act(ActionKind.SEARCH_WEB, text=""), "nothing to search"),
    ],
    ids=lambda v: v if isinstance(v, str) else v.kind.value,
)
def test_dry_run_reports_would_fail(action, why):
    snap = snap_of(
        el("e01", "button", "Send"),
        el("e02", "text field", "To", focused=True),
        el("e03", "button", "Archive", enabled=False),
        el("e04", "password field", "Password", secure=True),
        el("e05", "text field", "Card number"),
        focused="e02",
    )
    r = Executor(Exploding("observer"), HarnessConfig(), ui=Exploding("ui")).run(action, snap, threading.Event())
    assert not r.ok and r.dry_run
    assert r.detail.startswith("would fail:") and why in r.detail


def test_dry_run_refuses_typing_into_focused_secure_field():
    snap = snap_of(el("e01", "password field", "Password", secure=True, focused=True), focused="e01")
    r = Executor(None, HarnessConfig()).run(act(ActionKind.TYPE_TEXT, text="hunter2"), snap, threading.Event())
    assert not r.ok and "password" in r.detail


def test_cancelled_before_start():
    cancel = threading.Event()
    cancel.set()
    r = Executor(Exploding("observer"), HarnessConfig(dry_run=False), ui=Exploding("ui")).run(
        act(ActionKind.PRESS_KEY, key="return"), None, cancel
    )
    assert not r.ok and "cancelled" in r.detail


def test_undo_dry_run():
    rec = ActionRecord(said="type hello", action=act(ActionKind.TYPE_TEXT, text="hello"), outcome="ok", t=0.0,
                       undo_token={"kind": "type_text", "n16": 5, "digest": text_digest("hello")})
    r = Executor(Exploding("observer"), HarnessConfig(), ui=Exploding("ui")).undo(rec)
    assert r.ok and r.dry_run and r.detail.startswith("would undo type_text")
    rec2 = ActionRecord(said="new tab", action=act(ActionKind.NEW_TAB), outcome="ok", t=0.0, undo_token=None)
    assert "cmd+z" in Executor(None, HarnessConfig()).undo(rec2).detail


@pytest.mark.slow  # a fresh interpreter importing AppKit can take >10 s on a cold, locked machine
def test_dry_run_imports_no_event_machinery():
    """A fresh interpreter running dry-runs of every action never loads Quartz or ApplicationServices."""
    code = (
        "import sys, threading\n"
        "from jev_local.harness.execute import Executor\n"
        "from jev_local.harness.types import Action, ActionKind, HarnessConfig\n"
        "x = Executor(None, HarnessConfig())\n"
        "for k in ActionKind:\n"
        "    x.run(Action(kind=k, source_vid='v', confidence=1.0, app='Finder', text='hi', key='return',"
        " url='example.com', folder='home', amount=1), None, threading.Event())\n"
        "print(int('Quartz' in sys.modules), int('ApplicationServices' in sys.modules), int(x._ui is None))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["0", "0", "1"]


# ---------------------------------------------------------------- live logic against a fake UI


def test_click_prefers_axpress():
    send = Ref("send", actions=("AXPress", "AXShowMenu"), AXEnabled=True)
    ui = FakeUI()
    obs = FakeObserver({"e01": send}, front=Front(4242))
    r = live(obs, ui).run(act(ActionKind.CLICK, target_eid="e01"), snap_of(el("e01", "button", "Send")), threading.Event())
    assert r.ok and r.changed and "AXPress" in r.detail and not r.dry_run
    assert ui.writes() == [("perform", "send", "AXPress")]
    assert obs.invalidated == 1


def test_click_press_timeout_counts_as_delivered():
    menu = Ref("file", actions=("AXPress",))
    menu.rc["AXPress"] = -25204
    ui = FakeUI()
    r = live(FakeObserver({"e01": menu}, front=Front(4242)), ui).run(
        act(ActionKind.CLICK, target_eid="e01"), snap_of(el("e01", "menu", "File")), threading.Event()
    )
    assert r.ok and "no reply" in r.detail


def test_click_falls_back_to_selected_then_pointer():
    row = Ref("row", settable=("AXSelected",), AXSelected=False)
    ui = FakeUI()
    r = live(FakeObserver({"e01": row}, front=Front(4242)), ui).run(
        act(ActionKind.CLICK, target_eid="e01"), snap_of(el("e01", "row", "Downloads")), threading.Event()
    )
    assert r.ok and r.changed and "AXSelected" in r.detail
    assert ("set", "row", "AXSelected", True) in ui.calls

    icon = Ref("icon", actions=("AXOpen",), frame=(10.0, 20.0, 64.0, 64.0))
    label = Ref("icon-label")
    label.parent = icon
    ui = FakeUI(hit=label)
    r = live(FakeObserver({"e01": icon}, front=Front(4242)), ui).run(
        act(ActionKind.CLICK, target_eid="e01"), snap_of(el("e01", "icon", "Secrets")), threading.Event()
    )
    assert r.ok and "pointer click" in r.detail
    assert ("click_at", 42, 52) in ui.calls


def test_click_refuses_when_occluded_or_disabled_or_stale():
    icon = Ref("icon", frame=(10.0, 20.0, 64.0, 64.0))
    ui = FakeUI(hit=Ref("someone-elses-window"))
    r = live(FakeObserver({"e01": icon}, front=Front(4242)), ui).run(
        act(ActionKind.CLICK, target_eid="e01"), snap_of(el("e01", "icon", "Secrets")), threading.Event()
    )
    assert not r.ok and "on top" in r.detail
    assert not [c for c in ui.calls if c[0] == "click_at"]

    off = Ref("off", actions=("AXPress",), AXEnabled=False)
    ui = FakeUI()
    r = live(FakeObserver({"e01": off}, front=Front(4242)), ui).run(
        act(ActionKind.CLICK, target_eid="e01"), snap_of(el("e01", "button", "Send")), threading.Event()
    )
    assert not r.ok and "disabled" in r.detail and ui.writes() == []

    ui = FakeUI()
    r = live(FakeObserver({}, front=Front(4242)), ui).run(
        act(ActionKind.CLICK, target_eid="e01"), snap_of(el("e01", "button", "Send")), threading.Event()
    )
    assert not r.ok and "no longer on screen" in r.detail


def test_ui_actions_refused_when_front_app_changed():
    ui = FakeUI()
    obs = FakeObserver({"e01": Ref("send", actions=("AXPress",))}, front=Front(999, name="Terminal"))
    for a in (act(ActionKind.CLICK, target_eid="e01"), act(ActionKind.PRESS_KEY, key="return"),
              act(ActionKind.TYPE_TEXT, text="rm -rf"), act(ActionKind.CLOSE_TAB)):
        r = live(obs, ui).run(a, snap_of(el("e01", "button", "Send")), threading.Event())
        assert not r.ok and "frontmost app changed" in r.detail
    assert ui.writes() == []


def test_live_refused_while_screen_locked():
    ui = FakeUI(locked=True)
    r = live(FakeObserver(front=Front(4242)), ui).run(act(ActionKind.OPEN_URL, url="example.com"), None, threading.Event())
    assert not r.ok and "locked" in r.detail and ui.writes() == []


def test_type_via_selected_text_and_token_has_no_text():
    field_ = Ref("to", settable=("AXSelectedText", "AXFocused"), AXRole="AXTextField", AXValue="", AXFocused=True)
    ui = FakeUI()
    r = live(FakeObserver(front=Front(4242), focused=field_), ui).run(
        act(ActionKind.TYPE_TEXT, text="hello world"), snap_of(el("e01", "text field", "To", focused=True)),
        threading.Event(),
    )
    assert r.ok and r.changed and "AXSelectedText" in r.detail and "unverified" not in r.detail
    assert not [c for c in ui.calls if c[0] == "type"]
    tok = r.undo_token
    assert tok["kind"] == "type_text" and tok["n16"] == 11 and tok["digest"] == text_digest("hello world")
    assert "hello world" not in repr(tok)


def test_type_falls_back_to_keystrokes_when_ax_write_ignored():
    field_ = Ref("web", settable=("AXSelectedText",), AXRole="AXTextArea", AXValue="abc")
    field_.ignore_writes = True
    ui = FakeUI()
    r = live(FakeObserver(front=Front(4242), focused=field_), ui).run(
        act(ActionKind.TYPE_TEXT, text="hi"), snap_of(el("e01", "text area", "Message")), threading.Event()
    )
    assert r.ok and "keystrokes" in r.detail
    assert ("type", "hi", 4242) in ui.calls


def test_type_into_target_focuses_it_first():
    field_ = Ref("search", settable=("AXSelectedText", "AXFocused"), AXRole="AXTextField", AXValue="", AXFocused=False)
    ui = FakeUI()
    r = live(FakeObserver({"e02": field_}, front=Front(4242)), ui).run(
        act(ActionKind.TYPE_TEXT, text="cats", target_eid="e02"),
        snap_of(el("e01", "button", "Go"), el("e02", "search field", "Search")), threading.Event(),
    )
    assert r.ok
    assert ui.writes()[0] == ("set", "search", "AXFocused", True)


@pytest.mark.parametrize(
    "attrs",
    [
        {"AXRole": "AXSecureTextField"},
        {"AXRole": "AXTextField", "AXSubrole": "AXSecureTextField"},
        {"AXRole": "AXTextField", "AXPlaceholderValue": "Enter your password"},
        {"AXRole": "AXTextField", "AXTitle": "CVV"},
    ],
)
def test_type_refuses_live_secure_field(attrs):
    pw = Ref("pw", settable=("AXSelectedText",), AXValue="", **attrs)
    ui = FakeUI()
    r = live(FakeObserver(front=Front(4242), focused=pw), ui).run(
        act(ActionKind.TYPE_TEXT, text="hunter2"), snap_of(el("e01", "text field", "Name")), threading.Event()
    )
    assert not r.ok and "secure" in r.detail
    assert ui.writes() == []


def test_type_needs_a_focused_element():
    r = live(FakeObserver(front=Front(4242), focused=None), FakeUI()).run(
        act(ActionKind.TYPE_TEXT, text="x"), snap_of(), threading.Event()
    )
    assert not r.ok and "no focused" in r.detail


def test_press_key_and_shortcuts_go_to_snapshot_pid():
    ui = FakeUI()
    x = live(FakeObserver(front=Front(4242)), ui)
    s = snap_of(pid=4242)
    assert x.run(act(ActionKind.PRESS_KEY, key="cmd+shift+z"), s, threading.Event()).ok
    assert x.run(act(ActionKind.GO_BACK), s, threading.Event()).ok
    r = x.run(act(ActionKind.NEW_TAB), s, threading.Event())
    assert r.undo_token == {"kind": "new_tab", "pid": 4242, "bundle_id": "com.apple.mail"}
    assert x.run(act(ActionKind.CLOSE_TAB), s, threading.Event()).ok
    assert x.run(act(ActionKind.UNDO), s, threading.Event()).ok
    assert ui.writes() == [
        ("key", "cmd+shift+z", 4242), ("key", "cmd+[", 4242), ("key", "cmd+t", 4242),
        ("key", "cmd+w", 4242), ("key", "cmd+z", 4242),
    ]


def test_scroll_to_end_uses_scroll_bar():
    bar = Ref("bar", settable=("AXValue",), AXValue=0.3)
    area = Ref("area", AXVerticalScrollBar=bar)
    obs = FakeObserver(front=Front(4242), areas=[(area, (0.0, 100.0, 800.0, 600.0), (0,))])
    ui = FakeUI()
    r = live(obs, ui).run(act(ActionKind.SCROLL_DOWN, amount=2), snap_of(), threading.Event())
    assert r.ok and r.changed and "scroll bar" in r.detail
    assert ui.writes() == [("set", "bar", "AXValue", 1.0)]
    assert r.undo_token == {"kind": "scroll", "dir": "down", "amount": 2, "pid": 4242}


def test_scroll_wheel_located_over_innermost_area():
    big = Ref("big")
    small = Ref("small")
    areas = [(big, (0.0, 0.0, 1000.0, 800.0), (0,)), (small, (500.0, 100.0, 400.0, 300.0), (0, 1))]
    ui = FakeUI()
    x = live(FakeObserver(front=Front(4242), areas=areas), ui)
    s = snap_of(el("e01", "row", "Item", frame=(600.0, 150.0, 100.0, 20.0)))
    assert x.run(act(ActionKind.SCROLL_DOWN, amount=1, target_eid="e01"), s, threading.Event()).ok
    assert x.run(act(ActionKind.SCROLL_UP, amount=0), s, threading.Event()).ok
    assert ui.writes() == [
        ("scroll", -255, "pixel", (700.0, 250.0), 4242),  # 85% of the small area's height, downwards
        ("scroll", 3, "line", (500.0, 400.0), 4242),  # no anchor: the biggest area, 3 lines up
    ]


def test_open_url_search_folder():
    ui = FakeUI()
    x = live(FakeObserver(front=Front(4242)), ui)
    assert x.run(act(ActionKind.OPEN_URL, url="GitHub.com/anthropics"), None, threading.Event()).ok
    assert x.run(act(ActionKind.SEARCH_WEB, text="rust & go"), None, threading.Event()).ok
    r = x.run(act(ActionKind.OPEN_URL, url="javascript:alert(1)"), None, threading.Event())
    assert not r.ok
    calls = ui.writes()
    assert calls[:2] == [("open_url", "https://GitHub.com/anthropics"),
                         ("open_url", "https://www.google.com/search?q=rust+%26+go")]
    assert len(calls) == 2
    if os.path.isdir(os.path.expanduser("~/Downloads")):
        assert x.run(act(ActionKind.OPEN_FOLDER, folder="downloads"), None, threading.Event()).ok
        assert ui.writes()[-1] == ("open_folder", os.path.expanduser("~/Downloads"))


def test_open_app_waits_for_frontmost(monkeypatch):
    monkeypatch.setattr(ex.apps, "app_path", lambda n: "/System/Applications/Notes.app" if n.lower() == "notes" else None)
    monkeypatch.setattr(ex.apps, "bundle_id", lambda p: "com.apple.Notes")
    monkeypatch.setattr(ex.apps, "running", lambda regular_only=True: [])
    obs = FakeObserver(front=Front(4242))
    ui = FakeUI()
    orig_open = ui.open_app

    def open_and_activate(path):
        obs.front = Front(777, "Notes", "com.apple.Notes", path)
        return orig_open(path)

    ui.open_app = open_and_activate  # type: ignore[method-assign]
    r = live(obs, ui).run(act(ActionKind.OPEN_APP, app="Notes"), snap_of(), threading.Event())
    assert r.ok and r.changed, r.detail
    assert r.undo_token["launched"] is True and r.undo_token["pid"] == 777
    # already frontmost: nothing to do
    r2 = live(obs, ui).run(act(ActionKind.OPEN_APP, app="Notes"), snap_of(), threading.Event())
    assert r2.ok and not r2.changed and "already frontmost" in r2.detail
    assert [c for c in ui.calls if c[0] == "open_app"] == [("open_app", "/System/Applications/Notes.app")]


def test_quit_app_uses_terminate(monkeypatch):
    target = ex.apps.RunningApp("TextEdit", "com.apple.TextEdit", 555, "/System/Applications/TextEdit.app", False)
    monkeypatch.setattr(ex.apps, "find_running", lambda n: target if n == "TextEdit" else None)
    ui = FakeUI(running_pids={555})
    orig = ui.terminate

    def term(pid):
        ui.running_pids.discard(pid)
        return orig(pid)

    ui.terminate = term  # type: ignore[method-assign]
    r = live(FakeObserver(front=Front(4242)), ui).run(act(ActionKind.QUIT_APP, app="TextEdit"), None, threading.Event())
    assert r.ok and r.changed and ui.writes() == [("terminate", 555)]
    r = live(FakeObserver(front=Front(4242)), FakeUI()).run(act(ActionKind.QUIT_APP, app="Nope"), None, threading.Event())
    assert not r.ok and "not running" in r.detail


def test_handler_exception_becomes_failed_result():
    class Boom(FakeUI):
        def post_key(self, stroke, pid):
            raise RuntimeError("kaboom")

    r = live(FakeObserver(front=Front(4242)), Boom()).run(act(ActionKind.PRESS_KEY, key="return"), snap_of(), threading.Event())
    assert not r.ok and "kaboom" in r.detail


# ---------------------------------------------------------------- undo (live logic, fake UI)


def _typed_field(value: str, caret: int) -> Ref:
    f = Ref("field", settable=("AXSelectedText", "AXSelectedTextRange"), AXValue=value)
    f.attrs["AXSelectedTextRange"] = (caret, 0)
    return f


def test_undo_typing_deletes_exactly_the_typed_text():
    f = _typed_field("Dear bob, hello world", 21)
    tok = {"kind": "type_text", "method": "AXSelectedText", "n16": 11, "digest": text_digest("hello world"), "pid": 4242}
    rec = ActionRecord("type hello world", act(ActionKind.TYPE_TEXT, text="hello world"), "ok", 0.0, tok)
    ui = FakeUI()
    r = live(FakeObserver(front=Front(4242), focused=f), ui).undo(rec)
    assert r.ok and "deleted the 11 typed characters" in r.detail
    assert ("range", "field", 10, 11) in ui.calls and ("set", "field", "AXSelectedText", "") in ui.calls
    assert not [c for c in ui.calls if c[0] == "key"]


def test_undo_typing_falls_back_to_cmd_z_when_text_differs():
    f = _typed_field("Dear bob, jello world", 21)  # the field no longer ends with what we typed
    tok = {"kind": "type_text", "n16": 11, "digest": text_digest("hello world"), "pid": 4242}
    rec = ActionRecord("type hello world", act(ActionKind.TYPE_TEXT, text="hello world"), "ok", 0.0, tok)
    ui = FakeUI()
    r = live(FakeObserver(front=Front(4242), focused=f), ui).undo(rec)
    assert r.ok and "cmd+z" in r.detail
    assert ("range", "field", 21, 0) in ui.calls  # caret restored before falling back
    assert ("set", "field", "AXSelectedText", "") not in ui.calls


def test_undo_new_tab_scroll_close_tab_generic():
    ui = FakeUI()
    x = live(FakeObserver(front=Front(4242)), ui)
    mk = lambda kind, tok: ActionRecord("x", act(kind), "ok", 0.0, tok)  # noqa: E731
    assert x.undo(mk(ActionKind.NEW_TAB, {"kind": "new_tab", "pid": 4242})).ok
    assert x.undo(mk(ActionKind.SCROLL_DOWN, {"kind": "scroll", "dir": "down", "amount": 0, "pid": 4242})).ok
    assert x.undo(mk(ActionKind.CLOSE_TAB, {"kind": "close_tab", "pid": 4242, "bundle_id": "com.google.Chrome"})).ok
    assert not x.undo(mk(ActionKind.CLOSE_TAB, {"kind": "close_tab", "pid": 4242, "bundle_id": "com.apple.mail"})).ok
    assert x.undo(mk(ActionKind.CLICK, None)).ok
    assert ui.writes() == [
        ("key", "cmd+w", 4242),
        ("scroll", 3, "line", None, 4242),  # opposite direction
        ("key", "cmd+shift+t", 4242),
        ("key", "cmd+z", 4242),
    ]
    r = live(FakeObserver(front=Front(1)), FakeUI()).undo(mk(ActionKind.NEW_TAB, {"kind": "new_tab", "pid": 4242}))
    assert not r.ok and "no longer frontmost" in r.detail


# ---------------------------------------------------------------- deny list (defense in depth)


def test_denied_app_matching():
    assert ex.denied_app("Terminal", "com.apple.Terminal") == "terminal"
    assert ex.denied_app("System Settings") == "system settings"
    assert ex.denied_app("1Password 7") == "1password"
    assert ex.denied_app("Notes", "com.apple.Notes") is None
    assert ex.denied_app("Terminal", allow=("terminal",)) is None
    assert ex.denied_app("", None) is None


def test_executor_refuses_deny_listed_apps_even_in_dry_run():
    term = Snapshot("Terminal", "com.apple.Terminal", 701, "zsh", (el("e01", "text area", "shell", focused=True),),
                    1.0, focused_eid="e01")
    x = Executor(Exploding("observer"), HarnessConfig(), ui=Exploding("ui"))
    for a in (act(ActionKind.TYPE_TEXT, text="rm -rf ~"), act(ActionKind.PRESS_KEY, key="return"),
              act(ActionKind.CLICK, target_eid="e01")):
        r = x.run(a, term, threading.Event())
        assert not r.ok and "deny list" in r.detail
    for app in ("System Settings", "Keychain Access", "Passwords"):
        r = x.run(act(ActionKind.OPEN_APP, app=app), None, threading.Event())
        assert not r.ok and "deny list" in r.detail
    # allow-listed by the user: permitted
    ok = Executor(None, HarnessConfig(allow_apps=("terminal",))).run(
        act(ActionKind.PRESS_KEY, key="return"), term, threading.Event()
    )
    assert ok.ok and ok.detail == "would press return"


def test_live_refuses_deny_listed_app_without_touching_ui():
    ui = FakeUI()
    term = Snapshot("iTerm2", "com.googlecode.iterm2", 700, None, (), 1.0)
    r = live(FakeObserver(front=Front(700, "iTerm2")), ui).run(act(ActionKind.PRESS_KEY, key="return"), term, threading.Event())
    assert not r.ok and "deny list" in r.detail and ui.writes() == []


def test_run_never_raises_even_if_validation_breaks(monkeypatch):
    monkeypatch.setattr(ex.apps, "app_path", lambda n: (_ for _ in ()).throw(OSError("disk gone")))
    for dry in (True, False):
        r = Executor(None, HarnessConfig(dry_run=dry), ui=FakeUI()).run(
            act(ActionKind.OPEN_APP, app="Notes"), None, threading.Event()
        )
        assert not r.ok and "disk gone" in r.detail and r.dry_run == dry
