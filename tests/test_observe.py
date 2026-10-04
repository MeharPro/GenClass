"""Observer tests. Pure helpers run anywhere; `macos` tests take real, read-only AX snapshots
(never setting AXManualAccessibility) of Finder and of whatever app is frontmost."""

from __future__ import annotations

import dataclasses
import re
import statistics
import sys
import time

import pytest

if sys.platform != "darwin":  # observe.py imports pyobjc ApplicationServices at module level (macOS only)
    pytest.skip("macOS Accessibility (pyobjc) only", allow_module_level=True)

from jev_local.harness import observe  # noqa: E402
from jev_local.harness.observe import (
    FOCUS_ROOT,
    MENU_ROOT,
    Observer,
    PermissionMissing,
    _Cand,
    _Walk,
    ax_error_name,
    clean_text,
    dedupe_labels,
    format_value,
    humanize_role,
    is_chromium_like,
)
from jev_local.harness.types import Element, Snapshot

# ---------------------------------------------------------------- pure helpers


@pytest.mark.parametrize(
    "role,sub,want",
    [
        ("AXButton", "", "button"),
        ("AXButton", "AXCloseButton", "close button"),
        ("AXTextField", "AXSearchField", "search field"),
        ("AXTextField", "", "text field"),
        ("AXSecureTextField", "", "password field"),
        ("AXRadioButton", "AXTabButton", "tab"),
        ("AXCheckBox", "AXSwitch", "switch"),
        ("AXPopUpButton", "", "pop-up button"),
        ("AXRow", "AXOutlineRow", "row"),
        ("AXMenuBarItem", "", "menu"),
        ("AXLevelIndicator", "", "level indicator"),
        ("", "", "element"),
    ],
)
def test_humanize_role(role, sub, want):
    assert humanize_role(role, sub) == want


def test_clean_text():
    assert clean_text("  Hello\n\tworld  ") == "Hello world"
    long = "x" * 70
    out = clean_text(long)
    assert len(out) == 60 and out.endswith("…")
    assert clean_text(None) == "" and clean_text(True) == ""
    assert clean_text("a\u0000b") == "ab"
    assert clean_text(42) == "42"


@pytest.mark.parametrize(
    "role,sub,v,want",
    [
        ("AXCheckBox", "", 1, "on"),
        ("AXCheckBox", "", 0, "off"),
        ("AXCheckBox", "", 2, "mixed"),
        ("AXCheckBox", "AXSwitch", True, "on"),
        ("AXRadioButton", "AXTabButton", True, "selected"),
        ("AXRadioButton", "", 0, None),
        ("AXSlider", "", 0.5, "0.5"),
        ("AXTextField", "", "  hi\nthere ", "hi there"),
        ("AXTextField", "", "", None),
        ("AXButton", "", False, None),
        ("AXTextField", "", None, None),
    ],
)
def test_format_value(role, sub, v, want):
    assert format_value(role, sub, v) == want


def test_dedupe_labels():
    pairs = [("button", "Reply"), ("button", "reply"), ("link", "Reply"), ("button", "Reply"), ("button", ""), ("button", "")]
    assert dedupe_labels(pairs) == ["Reply (1 of 3)", "reply (2 of 3)", "Reply", "Reply (3 of 3)", "", ""]


def test_ax_error_names():
    assert "stale" in ax_error_name(-25202)
    assert "did not answer" in ax_error_name(-25204)
    assert "permission" in ax_error_name(-25211)
    assert "no value" in ax_error_name(-25212)
    assert "unknown" in ax_error_name(-1)
    assert all(-25214 <= c <= 0 for c in observe.AX_ERRORS)


def test_is_chromium_like(tmp_path):
    assert is_chromium_like("com.google.Chrome", None)
    assert is_chromium_like("com.brave.Browser", "/nonexistent")
    electron = tmp_path / "Slack.app"
    (electron / "Contents/Frameworks/Electron Framework.framework").mkdir(parents=True)
    assert is_chromium_like("com.tinyspeck.slackmacgap", str(electron))
    native = tmp_path / "Notes.app"
    (native / "Contents/MacOS").mkdir(parents=True)
    assert not is_chromium_like("com.apple.Notes", str(native))


def test_rect_helpers():
    assert observe._intersects((0, 0, 10, 10), (5, 5, 10, 10))
    assert not observe._intersects((0, 0, 10, 10), (10, 0, 5, 5))  # touching edges don't overlap
    assert observe._intersect(None, (1, 2, 3, 4)) == (1, 2, 3, 4)
    assert observe._intersect((0, 0, 10, 10), (5, 5, 10, 10)) == (5, 5, 5, 5)
    assert observe._intersect((0, 0, 1, 1), (5, 5, 1, 1))[2:] == (0.0, 0.0)
    assert observe._expand((10, 10, 5, 5), 2) == (8, 8, 9, 9)


def test_context_rules():
    ctx = Observer._context
    assert ctx("AXToolbar", "", "", None) == "toolbar"
    assert ctx("AXOutline", "AXSourceList", "", None) == "sidebar"
    assert ctx("AXOutline", "", "Sidebar", None) == "sidebar"
    assert ctx("AXTable", "", "", None) == "list"
    assert ctx("AXScrollArea", "", "desktop", None) == "desktop"
    assert ctx("AXTabGroup", "", "", "toolbar") == "tab bar"
    assert ctx("AXGroup", "", "", "toolbar") == "toolbar"  # inherited


def _cand(order: int, role_ax: str = "AXButton", label: str = "x", **kw) -> _Cand:
    base = dict(
        ref=object(), order=order, role_ax=role_ax, subrole="", role=humanize_role(role_ax), label=label,
        value=None, frame=(0.0, 0.0, 10.0, 10.0), path=(order,), secure=False, focused=False, enabled=True,
        context=None, modal=False,
    )
    base.update(kw)
    return _Cand(**base)


def test_select_ranks_trims_and_keeps_walk_order():
    obs = Observer(max_elements=4, watch=False, manual_ax=False)
    w = _Walk(deadline=0, max_nodes=0, max_children=0)
    w.cands = [
        _cand(0, "AXMenuBarItem", "File", menu_bar=True, path=(MENU_ROOT, 0)),
        _cand(1, "AXButton", ""),  # unlabeled: dropped
        _cand(2, "AXButton", "Send"),
        _cand(3, "AXTextField", ""),  # unlabeled text field: kept (you can still type into it)
        _cand(4, "AXButton", "Hidden", visible=False),  # zero-size: dropped
        _cand(5, "AXRow", "Inbox"),
        _cand(6, "AXButton", "Disabled", enabled=False),
        _cand(7, "AXLink", "Docs"),
        _cand(8, "AXGroup", "Deep focus", focused=True, visible=False),  # focused always survives
        _cand(9, "AXButton", "", subrole="AXCloseButton", role="close button"),
    ]
    kept = obs._select(w)
    orders = [c.order for c in kept]
    assert orders == sorted(orders)
    assert 8 in orders  # focused
    assert 0 not in orders  # menu bar loses to window content
    assert 1 not in orders and 4 not in orders
    assert len(kept) == 4


def test_permission_missing(monkeypatch):
    monkeypatch.setattr(observe.AX, "AXIsProcessTrusted", lambda: False)
    with pytest.raises(PermissionMissing) as e:
        Observer(watch=False, manual_ax=False).snapshot()
    assert "Accessibility" in str(e.value)


def test_locked_screen_gives_empty_snapshot(monkeypatch):
    monkeypatch.setattr(observe.AX, "AXIsProcessTrusted", lambda: True)
    monkeypatch.setattr(observe, "screen_locked", lambda: True)
    obs = Observer(watch=False, manual_ax=False)
    monkeypatch.setattr(obs, "frontmost", lambda: None)
    s = obs.snapshot(allow_cached=False)
    assert s.elements == () and s.partial and s.app_name == "loginwindow"
    # right after unlocking, loginwindow may still be the active app: treat it the same
    monkeypatch.setattr(observe, "screen_locked", lambda: False)
    monkeypatch.setattr(obs, "frontmost", lambda: observe.FrontApp(431, "loginwindow", observe.LOGINWINDOW, None))
    monkeypatch.setattr(obs, "_walk", lambda *a, **k: pytest.fail("walked the login window"))
    s = obs.snapshot(allow_cached=False)
    assert s.elements == () and s.app_name == "loginwindow"


def test_watcher_callback_marks_cache_dirty():
    obs = Observer(watch=False, manual_ax=False)
    snap = Snapshot("X", "x", 1, None, (), time.monotonic())
    obs._cache = observe._Cached(snap, None, obs._events, obs._gen)
    front = observe.FrontApp(1, "X", "x", None)

    class W:
        active_pid = 1

    obs._watcher = W()  # type: ignore[assignment]
    assert obs._cached_for(front, 5.0) is snap
    obs._on_notification("AXFocusedWindowChanged")
    assert obs._cached_for(front, 5.0) is None
    obs._cache = observe._Cached(snap, None, obs._events, obs._gen)
    obs.invalidate()
    assert obs._cached_for(front, 5.0) is None


# ---------------------------------------------------------------- live, read-only (macOS)

LABEL_RE = re.compile(r"^.{0,60}( \(\d+ of \d+\))?$", re.S)


def _require_live(frontmost: bool = False) -> None:
    if not observe.AX.AXIsProcessTrusted():
        pytest.skip("host process is not AX-trusted")
    if observe.screen_locked():
        pytest.skip("screen is locked: apps don't answer AX queries usefully")
    if frontmost:
        front = Observer(watch=False, manual_ax=False).frontmost()
        if front is not None and front.bundle_id == observe.LOGINWINDOW:
            pytest.skip("the login window is frontmost (just unlocked); nothing of the user's to read")


def _finder_pid() -> int:
    from jev_local.harness import apps

    r = next((r for r in apps.running(regular_only=False) if r.bundle_id == "com.apple.finder"), None)
    if r is None:
        pytest.skip("Finder is not running")
    return r.pid


def _check_snapshot(s: Snapshot, max_elements: int = 60) -> None:
    assert isinstance(s, Snapshot)
    assert s.pid > 0 and s.walk_ms > 0
    assert len(s.elements) <= max_elements
    eids = [e.eid for e in s.elements]
    assert eids == [f"e{i:02d}" for i in range(1, len(eids) + 1)]
    for e in s.elements:
        assert isinstance(e, Element)
        assert e.role and isinstance(e.role, str)
        assert LABEL_RE.match(e.label), e.label
        assert e.value is None or len(e.value) <= 60
        assert len(e.frame) == 4 and all(isinstance(v, float) for v in e.frame)
        assert all(isinstance(i, int) for i in e.path)
        assert all(isinstance(a, str) for a in e.actions)
        if e.secure:
            assert e.value is None
    assert s.focused_eid is None or s.focused_eid in eids


def _measure(obs: Observer, pid: int | None, runs: int = 10) -> tuple[Snapshot, list[float]]:
    times, snap = [], None
    for _ in range(runs):
        snap = obs.snapshot(allow_cached=False, pid=pid)
        times.append(snap.walk_ms)
    assert snap is not None
    return snap, times


@pytest.mark.macos
def test_live_finder_snapshot_readonly():
    _require_live()
    pid = _finder_pid()
    obs = Observer(manual_ax=False, watch=False)
    snap, times = _measure(obs, pid)
    _check_snapshot(snap)
    assert snap.bundle_id == "com.apple.finder"
    print(f"\nFinder: {len(snap.elements)} elements, walk_ms p50 {statistics.median(times):.1f} "
          f"(min {min(times):.1f}, max {max(times):.1f}, first {times[0]:.1f}); stats {obs.last_stats}")
    # Finder always has a menu bar or a desktop, so it is never empty.
    assert len(snap.elements) > 0
    assert statistics.median(times) < obs.deadline_s * 1000 + 100


@pytest.mark.macos
def test_live_frontmost_snapshot_readonly():
    _require_live(frontmost=True)
    obs = Observer(manual_ax=False, watch=False)
    front = obs.frontmost()
    assert front is not None and front.pid > 0
    snap, times = _measure(obs, None)
    if snap.pid != front.pid:
        pytest.skip("the frontmost app changed during the measurement")
    _check_snapshot(snap)
    print(f"\nfrontmost {snap.app_name!r}: {len(snap.elements)} elements, walk_ms p50 {statistics.median(times):.1f} "
          f"(min {min(times):.1f}, max {max(times):.1f}, first {times[0]:.1f}); stats {obs.last_stats}")
    obs.focused_element()  # must not raise, whatever it returns


@pytest.mark.macos
def test_live_resolve_by_registry_and_by_path():
    _require_live()
    obs = Observer(manual_ax=False, watch=False)
    snap = obs.snapshot(allow_cached=False, pid=_finder_pid())
    if not snap.elements:
        pytest.skip("Finder showed no elements")
    # A copy with another taken_at misses the registry, so resolve() must re-find elements by path.
    other = dataclasses.replace(snap, taken_at=snap.taken_at - 1000.0)
    checked = 0
    for e in snap.elements[:8] + snap.elements[-3:]:
        ref = obs.resolve(snap, e.eid)
        assert ref is not None, e
        if e.path[:1] == (FOCUS_ROOT,):
            continue
        by_path = obs.resolve(other, e.eid)
        if by_path is None:  # the UI changed between the walk and now
            continue
        assert by_path == ref, (e, by_path, ref)
        checked += 1
    assert checked > 0
    assert obs.resolve(snap, "e999") is None


@pytest.mark.macos
def test_live_cache_invalidate_prefetch():
    _require_live()
    pid = _finder_pid()
    obs = Observer(manual_ax=False, watch=False)
    s1 = obs.snapshot(pid=pid)
    t = time.perf_counter()
    s2 = obs.snapshot(pid=pid, allow_cached=True, max_age_s=30)
    hit_ms = (time.perf_counter() - t) * 1000
    assert s2 is s1
    print(f"\ncache hit: {hit_ms:.2f} ms")
    obs.invalidate()
    s3 = obs.snapshot(pid=pid, allow_cached=True, max_age_s=30)
    assert s3 is not s1
    # prefetch warms the cache off-thread; the next call reuses it. Point "frontmost" at Finder so
    # the test doesn't depend on which app the user has in front.
    finder = obs.app_info(pid)
    obs.frontmost = lambda: finder  # type: ignore[method-assign]
    obs.invalidate()
    obs.prefetch()
    deadline = time.monotonic() + 5
    while obs._cache is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert obs._cache is not None
    t = time.perf_counter()
    assert obs.snapshot() is obs._cache.snap
    print(f"after prefetch: {(time.perf_counter() - t) * 1000:.2f} ms")


@pytest.mark.macos
def test_live_watcher_registers_readonly():
    _require_live()
    pid = _finder_pid()
    obs = Observer(manual_ax=False, watch=True)
    try:
        obs.snapshot(pid=pid)
        deadline = time.monotonic() + 3
        while obs._watcher.active_pid != pid and pid not in obs._watcher.failed and time.monotonic() < deadline:
            time.sleep(0.05)
        assert obs._watcher.active_pid == pid or pid in obs._watcher.failed
    finally:
        obs.close()
