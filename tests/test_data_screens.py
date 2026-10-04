"""Procedural screens: realistic, bounded, and exactly what the runtime observer would emit."""

from __future__ import annotations

import random
import re
from collections import Counter

import pytest

from jev_local.data.screens import (
    APP_BY_NAME, APPS, HELDOUT_APPS, HELDOUT_SCREEN_TYPES, INJECTIONS, SCREEN_TYPES, apps_for_screen, make_screen,
    screens_for_app, stable_hash,
)
from jev_local.harness.questions import element_line

EID_RE = re.compile(r"^e\d{2}$")
DUP_RE = re.compile(r" \((\d+) of (\d+)\)$")


def _screens(n_per_type: int = 12, seed: int = 0):
    rng = random.Random(seed)
    for st in SCREEN_TYPES:
        apps = apps_for_screen(st, set(APP_BY_NAME))
        for _ in range(n_per_type):
            yield st, make_screen(rng, st, rng.choice(apps))


def test_app_catalog_is_realistic_and_consistent():
    names = [a.name for a in APPS]
    assert len(names) == len(set(names)), "duplicate app names"
    assert 130 <= len(names) <= 200, len(names)  # "about 150 macOS app names"
    for a in APPS:
        assert a.aliases, a.name
        assert all(al == al.strip() and al for al in a.aliases), a.name
    assert HELDOUT_APPS <= set(names)
    assert len(HELDOUT_APPS) / len(names) >= 0.15
    assert not any(APP_BY_NAME[n].core for n in HELDOUT_APPS)


def test_screen_type_coverage_and_heldout_fraction():
    assert len(SCREEN_TYPES) >= 25
    assert HELDOUT_SCREEN_TYPES <= set(SCREEN_TYPES)
    assert len(HELDOUT_SCREEN_TYPES) / len(SCREEN_TYPES) >= 0.15
    # every screen type can be shown by at least one non-held-out app (train needs it) unless held out
    for st in SCREEN_TYPES:
        apps = apps_for_screen(st, set(APP_BY_NAME))
        assert apps, st
        if st not in HELDOUT_SCREEN_TYPES and st != "empty_desktop":
            assert set(apps) - HELDOUT_APPS, st
    for a in APPS:
        for st in screens_for_app(a.name):
            assert st in SCREEN_TYPES, (a.name, st)


def test_screens_are_well_formed():
    for st, scr in _screens():
        els = scr.snap.elements
        assert 5 <= len(els) <= 60, (st, len(els))
        assert [e.eid for e in els] == [f"e{i + 1:02d}" for i in range(len(els))]
        assert [m.eid for m in scr.els] == [e.eid for e in els]
        focused = [e for e in els if e.focused]
        assert len(focused) <= 1, st
        assert scr.snap.focused_eid == (focused[0].eid if focused else None)
        for e, m in zip(els, scr.els):
            assert EID_RE.match(e.eid)
            assert e.role and e.role == e.role.lower(), e.role
            assert len(e.label) <= 60, e.label
            assert e.value is None or len(e.value) <= 60
            assert m.label == e.label and m.role == e.role
            # the model sees exactly questions.element_line of the runtime Element
            assert element_line(e).startswith(e.role)


def test_duplicates_are_numbered_like_the_observer():
    seen_dup = 0
    for st, scr in _screens(8, seed=1):
        by_base: dict[str, list] = {}
        for m in scr.els:
            if m.dup_n:
                by_base.setdefault(m.base_label, []).append(m)
        for base, group in by_base.items():
            seen_dup += 1
            n = len(group)
            assert all(m.dup_n == n for m in group)
            assert sorted(m.dup_k for m in group) == list(range(1, n + 1))
            for m in group:
                mm = DUP_RE.search(m.label)
                assert mm and int(mm.group(1)) == m.dup_k and int(mm.group(2)) == n, m.label
        # labels that are not duplicated never carry a "(k of n)" suffix added by us
        counts = Counter(m.base_label for m in scr.els if m.base_label)
        for m in scr.els:
            if m.base_label and counts[m.base_label] == 1:
                assert m.dup_n == 0
    assert seen_dup > 20


def test_realism_features_present_across_screens():
    roles, disabled, focused, injected, contexts = Counter(), 0, 0, 0, Counter()
    for st, scr in _screens(10, seed=2):
        for e, m in zip(scr.snap.elements, scr.els):
            roles[e.role] += 1
            disabled += not e.enabled
            focused += e.focused
            injected += m.injected
            contexts[e.context] += 1
            if m.injected:
                assert m.base_label in INJECTIONS
    for r in ("button", "link", "row", "text field", "checkbox", "tab", "search field", "pop-up button"):
        assert roles[r] > 0, r
    assert len(roles) >= 12
    assert disabled > 10 and focused > 50 and injected > 5
    assert len(contexts) >= 10


def test_dialog_screens_have_standard_buttons():
    rng = random.Random(3)
    labels = set()
    for _ in range(40):
        for st in ("dialog_save", "dialog_alert"):
            scr = make_screen(rng, st, "TextEdit")
            labels |= {m.base_label for m in scr.els}
    for want in ("OK", "Cancel", "Delete", "Don't Save", "Save"):
        assert want in labels, want


def test_make_screen_is_deterministic():
    a = make_screen(random.Random(7), "mail_inbox", "Mail")
    b = make_screen(random.Random(7), "mail_inbox", "Mail")
    assert a.snap == b.snap and a.els == b.els


@pytest.mark.parametrize("s", ["a", "hello", "dev:click/mail_inbox/t03"])
def test_stable_hash_is_process_independent(s):
    # Python's hash() is salted per process; splits must not depend on it
    import hashlib

    assert stable_hash(s) == int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)
