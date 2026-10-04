"""Exact label semantics of the command grammar (docs/CONTRACT.md "D").

Hand-built utterances pin the rules down word by word; a randomized sweep over every intent checks
the same invariants on generated commands.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from jev_local.data import grammar as G
from jev_local.data.asr_noise import Tok
from jev_local.data.screens import APP_BY_NAME, APPS, HELDOUT_APPS, SCREEN_TYPES, apps_for_screen, make_screen
from jev_local.harness.catalog import INTENTS, PAYLOAD_INTENTS, RISK_RE
from jev_local.harness.questions import build_questions, rank_apps
from jev_local.harness.spans import extract_text_candidates, extract_url_candidates

TRAIN_APPS = [a.name for a in APPS if a.name not in HELDOUT_APPS]


def _ctx(rng: random.Random, screen_type: str = "mail_compose", app: str = "Mail") -> G.Ctx:
    scr = make_screen(rng, screen_type, app, inject_p=0.0)
    installed = sorted(set(rng.sample(TRAIN_APPS, 60)) | {app, "Safari", "Notes", "Spotify", "Google Chrome"})
    return G.Ctx(rng=rng, screen=scr, apps=TRAIN_APPS, installed=installed, running=[app, "Safari"])


def _labels_all_prefixes(ctx: G.Ctx, toks: list[Tok], gold: G.Gold, kind: str = "command", destructive: float | None = None):
    if destructive is None:
        destructive = G.destructive_gold(toks, 1, gold) if kind != "side" else 0.0
    out = []
    for k in range(1, len(toks) + 1):
        text = " ".join(t.text for t in toks[:k])
        qs = build_questions(ctx.screen.snap, rank_apps(text, ctx.installed, ctx.running), extract_text_candidates(text),
                             extract_url_candidates(text))
        lab, drop = G.compute_labels(toks, k, kind, gold, 1, qs, destructive)
        out.append((text, lab, drop, qs))
    return out


def _lab(lab: dict, qid: str):
    v = lab.get(qid)
    if v is None:
        return None
    return v.get("label", v.get("p", v.get("level", v.get("dist"))))


def _eid(ctx: G.Ctx, base: str) -> str:
    return next(e.eid for e in ctx.screen.els if e.base_label == base)


# ----------------------------------------------------------------------------- hand-built cases


def test_open_app_waits_until_the_app_is_named():
    ctx = _ctx(random.Random(0))
    toks = G.lit_toks("please") + G.lit_toks("open") + G.arg_toks(["Safari"], "app")
    toks[0] = replace(toks[0], slot="lead", part=0)
    G._set_commit(toks, "app")
    rows = _labels_all_prefixes(ctx, toks, G.Gold("open_app", app="Safari"))
    (t1, l1, _, _), (t2, l2, _, _), (t3, l3, _, _) = rows
    assert t2 == "please open"
    for lab in (l1, l2):
        assert _lab(lab, "intent") == "wait" and _lab(lab, "complete") == 0.0 and _lab(lab, "app") == "none"
        assert _lab(lab, "is_command") == 1.0
    assert _lab(l3, "intent") == "open_app" and _lab(l3, "complete") == 1.0 and _lab(l3, "app") == "Safari"
    assert _lab(l3, "destructive") == 0.0
    for lab in (l1, l2, l3):
        assert _lab(lab, "key") == "none" and _lab(lab, "folder") == "none" and _lab(lab, "target") == "none"
        assert "scroll_amount" not in lab


def test_click_commits_at_the_target_and_destructive_follows_the_words():
    ctx = _ctx(random.Random(1))
    send = _eid(ctx, "Send")
    toks = G.lit_toks("click") + G.arg_toks(["the", "send", "button"], "target", role_words={"button"})
    G._set_commit(toks, "target")
    gold = G.Gold("click", target=send, target_base="Send", ref_kind="label")
    rows = _labels_all_prefixes(ctx, toks, gold)
    by_text = {t: lab for t, lab, _, _ in rows}
    assert _lab(by_text["click"], "intent") == "wait"
    assert _lab(by_text["click the"], "intent") == "wait"
    # nothing said so far is destructive: "click the" could still be "click the subject field"
    assert _lab(by_text["click the"], "destructive") == 0.0
    assert _lab(by_text["click the"], "target") == "none"
    lab = by_text["click the send"]
    assert _lab(lab, "intent") == "click" and _lab(lab, "complete") == 1.0 and _lab(lab, "target") == send
    assert _lab(lab, "destructive") == 1.0  # "send" is in catalog.RISK_RE
    lab = by_text["click the send button"]
    assert _lab(lab, "complete") == 1.0 and _lab(lab, "target") == send and _lab(lab, "destructive") == 1.0


def test_click_on_absent_element_is_none():
    ctx = _ctx(random.Random(2))
    toks = G.lit_toks("click") + G.arg_toks(["the", "blue", "one"], "target")
    G._set_commit(toks, "target")
    rows = _labels_all_prefixes(ctx, toks, G.Gold("click", target="none", ref_kind="none"))
    t, lab, _, qs = rows[-1]
    assert "target" in qs and _lab(lab, "target") == "none"
    assert _lab(lab, "intent") == "click"


def test_type_text_span_is_a_candidate_and_verbatim():
    ctx = _ctx(random.Random(3))
    toks = G.lit_toks("type") + G.payload_toks("see you at noon") + G.lit_toks("please")
    toks[-1] = replace(toks[-1], slot="tail", part=0, content=False, droppable=True)
    G._set_commit(toks, "payload")
    rows = _labels_all_prefixes(ctx, toks, G.Gold("type_text"))
    assert _lab(rows[0][1], "intent") == "wait"  # "type"
    assert _lab(rows[0][1], "text_span") in (None, "none")
    for text, lab, drop, qs in rows[1:]:
        assert drop is None, text
        span = _lab(lab, "text_span")
        assert span in qs["text_span"].criteria and span in text
        assert _lab(lab, "intent") == "type_text"
    assert _lab(rows[1][1], "text_span") == "see" and _lab(rows[1][1], "complete") == 0.0
    assert _lab(rows[-1][1], "text_span") == "see you at noon"  # "please" is not typed
    assert _lab(rows[-1][1], "complete") == 1.0 and _lab(rows[-2][1], "complete") == 1.0


def test_payload_risk_words_are_not_destructive():
    ctx = _ctx(random.Random(4))
    toks = G.lit_toks("type") + G.payload_toks("delete this later")
    G._set_commit(toks, "payload")
    assert G.destructive_gold(toks, 1, G.Gold("type_text")) == 0.0
    rows = _labels_all_prefixes(ctx, toks, G.Gold("type_text"))
    assert all(_lab(lab, "destructive") == 0.0 for _, lab, _, _ in rows)


def test_quit_and_close_are_destructive_once_committed():
    ctx = _ctx(random.Random(5))
    toks = G.lit_toks("can") + G.lit_toks("you") + G.lit_toks("quit*") + G.arg_toks(["Spotify"], "app")
    G._set_commit(toks, "app")
    rows = _labels_all_prefixes(ctx, toks, G.Gold("quit_app", app="Spotify"))
    assert [_lab(l, "intent") for _, l, _, _ in rows] == ["wait", "wait", "quit_app", "quit_app"]
    assert [_lab(l, "destructive") for _, l, _, _ in rows] == [0.0, 0.0, 1.0, 1.0]
    assert [_lab(l, "complete") for _, l, _, _ in rows] == [0.0, 0.0, 0.0, 1.0]
    assert _lab(rows[-1][1], "app") == "Spotify"


def test_scroll_amount_only_for_scroll_and_follows_the_amount_words():
    ctx = _ctx(random.Random(6))
    toks = G.lit_toks("scroll") + G.lit_toks("down*")
    amt = [Tok(w, arg="amount", content=True) for w in ("a", "little")]
    amt[0].ident = True
    toks += amt
    rows = _labels_all_prefixes(ctx, toks, G.Gold("scroll_down", amount=0))
    assert _lab(rows[0][1], "intent") == "wait" and "scroll_amount" not in rows[0][1]
    assert _lab(rows[1][1], "intent") == "scroll_down"
    assert rows[1][1]["scroll_amount"]["dist"] == list(G.DEFAULT_SCROLL_DIST)  # amount not said yet
    assert _lab(rows[-1][1], "scroll_amount") == 0 and _lab(rows[-1][1], "complete") == 1.0


def test_side_talk_is_not_a_command():
    ctx = _ctx(random.Random(7))
    toks = [Tok(w) for w in "can you pass the salt".split()]
    for text, lab, drop, qs in _labels_all_prefixes(ctx, toks, None, kind="side", destructive=0.0):
        assert _lab(lab, "intent") == "none" and _lab(lab, "is_command") == 0.0
        assert _lab(lab, "complete") == 0.0 and _lab(lab, "destructive") == 0.0
        for q in ("target", "app", "key", "folder", "text_span", "url_span"):
            assert _lab(lab, q) in (None, "none")


def test_confirm_inherits_risk_from_the_pending_action():
    rng = random.Random(8)
    ctx = _ctx(rng)
    ctx.pending = G.Action(G.ActionKind.CLICK, "u1+0", 0.9, target_label='button "Delete"')
    cmd = None
    for _ in range(50):
        cmd = G.build_fixed(rng, ctx, "confirm")
        if cmd:
            break
    assert cmd is not None and cmd.gold.extra_risk
    assert G.destructive_gold(cmd.toks, 1, cmd.gold) == 1.0


# ----------------------------------------------------------------------------- templates and held-out


def test_every_intent_has_templates_and_a_heldout_share():
    assert set(G.TEMPLATES) == set(INTENTS) - {"wait", "none"}
    for intent, tl in G.TEMPLATES.items():
        assert len(tl) >= 8, intent
        held = G.HELDOUT_TEMPLATES[intent]
        assert len(held) >= max(1, 0.15 * len(tl)), intent
        assert len(held) < len(tl)
        assert G.template_heldout(G.template_id(intent, next(iter(held))))
        assert not G.template_heldout(G.template_id(intent, next(i for i in range(len(tl)) if i not in held)))
    assert len(G.HELDOUT_SIDE) >= 0.15 * len(G.SIDE_TALK)
    assert len(G.HELDOUT_CORR) >= 1 and G.template_heldout(f"click/corr{next(iter(G.HELDOUT_CORR))}")
    assert len(G.HELDOUT_PAYLOADS) >= 0.15 * len(G.TYPE_PAYLOADS)
    assert len(G.HELDOUT_URLS) >= 0.15 * len(G.URL_SITES)


def test_train_contexts_never_draw_heldout_templates():
    rng = random.Random(9)
    for _ in range(300):
        ctx = _ctx(rng)
        intent = rng.choice(sorted(G.TEMPLATES))
        cmd = G.build_command(rng, ctx, intent)
        if cmd is not None:
            assert not G.template_heldout(cmd.template), cmd.template


# ----------------------------------------------------------------------------- randomized sweep


def _sweep_cmds(n: int, seed: int):
    rng = random.Random(seed)
    types = [s for s in SCREEN_TYPES]
    made = 0
    while made < n:
        st = rng.choice(types)
        app = rng.choice(apps_for_screen(st, set(TRAIN_APPS)) or ["Finder"])
        ctx = _ctx(rng, st, app)
        intent = rng.choice(sorted(G.TEMPLATES))
        if intent in ("confirm", "cancel") and rng.random() < 0.8:
            ctx.pending = G.make_pending(rng, ctx)
        if intent == "undo":
            ctx.history = G.make_history(rng, ctx, 2)
        cmd = G.build_command(rng, ctx, intent)
        if cmd is None:
            continue
        made += 1
        yield ctx, cmd


@pytest.mark.parametrize("seed", [0, 1])
def test_generated_commands_label_invariants(seed):
    seen = set()
    for ctx, cmd in _sweep_cmds(250, seed):
        utt = G.compose(ctx.rng, cmd)
        toks = utt.toks
        rows = _labels_all_prefixes(ctx, toks, cmd.gold)
        committed = False
        completed = False
        for text, lab, drop, qs in rows:
            intent = _lab(lab, "intent")
            assert intent in INTENTS
            # every hard choice label is one of that question's options
            for qid, v in lab.items():
                if v["type"] == "choice" and "label" in v:
                    assert v["label"] in qs[qid].criteria, (qid, v, text)
                if v["type"] == "choice" and "dist" in v:
                    assert set(v["dist"]) <= set(qs[qid].criteria) and abs(sum(v["dist"].values()) - 1) < 1e-3
            if intent == "wait":
                assert not committed, f"intent went back to wait: {text!r}"
                assert _lab(lab, "complete") == 0.0
                for q in ("app", "key", "folder", "text_span", "url_span"):
                    assert _lab(lab, q) in (None, "none"), (q, text)
                if _lab(lab, "destructive") == 1.0:
                    assert RISK_RE.search(text), text
            else:
                committed = True
                assert intent == cmd.intent
            if _lab(lab, "complete") == 1.0:
                completed = True
            else:
                assert not completed, f"complete went 1 -> 0 at {text!r}"
            assert ("scroll_amount" in lab) == (intent in G.SCROLL_INTENTS)
            span = _lab(lab, "text_span")
            if span not in (None, "none"):
                assert intent in PAYLOAD_INTENTS and span in text and span in qs["text_span"].criteria
        # the full utterance is a finished command
        t, lab, drop, qs = rows[-1]
        if drop is None:
            assert _lab(lab, "intent") == cmd.intent and _lab(lab, "complete") == 1.0, t
            if cmd.gold.app and cmd.gold.app in qs["app"].criteria:
                assert _lab(lab, "app") == cmd.gold.app
            if cmd.gold.key:
                assert _lab(lab, "key") == cmd.gold.key
            if cmd.gold.folder:
                assert _lab(lab, "folder") == cmd.gold.folder
            if cmd.intent in PAYLOAD_INTENTS:
                assert _lab(lab, "text_span") not in (None, "none")
            if cmd.intent == "open_url":
                assert _lab(lab, "url_span") == cmd.gold.url
            if cmd.intent in ("quit_app", "close_tab"):
                assert _lab(lab, "destructive") == 1.0
        seen.add(cmd.intent)
    assert seen == set(G.TEMPLATES)


def test_click_gold_is_never_an_injected_element_unless_referenced():
    rng = random.Random(11)
    n = 0
    for _ in range(400):
        scr = make_screen(rng, rng.choice(["mail_inbox", "browser_article", "chat_app"]), "Safari", inject_p=1.0)
        ctx = G.Ctx(rng=rng, screen=scr, apps=TRAIN_APPS, installed=TRAIN_APPS[:50], running=["Safari"])
        cmd = G.build_click(rng, ctx)
        if cmd is None or not scr.injected_eids:
            continue
        n += 1
        gold = cmd.gold.target
        ids = {gold} if isinstance(gold, str) else set(gold)
        assert not ids & set(scr.injected_eids)
    assert n > 100
