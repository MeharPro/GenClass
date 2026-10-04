"""The computer-use corpus: runtime-identical inputs, exactly-right labels, honest held-out split.

A module fixture generates a few hundred base utterances through `synth_cu.generate_base` while
recording the context each example was built from, so the tests can rebuild `state` and
`questions` with the shared harness functions and compare them byte for byte.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from typing import Any

import pytest

from jev_local.data import grammar as G
from jev_local.data import synth_cu as S
from jev_local.data.screens import HELDOUT_APPS, HELDOUT_SCREEN_TYPES
from jev_local.data.stats import iter_corpus, validate_example
from jev_local.harness.catalog import INTENTS, PAYLOAD_INTENTS, RISK_RE
from jev_local.harness.questions import build_questions, rank_apps
from jev_local.harness.spans import extract_text_candidates, extract_url_candidates
from jev_local.harness.state import build_state
from jev_local.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, question_from_json, question_to_json

N_BASES = 420
HELD_APP_RE = re.compile(r"\b(" + "|".join(re.escape(a) for a in sorted(HELDOUT_APPS, key=len, reverse=True)) + r")\b")


def _norm(x: Any) -> Any:
    return json.loads(json.dumps(x, ensure_ascii=False))


@pytest.fixture(scope="module")
def sample() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """[(example, recorded context)] for N_BASES base utterances, half of them forced to test."""
    recorded: list[dict[str, Any]] = []
    orig = S._example

    def spy(rng, ex_id, split, family, full, k, kind, cmd, part, ctx, destructive, stats, meta_extra):
        ex = orig(rng, ex_id, split, family, full, k, kind, cmd, part, ctx, destructive, stats, meta_extra)
        if ex is not None:
            recorded.append({
                "id": ex["id"], "snap": ctx.screen.snap, "installed": list(ctx.installed), "running": list(ctx.running),
                "history": list(ctx.history), "pending": ctx.pending, "cmd": cmd, "part": part,
            })
        return ex

    S._example = spy
    try:
        stats = S.Stats()
        exs = []
        for base in range(N_BASES):
            split = "test" if base % 4 == 0 else "train"
            exs += list(S.generate_base(random.Random(1000 + base), base, split, stats))
    finally:
        S._example = orig
    ctx_by_id = {r["id"]: r for r in recorded}
    assert len(ctx_by_id) == len(exs)
    return [(ex, ctx_by_id[ex["id"]]) for ex in exs]


def _lab(ex, qid):
    v = ex["labels"].get(qid)
    if v is None:
        return None
    return v.get("label", v.get("p", v.get("level", v.get("dist"))))


# ----------------------------------------------------------------------------- inputs match the runtime


def test_state_and_questions_are_exactly_the_runtime_ones(sample):
    for ex, rec in sample:
        tr = ex["state"]["transcript"]
        apps = rank_apps(tr, rec["installed"], rec["running"], max_n=24)
        qs = build_questions(rec["snap"], apps, extract_text_candidates(tr), extract_url_candidates(tr), max_elements=60)
        assert ex["questions"] == _norm({qid: question_to_json(q) for qid, q in qs.items()}), ex["id"]
        assert list(ex["questions"]) == list(qs)
        st = build_state(tr, rec["snap"], rec["history"], rec["pending"])
        assert ex["state"] == _norm(st), ex["id"]


def test_examples_round_trip_as_json_and_parse(sample):
    for ex, _ in sample:
        back = json.loads(json.dumps(ex, ensure_ascii=False))
        assert back == _norm(ex)
        for qj in back["questions"].values():
            question_from_json(qj)
        assert set(back) == {"id", "split", "family", "state", "questions", "labels", "meta"}
        assert back["split"] in ("train", "dev", "test")
        m = back["meta"]
        assert m["n_words"] == len(back["state"]["transcript"].split())
        assert isinstance(m["prefix"], bool)


def test_validator_finds_no_problems(sample):
    errs = Counter()
    for ex, _ in sample:
        for e in validate_example(_norm(ex)):
            errs[e] += 1
    assert not errs, errs


# ----------------------------------------------------------------------------- label consistency


def test_every_label_is_an_option_of_its_question(sample):
    for ex, _ in sample:
        for qid, lab in ex["labels"].items():
            q = question_from_json(ex["questions"][qid])
            assert lab["type"] == q.type, (ex["id"], qid)
            if isinstance(q, ChoiceQuestion):
                if "label" in lab:
                    assert lab["label"] in q.criteria, (ex["id"], qid, lab)
                else:
                    assert set(lab["dist"]) <= set(q.criteria) and abs(sum(lab["dist"].values()) - 1) < 1e-3
            elif isinstance(q, NoulQuestion):
                assert lab["p"] in (0.0, 1.0)
            elif isinstance(q, ScoreQuestion):
                if "level" in lab:
                    assert 0 <= lab["level"] < len(q.criteria)
                else:
                    assert len(lab["dist"]) == len(q.criteria) and abs(sum(lab["dist"]) - 1) < 1e-3
        # every harness head except scroll_amount is supervised on every example
        for q in ("intent", "complete", "is_command", "destructive", "app", "key", "folder"):
            assert q in ex["labels"], (ex["id"], q)
        for q in ("target", "text_span"):
            assert (q in ex["labels"]) == (q in ex["questions"]), (ex["id"], q)


def test_wait_none_and_complete_semantics(sample):
    for ex, rec in sample:
        intent, comp, isc = _lab(ex, "intent"), _lab(ex, "complete"), _lab(ex, "is_command")
        side = ex["meta"]["utterance"] == "side"
        assert (intent == "none") == side == (isc == 0.0), ex["id"]
        if intent in ("wait", "none"):
            assert comp == 0.0
            for q in ("app", "key", "folder", "text_span", "url_span"):
                assert _lab(ex, q) in (None, "none"), (ex["id"], q)
        if side:
            assert _lab(ex, "destructive") == 0.0 and _lab(ex, "target") in (None, "none")
        if not ex["meta"]["prefix"] and not side:
            # the full transcript is always a finished first command
            assert intent == rec["cmd"].intent and comp == 1.0, (ex["id"], ex["state"]["transcript"])
        if intent == "wait" and _lab(ex, "destructive") == 1.0:
            assert RISK_RE.search(ex["state"]["transcript"]), ex["id"]  # only from words already spoken


def test_prefix_labels_are_monotone_within_an_utterance(sample):
    groups: dict[str, list] = defaultdict(list)
    for ex, _ in sample:
        groups[ex["id"].rsplit("-", 1)[0]].append(ex)
    n_multi = 0
    for key, exs in groups.items():
        exs.sort(key=lambda e: e["meta"]["n_words"])
        if len(exs) > 1:
            n_multi += 1
        intents = [_lab(e, "intent") for e in exs]
        comps = [_lab(e, "complete") for e in exs]
        dests = [_lab(e, "destructive") for e in exs]
        committed = [i for i in intents if i != "wait"]
        assert len(set(committed)) <= 1, (key, intents)
        if committed:
            assert intents.index(committed[0]) == len(intents) - len(committed), (key, intents)
        assert comps == sorted(comps), (key, comps)
        assert dests == sorted(dests), (key, dests)
        # prefixes are prefixes of the full transcript
        full = exs[-1]["state"]["transcript"].split()
        for e in exs:
            words = e["state"]["transcript"].split()
            assert [w.rstrip(",.;:!?") for w in words] == [w.rstrip(",.;:!?") for w in full[: len(words)]], key
    assert n_multi > N_BASES // 2


def test_argument_labels(sample):
    for ex, rec in sample:
        intent, cmd = _lab(ex, "intent"), rec["cmd"]
        g = cmd.gold if cmd else None
        # arguments of other intents are none
        if intent not in ("open_app", "quit_app"):
            assert _lab(ex, "app") == "none", ex["id"]
        if intent != "press_key":
            assert _lab(ex, "key") == "none", ex["id"]
        if intent != "open_folder":
            assert _lab(ex, "folder") == "none", ex["id"]
        if intent not in PAYLOAD_INTENTS:
            assert _lab(ex, "text_span") in (None, "none"), ex["id"]
        if intent != "open_url":
            assert _lab(ex, "url_span") in (None, "none"), ex["id"]
        if intent not in ("click", "type_text", "search_web", "wait"):
            assert _lab(ex, "target") in (None, "none"), ex["id"]
        # payload span: a verbatim candidate
        span = _lab(ex, "text_span")
        if span not in (None, "none"):
            assert span in ex["questions"]["text_span"]["criteria"] and span in ex["state"]["transcript"]
        if ex["meta"]["prefix"] or g is None:
            continue
        # full transcripts carry the gold argument
        if g.app and g.app in ex["questions"]["app"]["criteria"]:
            assert _lab(ex, "app") == g.app, ex["id"]
        if g.key:
            assert _lab(ex, "key") == g.key, ex["id"]
        if g.folder:
            assert _lab(ex, "folder") == g.folder, ex["id"]
        if intent in PAYLOAD_INTENTS:
            assert span not in (None, "none"), ex["id"]
        if intent == "open_url":
            assert _lab(ex, "url_span") == g.url, ex["id"]
            assert g.url in extract_url_candidates(ex["state"]["transcript"])
        if intent == "click" and "target" in ex["labels"]:
            lab = ex["labels"]["target"]
            if isinstance(g.target, str):
                assert lab.get("label") == g.target, ex["id"]
            else:
                assert lab.get("dist") and set(lab["dist"]) == set(g.target), ex["id"]
        if intent in G.SCROLL_INTENTS:
            assert "scroll_amount" in ex["labels"]


def test_injected_elements_are_never_gold_unless_referenced(sample):
    n_inj = 0
    for ex, _ in sample:
        inj = set(ex["meta"]["injected"]) - {"title"}
        if not inj:
            continue
        n_inj += 1
        lab = ex["labels"].get("target", {})
        gold = {lab["label"]} if "label" in lab else set(lab.get("dist", {}))
        if ex["meta"]["gold"].get("ref") != "injected":
            assert not gold & inj, ex["id"]
    assert n_inj > 10


def test_destructive_uses_the_risk_lexicon_on_command_and_target(sample):
    checked = 0
    for ex, rec in sample:
        m = ex["meta"]
        if m["prefix"] or m["utterance"] != "command":
            continue
        intent, d = _lab(ex, "intent"), _lab(ex, "destructive")
        text = ex["state"]["transcript"]
        payload = m["gold"].get("text")
        cmd_text = text.replace(payload, " ") if payload else text
        tgt = ex["labels"].get("target", {}).get("label")
        tgt_line = ex["questions"]["target"]["criteria"][tgt] if tgt not in (None, "none") else ""
        if intent == "cancel":
            assert d == 0.0
            continue
        expect = (intent in ("quit_app", "close_tab") or rec["cmd"].gold.extra_risk or bool(RISK_RE.search(cmd_text))
                  or bool(RISK_RE.search(re.sub(r" \(\d+ of \d+\)", "", tgt_line.split('"')[1] if '"' in tgt_line else ""))))
        assert d == float(expect), (ex["id"], text, tgt_line)
        checked += 1
    assert checked > 200


# ----------------------------------------------------------------------------- coverage


def test_coverage_of_intents_screens_and_phenomena(sample):
    intents = Counter(_lab(ex, "intent") for ex, _ in sample)
    assert set(intents) == set(INTENTS), set(INTENTS) - set(intents)
    kinds = Counter(ex["meta"]["utterance"] for ex, _ in sample)
    assert {"command", "chain", "side", "correction"} <= set(kinds)
    screens = Counter(ex["meta"]["screen_type"] for ex, _ in sample)
    assert len(screens) >= 30
    n_el = [ex["meta"]["n_elements"] for ex, _ in sample if ex["meta"]["screen_type"] != "no_elements"]
    assert min(n_el) >= 5 and max(n_el) <= 60
    refs = Counter(ex["meta"]["gold"].get("ref") for ex, _ in sample)
    for r in ("label", "keyword", "none", "ordinal"):
        assert refs[r] > 0, r
    assert any(ex["meta"].get("step") == 2 for ex, _ in sample)
    assert any(ex["state"]["pending"] != "none" for ex, _ in sample)
    assert any(ex["state"]["recent_actions"] != "none" for ex, _ in sample)
    noise = Counter(op for ex, _ in sample for op in ex["meta"]["noise"])
    assert {"lower", "nopunct", "drop", "repeat", "filler", "homophone"} <= set(noise)
    dist_targets = sum(1 for ex, _ in sample if "dist" in ex["labels"].get("target", {}))
    assert dist_targets > 0  # ambiguous duplicates get soft labels
    prefix = sum(ex["meta"]["prefix"] for ex, _ in sample)
    assert 0.3 < prefix / len(sample) < 0.8


# ----------------------------------------------------------------------------- splits


def _heldout_present(ex) -> set[str]:
    m = ex["meta"]
    dims = set()
    if G.template_heldout(m["template"]):
        dims.add("template")
    if m["screen_type"] in HELDOUT_SCREEN_TYPES:
        dims.add("screen")
    if m["app"] in HELDOUT_APPS or m["gold"].get("app") in HELDOUT_APPS:
        dims.add("app")
    return dims


def test_train_and_dev_never_see_heldout_items(sample):
    urls = {G.URL_SITES[i] for i in G.HELDOUT_URLS}
    for ex, rec in sample:
        if ex["split"] == "test":
            continue
        assert not _heldout_present(ex), ex["id"]
        assert not HELD_APP_RE.search(json.dumps(ex["state"])), ex["id"]
        assert not set(ex["questions"]["app"]["criteria"]) & HELDOUT_APPS, ex["id"]
        assert not set(rec["installed"]) & HELDOUT_APPS
        assert ex["meta"]["gold"].get("url") not in urls
        assert "|heldout:" not in ex["family"]


def test_test_examples_always_show_a_heldout_dimension(sample):
    dims = Counter()
    for ex, _ in sample:
        if ex["split"] != "test":
            continue
        present = _heldout_present(ex)
        assert present, ex["id"]
        assert set(ex["meta"]["heldout"]) == present
        assert ex["family"].endswith("|heldout:" + "+".join(sorted(present, key=["template", "screen", "app"].index)))
        dims.update(present)
    assert {"template", "screen", "app"} <= set(dims)


def test_family_sets_are_disjoint_across_splits(sample):
    fams: dict[str, set[str]] = defaultdict(set)
    for ex, _ in sample:
        if ex["meta"].get("step") == 2:
            continue  # a chain's second step follows its base utterance's split (no near-duplicates across splits)
        fams[ex["split"]].add(ex["family"])
    assert not fams["train"] & fams["test"]
    assert not fams["train"] & fams["dev"]


# ----------------------------------------------------------------------------- driver


def test_generate_writes_splits_manifest_and_is_deterministic(tmp_path):
    a = S.generate(120, tmp_path / "a", seed=3, log_every=0)
    b = S.generate(120, tmp_path / "b", seed=3, log_every=0)
    for s in ("train", "dev", "test"):
        assert (tmp_path / "a" / f"{s}.jsonl").read_bytes() == (tmp_path / "b" / f"{s}.jsonl").read_bytes()
    assert a["n_written"] >= 120 and a["n_written"] == sum(a["splits"].values())
    man = json.loads((tmp_path / "a" / "manifest.json").read_text())
    assert man["heldout"]["screen_types"] and man["heldout"]["apps"] and "drops" in man
    exs = list(iter_corpus(tmp_path / "a"))
    assert len(exs) == a["n_written"]
    assert len({e["id"] for e in exs}) == len(exs)
    for e in exs:
        assert e["split"] in ("train", "dev", "test")
        assert not validate_example(e), e["id"]
