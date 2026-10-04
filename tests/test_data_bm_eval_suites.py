"""bm-z-data eval-suite fetch (jev_local/data/bm/eval_suites.py): pure-Python parts.

No network, no numpy/torch/datasets: the pinned-revision picker, the item writer's shape (read back by
decontam.read_bench), the reproduced AbdelStark sampler, the DMB NLU++ decision rule, the Jevals serialisation
candidates, the targets.json mappings (Deußer task -> target, DI dataset -> rows) and the coverage invariant
(every counted targets row is fetched by a suite or recorded as gated / unavailable)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev_local.data.bm import eval_suites as E

TARGETS = Path(__file__).resolve().parents[1] / "bench" / "public" / "targets.json"


@pytest.fixture(scope="module")
def targets():
    return E.load_targets(TARGETS)


# ------------------------------------------------------------------------------------------- pure helpers
def test_pick_commit_last_on_or_before_cutoff():
    commits = [("c3", "2026-09-30T10:00:00+00:00"), ("c2", "2026-09-26T23:00:00+00:00"), ("c1", "2024-03-07T12:02:37+00:00")]
    assert E.pick_commit(commits, "2026-09-26T23:59:59+00:00") == ("c2", "2026-09-26T23:00:00+00:00")
    assert E.pick_commit(commits, "2025-01-01T00:00:00+00:00") == ("c1", "2024-03-07T12:02:37+00:00")
    assert E.pick_commit(commits, "2020-01-01T00:00:00+00:00") is None
    assert E.pick_commit([("z", "2026-09-26T12:00:00Z")], E.CUTOFF)[0] == "z"  # Z suffix


def test_sha256_lines_variants():
    lines = ['{"a": 1}', '{"b": 2}']
    import hashlib

    assert E.sha256_lines(lines, True) == hashlib.sha256(b'{"a": 1}\n{"b": 2}\n').hexdigest()
    assert E.sha256_lines(lines, False) == hashlib.sha256(b'{"a": 1}\n{"b": 2}').hexdigest()


def test_class_count_and_balanced_indices():
    assert E.class_count(["t1", "t1", "t1", "t2", "t2", "t2"]) == 3
    with pytest.raises(ValueError):
        E.class_count(["same", "same"])
    targets = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    chosen = E.balanced_indices(targets, 6, seed=20260917)
    assert chosen == sorted(chosen) and len(chosen) == 6
    assert sorted(targets[i] for i in chosen) == [0, 0, 1, 1, 2, 2]  # round-robin over classes
    assert E.balanced_indices(targets, 6, seed=20260917) == chosen  # deterministic
    assert len(E.balanced_indices([0, 0, 1], 10, seed=1)) == 3  # limit above n


def test_nlupp_decisions_domain_plus_general():
    onto = {"affirm": {"domain": ["general"]}, "open_account": {"domain": ["banking"]}, "book_room": {"domain": ["hotels"]},
            "cancel": {"domain": ["banking", "hotels"]}}
    assert E.nlupp_decisions([], onto, "banking") == ["affirm", "cancel", "open_account"]
    assert E.nlupp_decisions([], onto, "hotels") == ["affirm", "book_room", "cancel"]


def test_jevals_serializers_cover_js_stringify():
    sers = E.jevals_serializers()
    assert len(sers) == 8
    obj = {"question": "Is x?", "context": {"contexts": ["a", "b"]}}
    assert sers["json:utf8:compact:insertion"](obj) == '{"question":"Is x?","context":{"contexts":["a","b"]}}'  # JSON.stringify
    assert sers["json:utf8:compact:sorted"](obj).startswith('{"context"')


# ------------------------------------------------------------------------------------------- writer / ctx
def test_item_writer_rows_are_read_by_decontam_read_bench(tmp_path):
    from jev_local.data.v2.decontam import read_bench

    w = E.ItemWriter(tmp_path / "suite", "x")
    q = {"answer": {"type": "choice", "instructions": "Topic?", "criteria": {"World": None, "Sports": None}}}
    w.add("suite/ds", 0, {"article": "A long enough article about sports and the world cup."}, q, "Sports", {"subset": "en"})
    w.add("suite/ds", 1, "plain string state", {}, None)
    w.add("suite/ds", 1, "plain string state", {})  # same item + same questions -> deduplicated by read_bench
    info = w.close()
    assert info["rows"] == 3 and info["by_dataset"] == {"suite/ds": 3} and len(info["sha256"]) == 64
    bench = tmp_path / "bench"
    (bench / "test").mkdir(parents=True)
    (bench / "test" / "suite__x.jsonl").symlink_to(Path(info["path"]))
    items, hashes = read_bench(bench)
    assert [it.dataset for it in items] == ["suite/ds", "suite/ds"]
    assert items[0].state == {"article": "A long enough article about sports and the world cup."}
    assert items[0].questions == q and items[0].role == "test"
    assert set(hashes) == {"test/suite__x.jsonl"}


def _toy_targets():
    return {"targets": [{"id": "T001", "n": 10, "counted": True, "spec_id": "toy"}, {"id": "T002", "n": 5, "counted": True, "spec_id": "toy"}]}


def test_ctx_check_modes(tmp_path):
    ctx = E.Ctx("toy", tmp_path, _toy_targets())
    assert ctx.check("a", ["T001"], 10).ok is True
    assert ctx.check("b", ["T001"], 9).ok is False
    assert ctx.check("c", ["T001"], 11, "superset").ok is True
    assert ctx.check("d", ["T001", "T002"], 5).expected == 5  # the smallest n among the rows (the sample size)
    assert ctx.check("e", [], 3).ok is False and ctx.check("f", [], 3, "info").ok is None
    ctx.pend("x", "why", ["T002"])
    assert ctx.pending[0]["target_ids"] == ["T002"]


def test_run_suite_writes_manifest(tmp_path, monkeypatch):
    def toy(ctx: E.Ctx):
        w = ctx.writer("items")
        for i in range(10):
            w.add("toy/a", i, f"state number {i} with enough words", {})
        ctx.done(w)
        ctx.pin("hf:toy", "abc")
        ctx.check("toy/a", ["T001"], 10)
        return {"k": 1}

    monkeypatch.setitem(E.SUITES, "toy", toy)
    man = E.run_suite("toy", _toy_targets(), eval_dir=tmp_path)
    assert man["rows"] == 10 and man["checks_ok"] and man["pins"] == {"hf:toy": "abc"} and man["extra"] == {"k": 1}
    on_disk = json.loads((tmp_path / "toy" / "manifest.json").read_text())
    assert on_disk["datasets"] == {"toy/a": 10} and on_disk["files"][0]["rows"] == 10


# ------------------------------------------------------------------------------------------- targets.json maps
def test_deusser_task_targets(targets):
    m = E.deusser_task_targets(targets)
    assert len(m) == 37
    assert m["ag_news"] == ("T000", 7600) and m["belebele"] == ("T017", 109800) and m["helpsteer2"] == ("T036", 1038)
    assert m["llm_aggrefact"][0] == "T015" and m["toxigen"][0] == "T027"  # gated: pending, never fetched


def test_di_target_rows_all_matched(targets):
    rows = E.di_target_rows(targets)
    kit_names = set(E.DI_DATASETS.values())
    unmatched = sorted(k for k in rows if k not in kit_names)
    assert set(unmatched) <= E.DI_NOT_IN_REBUILD, unmatched  # shown-not-indexed extras only; no diagnostics
    assert {r["id"] for r in rows["BANKING77"]} >= {"T057"} and {r["id"] for r in rows["ToolRet"]} == {"T055"}
    assert "GPQA Diamond" in rows and "HLE" in rows and "OpenBookQA" in rows
    assert all(r["role"] in ("headline", "secondary", "s_bar", "context") for rs in rows.values() for r in rs)


def test_coverage_has_no_gap(targets):
    cov = E.coverage(targets)
    assert cov["gap"] == [], cov["gap"]
    counted = sum(1 for r in targets["targets"] if r["counted"])
    assert len(cov["suite"]) + len(cov["pending"]) == counted == 154
    assert cov["suite"]["T000"] == "deusser" and cov["suite"]["T112"] == "dmb" and cov["suite"]["T521"] == "studies"
    assert cov["pending"]["T015"] == "gated" and "T469" in cov["pending"]


def test_gated_and_unavailable_ids_exist(targets):
    ids = {r["id"] for r in targets["targets"]}
    for tids in E.GATED.values():
        assert set(tids) <= ids
    assert set(E.UNAVAILABLE) <= ids
    assert all(len(v) == 40 for k, v in E.PINS.items() if k in ("deusser", "decision_index", "jevbench", "abdelstark", "jevals", "dmb", "workflowevals", "typed_decisions", "btzsc"))


def test_hf_sets_have_targets_and_state_fn():
    keys = [s.key for s in E.HF_SETS]
    assert len(keys) == len(set(keys))
    for s in E.HF_SETS:
        assert s.target_ids and s.mode in ("exact", "superset", "info", "none")
    row = {"text": "hi", "Consumer Complaint": "c", "utterance": "u"}
    assert next(s for s in E.HF_SETS if s.key == "hwu64_test").state(row) == "u"
    assert next(s for s in E.HF_SETS if s.key == "cfpb_test").state(row) == "c"
