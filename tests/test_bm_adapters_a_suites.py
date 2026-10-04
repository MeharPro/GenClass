"""Deußer / Decision Index / jev-bench adapter plumbing (pure Python, no datasets, no kit, no network)."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from jev_local.bench.benchmax.adapters_a import decision_index as DI
from jev_local.bench.benchmax.adapters_a import deusser as D
from jev_local.bench.benchmax.adapters_a import jevbench_hf as J

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ Deußer


def test_deusser_request_key_matches_their_cache_py():
    """`cache.request_key` = sha256 of json.dumps({"model","state","questions"}, ensure_ascii=False); no sort_keys."""
    model, state, questions = "meharsjev-68m", {"query": "héllo"}, {"answer": {"type": "choice", "instructions": "i", "criteria": {"b": None, "a": None}}}
    expected = hashlib.sha256(json.dumps({"model": model, "state": state, "questions": questions}, ensure_ascii=False).encode()).hexdigest()
    local = ROOT / "bench" / "sources" / "deusser"
    if not (local / "COMMIT").exists():
        pytest.skip("bench/sources/deusser (the Mac's read-only copy of their cache.py) is not on this machine")
    assert (local / "COMMIT").read_text().strip() == D.PINNED_COMMIT
    # our own vendored copy of the key function must agree (the real run imports theirs)
    assert D.shard_of(expected, 1) == 0 and 0 <= D.shard_of(expected, 7) < 7
    rec = D.response_record(expected, "banking77", "12", model, {"answers": {"answer": {"type": "choice", "choice": "a", "confidence": 0.5, "probabilities": {"b": 0.25, "a": 0.75}}}, "usage": {"input_tokens": 9, "output_tokens": 3}}, 0.0123)
    assert set(rec) == {"key", "task", "uid", "response", "latency_s"} and rec["response"]["model"] == model
    assert rec["response"]["answers"]["answer"]["choice"] == "a" and rec["latency_s"] == 0.0123
    with pytest.raises(ValueError):
        D.response_record(expected, "t", "u", "jev-1.13.0", {"answers": {}}, 0.0)


def test_deusser_expected_n_from_targets_and_source_url_mapping():
    assert D.task_of_source_url("https://github.com/AppliedMachineLearning-Lab/jev-benchmarking/blob/main/results/eval/ag_news.json") == "ag_news"
    assert D.task_of_source_url(None) is None and D.task_of_source_url("https://x/y.json") is None
    n = D.expected_n_by_task("deusser_exact@6bbdeb33")
    assert n["ag_news"] == 7600 and n["banking77"] == 3076 and n["imdb"] == 25000 and n["prompt_injections"] == 116
    assert len(n) == 37, "one headline row per Deußer task"


def test_deusser_missing_wrong_variant():
    assert D.missing_wrong("accuracy", 0.9, 90, 100) == pytest.approx(0.81)
    assert D.missing_wrong("f1", 0.9, 90, 100) is None and D.missing_wrong("accuracy", None, 1, 1) is None


def test_deusser_adapter_verify_without_checkout(tmp_path):
    from jev_local.bench.benchmax import get_spec
    from jev_local.bench.benchmax.runner import RunContext

    spec = get_spec("deusser_exact@6bbdeb33")
    args = types.SimpleNamespace(harness_dir=str(tmp_path / "nope"), tasks=["all"], split="eval", limit=None, subsets=None, shard=0, num_shards=1,
                                 revisions=None, no_evaluate=False, no_ci=False, harness_python=None)
    rep = D.DeusserAdapter(spec, args).verify(RunContext(spec, tmp_path, "meharsjev-68m", {}))
    assert rep["ok"] is False and rep["commit"]["present"] is False and rep["expected_n"]["sst2"] == 872
    assert D.DeusserAdapter.default_overflow == "truncate" and D.SELF_CHECK_TASKS == ("sst2", "emotion", "paws", "stsb", "go_emotions")


# ------------------------------------------------------------------ Decision Index


def test_di_constants_match_the_specs_note():
    assert DI.ROWS_SHA256.startswith("b2b56d6f") and DI.ADDED_SHA256.startswith("7429f3c9") and DI.EXCLUSIONS_SHA256.startswith("331df32d")
    assert DI.PINNED_COMMIT.startswith("87d4650b") and DI.EDITION == "0.2.1"
    assert DI.DecisionIndexAdapter.default_overflow == "refuse" and DI.DecisionIndexAdapter.allow_truncate is False


def test_di_gunzipped_hash_and_fixture_parsing(tmp_path):
    import gzip

    data = b'{"a":1}\n'
    (tmp_path / "rows.jsonl.gz").write_bytes(gzip.compress(data))
    (tmp_path / "rows.jsonl").write_bytes(data)
    assert DI.sha256_gunzipped(tmp_path / "rows.jsonl.gz") == DI.sha256_gunzipped(tmp_path / "rows.jsonl") == hashlib.sha256(data).hexdigest()
    fx = tmp_path / "tests" / "fixtures"
    fx.mkdir(parents=True)
    (fx / "board-0.2.1.json").write_text(json.dumps({"entrants": [{"name": "Other", "benchmarks": {}}, {"name": "Jev", "scores": {"balanced_skill": 57.89},
                                                     "benchmarks": {"4": {"board": {"raw": 0.7974, "skill": 0.7948, "coverage": 1.0}, "result": {}}}}]}))
    ref = DI.jev_fixture_values(tmp_path)
    assert ref["index"] == {"balanced_skill": 57.89} and ref["benchmarks"]["4"]["raw"] == 0.7974
    assert DI.jev_fixture_values(tmp_path / "nowhere") == {}


def test_di_engine_subclass_maps_refusal_to_unsupported(monkeypatch):
    """Inject a stub `decision_index.engines.base` so the lazily built Engine subclass can be exercised here."""
    base = types.ModuleType("decision_index.engines.base")

    class Unsupported(ValueError):
        pass

    class Engine:
        def __init__(self, **options):
            self.options = options

    base.Engine, base.Unsupported = Engine, Unsupported
    pkg = types.ModuleType("decision_index")
    eng_pkg = types.ModuleType("decision_index.engines")
    monkeypatch.setitem(sys.modules, "decision_index", pkg)
    monkeypatch.setitem(sys.modules, "decision_index.engines", eng_pkg)
    monkeypatch.setitem(sys.modules, "decision_index.engines.base", base)

    from jev_local.bench.benchmax import runner as R

    class FakeClient:
        kind = "fake"

        def __init__(self, *a, **kw):
            self.model_id = "meharsjev-68m"

        def describe(self):
            return {"engine": "fake", "max_tokens": 100}

        def system_one(self, state, questions):
            if "long" in json.dumps(state):
                return R.Outcome("refused", None, R.refusal_body("context", model_id="meharsjev-68m", need=9, limit=1), 400)
            if "boom" in json.dumps(state):
                return R.Outcome("error", None, {"detail": "boom"}, 500)
            return R.Outcome("ok", {"model": "meharsjev-68m", "answers": {"q": {"type": "choice", "choice": "A", "probabilities": {"A": 0.6, "B": 0.4}}}, "usage": {}}, None, 200, 0.01)

    monkeypatch.setattr(R, "LocalClient", FakeClient)
    monkeypatch.setattr(R, "HttpClient", FakeClient)
    cls = DI.make_engine_class()
    e = cls(ckpt="/x", model="meharsjev-68m")
    assert e.provenance["policy"].startswith("Unmodified")
    resp, raw = e("ok", {"q": {"type": "choice", "criteria": {"A": "a", "B": "b"}}})
    assert resp["answers"]["q"]["choice"] == "A" and "latency_s" in raw
    with pytest.raises(Unsupported) as ei:
        e("long", {"q": {}})
    assert R.CONTEXT_MARKER in str(ei.value)
    with pytest.raises(RuntimeError):
        e("boom", {"q": {}})
    with pytest.raises(ValueError):
        cls()  # neither ckpt nor base_url
    assert DI.__getattr__("MeharsjevEngine") is not None
    with pytest.raises(AttributeError):
        DI.__getattr__("Nope")


def test_di_server_config_problems():
    good = {"precision": "exact", "overflow": "refuse", "refusal_status": 400, "refusal_style": "jev", "name": "di"}
    assert DI.config_problems(good) == []
    bad = DI.config_problems({"precision": "round2", "overflow": "refuse", "refusal_style": "local"})
    assert len(bad) == 2 and any("precision" in p for p in bad) and any("refusal_style" in p for p in bad)
    assert DI.config_problems(None) == ["no benchmax config in /healthz (old server?)"]


def test_di_results_summary(tmp_path):
    p = tmp_path / "results.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"run_id": "a", "status": "ok"}, {"run_id": "b", "status": "unsupported", "error": "maximum context length"}, {"run_id": "c", "status": "error", "error": "x"}]) + "\n")
    s = DI.DecisionIndexAdapter._summarize_results(p)
    assert s["rows"] == 3 and s["counts"] == {"ok": 1, "unsupported": 1, "error": 1} and len(s["samples"]) == 2
    assert DI.DecisionIndexAdapter._summarize_results(tmp_path / "none.jsonl") == {"present": False}


# ------------------------------------------------------------------ jev-bench (HF)


def _row(primitive, question, label, soft=None, state="the text"):
    return {"id": f"{primitive}/x/1", "source": "src", "primitive": primitive, "split": "validation", "state": json.dumps(state),
            "question": json.dumps(question), "label": label, "soft_label": None if soft is None else json.dumps(soft),
            "meta": json.dumps({"hf_id": "a/b"})}


def test_jevbench_record_parsing_and_request_shape():
    q = {"type": "choice", "instructions": "Which?", "criteria": {"b": "B", "a": "A"}}
    rec = J.parse_row(_row("choice", q, "a"))
    assert rec.state == "the text" and rec.option_keys() == ["b", "a"] and rec.label_index() == 1
    body = J.request_body(rec, "meharsjev-68m")
    assert body == {"state": "the text", "model": "meharsjev-68m", "questions": {"q": q}}
    s = J.parse_row(_row("score", {"type": "score", "criteria": ["l0", "l1", "l2"]}, "2"))
    assert s.label == 2 and s.option_keys() == ["0", "1", "2"] and s.label_index() == 2
    n = J.parse_row(_row("noul", {"type": "noul", "instructions": "yes?"}, "1", state={"passage": "p", "question": "q"}))
    assert n.label == 1 and n.option_keys() == ["0", "1"] and n.state == {"passage": "p", "question": "q"}


def test_jevbench_predictions_follow_jevify_rules():
    q = {"type": "choice", "criteria": {"a": "A", "b": "B", "c": "C"}}
    rec = J.parse_row(_row("choice", q, "b"))
    p = J.probs_of(rec, {"type": "choice", "choice": "b", "probabilities": {"a": 0.3, "b": 0.3, "c": 0.4}})
    assert J.predicted_index(rec, p) == 2
    assert J.predicted_index(rec, [0.4, 0.4, 0.2]) == 0, "np.argmax: first maximum"
    n = J.parse_row(_row("noul", {"type": "noul"}, "1"))
    assert J.predicted_index(n, J.probs_of(n, {"type": "noul", "noul": 0.5})) == 1 and J.predicted_index(n, J.probs_of(n, {"noul": 0.49})) == 0
    s = J.parse_row(_row("score", {"type": "score", "criteria": ["l0", "l1", "l2"]}, "1"))
    assert J.predicted_index(s, J.probs_of(s, {"type": "score", "score": 1.0, "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2}})) == 1


def test_jevbench_report_and_macro():
    q = {"type": "choice", "criteria": {"a": "A", "b": "B"}}
    recs = [J.parse_row(_row("choice", q, "a", soft={"a": 0.8, "b": 0.2})), J.parse_row(_row("choice", q, "b")), J.parse_row(_row("choice", q, "a"))]
    answers = [{"type": "choice", "choice": "a", "probabilities": {"a": 0.9, "b": 0.1}}, {"type": "choice", "choice": "a", "probabilities": {"a": 0.6, "b": 0.4}}, None]
    r = J.report(recs, answers)
    assert r["n"] == 3 and r["n_answered"] == 2 and r["n_missing"] == 1
    assert r["accuracy"] == pytest.approx(1 / 3) and r["accuracy_answered"] == pytest.approx(0.5)
    assert r["brier"] == pytest.approx(((0.1 ** 2 + 0.1 ** 2) + (0.6 ** 2 + 0.6 ** 2)) / 2)
    assert r["tvd_to_human"] == pytest.approx(0.1)
    assert 0 <= r["ece"] <= 1
    m = J.macro({"c1": r, "c2": {"accuracy": 1.0, "ece": 0.0, "brier": 0.0}}, {"c1": "choice", "c2": "noul"})
    assert m["macro_accuracy"] == pytest.approx((1 / 3 + 1.0) / 2) and m["choice_acc"] == pytest.approx(1 / 3) and m["noul_acc"] == 1.0 and m["n_configs"] == 2


def test_jevbench_ece15_matches_jevify_binning():
    # all in the top bin, 90% confident, 90% right -> 0
    assert J.ece15([0.9] * 10, [1.0] * 9 + [0.0]) == pytest.approx(0.0)
    # conf 1.0 (bin 14), accuracy 0 -> 1.0 ; a 0.0 confidence lands in the first (closed) bin
    assert J.ece15([1.0, 1.0], [0.0, 0.0]) == pytest.approx(1.0)
    assert J.ece15([0.0], [0.0]) == pytest.approx(0.0)


def test_jevbench_expected_counts_and_targets_mapping():
    assert len(J.CONFIGS) == 22 and sum(J.EXPECTED_TEST_N.values()) == 22773
    assert J.EXPECTED_TEST_N["chaosnli"] == 1599 and J.EXPECTED_TEST_N["civil_comments"] == 2000 and J.EXPECTED_TEST_N["sms_spam"] == 800
    assert J.config_of_target({"dataset": "chaosnli (choice)"}) == "chaosnli" and J.config_of_target({"dataset": "LexGLUE LEDGAR"}) is None
    from jev_local.bench.benchmax.runner import load_targets

    rows = load_targets("jevbench_hf_praveenrajus_v0.1.1")
    head = {J.config_of_target(r): r["n"] for r in rows if r["role"] == "headline"}
    assert len(head) == 13 and all(J.EXPECTED_TEST_N[c] == n for c, n in head.items())
    assert J.JEV_CARD_MACRO["macro_accuracy"] == 0.733 and set(J.JEV_CARD_ACC) == set(J.CONFIGS)
