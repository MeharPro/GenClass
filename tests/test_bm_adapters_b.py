"""Pure-Python tests for the group-B benchmax adapters (no torch / datasets / transformers imports)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TARGETS = json.loads((ROOT / "bench" / "public" / "targets.json").read_text())["targets"]
HEAVY = ("torch", "transformers", "datasets", "sklearn", "pandas", "pyarrow")


def _answers(mapping):
    return {k: {"ok": True, "answers": v} for k, v in mapping.items()}


def _work(tmp_path):
    return argparse.Namespace(work=str(tmp_path))


# ---------------------------------------------------------------------------------------------- registry


def test_specs_b_registers_every_target_row():
    from jev_local.bench.benchmax import load_specs
    from jev_local.bench.benchmax import specs_b

    specs = load_specs()
    assert set(specs_b.SPEC_IDS) <= set(specs)
    rows_by_spec, counted_by_spec = {}, {}
    for r in TARGETS:
        rows_by_spec.setdefault(r.get("spec_id"), set()).add(r["id"])
        if r.get("counted"):
            counted_by_spec.setdefault(r.get("spec_id"), set()).add(r["id"])
    for sid in specs_b.SPEC_IDS:
        assert sid in rows_by_spec, f"{sid} is not a spec_id in targets.json"
        cls = specs_b.adapter_class(sid)
        assert cls.SPEC_ID == sid
        # every implemented target is a row of this spec, and every counted row is implemented
        # (DMB's uncounted context rows T113-T119 - pilots, v1.1 historical S1/S2/S4/S5 - are a documented gap)
        assert set(cls.TARGETS) <= rows_by_spec[sid], (sid, set(cls.TARGETS) - rows_by_spec[sid])
        assert counted_by_spec.get(sid, set()) <= set(cls.TARGETS), (sid, counted_by_spec.get(sid, set()) - set(cls.TARGETS))
        assert specs_b.spec(sid).owner == "b"
        assert cls.TASKS and cls.SPLITS
    # module-level import hygiene, checked in a fresh interpreter (in a full-suite run other tests have already
    # imported torch / pandas into this process)
    code = ("import sys\nfrom jev_local.bench.benchmax import specs_b\n"
            "for sid in specs_b.SPEC_IDS:\n    specs_b.adapter_class(sid)\n"
            f"print(','.join(h for h in {HEAVY!r} if h in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()
    assert out == "", f"imported at module level: {out}"


def test_adapters_construct_without_network(tmp_path):
    from jev_local.bench.benchmax import make_adapter, specs_b

    for sid in specs_b.SPEC_IDS:
        ad = make_adapter(specs_b.spec(sid), _work(tmp_path))
        assert ad.work == tmp_path
        assert ad.spec.id == sid
        assert isinstance(ad.expected_counts("test"), dict)
        assert ad.raw_dir("x").is_dir()
    with pytest.raises(KeyError):
        specs_b.spec("nope")


def test_driver_guard_and_adapter_factory(tmp_path):
    from jev_local.bench.benchmax.adapters_b import driver

    ad = driver._adapter("stperic_medhallu", str(tmp_path))
    assert ad.SPEC_ID == "stperic_medhallu"
    with pytest.raises(SystemExit):
        driver._guard_test(argparse.Namespace(split="test", allow_test=False))
    driver._guard_test(argparse.Namespace(split="test", allow_test=True))
    driver._guard_test(argparse.Namespace(split="validation", allow_test=False))


# ---------------------------------------------------------------------------------------------- common


def test_common_metrics():
    from jev_local.bench.benchmax.adapters_b import common as c

    assert c.macro_f1([0, 1, 1], [0, 1, 2], labels=[0, 1, 2]) == pytest.approx((1.0 + 2 / 3 + 0.0) / 3)
    assert c.accuracy(["a", "b"], ["a", "c"]) == 0.5
    assert c.ndcg_at_k(["d1", "d2", "d3"], {"d2": 1}, k=10, exponential=True) == pytest.approx(1 / __import__("math").log2(3))
    assert c.roc_auc([0.9, 0.8, 0.3, 0.1], [True, True, False, False]) == 1.0
    assert c.average_precision([0.9, 0.1, 0.8], [True, False, True]) == 1.0
    lo, hi = c.wilson_ci(50, 100)
    assert lo < 0.5 < hi
    assert c.choice_pick({"choice": "b", "probabilities": {"a": 0.6, "b": 0.4}}, ["a", "b"]) == "b"
    assert c.choice_pick({"choice": "zzz", "probabilities": {"a": 0.6, "b": 0.4}}, ["a", "b"]) == "a"
    assert c.noul_p({"noul": 0.25}) == 0.25 and c.noul_p({"choice": "x"}) is None
    assert c.spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)
    assert c.answers_of({"ok": False, "answers": {}}) is None


def test_engine_client_run_items_resume(tmp_path):
    from jev_local.bench.benchmax.adapters_b.common import Item
    from jev_local.bench.benchmax.adapters_b.engine_client import AnswerError, load_answers, run_items

    items = [Item(f"t/{i}", "t", {"state": str(i), "questions": {"q": {"type": "noul", "instructions": "?"}}}, gold=i) for i in range(4)]
    calls = []

    def answerer(req):
        calls.append(req["state"])
        if req["state"] == "2":
            raise AnswerError(400, {"error": "bad"})
        return {"answers": {"q": {"noul": 0.5}}, "usage": {"input_tokens": 3}, "model": "meharsjev-68m", "truncated": False}

    out = tmp_path / "answers.jsonl"
    run_items(items, answerer, out, resume=True, log_every=0)
    recs = load_answers(out)
    assert {k for k, v in recs.items() if v["ok"]} == {"t/0", "t/1", "t/3"}
    assert recs["t/2"]["ok"] is False and recs["t/2"]["status"] == 400
    n = len(calls)
    run_items(items, answerer, out, resume=True, log_every=0)  # only the failed non-permanent row may be retried
    assert len(calls) - n <= 1


# ---------------------------------------------------------------------------------------------- chepyle


def test_chepyle_requests_and_scoring():
    from jev_local.bench.benchmax.adapters_b import chepyle_lexglue as m

    short, flag = m.truncate("abc", 48000)
    assert (short, flag) == ("abc", False)
    long, flag = m.truncate("x" * 60000, 48000)
    assert flag and len(long) == 48000 and m.MARKER in long
    req, info = m.build_request("ecthr_a", "facts", None)
    k = len(m.task_entry("ecthr_a")["codes"])
    assert set(req["questions"]) == {f"label_{i}" for i in range(k)} and req["state"] == {"document": "facts"}
    q0 = req["questions"]["label_0"]
    assert q0["type"] == "noul" and q0["instructions"].startswith(m.INSTRUCTIONS["ecthr_a"]) and set(q0["criteria"]) == {"true", "false"}
    req, _ = m.build_request("case_hold", "ctx", ["h1", "h2", "h3", "h4", "h5"])
    assert list(req["questions"]["label"]["criteria"]) == ["0", "1", "2", "3", "4"]
    with pytest.raises(ValueError):
        m.build_request("case_hold", "ctx", ["only", "four", "ho", "ld"])
    answers = {f"label_{i}": {"noul": 0.9 if i in (0, 2) else 0.1} for i in range(k)}
    pred, probs = m.predictions_from_answers("ecthr_a", answers)
    assert pred == [0, 2] and len(probs) == k
    pred, _ = m.predictions_from_answers("ecthr_a", answers, [0.95] * k)
    assert pred == []
    assert m.predictions_from_answers("ecthr_a", None) == (None, None)
    s = m.score_task("ecthr_a", [[0, 2], []], [[0, 2], []])
    assert s["micro_f1"] == 1.0  # the none column makes the empty row count
    gold = [[0], [1], [0, 1], []]
    probs = [[0.9, 0.2], [0.3, 0.8], [0.7, 0.7], [0.1, 0.1]]
    th = m.tune_thresholds("unfair_tos", gold, [row + [0.0] * (len(m.task_entry("unfair_tos")["codes"]) - 2) for row in probs])
    assert 0 < th["global"] < 1 and len(th["per_label"]) == len(m.task_entry("unfair_tos")["codes"])
    agg = m.aggregate({t: {"micro_f1": 0.5, "macro_f1": 0.25} for t in m.LEXGLUE_TASKS})
    assert agg["micro_f1"]["arithmetic"] == 0.5 and agg["micro_f1"]["n_tasks"] == 7


# ---------------------------------------------------------------------------------------------- mbburabak


def test_mbburabak_questions_and_state():
    from jev_local.bench.benchmax.adapters_b import mbburabak_safety as m

    for bench in m.BENCHMARKS:
        qs = m.build_questions(bench)
        assert qs and list(qs) == sorted(qs)
        assert all(q["type"] in {"noul", "choice", "score"} for q in qs.values())
        assert "boolean" not in {q["type"] for q in qs.values()}
    st = m.build_state("toxicchat", "hello", None)
    assert st["content_type"] == "message" and st["text"] == "hello" and "response" not in st
    assert set(st["policy"]) == {"toxicchat"} and len(st["policy"]["toxicchat"]) > 100
    st = m.build_state("harmbench", "req", "resp")
    assert st["response"] == "resp"
    assert m.band_of(0.5) == "medium" and m.band_of(0.9) == "high" and m.band_of(0.1) == "low"
    assert m.unsafe_majority(3, ["Safe", "Unsafe", "Unsafe"]) is True
    assert m.unsafe_majority(3, ["Safe", "Safe", "Unsafe"]) is False
    stats = m.confusion_stats([(True, True), (True, False), (False, False), (False, True)])
    assert stats["tp"] == 1 and stats["fp"] == 1 and stats["fn"] == 1 and stats["tn"] == 1
    assert set(m.TARGET_OF_TASK.values()) == set(m.TARGETS)


# ---------------------------------------------------------------------------------------------- dmb


def test_dmb_requests_and_thresholds():
    from jev_local.bench.benchmax.adapters_b import dmb_expanded as m

    m.set_instructions_for_tests("TEST INSTRUCTIONS")
    try:
        req = m.build_request(" hello ", ["a", "b"])
        assert req["state"] == "hello" and req["questions"]["decision"]["criteria"] == {"a": "a", "b": "b"}
        assert req["questions"]["decision"]["instructions"] == "TEST INSTRUCTIONS"
    finally:
        m.set_instructions_for_tests(None)
    train = [{"text": f"t{i}-{lab}", "category": lab} for lab in ("x", "y") for i in range(12)]
    test = [{"text": "t0-x", "category": "x"}, {"text": "new", "category": "y"}]
    out = m.banking_items(train, test)
    assert len(out["test"]) == 2 and len(out["validation"]) == 20
    assert all(it["options"] == ["x", "y"] for it in out["validation"])
    assert "t0-x" not in {it["text"] for it in out["validation"]}
    assert out["validation"] == m.banking_items(train, test)["validation"]  # deterministic
    # errors at ranks 0, 50, 100, 150 (2% of the top 100, 1.5% of the top 199) and a dense error tail
    units = [{"valid": True, "correct": (i % 50 != 0) and i < 190, "score": 1 - i / 200, "group": str(i)} for i in range(200)]
    th = m.choose_threshold(units, max_error=0.05, min_accepted=100)
    assert th["threshold"] is not None and 100 <= th["accepted"] < 200 and th["error_rate"] <= 0.05
    ap = m.apply_threshold(units, th["threshold"])
    assert ap["accepted"] == th["accepted"] and ap["correct_all_requested"] == (200 - 4 - 10) / 200
    assert m.choose_threshold(units, max_error=0.0, min_accepted=100)["threshold"] is None
    assert m.apply_threshold(units, None)["accepted"] == 0
    assert m.text_hash("A  b") == m.text_hash("a b")


# ---------------------------------------------------------------------------------------------- rerank


def test_rerank_requests_and_ranking():
    from jev_local.bench.benchmax.adapters_b import rerank_scripts as m

    req = m.denser_request("q", ["p one", "p two"])
    assert req["state"] == {"query": "q"} and list(req["questions"]) == ["p0", "p1"]
    assert req["questions"]["p1"]["instructions"] == {"question": "Is this passage relevant to the query?", "passage": "p two"}
    assert req["questions"]["p1"]["type"] == "noul"
    assert m.denser_passage({"title": "T", "text": "x"}) == "T x" and m.denser_passage({"text": "x"}) == "x"
    req = m.hev_request("q", [{"title": "a", "text": "b"}, {"text": "c"}])
    assert list(req["state"]["documents"]) == ["D00", "D01"] and req["state"]["documents"]["D01"] == {"text": "c"}
    assert "D01" in req["questions"]["D01"]["instructions"]
    req = m.aness_score_batch_request("q", ["d1", "d2"])
    assert list(req["state"]["passages"]) == ["p01", "p02"] and req["questions"]["p01"]["type"] == "score"
    assert req["questions"]["p01"]["criteria"] == m.ANESS_RUBRIC
    legend = {str(i): lvl for i, lvl in enumerate(m.ANESS_RUBRIC)}
    assert m.aness_expected_level({"legend": legend, "probabilities": {"0": 0, "1": 0, "2": 0, "3": 1.0}}) == 1.0
    assert m.aness_expected_level({"score": 2}) == 2.0 and m.aness_expected_level(None) is None
    assert m.order_by_score(["a", "b", "c"], [0.1, 0.9, 0.1]) == ["b", "a", "c"]
    assert m.aness_truncate("x" * 3000) == "x" * 2000
    bm = m.BM25Okapi([m.tokenize(t) for t in ("the cat sat", "a dog ran", "fish swim deep", "birds fly high")])
    s = bm.get_scores(m.tokenize("cat"))
    assert s[0] > 0 and s[1] == 0 and s[2] == 0 and s[3] == 0
    assert m.tokenize("The Cat, sat!") == ["cat", "sat"]  # Lucene stop words dropped, lower-cased alphanumerics


# ---------------------------------------------------------------------------------------------- elcronos


def test_elcronos_requests_and_metrics():
    from jev_local.bench.benchmax.adapters_b import elcronos as m

    req = m.build_request("emotion", "i feel fine")
    assert req["state"] == "i feel fine" and list(req["questions"]) == ["emotion"]
    assert req["questions"]["emotion"]["criteria"] == {lab: "" for lab in m.EMOTION_LABELS}
    req = m.build_request("emotion", "x", "defined")
    assert req["questions"]["emotion"]["criteria"]["joy"] == m.EMOTION_DEFINITIONS["joy"]
    with pytest.raises(ValueError):
        m.build_request("fin_topic", "x", "defined")
    assert list(m.build_request("tweet_topic", "t")["questions"]) == ["topic"]
    assert m.DATASETS["tweet_topic"]["labels"][0] == "arts & culture"
    assert len(m.FIN_TOPIC_LABELS) == 20 and len(m.DAILY_DIALOG_LABELS) == 7
    labs = m.EMOTION_LABELS
    assert m.probs_from_answer({"choice": "joy", "probabilities": {"joy": 0.5, "fear": 0.5}}, labs) == [0, 0.5, 0, 0, 0.5, 0]
    assert m.probs_from_answer({"choice": "joy", "probabilities": {}}, labs) == [0, 1.0, 0, 0, 0, 0]
    assert m.probs_from_answer({"choice": "nope", "probabilities": {}}, labs) is None
    r = m.metrics([0, 1, 1], [[0.9, 0.1, 0], [0.2, 0.8, 0], [0.6, 0.4, 0]], 3)
    assert r["accuracy"] == pytest.approx(2 / 3) and r["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)
    assert r["majority_class_accuracy"] == pytest.approx(2 / 3) and 0 <= r["ece_15"] <= 1


# ---------------------------------------------------------------------------------------------- thisisandreeeee


def test_thisisandreeeee_requests_and_metrics():
    from jev_local.bench.benchmax.adapters_b import thisisandreeeee as m

    labels = m.space2_labels("hwu64")
    assert len(labels) == 64 and len(m.space2_labels("clinc150")) == 150 and len(m.space2_labels("banking77")) == 77
    req = m.intent_request("hwu64", "wake me", labels)
    assert req["state"] == "wake me" and req["questions"]["answer"]["criteria"] == {lab: None for lab in labels}
    assert m.sst2_request("good")["questions"]["answer"] == {"type": "noul", "instructions": m.SST2_INSTRUCTION}
    req = m.stsb_request("a", "b")
    assert req["state"] == "Sentence 1: a\nSentence 2: b" and len(req["questions"]["answer"]["criteria"]) == 6
    recs = [("x", [0.7, 0.3], "x", ["x", "y"]), ("y", [0.6, 0.4], "x", ["x", "y"])]
    r = m.classification_metrics(recs)
    assert r["accuracy"] == 0.5 and r["mean_confidence"] == pytest.approx(0.65) and 0 <= r["expected_calibration_error"] <= 1
    r = m.noul_metrics([(True, 0.9), (False, 0.6)])
    assert r["accuracy"] == 0.5
    d1 = m.row_digest_intents([{"text": "a", "label": "b"}])
    assert d1 == m.row_digest_intents([{"text": "a", "label": "b"}]) and d1 != m.row_digest_intents([{"text": "a", "label": "c"}])


# ---------------------------------------------------------------------------------------------- zhuyansen


def test_zhuyansen_requests_and_sampling():
    from jev_local.bench.benchmax.adapters_b import zhuyansen as m

    assert m.key_of("Sci/Tech") == "sci_tech" and m.key_of("arts & culture") == "arts_culture"
    labels = m.load_labels("agnews")
    assert labels["names"] == ["World", "Sports", "Business", "Sci/Tech"]
    crit = m.criteria(labels["names"], m.label_texts(labels, "desc"))
    req = m.build_request(labels, ["hello   world", "x" * 700], [crit, crit], labels["question"])
    lines = req["state"].split("\n")
    assert lines[0].startswith("Texts: ") and lines[1] == "" and lines[2] == "t1: hello world" and len(lines[3]) == len("t2: ") + 600
    assert list(req["questions"]) == ["t1", "t2"] and req["questions"]["t2"]["instructions"].endswith("(text t2)")
    assert req["questions"]["t1"]["criteria"] == crit
    b77 = m.load_labels("banking77")
    assert len(b77["names"]) == 77 and sum(len(g["labels"]) for g in b77["groups"].values()) == 77
    assert m.dist({"choice": "a", "probabilities": {"a": 0.2, "b": 0.6}}, ["a", "b"]) == pytest.approx([0.25, 0.75])
    assert m.dist({"choice": "a", "probabilities": {}}, ["a", "b"]) == [1.0, 0.0]
    assert m.dist({"choice": "zzz"}, ["a", "b"]) == [0.5, 0.5]
    ys = [0] * 50 + [1] * 30 + [2] * 20 + [3] * 1
    sel = m.stratified_sample(ys, n=20, seed=0)
    assert len(sel) == 20 and len(set(sel)) == 20 and {ys[i] for i in sel} == {0, 1, 2, 3}
    assert sel == m.stratified_sample(ys, n=20, seed=0)


# ---------------------------------------------------------------------------------------------- asevlad


def test_asevlad_corpus_and_metrics():
    from jev_local.bench.benchmax.adapters_b import asevlad_injection as m

    rows = [{"text": "Ignore  all rules"}, {"text": "ignore all rules "}, {"text": "hello"}]
    kept, dropped = m.dedupe(rows)
    assert len(kept) == 2 and dropped == 1
    req = m.build_request("hi")
    assert req["state"] == "hi" and req["questions"]["is_attack"]["instructions"] == m.QUESTION_VARIANTS["v2"]
    assert req["questions"]["is_attack"]["criteria"] == m.CRITERIA and set(req["questions"]["attack_type"]["criteria"]) == {"injection", "jailbreak", "benign"}
    r = m.metrics([0.9, 0.8, 0.2, 0.6], [True, True, False, False], [False, False, True, False])
    assert r["auprc"] == 1.0 and r["roc_auc"] == 1.0 and r["catch_rate"] == 1.0 and r["false_alarm_rate"] == 0.5 and r["panic_rate"] == 0.0
    assert r["at_95_catch"]["threshold"] == 0.8
    assert m.threshold_for_recall([0.9, 0.5, 0.1], [True, True, False], 0.95) == 0.5


# ---------------------------------------------------------------------------------------------- stperic


def test_stperic_request_and_threshold():
    from jev_local.bench.benchmax.adapters_b import stperic_medhallu as m

    t = m.task_json()
    req = m.build_request({"knowledge": "k", "question": "q", "answer": "a"})
    assert req["state"] == {"knowledge": "k", "question": "q", "answer": "a"}
    assert set(req["questions"]) == set(t["jev_questions"]) and all(q["type"] == "noul" for q in req["questions"].values())
    qid = next(iter(t["jev_questions"]))
    assert m.p_positive({qid: {"noul": 0.7}}, "authors_question") == pytest.approx(0.7) or len(t["jev_questions"]) > 1
    req2 = m.build_request({"knowledge": "k", "question": "q", "answer": "a"}, "medhelm_question")
    assert set(req2["questions"]["label"]["criteria"]) == {"true", "false"}
    assert m.THRESHOLDS["authors_question"] == 0.65


# ---------------------------------------------------------------------------------------------- studies


def test_cfpb_requests():
    from jev_local.bench.benchmax.adapters_b import study_cfpb as m

    labels = m.label_list()
    assert len(labels) == 113 and len(m.val_indices()) == 6430
    req = m.build_request("my bank", "bare")
    assert req["state"] == {"complaint": "my bank"} and list(req["questions"]["issue"]["criteria"]) == labels
    assert set(req["questions"]["issue"]["criteria"].values()) == {None}
    instr, crit = m.criteria_variant("v1")
    assert instr != m.INSTRUCTIONS_BARE and sum(v is not None for v in crit.values()) == 34
    assert all(isinstance(v, dict) and "what" in v and set(v) <= {"what", "not_for"} for v in crit.values() if v is not None)
    assert sum("not_for" in v for v in crit.values() if v is not None) == 6
    instr2, crit2 = m.criteria_variant("v1instr")
    assert instr2 == instr and set(crit2.values()) == {None}


def test_trec_requests():
    from jev_local.bench.benchmax.adapters_b import study_trec50 as m

    fine = ["ENTY:cremat", "ABBR:abb", "NUM:date", "HUM:ind"]
    ids = m.identifiers(fine)
    assert ids == {"ABBR:abb": "L000", "ENTY:cremat": "L001", "HUM:ind": "L002", "NUM:date": "L003"}
    assert m.fine_description("ENTY:cremat") == "Entity: cremat"
    assert m.permutation_seed("trec:test:0") == m.permutation_seed("trec:test:0") != m.permutation_seed("trec:test:1")
    req = m.build_request("What is X ?", "trec:test:0", fine, {f: m.fine_description(f) for f in fine}, coarse=False)
    assert req["state"] == "What is X ?" and set(req["questions"]["classification"]["criteria"]) == set(ids.values())
    assert req["questions"]["classification"]["criteria"]["L001"] == "Entity: cremat"
    assert req == m.build_request("What is X ?", "trec:test:0", fine, {f: m.fine_description(f) for f in fine}, coarse=False)


def test_nslkdd_request_and_score(tmp_path):
    from jev_local.bench.benchmax.adapters_b import study_nslkdd as m
    from jev_local.bench.benchmax.adapters_b.common import Item

    cfg = m.card()
    assert len(cfg["features"]) == 41 and cfg["benign"] == "normal"
    row = {n: str(i) for i, n in enumerate(cfg["features"])}
    row.update({"row_id": "7", "category": "dos", "novel_attack": "1"})
    flow = m.flow_from_row(row, cfg)
    assert flow["is_attack"] and flow["novel_attack"] and flow["attributes_csv"].startswith("0,1,2,")
    req = m.request_body(flow)
    assert req["state"]["flows"] == {"under_test": flow["attributes_csv"]} and "examples" not in req["state"]
    assert req["questions"]["category"]["criteria"] == req["state"]["categories"]
    assert set(req["questions"]) == {"is_attack", "category"} and "model" not in req
    req2 = m.request_body(flow, [flow])
    assert req2["state"]["examples"] == [{"record": flow["attributes_csv"], "category": "dos"}]
    ad = m.Adapter(None, _work(tmp_path))
    items = [Item("k0/1", "k0", req, {"is_attack": True, "category": "dos"}, meta={"novel_attack": True, "row_id": 1}),
             Item("k0/2", "k0", req, {"is_attack": False, "category": "normal"}, meta={"novel_attack": False, "row_id": 2}),
             Item("k0/3", "k0", req, {"is_attack": True, "category": "probe"}, meta={"novel_attack": False, "row_id": 3})]
    ans = _answers({"k0/1": {"is_attack": {"noul": 0.9}, "category": {"choice": "dos", "probabilities": {"dos": 1}}},
                    "k0/2": {"is_attack": {"noul": 0.2}, "category": {"choice": "normal", "probabilities": {"normal": 1}}}})
    r = ad.score(items, ans, "validation")["tasks"]["k0"]
    assert r["recall"] == 0.5 and r["precision"] == 1.0 and r["f1"] == pytest.approx(2 / 3) and r["error_rate"] == pytest.approx(1 / 3)
    assert r["recall_novel"] == 1.0 and r["recall_known"] == 0.0 and r["category_accuracy"] == pytest.approx(2 / 3)


def test_cesnet_selection_and_request():
    import datetime as dt

    from jev_local.bench.benchmax.adapters_b import study_cesnet as m

    assert m.week_dates(1)[0] == dt.date(2024, 6, 1) and len(m.week_dates(1)) == 7
    assert m.week_dates(5)[0] == dt.date(2024, 6, 29) and m.week_dates(30)[-1] == dt.date(2024, 12, 27)
    assert dt.date(2024, 10, 31) not in m.week_dates(22) and len(m.week_dates(22)) == 6
    assert sum(len(m.week_dates(w)) for w in m.TRAIN_WEEKS + m.TEST_WEEKS) == 205
    ipt, dr, sz = [0.0, 3.0] + [1.5] * 8, [1, -1] * 5, [1250] * 10
    pk = m.packets_json(ipt, dr, sz)
    assert pk[0] == {"ipt_ms": 0, "direction": 1, "size": 1250} and pk[2]["ipt_ms"] == 1.5 and len(pk) == 10
    req = m.build_request(pk)
    assert set(req["state"]) == {"instructions", "fields", "packets"} and list(req["questions"]["application"]["criteria"]) == list(m.CLASSES)
    req = m.build_request(pk, [{"packets": pk, "label": "youtube"}])
    assert req["state"]["examples"] == [{"packets": pk, "application": "youtube"}]
    recs = [{"flow_id": str(i), "priority": i, "fingerprint": f"fp{i % 5}", "label": "youtube"} for i in range(50)]
    seen_ids, seen_fps = {"0"}, set()
    out = m.select_week(recs, seen_ids, seen_fps, candidates=20, keep=4)
    assert [r["flow_id"] for r in out] == ["1", "2", "3", "4"] and "0" not in {r["flow_id"] for r in out}
    with pytest.raises(RuntimeError):
        m.select_week(recs, set(), set(), candidates=3, keep=4)
    train = [{"flow_id": f"{w}{c}{k}", "priority": k, "label": c, "week": w} for w in m.TRAIN_WEEKS for c in m.CLASSES for k in (2, 1)]
    ex = m.pick_examples(train)
    assert len(ex) == 40 and all(e["priority"] == 1 for e in ex) and ex[0]["week"] == 1 and ex[-1]["label"] == m.CLASSES[-1]


def test_koa_request():
    from jev_local.bench.benchmax.adapters_b import study_amazon_polarity as m

    req = m.build_request("great product")
    assert req["state"] == "great product" and req["questions"]["sentiment"]["criteria"] == {"Negative": None, "Positive": None}
