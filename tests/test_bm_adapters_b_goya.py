"""Pure-Python tests for the goya_rm adapter port (protocols, parse_prediction, official aggregations)."""

from __future__ import annotations

import hashlib

import pytest

from jev_local.bench.benchmax.adapters_b import goya_rm as g


def test_swap_and_permutation_match_upstream_definitions():
    assert g.swap("x") == (hashlib.sha256(b"x").digest()[0] & 1 == 1)
    p = g.permutation("rm_bench:1", 6)
    assert sorted(p) == list(range(6)) and p == g.permutation("rm_bench:1", 6)
    ex = g.pair(benchmark="rewardbench1", example_id="0:1", subset="s", prompt="p", preferred="good", rejected="bad")
    assert ex.candidates[ex.correct[0]] == "good" and ex.mode == "pairwise"


def test_build_request_all_modes():
    ex = g.pair(benchmark="rubric_bench", example_id="c1", subset="math", prompt="p", preferred="A1", rejected="B1", rubrics=["r1", "r2"])
    req = g.build_request(ex)
    q = req["questions"]["preference"]
    assert req["state"]["responses"] == {"A": ex.candidates[0], "B": ex.candidates[1]}
    assert q["instructions"]["required_rubrics"] == ["r1", "r2"] and q["instructions"]["decision"] == g.PAIRWISE_INSTRUCTIONS
    assert q["criteria"] == {"A": "The candidate at `responses.A` is better.", "B": "The candidate at `responses.B` is better."}
    tie = g.Example("ppe", "q:0", "English", "p", ["x", "y"], [0, 1], "pairwise_tie")
    assert "TIE" in g.build_request(tie)["questions"]["preference"]["criteria"]
    lw = g.Example("rewardbench2", "0:1", "Focus", "p", ["a", "b", "c"], [1], "listwise")
    assert list(g.build_request(lw)["questions"]["best_response"]["criteria"]) == ["R0", "R1", "R2"]
    rt = g.Example("rewardbench2", "0:2", "Ties", "p", ["a", "b"], [0], "ratings")
    qs = g.build_request(rt)["questions"]
    assert set(qs) == {"quality_0", "quality_1"} and qs["quality_0"]["criteria"] == g.QUALITY_LEVELS and qs["quality_0"]["type"] == "score"
    grid = g.Example("rm_bench", "1", "chat", "p", [f"s{i}" for i in range(6)], [0, 1, 2], "rm_style_grid",
                     metadata={"grid_pairs": [[c, 3 + r] for c in range(3) for r in range(3)]})
    qs = g.build_request(grid)["questions"]
    assert len(qs) == 9 and set(qs["pair_4"]["criteria"]) == {"S1", "S4"}
    pb = g.Example("processbench", "1", "gsm8k", "prob", ["s0", "s1"], [-1], "process_first_error")
    crit = g.build_request(pb)["questions"]["first_error"]["criteria"]
    assert list(crit) == ["CLEAN", "STEP_0", "STEP_1"]
    prm = g.Example("prmbench", "0:1", "redundency", "prob", ["s0", "s1"], [1], "prm_steps")
    qs = g.build_request(prm)["questions"]
    assert set(qs) == {"validity_0", "redundancy_0", "validity_1", "redundancy_1"}
    oq = g.Example("rm_bench_pointwise", "1:chosen:0", "chat", "p", ["resp"], [], "ordinal_quality")
    req = g.build_request(oq)
    assert req["state"] == {"user_prompt": "p", "response": "resp"} and list(req["questions"]["quality"]["criteria"]) == ["Q0", "Q1", "Q2", "Q3", "Q4"]
    with pytest.raises(ValueError):
        g.build_request(g.Example("x", "1", "s", "p", ["a"], [], "binary_correctness"))


def test_parse_prediction_and_metrics():
    pred = g.parse_prediction("pairwise", 2, {}, {"preference": {"choice": "B", "probabilities": {"A": 0.3, "B": 0.7}, "confidence": 0.7}})
    assert pred == {"predicted": 1, "probabilities": {"0": 0.3, "1": 0.7}, "confidence": 0.7}
    assert g.is_correct("pairwise", [1], pred, {}) and not g.is_correct("pairwise", [0], pred, {})
    pred = g.parse_prediction("process_first_error", 2, {}, {"first_error": {"choice": "STEP_1", "probabilities": {"CLEAN": 0.1, "STEP_0": 0.2, "STEP_1": 0.7}, "confidence": 0.7}})
    assert pred["predicted"] == 1 and pred["probabilities"] == {"-1": 0.1, "0": 0.2, "1": 0.7}
    pred = g.parse_prediction("ordinal_quality", 1, {}, {"quality": {"choice": "Q3", "probabilities": {"Q0": 0, "Q1": 0, "Q2": 0.5, "Q3": 0.5, "Q4": 0}, "confidence": 0.5}})
    assert pred["reward"] == 2.5 and pred["predicted"] == 3
    with pytest.raises(KeyError):
        g.parse_prediction("pairwise", 2, {}, {"preference": {"choice": "C", "probabilities": {}, "confidence": 1}})

    def row(subset, correct, mode="pairwise", pred=None, metadata=None, correct_indices=(0,)):
        return {"subset": subset, "correct": correct, "mode": mode, "prediction": pred or {"predicted": 0, "probabilities": {"0": 0.8, "1": 0.2}, "confidence": 0.8},
                "metadata": metadata or {}, "correct_indices": list(correct_indices)}

    rows = [row("alpacaeval-easy", True), row("alpacaeval-easy", False), row("math-prm", True), row("hep-cpp", True), row("donotanswer", False),
            row("mt-bench-hard", True)]
    m = g.rewardbench1_metrics(rows)
    assert m["sections"]["Chat"] == 0.5 and m["sections"]["Safety"] == 0.0 and m["sections"]["Reasoning"] == 1.0 and m["sections"]["Chat Hard"] == 1.0
    assert m["official_macro"] == pytest.approx((0.5 + 1.0 + 0.0 + 1.0) / 4) and 0 <= m["calibration"]["ece_confidence"] <= 1
    grid_rows = [row("chat", True, "rm_style_grid", {"grid_correct": [True] * 9, "confidence": 1}),
                 row("safety-response", True, "rm_style_grid", {"grid_correct": [True, False, False, True, True, False, True, True, True], "confidence": 1})]
    m = g.rm_bench_metrics(grid_rows)
    assert m["domains"]["chat"]["all"] == 1.0 and m["domains"]["safety"]["all"] == pytest.approx(6 / 9)
    # rows = chosen style i, cols = rejected style j: strictly-upper = hard, diagonal = normal, strictly-lower = easy
    assert m["domains"]["safety"]["hard"] == 0.0 and m["domains"]["safety"]["normal"] == 1.0 and m["domains"]["safety"]["easy"] == 1.0
    assert m["official_domain_macro"] == pytest.approx((1.0 + 6 / 9) / 2)
    pb = [row("gsm8k", True, "process_first_error", {"predicted": -1}, {"label": -1}), row("gsm8k", False, "process_first_error", {"predicted": 2}, {"label": 1}),
          row("math", True, "process_first_error", {"predicted": 1}, {"label": 1})]
    m = g.processbench_metrics(pb)
    assert m["subsets"]["gsm8k"]["f1_harmonic"] == 0.0 and m["subsets"]["math"]["clean_accuracy"] is None and m["official_macro"] == 0.0
    prm = [row("redundency", False, "prm_steps", {"validity_labels": [True, True], "redundancy_labels": [False, True]}, {"classification": "redundency", "error_steps": [1]}),
           row("confidence", False, "prm_steps", {"validity_labels": [True, False, True], "redundancy_labels": [False] * 3}, {"classification": "confidence", "error_steps": [1]})]
    m = g.prmbench_metrics(prm)
    assert m["confusion"] == {"TP": 3, "FP": 0, "TN": 2, "FN": 0} and m["prm_score"] == 1.0 and m["exact_trace_accuracy"] == 1.0
    pw = []
    for pid, dom in (("1", "chat"), ("2", "safety-response")):
        for style in range(3):
            pw.append(row(dom, False, "ordinal_quality", {"reward": 3 - 0.1 * style}, {"prompt_id": pid, "preferred": True, "style": style, "domain": dom}))
            pw.append(row(dom, False, "ordinal_quality", {"reward": 2 + 0.5 * style}, {"prompt_id": pid, "preferred": False, "style": style, "domain": dom}))
    m = g.rm_bench_pointwise_metrics(pw)
    assert m["complete_prompts"] == 2 and m["headline_domains"]["safety"]["all"] == pytest.approx(m["headline_domains"]["chat"]["all"])
    assert m["official_domain_macro"] == pytest.approx(sum(1 for i in range(3) for j in range(3) if 3 - 0.1 * i > 2 + 0.5 * j) / 9)
    ppe = [row("English", True, "pairwise_tie", {"predicted": 0, "probabilities": {"0": 0.6, "1": 0.3, "-1": 0.1}, "confidence": 0.6},
               {"winner": "model_a", "hard_prompt": True, "easy_prompt": False, "if_prompt": False, "math_prompt": False, "is_code": False, "language": "English"}),
           row("English", False, "pairwise_tie", {"predicted": 0, "probabilities": {}, "confidence": 0.5},
               {"winner": "tie", "hard_prompt": False, "easy_prompt": False, "if_prompt": False, "math_prompt": False, "is_code": False, "language": "English"}, correct_indices=(0, 1))]
    m = g.ppe_metrics(ppe)
    assert m["accuracy"] == 1.0 and m["tie_rows"] == 1 and m["categories"]["hard_prompt"]["n"] == 1
    rb2 = [row("Focus", True, "listwise"), row("Ties", False, "ratings", {"predicted": 0, "scores": [3.0, 2.0, 1.0]}, {"original_id": "ref:1"}, correct_indices=(0,)),
           row("Ties", False, "ratings", {"predicted": 0, "scores": [3.0, 3.0, 1.0]}, {"original_id": "tied:1"}, correct_indices=(0, 1))]
    m = g.rewardbench2_metrics(rb2)
    assert m["subsets"]["Focus"] == 1.0 and m["subsets"]["Ties"] is not None and m["official_macro"] == pytest.approx((1.0 + m["subsets"]["Ties"]) / 2)
