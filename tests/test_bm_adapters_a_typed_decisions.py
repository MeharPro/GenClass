"""typed-decisions scorer: definitions, prior/uniform rows, pinning (pure Python, synthetic cases)."""

from __future__ import annotations

import json
import math

import pytest

from jev_local.bench.benchmax.adapters_a import typed_decisions as T

Q = {
    "verdict": {"type": "noul", "instructions": "Should this be escalated?", "criteria": {"true": "yes", "false": "no"}},
    "category": {"type": "choice", "instructions": "Which category?", "criteria": {"billing": "money", "tech": "bugs", "other": "else"}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]},
}


def case(cid, wf, gold):
    return T.Case(cid, wf, {"text": f"case {cid}"}, Q, gold)


def gold(noul_p, noul_lab, choice_probs, choice_lab, score_probs, score_lab):
    return {
        "verdict": {"type": "noul", "noul": noul_p, "label": noul_lab},
        "category": {"type": "choice", "label": choice_lab, "probabilities": choice_probs},
        "urgency": {"type": "score", "label": score_lab, "score": sum(i * p for i, p in enumerate(score_probs)), "probabilities": {str(i): p for i, p in enumerate(score_probs)}},
    }


CASES = [
    case("1", "cs", gold(0.9, "true", {"billing": 0.7, "tech": 0.2, "other": 0.1}, "billing", [0.1, 0.3, 0.6], 2)),
    case("2", "cs", gold(0.2, "false", {"billing": 0.1, "tech": 0.8, "other": 0.1}, "tech", [0.6, 0.3, 0.1], 0)),
    case("3", "cs", gold(0.5, "true", {"billing": 0.4, "tech": 0.4, "other": 0.2}, "billing", [0.2, 0.5, 0.3], 1)),
    case("4", "cs", gold(0.1, "false", {"billing": 0.2, "tech": 0.3, "other": 0.5}, "other", [0.3, 0.3, 0.4], 2)),
]


def test_parse_row_decodes_json_strings():
    row = {"id": "x", "workflow": "cs", "state": json.dumps({"a": 1}), "questions": json.dumps(Q), "gold": json.dumps(CASES[0].gold)}
    c = T.parse_row(row)
    assert c.state == {"a": 1} and c.questions == Q and c.gold["verdict"]["noul"] == 0.9


def test_gold_and_model_dists():
    g, idx = T.gold_dist(Q["verdict"], CASES[0].gold["verdict"])
    assert g == [pytest.approx(0.1), 0.9] and idx == 1
    g, idx = T.gold_dist(Q["category"], CASES[0].gold["category"])
    assert g == [0.7, 0.2, 0.1] and idx == 0
    g, idx = T.gold_dist(Q["urgency"], CASES[0].gold["urgency"])
    assert g == [0.1, 0.3, 0.6] and idx == 2
    m = T.model_dist(Q["category"], {"type": "choice", "choice": "tech", "probabilities": {"billing": 0.25, "tech": 0.5, "other": 0.25}})
    assert m == [0.25, 0.5, 0.25]
    assert T.model_dist(Q["verdict"], {"type": "noul", "noul": 0.3}) == [pytest.approx(0.7), 0.3]
    assert T.model_dist(Q["urgency"], None) is None
    assert T.uniform_dist(Q["urgency"]) == [1 / 3] * 3


def test_predict_tie_and_noul_rules():
    cfg = T.ScorerConfig()
    assert T.predict("choice", [0.4, 0.4, 0.2], cfg) == 0
    assert T.predict("choice", [0.4, 0.4, 0.2], T.ScorerConfig(tie="last")) == 1
    assert T.predict("noul", [0.5, 0.5], cfg) == 1 and T.predict("noul", [0.5, 0.5], T.ScorerConfig(noul_rule="gt")) == 0


def test_kl_brier_ece_definitions():
    assert T.kl_div([1.0, 0.0], [1.0, 0.0], 1e-6) == 0.0
    assert T.kl_div([0.5, 0.5], [0.25, 0.75], 1e-6) == pytest.approx(0.5 * math.log(2) + 0.5 * math.log(0.5 / 0.75))
    assert T.kl_div([1.0, 0.0], [0.0, 1.0], 1e-3) == pytest.approx(math.log(1000))
    cfg = T.ScorerConfig()
    assert T.brier_score("choice", [1, 0, 0], [0.5, 0.5, 0], cfg) == pytest.approx(0.5)
    assert T.brier_score("choice", [1, 0, 0], [0.5, 0.5, 0], T.ScorerConfig(brier="mean")) == pytest.approx(0.5 / 3)
    assert T.brier_score("noul", [0.2, 0.8], [0.5, 0.5], cfg) == pytest.approx(0.18)  # two-class
    assert T.brier_score("noul", [0.2, 0.8], [0.5, 0.5], T.ScorerConfig(noul_brier="binary")) == pytest.approx(0.09)
    # ECE: two bins, perfectly calibrated halves -> 0; all confident and wrong -> 1
    assert T.ece_score([0.9, 0.9], [1.0, 1.0], 10, "floor") == pytest.approx(0.1)
    assert T.ece_score([1.0, 1.0], [0.0, 0.0], 10, "floor") == pytest.approx(1.0)
    assert T.ece_score([1.0, 1.0], [0.0, 0.0], 10, "right_closed") == pytest.approx(1.0)
    assert math.isnan(T.ece_score([], [], 10, "floor"))


def test_uniform_row_accuracy_is_first_option_share_and_missing_is_wrong():
    cfg = T.ScorerConfig()
    uni = [T.uniform_answers(c) for c in CASES]
    r = T.score_cases(CASES, uni, cfg)
    # uniform: noul p=0.5 -> true (ge) matches labels true,false,true,false -> 2/4; choice first option billing -> 2/4; score first level -> 1/4
    assert r["accuracy"] == pytest.approx((2 + 2 + 1) / 12)
    assert r["n_decisions"] == 12 and r["n_cases"] == 4 and r["n_missing_answers"] == 0
    assert set(r["by_type"]) == {"noul", "choice", "score"} and r["by_workflow"] == {"cs": pytest.approx(5 / 12)}
    r2 = T.score_cases(CASES, [None] * 4, cfg)
    assert r2["accuracy"] == 0.0 and r2["n_missing_answers"] == 12 and r2["kl"] == pytest.approx(r["kl"])
    # case averaging with equal-sized cases equals decision averaging
    r3 = T.score_cases(CASES, uni, T.ScorerConfig(average="case"))
    assert r3["accuracy"] == pytest.approx(r["accuracy"]) and r3["kl"] == pytest.approx(r["kl"])


def test_prior_table_hard_soft_and_pooling():
    wf = T.ScorerConfig(prior_source="hard", prior_key="wf_qid")
    hard = T.prior_table(CASES, wf)
    assert hard[("cs", "category")] == {"billing": 0.5, "tech": 0.25, "other": 0.25}
    assert hard[("cs", "verdict")] == {"true": 0.5, "false": 0.5}
    soft = T.prior_table(CASES, T.ScorerConfig(prior_source="soft", prior_key="wf_qid"))
    assert soft[("cs", "category")]["billing"] == pytest.approx((0.7 + 0.1 + 0.4 + 0.2) / 4)
    ans = T.prior_answers(CASES[0], hard, wf)
    assert ans["category"]["choice"] == "billing" and ans["verdict"]["noul"] == 0.5 and ans["urgency"]["probabilities"]["2"] == 0.5
    # pooled by question id across workflows: a second workflow with the same qid shares the prior
    other = T.Case("9", "sec", {"text": "x"}, Q, CASES[1].gold)
    pooled = T.prior_table(CASES + [other], T.ScorerConfig(prior_source="hard", prior_key="qid"))
    assert set(pooled) == {("*", "verdict"), ("*", "category"), ("*", "urgency")}
    assert pooled[("*", "category")] == {"billing": 0.4, "tech": 0.4, "other": 0.2}
    per_wf = T.prior_table(CASES + [other], wf)
    assert per_wf[("sec", "category")] == {"tech": 1.0}


def test_confidence_modes():
    assert T.confidence_of("choice", [0.5, 0.25, 0.25], "max") == 0.5
    assert T.confidence_of("choice", [0.5, 0.25, 0.25], "wire") == pytest.approx((3 * 0.5 - 1) / 2)
    assert T.confidence_of("noul", [0.3, 0.7], "wire") == pytest.approx(0.7) and T.confidence_of("noul", [0.3, 0.7], "max") == pytest.approx(0.7)
    assert T.confidence_of("score", [1.0, 0.0, 0.0], "wire") == pytest.approx(1.0)
    assert T.confidence_of("choice", [0.25] * 4, "wire") == pytest.approx(0.0)


def test_pin_scorer_finds_the_generating_variant():
    """Compute reference rows with one known config; the enumeration must recover a matching config."""
    truth = T.ScorerConfig(tie="first", noul_rule="gt", kl_eps=1e-6, brier="mean", noul_brier="binary", ece_bins=15, ece_mode="right_closed",
                           ece_conf="wire", average="decision", prior_source="soft", prior_key="qid")
    table = T.prior_table(CASES, truth)
    ref = {
        "prior": {k: round(v, 3) for k, v in T.score_cases(CASES, [T.prior_answers(c, table, truth) for c in CASES], truth).items() if k in T.METRICS},
        "uniform": {k: round(v, 3) for k, v in T.score_cases(CASES, [T.uniform_answers(c) for c in CASES], truth).items() if k in T.METRICS},
    }
    rep = T.pin_scorer(CASES, CASES, reference=ref)
    assert rep["pinned"] and rep["comparable"] and rep["n_matching"] >= 1 and rep["pinned_metrics"] == list(T.METRICS) and rep["unpinned_metrics"] == []
    cfgs = [T.ScorerConfig.from_json(c) for c in rep["all_matching"]]
    assert truth in cfgs and len(cfgs) == rep["n_matching"]
    # every matching variant reproduces both rows; fields that stay free are those the two rows cannot identify
    # (with brier="mean" the two noul Brier conventions coincide: mean over 2 classes of 2(p-g)^2 == (p-g)^2;
    # one workflow only, so qid and wf_qid pooling coincide)
    for c in cfgs:
        assert c.brier == "mean" and c.prior_source == "soft"
    assert set(rep["free_fields"]) <= {"tie", "noul_rule", "kl_eps", "ece_bins", "ece_mode", "ece_conf", "average", "noul_brier", "prior_key"}
    assert "brier" not in rep["free_fields"] and "prior_source" not in rep["free_fields"]
    assert rep["closest"]["l1_distance"] <= 8 * 0.0005 + 1e-9  # the reference rows are rounded to 3 dp
    # accuracy / ECE unreproducible (e.g. random tie-breaking upstream) but KL / Brier exact: comparable, not pinned
    part = {"prior": dict(ref["prior"], accuracy=0.999, ece=0.999), "uniform": dict(ref["uniform"], accuracy=0.999, ece=0.999)}
    rep3 = T.pin_scorer(CASES, CASES, reference=part)
    assert not rep3["pinned"] and rep3["comparable"] and rep3["pinned_metrics"] == ["kl", "brier"] and rep3["unpinned_metrics"] == ["accuracy", "ece"]
    assert rep3["config"] is not None and T.ScorerConfig.from_json(rep3["config"]).brier == "mean"
    # a reference nothing reproduces is reported as unpinned and not comparable, with the closest candidate
    bad = {"prior": {"accuracy": 0.123, "kl": 9.0, "brier": 9.0, "ece": 0.5}, "uniform": {"accuracy": 0.9, "kl": 9.0, "brier": 9.0, "ece": 0.5}}
    rep2 = T.pin_scorer(CASES, CASES, reference=bad, variants={"tie": ("first",), "kl_eps": (1e-6,)})
    assert not rep2["pinned"] and not rep2["comparable"] and rep2["n_matching"] == 0 and rep2["closest"]["config"] is not None


def test_scorer_config_json_round_trip_and_reference_constants():
    cfg = T.ScorerConfig(tie="last", ece_bins=15, prior_key="wf_qid")
    assert T.ScorerConfig.from_json(json.loads(json.dumps(cfg.to_json()))) == cfg
    assert T.ScorerConfig.from_json({"tie": "last"}).prior_key == "qid"  # older pinned files without the field
    assert T.COMPARABLE_METRICS == ("kl", "brier") and set(T.VARIANTS) == set(T.ScorerConfig.__dataclass_fields__)
    assert T.REFERENCE_ROWS["prior"] == {"accuracy": 0.470, "kl": 0.347, "brier": 0.189, "ece": 0.088}
    assert T.REFERENCE_ROWS["uniform"] == {"accuracy": 0.308, "kl": 0.444, "brier": 0.238, "ece": 0.169}
    assert T.JEV_CARD["accuracy"] == 0.727 and T.PINNED_REVISION.startswith("d0e2f0c4")
