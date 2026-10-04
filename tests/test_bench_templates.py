"""jevbench templates: mappers produce valid string-only requests; variants behave (pure Python)."""

import json
import random

from jev_local.bench import templates as T
from jev_local.bench.build import proportional_quotas


def _ok(item):
    assert T.validate_request({"state": item.state, "questions": item.questions}) == []
    assert set(item.gold) <= set(item.questions)
    return item


def test_humanize():
    assert T.humanize("card_arrival") == "card arrival"
    assert T.humanize("AddToPlaylist") == "add to playlist"
    assert T.humanize("arts_&_culture") == "arts & culture"
    assert T.humanize("IPO") == "IPO"


def test_choice_mappers_and_btzsc_fallback():
    ctx = T.Ctx(names=["card_arrival", "age_limit"], btzsc={"card arrival": "This banking customer example message is about card arrival."})
    it = _ok(T.m_banking77({"text": "where is my card", "label_text": "card_arrival"}, "test:0", ctx))
    crit = it.questions["answer"]["criteria"]
    assert list(crit) == ["card_arrival", "age_limit"] and crit["age_limit"].endswith("about age limit.")
    assert ctx.btzsc_missing == ["age_limit"]
    assert it.gold["answer"] == {"type": "choice", "label": "card_arrival"}


def test_noul_criteria_both_or_neither():
    try:
        T.noul("x?", true="yes only")
    except ValueError:
        pass
    else:
        raise AssertionError("one-sided noul criteria must be rejected")
    bad = {"state": "s", "questions": {"q": {"type": "noul", "instructions": "x?", "criteria": {"true": "t"}}}}
    assert T.validate_request(bad)


def test_sst2_has_choice_variant():
    it = _ok(T.m_sst2({"sentence": "great film", "label": 1}, "validation:0", T.Ctx(names=["negative", "positive"])))
    assert it.gold["answer"] == {"type": "noul", "label": True}
    qs, gold = it.extra["choice"]
    assert gold["answer"]["label"] == "positive" and qs["answer"]["type"] == "choice"


def test_multi_question_mappers_are_string_only():
    it = _ok(T.m_openai_moderation({"prompt": "hi", "S": 0, "H": 1, "V": 0, "HR": 0, "SH": 0, "S3": 0, "H2": 0, "V2": 0}, "train:0", T.Ctx()))
    assert len(it.questions) == 8 and it.gold["H"]["label"] is True and it.stratum == "pos"
    it = _ok(T.m_unfair_tos({"text": "we may terminate", "labels": [1]}, "test:0", T.Ctx()))
    assert it.gold["unilateral_termination"]["label"] is True and len(it.questions) == 8
    it = _ok(T.m_go_emotions({"text": "yay", "labels": [17], "id": "x"}, "test:0", T.Ctx()))
    assert len(it.questions) == 28 and it.gold["joy"]["label"] is True and "criteria" not in it.questions["joy"]
    it = _ok(T.m_helpsteer2({"prompt": "p", "response": "r", "helpfulness": 3, "correctness": 2, "coherence": 4,
                             "complexity": 1, "verbosity": 0}, "validation:0", T.Ctx()))
    assert it.gold["helpfulness"] == {"type": "score", "label": 3, "value": 3.0}


def test_typed_decisions_native():
    row = {"id": "w_1", "workflow": "w", "state": json.dumps({"a": 1}),
           "questions": json.dumps({"c": {"type": "choice", "instructions": "i", "criteria": {"x": "dx", "y": "dy"}},
                                    "n": {"type": "noul", "instructions": "n?"},
                                    "s": {"type": "score", "instructions": "s?", "criteria": ["lo", "mid", "hi"]}}),
           "gold": json.dumps({"c": {"type": "choice", "label": "x", "probabilities": {"x": 0.7, "y": 0.3}},
                               "n": {"type": "noul", "label": "true", "noul": 0.8},
                               "s": {"type": "score", "label": "1", "score": 1.1, "probabilities": {"0": 0.1, "1": 0.7, "2": 0.2}}})}
    it = _ok(T.m_typed_decisions(row, "test:0", T.Ctx()))
    assert it.state == {"a": 1}
    assert it.gold["s"]["dist"] == [0.1, 0.7, 0.2] and it.gold["n"]["p"] == 0.8


def test_stsb_and_scores():
    it = _ok(T.m_stsb({"sentence1": "a", "sentence2": "b", "score": 0.76}, "test:0", T.Ctx()))
    assert it.gold["answer"]["label"] == 4 and abs(it.gold["answer"]["value"] - 3.8) < 1e-9


def test_variants():
    qs = {"q": T.choice("i", {"a": "da", "b": "db", "c": "dc"}), "n": T.noul("n?")}
    b = T.bare(qs)
    assert b["q"]["criteria"] == {"a": None, "b": None, "c": None} and b["n"] == qs["n"]
    s = T.shuffled(qs, random.Random(0))
    assert list(s["q"]["criteria"]) != list(qs["q"]["criteria"])
    assert s["q"]["criteria"]["a"] == "da" and set(s["q"]["criteria"]) == set(qs["q"]["criteria"])
    two = T.shuffled({"q": T.choice("i", {"a": None, "b": None})}, random.Random(1))
    assert list(two["q"]["criteria"]) == ["b", "a"]


def test_proportional_quotas():
    q = proportional_quotas({"a": 500, "b": 300, "c": 200}, 100)
    assert q == {"a": 50, "b": 30, "c": 20}
    q = proportional_quotas({"a": 1, "b": 1, "c": 1}, 2)
    assert sum(q.values()) == 2 and q == {"a": 1, "b": 1, "c": 0}
