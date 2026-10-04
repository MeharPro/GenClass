"""Answer math (CONTRACT A / SPEC §1.4). Documented values are reproduced within ±0.01.

The docs display probabilities already rounded to 2 dp, and their confidences were computed from
the unrounded ones, so a ±0.01 tolerance is the tightest that any formula can meet on these rows
(e.g. 0.88/0.12/0 gives exactly 0.82, the docs show 0.81, consistent with a true pmax ~0.875).
"""

from __future__ import annotations

import math

import pytest

from jev_local.confidence import (
    build_answer,
    choice_confidence,
    mad_uniform,
    normalize,
    score_center,
    score_confidence,
    score_value,
)
from jev_local.engine.base import RawDist
from jev_local.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion

TOL = 0.01 + 1e-9


@pytest.mark.parametrize(
    "p, expected",
    [
        ((0.88, 0.12, 0.0), 0.81),
        ((0.61, 0.35, 0.04), 0.42),
        ((0.74, 0.065, 0.065, 0.065, 0.065), 0.67),  # pmax 0.74, K=5
        ((0.40, 0.34, 0.24, 0.02), 0.20),
        ((0.01, 0.99), 0.97),
    ],
)
def test_choice_confidence_documented(p, expected):
    assert abs(choice_confidence(p) - expected) <= TOL


def test_choice_confidence_edges():
    assert choice_confidence([1.0]) == 1.0  # K=1
    assert choice_confidence([0.5, 0.5]) == 0.0
    assert choice_confidence([1.0, 0.0, 0.0]) == 1.0
    assert choice_confidence([0.25] * 4) == pytest.approx(0.0, abs=1e-12)


# (p, score, confidence). Each row records which centre reproduces it: on all four documented
# rows mode and median pick the same level, so both reproduce every case.
SCORE_CASES = [
    ((0.0, 0.57, 0.43), 1.43, 0.35, {"mode": 1, "median": 1}),
    ((0.0, 0.95, 0.05), 1.05, 0.92, {"mode": 1, "median": 1}),
    ((0.0, 0.16, 0.84), 1.84, 0.77, {"mode": 2, "median": 2}),
    ((0.37, 0.03, 0.25, 0.35), None, 0.0, {"mode": 0, "median": 2}),
]


@pytest.mark.parametrize("p, score, conf, centers", SCORE_CASES)
@pytest.mark.parametrize("center", ["mode", "median"])
def test_score_documented(p, score, conf, centers, center):
    if score is not None:
        assert abs(score_value(p) - score) <= 1e-9
    assert score_center(p, center) == centers[center]
    assert abs(score_confidence(p, center) - conf) <= TOL


def test_mode_and_median_differ_off_the_documented_rows():
    # Skewed mass: the mode is level 1 but half the mass lies at or above level 2.
    p = (0.0, 0.4, 0.35, 0.25)
    assert score_center(p, "mode") == 1 and score_center(p, "median") == 2
    assert score_confidence(p, "mode") == pytest.approx(0.15)
    assert score_confidence(p, "median") == pytest.approx(0.35)


def test_mad_uniform():
    assert mad_uniform(2) == pytest.approx(0.5)
    assert mad_uniform(3) == pytest.approx(2 / 3)
    assert mad_uniform(4) == pytest.approx(1.0)
    assert mad_uniform(5) == pytest.approx(1.2)


def test_score_confidence_one_hot_and_uniform():
    assert score_confidence([0.0, 1.0, 0.0]) == 1.0
    assert score_confidence([1 / 3] * 3, "median") == pytest.approx(0.0, abs=1e-9)


def test_normalize_guards_wire_format():
    assert normalize([0.2, 0.2]) == [0.5, 0.5]
    assert normalize([float("nan"), 1.0]) == [0.0, 1.0]
    assert normalize([-0.1, 0.0]) == [0.5, 0.5]  # all mass invalid -> uniform


# ---------------------------------------------------------------- build_answer


def test_build_choice_answer_ordering_rounding_and_ties():
    q = ChoiceQuestion(instructions="x", criteria={"billing": "a", "technical": "b", "sales": None})
    a = build_answer(q, RawDist("choice", (0.875, 0.125, 0.0), ("billing", "technical", "sales")))
    assert a.type == "choice" and a.choice == "billing"
    assert list(a.probabilities) == ["billing", "technical", "sales"]
    assert a.probabilities == {"billing": 0.88, "technical": 0.12, "sales": 0.0}
    # computed from unrounded p (0.8125 -> 0.81), matching the documented example exactly
    assert a.confidence == 0.81

    tie = build_answer(q, RawDist("choice", (0.4, 0.4, 0.2), ("billing", "technical", "sales")))
    assert tie.choice == "billing"  # first label in input order


def test_build_choice_reorders_engine_labels_to_request_order():
    q = ChoiceQuestion(criteria={"a": None, "b": None, "c": None})
    a = build_answer(q, RawDist("choice", (0.7, 0.2, 0.1), ("c", "a", "b")))
    assert list(a.probabilities) == ["a", "b", "c"]
    assert a.probabilities == {"a": 0.2, "b": 0.1, "c": 0.7}
    assert a.choice == "c"


def test_build_choice_rejects_wrong_arity():
    q = ChoiceQuestion(criteria={"a": None, "b": None})
    with pytest.raises(ValueError):
        build_answer(q, RawDist("choice", (1.0,), ("a",)))


def test_build_score_answer_echoes_legend_objects():
    levels = [{"what": "cosmetic", "examples": ["typo"]}, "degraded", ["blocking", "no workaround"]]
    q = ScoreQuestion(instructions="How severe?", criteria=levels)
    a = build_answer(q, RawDist("score", (0.0, 0.91, 0.09), ("0", "1", "2")))
    assert a.type == "score"
    assert a.legend == {"0": levels[0], "1": levels[1], "2": levels[2]}
    assert a.probabilities == {"0": 0.0, "1": 0.91, "2": 0.09}
    assert a.score == 1.09
    assert 0.0 <= a.confidence <= 1.0


def test_build_score_matches_documented_row():
    q = ScoreQuestion(criteria=["low", "mid", "high"])
    a = build_answer(q, RawDist("score", (0.0, 0.57, 0.43), ("0", "1", "2")))
    assert a.score == 1.43
    assert abs(a.confidence - 0.35) <= TOL


def test_build_noul_answer_rounds_and_clips_without_clamping_range():
    q = NoulQuestion(instructions="urgent?")
    assert build_answer(q, RawDist("noul", (0.8149,))).noul == 0.81
    assert build_answer(q, RawDist("noul", (1.0,))).noul == 1.0  # no [0.02, 0.98] clamp (SPEC C6)
    assert build_answer(q, RawDist("noul", (1.0000001,))).noul == 1.0
    assert build_answer(q, RawDist("noul", (float("nan"),))).noul == 0.5
    neg = build_answer(q, RawDist("noul", (-1e-9,))).noul
    assert neg == 0.0 and math.copysign(1.0, neg) == 1.0  # never "-0.0" on the wire


def test_build_answer_kind_mismatch():
    with pytest.raises(ValueError):
        build_answer(NoulQuestion(), RawDist("choice", (0.5, 0.5), ("a", "b")))


def test_round_digits():
    q = ChoiceQuestion(criteria={"a": None, "b": None})
    a = build_answer(q, RawDist("choice", (0.12345, 0.87655), ("a", "b")), round_digits=3)
    assert a.probabilities == {"a": 0.123, "b": 0.877}
