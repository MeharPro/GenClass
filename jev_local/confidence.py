"""Answer math: raw distributions -> Jev-shaped answers (docs/research/SPEC.md §1.4, §5.3).

Every engine returns unrounded `RawDist`s; this module is the only place that turns them into wire
answers, so all engines produce identical shapes, rounding and confidence semantics.

Formulas (reproduce every documented jev-1.13 example within ±0.01; the docs round the displayed
probabilities, so exact agreement on rounded inputs is not possible, e.g. 0.88/0.12/0 gives 0.82
from the rounded numbers while the docs show 0.81):

- choice confidence = (K·pmax − 1)/(K − 1)
- score             = Σ i·pᵢ
- score confidence  = max(0, 1 − Σ pᵢ·|i − c| / MAD_uniform(K)), c = mode (official adapter) or
  median (first i with cumulative p ≥ 0.5). Both centres reproduce all four documented score
  cases; mode is the default because the official adapter uses it.

Precision modes (benchmax W3, suite-reproduction-specs.md §1.2):
- "round": every number rounded independently to `round_digits` (Jev returns 2 dp). Σp can drift from 1 by up
  to K/2 units in the last place, which breaks the Decision Index |Σp − 1| ≤ 0.01 check for 77–255 options.
- "lr":    probabilities rounded by the largest-remainder method to `round_digits`, so they sum to exactly 1.00
  in decimal; `choice`, `score` and `confidence` still come from the unrounded distribution.
- "exact": nothing is rounded (leaderboard precision mode).
`choice` is always the argmax of the UNROUNDED probabilities, ties to the first key in request order (W1).
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

Precision = Literal["round", "lr", "exact"]

from jev_local.engine.base import RawDist
from jev_local.schema import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)


def choice_confidence(p: Sequence[float]) -> float:
    k = len(p)
    if k <= 1:
        return 1.0
    return _clip01((k * max(p) - 1.0) / (k - 1.0))


def score_value(p: Sequence[float]) -> float:
    return sum(i * pi for i, pi in enumerate(p))


def mad_uniform(k: int) -> float:
    """Mean |i − (K−1)/2| over a uniform distribution on 0..K−1 (0.5, 2/3, 1.0, 1.2 for K=2..5)."""
    c = (k - 1) / 2.0
    return sum(abs(i - c) for i in range(k)) / k


def score_center(p: Sequence[float], center: Literal["mode", "median"] = "mode") -> int:
    if center == "mode":
        return max(range(len(p)), key=lambda i: (p[i], -i))  # first index on ties
    if center == "median":
        acc = 0.0
        for i, pi in enumerate(p):
            acc += pi
            if acc >= 0.5 - 1e-12:
                return i
        return len(p) - 1
    raise ValueError(f"unknown center: {center!r}")


def score_confidence(p: Sequence[float], center: Literal["mode", "median"] = "mode") -> float:
    k = len(p)
    if k <= 1:
        return 1.0
    c = score_center(p, center)
    spread = sum(pi * abs(i - c) for i, pi in enumerate(p))
    return max(0.0, 1.0 - spread / mad_uniform(k))


# ---------------------------------------------------------------- building answers


def _clip01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    return min(1.0, max(0.0, x)) + 0.0  # "+ 0.0" turns -0.0 into 0.0 so JSON never shows "-0.0"


def _round(x: float, digits: int) -> float:
    return round(_clip01(x), digits) + 0.0


def normalize(probs: Sequence[float]) -> list[float]:
    """Clip to [0, 1], drop NaN/inf, renormalise; an all-zero input becomes uniform.

    Engines are expected to return proper distributions already. This guards the wire format
    (probabilities in [0, 1] that sum to ~1) against numerical slop such as fp16 softmax drift.
    """
    clean = [x if math.isfinite(x) and x > 0 else 0.0 for x in probs]
    total = sum(clean)
    if total <= 0:
        return [1.0 / len(clean)] * len(clean) if clean else []
    return [x / total for x in clean]


def largest_remainder(p: Sequence[float], digits: int = 2) -> list[float]:
    """Round a distribution to `digits` decimals so the rounded values sum to exactly 10^-digits × 10^digits.

    Hamilton's method: floor every value in units of 10^-digits, then hand the missing units to the entries
    with the largest remainders (ties: earlier index first), so the result is deterministic and the decimal
    sum is exactly 1. The input must already be a normalised distribution (see `normalize`).
    """
    if not p:
        return []
    scale = 10**digits
    raw = [x * scale for x in p]
    units = [math.floor(r) for r in raw]
    short = scale - sum(units)
    if short > 0:
        order = sorted(range(len(p)), key=lambda i: (-(raw[i] - units[i]), i))
        for i in order[:short]:
            units[i] += 1
    elif short < 0:  # only possible if the input sums above 1 (floating slop): take from the smallest remainders
        order = sorted(range(len(p)), key=lambda i: ((raw[i] - units[i]), -i))
        for i in order[: -short]:
            units[i] -= 1
    return [u / scale + 0.0 for u in units]


def round_probs(p: Sequence[float], precision: Precision, digits: int) -> list[float]:
    if precision == "exact":
        return [x + 0.0 for x in p]
    if precision == "lr":
        return largest_remainder(p, digits)
    if precision == "round":
        return [_round(x, digits) for x in p]
    raise ValueError(f"unknown precision {precision!r}")


def _round_scalar(x: float, precision: Precision, digits: int) -> float:
    return x + 0.0 if precision == "exact" else round(x, digits) + 0.0


def _ordered_probs(dist: RawDist, labels: Sequence[str]) -> list[float]:
    """Probabilities in the question's label order, whatever order the engine used."""
    if len(dist.probs) != len(labels):
        raise ValueError(f"engine returned {len(dist.probs)} probabilities for {len(labels)} options")
    if not dist.labels or tuple(dist.labels) == tuple(labels):
        return list(dist.probs)
    by_label = dict(zip(dist.labels, dist.probs))
    missing = [lab for lab in labels if lab not in by_label]
    if missing:
        raise ValueError(f"engine returned no probability for options {missing[:5]!r}")
    return [by_label[lab] for lab in labels]


def build_answer(
    q: Question,
    d: RawDist,
    round_digits: int = 2,
    center: Literal["mode", "median"] = "mode",
    precision: Precision = "round",
) -> Answer:
    if isinstance(q, NoulQuestion):
        _check_kind(d, "noul")
        p = d.probs[0] if d.probs else float("nan")
        p = _clip01(p if math.isfinite(p) else 0.5)
        return NoulAnswer(noul=_round_scalar(p, precision, round_digits))

    if isinstance(q, ChoiceQuestion):
        _check_kind(d, "choice")
        labels = list(q.criteria.keys())
        p = normalize(_ordered_probs(d, labels))
        best = max(range(len(p)), key=lambda i: (p[i], -i))  # unrounded argmax; ties -> first label (W1)
        return ChoiceAnswer(
            choice=labels[best],
            confidence=_round_scalar(choice_confidence(p), precision, round_digits),
            probabilities=dict(zip(labels, round_probs(p, precision, round_digits))),
        )

    if isinstance(q, ScoreQuestion):
        _check_kind(d, "score")
        levels = [str(i) for i in range(len(q.criteria))]
        p = normalize(_ordered_probs(d, levels))
        return ScoreAnswer(
            score=_round_scalar(score_value(p), precision, round_digits),  # Σ i·pᵢ, unrounded in exact mode (W2)
            confidence=_round_scalar(score_confidence(p, center), precision, round_digits),
            legend={lvl: crit for lvl, crit in zip(levels, q.criteria)},  # echoed verbatim, objects kept
            probabilities=dict(zip(levels, round_probs(p, precision, round_digits))),
        )

    raise TypeError(f"unsupported question: {type(q).__name__}")


def _check_kind(d: RawDist, kind: str) -> None:
    if d.kind != kind:
        raise ValueError(f"engine returned a {d.kind} distribution for a {kind} question")
