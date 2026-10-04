"""Request limits that the wire schema cannot express (docs/research/SPEC.md §5.2).

Schema errors (missing fields, wrong types, empty `questions`) are 422s raised by pydantic/FastAPI
before this runs. Count limits are 400s, matching what clients observed from the hosted API.

Benchmax W4 (suite-reproduction-specs.md §1.2): the Decision Index kit only treats a refusal as
`unsupported` (counted wrong, not retried) when the body carries one of its capacity markers; for the
option cap that marker is the phrase `options per choice`. `style="jev"` produces that wording and
`status` lets a run choose 400 (Jev parity, default) or 422 (JevBench). `style="local"` keeps the
original local wording, which the v1 tests pin.
"""

from __future__ import annotations

from typing import Literal

from jev_local.engine.base import EngineError
from jev_local.schema import ChoiceQuestion, ScoreQuestion, SystemOneRequest

MIN_CHOICE_OPTIONS = 2
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

OPTION_CAP_MARKER = "options per choice"  # Decision Index `http.CAPACITY_MARKERS` (option cap)

RefusalStyle = Literal["local", "jev"]


class OptionCapExceeded(EngineError):
    """Too many choice options: a capacity refusal (W4), distinct from other 400s for run accounting."""


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def option_cap_message(qid: str, n: int, style: RefusalStyle = "local") -> str:
    if style == "jev":
        return f"Choice question '{qid}' has {_plural(n, 'option')}; the maximum is {MAX_CHOICE_OPTIONS} {OPTION_CAP_MARKER}"
    return f"Choice question '{qid}' has {_plural(n, 'option')}; maximum is {MAX_CHOICE_OPTIONS}"


def validate_limits(req: SystemOneRequest, *, status: int = 400, style: RefusalStyle = "local") -> None:
    """`status`/`style` apply to the option-cap refusal only (W4); the other limits stay 400 / 422."""
    for qid, q in req.questions.items():
        if isinstance(q, ChoiceQuestion):
            n = len(q.criteria)
            if n > MAX_CHOICE_OPTIONS:
                raise OptionCapExceeded(option_cap_message(qid, n, style), status=status)
            if n < MIN_CHOICE_OPTIONS:
                raise EngineError(
                    f"Choice question '{qid}' has {_plural(n, 'option')}; minimum is {MIN_CHOICE_OPTIONS}", status=400
                )
        elif isinstance(q, ScoreQuestion):
            n = len(q.criteria)
            if n == 0:
                # The hosted OpenAPI declares minItems 1 here, so an empty list is a schema error
                # (422 in FastAPI's shape), not a count limit. schema.py does not encode minItems.
                raise EngineError(
                    [
                        {
                            "type": "too_short",
                            "loc": ["body", "questions", qid, "score", "criteria"],
                            "msg": "List should have at least 1 item after validation, not 0",
                            "input": [],
                            "ctx": {"field_type": "List", "min_length": 1, "actual_length": 0},
                        }
                    ],
                    status=422,
                )
            if n > MAX_SCORE_LEVELS:
                raise EngineError(
                    f"Score question '{qid}' has {_plural(n, 'level')}; maximum is {MAX_SCORE_LEVELS}", status=400
                )
            if n < MIN_SCORE_LEVELS:
                raise EngineError(
                    f"Score question '{qid}' has {_plural(n, 'level')}; minimum is {MIN_SCORE_LEVELS}", status=400
                )
