"""Engine interface shared by every decision engine (fast encoder, MLX decoder, test fakes).

An engine turns (state, questions) into raw probability distributions. Turning those into
Jev-shaped answers (rounding, confidence formulas, legends) is done once, in
`jev_local.api.build_answer`, so every engine produces byte-identical response shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Protocol, runtime_checkable

from jev_local.schema import Entry, Question, SystemOneRequest

Kind = Literal["noul", "choice", "score"]


@dataclass(frozen=True)
class RawDist:
    """Unrounded distribution for one question.

    noul:   probs == (p_yes,), labels == ()
    choice: probs[i] is the probability of labels[i]; labels in the request's criteria order
    score:  probs[i] is the probability of level i; labels == ("0", "1", ...)
    """

    kind: Kind
    probs: tuple[float, ...]
    labels: tuple[str, ...] = ()


@dataclass
class EngineResult:
    dists: dict[str, RawDist]
    input_tokens: int
    output_tokens: int
    engine: str  # concrete versioned id, e.g. "jev-local-fast-0.1.0"
    cached_tokens: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)  # serialize/pack/forward/heads/total


@runtime_checkable
class Engine(Protocol):
    name: str  # concrete versioned id
    max_tokens: int

    def supports(self, req: SystemOneRequest) -> bool: ...

    def count_tokens(self, req: SystemOneRequest) -> int: ...

    def evaluate(self, state: Entry, questions: Mapping[str, Question]) -> EngineResult: ...


class EngineError(Exception):
    """Raised by engines for request-level problems (e.g. too long). Mapped to HTTP 400."""

    def __init__(self, detail: object, status: int = 400):
        super().__init__(str(detail))
        self.detail = detail
        self.status = status
