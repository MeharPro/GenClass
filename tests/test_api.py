"""api.route / api.system_one with stub engines (no models)."""

from __future__ import annotations

from typing import Mapping

import pytest

from jev_local import api
from jev_local.engine.base import EngineError, EngineResult, RawDist
from jev_local.engine.heuristic import HeuristicEngine
from jev_local.schema import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion, SystemOneRequest


class Stub:
    """Deterministic engine: choice -> first option 0.7, score -> level 1, noul -> 0.8."""

    def __init__(self, name: str, max_tokens: int = 10_000, tokens: int = 100, supports: bool = True, raise_on_eval=None):
        self.name = name
        self.max_tokens = max_tokens
        self.tokens = tokens
        self._supports = supports
        self.raise_on_eval = raise_on_eval
        self.calls = 0
        self.counted = 0

    def supports(self, req: SystemOneRequest) -> bool:
        return self._supports

    def count_tokens(self, req: SystemOneRequest) -> int:
        self.counted += 1
        return self.tokens

    def evaluate(self, state, questions: Mapping[str, Question]) -> EngineResult:
        self.calls += 1
        if self.raise_on_eval is not None:
            raise self.raise_on_eval
        dists = {}
        for qid, q in questions.items():
            if isinstance(q, ChoiceQuestion):
                labels = tuple(q.criteria)
                rest = 0.3 / (len(labels) - 1)
                dists[qid] = RawDist("choice", (0.7,) + (rest,) * (len(labels) - 1), labels)
            elif isinstance(q, ScoreQuestion):
                n = len(q.criteria)
                p = [0.0] * n
                p[1] = 1.0
                dists[qid] = RawDist("score", tuple(p), tuple(str(i) for i in range(n)))
            else:
                dists[qid] = RawDist("noul", (0.8,))
        return EngineResult(dists, self.tokens, 0, self.name)


def make_req(model: str = "jev-local", **extra) -> SystemOneRequest:
    questions = {
        "department": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "pay", "technical": None, "sales": None}},
        "frustration": {"type": "score", "criteria": ["calm", "annoyed", "angry"]},
        "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    }
    questions.update(extra)
    return SystemOneRequest.model_validate({"state": "My payouts fail.", "model": model, "questions": questions})


def engines(**kw) -> dict:
    return {k: v for k, v in kw.items() if v is not None}


@pytest.mark.parametrize("mid", ["jev-local", "jev", "jev-latest", "jev-preview", "jev-1.13", "jev-1.13.0", "typesafe/jev-1.13"])
def test_auto_ids_prefer_fast(mid):
    fast, general, heur = Stub("jev-local-fast-0.1.0"), Stub("jev-local-general-0.1.0"), HeuristicEngine()
    assert api.route(make_req(mid), engines(fast=fast, general=general, heuristic=heur)) is fast
    assert general.counted == 0  # lazy: stops at the first fit


def test_auto_falls_back_when_fast_absent_unsupported_or_too_long():
    general, heur = Stub("g"), HeuristicEngine()
    assert api.route(make_req(), engines(general=general, heuristic=heur)) is general
    assert api.route(make_req(), engines(fast=Stub("f", supports=False), general=general)) is general
    assert api.route(make_req(), engines(fast=Stub("f", max_tokens=50), general=general)) is general
    assert api.route(make_req(), engines(heuristic=heur)) is heur


@pytest.mark.parametrize(
    "mid, key",
    [
        ("jev-local-fast", "fast"),
        ("jev-local-fast-0.1.0", "fast"),
        ("jev-local-general", "general"),
        ("jev-local-general-0.1.0", "general"),
        ("jev-local-heuristic", "heuristic"),
        ("jev-local-heuristic-0.1.0", "heuristic"),
    ],
)
def test_explicit_ids(mid, key):
    es = engines(fast=Stub("f"), general=Stub("g"), heuristic=HeuristicEngine())
    assert api.route(make_req(mid), es) is es[key]


@pytest.mark.parametrize("mid", ["gpt-4o", "jev-1.12", "JEV-LOCAL", "", "jev-local2"])
def test_unknown_model_404(mid):
    with pytest.raises(EngineError) as e:
        api.route(make_req(mid), engines(heuristic=HeuristicEngine()))
    assert e.value.status == 404
    assert e.value.detail == f"Model not found: {mid}"


def test_explicit_engine_not_installed_is_404():
    with pytest.raises(EngineError) as e:
        api.route(make_req("jev-local-fast"), engines(heuristic=HeuristicEngine()))
    assert e.value.status == 404 and str(e.value.detail).startswith("Model not found: jev-local-fast")


def test_too_long_everywhere_is_400_max_tokens_exceeded():
    es = engines(fast=Stub("f", max_tokens=10), general=Stub("g", max_tokens=10))
    with pytest.raises(EngineError) as e:
        api.route(make_req(), es)
    assert e.value.status == 400
    assert e.value.detail[0]["type"] == "max_tokens_exceeded"
    with pytest.raises(EngineError) as e:
        api.route(make_req("jev-local-general"), es)
    assert e.value.detail[0]["type"] == "max_tokens_exceeded"


def test_no_engines_is_503():
    with pytest.raises(EngineError) as e:
        api.route(make_req(), {})
    assert e.value.status == 503


def test_system_one_builds_jev_shaped_response():
    fast = Stub("jev-local-fast-0.1.0", tokens=318)
    resp = api.system_one(make_req(), engines(fast=fast))
    assert resp.model == "jev-local-fast-0.1.0"  # concrete versioned id, not the alias
    d = resp.answers["department"]
    assert d.type == "choice" and d.choice == "billing" and d.probabilities == {"billing": 0.7, "technical": 0.15, "sales": 0.15}
    assert d.confidence == 0.55
    s = resp.answers["frustration"]
    assert s.type == "score" and s.score == 1.0 and s.confidence == 1.0
    assert s.legend == {"0": "calm", "1": "annoyed", "2": "angry"}
    assert resp.answers["urgent"].noul == 0.8
    assert list(resp.answers) == ["department", "frustration", "urgent"]  # request order
    # input: the engine's count; output: Σ(items + 1) = (3+1) + (3+1) + (2+1)
    assert resp.usage.input_tokens == 318 and resp.usage.output_tokens == 11


def test_system_one_validates_before_routing():
    fast = Stub("f")
    big = {"type": "choice", "criteria": {f"o{i}": None for i in range(256)}}
    with pytest.raises(EngineError) as e:
        api.system_one(make_req(big=big), engines(fast=fast))
    assert e.value.status == 400 and fast.calls == 0


def test_auto_retries_next_engine_when_packing_exceeds_estimate():
    too_long = EngineError([{"type": "max_tokens_exceeded", "msg": "2,100 > 2,048"}], 400)
    fast, general = Stub("f", raise_on_eval=too_long), Stub("g")
    served = api.serve_request(make_req(), engines(fast=fast, general=general))
    assert served.engine_key == "general" and served.response.model == "g"
    # explicit ids never fall back
    with pytest.raises(EngineError):
        api.serve_request(make_req("jev-local-fast"), engines(fast=fast, general=general))
    # auto with nothing left reports the engine's own error
    with pytest.raises(EngineError) as e:
        api.serve_request(make_req(), engines(fast=fast))
    assert e.value.detail[0]["msg"] == "2,100 > 2,048"


@pytest.mark.parametrize("mid", ["jev-local", "jev-local-fast"])
def test_engine_too_long_errors_are_normalised_to_the_wire_shape(mid):
    # The encoder's packer reports {"detail": "max_tokens_exceeded", "tokens": n, "max_tokens": m}.
    raw = EngineError({"detail": "max_tokens_exceeded", "tokens": 2100, "max_tokens": 2048}, 400)
    with pytest.raises(EngineError) as e:
        api.serve_request(make_req(mid), engines(fast=Stub("jev-local-fast-0.1.0", raise_on_eval=raw)))
    assert e.value.status == 400
    assert e.value.detail == [
        {
            "type": "max_tokens_exceeded",
            "msg": "state+questions need 2,100 tokens; jev-local-fast-0.1.0 accepts at most 2,048 tokens (local limit)",
        }
    ]


def test_other_engine_errors_propagate():
    boom = EngineError("bad request for this engine", 400)
    with pytest.raises(EngineError) as e:
        api.system_one(make_req(), engines(fast=Stub("f", raise_on_eval=boom), general=Stub("g")))
    assert e.value.detail == "bad request for this engine"


def test_missing_answer_is_an_engine_bug():
    class Lossy(Stub):
        def evaluate(self, state, questions):
            r = super().evaluate(state, questions)
            r.dists.pop("urgent")
            return r

    with pytest.raises(RuntimeError):
        api.system_one(make_req(), engines(fast=Lossy("f")))


def test_output_tokens_counts_items_plus_one():
    r = SystemOneRequest(
        state="x",
        model="jev-local",
        questions={
            "n": NoulQuestion(),
            "c": ChoiceQuestion(criteria={"a": None, "b": None, "c": None, "d": None}),
            "s": ScoreQuestion(criteria=["0", "1"]),
        },
    )
    assert api.output_tokens(r) == 3 + 5 + 3


def test_system_one_with_heuristic_end_to_end():
    resp = api.system_one(make_req(), engines(heuristic=HeuristicEngine()))
    assert resp.model == "jev-local-heuristic-0.1.0"
    assert resp.usage.input_tokens > 0
    assert set(resp.answers) == {"department", "frustration", "urgent"}
