from __future__ import annotations

import pytest
from pydantic import ValidationError

from jev_local.engine.base import EngineError
from jev_local.schema import SystemOneRequest
from jev_local.validate import validate_limits


def req(**questions) -> SystemOneRequest:
    return SystemOneRequest.model_validate({"state": "s", "model": "jev-local", "questions": questions})


def choice(n: int) -> dict:
    return {"type": "choice", "criteria": {f"o{i}": None for i in range(n)}}


def score(n: int) -> dict:
    return {"type": "score", "criteria": [f"level {i}" for i in range(n)]}


@pytest.mark.parametrize("n", [2, 3, 255])
def test_choice_within_limits(n):
    validate_limits(req(q=choice(n)))


@pytest.mark.parametrize("n", [2, 5, 10])
def test_score_within_limits(n):
    validate_limits(req(q=score(n)))


def test_choice_too_many():
    with pytest.raises(EngineError) as e:
        validate_limits(req(ok=choice(3), big=choice(256)))
    assert e.value.status == 400
    assert e.value.detail == "Choice question 'big' has 256 options; maximum is 255"


@pytest.mark.parametrize("n, word", [(1, "1 option"), (0, "0 options")])
def test_choice_too_few(n, word):
    with pytest.raises(EngineError) as e:
        validate_limits(req(q=choice(n)))
    assert e.value.status == 400
    assert e.value.detail == f"Choice question 'q' has {word}; minimum is 2"


def test_score_too_many():
    with pytest.raises(EngineError) as e:
        validate_limits(req(q=score(11)))
    assert e.value.status == 400
    assert e.value.detail == "Score question 'q' has 11 levels; maximum is 10"


def test_score_too_few():
    with pytest.raises(EngineError) as e:
        validate_limits(req(q=score(1)))
    assert e.value.status == 400
    assert e.value.detail == "Score question 'q' has 1 level; minimum is 2"


def test_empty_score_is_a_schema_error():
    # The hosted OpenAPI declares minItems 1 for score criteria, so this is a 422 in FastAPI shape.
    with pytest.raises(EngineError) as e:
        validate_limits(req(q=score(0)))
    assert e.value.status == 422
    [err] = e.value.detail
    assert err["loc"] == ["body", "questions", "q", "score", "criteria"] and err["type"] == "too_short"


def test_nouls_have_no_limits():
    validate_limits(req(a={"type": "noul"}, b={"type": "noul", "instructions": {"q": "x"}}))


def test_schema_errors_are_pydantic_not_validate():
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": "s", "model": "m", "questions": {}})
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": None, "model": "m", "questions": {"a": {"type": "noul"}}})
