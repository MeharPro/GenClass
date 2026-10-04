"""Wire models for the Jev-compatible System One API (docs/research/SPEC.md §1.3).

Request:  {state, model, questions: {id: Noul|Choice|Score}}
Response: {model, answers: {id: Answer}, usage: {input_tokens, output_tokens}}
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

# An "entry" is any JSON text-ish value: instructions, criteria descriptions, legend levels.
Entry = Union[str, dict[str, Any], list[Any]]


class NoulCriteria(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    true: Entry | None = None
    false: Entry | None = None


class NoulQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["noul"] = "noul"
    instructions: Entry | None = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["choice"] = "choice"
    instructions: Entry | None = None
    # label -> description (None means "read the label by its name alone"). Order is meaningful
    # only for tie-breaking and for the order of `probabilities` in the answer.
    criteria: dict[str, Entry | None]


class ScoreQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["score"] = "score"
    instructions: Entry | None = None
    criteria: list[Entry]  # index = level


Question = Annotated[Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    state: Entry
    model: str
    questions: dict[str, Question] = Field(min_length=1)


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: dict[str, Entry]
    probabilities: dict[str, float]


Answer = Annotated[Union[NoulAnswer, ChoiceAnswer, ScoreAnswer], Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage


def question_from_json(obj: dict[str, Any]) -> NoulQuestion | ChoiceQuestion | ScoreQuestion:
    """Parse one question dict (as found in a request or a training example)."""
    kind = obj.get("type")
    if kind == "noul":
        return NoulQuestion.model_validate(obj)
    if kind == "choice":
        return ChoiceQuestion.model_validate(obj)
    if kind == "score":
        return ScoreQuestion.model_validate(obj)
    raise ValueError(f"unknown question type: {kind!r}")


def question_to_json(q: NoulQuestion | ChoiceQuestion | ScoreQuestion) -> dict[str, Any]:
    return q.model_dump(mode="json", exclude_none=True)
