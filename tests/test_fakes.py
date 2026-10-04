"""Fakes: ScriptedEngine distributions satisfy the Engine protocol; RuleModel parses like a model."""

from __future__ import annotations

import pytest

from jev_local.engine.base import Engine
from jev_local.harness.fakes import (
    FakeExecutor,
    FakeObserver,
    RuleModel,
    ScriptedEngine,
    answers_for,
    mail_snapshot,
    spec_to_dist,
)
from jev_local.harness.questions import build_questions
from jev_local.harness.spans import extract_text_candidates, extract_url_candidates
from jev_local.harness.types import Action, ActionKind
from jev_local.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion

APPS = ["Notes", "Safari", "Visual Studio Code", "Visual Studio"]


def test_spec_to_dist_shapes():
    cq = ChoiceQuestion(criteria={"a": None, "b": None, "none": None})
    d = spec_to_dist(cq, "a")
    assert d.labels == ("a", "b", "none") and d.probs[0] == pytest.approx(0.9) and sum(d.probs) == pytest.approx(1)
    d = spec_to_dist(cq, {"a": 0.5, "b": 0.3})
    assert d.probs == pytest.approx((0.5, 0.3, 0.2))
    d = spec_to_dist(cq, {"a": 3, "b": 1})  # over-full: normalised
    assert d.probs == pytest.approx((0.75, 0.25, 0.0))
    assert spec_to_dist(cq, None).probs[2] == pytest.approx(0.9)
    assert spec_to_dist(NoulQuestion(), 0.3).probs == (0.3,)
    sq = ScoreQuestion(criteria=["x", "y", "z"])
    assert spec_to_dist(sq, 2).probs[2] == pytest.approx(0.9)
    assert spec_to_dist(sq, [0.2, 0.3, 0.5]).probs == pytest.approx((0.2, 0.3, 0.5))


def test_scripted_engine_is_an_engine():
    eng = ScriptedEngine(lambda t, q: {"intent": "click"})
    assert isinstance(eng, Engine)
    qs = build_questions(mail_snapshot(), APPS, [], [])
    res = eng.evaluate({"transcript": "click send"}, qs)
    assert set(res.dists) == set(qs) and res.engine == eng.name and eng.calls == ["click send"]
    resp = answers_for(qs, {"intent": "click", "target": "e01"})
    assert resp.answers["intent"].choice == "click" and resp.answers["target"].choice == "e01"


def parse(text: str, snap=None, model: RuleModel | None = None) -> dict:
    snap = snap if snap is not None else mail_snapshot()
    qs = build_questions(snap, APPS, extract_text_candidates(text), extract_url_candidates(text))
    resp = answers_for(qs, (model or RuleModel())(text, qs))
    out = {qid: getattr(a, "choice", getattr(a, "noul", None)) for qid, a in resp.answers.items()}
    return out


@pytest.mark.parametrize("text,intent,arg,complete", [
    ("open", "wait", None, False),
    ("open notes", "open_app", ("app", "Notes"), True),
    ("open notes and", "open_app", ("app", "Notes"), True),
    ("open visual", "wait", None, False),
    ("open visual studio code", "open_app", ("app", "Visual Studio Code"), True),
    ("open downloads", "open_folder", ("folder", "downloads"), True),
    ("type", "wait", None, False),
    ("type hello world and press enter", "type_text", ("text_span", "hello world"), True),
    ("search for salt and pepper", "search_web", ("text_span", "salt and pepper"), True),
    ("go to github dot com", "open_url", ("url_span", "github.com"), True),
    ("click send", "click", ("target", "e01"), True),
    ("click the archive button", "click", ("target", "e05"), True),
    ("click the", "wait", None, False),
    ("press enter", "press_key", ("key", "return"), True),
    ("copy that", "press_key", ("key", "cmd+c"), True),
    ("scroll down a little", "scroll_down", None, True),
    ("quit safari", "quit_app", ("app", "Safari"), True),
    ("go back", "go_back", None, True),
    ("yes", "confirm", None, True),
    ("never mind", "cancel", None, True),
    ("can you pass the salt", "none", None, None),
])
def test_rule_model_parses_first_command(text, intent, arg, complete):
    out = parse(text)
    assert out["intent"] == intent
    if arg:
        assert out[arg[0]] == arg[1]
    if complete is not None:
        assert (out["complete"] >= 0.65) is complete
    if intent == "none":
        assert out["is_command"] < 0.5


def test_rule_model_ambiguous_target_splits_mass():
    out = parse("click reply")
    assert out["intent"] == "click"
    qs = build_questions(mail_snapshot(), APPS, [], [])
    resp = answers_for(qs, RuleModel()("click reply", qs))
    assert resp.answers["target"].probabilities["e02"] < 0.35  # three "Reply" buttons


def test_rule_model_overrides_and_destructive():
    m = RuleModel(overrides={"click ok": {"intent": {"none": 0.9}}})
    assert parse("click ok", model=m)["intent"] == "none"
    assert parse("click delete")["destructive"] >= 0.5
    assert parse("click archive")["destructive"] < 0.5


def test_fake_observer_and_executor():
    obs = FakeObserver(mail_snapshot())
    s = obs.snapshot(allow_cached=True)
    obs.prefetch()
    obs.invalidate()
    assert obs.snapshot_calls == [True] and obs.prefetch_calls == obs.invalidate_calls == 1
    assert obs.resolve(s, "e01").label == "Send" and obs.focused_element().label == "Message body"
    ex = FakeExecutor(fail_kinds=[ActionKind.CLICK])
    r = ex.run(Action(ActionKind.SCROLL_DOWN, "u1+0", 0.9, amount=1), s)
    assert r.ok and r.dry_run and ex.kinds == ["scroll_down"]
    assert not ex.run(Action(ActionKind.CLICK, "u1+0", 0.9, target_eid="e01"), s).ok
