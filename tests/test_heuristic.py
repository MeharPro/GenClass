"""HeuristicEngine: protocol conformance, determinism, speed, and sane harness answers."""

from __future__ import annotations

import time

import pytest

from jev_local.api import system_one
from jev_local.engine.base import Engine
from jev_local.engine.heuristic import HeuristicEngine, reference_text
from jev_local.harness.questions import build_questions, rank_apps
from jev_local.harness.spans import extract_text_candidates, extract_url_candidates
from jev_local.harness.state import build_state
from jev_local.harness.types import Element, Snapshot
from jev_local.schema import ChoiceQuestion, NoulCriteria, NoulQuestion, ScoreQuestion, SystemOneRequest
from jev_local.serialize import state_segments

APPS = [
    "Notes", "Safari", "Mail", "Messages", "Slack", "Visual Studio Code", "System Settings", "Finder",
    "Spotify", "Calendar", "Xcode", "Terminal", "Google Chrome", "Music", "Photos", "Preview",
]


def mail_snapshot(n_filler: int = 50) -> Snapshot:
    els = [
        Element("e01", "button", "Reply (1 of 2)"),
        Element("e02", "button", "Reply All"),
        Element("e03", "button", "Forward"),
        Element("e04", "button", "Send"),
        Element("e05", "text field", "To", value="bob@example.com"),
        Element("e06", "text field", "Subject", focused=True),
        Element("e07", "button", "Delete"),
        Element("e08", "checkbox", "Remember me"),
        Element("e09", "link", "Privacy Policy"),
        Element("e10", "button", "Reply (2 of 2)"),
    ]
    roles = ["button", "link", "menu item", "checkbox", "text field"]
    els += [Element(f"e{i:02d}", roles[i % 5], f"Mailbox folder {i}") for i in range(11, 11 + n_filler)]
    return Snapshot("Mail", "com.apple.mail", 1, "Inbox", tuple(els), 0.0)


def harness_request(text: str, snap: Snapshot | None = None) -> SystemOneRequest:
    snap = snap or mail_snapshot()
    qs = build_questions(snap, rank_apps(text, APPS), extract_text_candidates(text), extract_url_candidates(text))
    return SystemOneRequest(state=build_state(text, snap), model="jev-local-heuristic", questions=qs)


def answers(text: str):
    return system_one(harness_request(text), {"heuristic": HeuristicEngine()}).answers


def test_implements_engine_protocol():
    e = HeuristicEngine()
    assert isinstance(e, Engine)
    assert e.name == "jev-local-heuristic-0.1.0"
    assert e.supports(harness_request("open notes"))


def test_deterministic():
    req = harness_request("open notes and type hello world")
    e = HeuristicEngine()
    a = e.evaluate(req.state, req.questions).dists
    b = HeuristicEngine().evaluate(req.state, req.questions).dists
    assert a == b


def test_fast_on_a_60_element_harness_request():
    req = harness_request("okay open notes and then type hello world into the note and click send")
    assert len(req.questions["target"].criteria) == 61
    e = HeuristicEngine()
    e.evaluate(req.state, req.questions)  # warm caches/imports
    times = []
    for _ in range(30):
        t = time.perf_counter()
        e.count_tokens(req)
        e.evaluate(req.state, req.questions)
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    # Contract: < 5 ms. The minimum is robust to other processes loading the machine; the median
    # bound is loose for the same reason (measured p50 ~2-3 ms on a busy M1).
    assert times[0] < 5.0, times
    assert times[len(times) // 2] < 15.0, times


def test_count_tokens_is_plausible_and_memoised():
    req = harness_request("open notes")
    e = HeuristicEngine()
    n = e.count_tokens(req)
    assert 800 < n < 4000
    assert e.evaluate(req.state, req.questions).input_tokens == n


@pytest.mark.parametrize(
    "text, intent",
    [
        ("open notes", "open_app"),
        ("open notes and type hello world", "open_app"),  # the first command
        ("click the send button", "click"),
        ("type hello world", "type_text"),
        ("search for cats", "search_web"),
        ("press enter", "press_key"),
        ("copy that", "press_key"),
        ("scroll down a little", "scroll_down"),
        ("quit safari", "quit_app"),
        ("close the tab", "close_tab"),
        ("go back", "go_back"),
        ("undo that", "undo"),
        ("yes", "confirm"),
        ("no", "cancel"),
        ("open downloads folder", "open_folder"),
        ("new tab", "new_tab"),
        ("can you pass the salt", "none"),
    ],
)
def test_intent_on_easy_commands(text, intent):
    assert answers(text)["intent"].choice == intent


def test_arguments():
    a = answers("click the send button")
    assert a["target"].choice == "e04"
    assert answers("click reply all")["target"].choice == "e02"
    assert answers("open notes")["app"].choice == "Notes"
    assert answers("quit safari")["app"].choice == "Safari"
    assert answers("press enter")["key"].choice == "return"
    assert answers("copy that")["key"].choice == "cmd+c"
    assert answers("open downloads folder")["folder"].choice == "downloads"
    assert answers("type hello world")["text_span"].choice == "hello world"
    assert answers("go to github dot com")["url_span"].choice == "github.com"
    assert answers("open notes")["target"].choice == "none"
    assert answers("scroll down a little")["scroll_amount"].score < answers("scroll to the bottom")["scroll_amount"].score


def test_nouls():
    assert answers("open")["complete"].noul < 0.3
    assert answers("click the")["complete"].noul < 0.3
    assert answers("type hello world")["complete"].noul > 0.7
    assert answers("scroll down")["complete"].noul > 0.7
    assert answers("open notes")["is_command"].noul > 0.7
    assert answers("can you pass the salt")["is_command"].noul < 0.3
    assert answers("delete this email")["destructive"].noul > 0.5
    assert answers("open notes")["destructive"].noul < 0.5


def test_noul_without_cues_is_half():
    e = HeuristicEngine()
    r = e.evaluate("anything", {"q": NoulQuestion()})
    assert r.dists["q"].probs == (0.5,)
    r = e.evaluate("anything", {"q": NoulQuestion(instructions="Is `x` true?")})
    assert r.dists["q"].probs == (0.5,)


def test_noul_keyword_cues_and_negation():
    e = HeuristicEngine()
    q = {"q": NoulQuestion(instructions="Does the customer ask for a refund?")}
    assert e.evaluate("I want a refund for order 12", q).dists["q"].probs[0] > 0.6
    assert e.evaluate("I do not want a refund", q).dists["q"].probs[0] < 0.5
    assert e.evaluate("The weather is nice", q).dists["q"].probs[0] < 0.5


def test_noul_criteria_sides():
    e = HeuristicEngine()
    q = {"q": NoulQuestion(instructions="Is it spam?", criteria=NoulCriteria(true="buy cheap pills now", false="lunch at noon?"))}
    assert e.evaluate("buy cheap pills now!!!", q).dists["q"].probs[0] > 0.8
    assert e.evaluate("lunch at noon?", q).dists["q"].probs[0] < 0.2


def test_choice_uniform_without_evidence_and_permutation_invariant():
    e = HeuristicEngine()
    q = ChoiceQuestion(criteria={"red": None, "green": None, "blue": None})
    r = e.evaluate("nothing relevant here", {"q": q}).dists["q"]
    assert r.probs == pytest.approx((1 / 3,) * 3)
    q1 = ChoiceQuestion(criteria={"billing": "payments and invoices", "technical": "bugs and outages", "sales": "pricing"})
    q2 = ChoiceQuestion(criteria={"sales": "pricing", "billing": "payments and invoices", "technical": "bugs and outages"})
    s = "The app has bugs and an outage since yesterday"
    d1, d2 = e.evaluate(s, {"q": q1}).dists["q"], e.evaluate(s, {"q": q2}).dists["q"]
    assert dict(zip(d1.labels, d1.probs)) == pytest.approx(dict(zip(d2.labels, d2.probs)))
    assert max(zip(d1.probs, d1.labels))[1] == "technical"


def test_questions_are_isolated():
    e = HeuristicEngine()
    base = {"a": ChoiceQuestion(criteria={"x": "cats", "y": "dogs"}), "b": NoulQuestion(instructions="about cats?")}
    more = dict(base, c=ScoreQuestion(criteria=["low", "high"]))
    r1, r2 = e.evaluate("I love cats", base).dists, e.evaluate("I love cats", more).dists
    assert r1["a"] == r2["a"] and r1["b"] == r2["b"]


def test_reference_text_follows_backticked_path():
    segs = state_segments({"screen": "Mail", "transcript": "open notes", "notes": "x"})
    assert reference_text(segs, "What does `transcript` ask?") == "open notes"
    assert reference_text(segs, "What does `notes.body` say?") == "x"
    assert reference_text(segs, "No path here") == "open notes"  # transcript-like key
    whole = reference_text(state_segments({"a": "1", "b": "2"}), "q")
    assert "a: 1" in whole and "b: 2" in whole
    assert reference_text(state_segments("plain text"), "q") == "plain text"


def test_score_is_uniformish():
    e = HeuristicEngine()
    q = {"s": ScoreQuestion(criteria=["calm", "annoyed", "furious and angry"])}
    p = e.evaluate("I am furious", q).dists["s"].probs
    assert max(p) < 0.8 and p[2] == max(p)
    assert e.evaluate("", q).dists["s"].probs == pytest.approx((1 / 3,) * 3)
