"""Policy gates, in the contract's order, on API-shaped answers built from scripted specs."""

from __future__ import annotations

from typing import Any

import pytest

from jev_local.harness.fakes import answers_for, finder_snapshot, mail_snapshot, make_snapshot
from jev_local.harness.policy import PolicyContext, evaluate_policy, pick_of
from jev_local.harness.questions import build_questions
from jev_local.harness.spans import extract_text_candidates, extract_url_candidates
from jev_local.harness.types import Action, ActionKind, Snapshot, Tail, Thresholds

T = Thresholds()
APPS = ["Notes", "Safari", "Mail", "Finder"]


def tail(text: str, *, final: bool = False, silent: int = 0, vid: str = "u1+0", cursor: int = 0) -> Tail:
    uid = vid.split("+")[0]
    return Tail(vid=vid, text=text, words=tuple(text.split()), cursor=cursor, is_final=final, silent_ms=silent, uid=uid)


def run(text: str, spec: dict[str, Any], *, snap: Snapshot | None = None, ctx: PolicyContext | None = None,
        **tail_kw: Any):
    snap = snap if snap is not None else finder_snapshot()
    qs = build_questions(snap, APPS, extract_text_candidates(text), extract_url_candidates(text))
    resp = answers_for(qs, spec)
    tl = tail(text, **tail_kw)
    return evaluate_policy(resp, tl, snap, ctx or PolicyContext(), T)


# ---------------------------------------------------------------- 2-3: command / intent


def test_side_talk_is_ignored():
    d = run("can you pass the salt", {"intent": "none", "is_command": 0.05})
    assert d.verdict == "ignore" and d.action is None


def test_intent_none_ignored_and_wait_waits():
    assert run("hmm", {"intent": "none"}).verdict == "ignore"
    assert run("open", {"intent": "wait", "complete": 0.1}).verdict == "wait"


def test_low_intent_waits():
    d = run("open notes", {"intent": {"open_app": 0.4, "quit_app": 0.35}, "app": "Notes"})
    assert d.verdict == "wait" and "low" in d.reason


def test_confirm_or_cancel_without_pending_is_ignored():
    assert run("yes", {"intent": "confirm"}).verdict == "ignore"
    assert run("cancel", {"intent": "cancel"}).verdict == "ignore"


# ---------------------------------------------------------------- 4: complete gate


def test_incomplete_partial_waits_with_retry():
    d = run("click the", {"intent": "click", "complete": 0.2}, silent=100)
    assert d.verdict == "wait" and d.retry_in_ms == T.silence_complete_ms - 100


def test_incomplete_after_silence_or_final_reaches_argument_gate():
    d = run("open", {"intent": "open_app", "complete": 0.2, "app": "none"}, silent=950)
    assert d.verdict == "clarify" and "app" in d.reason
    d = run("open", {"intent": "open_app", "complete": 0.2, "app": "none"}, final=True)
    assert d.verdict == "clarify"


# ---------------------------------------------------------------- 5: payload gate


def test_payload_waits_for_final_or_silence():
    spec = {"intent": "type_text", "complete": 0.97, "text_span": "hello world"}
    d = run("type hello world", spec)
    assert d.verdict == "wait" and d.retry_in_ms == T.payload_silence_ms
    d = run("type hello world", spec, silent=T.payload_silence_ms)
    assert d.verdict == "act" and d.action.text == "hello world" and d.action.kind == ActionKind.TYPE_TEXT
    d = run("type hello world", spec, final=True)
    assert d.verdict == "act" and d.action.consumed_words == 3


def test_stale_answer_never_passes_payload_or_silence_gates():
    spec = {"intent": "type_text", "complete": 0.97, "text_span": "hello"}
    d = run("type hello", spec, final=True, silent=2000, ctx=PolicyContext(stale=True))
    assert d.verdict == "wait"
    d = run("open", {"intent": "open_app", "complete": 0.2, "app": "none"}, silent=2000,
            ctx=PolicyContext(stale=True))
    assert d.verdict == "wait" and "incomplete" in d.reason


def test_stale_answer_may_commit_closed_set():
    d = run("open notes", {"intent": "open_app", "complete": 0.95, "app": "Notes"}, ctx=PolicyContext(stale=True))
    assert d.verdict == "act" and d.action.app == "Notes"


# ---------------------------------------------------------------- 6: stability


def test_stability_needs_high_complete_or_two_agreeing_evals():
    spec = {"intent": "open_app", "complete": 0.75, "app": "Notes"}
    d1 = run("open notes", spec)
    assert d1.verdict == "wait" and "unstable" in d1.reason and d1.action is not None
    ctx = PolicyContext(prev_pick=pick_of(d1.action))
    d2 = run("open notes and", spec, ctx=ctx)
    assert d2.verdict == "act" and d2.action.app == "Notes"
    # a different argument on the previous evaluation is not agreement
    ctx = PolicyContext(prev_pick=("u1+0", "open_app", "Safari"))
    assert run("open notes and", spec, ctx=ctx).verdict == "wait"
    # nor is the same pick from another virtual utterance
    ctx = PolicyContext(prev_pick=("u0+3", "open_app", "Notes"))
    assert run("open notes and", spec, ctx=ctx).verdict == "wait"
    # high complete acts on the first partial; a final needs no second opinion
    assert run("open notes", {**spec, "complete": 0.9}).verdict == "act"
    assert run("open notes", spec, final=True).verdict == "act"


# ---------------------------------------------------------------- 7: argument gates


def test_app_gate():
    d = run("open notes", {"intent": "open_app", "app": {"Notes": 0.4, "Safari": 0.35}}, final=True)
    assert d.verdict == "clarify"
    d = run("open notes", {"intent": "open_app", "app": {"Notes": 0.4, "Safari": 0.35}})
    assert d.verdict == "wait" and "not resolved" in d.reason  # mid-sentence: no clarify yet


def test_target_gate_and_element_checks():
    snap = mail_snapshot()
    d = run("click send", {"intent": "click", "target": "e01"}, snap=snap)
    assert d.verdict == "act" and d.action.target_eid == "e01" and d.action.target_label == 'button "Send"'
    d = run("click reply", {"intent": "click", "target": {"e02": 0.34, "e03": 0.33, "e04": 0.33}}, snap=snap,
            final=True)
    assert d.verdict == "clarify" and "which element" in d.reason
    disabled = make_snapshot("Mail", [("button", "Send", {"enabled": False}), ("button", "Archive")])
    d = run("click send", {"intent": "click", "target": "e01"}, snap=disabled, final=True)
    assert d.verdict == "clarify" and "disabled" in d.reason
    d = run("click send", {"intent": "click", "target": "none"}, snap=snap, final=True)
    assert d.verdict == "clarify"


def test_span_must_be_verbatim():
    # A misbehaving engine can only choose among the question's options, but guard anyway.
    snap = finder_snapshot()
    qs = build_questions(snap, APPS, ["hello world"], [])
    resp = answers_for(qs, {"intent": "type_text", "text_span": "hello world"})
    d = evaluate_policy(resp, tail("type goodbye", final=True), snap, PolicyContext(), T)
    assert d.verdict == "clarify" and "verbatim" in d.reason


def test_key_folder_url_scroll_arguments():
    d = run("press enter", {"intent": "press_key", "key": "return"}, final=True)
    assert d.verdict == "act" and d.action.key == "return"
    d = run("press enter", {"intent": "press_key", "key": {"return": 0.5, "escape": 0.4}}, final=True)
    assert d.verdict == "clarify"
    d = run("open downloads", {"intent": "open_folder", "folder": "downloads"}, final=True)
    assert d.verdict == "act" and d.action.folder == "downloads"
    d = run("go to github dot com", {"intent": "open_url", "url_span": "github.com"}, final=True)
    assert d.verdict == "act" and d.action.url == "https://github.com"
    d = run("scroll down all the way", {"intent": "scroll_down", "scroll_amount": 2}, final=True)
    assert d.verdict == "act" and d.action.amount == 2


def test_url_waits_until_the_address_is_finished():
    spec = {"intent": "open_url", "complete": 0.97, "url_span": "wikipedia.com"}
    # "go to wikipedia" may still become "wikipedia dot org": no guessing mid-sentence
    d = run("go to wikipedia", spec)
    assert d.verdict == "wait" and "address" in d.reason
    d = run("go to wikipedia", spec, silent=T.payload_silence_ms)
    assert d.verdict == "act" and d.action.url == "https://wikipedia.com"
    spec = {"intent": "open_url", "complete": 0.97, "url_span": "wikipedia.org"}
    assert run("go to wikipedia dot", spec).verdict == "wait"
    assert run("go to wikipedia dot org", spec).verdict == "wait"
    # a chain word after the TLD closes the address while the sentence goes on
    d = run("go to wikipedia dot org and", spec)
    assert d.verdict == "act" and d.action.url == "https://wikipedia.org"
    assert run("go to barnes and noble", {"intent": "open_url", "complete": 0.97, "url_span": "barnes.com"}).verdict == "wait"


# ---------------------------------------------------------------- 8: the action


def test_action_confidence_is_min_and_consumption_stops_at_chain_word():
    d = run("open notes and type hello", {"intent": {"open_app": 0.9}, "app": {"Notes": 0.6}, "complete": 0.95})
    assert d.verdict == "act"
    a = d.action
    assert a.consumed_words == 3  # "open notes and"
    assert a.source_vid == "u1+0"
    # min(intent conf, app conf) with conf=(K*p-1)/(K-1): app has 5 options -> (5*.6-1)/4 = .5
    assert a.confidence == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------- 1: pending confirmation


def pending_ctx(**kw: Any) -> PolicyContext:
    p = Action(ActionKind.QUIT_APP, source_vid="u1+0", confidence=0.9, app="Safari", consumed_words=2)
    return PolicyContext(pending=p, pending_mark=("u1", 2), **kw)


def test_confirm_from_new_utterance_after_prompt():
    d = run("yes", {"intent": "confirm", "complete": 0.95}, vid="u2+0", ctx=pending_ctx())
    assert d.verdict == "act" and d.action.kind == ActionKind.CONFIRM


def test_confirm_never_from_proposing_utterance_or_words_heard_before_prompt():
    d = run("yes", {"intent": "confirm"}, vid="u1+0", ctx=pending_ctx(), final=True)
    assert d.verdict == "wait" and "new utterance" in d.reason
    # a later virtual utterance of the same breath whose words were already heard at the prompt
    d = run("yes", {"intent": "confirm"}, vid="u1+1", cursor=1, ctx=pending_ctx(), final=True)
    assert d.verdict == "wait"
    # ... but words spoken after the prompt in the same physical utterance may confirm
    d = run("yes", {"intent": "confirm"}, vid="u1+1", cursor=2, ctx=pending_ctx(), final=True)
    assert d.verdict == "act" and d.action.kind == ActionKind.CONFIRM


def test_confident_mishear_does_not_confirm():
    d = run("know", {"intent": {"confirm": 0.97}}, vid="u2+0", ctx=pending_ctx(), final=True)
    assert d.verdict == "clarify"


def test_negation_cancels_even_if_model_says_confirm():
    d = run("no don't", {"intent": {"confirm": 0.9}}, vid="u2+0", ctx=pending_ctx())
    assert d.verdict == "act" and d.action.kind == ActionKind.CANCEL
    d = run("cancel", {"intent": "cancel"}, vid="u1+0", ctx=pending_ctx())
    assert d.action.kind == ActionKind.CANCEL  # cancelling from the same utterance is fine


def test_weak_or_unsettled_confirm_waits():
    d = run("yes", {"intent": {"confirm": 0.7, "none": 0.3}}, vid="u2+0", ctx=pending_ctx(), final=True)
    assert d.verdict == "wait"
    d = run("yes", {"intent": "confirm", "complete": 0.6}, vid="u2+0", ctx=pending_ctx())
    assert d.verdict == "wait"


def test_other_command_while_pending_falls_through():
    d = run("scroll down", {"intent": "scroll_down"}, vid="u2+0", ctx=pending_ctx())
    assert d.verdict == "act" and d.action.kind == ActionKind.SCROLL_DOWN
