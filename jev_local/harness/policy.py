"""Turn one System One response into a verdict for the current tail (SPEC §4.3 step 7, §2.5).

Pure function, no I/O. The order of the gates matters and follows the contract:

1. a pending confirmation is answered first (confirm / cancel);
2. `is_command` below threshold -> ignore (side talk);
3. intent none -> ignore; wait or low confidence -> wait;
4. `complete` gate, passed by the model, by 900 ms of silence, or by a final transcript;
5. payload gate: type_text / search_web only on a final transcript or 600 ms of silence, because
   a partial payload ("type hello wor") must never be typed; open_url likewise, unless the spoken
   address is already closed by a TLD and a chain word ("... dot org and ..."), because
   "go to wikipedia" may still become "wikipedia dot org" or "wikipedia dot org slash wiki";
6. stability: a closed-set act on a partial also needs `complete >= stable_complete`, or the same
   intent+argument on the previous evaluation of the same virtual utterance;
7. argument gates per intent (a failing gate -> clarify);
8. build the `Action`, whose confidence is the min of the judgements used.

Safety (risk classes, deny lists, the destructive noul) is applied afterwards by `safety.gate`.
A stale answer (computed on text that has since grown) is evaluated as a partial with no silence,
so it may commit a closed-set command but never a payload or a silence-gated one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from jev_local.harness.catalog import FOLDERS, KEYS, PAYLOAD_INTENTS
from jev_local.harness.questions import (
    NONE,
    Q_APP,
    Q_COMPLETE,
    Q_FOLDER,
    Q_INTENT,
    Q_IS_COMMAND,
    Q_KEY,
    Q_SCROLL,
    Q_TARGET,
    Q_TEXT,
    Q_URL,
)
from jev_local.harness.safety import AFFIRM_RE, NEGATE_RE, confirmation_allowed
from jev_local.harness.spans import _TLDS, consumed_for
from jev_local.harness.types import Action, ActionKind, ActionRecord, Decision, Snapshot, Tail, Thresholds
from jev_local.schema import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneResponse

Pick = tuple[str, str, str | None]  # (vid, intent, argument) for the stability rule



# A spoken URL followed by a chain word: the address is finished even though the sentence goes on.
_URL_CLOSED_RE = re.compile(r"(?:\bdot\s+|\.)(?:" + _TLDS + r")\b[^\s]*\s+(?:and|then)\b", re.IGNORECASE)

_BARE_DOMAIN_RE = re.compile(r"^(?:[a-z0-9-]+\.)+[a-z]{2,}(?::\d{1,5})?(?:/\S*)?$", re.IGNORECASE)

# The cancel intent is the safe direction, so its bar is lower than any acting intent's.
CANCEL_TOP_P = 0.5


@dataclass
class PolicyContext:
    prev_pick: Pick | None = None  # top pick of the previous evaluation (any vid; compared with vid)
    pending: Action | None = None  # a HIGH-risk action waiting for "yes"
    pending_mark: tuple[str, int, float] | None = None  # (uid, words heard, time) when `pending` was proposed
    tail_heard_at: float | None = None  # when the tail's first word was heard (same clock as the mark)
    stale: bool = False  # the tail grew while the request was in flight
    history: Sequence[ActionRecord] = ()


@dataclass(frozen=True)
class ChoicePick:
    label: str
    p: float  # raw probability of the chosen label
    conf: float  # Jev choice confidence (normalised by the number of options)


# ---------------------------------------------------------------- reading answers


def choice(resp: SystemOneResponse | None, qid: str) -> ChoicePick | None:
    a = resp.answers.get(qid) if resp is not None else None
    if not isinstance(a, ChoiceAnswer):
        return None
    return ChoicePick(a.choice, float(a.probabilities.get(a.choice, 0.0)), float(a.confidence))


def noul(resp: SystemOneResponse | None, qid: str, default: float) -> float:
    a = resp.answers.get(qid) if resp is not None else None
    return float(a.noul) if isinstance(a, NoulAnswer) else default


def choice_prob(resp: SystemOneResponse | None, qid: str, label: str | None) -> float:
    a = resp.answers.get(qid) if resp is not None else None
    if not isinstance(a, ChoiceAnswer) or label is None:
        return 0.0
    return float(a.probabilities.get(label, 0.0))


def score_level(resp: SystemOneResponse | None, qid: str, default: int) -> int:
    a = resp.answers.get(qid) if resp is not None else None
    if not isinstance(a, ScoreAnswer) or not a.probabilities:
        return default
    best = max(a.probabilities.items(), key=lambda kv: (kv[1], -int(kv[0])))
    return int(best[0])


def pick_of(action: Action | None) -> Pick | None:
    """The (vid, intent, argument) identity of an action, used by the stability rule."""
    if action is None:
        return None
    arg = action.target_eid or action.app or action.text or action.key or action.url or action.folder
    if arg is None and action.amount is not None:
        arg = str(action.amount)
    return (action.source_vid, action.kind.value, arg)


# ---------------------------------------------------------------- policy


def _d(verdict: str, reason: str, resp: SystemOneResponse, action: Action | None = None, retry: int | None = None) -> Decision:
    return Decision(verdict=verdict, action=action, reason=reason, seq=0, retry_in_ms=retry, answers=resp)  # type: ignore[arg-type]


def evaluate_policy(
    resp: SystemOneResponse,
    tail: Tail,
    snap: Snapshot | None,
    ctx: PolicyContext,
    T: Thresholds,
) -> Decision:
    # Stale answers never pass the final/silence gates (SPEC §4.3 step 6).
    is_final = tail.is_final and not ctx.stale
    silent = 0 if ctx.stale else tail.silent_ms
    settled = is_final or silent >= T.payload_silence_ms

    intent = choice(resp, Q_INTENT)
    if intent is None:
        return _d("wait", "no intent answer", resp)

    # 1. pending confirmation
    if ctx.pending is not None:
        d = _pending(resp, tail, intent, ctx, T, settled)
        if d is not None:
            return d

    # 2. side talk
    if noul(resp, Q_IS_COMMAND, 1.0) < T.is_command:
        return _d("ignore", "not a command (is_command low)", resp)

    # 3. intent none / wait / low
    if intent.label == NONE:
        return _d("ignore", "intent none", resp)
    if intent.label == "wait":
        return _d("wait", "intent wait: words do not commit to an action yet", resp)
    if intent.conf < T.intent_conf or intent.p < T.intent_top_p:
        return _d("wait", f"intent {intent.label} low (p={intent.p:.2f}, conf={intent.conf:.2f})", resp)
    if intent.label in (ActionKind.CONFIRM.value, ActionKind.CANCEL.value):
        return _d("ignore", f"{intent.label} with nothing pending", resp)
    try:
        kind = ActionKind(intent.label)
    except ValueError:
        return _d("ignore", f"unknown intent {intent.label!r}", resp)

    # 4. complete gate
    complete = noul(resp, Q_COMPLETE, 0.0)
    if complete < T.complete and not (is_final or silent >= T.silence_complete_ms):
        return _d("wait", f"incomplete (complete={complete:.2f})", resp, retry=max(0, T.silence_complete_ms - silent))

    # 5. payload gate
    if intent.label in PAYLOAD_INTENTS and not (is_final or silent >= T.payload_silence_ms):
        return _d("wait", f"{intent.label} waits for the final transcript or silence", resp,
                  retry=max(0, T.payload_silence_ms - silent))
    if (intent.label == ActionKind.OPEN_URL.value and not (is_final or silent >= T.payload_silence_ms)
            and not _URL_CLOSED_RE.search(tail.text)):
        return _d("wait", "open_url waits for the address to finish (final, silence, or 'and')", resp,
                  retry=max(0, T.payload_silence_ms - silent))

    # 7./8. argument gates and the action
    built = _build(kind, intent, resp, tail, snap, T)
    if isinstance(built, str):
        # Mid-sentence, more words may still resolve the argument ("click the ... send button"),
        # so only a finished phrase asks the user to clarify.
        if is_final or silent >= T.silence_complete_ms:
            return _d("clarify", built, resp)
        return _d("wait", f"argument not resolved yet: {built}", resp)
    action = built

    # 6. stability (checked after building, because it compares the argument too)
    if not settled and complete < T.stable_complete and ctx.prev_pick != pick_of(action):
        return _d("wait", f"unstable: {action.describe()} needs a second agreeing partial "
                  f"(complete={complete:.2f})", resp, action=action)

    return _d("act", f"{action.describe()} (conf={action.confidence:.2f})", resp, action=action)


def _pending(
    resp: SystemOneResponse, tail: Tail, intent: ChoicePick, ctx: PolicyContext, T: Thresholds, settled: bool
) -> Decision | None:
    """Confirm or cancel the pending action; None falls through to the normal policy."""
    pending = ctx.pending
    assert pending is not None
    text = tail.text
    negated = NEGATE_RE.search(text) is not None
    if (intent.label == ActionKind.CANCEL.value and intent.p >= CANCEL_TOP_P) or (
        negated and intent.label in ("cancel", "confirm", "none", "wait")
    ):
        n = min(len(tail.words), consumed_for(tail.words, "cancel"))
        a = Action(ActionKind.CANCEL, source_vid=tail.vid, confidence=intent.conf, consumed_words=n)
        return _d("act", f"cancel pending {pending.describe()}", resp, action=a)
    if intent.label != ActionKind.CONFIRM.value:
        return None
    if not confirmation_allowed(pending, tail, ctx.pending_mark, ctx.tail_heard_at):
        return _d("wait", "a confirmation must come from a new utterance, after the prompt", resp)
    if not AFFIRM_RE.search(text):
        # A confident mis-hear ("know" for "no") must not confirm: require an explicit yes-word.
        return _d("clarify", "say \"confirm\" (or yes) to confirm, or \"cancel\"", resp)
    if intent.p < T.confirm_p:
        return _d("wait", f"confirm too weak (p={intent.p:.2f})", resp)
    if not settled and noul(resp, Q_COMPLETE, 0.0) < T.stable_complete:
        return _d("wait", "confirm waits for a finished phrase", resp)
    n = min(len(tail.words), consumed_for(tail.words, "confirm"))
    a = Action(ActionKind.CONFIRM, source_vid=tail.vid, confidence=intent.conf, consumed_words=n)
    return _d("act", f"confirm {pending.describe()}", resp, action=a)


def _gate_choice(resp: SystemOneResponse, qid: str, top_p: float) -> ChoicePick | None:
    c = choice(resp, qid)
    if c is None or c.label == NONE or c.p < top_p:
        return None
    return c


def _build(
    kind: ActionKind, intent: ChoicePick, resp: SystemOneResponse, tail: Tail, snap: Snapshot | None, T: Thresholds
) -> Action | str:
    """The action with its arguments, or a clarify reason."""
    conf = intent.conf
    span: str | None = None
    kw: dict = {}
    if kind in (ActionKind.OPEN_APP, ActionKind.QUIT_APP):
        a = _gate_choice(resp, Q_APP, T.app_top_p)
        if a is None:
            return "which app?"
        kw["app"] = a.label
        conf = min(conf, a.conf)
    elif kind == ActionKind.CLICK:
        if snap is None or not snap.elements:
            return "no screen elements to click"
        t = _gate_choice(resp, Q_TARGET, T.target_top_p)
        if t is None or t.conf < T.target_conf:
            return "which element?"
        el = next((e for e in snap.elements if e.eid == t.label), None)
        if el is None:
            return f"element {t.label} is not on screen"
        if not el.enabled:
            return f'{el.role} "{el.label}" is disabled'
        kw["target_eid"] = el.eid
        kw["target_label"] = f'{el.role} "{el.label}"' if el.label else el.role
        conf = min(conf, t.conf)
    elif kind.value in PAYLOAD_INTENTS:
        s = _gate_choice(resp, Q_TEXT, T.span_top_p)
        if s is None:
            return "what text?"
        if s.label not in tail.text:
            # Select, don't generate: whatever is typed must be the user's own words.
            return "text span is not verbatim from the transcript"
        kw["text"] = span = s.label
        conf = min(conf, s.conf)
    elif kind == ActionKind.OPEN_URL:
        u = _gate_choice(resp, Q_URL, T.span_top_p)
        if u is None:
            return "which website?"
        # Only a bare domain gets a scheme; anything else ("javascript:...") is left for safety to deny.
        kw["url"] = f"https://{u.label}" if _BARE_DOMAIN_RE.match(u.label) else u.label
        conf = min(conf, u.conf)
    elif kind == ActionKind.PRESS_KEY:
        k = _gate_choice(resp, Q_KEY, T.key_top_p)
        if k is None or k.label not in KEYS:
            return "which key?"
        kw["key"] = k.label
        conf = min(conf, k.conf)
    elif kind == ActionKind.OPEN_FOLDER:
        f = _gate_choice(resp, Q_FOLDER, T.folder_top_p)
        if f is None or f.label not in FOLDERS:
            return "which folder?"
        kw["folder"] = f.label
        conf = min(conf, f.conf)
    elif kind in (ActionKind.SCROLL_DOWN, ActionKind.SCROLL_UP):
        kw["amount"] = score_level(resp, Q_SCROLL, 1)
    # Clamped: spans.consumed_for overshoots by one when the tail ends in a chain word ("open notes and").
    n = min(len(tail.words), consumed_for(tail.words, kind.value, span))
    return Action(kind, source_vid=tail.vid, confidence=round(conf, 4), consumed_words=n, **kw)


__all__ = [
    "ChoicePick",
    "PolicyContext",
    "Pick",
    "choice",
    "choice_prob",
    "evaluate_policy",
    "noul",
    "pick_of",
    "score_level",
]
