"""The transcript stream: a consumed-prefix cursor that makes mid-sentence acting safe.

SPEC §4.3 steps 1–2. Streaming STT keeps revising the words of one physical utterance, and the
controller acts on those words *before* they are final. Three rules stop anything from running
twice:

1. Once a command's words are executed they are **consumed**: the stream remembers their
   normalised text (case, punctuation, filler and chain words ignored). Later partials must still start
   with that prefix. If one does not, the recognizer rewrote words we already acted on. That is
   logged as `revised_after_act` and the utterance is frozen, because the action stands (the user
   can say "undo") and the rest of a rewritten utterance can't be trusted to line up.
2. Each consume starts a new **virtual utterance** `<uid>+<gen>`, whose tail holds only the words
   after the cursor, with leading chain and filler words ("and", "then", "um") removed.
3. A **fired set** of `(vid, cursor, n)` keys. The controller checks it before executing, so a
   decision computed twice for the same words (a coalesced or stale request) runs at most once.

Recognizers sometimes restart an utterance under a new uid and re-emit words already heard, for
example a final with a fresh id, or a restart after a pause without a final. A new uid that
repeats the consumed prefix of the previous one within a short window inherits that consumed
prefix (logged as `uid_echo`), so the echoed command does not run again.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any, Callable, Hashable, Sequence

from jev_local.harness.catalog import CHAIN_WORDS, FILLERS
from jev_local.harness.spans import strip_leading_chain
from jev_local.harness.types import Tail, TranscriptEvent

LogFn = Callable[..., None]

_NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)
_CHAIN_SINGLE = frozenset(w for w in CHAIN_WORDS if " " not in w)
_CHAIN_PAIRS = frozenset(tuple(w.split()) for w in CHAIN_WORDS if " " in w)
# Words the consumed-prefix comparison skips wherever they occur. Recognizers insert and drop
# fillers and chain words between revisions ("go back and open" <-> "go back open"); treating that
# as a rewrite would let a re-sent final under a fresh uid replay commands that already ran.
_IGNORABLE = FILLERS | _CHAIN_SINGLE

# Result of matching a new text against the consumed prefix when the text is still shorter than it
# (the recognizer briefly retracted words). There is nothing new to decide on; not a revision.
_SHORT = -1


def norm_word(w: str) -> str:
    """'Notes,' -> 'notes'; "don't" -> 'dont'; '--' -> ''."""
    return _NON_WORD_RE.sub("", w.lower())


def core_words(words: Sequence[str]) -> list[str]:
    """Normalised words without punctuation-only tokens, fillers or chain words (the comparison form)."""
    return [n for n in (norm_word(w) for w in words) if n and n not in _IGNORABLE]


def _required(core: list[str]) -> list[str]:
    """Drop a trailing two-word chain phrase ("after that"): consumed, but not needed to match."""
    r = list(core)
    while len(r) >= 2 and (r[-2], r[-1]) in _CHAIN_PAIRS:
        del r[-2:]
    return r


def match_prefix(words: Sequence[str], required: Sequence[str]) -> int | None:
    """Raw index in `words` just past `required` (ignorable words skipped). None = mismatch, -1 = too short."""
    j = 0
    for i, w in enumerate(words):
        if j == len(required):
            return i
        n = norm_word(w)
        if not n or n in _IGNORABLE:
            continue
        if n != required[j]:
            return None
        j += 1
    return len(words) if j == len(required) else _SHORT


def _skip_leading(words: Sequence[str]) -> int:
    """Leading chain/filler words plus punctuation-only tokens (',' '-')."""
    i = 0
    while i < len(words):
        k = strip_leading_chain(list(words[i:]))
        if k:
            i += k
        elif not norm_word(words[i]):
            i += 1
        else:
            break
    return min(i, len(words))  # strip_leading_chain overshoots by one on a lone trailing chain word


class Stream:
    """Per-utterance cursor over streaming transcripts. Not thread-safe; owned by the controller loop."""

    def __init__(
        self,
        log: LogFn | None = None,
        echo_window_s: float = 2.0,
        dup_window_s: float = 0.35,
    ):
        # A new uid repeating the consumed words of the previous one is an echo if the previous one
        # never got a final and the gap is short (a recognizer restart), or if the gap is so short
        # that no person could have started a new sentence (a duplicate final under a fresh id).
        self.echo_window_s = echo_window_s
        self.dup_window_s = dup_window_s
        self._log_fn = log
        self.revisions: list[dict[str, Any]] = []  # revised_after_act records, for tests and the HUD
        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        self.uid: str = ""
        self.gen = 0
        self.words: list[str] = []
        self._word_t: list[float] = []  # when each word position was first heard with its current text
        self.cursor = 0  # raw index into self.words: everything before it is consumed
        self.final = False
        self.joined = False  # no pause was observed before this physical utterance (TranscriptEvent.joined)
        self.frozen = False  # a revision rewrote consumed words; lifts if the recognizer agrees again
        self.dropped = False  # the rest of this physical utterance is ignored (after a deny)
        self.last_change_t = 0.0
        self.last_event_t = 0.0
        self.changed = False  # did the last update change the words?
        self._consumed_core: list[str] = []
        self._required: list[str] = []
        self._carried = False  # consumed prefix inherited from the previous uid, not yet confirmed
        self._carried_dropped = False  # ... from an utterance that was dropped (deny / risky proposal)
        self._fired: set[Hashable] = set()
        self._closed: deque[str] = deque(maxlen=64)

    @property
    def vid(self) -> str:
        return f"{self.uid}+{self.gen}"

    @property
    def heard_words(self) -> int:
        """Words heard so far in the current physical utterance."""
        return len(self.words)

    @property
    def consumed_text(self) -> str:
        return " ".join(self._consumed_core)

    # ------------------------------------------------------------------ updates

    def update(self, ev: TranscriptEvent, now: float) -> Tail | None:
        """Feed one recognizer event. Returns the unconsumed tail, or None when there is nothing to decide."""
        self.changed = False
        if ev.kind not in ("partial", "final"):
            return None
        if ev.uid != self.uid:
            if ev.uid in self._closed:
                self._log("late_event", uid=ev.uid, current=self.uid, event=ev.kind, text=ev.text)
                return None
            self._begin(ev.uid, ev.text.split(), now)
        self.joined = self.joined or ev.joined
        self.last_event_t = now
        if self.dropped:
            return None

        words = ev.text.split()
        pos = match_prefix(words, self._required)
        if pos is None and self._carried and self.gen == 0:
            # Not an echo after all: the new utterance merely started differently. Forget the
            # inherited prefix; this is fresh speech.
            self._log("echo_rejected", uid=self.uid, consumed=self.consumed_text, text=ev.text)
            self._consumed_core, self._required, self._carried = [], [], False
            pos = 0
        if pos is None:
            if not self.frozen:
                rec = {"uid": self.uid, "consumed": self.consumed_text, "text": ev.text, "t": now}
                self.revisions.append(rec)
                self._log("revised_after_act", **rec)
            self.frozen = True
            return None
        if pos == _SHORT:
            return None  # a transient retraction below the consumed prefix: nothing new yet
        if self._carried and self._required:
            self._carried = False
            self._log("uid_echo", uid=self.uid, consumed=self.consumed_text, text=ev.text)
            if self._carried_dropped:
                self.dropped = True  # an echo of a dropped utterance stays dropped
                return None
        self.frozen = False  # a flip-flopping recognizer can come back into agreement

        if words != self.words:
            self.changed = True
            self.last_change_t = now
            self._word_t = [
                self._word_t[i] if i < len(self._word_t) and norm_word(self.words[i]) == norm_word(w) else now
                for i, w in enumerate(words)
            ]
        self.words = words
        self.cursor = max(pos, 0)
        if ev.kind == "final":
            self.final = True
        return self.current(now)

    def _begin(self, uid: str, words: list[str], now: float) -> None:
        carry = False
        if self.uid and self._required:
            gap = now - self.last_event_t
            echo_ok = (not self.final and gap <= self.echo_window_s) or gap <= self.dup_window_s
            carry = echo_ok and match_prefix(words, self._required) is not None
        if self.uid:
            self._closed.append(self.uid)
        consumed, required = (self._consumed_core, self._required) if carry else ([], [])
        self._carried_dropped = carry and self.dropped
        self.uid, self.gen = uid, 0
        self.words, self._word_t, self.cursor = [], [], 0
        self.final = self.frozen = self.dropped = self.joined = False
        self._consumed_core, self._required, self._carried = consumed, required, carry
        self.last_change_t = now

    # ------------------------------------------------------------------ reading

    def _tail_start(self) -> int:
        return self.cursor + _skip_leading(self.words[self.cursor :])

    def current(self, now: float) -> Tail | None:
        """The unconsumed tail as of `now` (silent_ms = time since the words last changed)."""
        if not self.uid or self.frozen or self.dropped:
            return None
        start = self._tail_start()
        tw = tuple(self.words[start:])
        if not tw:
            return None
        return Tail(
            vid=self.vid,
            text=" ".join(tw),
            words=tw,
            cursor=start,
            is_final=self.final,
            silent_ms=max(0, int(round((now - self.last_change_t) * 1000))),
            uid=self.uid,
            joined=self.joined,
        )

    def heard_at(self, index: int) -> float | None:
        """When the word at `index` of the current physical utterance was first heard."""
        return self._word_t[index] if 0 <= index < len(self._word_t) else None

    def matches(self, tail: Tail, n_words: int) -> bool:
        """Does the current tail still begin with `tail.words[:n_words]` at the same position?

        The controller checks this before consuming for a decision made on an older tail (a stale
        request), so consumption always lines up with the words the decision was about.
        """
        if self.frozen or self.dropped or tail.vid != self.vid or tail.cursor != self._tail_start():
            return False
        now_words = self.words[tail.cursor : tail.cursor + n_words]
        return core_words(now_words) == core_words(tail.words[:n_words]) and len(now_words) == n_words

    # ------------------------------------------------------------------ consuming

    def consume(self, n_words: int) -> None:
        """Mark the first `n_words` of the current tail as done and start the next virtual utterance."""
        start = self._tail_start()
        end = min(len(self.words), start + max(0, n_words))
        self._consume_to(end)

    def skip_to_end(self) -> None:
        """Consume everything heard so far (cancel): only words spoken after this can act."""
        self._consume_to(len(self.words))

    def _consume_to(self, end: int) -> None:
        self._consumed_core = core_words(self.words[:end])
        self._required = _required(self._consumed_core)
        self._carried = False
        self.cursor = end
        self.gen += 1
        self._log("consume", uid=self.uid, vid=self.vid, cursor=end, consumed=self.consumed_text)

    def drop_utterance(self) -> None:
        """Ignore the rest of this physical utterance (after a deny): a chain never continues past it."""
        self.skip_to_end()
        self.dropped = True

    # ------------------------------------------------------------------ fired set

    def mark_fired(self, key: Hashable) -> None:
        self._fired.add(key)

    def already_fired(self, key: Hashable) -> bool:
        return key in self._fired

    # ------------------------------------------------------------------ misc

    def _log(self, kind: str, **fields: Any) -> None:
        if self._log_fn is not None:
            self._log_fn(kind, **fields)
