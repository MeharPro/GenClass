"""The incremental decision loop (SPEC §4.3): act mid-sentence, never twice, risky things only on "yes".

One `Controller` owns one `Stream` and runs on one asyncio loop:

    event -> Stream.update -> debounce (120 ms partial, 0 final) -> decide
    decide = snapshot (cached) -> build state+questions -> DecisionClient -> stale check
           -> policy -> safety.gate -> execute on a worker thread -> consume -> re-decide the rest

Concurrency: `decide` holds a lock, so at most one decision is in flight. While it runs, newer
triggers collapse into one coalesced "latest" request (`_want`) that runs next. Each request has
a monotonically increasing `seq`, and a result applies only if `seq > applied_seq`. A result
computed on a tail that has since grown is *stale*: it may still commit a closed-set command,
provided the words it decided on are still there, but never a payload or a silence-gated one
(policy.py).

Timers: silence retries at 600 ms (payload gate) and 900 ms (complete gate) after the words last
changed; a pending confirmation expires after `confirm_timeout_ms`.

Sources that can only change the transcript once per step (whisper-stream: one window per 500 ms
step) advertise `cadence_ms`. "No new word for 600 ms" then only means "one window without a new
word", so the silence gates are raised to `silence_gates_for(cadence_ms)` (two windows plus
jitter) and the partial debounce is dropped (one partial per step: nothing to coalesce).

`halt()` is the kill switch: unlike `cancel_all()` (Escape: keep listening), nothing is decided or
executed afterwards, even for transcript events that were already in flight.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Callable, Protocol, Sequence

from jev_local.harness.catalog import CHAIN_WORDS, PAYLOAD_INTENTS
from jev_local.harness.log import AuditLog
from jev_local.harness.policy import PolicyContext, choice, evaluate_policy, pick_of
from jev_local.harness.questions import NONE, Q_INTENT, build_questions, rank_apps
from jev_local.harness.safety import RateLimiter, gate
from jev_local.harness.spans import consumed_for, extract_text_candidates, extract_url_candidates
from jev_local.harness.state import build_state
from jev_local.harness.stream import Stream, core_words, norm_word
from jev_local.harness.types import (
    Action,
    ActionKind,
    ActionRecord,
    Decision,
    ExecResult,
    HarnessConfig,
    Snapshot,
    Tail,
    TranscriptEvent,
)
from jev_local.schema import SystemOneResponse

MAX_LOOPS = 4  # decisions per trigger: one breath can complete several chained commands
MAX_DEBOUNCE_MS = 300  # a burst of partials still gets a decision at least this often
SILENCE_MARGIN_MS = 15  # fire silence timers just past the threshold, so silent_ms >= threshold
RATE_MAX, RATE_WINDOW_S = 3, 1.0
UNDO_DEPTH = 20
# Used when the caller supplies no app names. The app Choice needs >= 2 options, and deny-listed
# apps are included on purpose so "open terminal" resolves (and is then denied) instead of guessing.
DEFAULT_APPS = (
    "Safari", "Notes", "Mail", "Messages", "Finder", "Calendar", "Reminders", "Music", "Photos", "Maps",
    "Preview", "TextEdit", "Google Chrome", "Slack", "Spotify", "Visual Studio Code", "System Settings", "Terminal",
)


def silence_gates_for(cadence_ms: float) -> tuple[int, int]:
    """(payload_silence_ms, silence_complete_ms) for a source that updates every `cadence_ms`:
    silence must span two whole windows plus inference jitter, never just one late window."""
    payload = int(round(2 * cadence_ms + 150))
    return payload, max(payload, 900)


class SpeechSource(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def events(self) -> Any: ...  # AsyncIterator[TranscriptEvent]


_CHAIN_SINGLE = frozenset(w for w in CHAIN_WORDS if " " not in w)


def command_of(a: Action) -> tuple:
    """What an action does, ignoring which words it came from and the scroll amount (a modifier)."""
    return (a.kind, a.target_eid, a.app, a.text, a.key, a.url, a.folder)


# Words that can finish a command's phrase after the command already fired on its first words
# ("open notes" + "app", "scroll down" + "a little more"). Kept per kind and deliberately small:
# absorbing one wrongly loses a word; it can never run anything. Payload commands have none,
# because what follows dictated text is more dictation.
_POLITE = frozenset("please now right away for me thanks thank you real quick quickly again".split())
_SCROLL_MODS = frozenset("a little bit lot few lines more some all the way to top bottom of further".split())
_ROLE_NOUNS = frozenset("button link icon item option box field checkbox row tab menu".split())
LEFTOVER_WORDS: dict[ActionKind, frozenset[str]] = {
    ActionKind.OPEN_APP: frozenset({"app", "application", "program"}),
    ActionKind.QUIT_APP: frozenset({"app", "application", "program"}),
    ActionKind.OPEN_FOLDER: frozenset({"folder", "directory", "window"}),
    ActionKind.CLICK: _ROLE_NOUNS,
    ActionKind.PRESS_KEY: frozenset({"key", "keys", "button"}),
    ActionKind.SCROLL_DOWN: _SCROLL_MODS,
    ActionKind.SCROLL_UP: _SCROLL_MODS,
    ActionKind.GO_BACK: frozenset({"a", "to", "the", "previous", "one"}),
    ActionKind.NEW_TAB: frozenset({"tab", "window"}),
    ActionKind.CLOSE_TAB: frozenset({"tab", "window"}),
    ActionKind.OPEN_URL: frozenset({"website", "site", "dot", "com", "org", "net", "io"}),
    ActionKind.UNDO: frozenset({"that", "it", "this", "last", "the", "thing"}),
}


def leftover_words(a: Action) -> frozenset[str]:
    if a.kind.value in PAYLOAD_INTENTS:
        return frozenset()
    words = LEFTOVER_WORDS.get(a.kind, frozenset()) | _POLITE
    if a.target_label:  # 'button "Archive"': "click archive" + "button"
        words |= frozenset(core_words(a.target_label.replace('"', " ").split()))
    return words


@dataclass
class _LastAct:
    """The last executed command of the current physical utterance, for the leftover/duplicate rules.

    Acting mid-sentence consumes a command before the speaker has finished its phrase, so the next
    tail can start with the rest of that phrase ("open the downloads" + "folder and scroll down",
    "open notes" + "app", "click archive" + "button and ..."). Those leftover words are not a new
    command. These rules keep them from blocking the chain or repeating the command:

    - absorb: right after a mid-phrase consume, leading words from the command's small
      `leftover_words` set are consumed once a word outside it follows ("app type hi" -> "type hi");
    - leftover: in that position, a decision for the *same* command is the rest of that phrase: its
      words are consumed without executing again; and a head the model calls not-a-command / wait,
      followed by a chain word and more words, is skipped;
    - duplicate: `consumed_for` assumes the tail starts with the command. When a leftover head
      precedes it ("button and press enter" -> press_key), the action consumes only the head, and
      the re-decision on "press enter" would run it again. A same-command decision on words the
      previous decision already saw, and that its own consumed words do not account for, is that
      repeat: consumed, not executed.
    """

    uid: str
    vid_after: str  # the virtual utterance the consume started
    command: tuple
    consumed_core: list[str]  # core words the command consumed
    remainder_core: list[str]  # core words already heard after them when it was decided
    mid_phrase: bool  # consumption did not end at a chain word: leftovers may follow
    leftovers: frozenset[str] = frozenset()


def _ends_with_chain(words: Sequence[str]) -> bool:
    if not words:
        return False
    last = norm_word(words[-1])
    two = " ".join(norm_word(w) for w in words[-2:])
    return last in _CHAIN_SINGLE or two in CHAIN_WORDS


class Controller:
    def __init__(
        self,
        source: SpeechSource | None,
        observer: Any,
        client: Any,
        executor: Any,
        cfg: HarnessConfig,
        hud: Any = None,
        apps_provider: Callable[[str], Sequence[str]] | None = None,
        clock: Callable[[], float] = time.monotonic,
        *,
        log: AuditLog | None = None,
        app_names: Sequence[str] = DEFAULT_APPS,
        running_apps: Callable[[], Sequence[str]] | None = None,
    ):
        self.source = source
        self.observer = observer
        self.client = client
        self.executor = executor
        self.cfg = cfg
        self.hud = hud
        self.apps_provider = apps_provider
        self.clock = clock
        self.log = log if log is not None else AuditLog(redact=cfg.log_redact)
        self.app_names = list(app_names) or list(DEFAULT_APPS)
        self.running_apps = running_apps
        self.stream = Stream(log=self._log)

        self.history: list[ActionRecord] = []
        # Recent (time, tail, decision) for replay reports and debugging; each holds a full response.
        self.decisions: deque[tuple[float, Tail, Decision]] = deque(maxlen=1000)
        self.pending: Action | None = None
        self._pending_snap: Snapshot | None = None
        self._pending_mark: tuple[str, int, float] | None = None
        self._pending_timer: asyncio.TimerHandle | None = None
        self._undo_stack: list[ActionRecord] = []

        self._seq = 0
        self.applied_seq = 0
        self._prev_pick = None
        self._last_key: tuple | None = None
        self._last_act: _LastAct | None = None
        self._lock = asyncio.Lock()
        self._want: str | None = None
        self._runner: asyncio.Task | None = None
        self._debounce: asyncio.TimerHandle | None = None
        self._debounce_since = 0.0
        self._silence: list[asyncio.TimerHandle] = []
        self._retry: asyncio.TimerHandle | None = None  # rate-limit retry
        self._cancel = threading.Event()
        self._rate = RateLimiter(RATE_MAX, RATE_WINDOW_S)
        self._closed = False
        self.halted = False  # kill switch: latched, nothing runs after it
        self.stats: dict[str, int] = {
            "events": 0, "decisions": 0, "engine_calls": 0, "stale_dropped": 0, "stale_applied": 0,
            "same_text_skipped": 0, "executed": 0, "rate_limited": 0, "errors": 0,
        }
        self._log("session_start", dry_run=cfg.dry_run, backend=cfg.backend, model=cfg.model,
                  thresholds=self.T, allow_apps=cfg.allow_apps)

    # ------------------------------------------------------------------ thresholds

    def _cadence_ms(self) -> float | None:
        c = getattr(self.source, "cadence_ms", None)
        try:
            c = c() if callable(c) else c
            return float(c) if c else None
        except Exception:  # noqa: BLE001
            return None

    @property
    def thresholds(self):  # noqa: ANN201 - Thresholds
        """The thresholds in force: the config's, with the silence gates raised to the source cadence."""
        T = self.cfg.thresholds
        c = self._cadence_ms()
        if c is None:
            return T
        payload, complete = silence_gates_for(c)
        if payload <= T.payload_silence_ms and complete <= T.silence_complete_ms and T.debounce_ms == 0:
            return T
        return replace(T, payload_silence_ms=max(T.payload_silence_ms, payload),
                       silence_complete_ms=max(T.silence_complete_ms, complete), debounce_ms=0)

    # ------------------------------------------------------------------ public API

    async def run(self) -> None:
        """Consume the speech source until it ends."""
        assert self.source is not None, "Controller.run needs a speech source"
        await self.source.start()
        try:
            async for ev in self.source.events():
                await self.on_event(ev)
        finally:
            try:
                await self.source.stop()
            finally:
                await self.aclose()

    @property
    def T(self):  # noqa: ANN201, N802 - Thresholds
        return self.thresholds

    async def on_event(self, ev: TranscriptEvent) -> None:
        now = self.clock()
        self.stats["events"] += 1
        if self.halted:
            self._log("transcript_after_halt", event=ev.kind, uid=ev.uid, text=ev.text)
            return
        self._log("transcript", event=ev.kind, uid=ev.uid, seq=ev.seq, text=ev.text, error=ev.error)
        if ev.kind == "speech_start":
            self._call(self.observer, "prefetch")  # walk the AX tree while the user is still talking
            return
        if ev.kind == "error":
            self._hud("error", f"speech: {ev.error or ev.text}")
            return
        if ev.kind == "speech_end":
            return
        tail = self.stream.update(ev, now)
        if tail is None:
            return
        self._hud("transcript", tail)
        if self.stream.changed:
            self._arm_silence()
        if ev.kind == "final":
            self._clear_debounce()
            self._request("final")
        else:
            self._debounce_request("partial", self.T.debounce_ms)

    async def decide(self, reason: str = "partial") -> Decision | None:
        """One decision on the current tail, executing and re-deciding on the rest up to MAX_LOOPS times.

        Returns the first decision made (None when there was nothing new to decide on).
        """
        async with self._lock:
            first: Decision | None = None
            r = reason
            for _ in range(MAX_LOOPS):
                d, again = await self._decide_once(r)
                if first is None:
                    first = d
                if not again:
                    break
                r = "post_action"
            return first

    def cancel_all(self) -> None:
        """Escape / kill switch: drop queued decisions, abort the executor, forget the pending action,
        and jump the cursor to the end of what has been heard."""
        self._want = None
        self._clear_debounce()
        for h in self._silence:
            h.cancel()
        self._silence.clear()
        self._cancel.set()
        self._cancel = threading.Event()  # the running execution keeps the set one
        self._drop_pending("cancel_all")
        self.stream.skip_to_end()
        self._last_act = None
        self._log("cancel_all")

    def halt(self) -> None:
        """Kill switch: cancel everything and latch. Later events, timers and queued decisions do nothing,
        so words whisper had already written before it was killed can never run."""
        self.halted = True
        self.cancel_all()
        self._cancel.set()  # also refuse any execution that starts from here on
        self._cancel_handle(self._retry)
        self._cancel_handle(self._pending_timer)
        self._log("halt")

    def idle(self) -> bool:
        """Nothing queued, running, or scheduled that could still decide (the pending-confirm timeout
        only expires; it never acts)."""
        loop = asyncio.get_running_loop()
        now = loop.time()
        timers = [self._debounce, self._retry, *self._silence]
        return (
            (self._runner is None or self._runner.done())
            and self._want is None
            and not any(h is not None and not h.cancelled() and h.when() > now for h in timers)
        )

    async def drain(self) -> None:
        """Wait until queued and in-flight decisions have finished (timers may still fire later)."""
        while self._runner is not None and not self._runner.done():
            await asyncio.shield(self._runner)

    async def aclose(self) -> None:
        self._closed = True
        self._clear_debounce()
        self._cancel_handle(self._pending_timer)
        self._cancel_handle(self._retry)
        for h in self._silence:
            h.cancel()
        if self._runner is not None and not self._runner.done():
            try:
                await self._runner
            except Exception:  # noqa: BLE001 - already logged in _drain
                pass
        close = getattr(self.client, "aclose", None)
        if close is not None:
            await close()
        await asyncio.to_thread(self.log.flush, 2.0)

    # ------------------------------------------------------------------ scheduling

    def _request(self, reason: str) -> None:
        """Ask for a decision; coalesces with any already queued (one in flight + one latest)."""
        if self._closed or self.halted:
            return
        self._want = reason
        if self._runner is None or self._runner.done():
            self._runner = asyncio.get_running_loop().create_task(self._drain_wants())

    async def _drain_wants(self) -> None:
        while self._want is not None and not self._closed:
            reason, self._want = self._want, None
            try:
                await self.decide(reason)
            except Exception as e:  # noqa: BLE001 - the loop must survive any single failure
                self.stats["errors"] += 1
                self._log("error", stage="decide", error=repr(e))
                self._hud("error", f"decide failed: {e!r}")

    def _debounce_request(self, reason: str, delay_ms: int) -> None:
        """Trailing debounce, capped: partials faster than the debounce still get a decision every
        MAX_DEBOUNCE_MS instead of starving until the speaker pauses."""
        loop = asyncio.get_running_loop()
        now = loop.time()
        if self._debounce is None:
            self._debounce_since = now
        else:
            self._debounce.cancel()
        fire_at = min(now + delay_ms / 1000.0, self._debounce_since + MAX_DEBOUNCE_MS / 1000.0)
        self._debounce = loop.call_at(fire_at, self._debounce_fire, reason)

    def _clear_debounce(self) -> None:
        self._cancel_handle(self._debounce)
        self._debounce = None

    def _debounce_fire(self, reason: str) -> None:
        self._debounce = None
        self._request(reason)

    def _arm_silence(self) -> None:
        for h in self._silence:
            h.cancel()
        loop = asyncio.get_running_loop()
        self._silence = [
            loop.call_later((ms + SILENCE_MARGIN_MS) / 1000.0, self._request, "silence")
            for ms in sorted({self.T.payload_silence_ms, self.T.silence_complete_ms})
        ]

    @staticmethod
    def _cancel_handle(h: asyncio.TimerHandle | None) -> None:
        if h is not None:
            h.cancel()

    # ------------------------------------------------------------------ one decision

    def _silence_bucket(self, tail: Tail) -> int:
        if tail.is_final:
            return 2  # a final already passes every silence gate: silence timers add nothing
        return (tail.silent_ms >= self.T.payload_silence_ms) + (tail.silent_ms >= self.T.silence_complete_ms)

    async def _decide_once(self, reason: str) -> tuple[Decision | None, bool]:
        now = self.clock()
        tail = self.stream.current(now)
        if tail is not None:
            self._sync_last_act(tail)
            if self._absorb_leftovers(tail):
                tail = self.stream.current(now)
        if tail is None:
            return None, False
        # Same-text cache: the gates only change with the text, finality, a silence threshold, or
        # the pending action, so identical inputs are not re-sent.
        key = (tail.vid, tail.cursor, tail.text, tail.is_final, self._silence_bucket(tail), id(self.pending))
        if key == self._last_key:
            self.stats["same_text_skipped"] += 1
            return None, False
        self._last_key = key
        self._seq += 1
        seq = self._seq

        t0 = time.perf_counter()
        snap = await self._snapshot()
        t_snap = time.perf_counter()
        apps = self._apps(tail.text)
        state = build_state(tail.text, snap, self.history, self.pending)
        questions = build_questions(snap, apps, extract_text_candidates(tail.text),
                                    extract_url_candidates(tail.text), self.cfg.max_elements)
        t_build = time.perf_counter()
        try:
            self.stats["engine_calls"] += 1
            resp: SystemOneResponse = await self.client.system_one(state, questions)
        except Exception as e:  # noqa: BLE001 - engine/HTTP failures skip this decision only
            self.stats["errors"] += 1
            self._last_key = None  # allow a retry on the next trigger
            self._log("error", stage="client", seq=seq, vid=tail.vid, error=repr(e))
            self._hud("error", f"decision failed: {e}")
            return None, False
        t_client = time.perf_counter()

        if seq <= self.applied_seq:
            return None, False
        cur = self.stream.current(self.clock())
        if cur is None or cur.vid != tail.vid or cur.cursor != tail.cursor:
            # The words this answer was about were consumed, cancelled or rewritten meanwhile.
            self.stats["stale_dropped"] += 1
            self._log("stale_dropped", seq=seq, vid=tail.vid, text=tail.text,
                      current=None if cur is None else cur.text)
            return None, False
        stale = cur.text != tail.text

        ctx = PolicyContext(prev_pick=self._prev_pick, pending=self.pending, pending_mark=self._pending_mark,
                            tail_heard_at=self.stream.heard_at(tail.cursor), stale=stale,
                            history=tuple(self.history[-3:]))
        decision = evaluate_policy(resp, tail, snap, ctx, self.T)
        a = decision.action
        if a is None or a.kind not in (ActionKind.CONFIRM, ActionKind.CANCEL):
            self._prev_pick = pick_of(a)
        said = " ".join(tail.words[: a.consumed_words]) if a is not None and a.consumed_words else tail.text
        decision = gate(decision, snap, self.cfg, said=said)
        t_policy = time.perf_counter()
        decision = replace(decision, seq=seq, latency_ms=round((t_client - t_build) * 1000.0, 2))
        self.applied_seq = seq
        self.stats["decisions"] += 1
        if stale:
            self.stats["stale_applied"] += 1
        self.decisions.append((now, tail, decision))
        self._log(
            "decision", seq=seq, trigger=reason, vid=tail.vid, text=tail.text, is_final=tail.is_final,
            silent_ms=tail.silent_ms, stale=stale, verdict=decision.verdict, reason=decision.reason,
            action=decision.action, risk=decision.risk, retry_in_ms=decision.retry_in_ms,
            snapshot=None if snap is None else {"app": snap.app_name, "n": len(snap.elements), "walk_ms": snap.walk_ms},
            timings_ms={"snapshot": _ms(t0, t_snap), "build": _ms(t_snap, t_build),
                        "client": _ms(t_build, t_client), "policy": _ms(t_client, t_policy)},
            answers=resp,
        )
        self._hud("decision", decision)
        return decision, await self._apply(decision, tail, snap, said)

    async def _apply(self, decision: Decision, tail: Tail, snap: Snapshot | None, said: str) -> bool:
        """Carry out a decision. Returns True when words were consumed (re-decide on the rest)."""
        a = decision.action
        v = decision.verdict
        self._sync_last_act(tail)
        if v in ("ignore", "wait"):
            return self._skip_leftover_head(decision, tail)
        if a is None or v not in ("act", "confirm", "deny"):
            return False
        n = a.consumed_words
        key = (tail.vid, tail.cursor, n)
        if self.stream.already_fired(key) or not self.stream.matches(tail, n):
            self._log("not_fired", seq=decision.seq, vid=tail.vid, reason="already fired or words changed")
            return False

        if v in ("act", "confirm") and a.kind not in (ActionKind.CONFIRM, ActionKind.CANCEL):
            why = self._repeat_of_last(a, tail)
            if why is not None:
                # The command already ran: these words belong to it (see _LastAct). Consume, don't re-run.
                self.stream.mark_fired(key)
                self.stream.consume(n)
                self._log(why, seq=decision.seq, vid=tail.vid, action=a, consumed=" ".join(tail.words[:n]))
                self._remember(tail, n, extend=True)
                return True

        if v == "deny":
            self.stream.mark_fired(key)
            self.stream.drop_utterance()  # a denied command ends the chain it was part of
            return False

        if v == "confirm":
            # A risky step ends its sentence: later chained steps ("... and press enter") must never
            # run while an earlier one is unconfirmed, and the "yes" must be a new utterance.
            self.stream.mark_fired(key)
            self._propose(a, snap, tail)
            self.stream.drop_utterance()
            return False

        if a.kind == ActionKind.CANCEL:
            self.stream.mark_fired(key)
            self.stream.consume(n)
            self._last_act = None
            self._drop_pending("cancelled by user")
            return True

        if a.kind == ActionKind.CONFIRM:
            if self.pending is None:
                return False
            # A confirmed action is an execution too: it counts against the rate limit.
            if self._rate_limited(decision, self.pending):
                return False
            self.stream.mark_fired(key)
            self.stream.consume(n)
            self._last_act = None  # the confirmed command's words were in another utterance
            pending, psnap = self.pending, self._pending_snap
            self._clear_pending()
            self._log("confirmed", action=pending, by=tail.vid, said=said)
            await self._execute(pending, psnap, f"(confirmed by: {said})")
            return True

        if self._rate_limited(decision, a):
            return False

        self.stream.mark_fired(key)
        self.stream.consume(n)  # before executing: a failure or a racing event can never re-run it
        self._remember(tail, n, action=a)
        if self.pending is not None:
            self._drop_pending(f"superseded by {a.describe()}")
        await self._execute(a, snap, said)
        return True

    # ------------------------------------------------------------------ leftovers and repeats

    def _sync_last_act(self, tail: Tail) -> None:
        """Forget the last command on a new physical utterance, unless the stream carried its consumed
        prefix over (a recognizer restart re-sending the same words): then it is the same speech."""
        L = self._last_act
        if L is None or tail.uid == L.uid:
            return
        if self.stream.gen == 0 and self.stream.consumed_text and L.vid_after.startswith(f"{L.uid}+"):
            L.uid, L.vid_after = tail.uid, tail.vid
        else:
            self._last_act = None

    def _remember(self, tail: Tail, n: int, *, action: Action | None = None, extend: bool = False) -> None:
        """Record what the consume of `tail.words[:n]` covered. `extend`: the words belong to the command
        already recorded (a leftover or repeat), so they add to its words instead of starting anew."""
        words = tail.words[:n]
        L = self._last_act
        if extend and L is not None:
            L.consumed_core = L.consumed_core + core_words(words)
        elif action is not None:
            L = self._last_act = _LastAct(tail.uid, "", command_of(action), core_words(words), [], False,
                                          leftover_words(action))
        else:
            return
        L.vid_after = self.stream.vid
        L.remainder_core = core_words(tail.words[n:])  # words this decision saw beyond its own
        L.mid_phrase = not _ends_with_chain(words)

    def _absorb_leftovers(self, tail: Tail) -> bool:
        """Consume leading leftover words of the command that just fired mid-phrase, once a word
        that is not one follows them (so the phrase has visibly ended). No model call needed."""
        L = self._last_act
        if L is None or not L.mid_phrase or tail.vid != L.vid_after or not L.leftovers:
            return False
        k = 0
        while k < len(tail.words) and norm_word(tail.words[k]) in L.leftovers:
            k += 1
        if k == 0 or k == len(tail.words):
            return False
        self.stream.consume(k)
        self._log("leftover_absorbed", vid=tail.vid, consumed=" ".join(tail.words[:k]), rule="words")
        L.consumed_core = L.consumed_core + core_words(tail.words[:k])
        L.vid_after, L.remainder_core = self.stream.vid, []  # still right after the command: mid_phrase stays
        return True

    def _repeat_of_last(self, a: Action, tail: Tail) -> str | None:
        """'leftover' / 'duplicate' when this decision is the last command again (see _LastAct)."""
        L = self._last_act
        if L is None or tail.vid != L.vid_after or command_of(a) != L.command:
            return None
        if L.mid_phrase:
            return "leftover_absorbed"
        seen = bool(L.remainder_core) and core_words(tail.words)[: len(L.remainder_core)] == L.remainder_core
        mine = core_words(tail.words[: a.consumed_words])
        if seen and not set(mine) <= set(L.consumed_core):
            return "duplicate_skipped"
        return None  # a genuine repeat ("scroll down and scroll down")

    def _skip_leftover_head(self, decision: Decision, tail: Tail) -> bool:
        """Right after a mid-phrase consume, skip a head the model calls not-a-command (or intent
        `wait`) once a chain word and more words follow it: "folder and scroll down" -> "scroll down"."""
        L = self._last_act
        if L is None or not L.mid_phrase or tail.vid != L.vid_after:
            return False
        if decision.verdict == "wait":
            intent = choice(decision.answers, Q_INTENT)
            if intent is None or intent.label != "wait":
                return False  # e.g. a payload waiting for silence: those words are a command
        n = consumed_for(tail.words, NONE)  # through the first chain word after the head
        if n >= len(tail.words):
            return False  # no chain word yet, or nothing after it
        key = (tail.vid, tail.cursor, n)
        if self.stream.already_fired(key) or not self.stream.matches(tail, n):
            return False
        self.stream.mark_fired(key)
        self.stream.consume(n)
        self._log("leftover_skipped", seq=decision.seq, vid=tail.vid, consumed=" ".join(tail.words[:n]))
        # The rest starts a new clause, and this decision did not pick the last command for it.
        L.vid_after, L.remainder_core, L.mid_phrase = self.stream.vid, [], False
        return True

    def _rate_limited(self, decision: Decision, action: Action) -> bool:
        """True (and a retry is scheduled) when executing now would exceed RATE_MAX per RATE_WINDOW_S.
        Nothing is consumed, so the retry re-decides on the same words."""
        wait_s = self._rate.wait_s(self.clock())
        if wait_s <= 0:
            return False
        self.stats["rate_limited"] += 1
        self._last_key = None
        self._log("rate_limited", seq=decision.seq, action=action, retry_in_ms=int(wait_s * 1000))
        self._cancel_handle(self._retry)
        self._retry = asyncio.get_running_loop().call_later(wait_s + 0.005, self._request, "rate_limit")
        return True

    # ------------------------------------------------------------------ execution

    async def _execute(self, action: Action, snap: Snapshot | None, said: str) -> ExecResult:
        if self.halted:
            self._log("not_executed", action=action, why="halted")
            return ExecResult(ok=False, changed=False, detail="halted", elapsed_ms=0.0, dry_run=self.cfg.dry_run)
        t = self.clock()
        self._rate.record(t)
        t0 = time.perf_counter()
        try:
            if action.kind == ActionKind.UNDO and self._undo_stack:
                rec = self._undo_stack.pop()
                result = await asyncio.to_thread(self.executor.undo, rec)
            else:
                result = await asyncio.to_thread(self.executor.run, action, snap, self._cancel)
        except Exception as e:  # noqa: BLE001 - an executor crash is an outcome, not a harness crash
            self.stats["errors"] += 1
            result = ExecResult(ok=False, changed=False, detail=f"error: {e!r}",
                                elapsed_ms=(time.perf_counter() - t0) * 1000.0, dry_run=self.cfg.dry_run)
        outcome = f"failed: {result.detail}" if not result.ok else ("dry-run" if result.dry_run else "ok")
        rec = ActionRecord(said=said, action=action, outcome=outcome, t=t, undo_token=result.undo_token)
        self.history.append(rec)
        if len(self.history) > 500:
            del self.history[:100]
        if action.kind != ActionKind.UNDO and result.ok:
            self._undo_stack.append(rec)
            del self._undo_stack[:-UNDO_DEPTH]
        self.stats["executed"] += 1
        self._call(self.observer, "invalidate")  # the screen changed (or will): next decision re-walks
        if result.ok and not result.dry_run:
            # Walk the new screen now (in the background), not when the next command arrives: after
            # "open textedit" the first walk of the new app is what the "and type ..." waits for.
            self._call(self.observer, "prefetch")
        self._log("exec", action=action, said=said, outcome=outcome, detail=result.detail,
                  elapsed_ms=result.elapsed_ms, dry_run=result.dry_run)
        self._hud("executed", action, result)
        return result

    # ------------------------------------------------------------------ pending confirmation

    def _propose(self, action: Action, snap: Snapshot | None, tail: Tail) -> None:
        if self.pending is not None:
            self._drop_pending(f"replaced by {action.describe()}")
        self.pending = action
        self._pending_snap = snap
        # Words of this physical utterance heard so far were spoken before the prompt appeared.
        self._pending_mark = (tail.uid, self.stream.heard_words, self.clock())
        self._pending_timer = asyncio.get_running_loop().call_later(
            self.T.confirm_timeout_ms / 1000.0, self._expire_pending, action)
        self._log("pending", action=action, mark=self._pending_mark)
        self._hud("pending", action)

    def _expire_pending(self, action: Action) -> None:
        if self.pending is action:
            self._drop_pending("timeout")

    def _clear_pending(self) -> None:
        self._cancel_handle(self._pending_timer)
        self.pending, self._pending_snap, self._pending_mark, self._pending_timer = None, None, None, None

    def _drop_pending(self, why: str) -> None:
        if self.pending is None:
            return
        self._log("pending_dropped", action=self.pending, why=why)
        self._hud("error", f"not done: {self.pending.describe()} ({why})")
        self._clear_pending()

    # ------------------------------------------------------------------ helpers

    async def _snapshot(self) -> Snapshot | None:
        try:
            return await asyncio.to_thread(self.observer.snapshot, allow_cached=True)
        except Exception as e:  # noqa: BLE001 - PermissionMissing etc.: decide without a screen
            self._log("error", stage="snapshot", error=repr(e))
            return None

    def _apps(self, tail_text: str) -> list[str]:
        if self.apps_provider is not None:
            names = [a for a in self.apps_provider(tail_text) if a]
        else:
            running = list(self.running_apps()) if self.running_apps is not None else []
            names = rank_apps(tail_text, self.app_names, running=running, max_n=self.cfg.max_apps)
        return names or list(DEFAULT_APPS[: self.cfg.max_apps])

    def _log(self, kind: str, **fields: Any) -> None:
        self.log.write(kind, t_mono=self.clock(), **fields)

    def _hud(self, method: str, *args: Any) -> None:
        if self.hud is not None:
            self._call(self.hud, method, *args)

    def _call(self, obj: Any, method: str, *args: Any) -> None:
        fn = getattr(obj, method, None)
        if fn is None:
            return
        try:
            fn(*args)
        except Exception as e:  # noqa: BLE001 - a HUD or prefetch failure must not stop the loop
            self._log("error", stage=method, error=repr(e))


def _ms(a: float, b: float) -> float:
    return round((b - a) * 1000.0, 3)
