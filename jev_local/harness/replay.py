"""Replay word-timed transcripts through a Controller and measure when actions fire.

`replay(script, controller_factory)` plays each `ReplayUtterance` the way a streaming recognizer
delivers it:

    speech_start, then one partial per word every `word_ms` (each adds one word),
    the final `final_delay_ms` after the last word, speech_end, then `pause_ms` of silence.

Explicit `partials` let a script inject STT revisions (re-cased, re-punctuated or rewritten
words) and `final` overrides or suppresses the final. The report says, per utterance, which
actions executed, how long after the utterance's last word each one started (negative = before
the user finished speaking), at which word it fired, and whether it fired **before the final
transcript**, the mid-sentence metric.

Timing runs on the event loop's clock. `run_replay(..., virtual=True)` uses `VirtualTimeLoop`:
timers fire instantly in virtual time, and time freezes while worker threads (engine,
executor) run. Replays are then deterministic and take milliseconds regardless of machine load.
With `virtual=False` it runs in real time, which is what latency measurements should use.
"""

from __future__ import annotations

import asyncio
import json
import selectors
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

from jev_local.harness.types import ActionRecord, Decision, TranscriptEvent

# ---------------------------------------------------------------- virtual time


class _VirtualSelector(selectors.DefaultSelector):  # type: ignore[misc,valid-type]
    loop: "VirtualTimeLoop | None" = None

    def select(self, timeout: float | None = None):  # type: ignore[override]
        events = super().select(0)
        loop = self.loop
        if events or loop is None or timeout == 0:  # timeout 0: callbacks are ready, never block
            return events
        if loop._threads > 0:
            # A worker thread is running: wait for it in real time with the virtual clock frozen,
            # so thread work takes zero virtual time and ordering stays deterministic.
            return super().select(0.05)
        if timeout is None:
            return super().select(None)  # nothing scheduled at all: block on real I/O
        if timeout > 0:
            loop._vtime += timeout
        return []


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    """An event loop whose clock jumps straight to the next timer when there is nothing to do."""

    def __init__(self) -> None:
        sel = _VirtualSelector()
        super().__init__(sel)
        sel.loop = self
        self._vtime = 0.0
        self._threads = 0

    def time(self) -> float:
        return self._vtime

    def run_in_executor(self, executor, func, *args):  # type: ignore[override]
        fut = super().run_in_executor(executor, func, *args)
        self._threads += 1
        fut.add_done_callback(self._thread_done)
        return fut

    def _thread_done(self, _fut: Any) -> None:
        self._threads -= 1

    async def shutdown_default_executor(self, timeout: float | None = None) -> None:
        self._threads += 1  # the joining thread is real work too
        try:
            await super().shutdown_default_executor(timeout)
        finally:
            self._threads -= 1


def run_virtual(coro: Awaitable[Any]) -> Any:
    loop = VirtualTimeLoop()
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
        finally:
            loop.close()


# ---------------------------------------------------------------- script and report


@dataclass
class ReplayUtterance:
    text: str
    # Expected executions, matched in order against actions *started* during this utterance's
    # window, e.g. {"kind": "open_app", "app": "Notes", "before_final": True}. A {"verdict": ...}
    # entry instead requires some decision of this utterance to have that verdict. [] = nothing runs.
    expect: list[dict[str, Any]] | None = None
    partials: list[str] | None = None  # explicit partial texts (STT revisions); default word by word
    final: str | None = None  # final text; default `text`; "" = the recognizer never sends one
    pause_ms: int | None = None  # silence after this utterance; default replay's pause_ms
    word_ms: int | None = None
    uid: str | None = None
    final_uid: str | None = None  # a recognizer that finalizes under a fresh uid


@dataclass
class ActionReport:
    kind: str
    describe: str
    source_vid: str
    t_ms: float  # action start minus the utterance's last word (negative = mid-sentence)
    at_word: int  # words of the utterance heard when it started
    before_final: bool
    outcome: str
    record: ActionRecord


@dataclass
class UtteranceReport:
    index: int
    uid: str
    text: str
    t_start: float
    t_last_word: float
    t_final: float | None
    n_words: int
    actions: list[ActionReport] = field(default_factory=list)
    verdicts: dict[str, int] = field(default_factory=dict)
    decisions: list[Decision] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    marks: list[tuple[float, int]] = field(default_factory=list)  # (partial time, words in it)

    @property
    def ok(self) -> bool:
        return not self.failures


@dataclass
class ReplayReport:
    utterances: list[UtteranceReport]
    word_ms: int
    final_delay_ms: int
    stats: dict[str, int] = field(default_factory=dict)
    revisions: int = 0

    @property
    def actions(self) -> list[ActionReport]:
        return [a for u in self.utterances for a in u.actions]

    @property
    def failures(self) -> list[str]:
        return [f"#{u.index} {u.text!r}: {f}" for u in self.utterances for f in u.failures]

    @property
    def ok(self) -> bool:
        return not self.failures

    def summary(self) -> dict[str, Any]:
        acts = self.actions
        lat = [a.t_ms for a in acts]
        closed = [a for a in acts if a.kind not in ("type_text", "search_web")]
        return {
            "utterances": len(self.utterances),
            "actions": len(acts),
            "before_final": sum(a.before_final for a in acts),
            "closed_set_before_final_rate": round(sum(a.before_final for a in closed) / len(closed), 3) if closed else None,
            "last_word_to_action_ms_p50": round(statistics.median(lat), 1) if lat else None,
            "last_word_to_action_ms_p95": round(_pct(lat, 95), 1) if lat else None,
            "failures": len(self.failures),
            "revisions_after_act": self.revisions,
            **{f"stat_{k}": v for k, v in self.stats.items()},
        }

    def table(self) -> str:
        lines = [f"{'#':>2}  {'utterance':<44} {'action':<32} {'t_ms':>7} {'word':>6} {'pre-final':>9}"]
        for u in self.utterances:
            if not u.actions:
                v = ",".join(f"{k}:{n}" for k, n in sorted(u.verdicts.items()))
                lines.append(f"{u.index:>2}  {u.text[:44]:<44} {'-- ' + v:<32}")
            for i, a in enumerate(u.actions):
                lines.append(
                    f"{u.index if i == 0 else '':>2}  {(u.text if i == 0 else '')[:44]:<44} {a.describe[:32]:<32} "
                    f"{a.t_ms:>7.0f} {f'{a.at_word}/{u.n_words}':>6} {'yes' if a.before_final else 'no':>9}"
                )
            for f in u.failures:
                lines.append(f"    FAIL: {f}")
        return "\n".join(lines)


def _pct(xs: Sequence[float], q: float) -> float:
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(q / 100.0 * (len(s) - 1)))))
    return s[k]


# ---------------------------------------------------------------- driver

ControllerFactory = Callable[[Callable[[], float]], Any]  # clock -> Controller
MAX_SETTLE_S = 30.0  # after the script, wait at most this long for queued work to finish


async def replay(
    script: Sequence[ReplayUtterance],
    controller_factory: ControllerFactory,
    word_ms: int = 280,
    final_delay_ms: int = 500,
    pause_ms: int = 1500,
    tail_ms: int | None = None,
    between: Callable[[int, Any], Awaitable[None] | None] | None = None,
) -> ReplayReport:
    """Drive `controller_factory(clock)` with timed events; see the module docstring.

    `between(i, controller)` runs before utterance i starts (tests use it to change the screen or
    to call `cancel_all`). `tail_ms` is how long to keep the loop running after the last
    utterance; the default outlasts the silence timers and the confirmation timeout.
    """
    loop = asyncio.get_running_loop()
    clock = loop.time
    ctl = controller_factory(clock)
    T = ctl.T
    windows: list[UtteranceReport] = []
    seq = 0
    t_origin = clock()

    async def at(t: float) -> None:
        d = t - clock()
        if d > 0:
            await asyncio.sleep(d)

    async def emit(kind: str, uid: str, text: str) -> None:
        nonlocal seq
        seq += 1
        await ctl.on_event(TranscriptEvent(kind=kind, seq=seq, uid=uid, text=text, t_mono=clock()))  # type: ignore[arg-type]

    t = t_origin
    for i, u in enumerate(script):
        if between is not None:
            r = between(i, ctl)
            if asyncio.iscoroutine(r):
                await r
        await at(t)
        uid = u.uid or f"u{i + 1}"
        wms = u.word_ms if u.word_ms is not None else word_ms
        partials = u.partials if u.partials is not None else _word_prefixes(u.text)
        start = clock()
        marks: list[tuple[float, int]] = []
        await emit("speech_start", uid, "")
        for k, p in enumerate(partials, 1):
            await at(start + k * wms / 1000.0)
            marks.append((clock(), len(p.split())))
            await emit("partial", uid, p)
        t_last = start + len(partials) * wms / 1000.0
        final_text = u.text if u.final is None else u.final
        t_final: float | None = None
        if final_text != "":
            await at(t_last + final_delay_ms / 1000.0)
            t_final = clock()
            await emit("final", u.final_uid or uid, final_text)
        await emit("speech_end", uid, "")
        windows.append(UtteranceReport(i, uid, u.text, start, t_last, t_final, len(u.text.split()), marks=marks))
        t_end = max(clock(), t_last + final_delay_ms / 1000.0)
        t = t_end + (u.pause_ms if u.pause_ms is not None else pause_ms) / 1000.0

    end_wait = tail_ms if tail_ms is not None else max(pause_ms, T.silence_complete_ms + 300)
    await at((t_end if script else t) + end_wait / 1000.0)
    # Then until the controller has nothing left that could act (e.g. a rate-limited retry).
    deadline = clock() + MAX_SETTLE_S
    while clock() < deadline:
        await ctl.drain()
        if ctl.idle():
            break
        await asyncio.sleep(0.05)
    await ctl.drain()
    report = _build_report(script, windows, ctl, word_ms, final_delay_ms)
    await ctl.aclose()
    return report


def _word_prefixes(text: str) -> list[str]:
    words = text.split()
    return [" ".join(words[:k]) for k in range(1, len(words) + 1)]


def _window_index(windows: list[UtteranceReport], t: float) -> int:
    idx = 0
    for i, w in enumerate(windows):
        if t >= w.t_start:
            idx = i
    return idx


def _build_report(
    script: Sequence[ReplayUtterance], windows: list[UtteranceReport], ctl: Any, word_ms: int, final_delay_ms: int
) -> ReplayReport:
    # Attribute each execution and decision to the utterance being spoken (or just finished) when
    # it happened. A confirmed action runs during the "yes" utterance, not the one that proposed it.
    for rec in ctl.history:
        w = windows[_window_index(windows, rec.t)]
        heard = max((n for tm, n in w.marks if tm <= rec.t), default=0)
        w.actions.append(ActionReport(
            kind=rec.action.kind.value,
            describe=rec.action.describe(),
            source_vid=rec.action.source_vid,
            t_ms=round((rec.t - w.t_last_word) * 1000.0, 1),
            at_word=heard,
            before_final=w.t_final is not None and rec.t < w.t_final,
            outcome=rec.outcome,
            record=rec,
        ))
    for t, _tail, d in ctl.decisions:
        w = windows[_window_index(windows, t)]
        w.verdicts[d.verdict] = w.verdicts.get(d.verdict, 0) + 1
        w.decisions.append(d)
    for w, u in zip(windows, script):
        if u.expect is not None:
            w.failures.extend(check_expect(u.expect, w))
    return ReplayReport(windows, word_ms, final_delay_ms, dict(ctl.stats), len(ctl.stream.revisions))


def _field(a: ActionReport, key: str) -> Any:
    if key in ("kind", "before_final", "outcome", "at_word", "t_ms"):
        return getattr(a, key)
    return getattr(a.record.action, key, None)


def check_expect(expect: list[dict[str, Any]], w: UtteranceReport) -> list[str]:
    fails: list[str] = []
    acts = list(w.actions)
    exec_expect = [e for e in expect if "verdict" not in e]
    for e in expect:
        if "verdict" in e:
            want_kind = e.get("kind")
            if not any(d.verdict == e["verdict"] and (want_kind is None or (d.action and d.action.kind.value == want_kind))
                       for d in w.decisions):
                fails.append(f"expected a {e['verdict']!r} decision{f' for {want_kind}' if want_kind else ''}; "
                             f"got {sorted(w.verdicts.items())}")
    if len(acts) != len(exec_expect):
        fails.append(f"expected {len(exec_expect)} action(s) {[e.get('kind') for e in exec_expect]}, "
                     f"got {[a.describe for a in acts]}")
    for e, a in zip(exec_expect, acts):
        for k, v in e.items():
            got = _field(a, k)
            if k == "max_t_ms":
                if a.t_ms > v:
                    fails.append(f"{a.describe}: t_ms {a.t_ms} > {v}")
            elif got != v:
                fails.append(f"{a.describe}: {k}={got!r}, expected {v!r}")
    return fails


def run_replay(
    script: Sequence[ReplayUtterance], controller_factory: ControllerFactory, *, virtual: bool = True, **kw: Any
) -> ReplayReport:
    """Synchronous wrapper: virtual time by default (tests), real time with virtual=False (benchmarks)."""
    coro = replay(script, controller_factory, **kw)
    return run_virtual(coro) if virtual else asyncio.run(coro)


def load_script(path: str | Path) -> list[ReplayUtterance]:
    """JSONL, one utterance per line: {"text": ..., "expect": [...], "partials": [...], ...}."""
    out: list[ReplayUtterance] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        obj = json.loads(line)
        if isinstance(obj, str):
            obj = {"text": obj}
        out.append(ReplayUtterance(**obj))
    return out
