"""Adversarial review of the voice demo: mid-sentence correctness and latency with whisper-stream.

Every test in this file FAILS on the reviewed code and pins one real bug (see each docstring for
the scenario and the suggested fix). Pure Python: no torch, no microphone, no UI actions.

whisper-stream facts these tests rely on (examples/stream/stream.cpp, whisper.cpp v1.8.3, the
Homebrew build used by the demo):
- one inference per `--step` of audio; the window is the whole current line (it only grows, so
  words never drop off the front with the demo's flags) and the line resets every
  max(1, length/step - 1) steps (7.5 s at 500/8000), keeping only `--keep` (200 ms) of audio;
- the transcript can therefore change at most once per step, plus inference-latency jitter;
- `audio.get(step_ms)` returns at most step_ms of audio, so when inference is slower than the
  step the extra audio is silently dropped; "cannot process audio fast enough" is unreachable.
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
import wave
from pathlib import Path

import pytest

from jev_local import demo
from jev_local.harness import whisper_source as ws
from jev_local.harness.client import DecisionClient
from jev_local.harness.controller import Controller
from jev_local.harness.fakes import FakeExecutor, RuleModel, ScriptedEngine
from jev_local.harness.log import AuditLog
from jev_local.harness.replay import run_virtual
from jev_local.harness.simworld import SimExecutor, SimWorld
from jev_local.harness.spans import consumed_for, extract_text_candidates
from jev_local.harness.types import HarnessConfig
from jev_local.harness.whisper_source import (
    CLEAR,
    FileWhisperSource,
    UtteranceTracker,
    WhisperSource,
    WhisperStreamParser,
)

WIPE = f"{CLEAR}\r{' ' * 100}{CLEAR}\r"
LATENCY_S = 0.15  # whisper inference per window (base.en on an M1, Metal); the step dominates anyway


# ---------------------------------------------------------------- helpers


def demo_cfg(step_ms: int = 500) -> HarnessConfig:
    """The HarnessConfig the demo runs with. Today `demo.amain` builds `HarnessConfig(dry_run=...)`
    with the default Thresholds, whatever --step-ms is. A fix may add `demo.harness_config(step_ms=)`."""
    f = getattr(demo, "harness_config", None)
    return f(step_ms=step_ms) if f is not None else HarnessConfig()


def cadence(texts: list[str | None], step_ms: int = 500, jitter: dict[int, float] | None = None,
            pad_s: float = 3.0) -> list[tuple[float, str | None]]:
    """(time, window text) as whisper-stream prints them: one window per step, LATENCY_S after the
    step's audio ends (+ optional per-window extra latency). The last text repeats for `pad_s`
    (whisper keeps re-transcribing the same words over silence), so the tracker can finalise."""
    step = step_ms / 1000.0
    texts = list(texts) + [texts[-1]] * int(pad_s / step)
    out, k = [], 0
    for t in texts:
        if t is None:  # the newline at a line reset comes right after the window it ends
            out.append((out[-1][0] if out else 0.0, None))
            continue
        k += 1
        out.append((k * step + LATENCY_S + (jitter or {}).get(k, 0.0), t))
    return out


class TimedWhisper:
    """whisper-stream stdout at chosen (virtual) times, through the real parser and tracker,
    exactly as `WhisperSource._emit` feeds them (minus the 25 ms idle flush)."""

    def __init__(self, windows: list[tuple[float, str | None]]):
        self.windows = windows
        self.tracker: UtteranceTracker | None = None

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def events(self):
        loop = asyncio.get_running_loop()
        parser = WhisperStreamParser()
        self.tracker = tr = UtteranceTracker(clock=loop.time)
        t0 = loop.time()
        for t, text in self.windows:
            d = t0 + t - loop.time()
            if d > 0:
                await asyncio.sleep(d)
            chunk = "\n" if text is None else WIPE + " " + text
            for item in parser.feed(chunk) + parser.flush():
                for ev in tr.handle(item, loop.time()):
                    yield ev
        for ev in tr.close(loop.time()):
            yield ev


def drive(windows, cfg: HarnessConfig | None = None, model: RuleModel | None = None, app_names=None):
    """TimedWhisper -> Controller(RuleModel) -> SimExecutor(FakeExecutor), in virtual time."""

    async def main():
        loop = asyncio.get_running_loop()
        world = SimWorld()
        ex = FakeExecutor()
        kw = {"app_names": app_names} if app_names else {}
        ctl = Controller(TimedWhisper(windows), world,
                         DecisionClient("inproc", {"fast": ScriptedEngine(model or RuleModel())}),
                         SimExecutor(ex, world), cfg or demo_cfg(), clock=loop.time, log=AuditLog(enabled=False), **kw)
        await ctl.run()
        return ctl, ex

    return run_virtual(main())


def executed(ex: FakeExecutor) -> list[tuple[str, str | None]]:
    return [(a.kind.value, (a.app or a.text or a.key or a.url or None) and
             (a.app or a.text or a.key or a.url).rstrip(".")) for a in ex.actions]


def track(windows: list[str | None], dt: float = 0.5) -> list[tuple[float, str, str]]:
    """Windows one step apart through parser + tracker; returns (t, kind, text) events."""
    p, tr = WhisperStreamParser(), UtteranceTracker()
    out, t = [], 0.0
    for w in windows:
        items = p.feed("\n" if w is None else WIPE + " " + w) + p.flush()
        for it in items:
            if it[0] == "window":
                t += dt
            for ev in tr.handle(it, t):
                out.append((t, ev.kind, ev.text))
    for ev in tr.close(t + 10):
        out.append((t + 10, ev.kind, ev.text))
    return out


def finals(evs) -> list[str]:
    return [x for _, k, x in evs if k == "final"]


# ---------------------------------------------------------------- 1. silence gates vs the whisper step (HIGH)


@pytest.mark.parametrize("step_ms,texts", [
    # a breath (or a long word) inside the payload: one 500 ms window shows no new word
    (500, ["[BLANK_AUDIO]", "Type", "Type hello", "Type hello", "Type hello world."]),
    # DEMO.md's own troubleshooting advice is --step-ms 800: every window then arrives after the
    # 615 ms payload timer, so EVERY payload is cut at its first window, even in continuous speech
    (800, ["[BLANK_AUDIO]", "Type hello", "Type hello world."]),
], ids=["step500-one-quiet-window", "step800-continuous-speech"])
def test_payload_is_not_cut_by_one_whisper_step(step_ms, texts):
    """payload_silence_ms=600 is 'ms since the transcript last changed', but whisper-stream can only
    change it once per step (+ jitter). 600 ms therefore means 'one window without a new word', not
    silence. Observed: type_text('hello') runs, then 'world.' is ignored as side talk.
    Fix: derive the silence gates from the source cadence, e.g. payload_silence_ms >= 2*step + 150
    and silence_complete_ms >= payload (or gate payloads on the tracker's final, which already needs
    2 unchanged windows), and set them in demo.py from --step-ms / the measured window period."""
    ctl, ex = drive(cadence(texts, step_ms), demo_cfg(step_ms))
    assert executed(ex) == [("type_text", "hello world")]


def test_stability_rule_needs_a_second_whisper_window_not_a_timer():
    """With complete < stable_complete the policy wants 'a second agreeing partial'. With whisper the
    second evaluation is the 615 ms silence timer (settled=True skips the stability rule), so one
    window that is 150 ms late lets a single, still-revisable window act. Here whisper hears
    'Open notes' while the user is saying 'Open Notability'; Notes opens, then the revision freezes
    the utterance and Notability never opens.
    Fix: same as above (silence gates > step + jitter); optionally have the tracker expose the
    number of unchanged windows and require >= 1 for any partial act."""
    texts = ["[BLANK_AUDIO]", "Open", "Open notes", "Open Notability."]
    ctl, ex = drive(cadence(texts, 500, jitter={4: 0.15}), demo_cfg(500), model=RuleModel(complete_p=0.75),
                    app_names=["Notes", "Notability", "Safari", "TextEdit"])
    assert executed(ex) == [("open_app", "Notability")]


def test_partial_to_action_has_no_debounce_delay_at_whisper_cadence():
    """whisper-stream delivers one partial per 500 ms step, so the 120 ms partial debounce never
    coalesces anything; it only adds 120 ms to every mid-sentence command.
    Fix: debounce_ms ~0-20 for the whisper source (demo.py), keep 120 for word-level recognizers."""
    texts = ["[BLANK_AUDIO]", "Open", "Open notes"]
    windows = cadence(texts, 500)
    t_partial = windows[2][0]
    ctl, ex = drive(windows, demo_cfg(500))
    assert executed(ex) == [("open_app", "Notes")]
    assert ctl.history[0].t - t_partial <= 0.03, f"{(ctl.history[0].t - t_partial) * 1000:.0f} ms after the partial"


# ---------------------------------------------------------------- 2. whisper's sentence punctuation (HIGH)


def test_sentence_end_is_a_command_boundary_for_consumption():
    """Whisper punctuates: 'open notes [short pause] scroll down' arrives as 'Open notes. Scroll down.'
    consumed_for only stops at chain words, so open_app consumes the whole tail and scroll_down never
    runs. (The training data only joins chained commands with chain words, CHAIN_JOIN.)
    Fix: in spans.consumed_for (non-payload) stop after a word ending in . ! ?; add ". "-joined
    chains to the synthetic data (grammar.CHAIN_JOIN) and to RuleModel._first_clause."""
    assert consumed_for("Open notes. Scroll down.".split(), "open_app") == 2


def test_sentence_end_cuts_payload_candidates():
    """'Type hello world. Press enter.' has no candidate 'hello world' (variants are cut at chain
    words only), so the text question can only choose 'hello world. Press enter' and the next
    command gets typed. Fix: spans._variants also cuts `rest` at each sentence end ([.!?] + space)."""
    assert "hello world" in extract_text_candidates("Type hello world. Press enter.")


def test_e2e_open_then_scroll_without_chain_word():
    """End to end: the first window 'Open notes' is not confident enough to act alone; the next one
    brings the second sentence and the open consumes it."""
    texts = ["[BLANK_AUDIO]", "Open", "Open notes", "Open notes. Scroll down."]
    ctl, ex = drive(cadence(texts, 500), demo_cfg(500), model=RuleModel(complete_p=0.75))
    assert [k for k, _ in executed(ex)] == ["open_app", "scroll_down"]


# ---------------------------------------------------------------- 3. tracker boundaries (HIGH / MEDIUM)


@pytest.mark.parametrize("junk,real", [("You", "Undo that."), ("The", "Open notes."), ("I", "Scroll down.")])
def test_hallucinated_first_window_does_not_eat_the_first_real_word(junk, real):
    """The first window of an utterance often holds only the onset of the first word, which whisper
    renders as 'You' / 'The' / 'I'. The tracker drops it as a hallucination but stores it in _done;
    the next window does not align with it, and align_skip's fallback skips len(done) words BY
    COUNT, eating the real first word: 'Open notes.' becomes 'notes.' and nothing runs.
    Fix: don't put a dropped hallucination into _done (or skip it only on an exact normalised
    match at the window start); in align_skip return 0 when no block matches at all."""
    evs = track([junk, real, real, real, real])
    assert finals(evs) == [real]


def test_word_cut_at_the_line_reset_is_not_duplicated():
    """Every 7.5 s (length/step - 1 steps, on a fixed audio clock) whisper-stream starts a new line
    from 200 ms of old audio. A ~2.5 s command straddles that reset about 1 time in 3. The word cut
    by the reset is printed truncated at the end of the old line ('wor-') and whole at the start of
    the new one; the overlap dedupe only handles an exact repeat, so the dictated text becomes
    'hello wor world.'. Fix: in UtteranceTracker.window, when the new line's first word extends,
    is extended by, or fuzzily equals (ratio >= 0.75) the base's last word, replace that base word."""
    evs = track(["Type", "Type hello", "Type hello wor-", None, "world.", "world.", "world.", "world."])
    assert finals(evs) == ["Type hello world."]


def test_keep_audio_after_a_reset_does_not_start_a_phantom_utterance():
    """After 'Scroll down.' is final, a line reset keeps its last 200 ms; whisper can transcribe that
    tail ('down.'). commit() clears _done when no utterance is active, so the fragment starts a new
    utterance 'down.' that the model may act on (a second scroll).
    Fix: on commit, remember the previous line's last word(s) and skip them at the start of the
    next line whether or not an utterance is active (same rule as the in-utterance overlap)."""
    evs = track(["Scroll down.", "Scroll down.", "Scroll down.", "Scroll down.", None, "down.", "down.", "down."])
    assert finals(evs) == ["Scroll down."]


def test_flip_flopping_last_word_still_finalises():
    """Over trailing silence whisper re-transcribes the same audio each step and can alternate a
    word ('world.' / 'word.'). Each flip is a normalised change: a new partial, the controller's
    silence timers re-arm, and the tracker's final never comes until the 7.5 s line reset, so the
    payload waits up to ~8 s. Fix: treat a window equal to a variant emitted in the last ~3 windows
    as unchanged (no partial, _same += 1)."""
    flips = ["Type hello world.", "Type hello word."] * 6
    evs = track(["Type", "Type hello"] + flips)
    t_first = next(t for t, k, x in evs if k == "partial" and x.startswith("Type hello wor"))
    t_final = next((t for t, k, _ in evs if k == "final"), None)
    assert t_final is not None and t_final - t_first <= 1.5, (t_first, t_final)


# ---------------------------------------------------------------- 4. whisper slower than the step (MEDIUM)


def _silent_wav(path: Path, seconds: float) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(ws.SAMPLE_RATE)
        w.writeframes(b"\x00\x00" * int(ws.SAMPLE_RATE * seconds))
    return path


def test_file_replay_models_whisper_dropping_audio_when_slow(tmp_path):
    """When inference (L) is slower than the step, real whisper-stream takes the LAST step_ms of audio
    after each inference and drops the rest, so the lag stays ~L and words are lost.
    FileWhisperSource instead queues every window (busy_until), so the lag grows by L - step per
    window (4.5 s after one line at L=800) and no audio is lost: the offline proof then reports
    wrong latencies and better accuracy than the microphone gets.
    Fix: in stream_windows/terminal_output, when latency > step, start each window's new audio at
    max(prev_end, t_emit_prev - step) (i.e. the last step_ms before the inference starts)."""
    src = FileWhisperSource([_silent_wav(tmp_path / "a.wav", 8.0)], "base.en", texts=["x"], latency_ms=800,
                            lead_ms=0, trail_ms=0)
    src.terminal_output()
    lags = [emit - end for end, emit, _ in src.info.windows]
    assert max(lags) <= 0.8 + 0.05, f"lag grows to {max(lags):.2f} s"


FAKE_STREAM = textwrap.dedent(
    """
    import json, sys, time
    script = json.load(open(sys.argv[1]))
    out = sys.stdout
    out.write("[Start speaking]\\n"); out.flush()
    WIPE = "\\x1b[2K\\r" + " " * 100 + "\\x1b[2K\\r"
    for item in script["steps"]:
        time.sleep(script["dt"])
        out.write(WIPE + " " + item); out.flush()
    """
)


def test_slow_whisper_is_reported(tmp_path):
    """SourceInfo.dropped_audio counts "cannot process audio fast enough", which stream.cpp can never
    print in sliding-window mode (audio.get(step) is capped at step, so the size check never trips).
    Windows arriving every 250 ms with --step 100 mean 60% of the audio is being dropped, and the
    HUD says nothing (DEMO.md's troubleshooting row points at a message that never appears).
    Fix: measure the window period in _read_stdout; when it exceeds ~1.3 x step_ms, count it in
    dropped_audio and emit an error/hint event ('whisper is slower than real time: raise --step-ms
    or use -ac 512')."""
    exe = tmp_path / "fake.py"
    exe.write_text(FAKE_STREAM)
    sc = tmp_path / "s.json"
    sc.write_text(json.dumps({"steps": ["Open", "Open notes", "Open notes.", "Open notes.", "Open notes.",
                                        "Open notes."], "dt": 0.25}))

    async def main():
        src = WhisperSource("base.en", binary=[sys.executable, str(exe), str(sc)], step_ms=100, pidfile=None)
        await src.start()
        _ = [ev async for ev in src.events()]
        await src.stop()
        return src

    src = asyncio.run(main())
    assert src.info.dropped_audio > 0
