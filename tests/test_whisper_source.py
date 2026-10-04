"""whisper.cpp source: terminal parser, utterance tracker, window emulation, and the controller
end to end on synthetic whisper-stream output. Pure Python: no whisper run, no microphone, no torch."""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
import time
import wave
from pathlib import Path

import pytest

from jev_local.harness import whisper_source as ws
from jev_local.harness.fakes import FakeExecutor, RuleModel, ScriptedEngine
from jev_local.harness.replay import run_virtual
from jev_local.harness.whisper_source import (
    CLEAR,
    FileWhisperSource,
    UtteranceTracker,
    WhisperSource,
    WhisperStreamParser,
    align_skip,
    clean_text,
    stabilize,
    stream_windows,
    strip_trailing_hallucination,
)

WIPE = f"{CLEAR}\r{' ' * 100}{CLEAR}\r"


def step(text: str, newline: bool = False) -> str:
    """One whisper-stream step exactly as stream.cpp prints it (sliding-window mode)."""
    return WIPE + " " + text + ("\n" if newline else "")


def run_tracker(windows: list[str | None], dt: float = 0.5, **kw) -> list[tuple[float, str, str, str]]:
    """Feed windows (None = newline/commit) one step apart; returns (t, kind, uid, text) events."""
    p, tr = WhisperStreamParser(), UtteranceTracker(**kw)
    out = []
    t = 0.0
    data = "[Start speaking]\n"
    for w in windows:
        data += "\n" if w is None else step(w)
    # feed in awkward 7-byte pieces: escape sequences and words get split across reads
    items = []
    for i in range(0, len(data), 7):
        items += p.feed(data[i : i + 7])
    items += p.flush()
    for it in items:
        if it[0] == "window":
            t += dt
        for ev in tr.handle(it, t):
            out.append((t, ev.kind, ev.uid, ev.text))
    for ev in tr.close(t + 5):
        out.append((t + 5, ev.kind, ev.uid, ev.text))
    return out


# ---------------------------------------------------------------- text helpers


def test_clean_text_drops_annotations():
    assert clean_text(" [BLANK_AUDIO]") == ""
    assert clean_text(" (music) Open notes ♪ la la ♪") == "Open notes"
    assert clean_text("*sigh* scroll down [Music]") == "scroll down"
    assert clean_text(">> open safari (") == "open safari"
    assert clean_text(" - Open notes.") == "Open notes."


def test_strip_trailing_hallucination():
    assert strip_trailing_hallucination("Open notes. Thank you.") == "Open notes."
    assert strip_trailing_hallucination("Thank you.") == "Thank you."  # alone: the tracker decides
    assert strip_trailing_hallucination("open notes thank you") == "open notes thank you"


def test_stabilize_keeps_emitted_tokens_on_respacing():
    old = "Open TextEdit and type".split()
    assert stabilize(old, "Open text edit and type hello.".split()) == "Open TextEdit and type hello.".split()
    assert stabilize(old, "open textedit, and".split()) == "open textedit, and".split()  # retraction
    assert stabilize(old, "Close notes".split()) == "Close notes".split()
    assert stabilize([], ["a"]) == ["a"]


def test_align_skip():
    assert align_skip(["open", "notes"], "Open Notes. Scroll down".split()) == 2
    assert align_skip(["open", "textedit"], "Open text edit. Scroll".split()) in (2, 3)
    assert align_skip([], ["a"]) == 0
    assert align_skip(["scroll", "down"], "Scroll down. Scroll down.".split()) == 2


# ---------------------------------------------------------------- window emulation


def test_stream_windows_matches_stream_cpp():
    sr = ws.SAMPLE_RATE
    w = stream_windows(10 * sr, 500, 8000, 200)
    step_n, keep = sr // 2, sr // 5
    assert w[0] == (0, step_n, False) and w[1] == (0, 2 * step_n, False)
    n_new_line = 8000 // 500 - 1  # 15
    assert [i for i, x in enumerate(w) if x[2]] == [n_new_line - 1]
    # after the newline the window restarts from the last keep_ms of audio
    s, e, _ = w[n_new_line]
    assert e - s == keep + step_n
    assert all(e - s <= keep + 8 * sr for s, e, _ in w)


# ---------------------------------------------------------------- tracker


def kinds(evs):
    return [(k, txt) for _, k, _, txt in evs]


def test_tracker_growing_utterance_and_final():
    evs = run_tracker(["[BLANK_AUDIO]", "Open", "Open notes", "Open notes.", "Open Notes.", "Open notes."])
    assert kinds(evs) == [("speech_start", ""), ("partial", "Open"), ("partial", "Open notes"),
                          ("final", "Open notes"), ("speech_end", "")]
    # final after 2 unchanged windows (>= 900 ms after the last change)
    t_last_change = next(t for t, k, _, x in evs if k == "partial" and x == "Open notes")
    t_final = next(t for t, k, *_ in evs if k == "final")
    assert t_final - t_last_change == pytest.approx(1.0)


def test_tracker_case_and_punctuation_flips_emit_nothing():
    evs = run_tracker(["Scroll down", "scroll down.", "Scroll Down!", "Scroll down a bit"], final_windows=5)
    parts = [x for _, k, _, x in evs if k == "partial"]
    assert parts == ["Scroll down", "Scroll down a bit"]


def test_tracker_second_utterance_in_same_line_skips_finished_words():
    evs = run_tracker(["Open notes", "Open notes.", "Open notes.", "Open notes.",
                       "Open notes. Scroll", "Open Notes, scroll down.", "Open notes. Scroll down.",
                       "Open notes. Scroll down.", "Open notes. Scroll down."])
    finals = [x.lower() for _, k, _, x in evs if k == "final"]
    assert finals == ["open notes", "scroll down."]
    uids = {u for _, k, u, _ in evs if k == "final"}
    assert uids == {"w1", "w2"}


def test_tracker_utterance_across_line_reset_and_overlap_word():
    evs = run_tracker(["Open safari and", "Open safari and go", None, "go to", "go to wikipedia dot org.",
                       "go to wikipedia dot org.", "go to wikipedia dot org."])
    finals = [x for _, k, _, x in evs if k == "final"]
    assert finals == ["Open safari and go to wikipedia dot org."]


def test_tracker_drops_hallucinations_and_blanks():
    evs = run_tracker(["[BLANK_AUDIO]", "(silence)", "Thank you.", "you", "[Music]", "Thank you. Open notes",
                       "Thank you. Open notes.", "Thank you. Open notes.", "Thank you. Open notes."])
    finals = [x for _, k, _, x in evs if k == "final"]
    assert finals == ["Open notes"]


def test_tracker_stall_tick_finalizes():
    tr = UtteranceTracker()
    tr.window("Open notes", 1.0)
    assert tr.tick(2.0) == []
    evs = tr.tick(4.0)
    assert [e.kind for e in evs] == ["final", "speech_end"]


# ---------------------------------------------------------------- live source over a fake whisper-stream

FAKE_STREAM = textwrap.dedent(
    """
    import json, sys, time, signal
    script = json.load(open(sys.argv[1]))
    sys.stderr.write("init: found 1 capture devices\\n"); sys.stderr.flush()
    out = sys.stdout
    out.write("[Start speaking]\\n"); out.flush()
    WIPE = "\\x1b[2K\\r" + " " * 100 + "\\x1b[2K\\r"
    for item in script["steps"]:
        time.sleep(script.get("dt", 0.05))
        if item is None:
            out.write("\\n")
        else:
            out.write(WIPE + " " + item)
        out.flush()
    if script.get("hang"):
        time.sleep(60)
    """
)


def _fake_bin(tmp_path: Path, steps, dt=0.05, hang=False) -> list[str]:
    exe = tmp_path / "fake_stream.py"
    exe.write_text(FAKE_STREAM)
    sc = tmp_path / "script.json"
    sc.write_text(json.dumps({"steps": steps, "dt": dt, "hang": hang}))
    return [sys.executable, str(exe), str(sc)]


def test_whisper_source_over_fake_binary(tmp_path):
    steps = ["[BLANK_AUDIO]", "Open", "Open notes", "Open notes.", "Open notes.", "Open notes."]

    async def main():
        src = WhisperSource("base.en", binary=_fake_bin(tmp_path, steps), final_after_ms=90, pidfile=None)
        await src.start()
        evs = []
        async for ev in src.events():
            evs.append(ev)
        await src.stop()
        return src, evs

    src, evs = asyncio.run(main())
    assert [(e.kind, e.text) for e in evs if e.kind != "error"] == [
        ("speech_start", ""), ("partial", "Open"), ("partial", "Open notes"), ("final", "Open notes"),
        ("speech_end", "")]
    assert src.info.ready_at is not None
    assert src.info.exit_code == 0
    assert "-m" in src.info.cmd and "--step" in src.info.cmd


def test_whisper_source_stop_kills_process(tmp_path):
    async def main():
        src = WhisperSource("base.en", binary=_fake_bin(tmp_path, ["Open"], hang=True), pidfile=tmp_path / "pid")
        await src.start()
        pid = src.info.pid
        assert (tmp_path / "pid").read_text() == str(pid)
        await asyncio.sleep(0.3)
        t0 = time.monotonic()
        await src.stop()
        return pid, time.monotonic() - t0

    pid, dt = asyncio.run(main())
    assert dt < 2.5
    import os

    with pytest.raises(OSError):
        os.kill(pid, 0)  # gone
    assert not (tmp_path / "pid").exists()


def test_kill_stale_ignores_non_whisper_pid(tmp_path):
    import os

    pf = tmp_path / "pid"
    pf.write_text(str(os.getpid()))  # this python process: not whisper, must not be killed
    assert ws.kill_stale(pf) is None
    assert not pf.exists()


# ---------------------------------------------------------------- end to end through the Controller


def _silent_wav(path: Path, seconds: float) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(ws.SAMPLE_RATE)
        w.writeframes(b"\x00\x00" * int(ws.SAMPLE_RATE * seconds))
    return path


def _drive(tmp_path: Path, texts: list[str], seconds: float, n_files: int = 1, gap_ms: int = 2500):
    """FileWhisperSource with precomputed window texts -> Controller(RuleModel) in virtual time."""
    from jev_local.harness.client import DecisionClient
    from jev_local.harness.controller import Controller
    from jev_local.harness.log import AuditLog
    from jev_local.harness.simworld import SimExecutor, SimWorld
    from jev_local.harness.types import HarnessConfig

    files = [_silent_wav(tmp_path / f"a{i}.wav", seconds) for i in range(n_files)]

    async def main():
        loop = asyncio.get_running_loop()
        src = FileWhisperSource(files, "base.en", texts=texts, latency_ms=150, lead_ms=0, gap_ms=gap_ms,
                                trail_ms=2500, clock=loop.time)
        src.prepare()
        world = SimWorld()
        ex = FakeExecutor()
        ctl = Controller(src, world, DecisionClient("inproc", {"fast": ScriptedEngine(RuleModel())}), SimExecutor(ex, world),
                         HarnessConfig(), clock=loop.time, log=AuditLog(enabled=False))
        finals = []
        inner = ctl.on_event

        async def tap(ev):
            if ev.kind == "final":
                finals.append((loop.time(), ev.text))
            await inner(ev)

        ctl.on_event = tap
        await ctl.run()
        return ctl, ex, finals, src

    return run_virtual(main())


def _pad(texts: list[str], n: int) -> list[str]:
    return texts + [texts[-1]] * (n - len(texts))


def test_e2e_open_fires_mid_sentence_and_type_after(tmp_path):
    n = int((3.0 + 2.5) / 0.5)
    texts = _pad(["[BLANK_AUDIO]", "Open", "Open text", "Open TextEdit and", "Open TextEdit and type",
                  "Open text edit and type hello.", "Open TextEdit and type hello world."], n)
    ctl, ex, finals, src = _drive(tmp_path, texts, 3.0)
    assert ex.kinds == ["open_app", "type_text"], ex.kinds
    assert ex.actions[0].app == "TextEdit"
    assert ex.actions[1].text.rstrip(".") == "hello world"
    t_open = ctl.history[0].t
    t_final = finals[0][0]
    assert t_open < t_final  # mid-sentence
    assert ctl.history[1].t >= t_open
    assert ctl.stats["executed"] == 2


def test_e2e_revisions_never_double_execute(tmp_path):
    # Whisper flip-flops the words already acted on; open_app must run exactly once.
    n = int((3.0 + 2.5) / 0.5)
    texts = _pad(["Open notes", "Open notes and", "Open Notes, and scroll", "Open notes and scroll down.",
                  "Open notes, and scroll down.", "open notes and scroll down"], n)
    ctl, ex, finals, src = _drive(tmp_path, texts, 3.0)
    assert ex.kinds.count("open_app") == 1, ex.kinds
    assert ex.kinds == ["open_app", "scroll_down"], ex.kinds


def test_e2e_side_talk_ignored(tmp_path):
    n = int((3.0 + 2.5) / 0.5)
    texts = _pad(["Hey", "Hey, can you", "Hey, can you pass the", "Hey, can you pass the salt?"], n)
    ctl, ex, finals, src = _drive(tmp_path, texts, 3.0)
    assert ex.kinds == []
    assert finals and "salt" in finals[0][1]


def test_e2e_quit_needs_confirm_from_next_utterance(tmp_path):
    # two files: "quit textedit" then "confirm", joined with 2.5 s of silence by the source
    per = int((2.0 + 2.5) / 0.5)
    first = _pad(["Quit", "Quit TextEdit.", "Quit TextEdit."], per)
    second = _pad(["Quit TextEdit. Confirm", "Quit TextEdit. Confirm."], 6)
    # window 15 ends the 8 s line: the next line holds only fresh (silent) audio
    texts = first + second + ["[BLANK_AUDIO]"] * 3
    ctl, ex, finals, src = _drive(tmp_path, texts, 2.0, n_files=2)
    assert ex.kinds == ["quit_app"], ex.kinds
    # it ran during the second utterance, never during the first
    assert ctl.history[0].t > finals[0][0]
    assert [t for _, t in finals] == ["Quit TextEdit.", "Confirm"]


def test_kill_now_is_immediate(tmp_path):
    import os

    async def main():
        src = WhisperSource("base.en", binary=_fake_bin(tmp_path, ["Open"], hang=True), pidfile=None)
        await src.start()
        await asyncio.sleep(0.3)
        pid = src.info.pid
        t0 = time.monotonic()
        src.kill_now()  # what control-option-escape / Ctrl-C call: synchronous SIGKILL
        evs = [ev async for ev in src.events()]  # the reader sees EOF and ends the stream
        return pid, time.monotonic() - t0, evs

    pid, dt, evs = asyncio.run(main())
    assert dt < 1.0
    with pytest.raises(OSError):
        os.kill(pid, 0)
    assert [e.kind for e in evs] == ["speech_start", "partial", "final", "speech_end"]  # no error event for our kill
