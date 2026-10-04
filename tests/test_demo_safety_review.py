"""Adversarial safety review of the live voice demo (whisper.cpp -> Controller -> Executor).

Every test here demonstrates a way the live demo can do something harmful or unintended on the
real Mac. They are expected to FAIL until the corresponding issue is fixed. Pure Python: no torch,
no microphone, no whisper run, no UI actions (live code paths use recording fakes only).
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import threading
import wave
from pathlib import Path
from typing import Any

import pytest

from jev_local.harness import whisper_source as ws
from jev_local.harness.client import DecisionClient
from jev_local.harness.controller import Controller
from jev_local.harness.fakes import FakeExecutor, RuleModel, ScriptedEngine, answers_for, make_snapshot
from jev_local.harness.log import AuditLog
from jev_local.harness.policy import PolicyContext, evaluate_policy
from jev_local.harness.questions import Q_INTENT, Q_KEY, Q_TEXT, build_questions
from jev_local.harness.replay import run_virtual
from jev_local.harness.safety import gate
from jev_local.harness.simworld import SimExecutor, SimWorld
from jev_local.harness.spans import extract_text_candidates, extract_url_candidates
from jev_local.harness.types import Action, ActionKind, ActionRecord, HarnessConfig, Snapshot, Tail
from jev_local.harness.whisper_source import FileWhisperSource, UtteranceTracker

CFG = HarnessConfig()
APPS = ["Notes", "Safari", "Mail", "Finder", "TextEdit", "Messages", "Slack"]


# ---------------------------------------------------------------- helpers


def _silent_wav(path: Path, seconds: float) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(ws.SAMPLE_RATE)
        w.writeframes(b"\x00\x00" * int(ws.SAMPLE_RATE * seconds))
    return path


def _pad(texts: list[str], n: int) -> list[str]:
    return texts + [texts[-1]] * (n - len(texts))


def drive_whisper(tmp_path: Path, texts: list[str], seconds: float, *, latency_ms: float = 150,
                  model: RuleModel | None = None):
    """whisper-stream terminal output (precomputed window texts, real windowing and latency model)
    -> WhisperStreamParser -> UtteranceTracker -> Controller(RuleModel) -> SimExecutor(dry-run).
    Virtual time. Returns (controller, recorded executor, [(t, final text)])."""
    wav = _silent_wav(tmp_path / "a.wav", seconds)

    async def main():
        loop = asyncio.get_running_loop()
        src = FileWhisperSource([wav], "base.en", texts=texts, latency_ms=latency_ms, lead_ms=0, trail_ms=2500,
                                clock=loop.time)
        src.prepare()
        world = SimWorld()
        ex = FakeExecutor()
        eng = ScriptedEngine(model or RuleModel())
        ctl = Controller(src, world, DecisionClient("inproc", {"fast": eng}), SimExecutor(ex, world),
                         HarnessConfig(), clock=loop.time, log=AuditLog(enabled=False))
        finals: list[tuple[float, str]] = []
        inner = ctl.on_event

        async def tap(ev):
            if ev.kind == "final":
                finals.append((round(loop.time(), 2), ev.text))
            await inner(ev)

        ctl.on_event = tap
        await ctl.run()
        return ctl, ex, finals

    return run_virtual(main())


class ScriptedWhisper:
    """A speech source that runs a real UtteranceTracker over scripted (t, "window", text) / (t, "tick")
    steps on the loop clock: whisper-stream timing, including stalls, without a subprocess."""

    def __init__(self, steps: list[tuple], clock):
        self.steps = steps
        self.clock = clock
        self.tracker = UtteranceTracker(clock=clock)

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def events(self):
        t0 = self.clock()
        for t, kind, *rest in self.steps:
            d = t0 + t - self.clock()
            if d > 0:
                await asyncio.sleep(d)
            evs = self.tracker.window(rest[0], self.clock()) if kind == "window" else self.tracker.tick(self.clock())
            for e in evs:
                yield e
        await asyncio.sleep(10.0)  # let the confirmation / silence timers play out
        for e in self.tracker.close(self.clock()):
            yield e


def drive_scripted(steps: list[tuple]):
    async def main():
        loop = asyncio.get_running_loop()
        src = ScriptedWhisper(steps, loop.time)
        world = SimWorld(front="TextEdit")
        ex = FakeExecutor()
        ctl = Controller(src, world, DecisionClient("inproc", {"fast": ScriptedEngine(RuleModel())}),
                         SimExecutor(ex, world), HarnessConfig(), clock=loop.time, log=AuditLog(enabled=False))
        await ctl.run()
        return ctl, ex

    return run_virtual(main())


def decide(text: str, spec: dict[str, Any], snap: Snapshot | None, cfg: HarnessConfig = CFG):
    """policy + safety gate exactly as the controller runs them, on a final transcript."""
    qs = build_questions(snap, APPS, extract_text_candidates(text), extract_url_candidates(text))
    resp = answers_for(qs, spec)
    tl = Tail(vid="u1+0", text=text, words=tuple(text.split()), cursor=0, is_final=True, silent_ms=0, uid="u1")
    d = evaluate_policy(resp, tl, snap, PolicyContext(), cfg.thresholds)
    said = " ".join(tl.words[: d.action.consumed_words]) if d.action and d.action.consumed_words else text
    return gate(d, snap, cfg, said=said)


TYPE_LS = {Q_INTENT: {"type_text": 0.97}, Q_TEXT: {"ls": 0.95}}
PRESS_RETURN = {Q_INTENT: {"press_key": 0.97}, Q_KEY: {"return": 0.95}}


# ---------------------------------------------------------------- 1. dictation typed half-way (silence gate vs whisper step)


def test_payload_not_typed_when_one_whisper_window_repeats(tmp_path):
    """The 600 ms payload-silence gate measures time since the transcript last changed. whisper-stream
    only produces a window every 500 ms (+ inference jitter), so ONE window that does not yet contain
    the next word ("Type hello" twice) passes the gate and half the dictation is typed while the user
    is still speaking. The rest ("world, how are you?") is then never typed."""
    n = int((3.0 + 2.5) / 0.5)
    texts = _pad(["Type", "Type hello", "Type hello", "Type hello world, how are you?"], n)
    ctl, ex, finals = drive_whisper(tmp_path, texts, 3.0)
    typed = [a.text for a in ex.actions if a.kind == ActionKind.TYPE_TEXT]
    assert len(typed) == 1 and "how are you" in typed[0], f"typed mid-dictation: {typed!r}; final {finals!r}"


def test_payload_not_typed_mid_dictation_when_whisper_is_slower_than_realtime(tmp_path):
    """When whisper runs slower than real time (small.en, a loaded GPU, swapping: the demo itself warns
    'slower than real time'), every window arrives > 600 ms after the previous one, so every partial
    passes the payload-silence gate."""
    n = int((3.0 + 2.5) / 0.5)
    texts = _pad(["Type", "Type hello", "Type hello world", "Type hello world how", "Type hello world how are you."], n)
    ctl, ex, finals = drive_whisper(tmp_path, texts, 3.0, latency_ms=700)
    typed = [a.text for a in ex.actions if a.kind == ActionKind.TYPE_TEXT]
    assert len(typed) == 1 and "how are you" in typed[0], f"typed mid-dictation: {typed!r}; final {finals!r}"


def test_url_not_opened_on_one_stale_whisper_window(tmp_path):
    """Same gate, worse outcome: policy.py promises "go to wikipedia" may still become "wikipedia dot
    org", yet one repeated window opens https://wikipedia.com (a different, possibly squatted site)
    and the real "dot org" is then ignored."""
    n = int((3.0 + 2.5) / 0.5)
    texts = _pad(["Go", "Go to Wikipedia", "Go to Wikipedia", "Go to Wikipedia dot org."], n)
    ctl, ex, finals = drive_whisper(tmp_path, texts, 3.0)
    urls = [a.url for a in ex.actions if a.kind == ActionKind.OPEN_URL]
    assert urls == ["https://wikipedia.org"], f"opened {urls!r} for {finals!r}"


# ---------------------------------------------------------------- 2. whisper repetition re-runs the last command


def test_whisper_repetition_after_final_does_not_rerun_command(tmp_path):
    """Greedy whisper decoding on the trailing silence of an 8 s line commonly repeats the last phrase
    ("Press enter. Press enter."). The tracker skips the finished words and starts a NEW utterance
    with the repeat, the stream's echo window (0.35 s after a final) has passed, so the command runs
    again: Return is pressed three times for one spoken "press enter"."""
    n = int((3.0 + 2.5) / 0.5)
    texts = _pad(["Press", "Press enter.", "Press enter.", "Press enter.", "Press enter.",
                  "Press enter. Press enter.", "Press enter. Press enter.", "Press enter. Press enter.",
                  "Press enter. Press enter.", "Press enter. Press enter. Press enter."], n)
    ctl, ex, finals = drive_whisper(tmp_path, texts, 3.0)
    keys = [a.key for a in ex.actions if a.kind == ActionKind.PRESS_KEY]
    assert keys == ["return"], f"one spoken 'press enter' pressed Return {len(keys)}x; finals {finals!r}"


# ---------------------------------------------------------------- 3. confirmations that the user never gave


@pytest.mark.parametrize("ack", ["Yeah.", "Yeah, yeah.", "Sure."])
def test_backchannel_after_proposal_never_confirms(tmp_path, ack):
    """After "Quit TextEdit." whisper keeps re-transcribing the same 8 s line while the user is silent;
    words it appends there become a new utterance. "Yeah." is a classic whisper breath/noise
    hallucination and, like "Sure.", the backchannel people say in side talk; AFFIRM_RE accepts both.
    ("Okay." is handled: it is a FILLER and in HALLUCINATIONS.) The risky quit is executed without
    anyone saying the documented word "confirm"."""
    n = int((3.0 + 2.5) / 0.5)
    texts = _pad(_pad(["Quit", "Quit TextEdit.", "Quit TextEdit."], 6) + [f"Quit TextEdit. {ack}"], n)
    ctl, ex, finals = drive_whisper(tmp_path, texts, 3.0)
    assert ex.kinds == [], f"{ack!r} confirmed: {ex.kinds}; finals {finals!r}"


def test_whisper_stall_splits_one_breath_into_proposal_and_confirmation():
    """'quit textedit yes do it' said in ONE breath must never confirm (safety.confirmation_allowed).
    If whisper stalls for 2.5 s (memory pressure / GPU contention on this 8 GB M1), the tracker's
    stall rule finalizes "Quit TextEdit" mid-breath; the delayed window then delivers "yes, do it"
    as a new uid, which passes every confirmation check (other uid => no reaction-time check), and
    TextEdit is quit."""
    steps: list[tuple] = [(0.5, "window", "Quit"), (1.0, "window", "Quit TextEdit")]
    steps += [(round(1.1 + 0.1 * k, 2), "tick") for k in range(26)]  # no window for 2.6 s: stall final
    steps += [(3.75, "window", "Quit TextEdit, yes, do it."), (4.25, "window", "Quit TextEdit, yes, do it."),
              (4.75, "window", "Quit TextEdit, yes, do it."), (5.25, "window", "Quit TextEdit, yes, do it.")]
    ctl, ex = drive_scripted(steps)
    assert ex.kinds == [], f"one breath proposed AND confirmed: {ex.kinds}"


def test_speaker_change_marker_is_not_merged_into_the_users_speech():
    """whisper marks a speaker turn with '>>'; clean_text deletes the marker, so another person's
    "Yes, do it." becomes the user's words (and can confirm a pending risky action)."""
    tr = UtteranceTracker()
    evs = []
    t = 0.0
    for w in ["Quit TextEdit.", "Quit TextEdit.", "Quit TextEdit.", "Quit TextEdit.",
              "Quit TextEdit. >> Yes, do it.", "Quit TextEdit. >> Yes, do it.", "Quit TextEdit. >> Yes, do it."]:
        t += 0.5
        evs += tr.window(w, t)
    evs += tr.close(t + 5)
    leaked = [e.text for e in evs if e.kind in ("partial", "final") and "do it" in e.text]
    assert leaked == [], f"other speaker's words emitted as the user's: {leaked!r}"


# ---------------------------------------------------------------- 4. deny list gaps (terminals, agents, password managers)


@pytest.mark.parametrize("app", ["kitty", "Alacritty", "WezTerm", "Hyper", "Tabby"])
def test_typing_is_denied_in_every_terminal_emulator(app):
    """safety.TERMINAL_RE knows kitty/Alacritty/WezTerm/Hyper (Return is denied there), but
    DENY_APP_PATTERNS does not, so dictated text is typed straight into a shell prompt; Tabby is in
    neither list, so 'type ... and press enter' runs a shell command."""
    from jev_local.harness.execute import Executor

    snap = make_snapshot(app, [("text area", "shell")], focused="shell")
    assert decide("type ls", TYPE_LS, snap).verdict == "deny"
    assert decide("press enter", PRESS_RETURN, snap).verdict == "deny"
    r = Executor(None, CFG).run(Action(ActionKind.TYPE_TEXT, "u1+0", 0.9, text="ls"), snap, threading.Event())
    assert not r.ok, f"executor (defense in depth) would type into {app}: {r.detail}"


@pytest.mark.parametrize("app", ["Claude", "Script Editor", "Automator"])
def test_typing_is_denied_in_apps_that_execute_what_is_typed(app):
    """DEMO.md has the user launch the demo from Claude Code ('Accessibility is already granted to the
    processes Claude Code starts'): the Claude desktop app is frontmost at start and hosts an agent and
    a shell panel, Script Editor/Automator run what is typed on cmd+r (a catalog key). None is
    deny-listed: 'type <command>' + 'press enter' / 'press command r' executes code with no confirmation."""
    snap = make_snapshot(app, [("text area", "Message input")], focused="Message input")
    assert decide("type ls", TYPE_LS, snap).verdict == "deny"
    assert decide("press enter", PRESS_RETURN, snap).verdict == "deny"


@pytest.mark.parametrize("app", ["KeePassXC", "Keeper Password Manager", "Proton Pass", "NordPass", "Enpass"])
def test_password_managers_are_deny_listed(app):
    """DEMO.md: 'Never: ... password managers'. DENY_APP_PATTERNS only has 1password/bitwarden/
    lastpass/dashlane/'passwords' (Apple), so other managers can be opened, clicked and typed into."""
    from jev_local.harness.safety import is_denied_app

    assert is_denied_app(app, CFG), f"{app} is not deny-listed"


# ---------------------------------------------------------------- 5. destructive Return without confirmation


@pytest.mark.parametrize("app, field", [("Messages", "iMessage"), ("Slack", "Message #general"),
                                        ("Discord", "Message @alex")])
def test_return_in_a_message_composer_needs_confirmation(app, field):
    """Return in a chat composer SENDS the message (to real people), but classify_risk only looks at
    the focused control's label for risk words, and 'iMessage' / 'Message #general' have none:
    'press enter' (also misheard or side talk) sends with no confirmation."""
    snap = make_snapshot(app, [("text field", field, {"value": "running late, sorry"}), ("button", "Emoji")],
                         focused=field)
    assert decide("press enter", PRESS_RETURN, snap).verdict == "confirm"


def test_return_with_a_destructive_default_button_needs_confirmation():
    """In a macOS alert Return activates the DEFAULT button, not the focused control. With Finder's
    'permanently erase the items in the Trash?' alert up, 'press enter' empties the Trash: the risk
    check sees no focused element and the visible 'Empty Trash' button is never considered."""
    snap = make_snapshot("Finder", [("button", "Cancel", {"context": "dialog"}),
                                    ("button", "Empty Trash", {"context": "dialog"})],
                         window_title="Are you sure you want to permanently erase the items in the Trash?")
    assert decide("press enter", PRESS_RETURN, snap).verdict == "confirm"


def test_undo_of_open_url_never_closes_another_apps_window():
    """'undo that' after 'open wikipedia dot org' posts cmd+w to whatever app is frontmost NOW (the
    open_url undo token has no pid/bundle): with Mail in front it closes the message being written.
    Undo is LOW risk, so no confirmation, and Executor.undo skips validate()/_front_guard."""
    from tests.test_execute import FakeObserver, FakeUI, Front, live

    rec = ActionRecord("open wikipedia dot org", Action(ActionKind.OPEN_URL, "u1+0", 0.9, url="https://wikipedia.org"),
                       "ok", 0.0, {"kind": "open_url"})
    ui = FakeUI()
    live(FakeObserver(front=Front(4242, "Mail", "com.apple.mail")), ui).undo(rec)
    assert ("key", "cmd+w", 4242) not in ui.calls, f"closed a Mail window: {ui.writes()}"


# ---------------------------------------------------------------- 6. demo entry point: gates and kill switch


@pytest.fixture
def no_live_world(monkeypatch):
    """Whatever a (fixed or unfixed) demo does, the screen is a fake and the executor only records."""
    from jev_local import demo
    from jev_local.harness.fakes import FakeObserver

    rec = FakeExecutor()
    monkeypatch.setattr(demo, "_make_world", lambda args, cfg: (FakeObserver(), rec, lambda: []))
    return rec



def test_live_enter_gate_does_not_accept_eof(monkeypatch, no_live_world):
    """`--live` 'Press Enter to start' reads stdin; on a non-interactive stdin (an agent's shell, a
    pipe, nohup) readline() returns '' at once and live control starts with nobody having pressed
    Enter. The demo should refuse instead."""
    from jev_local import demo

    reached = []

    def no_engine(*a, **k):
        reached.append(True)
        raise RuntimeError("live session started without an Enter key press")

    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(demo, "build_engines", no_engine)
    with pytest.raises(SystemExit):
        demo.main(["--live", "--text", "press enter", "--no-log", "--no-killswitch", "--no-warm"])
    assert not reached and no_live_world.actions == []


@pytest.mark.parametrize("flags, mode", [(["--no-killswitch"], None), ([], "off")])
def test_live_mode_requires_a_working_global_kill_switch(monkeypatch, no_live_world, flags, mode):
    """In --live the demo moves focus to other apps, so Ctrl-C (terminal-only) is gone; the only stop is
    the global ⌃⌥⎋ listener. `--live --no-killswitch`, or a KillSwitch that comes up "off" (no pyobjc
    Quartz), still starts live control with no way to stop it from the app being driven."""
    from jev_local import demo
    from jev_local.harness import killswitch

    class DeadSwitch:
        def __init__(self, on_trigger, **kw):
            pass

        def start(self):
            return mode

        def stop(self):
            pass

    monkeypatch.setattr(killswitch, "KillSwitch", DeadSwitch)
    with pytest.raises(SystemExit):
        demo.main(["--live", "--yes", "--engine", "rule", "--text", "scroll down", "--no-log", "--no-warm",
                   "--word-ms", "50", "--pause-ms", "200", *flags])
    assert no_live_world.actions == [], f"live actions ran without a kill switch: {no_live_world.kinds}"


def test_demo_refuses_to_start_without_memory_headroom(monkeypatch, no_live_world):
    """This 8 GB M1 has crashed from our workloads; the demo adds whisper-stream (~300 MB, Metal) and
    torch (~500 MB) with no memory gate (it only prints 'RAM free'). Under a swap storm the kill
    switch and Ctrl-C handling stall too, while live actions may be in flight."""
    from jev_local import demo

    reached = []

    def no_engine(*a, **k):
        reached.append(True)
        raise RuntimeError("engine loaded with 0.4 GB free")

    monkeypatch.setattr(demo, "ram_free_gb", lambda: 0.4)
    monkeypatch.setattr(demo, "build_engines", no_engine)
    with pytest.raises(SystemExit):
        demo.main(["--text", "open notes", "--no-log", "--no-killswitch", "--no-warm"])
    assert not reached


def test_kill_switch_latches_the_controller(monkeypatch, tmp_path):
    """⌃⌥⎋ / Ctrl-C call Controller.cancel_all(), which is *Escape* (keep listening): the controller
    still decides and executes every transcript event delivered during the up-to-3 s shutdown, e.g.
    the window whisper-stream had already written to the pipe when it was SIGKILLed (the reader
    drains the pipe before EOF). After the kill switch nothing may execute."""
    from jev_local import demo
    from jev_local.harness import killswitch
    from jev_local.harness.types import TranscriptEvent

    switch: dict[str, Any] = {}

    class FakeSwitch:
        def __init__(self, on_trigger, **kw):
            switch["fire"] = on_trigger

        def start(self):
            return "fake"

        def stop(self):
            pass

    class PipeRaceSource:
        """WhisperSource stand-in: the user says "press enter", hits ⌃⌥⎋, and the window carrying the
        words was already in the pipe; like the real reader it is delivered, then EOF."""

        def __init__(self, model, **kw):
            self.clock = kw["clock"]
            self.info = ws.SourceInfo()
            self.stats = ws.TrackerStats()
            self._q: asyncio.Queue = asyncio.Queue()

        async def start(self):
            self.info.started_at = self.info.ready_at = self.clock()
            self._producer = asyncio.get_running_loop().create_task(self._produce())

        async def _produce(self):
            await asyncio.sleep(0.3)
            switch["fire"]()  # ⌃⌥⎋ (call_soon_threadsafe(kill, ...))
            await asyncio.sleep(0)
            await asyncio.sleep(0)  # kill() has run: cancel_all, kill_now, stop.set()
            for kind, text in (("speech_start", ""), ("partial", "Press enter."), ("final", "Press enter.")):
                self._q.put_nowait(TranscriptEvent(kind, 0, "w1", text, self.clock()))  # type: ignore[arg-type]

        def kill_now(self):
            pass

        async def stop(self):
            await self._producer  # the reader drained the pipe before EOF
            self._q.put_nowait(None)

        async def events(self):
            while True:
                ev = await self._q.get()
                if ev is None:
                    return
                yield ev

    monkeypatch.setattr(killswitch, "KillSwitch", FakeSwitch)
    monkeypatch.setattr(ws, "WhisperSource", PipeRaceSource)
    monkeypatch.setattr(ws, "kill_stale", lambda *a, **k: None)
    rep = tmp_path / "r.json"
    demo.main(["--engine", "rule", "--no-warm", "--no-log", "--whisper-bin", "fake-whisper-stream", "--report", str(rep)])
    r = json.loads(rep.read_text())
    assert r["stats"]["executed"] == 0, f"executed after the kill switch: {r['tap']['utterances']}"
