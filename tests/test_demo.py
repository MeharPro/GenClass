"""Demo entry point, simulated screen and kill switch. No torch, no microphone, no UI actions."""

from __future__ import annotations

import asyncio
import io
import json
import threading
import time

import pytest

from jev_local import demo
from jev_local.harness.simworld import SimExecutor, SimWorld
from jev_local.harness.types import Action, ActionKind, ExecResult, TranscriptEvent


def _act(kind, **kw):
    return Action(kind=kind, source_vid="u1+0", confidence=0.9, **kw)


def test_simworld_follows_actions():
    w = SimWorld()
    assert w.snapshot().app_name == "Finder"
    w.apply(_act(ActionKind.OPEN_APP, app="TextEdit"))
    s = w.snapshot()
    assert s.app_name == "TextEdit" and s.focused_eid is not None
    w.apply(_act(ActionKind.TYPE_TEXT, text="hello world"))
    assert any(e.value == "hello world" for e in w.snapshot().elements)
    w.apply(_act(ActionKind.OPEN_URL, url="https://wikipedia.org"))
    assert w.snapshot().app_name == "Safari"
    w.apply(_act(ActionKind.QUIT_APP, app="Safari"))
    assert w.front == "TextEdit"
    assert w.running_apps()[0] == "TextEdit"


def test_simexecutor_only_updates_on_ok():
    class Inner:
        def __init__(self, ok):
            self.ok = ok

        def run(self, a, snap, cancel):
            return ExecResult(ok=self.ok, changed=False, detail="x", elapsed_ms=0.0, dry_run=True)

    w = SimWorld()
    SimExecutor(Inner(False), w).run(_act(ActionKind.OPEN_APP, app="Notes"), None, threading.Event())
    assert w.front == "Finder"
    SimExecutor(Inner(True), w).run(_act(ActionKind.OPEN_APP, app="Notes"), None, threading.Event())
    assert w.front == "Notes"


def test_parse_args_defaults_and_guards():
    a = demo.parse_args([])
    assert not a.live and a.screen == "sim" and a.model == "v1" and a.device == "cpu" and a.whisper_model == "base.en"
    assert demo.parse_args(["--live"]).screen == "real"
    with pytest.raises(SystemExit):
        demo.parse_args(["--live", "--screen", "sim"])
    with pytest.raises(SystemExit):
        demo.parse_args(["--text", "x", "--say", "y"])


def test_session_tap_word_index_and_before_final():
    t = [0.0]
    out = io.StringIO()
    tap = demo.SessionTap(lambda: t[0], out=out)

    def ev(kind, text=""):
        return TranscriptEvent(kind=kind, seq=0, uid="w1", text=text, t_mono=t[0])

    tap.event(ev("speech_start"))
    for k, words in enumerate(["open", "open notes", "open notes and", "open notes and scroll down"], 1):
        t[0] = k * 0.3
        tap.event(ev("partial", words))
        if k == 2:
            tap.executed(_act(ActionKind.OPEN_APP, app="Notes"),
                         ExecResult(ok=True, changed=False, detail="would open", elapsed_ms=0, dry_run=True))
    t[0] = 2.0
    tap.event(ev("final", "open notes and scroll down"))
    rep = tap.report()
    a = rep["utterances"][0]["actions"][0]
    assert a["word"] == "2/5" and a["before_final"] is True
    assert a["ms_after_end_of_speech"] == pytest.approx(-600.0)
    assert "fired at word 2 of 5" in out.getvalue()
    assert rep["closed_set_before_final"] == 1


def test_memory_helpers():
    assert demo.rss_mb() and demo.rss_mb() > 1
    assert demo.ram_free_gb() is None or demo.ram_free_gb() >= 0


@pytest.mark.macos
def test_demo_text_replay_rule_engine(tmp_path, capsys):
    rep_path = tmp_path / "r.json"
    rc = demo.main(["--engine", "rule", "--no-warm", "--no-log", "--no-killswitch", "--word-ms", "120", "--pause-ms", "1000",
                    "--text", "open textedit and type hello world", "--text", "quit textedit", "--text", "confirm",
                    "--text", "hey can you pass the salt", "--report", str(rep_path)])
    assert rc == 0
    r = json.loads(rep_path.read_text())
    utts = r["tap"]["utterances"]
    assert [a["kind"] for a in utts[0]["actions"]] == ["open_app", "type_text"]
    assert utts[0]["actions"][0]["before_final"] is True
    assert utts[1]["actions"] == [] and utts[1]["verdicts"].get("confirm") == 1
    assert [a["kind"] for a in utts[2]["actions"]] == ["quit_app"]
    assert utts[3]["actions"] == []
    assert all(a["outcome"] == "dry-run" for u in utts for a in u["actions"])


@pytest.mark.macos
def test_killswitch_starts_and_stops():
    from jev_local.harness.killswitch import KillSwitch

    hits = []
    ks = KillSwitch(lambda: hits.append(1), prefer_tap=False)
    assert ks.start() == "poll"
    time.sleep(0.1)
    ks.stop()
    assert hits == []  # nobody pressed control-option-escape


@pytest.mark.macos
def test_demo_mic_path_with_fake_whisper_stream(tmp_path):
    """The microphone code path end to end, with a fake whisper-stream printing scripted windows."""
    import shlex
    import sys

    from tests.test_whisper_source import FAKE_STREAM

    exe = tmp_path / "fake_stream.py"
    exe.write_text(FAKE_STREAM)
    steps = ["[BLANK_AUDIO]", "Open", "Open notes", "Open notes and", "Open Notes and scroll",
             "Open notes and scroll down.", *["Open notes and scroll down."] * 14]
    sc = tmp_path / "s.json"
    sc.write_text(json.dumps({"steps": steps, "dt": 0.25}))
    rep_path = tmp_path / "r.json"
    rc = demo.main(["--engine", "rule", "--no-warm", "--no-log", "--no-killswitch", "--report", str(rep_path),
                    "--whisper-bin", f"{shlex.quote(sys.executable)} {shlex.quote(str(exe))} {shlex.quote(str(sc))}"])
    assert rc == 0
    r = json.loads(rep_path.read_text())
    u = r["tap"]["utterances"]
    assert len(u) == 1
    acts = u[0]["actions"]
    assert [a["kind"] for a in acts] == ["open_app", "scroll_down"]
    assert acts[0]["before_final"] is True
    assert r["stats"]["executed"] == 2
