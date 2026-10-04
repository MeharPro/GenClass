from __future__ import annotations

import io

from jev_local.harness.hud import ConsoleHud
from jev_local.harness.types import Action, ActionKind, Decision, ExecResult, RiskLevel, Tail


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def tail(text: str, uid: str = "u1", final: bool = False, cursor: int = 0) -> Tail:
    return Tail(vid=f"{uid}+0", text=text, words=tuple(text.split()), cursor=cursor, is_final=final, silent_ms=0, uid=uid)


def action(kind=ActionKind.OPEN_APP, **kw) -> Action:
    kw.setdefault("app", "Notes")
    return Action(kind=kind, source_vid="u1+0", confidence=0.91, **kw)


def make(**kw):
    buf, clock = io.StringIO(), Clock()
    return ConsoleHud(stream=buf, color=False, clock=clock, **kw), buf, clock


def test_lines_are_timestamped_from_utterance_start():
    hud, buf, clock = make()
    hud.transcript(tail("open notes"))
    clock.t += 0.42
    hud.decision(Decision("act", action(), "complete command", seq=3, latency_ms=38.0))
    clock.t += 0.1
    hud.executed(action(), ExecResult(True, True, "opened Notes", 45.0))
    clock.t += 1.0
    hud.transcript(tail("open notes and type hello", final=True))
    lines = buf.getvalue().splitlines()
    assert lines[0].startswith("+ 0.00s hear") and '"open notes" [partial]' in lines[0]
    assert lines[1].startswith("+ 0.42s act") and "open_app('Notes') p=0.91 38ms - complete command" in lines[1]
    assert lines[2].startswith("+ 0.52s done") and "opened Notes" in lines[2]
    assert lines[3].startswith("+ 1.52s hear") and "[final]" in lines[3]
    # a new physical utterance resets the clock
    clock.t += 5
    hud.transcript(tail("scroll down", uid="u2"))
    assert buf.getvalue().splitlines()[-1].startswith("+ 0.00s hear")


def test_single_lines_only():
    hud, buf, _ = make()
    hud.transcript(tail("type this\nand that"))
    hud.error("boom\nsecond line")
    assert len(buf.getvalue().splitlines()) == 2


def test_repeated_waits_are_collapsed():
    hud, buf, _ = make()
    for _ in range(3):
        hud.decision(Decision("wait", None, "intent below threshold", seq=1))
    hud.decision(Decision("wait", None, "incomplete", seq=2))
    hud.decision(Decision("act", action(), "ok", seq=3))
    hud.decision(Decision("wait", None, "incomplete", seq=4))
    out = buf.getvalue().splitlines()
    assert len(out) == 4
    hud2, buf2, _ = make(quiet_waits=False)
    for _ in range(3):
        hud2.decision(Decision("wait", None, "same", seq=1))
    assert len(buf2.getvalue().splitlines()) == 3


def test_pending_dry_run_failure_error_risk():
    hud, buf, _ = make()
    quit_ = action(ActionKind.QUIT_APP, app="Mail")
    hud.decision(Decision("confirm", quit_, "high risk", seq=1, risk=RiskLevel.HIGH))
    hud.pending(quit_)
    hud.executed(action(), ExecResult(True, False, "would open 'Notes'", 0.3, dry_run=True))
    hud.executed(action(ActionKind.CLICK, target_label='button "Send"'), ExecResult(False, False, "target gone", 3.0))
    hud.executed(action(ActionKind.PRESS_KEY, key="return"), ExecResult(True, False, "pressed return", 3.0))
    hud.error("AX permission missing")
    out = buf.getvalue().splitlines()
    assert "confirm" in out[0] and "[HIGH]" in out[0]
    assert "confirm?" in out[1] and 'say "confirm" or "cancel"' in out[1]
    assert "dry-run" in out[2] and "would open" in out[2]
    assert "FAILED" in out[3] and "target gone" in out[3]
    assert "(no visible change)" in out[4]
    assert "error" in out[5] and "permission" in out[5]


def test_color_and_never_raises():
    buf = io.StringIO()
    hud = ConsoleHud(stream=buf, color=True, clock=Clock())
    hud.decision(Decision("deny", None, "denied app", seq=1))
    assert "\033[31m" in buf.getvalue()  # red
    # broken inputs must not propagate out of the HUD
    hud.transcript(None)  # type: ignore[arg-type]
    hud.decision(object())  # type: ignore[arg-type]
    hud.executed(None, None)  # type: ignore[arg-type]
    hud.pending(None)  # type: ignore[arg-type]


def test_default_stream_is_stderr_without_tty_color(capsys):
    hud = ConsoleHud()
    hud.error("to stderr")
    captured = capsys.readouterr()
    assert "to stderr" in captured.err and "\033[" not in captured.err
