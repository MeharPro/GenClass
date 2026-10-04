"""Console HUD: one compact, colored line per event on stderr.

Timestamps are seconds since the start of the current physical utterance, so a line like
`+0.84s act open_app('Notes')` next to `+1.30s hear ... [final]` shows at a glance whether the
action fired mid-sentence. A HUD must never break the loop, so every method swallows its errors.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Callable, TextIO

from jev_local.harness.types import Action, Decision, ExecResult, RiskLevel, Tail

_RESET = "\033[0m"
_STYLE = {
    "dim": "\033[2m",
    "bold": "\033[1m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "blue": "\033[34m",
}
_VERDICT_STYLE = {
    "act": "green",
    "confirm": "yellow",
    "clarify": "magenta",
    "deny": "red",
    "wait": "dim",
    "ignore": "dim",
}


def _clip(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


class ConsoleHud:
    def __init__(
        self,
        stream: TextIO | None = None,
        color: bool | None = None,
        clock: Callable[[], float] = time.monotonic,
        quiet_waits: bool = True,
        width: int = 100,
    ):
        self.stream = stream if stream is not None else sys.stderr
        if color is None:
            color = bool(getattr(self.stream, "isatty", lambda: False)()) and "NO_COLOR" not in os.environ
        self.color = color
        self.clock = clock
        self.quiet_waits = quiet_waits  # repeated identical wait/ignore lines are dropped
        self.width = width
        self._t0: float | None = None
        self._uid: str | None = None
        self._last_wait: tuple[str, str] | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------ public API

    def transcript(self, tail: Tail) -> None:
        try:
            uid = tail.uid or tail.vid.split("+", 1)[0]
            if uid != self._uid:
                self.start_utterance(uid)
            state = "final" if tail.is_final else "partial"
            cursor = f" @{tail.cursor}" if tail.cursor else ""
            silence = f" {tail.silent_ms}ms silent" if tail.silent_ms else ""
            text = _clip(tail.text, self.width - 30)
            self._line("hear", f'"{text}" [{state}{cursor}{silence}]', "bold" if tail.is_final else "dim")
        except Exception:
            pass

    def decision(self, decision: Decision) -> None:
        try:
            v = decision.verdict
            if v in ("wait", "ignore") and self.quiet_waits:
                key = (v, decision.reason)
                if key == self._last_wait:
                    return
                self._last_wait = key
            else:
                self._last_wait = None
            parts = []
            if decision.action is not None:
                parts.append(f"{decision.action.describe()} p={decision.action.confidence:.2f}")
            if decision.risk >= RiskLevel.HIGH:
                parts.append(f"[{decision.risk.name}]")
            if decision.latency_ms:
                parts.append(f"{decision.latency_ms:.0f}ms")
            if decision.reason:
                parts.append(f"- {decision.reason}")
            if decision.retry_in_ms:
                parts.append(f"(retry {decision.retry_in_ms}ms)")
            self._line(v, _clip(" ".join(parts), self.width), _VERDICT_STYLE.get(v, "blue"))
        except Exception:
            pass

    def pending(self, action: Action) -> None:
        try:
            self._line("confirm?", f'{action.describe()} - say "confirm" or "cancel"', "yellow", bold=True)
        except Exception:
            pass

    def executed(self, action: Action, result: ExecResult) -> None:
        try:
            if result.dry_run:
                tag, style = "dry-run", "cyan"
            elif result.ok:
                tag, style = "done", "green"
            else:
                tag, style = "FAILED", "red"
            body = f"{action.describe()} {result.elapsed_ms:.0f}ms - {result.detail}"
            if result.ok and not result.dry_run and not result.changed:
                body += " (no visible change)"
            self._line(tag, _clip(body, self.width), style, bold=not result.ok)
        except Exception:
            pass

    def error(self, msg: str) -> None:
        try:
            self._line("error", _clip(str(msg), self.width), "red", bold=True)
        except Exception:
            pass

    def start_utterance(self, uid: str | None = None) -> None:
        """Reset the clock origin (called automatically when a new physical utterance appears)."""
        self._uid = uid
        self._t0 = self.clock()
        self._last_wait = None

    # ------------------------------------------------------------ output

    def _paint(self, s: str, style: str, bold: bool = False) -> str:
        if not self.color:
            return s
        pre = _STYLE.get(style, "") + (_STYLE["bold"] if bold and style != "bold" else "")
        return f"{pre}{s}{_RESET}" if pre else s

    def _line(self, tag: str, body: str, style: str, bold: bool = False) -> None:
        now = self.clock()
        if self._t0 is None:
            self._t0 = now
        ts = f"+{now - self._t0:5.2f}s"
        line = f"{self._paint(ts, 'dim')} {self._paint(f'{tag:<8}', style, bold)} {self._paint(body, style if style != 'bold' else '', False)}"
        with self._lock:
            self.stream.write(line + "\n")
            self.stream.flush()
