"""Local audit log: every harness decision and execution, one JSON object per line (SPEC §4.7 layer 8).

Files are date-stamped (`YYYY-MM-DD.jsonl`, local date) under `~/.jev-local/log` by default and
never leave the machine. With `redact=True` (the default), what the user dictates is not kept:

- an action's `text` (typed or searched payload) becomes `<n chars>`;
- the `text_span` answer (the candidate payloads) is replaced the same way;
- in transcript-like fields (`text`, `tail`, `transcript`, `said`, ...), everything after a
  payload verb ("type", "write", "search for", ...) becomes `<n chars>`.

Writing never raises: a logging failure must not stop the harness, so the log disables itself
after the first OS error and remembers why.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import json
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from jev_local.schema import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneResponse

DEFAULT_DIR = "~/.jev-local/log"
TRANSCRIPT_KEYS = frozenset({"text", "tail", "transcript", "said", "partial", "consumed", "current"})
_PAYLOAD_VERB_RE = re.compile(
    r"\b(?:type(?:\s+(?:in|out))?|write(?:\s+down)?|dictate|put\s+in|fill\s+in|enter|"
    r"search(?:\s+(?:the\s+web|google|online))?(?:\s+for)?|look\s+up|google)\b[\s,:]*",
    re.IGNORECASE,
)
TOP_K = 3  # probabilities kept per choice answer (a 60-element target distribution is noise in a log)


def redact_transcript(s: str) -> str:
    """'open notes and type my pin is 1234' -> 'open notes and type <15 chars>'."""
    m = _PAYLOAD_VERB_RE.search(s)
    if not m or m.end() >= len(s):
        return s
    return s[: m.end()] + f"<{len(s) - m.end()} chars>"


def compact_answers(resp: SystemOneResponse) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for qid, a in resp.answers.items():
        if isinstance(a, NoulAnswer):
            out[qid] = a.noul
        elif isinstance(a, ChoiceAnswer):
            top = sorted(a.probabilities.items(), key=lambda kv: -kv[1])[:TOP_K]
            out[qid] = {"choice": a.choice, "confidence": a.confidence, "top": dict(top)}
        elif isinstance(a, ScoreAnswer):
            out[qid] = {"score": a.score, "confidence": a.confidence}
    return {"model": resp.model, "answers": out, "usage": resp.usage.model_dump()}


def to_jsonable(obj: Any) -> Any:
    """Dataclasses, enums, pydantic models, paths and tuples -> plain JSON values."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return round(obj, 4) if obj == obj else None
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, SystemOneResponse):
        return compact_answers(obj)
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


def _chars(s: str) -> str:
    return f"<{len(s)} chars>"


def redact(obj: Any, key: str | None = None) -> Any:
    """Redact a jsonable structure (see module docstring)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "text_span" and isinstance(v, dict):
                out[k] = {
                    "choice": _chars(v["choice"]) if v.get("choice") not in (None, "none") else v.get("choice"),
                    "confidence": v.get("confidence"),
                }
            elif k == "action" and isinstance(v, dict):
                a = dict(v)
                if isinstance(a.get("text"), str):
                    a["text"] = _chars(a["text"])
                out[k] = redact(a, k)
            else:
                out[k] = redact(v, k)
        return out
    if isinstance(obj, list):
        return [redact(v, key) for v in obj]
    if isinstance(obj, str) and key in TRANSCRIPT_KEYS:
        return redact_transcript(obj)
    return obj


def _collect_secrets(obj: Any, out: set[str]) -> None:
    """Payload strings in a jsonable record: action texts and the chosen text_span."""
    if isinstance(obj, dict):
        t = obj.get("text")
        if "source_vid" in obj and isinstance(t, str) and t:  # an Action
            out.add(t)
        ts = obj.get("text_span")
        if isinstance(ts, dict) and isinstance(ts.get("choice"), str) and ts["choice"] not in ("", "none"):
            out.add(ts["choice"])
        for v in obj.values():
            _collect_secrets(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_secrets(v, out)


def _scrub(obj: Any, secrets: list[str]) -> Any:
    """Replace every occurrence of a payload anywhere, e.g. inside `reason` strings."""
    if isinstance(obj, str):
        for sec in secrets:
            obj = obj.replace(sec, _chars(sec))
        return obj
    if isinstance(obj, dict):
        return {k: _scrub(v, secrets) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub(v, secrets) for v in obj]
    return obj


class AuditLog:
    """Append-only, date-stamped JSONL audit log. `write` never raises and never blocks on disk.

    Records are serialized (and redacted) on the caller's thread, then appended by a background
    writer thread, so a slow or full disk can never stall the decision loop. `flush()` waits for
    the writer; `read()` flushes first.
    """

    def __init__(
        self,
        dir: str | Path = DEFAULT_DIR,
        redact: bool = True,
        enabled: bool = True,
        wallclock: Callable[[], float] = time.time,
        background: bool = True,
    ):
        self.dir = Path(dir).expanduser()
        self.redact = redact
        self.enabled = enabled
        self.wallclock = wallclock
        self.background = background
        self.error: str | None = None
        self.n_written = 0
        self._n_dropped = 0  # records lost to a disk error
        self._n_queued = 0
        self._q: queue.SimpleQueue[tuple[Path, str] | None] = queue.SimpleQueue()
        self._cv = threading.Condition()
        self._thread: threading.Thread | None = None

    def path_for(self, t: float | None = None) -> Path:
        d = dt.datetime.fromtimestamp(self.wallclock() if t is None else t)
        return self.dir / f"{d:%Y-%m-%d}.jsonl"

    def write(self, kind: str, **fields: Any) -> None:
        if not self.enabled:
            return
        try:
            t = self.wallclock()
            rec = {"ts": dt.datetime.fromtimestamp(t).isoformat(timespec="milliseconds"), "kind": kind}
            rec.update(to_jsonable(fields))
            if self.redact:
                secrets: set[str] = set()
                _collect_secrets(rec, secrets)
                rec = redact(rec)
                if secrets:
                    rec = _scrub(rec, sorted(secrets, key=len, reverse=True))
            line = json.dumps(rec, ensure_ascii=False, separators=(",", ":"))
        except Exception as e:  # noqa: BLE001 - a record that cannot be serialized is dropped, not raised
            self.error = f"serialize {kind}: {e!r}"
            return
        item = (self.path_for(t), line)
        if not self.background:
            self._append([item])
            return
        with self._cv:
            self._n_queued += 1
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="jev-audit-log", daemon=True)
                self._thread.start()
        self._q.put(item)

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until everything written so far is on disk (or the log gave up). True if drained."""
        deadline = time.monotonic() + timeout
        with self._cv:
            while self._n_done() < self._n_queued:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(left)
        return True

    def close(self, timeout: float = 5.0) -> None:
        self.flush(timeout)
        if self._thread is not None and self._thread.is_alive():
            self._q.put(None)
            self._thread.join(timeout)

    def read(self, t: float | None = None) -> list[dict[str, Any]]:
        """All records in the file for `t`'s date (tests and `jev-local` tooling)."""
        self.flush()
        p = self.path_for(t)
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    # ------------------------------------------------------------------ writer thread

    def _n_done(self) -> int:
        return self.n_written + self._n_dropped

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            batch = [item]
            while True:  # drain what is queued: one open() per file per burst
                try:
                    nxt = self._q.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    self._append(batch)
                    return
                batch.append(nxt)
            self._append(batch)

    def _append(self, batch: list[tuple[Path, str]]) -> None:
        ok = 0
        if self.enabled or not self.background:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
                by_path: dict[Path, list[str]] = {}
                for path, line in batch:
                    by_path.setdefault(path, []).append(line)
                for path, lines in by_path.items():
                    with open(path, "a", encoding="utf-8") as f:
                        f.write("\n".join(lines) + "\n")
                ok = len(batch)
            except OSError as e:
                self.error = repr(e)
                self.enabled = False
        with self._cv:
            self.n_written += ok
            self._n_dropped += len(batch) - ok
            self._cv.notify_all()
