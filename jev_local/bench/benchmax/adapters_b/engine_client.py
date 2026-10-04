"""Answerers: turn an Item's request into Jev-shaped answers, through the local HTTP server or in process.

- `HttpAnswerer`: POST /v1/systemone (any server speaking the wire format; the model id is added here, W10).
- `LocalAnswerer` (VM only): `jev_local.bench.ours.OursRunner` for exact option/question chunking (truncation of
  over-long states is allowed and COUNTED for the group-B suites: PLAN §4.2 item 5), then
  `jev_local.confidence.build_answer` for the wire shapes (choice/confidence/score/legend, W1-W2) with 12-digit
  rounding, i.e. effectively unrounded (W3).
- `run_items`: resumable loop writing one JSONL record per item ({"id", "ok", "answers", "usage", "ms", "error"}).

Both answerers return the same record shape; failures are recorded, never raised past the loop.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

from jev_local.bench.benchmax.adapters_b.common import MODEL_ID, Item, read_jsonl

Record = dict[str, Any]


class AnswerError(RuntimeError):
    def __init__(self, status: int, body: Any):
        super().__init__(f"HTTP {status}: {str(body)[:300]}")
        self.status = status
        self.body = body


class Answerer(Protocol):
    name: str

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]: ...


class HttpAnswerer:
    """POST to a jev-local (or any wire-compatible) server. `model` defaults to meharsjev-68m (W10)."""

    def __init__(self, base_url: str = "http://127.0.0.1:8765", model: str = MODEL_ID, timeout: float = 600.0,
                 api_key: str | None = None):
        import httpx

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"http:{self.base_url}:{model}"
        headers = {"content-type": "application/json"}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(timeout=timeout, headers=headers)

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        body = {"state": request["state"], "model": self.model, "questions": request["questions"]}
        r = self._client.post(self.base_url + "/v1/systemone", json=body)
        if r.status_code != 200:
            try:
                detail = r.json()
            except ValueError:
                detail = r.text
            raise AnswerError(r.status_code, detail)
        data = r.json()
        return {"answers": data["answers"], "usage": data.get("usage", {}), "model": data.get("model"),
                "truncated": False}


class LocalAnswerer:
    """In-process checkpoint (VM only: imports torch). Exact chunking; long states truncated and flagged."""

    def __init__(self, ckpt: str | Path, threads: int = 8, max_tokens: int | None = None, round_digits: int = 12,
                 model: str = MODEL_ID):
        from jev_local.bench.ours import OursRunner

        self.runner = OursRunner(ckpt, threads=threads, max_tokens=max_tokens)
        self.round_digits = round_digits
        self.model = model
        self.name = f"local:{Path(ckpt).name}:{model}"

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        from jev_local.confidence import build_answer
        from jev_local.engine.base import RawDist
        from jev_local.schema import question_from_json

        raw, info = self.runner.answer(request)
        answers: dict[str, Any] = {}
        for qid, qjson in request["questions"].items():
            q = question_from_json(qjson)
            a = raw[qid]
            if a["type"] == "noul":
                dist = RawDist("noul", (float(a["noul"]),))
            elif a["type"] == "choice":
                dist = RawDist("choice", tuple(float(v) for v in a["probabilities"].values()),
                               tuple(a["probabilities"].keys()))
            else:
                dist = RawDist("score", tuple(float(v) for v in a["probabilities"].values()))
            answers[qid] = build_answer(q, dist, self.round_digits).model_dump(mode="json")
        return {"answers": answers, "usage": {"input_tokens": int(info["input_tokens"]), "passes": info["passes"]},
                "model": self.model, "truncated": bool(info["truncated"])}


def run_items(items: Sequence[Item], answerer: Callable[[dict[str, Any]], dict[str, Any]], out_path: str | Path,
              resume: bool = True, log_every: int = 50, log: Callable[[str], None] = print) -> dict[str, Record]:
    """Answer every item once; append-only JSONL so a crash resumes. Returns id -> record."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, Record] = {}
    if resume and out_path.exists():
        for rec in read_jsonl(out_path):
            if rec.get("ok") or rec.get("permanent"):
                done[rec["id"]] = rec
    todo = [it for it in items if it.id not in done]
    log(f"{out_path.name}: {len(done)} cached, {len(todo)} to answer")
    n_err = 0
    t_start = time.perf_counter()
    with out_path.open("a", encoding="utf-8") as f:
        for i, it in enumerate(todo, 1):
            t0 = time.perf_counter()
            try:
                res = answerer(it.request)
                rec = {"id": it.id, "ok": True, "answers": res["answers"], "usage": res.get("usage", {}),
                       "model": res.get("model"), "truncated": bool(res.get("truncated", False)),
                       "ms": (time.perf_counter() - t0) * 1000.0}
            except AnswerError as e:
                n_err += 1
                rec = {"id": it.id, "ok": False, "permanent": 400 <= e.status < 500, "status": e.status,
                       "error": str(e)[:500], "ms": (time.perf_counter() - t0) * 1000.0}
            except Exception as e:  # noqa: BLE001 - one bad request must not stop the run
                n_err += 1
                rec = {"id": it.id, "ok": False, "permanent": True, "error": f"{type(e).__name__}: {str(e)[:400]}",
                       "ms": (time.perf_counter() - t0) * 1000.0}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[it.id] = rec
            if (log_every and i % log_every == 0) or i == len(todo):
                el = time.perf_counter() - t_start
                log(f"  {i}/{len(todo)} answered, {n_err} errors, {el:.0f}s ({i / el:.2f}/s)")
    return done


def load_answers(path: str | Path) -> dict[str, Record]:
    out: dict[str, Record] = {}
    for rec in read_jsonl(path):
        prev = out.get(rec["id"])
        if prev is None or (not prev.get("ok") and rec.get("ok")):
            out[rec["id"]] = rec
    return out


def truncation_counts(records: Iterable[Record]) -> dict[str, int]:
    recs = list(records)
    return {"requests": len(recs), "truncated": sum(1 for r in recs if r.get("ok") and r.get("truncated")),
            "failed": sum(1 for r in recs if not r.get("ok"))}
