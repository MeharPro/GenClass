"""Shared benchmax plumbing (PLAN §4.2 item 8 = W1–W11, PLAN §6 logging rules).

Pieces
- `Client`s turn (state, questions) into an `Outcome`: `LocalClient` (a FastEngine checkpoint, VM only),
  `HttpClient` (any /v1/systemone server, e.g. our own), `HeuristicClient` (the model-free engine, for plumbing
  self-checks). All three produce the same wire shapes.
- `wire_answers` (W1–W3): `choice` = argmax of the UNROUNDED probabilities, ties to the first key in request
  order; `confidence` = (K·pmax − 1)/(K − 1); `score` = Σ i·pᵢ with `legend`; probabilities either unrounded
  ("unrounded", leaderboard mode) or 2-dp largest-remainder rounding that sums to exactly 1.00 ("2dp").
- `refusal_body` (W4): `{"detail": [{"type": "max_tokens_exceeded", "msg": "... maximum context length ..."}]}`
  so the Decision Index http engine records `unsupported` (CAPACITY_MARKERS) and Deußer's runner treats it as
  permanent; the option cap says "options per choice". The HTTP status is configurable (400 Jev parity, 422).
- overflow switch (W5): `refuse` (DI, jev-bench) or `truncate` (Deußer, typed-decisions; counted and disclosed).
- `RunContext` writes `runs/benchmax/<ckpt>/<spec>/run.json`: harness commit, dataset revisions, checkpoint
  sha256, calibration file, overflow mode + counts, CPU threads + hardware, wall-clock, request counts, the
  determinism report (W11) and the adapter's results. It is written at start and at the end (a crash keeps
  the partial record).
- `Session` wraps a client: counts outcomes, keeps the first N requests and replays them at the end for W11.

Nothing here imports torch, transformers or datasets at module level.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import platform
import re
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from jev_local.confidence import choice_confidence, normalize, score_confidence, score_value
from jev_local.engine.base import EngineError, RawDist
from jev_local.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, question_from_json

REPO_ROOT = Path(__file__).resolve().parents[3]
TARGETS_PATH = REPO_ROOT / "bench" / "public" / "targets.json"
DEFAULT_WORK = Path(os.environ.get("BENCHMAX_WORK", str(Path.home() / "bench_work")))
RUN_SCHEMA = "benchmax-run/1"

MODEL_ID_RE = re.compile(r"^(?:genclass|meharsjev)-[a-z0-9][A-Za-z0-9._-]*$")  # genclass-<size>; legacy meharsjev-<size>
MAX_TOKENS_EXCEEDED = "max_tokens_exceeded"
CONTEXT_MARKER = "maximum context length"  # Decision Index http.CAPACITY_MARKERS
OPTION_CAP_MARKER = "options per choice"  # idem
TOO_FEW_OPTIONS_MARKER = "a choice needs at least two options"  # idem
SCORE_LEVELS_MARKER = "a score takes 2 to 10 levels"  # idem
CAPACITY_MARKERS = (
    OPTION_CAP_MARKER, TOO_FEW_OPTIONS_MARKER, SCORE_LEVELS_MARKER, "the canvas holds", CONTEXT_MARKER,
    "maximum model length", "longer than the maximum model length", "context window", "too many tokens",
)
MAX_CHOICE_OPTIONS = 255
PRECISIONS = ("unrounded", "2dp")
OVERFLOWS = ("refuse", "truncate")
RETRY_HTTP = (408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529)


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def check_model_id(model_id: str) -> str:
    """W10: `meharsjev-<size>`, never `jev-*`."""
    if model_id.startswith("jev-") or model_id.startswith("typesafe/"):
        raise ValueError(f"model id {model_id!r} must not be a Jev id (W10)")
    if not MODEL_ID_RE.match(model_id):
        raise ValueError(f"model id {model_id!r} must look like genclass-<size> (W10)")
    return model_id


# ---------------------------------------------------------------------------------------- hashing / files


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_dir(path: str | Path, *, exclude: Sequence[str] = ("__pycache__", ".DS_Store")) -> dict[str, Any]:
    """One digest over every file of a directory (sorted relative paths, then contents), plus per-file digests
    of the small metadata files. Used for the checkpoint sha256 of run.json."""
    root = Path(path)
    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and not any(x in p.parts for x in exclude) and not p.name.startswith("._"))  # no AppleDouble sidecars
    h = hashlib.sha256()
    per_file: dict[str, str] = {}
    total = 0
    for p in files:
        rel = p.relative_to(root).as_posix()
        d = sha256_file(p)
        h.update(rel.encode("utf-8") + b"\0" + bytes.fromhex(d) + b"\0")
        total += p.stat().st_size
        if p.suffix in (".json", ".txt") or p.name == "heads.safetensors":
            per_file[rel] = d
    return {"sha256": h.hexdigest(), "files": len(files), "bytes": total, "per_file": per_file}


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def _json_default(o: Any) -> Any:
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


def read_jsonl(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_targets(spec_id: str | None = None, path: str | Path = TARGETS_PATH) -> list[dict]:
    """Rows of bench/public/targets.json, optionally filtered to one spec id."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = d["targets"]
    if spec_id is not None:
        rows = [r for r in rows if r.get("spec_id") == spec_id]
    return rows


def hardware_info(threads: int | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "threads": threads,
    }
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    info["mem_gb"] = round(int(line.split()[1]) / 1e6, 1)
                    break
    except OSError:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    if "torch" in sys.modules:
        try:
            info["torch_threads"] = sys.modules["torch"].get_num_threads()
            info["torch"] = sys.modules["torch"].__version__
        except Exception:
            pass
    return info


# ---------------------------------------------------------------------------------------- W3 precision


def largest_remainder_2dp(probs: Sequence[float]) -> list[float]:
    """Round a distribution to 2 dp so that the rounded values sum to exactly 1.00."""
    if not probs:
        return []
    scaled = [max(0.0, float(p)) * 100.0 for p in probs]
    total = sum(scaled)
    if total <= 0:
        scaled = [100.0 / len(probs)] * len(probs)
    elif abs(total - 100.0) > 1e-9:
        scaled = [x * 100.0 / total for x in scaled]
    floors = [math.floor(x + 1e-9) for x in scaled]
    rem = 100 - sum(floors)
    order = sorted(range(len(scaled)), key=lambda i: (-(scaled[i] - floors[i]), i))
    for i in order[:rem]:
        floors[i] += 1
    return [f / 100.0 for f in floors]


def _ordered_probs(d: RawDist, labels: Sequence[str]) -> list[float]:
    if len(d.probs) != len(labels):
        raise ValueError(f"engine returned {len(d.probs)} probabilities for {len(labels)} options")
    if not d.labels or tuple(d.labels) == tuple(labels):
        return list(d.probs)
    by = dict(zip(d.labels, d.probs))
    return [by[lab] for lab in labels]


def _fmt(x: float, precision: str) -> float:
    x = float(x)
    if x != x:
        x = 0.0
    return round(x, 2) + 0.0 if precision == "2dp" else x + 0.0


def wire_answers(questions: Mapping[str, Any], dists: Mapping[str, RawDist], precision: str = "unrounded") -> dict[str, dict]:
    """Raw distributions -> Jev wire answers (W1–W3). `questions` holds pydantic questions or raw dicts."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    out: dict[str, dict] = {}
    for qid, q in questions.items():
        if isinstance(q, dict):
            q = question_from_json(q)
        d = dists[qid]
        if isinstance(q, NoulQuestion):
            p = d.probs[0] if d.probs else 0.5
            p = min(1.0, max(0.0, p if math.isfinite(p) else 0.5))
            out[qid] = {"type": "noul", "noul": _fmt(p, precision)}
            continue
        if isinstance(q, ChoiceQuestion):
            labels = list(q.criteria.keys())
            p = normalize(_ordered_probs(d, labels))
            best = max(range(len(p)), key=lambda i: (p[i], -i))  # first key on ties
            shown = largest_remainder_2dp(p) if precision == "2dp" else [x + 0.0 for x in p]
            out[qid] = {
                "type": "choice",
                "choice": labels[best],
                "confidence": _fmt(choice_confidence(p), precision),
                "probabilities": dict(zip(labels, shown)),
            }
            continue
        if isinstance(q, ScoreQuestion):
            levels = [str(i) for i in range(len(q.criteria))]
            p = normalize(_ordered_probs(d, levels))
            shown = largest_remainder_2dp(p) if precision == "2dp" else [x + 0.0 for x in p]
            out[qid] = {
                "type": "score",
                "score": _fmt(score_value(p), precision),
                "confidence": _fmt(score_confidence(p, "mode"), precision),
                "legend": {lvl: crit for lvl, crit in zip(levels, q.criteria)},
                "probabilities": dict(zip(levels, shown)),
            }
            continue
        raise TypeError(f"unsupported question {type(q).__name__}")
    return out


# ---------------------------------------------------------------------------------------- W4 refusals


def refusal_body(kind: str, *, model_id: str = "", need: int | None = None, limit: int | None = None,
                 qid: str = "", n: int | None = None) -> dict[str, Any]:
    """The refusal body (SPEC §5.2 shape) with the Decision Index capacity markers inside `msg`."""
    if kind == "context":
        need_s = f"need {need:,} tokens" if isinstance(need, int) else "are too long"
        lim_s = f"{limit:,}" if isinstance(limit, int) else "the model's"
        msg = f"state+questions {need_s}; the {model_id or 'model'} {CONTEXT_MARKER} is {lim_s} tokens"
        return {"detail": [{"type": MAX_TOKENS_EXCEEDED, "msg": msg}]}
    if kind == "options":
        msg = f"Choice question '{qid}' has {n} options; the maximum is {MAX_CHOICE_OPTIONS} {OPTION_CAP_MARKER}"
        return {"detail": [{"type": "too_many_options", "msg": msg}]}
    if kind == "too_few_options":
        return {"detail": [{"type": "too_few_options", "msg": f"Choice question '{qid}' has {n} options; {TOO_FEW_OPTIONS_MARKER}"}]}
    if kind == "levels":
        return {"detail": [{"type": "bad_score_levels", "msg": f"Score question '{qid}' has {n} levels; {SCORE_LEVELS_MARKER}"}]}
    raise ValueError(kind)


def is_refusal_text(text: str) -> bool:
    return MAX_TOKENS_EXCEEDED in text or any(m in text for m in CAPACITY_MARKERS)


class Refused(Exception):
    """A W4 refusal; `status=None` means "the client's configured refusal status"."""

    def __init__(self, body: dict[str, Any], status: int | None = None):
        super().__init__(json.dumps(body))
        self.body = body
        self.status = status


# ---------------------------------------------------------------------------------------- outcomes / clients


@dataclass
class Outcome:
    status: str  # "ok" | "refused" | "error"
    response: dict[str, Any] | None = None  # {"model", "answers", "usage"}
    detail: Any = None  # refusal body or error text
    http_status: int | None = None
    latency_s: float = 0.0
    truncated: bool = False
    tokens_cut: int = 0
    engine_name: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class Client(Protocol):
    kind: str
    model_id: str

    def system_one(self, state: Any, questions: Mapping[str, Any]) -> Outcome: ...

    def describe(self) -> dict[str, Any]: ...


def _check_counts(questions: Mapping[str, Any], model_id: str) -> None:
    """Option / level counts the wire schema cannot express; refusals carry the DI markers."""
    for qid, q in questions.items():
        t = q.get("type") if isinstance(q, dict) else getattr(q, "type", None)
        if t == "choice":
            crit = q["criteria"] if isinstance(q, dict) else q.criteria
            n = len(crit)
            if n > MAX_CHOICE_OPTIONS:
                raise Refused(refusal_body("options", model_id=model_id, qid=qid, n=n))
            if n < 2:
                raise Refused(refusal_body("too_few_options", model_id=model_id, qid=qid, n=n))
        elif t == "score":
            crit = q["criteria"] if isinstance(q, dict) else q.criteria
            n = len(crit)
            if not 2 <= n <= 10:
                raise Refused(refusal_body("levels", model_id=model_id, qid=qid, n=n))


class InProcessClient:
    """Base for clients that hold an `Engine` in this process."""

    kind = "inprocess"

    def __init__(self, engine: Any, model_id: str, *, overflow: str = "refuse", precision: str = "unrounded",
                 refusal_status: int = 400):
        if overflow not in OVERFLOWS:
            raise ValueError(f"overflow must be one of {OVERFLOWS}")
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        self.engine = engine
        self.model_id = check_model_id(model_id)
        self.overflow = overflow
        self.precision = precision
        self.refusal_status = refusal_status

    def _fit(self, state: Any, qs: Mapping[str, Any]) -> tuple[Any, bool, int]:
        return state, False, 0

    def system_one(self, state: Any, questions: Mapping[str, Any]) -> Outcome:
        t0 = time.perf_counter()
        try:
            _check_counts(questions, self.model_id)
            qs = {qid: (q if not isinstance(q, dict) else question_from_json(q)) for qid, q in questions.items()}
            state2, truncated, cut = self._fit(state, qs)
            result = self.engine.evaluate(state2, qs)
            answers = wire_answers(qs, result.dists, self.precision)
            resp = {
                "model": self.model_id,
                "answers": answers,
                "usage": {"input_tokens": int(result.input_tokens), "output_tokens": int(result.output_tokens)},
            }
            return Outcome("ok", resp, None, 200, time.perf_counter() - t0, truncated, cut, result.engine)
        except Refused as r:
            status = r.status if r.status is not None else self.refusal_status
            return Outcome("refused", None, r.body, status, time.perf_counter() - t0)
        except EngineError as e:
            if _is_too_long(e):
                need, limit = _tokens_of(e)
                body = refusal_body("context", model_id=self.model_id, need=need, limit=limit)
                return Outcome("refused", None, body, self.refusal_status, time.perf_counter() - t0)
            return Outcome("error", None, {"detail": e.detail}, e.status, time.perf_counter() - t0)
        except Exception as e:  # engine bug: recorded, never hidden
            return Outcome("error", None, {"detail": f"{type(e).__name__}: {e}"}, None, time.perf_counter() - t0)

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "engine": getattr(self.engine, "name", type(self.engine).__name__),
                "max_tokens": getattr(self.engine, "max_tokens", None), "overflow": self.overflow,
                "precision": self.precision, "refusal_status": self.refusal_status}


def _is_too_long(e: EngineError) -> bool:
    d = e.detail
    if isinstance(d, dict) and "detail" in d:
        d = d["detail"]
    if isinstance(d, list):
        return any(isinstance(x, dict) and x.get("type") == MAX_TOKENS_EXCEEDED for x in d)
    if isinstance(d, dict):
        return d.get("type") == MAX_TOKENS_EXCEEDED or d.get("detail") == MAX_TOKENS_EXCEEDED
    return isinstance(d, str) and MAX_TOKENS_EXCEEDED in d


def _tokens_of(e: EngineError) -> tuple[int | None, int | None]:
    d = e.detail if isinstance(e.detail, dict) else {}
    need = d.get("tokens")
    limit = d.get("max_tokens")
    return (need if isinstance(need, int) else None, limit if isinstance(limit, int) else None)


class HeuristicClient(InProcessClient):
    """The model-free engine (rapidfuzz only): exercises every adapter end to end without weights."""

    kind = "heuristic"

    def __init__(self, model_id: str = "meharsjev-heuristic", **kw: Any):
        from jev_local.engine.heuristic import HeuristicEngine

        super().__init__(HeuristicEngine(), model_id, **kw)


class LocalClient(InProcessClient):
    """A FastEngine checkpoint in this process (torch: VM only).

    W5: `overflow="refuse"` turns an over-long request into a W4 refusal; `"truncate"` cuts the longest
    state field at its end until the request fits (the cut is counted and disclosed). W9: `calib_path`
    replaces the checkpoint's calibration.json for this run.
    """

    kind = "local"

    def __init__(self, ckpt: str | Path, model_id: str, *, threads: int = 8, calib_path: str | Path | None = None,
                 max_tokens: int | None = None, **kw: Any):
        import torch

        from jev_local.engine.encoder.engine import MAX_POSITIONS, FastEngine

        torch.set_num_threads(int(threads))
        self.ckpt = Path(ckpt).expanduser()
        meta = json.loads((self.ckpt / "meta.json").read_text()) if (self.ckpt / "meta.json").exists() else {}
        mt = int(max_tokens or meta.get("max_len") or 2048)
        engine = FastEngine(self.ckpt, device="cpu", dtype=torch.float32, max_tokens=min(mt, MAX_POSITIONS))
        super().__init__(engine, model_id, **kw)
        self.threads = int(threads)
        self.calib_path = Path(calib_path).expanduser() if calib_path else None
        if self.calib_path is not None:
            engine.calib = json.loads(self.calib_path.read_text())
        self.calib_sha256 = sha256_file(self.calib_path) if self.calib_path else (
            sha256_file(self.ckpt / "calibration.json") if (self.ckpt / "calibration.json").exists() else None)
        self._ckpt_hash: dict[str, Any] | None = None

    def ckpt_hash(self) -> dict[str, Any]:
        if self._ckpt_hash is None:
            self._ckpt_hash = sha256_dir(self.ckpt)
        return self._ckpt_hash

    def _fit(self, state: Any, qs: Mapping[str, Any]) -> tuple[Any, bool, int]:
        from jev_local.engine.encoder.engine import plan_passes
        from jev_local.serialize import Segment, question_block, state_segments

        eng = self.engine
        packer = eng.packer
        blocks = [question_block(qid, q) for qid, q in qs.items()]
        parts = packer._block_parts(blocks)
        segs = state_segments(state)
        truncated, cut_total = False, 0
        for _ in range(24):
            n_state = len(packer._state_ids(segs))
            try:
                plan_passes(n_state, blocks, parts, eng.max_tokens, eng.max_flat_tokens)
                return (_rebuild_state(state, segs) if truncated else state), truncated, cut_total
            except EngineError as e:
                if not _is_too_long(e):
                    raise
                need, limit = _tokens_of(e)
                if self.overflow == "refuse":
                    raise Refused(refusal_body("context", model_id=self.model_id, need=need, limit=limit or eng.max_tokens),
                                  self.refusal_status) from None
                excess = (need - (limit or eng.max_tokens)) if isinstance(need, int) else max(64, n_state // 4)
                excess = max(excess, 32) + 16
                if not segs:
                    raise Refused(refusal_body("context", model_id=self.model_id, need=need, limit=limit), self.refusal_status) from None
                i = max(range(len(segs)), key=lambda k: len(segs[k].text))
                ids = list(packer.encode([segs[i].text])[0])
                keep = len(ids) - excess
                if keep < 16:
                    raise Refused(refusal_body("context", model_id=self.model_id, need=need, limit=limit), self.refusal_status) from None
                segs = list(segs)
                segs[i] = Segment(segs[i].key, eng.tok.decode(ids[:keep]).strip())
                cut_total += len(ids) - keep
                truncated = True
        raise Refused(refusal_body("context", model_id=self.model_id), self.refusal_status)

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update(ckpt=str(self.ckpt), threads=self.threads, calibration=str(self.calib_path) if self.calib_path else "checkpoint",
                 calibration_sha256=self.calib_sha256)
        return d


def _rebuild_state(state: Any, segs: Sequence[Any]) -> Any:
    """A state whose `state_segments` render to exactly `segs` (after a truncation)."""
    if isinstance(state, dict):
        return {s.key: s.text for s in segs}
    if isinstance(state, list):
        return [s.text for s in segs]
    return segs[0].text if segs else ""


class HttpClient:
    """POST /v1/systemone against any System One server. 400/413/422 bodies with the refusal markers are
    `refused`; retryable statuses are retried with backoff; anything else is an `error`.

    `wire_model` is what goes on the wire when the server does not yet accept `meharsjev-*` (W10 pending on
    the api owner); the recorded response model is always `model_id` and the override is disclosed."""

    kind = "http"

    def __init__(self, base_url: str, model_id: str, *, wire_model: str | None = None, timeout: float = 600.0,
                 retries: int = 6, api_key: str | None = None):
        import httpx

        self.base_url = base_url.rstrip("/")
        self.model_id = check_model_id(model_id)
        self.wire_model = wire_model or self.model_id
        self.retries = retries
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, headers=headers)
        self._httpx = httpx

    def system_one(self, state: Any, questions: Mapping[str, Any]) -> Outcome:
        body = {"model": self.wire_model, "state": state, "questions": dict(questions)}
        delay = 1.0
        t0 = time.perf_counter()
        last: str = ""
        for attempt in range(self.retries + 1):
            try:
                r = self._client.post("/v1/systemone", json=body)
            except self._httpx.HTTPError as e:
                last = f"transport: {e}"
                if attempt == self.retries:
                    break
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            lat = time.perf_counter() - t0
            if r.status_code == 200:
                data = r.json()
                resp = {"model": self.model_id, "answers": data.get("answers", {}), "usage": data.get("usage", {})}
                return Outcome("ok", resp, None, 200, lat, engine_name=str(data.get("model", "")))
            if r.status_code in (400, 413, 422) and is_refusal_text(r.text):
                try:
                    detail = r.json()
                except ValueError:
                    detail = {"detail": r.text}
                return Outcome("refused", None, detail, r.status_code, lat)
            if r.status_code in RETRY_HTTP and attempt < self.retries:
                ra = r.headers.get("retry-after")
                try:
                    wait = float(ra) if ra else delay
                except ValueError:
                    wait = delay
                time.sleep(min(max(wait, 0.5), 60.0))
                delay = min(delay * 2, 30.0)
                continue
            last = f"http {r.status_code}: {r.text[:500]}"
            return Outcome("error", None, {"detail": last}, r.status_code, lat)
        return Outcome("error", None, {"detail": last or "exhausted retries"}, None, time.perf_counter() - t0)

    def describe(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.kind, "base_url": self.base_url, "wire_model": self.wire_model}
        for path in ("/healthz", "/v1/models"):
            try:
                r = self._client.get(path, timeout=10.0)
                d[path] = r.json() if r.status_code == 200 else {"status": r.status_code}
            except Exception as e:
                d[path] = {"error": f"{type(e).__name__}: {e}"}
        return d


def build_client(args: Any) -> Client:
    """From CLI args: --ckpt (local), --base-url (http) or --engine heuristic."""
    engine = getattr(args, "engine", None) or ("local" if getattr(args, "ckpt", None) else "http" if getattr(args, "base_url", None) else None)
    if engine == "local":
        if not args.ckpt:
            raise SystemExit("--ckpt is required for the local engine")
        return LocalClient(args.ckpt, args.model_id, threads=args.threads, calib_path=args.calib, max_tokens=getattr(args, "max_tokens", None),
                           overflow=args.overflow, precision=args.precision, refusal_status=args.refusal_status)
    if engine == "http":
        if not args.base_url:
            raise SystemExit("--base-url is required for the http engine")
        return HttpClient(args.base_url, args.model_id, wire_model=getattr(args, "wire_model", None), api_key=os.environ.get("BENCHMAX_API_KEY"))
    if engine == "heuristic":
        return HeuristicClient(args.model_id, overflow=args.overflow, precision=args.precision, refusal_status=args.refusal_status)
    raise SystemExit("pass --ckpt DIR, --base-url URL or --engine heuristic")


# ---------------------------------------------------------------------------------------- run context


@dataclass
class RunContext:
    spec: Any
    out: Path
    model_id: str
    args: dict[str, Any]
    engine: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    overflow: dict[str, Any] = field(default_factory=lambda: {"mode": "refuse", "truncated": 0, "refused": 0})
    precision: str = "unrounded"
    dataset_revisions: dict[str, Any] = field(default_factory=dict)
    harness: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    requests: dict[str, Any] = field(default_factory=lambda: {"sent": 0, "ok": 0, "refused": 0, "errors": 0, "truncated": 0})
    determinism: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    started: str = field(default_factory=utc_now)
    finished: str | None = None
    status: str = "running"

    def __post_init__(self) -> None:
        self.out = Path(self.out).expanduser().resolve()  # adapters hand it to subprocesses with other cwds

    @property
    def path(self) -> Path:
        return self.out / "run.json"

    def note(self, msg: str) -> None:
        self.notes.append(msg)
        print(f"[benchmax] {msg}", file=sys.stderr, flush=True)

    def to_dict(self) -> dict[str, Any]:
        t0 = _dt.datetime.fromisoformat(self.started)
        t1 = _dt.datetime.fromisoformat(self.finished) if self.finished else _dt.datetime.now(_dt.timezone.utc)
        return {
            "schema": RUN_SCHEMA,
            "spec": {"id": self.spec.id, "suite": self.spec.suite, "harness": self.spec.harness, "verification": self.spec.verification},
            "status": self.status,
            "model_id": self.model_id,
            "engine": self.engine,
            "calibration": self.calibration,
            "overflow": self.overflow,
            "precision": self.precision,
            "args": self.args,
            "harness": self.harness,
            "dataset_revisions": self.dataset_revisions,
            "hardware": self.hardware,
            "started": self.started,
            "finished": self.finished,
            "wall_clock_s": round((t1 - t0).total_seconds(), 1),
            "requests": self.requests,
            "determinism": self.determinism,
            "results": self.results,
            "notes": self.notes,
        }

    def write(self) -> Path:
        write_json(self.path, self.to_dict())
        return self.path

    def finish(self, status: str = "complete") -> Path:
        self.finished = utc_now()
        self.status = status
        return self.write()


class Session:
    """A client plus bookkeeping: counts, latency, the W11 replay set, an optional request/response trace."""

    def __init__(self, client: Client, ctx: RunContext, *, keep_first: int = 20, trace: Path | None = None):
        self.client = client
        self.ctx = ctx
        self.keep_first = keep_first
        self.kept: list[tuple[Any, dict, str]] = []
        self.latencies: list[float] = []
        self.trace = trace
        if trace is not None:
            trace.parent.mkdir(parents=True, exist_ok=True)

    @property
    def model_id(self) -> str:
        return self.client.model_id

    def ask(self, state: Any, questions: Mapping[str, Any], meta: Mapping[str, Any] | None = None) -> Outcome:
        qs = dict(questions)
        o = self.client.system_one(state, qs)
        r = self.ctx.requests
        r["sent"] += 1
        r["ok" if o.ok else "refused" if o.status == "refused" else "errors"] += 1
        if o.truncated:
            r["truncated"] += 1
            self.ctx.overflow["truncated"] = r["truncated"]
        if o.status == "refused":
            self.ctx.overflow["refused"] = r["refused"]
        if o.ok:
            self.latencies.append(o.latency_s)
            if len(self.kept) < self.keep_first:
                self.kept.append((state, qs, json.dumps(o.response["answers"], sort_keys=True)))
        if self.trace is not None:
            append_jsonl(self.trace, {"meta": dict(meta or {}), "state": state, "questions": qs, "status": o.status,
                                      "response": o.response, "detail": o.detail, "latency_s": round(o.latency_s, 4),
                                      "truncated": o.truncated, "tokens_cut": o.tokens_cut})
        if r["sent"] % 500 == 0:
            self.ctx.write()
        return o

    def latency_summary(self) -> dict[str, float]:
        if not self.latencies:
            return {}
        xs = sorted(self.latencies)
        return {"n": len(xs), "mean_ms": round(1000 * sum(xs) / len(xs), 2), "p50_ms": round(1000 * xs[len(xs) // 2], 2),
                "p95_ms": round(1000 * xs[min(len(xs) - 1, int(0.95 * len(xs)))], 2)}

    def determinism_check(self) -> dict[str, Any]:
        """W11: re-send the kept requests and compare the serialized answers byte for byte."""
        mism = 0
        for state, qs, before in self.kept:
            o = self.client.system_one(state, qs)
            after = json.dumps(o.response["answers"], sort_keys=True) if o.ok else f"<{o.status}>"
            if after != before:
                mism += 1
        rep = {"n_resent": len(self.kept), "mismatches": mism, "identical": mism == 0, "engine": getattr(self.client, "kind", "?")}
        self.ctx.determinism = rep
        return rep

    def close(self) -> None:
        self.ctx.results.setdefault("latency", self.latency_summary())


# ---------------------------------------------------------------------------------------- adapter base


class Adapter:
    """One spec's reproduction. Subclasses set the class attributes and implement `verify` and `run`."""

    default_overflow = "refuse"
    allow_truncate = True
    default_split = "eval"

    def __init__(self, spec: Any, args: Any):
        self.spec = spec
        self.args = args

    @classmethod
    def add_arguments(cls, p: Any) -> None:  # per-adapter CLI options
        return None

    def verify(self, ctx: RunContext) -> dict[str, Any]:  # hash / commit checks, no model
        raise NotImplementedError

    def run(self, session: Session, ctx: RunContext) -> dict[str, Any]:
        raise NotImplementedError


def git_head(path: str | Path) -> str | None:
    """HEAD of a checkout without shelling out (works on any machine with the clone)."""
    git = Path(path) / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
    except OSError:
        return None
    if head.startswith("ref:"):
        ref = head.split(" ", 1)[1].strip()
        try:
            return (git / ref).read_text().strip()
        except OSError:
            packed = git / "packed-refs"
            if packed.exists():
                for line in packed.read_text().splitlines():
                    if line.endswith(" " + ref):
                        return line.split()[0]
            return None
    return head


def check_commit(path: str | Path, pinned: str) -> dict[str, Any]:
    head = git_head(path)
    return {"path": str(path), "present": head is not None, "head": head, "pinned": pinned,
            "match": bool(head and pinned and head.startswith(pinned))}


def default_out(spec_id: str, args: Any) -> Path:
    ckpt = getattr(args, "ckpt", None)
    name = Path(ckpt).expanduser().name if ckpt else ("http" if getattr(args, "base_url", None) else getattr(args, "engine", None) or "engine")
    return REPO_ROOT / "runs" / "benchmax" / name / re.sub(r"[^A-Za-z0-9._-]+", "_", spec_id)
