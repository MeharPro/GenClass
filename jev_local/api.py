"""The System One call itself: validate -> route -> evaluate -> build answers (SPEC §5).

Shared by the HTTP server and the harness's in-process client, so both paths return identical
responses for the same request.

Benchmax (docs/research/benchmax/suite-reproduction-specs.md §1.2, PLAN §4.2 item 8) adds a per-run
`ServeConfig`, chosen once when the server (or in-process client) starts and never by inspecting requests:

    precision        "round2" (Jev-style 2 dp, default) | "lr2" (largest remainder, Σp == 1.00) | "exact"   (W3)
    overflow         "refuse" (default; Decision Index, JevBench) | "truncate" (Deußer, Jevals, DMB, ...)  (W5)
    refusal_status   400 (default, Jev parity) | 422 (JevBench)                                              (W4)
    refusal_style    "local" (v1 wording) | "jev" (Decision Index capacity markers)                        (W4)
    model_id         public id reported for the fast engine, e.g. "meharsjev-68m"                            (W10)
    label_alias      treat a choice description equal to its key like null / "" (read the label by name)   (W8)

`meharsjev-<size>` ids route to the fast engine; the local-only aliases (`jev-local*`, `jev-latest`, ...)
are kept for the voice harness and the SDK's default model.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Iterator, Literal, Mapping

from jev_local.confidence import Precision, build_answer
from jev_local.engine.base import Engine, EngineError, EngineResult
from jev_local.schema import ChoiceQuestion, Entry, SystemOneRequest, SystemOneResponse, Usage
from jev_local.serialize import question_block
from jev_local.validate import OptionCapExceeded, RefusalStyle, validate_limits

# Model ids that mean "pick the best local engine". The Jev names are accepted so existing
# clients (whose SDK default is jev-latest) work unchanged against the local server.
AUTO_IDS = frozenset({"jev-local", "jev", "jev-latest", "jev-preview", "jev-1.13", "jev-1.13.0"})
AUTO_ORDER = ("fast", "general", "heuristic")
EXPLICIT_PREFIXES = (
    ("jev-local-fast", "fast"),
    ("jev-local-general", "general"),
    ("jev-local-heuristic", "heuristic"),
)
_VENDOR_PREFIXES = ("typesafe/",)  # OpenRouter/LiteLLM-style "typesafe/jev-1.13"

PUBLIC_PREFIX = "genclass-"  # W10: the public model ids are genclass-<size> (renamed from meharsjev-), never jev-*
PUBLIC_PREFIXES = ("genclass-", "meharsjev-")  # legacy meharsjev-* ids still accepted
_PUBLIC_ID_RE = re.compile(r"^(?:genclass|meharsjev)-[a-z0-9][a-z0-9.\-]*$")

MAX_TOKENS_EXCEEDED = "max_tokens_exceeded"
CONTEXT_MARKER = "maximum context length"  # Decision Index `http.CAPACITY_MARKERS` (context)

Overflow = Literal["refuse", "truncate"]


# ------------------------------------------------------------------------------ run configuration


def parse_precision(spec: str) -> tuple[Precision, int]:
    """'round2' -> ('round', 2); 'lr2' -> ('lr', 2); 'exact' -> ('exact', 0)."""
    s = spec.strip().lower()
    if s == "exact":
        return "exact", 0
    m = re.fullmatch(r"(round|lr)(\d{1,2})?", s)
    if not m:
        raise ValueError(f"precision must be 'exact', 'round<N>' or 'lr<N>', not {spec!r}")
    return m.group(1), int(m.group(2) or 2)  # type: ignore[return-value]


@dataclass(frozen=True)
class ServeConfig:
    """Per-run serving options (module docstring). `ServeConfig()` is the v1 behaviour."""

    precision: str = "round2"  # "round<N>" | "lr<N>" | "exact"
    overflow: Overflow = "refuse"
    refusal_status: int = 400
    refusal_style: RefusalStyle = "local"
    model_id: str | None = None
    label_alias: bool = True
    center: Literal["mode", "median"] = "mode"
    name: str = "compat"

    def __post_init__(self) -> None:
        parse_precision(self.precision)
        if self.overflow not in ("refuse", "truncate"):
            raise ValueError(f"overflow must be 'refuse' or 'truncate', not {self.overflow!r}")
        if self.refusal_status not in (400, 413, 422):
            raise ValueError(f"refusal_status must be 400, 413 or 422, not {self.refusal_status!r}")
        if self.refusal_style not in ("local", "jev"):
            raise ValueError(f"refusal_style must be 'local' or 'jev', not {self.refusal_style!r}")
        if self.model_id is not None and not _PUBLIC_ID_RE.match(self.model_id):
            raise ValueError(f"model_id must look like {PUBLIC_PREFIX}<size> (lowercase), not {self.model_id!r}")

    @property
    def rounding(self) -> tuple[Precision, int]:
        return parse_precision(self.precision)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def preset(cls, name: str, **overrides: Any) -> "ServeConfig":
        """Named run configs for the published suites (see PRESETS); overrides are applied on top."""
        try:
            base = PRESETS[name]
        except KeyError:
            raise ValueError(f"unknown preset {name!r} (choose {sorted(PRESETS)})") from None
        return replace(base, **overrides)


PRESETS: dict[str, ServeConfig] = {
    # v1 server behaviour: 2-dp rounding, refuse, 400 with the local wording.
    "compat": ServeConfig(),
    # Same, with the Decision Index capacity markers in refusals (Jev wording). Recommended default.
    "jev": ServeConfig(refusal_style="jev", name="jev"),
    # Decision Index 0.2.1: unrounded probabilities (|Σp − 1| ≤ 0.01), refuse past the window, 400 + markers.
    "di": ServeConfig(precision="exact", overflow="refuse", refusal_status=400, refusal_style="jev", name="di"),
    # JevBench v1.5: strict SUM_TOL 1e-3, refuse, 422 (issue #113).
    "jevbench": ServeConfig(precision="exact", overflow="refuse", refusal_status=422, refusal_style="jev", name="jevbench"),
    # Deußer, Jevals, DMB, typed-decisions, AbdelStark, LexGLUE: answer every row (truncate, disclosed), unrounded.
    "deusser": ServeConfig(precision="exact", overflow="truncate", refusal_status=400, refusal_style="jev", name="deusser"),
    # Paired 2-dp view against Jev's 2-dp outputs (NLL/KL comparisons): largest-remainder rounding, truncate.
    "paired2dp": ServeConfig(precision="lr2", overflow="truncate", refusal_status=400, refusal_style="jev", name="paired2dp"),
}


# ------------------------------------------------------------------------------ model ids


def resolve_model_id(model: str, config: ServeConfig | None = None) -> str:
    """'auto' or an engine key ('fast' | 'general' | 'heuristic'); 404 for anything else.

    `meharsjev-<size>` is the fast engine. With `config.model_id` set, only that public id is accepted
    (a request for meharsjev-400m against a 68m server is a 404, not a silently mislabelled answer)."""
    mid = model
    for pre in _VENDOR_PREFIXES:
        if mid.startswith(pre):
            mid = mid[len(pre) :]
    if mid in AUTO_IDS:
        return "auto"
    for prefix, key in EXPLICIT_PREFIXES:
        if mid.startswith(prefix):
            return key
    if mid.startswith(PUBLIC_PREFIXES):
        public = config.model_id if config is not None else None
        if public is None or mid == public:
            return "fast"
    raise EngineError(f"Model not found: {model}", status=404)


def public_model_id(req_model: str, engine_key: str, result_engine: str, config: ServeConfig) -> str:
    """The `model` field of the response: the configured public id for the fast engine, else the public id
    the client asked for (meharsjev-*), else the engine's own versioned name."""
    if engine_key == "fast":
        if config.model_id:
            return config.model_id
        mid = req_model
        for pre in _VENDOR_PREFIXES:
            if mid.startswith(pre):
                mid = mid[len(pre) :]
        if mid.startswith(PUBLIC_PREFIXES):
            return mid
    return result_engine


# ------------------------------------------------------------------------------ refusals (W4)


def too_long_message(engine_name: str, limit: int, n: int | None, style: RefusalStyle) -> str:
    need = f"need {n:,} tokens" if isinstance(n, int) else "are too long"
    if style == "jev":
        # Decision Index capacity marker: the body must contain "maximum context length".
        return f"This model's {CONTEXT_MARKER} is {limit:,} tokens; the state plus its longest question {need}"
    return f"state+questions {need}; {engine_name} accepts at most {limit:,} tokens (local limit)"


def _too_long(engine: Engine, n: int | None, config: ServeConfig, limit: int | None = None) -> EngineError:
    limit = engine.max_tokens if limit is None else limit
    return EngineError(
        [{"type": MAX_TOKENS_EXCEEDED, "msg": too_long_message(engine.name, limit, n, config.refusal_style)}],
        status=config.refusal_status,
    )


def _canonical_too_long(e: EngineError, engine: Engine, config: ServeConfig) -> EngineError:
    """Engines report 'too long' in different shapes; the wire always gets SPEC §5.2's
    {"detail": [{"type": "max_tokens_exceeded", "msg": ...}]} with this run's status and wording."""
    d = e.detail
    if isinstance(d, list):
        return EngineError(d, status=config.refusal_status)
    n = d.get("tokens") if isinstance(d, dict) else None
    limit = d.get("max_tokens", engine.max_tokens) if isinstance(d, dict) else engine.max_tokens
    return _too_long(engine, n if isinstance(n, int) else None, config, limit=limit)


def is_too_long(e: EngineError) -> bool:
    d = e.detail
    if isinstance(d, dict) and "detail" in d:
        d = d["detail"]
    if isinstance(d, list):
        return any(isinstance(x, dict) and x.get("type") == MAX_TOKENS_EXCEEDED for x in d)
    return isinstance(d, str) and MAX_TOKENS_EXCEEDED in d


_is_too_long = is_too_long  # v1 name


def refusal_kind(e: EngineError) -> str | None:
    """'context' | 'options' for the two capacity refusals (W4), else None."""
    if isinstance(e, OptionCapExceeded):
        return "options"
    if is_too_long(e):
        return "context"
    return None


# ------------------------------------------------------------------------------ W8 normalisation


def normalize_request(req: SystemOneRequest) -> SystemOneRequest:
    """`null`, `""` and value == key all mean "read the label by its name" (W8).

    `serialize.question_block` already renders null and "" as the bare label; a description equal to its
    key (DMB, DI FinEntity: `{"Positive": "Positive"}`) would otherwise read "Positive: Positive". The
    rewrite is semantics-preserving on the wire: choice descriptions are never echoed in answers."""
    changed: dict[str, ChoiceQuestion] = {}
    for qid, q in req.questions.items():
        if isinstance(q, ChoiceQuestion) and any(_is_alias(k, v) for k, v in q.criteria.items()):
            changed[qid] = q.model_copy(update={"criteria": {k: (None if _is_alias(k, v) else v) for k, v in q.criteria.items()}})
    if not changed:
        return req
    return req.model_copy(update={"questions": {qid: changed.get(qid, q) for qid, q in req.questions.items()}})


def _is_alias(key: str, value: Entry | None) -> bool:
    return isinstance(value, str) and value.strip() == key.strip()


# ------------------------------------------------------------------------------ routing


@dataclass
class Candidate:
    key: str
    engine: Engine
    req: SystemOneRequest  # the request to evaluate (state truncated in overflow="truncate" mode)
    truncated_tokens: int = 0


def _prepare(req: SystemOneRequest, engine: Engine, config: ServeConfig) -> tuple[SystemOneRequest, int]:
    """overflow="truncate": ask the engine to cut the state so the request fits (engine.fit_state, W5)."""
    if config.overflow != "truncate":
        return req, 0
    fit = getattr(engine, "fit_state", None)
    if fit is None:
        return req, 0
    state, cut = fit(req.state, req.questions)
    if not cut:
        return req, 0
    return req.model_copy(update={"state": state}), int(cut)


def _needed(req: SystemOneRequest, engine: Engine) -> int:
    """Tokens compared against `engine.max_tokens`: the engine's own notion (`positions_needed`, e.g. state +
    longest question branch for the per-option-isolated encoder) else the full count."""
    fn = getattr(engine, "positions_needed", None)
    return int(fn(req)) if fn is not None else int(engine.count_tokens(req))


def _candidates(req: SystemOneRequest, engines: Mapping[str, Engine], config: ServeConfig) -> Iterator[Candidate]:
    """Engines that can serve `req`, best first. Lazy, so token counting stops at the first fit.

    Raises once nothing (more) fits: 404 unknown model / engine not installed, capacity refusal (W4) or
    400 unsupported, 503 when no engine is installed at all. The length check runs before `supports`, so a
    too-long request is always a capacity refusal and never "does not support this request".
    """
    key = resolve_model_id(req.model, config)
    if key != "auto":
        engine = engines.get(key)
        if engine is None:
            raise EngineError(f"Model not found: {req.model} (the {key} engine is not installed)", status=404)
        try:
            r, cut = _prepare(req, engine, config)
        except EngineError as e:
            raise (_canonical_too_long(e, engine, config) if is_too_long(e) else e) from None
        n = _needed(r, engine)
        if n > engine.max_tokens:
            raise _too_long(engine, n, config)
        if not engine.supports(r):
            raise EngineError(f"Model {engine.name} does not support this request", status=400)
        yield Candidate(key, engine, r, cut)
        return

    too_long: EngineError | None = None
    present = False
    for k in AUTO_ORDER:
        engine = engines.get(k)
        if engine is None:
            continue
        present = True
        try:
            r, cut = _prepare(req, engine, config)
        except EngineError as e:
            if not is_too_long(e):
                raise
            too_long = too_long or _canonical_too_long(e, engine, config)
            continue
        n = _needed(r, engine)
        if n > engine.max_tokens:
            too_long = too_long or _too_long(engine, n, config)
            continue
        if not engine.supports(r):
            continue
        yield Candidate(k, engine, r, cut)
    if too_long is not None:
        raise too_long
    if not present:
        raise EngineError("No decision engine is available", status=503)
    raise EngineError("No local engine supports this request", status=400)


def route(req: SystemOneRequest, engines: Mapping[str, Engine], config: ServeConfig | None = None) -> Engine:
    return route_key(req, engines, config)[1]


def route_key(req: SystemOneRequest, engines: Mapping[str, Engine], config: ServeConfig | None = None) -> tuple[str, Engine]:
    c = next(_candidates(req, engines, config or ServeConfig()))
    return c.key, c.engine


def output_tokens(req: SystemOneRequest) -> int:
    """Σ (items + 1) per question: options, levels, or the noul's true/false pair, plus one."""
    return sum(len(question_block(qid, q).items) + 1 for qid, q in req.questions.items())


# ------------------------------------------------------------------------------ serving


@dataclass
class Served:
    response: SystemOneResponse
    engine_key: str  # "fast" | "general" | "heuristic"
    result: EngineResult
    latency_ms: float  # validate + route + evaluate + answers
    truncated_tokens: int = 0  # state tokens cut in overflow="truncate" mode (0 = untouched)


def serve_request(
    req: SystemOneRequest,
    engines: Mapping[str, Engine],
    round_digits: int | None = None,
    config: ServeConfig | None = None,
) -> Served:
    """system_one plus the metadata the HTTP layer reports in headers. `round_digits` is the v1 knob
    (independent rounding); `config` is the benchmax run configuration and wins when both are given."""
    if config is None:
        config = ServeConfig() if round_digits is None else ServeConfig(precision=f"round{round_digits}")
    precision, digits = config.rounding
    t0 = time.perf_counter()
    validate_limits(req, status=config.refusal_status, style=config.refusal_style)
    if config.label_alias:
        req = normalize_request(req)
    auto = resolve_model_id(req.model, config) == "auto"
    cands = _candidates(req, engines, config)
    result: EngineResult | None = None
    too_long: EngineError | None = None
    while result is None:
        try:
            cand = next(cands)
        except (EngineError, StopIteration) as e:
            # Exhausted after an engine rejected the request as too long: report that engine's
            # own error rather than a generic "nothing fits".
            if too_long is not None:
                raise too_long from None
            if isinstance(e, StopIteration):
                raise RuntimeError("no engine produced a result") from None
            raise
        try:
            result = cand.engine.evaluate(cand.req.state, cand.req.questions)
        except EngineError as e:
            if not is_too_long(e):
                raise
            # An engine's token estimate can undershoot its real packing; in auto mode that is a
            # reason to try the next engine, not to fail the request.
            too_long = _canonical_too_long(e, cand.engine, config)
            if auto:
                continue
            raise too_long from None

    missing = [qid for qid in req.questions if qid not in result.dists]
    if missing:
        raise RuntimeError(f"engine {result.engine} returned no answer for {missing[:5]!r}")
    answers = {
        qid: build_answer(q, result.dists[qid], digits, config.center, precision) for qid, q in req.questions.items()
    }
    resp = SystemOneResponse(
        model=public_model_id(req.model, cand.key, result.engine or cand.engine.name, config),
        answers=answers,
        usage=Usage(input_tokens=int(result.input_tokens), output_tokens=output_tokens(cand.req)),
    )
    return Served(resp, cand.key, result, (time.perf_counter() - t0) * 1000.0, cand.truncated_tokens)


def system_one(req: SystemOneRequest, engines: Mapping[str, Engine], config: ServeConfig | None = None) -> SystemOneResponse:
    return serve_request(req, engines, config=config).response


def system_one_json(
    request: Mapping[str, Any], engines: Mapping[str, Engine], config: ServeConfig | None = None, model: str | None = None
) -> dict[str, Any]:
    """In-process adapter entry point: a wire-shaped request dict in, a wire-shaped response dict out
    (the same bytes the HTTP server would send). `model` fills in a missing "model" field.
    Harness adapters (Deußer `engine.system_one`, DI in-process Engine) should call this rather than
    read engine distributions themselves, so every suite gets W1–W3 answers from one code path."""
    return serve_json(request, engines, config, model)[0]


def serve_json(
    request: Mapping[str, Any], engines: Mapping[str, Engine], config: ServeConfig | None = None, model: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """`system_one_json` plus the run-accounting facts an adapter writes next to each answer: engine key,
    state tokens truncated (W5 disclosure `trunc:k`), input tokens and the server-side latency in ms."""
    body = dict(request)
    if model is not None and "model" not in body:
        body["model"] = model
    req = SystemOneRequest.model_validate(body)
    served = serve_request(req, engines, config=config)
    meta = {
        "engine": served.engine_key,
        "truncated_tokens": served.truncated_tokens,
        "input_tokens": served.response.usage.input_tokens,
        "latency_ms": served.latency_ms,
    }
    return served.response.model_dump(mode="json"), meta


# ------------------------------------------------------------------------------ W11 determinism report


def hardware_info() -> dict[str, Any]:
    """CPU model, cores, thread settings and library versions, for run.json and the determinism report."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "env_threads": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS") if k in os.environ},
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        info["cpu"] = platform.processor() or None
    torch = sys.modules.get("torch")  # never import torch here: this runs on the Mac too
    if torch is not None:
        info["torch"] = getattr(torch, "__version__", None)
        try:
            info["torch_threads"] = int(torch.get_num_threads())
        except Exception:
            pass
    return info


def _canon(resp: Mapping[str, Any]) -> str:
    return json.dumps(resp, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _probs(resp: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for qid, a in resp["answers"].items():
        if a["type"] == "noul":
            out[qid] = {"true": float(a["noul"])}
        else:
            out[qid] = {k: float(v) for k, v in a["probabilities"].items()}
    return out


def determinism_report(
    serve: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    requests: list[Mapping[str, Any]],
    repeats: int = 5,
    seed: int = 0,
    permute: bool = True,
) -> dict[str, Any]:
    """W11: send every request `repeats` times (identical bytes expected) and, with `permute`, once more with
    every choice question's options shuffled (per-label probabilities expected unchanged, argmax never flipping).
    `serve` is any wire-shaped callable: `lambda r: system_one_json(r, engines, cfg)` or an HTTP POST.
    Returns the counts, the worst absolute differences and `hardware_info()`; raises nothing on mismatch."""
    rng = random.Random(seed)
    n_req = len(requests)
    identical = 0
    worst_repeat = 0.0
    repeat_flips = 0
    perm_worst = 0.0
    perm_flips = 0
    perm_checked = 0
    perm_identical = 0
    latencies: list[float] = []
    per_request: list[dict[str, Any]] = []
    for r in requests:
        bodies: list[str] = []
        probs: list[dict[str, dict[str, float]]] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            resp = serve(r)
            latencies.append((time.perf_counter() - t0) * 1e3)
            bodies.append(_canon(resp))
            probs.append(_probs(resp))
        same = len(set(bodies)) == 1
        identical += same
        d_rep = 0.0
        flips = 0
        for p in probs[1:]:
            for qid, dist in probs[0].items():
                d_rep = max(d_rep, max(abs(dist[k] - p[qid][k]) for k in dist))
                if max(dist, key=dist.get) != max(p[qid], key=p[qid].get):
                    flips += 1
        worst_repeat = max(worst_repeat, d_rep)
        repeat_flips += flips
        entry: dict[str, Any] = {"identical": same, "max_abs_diff": d_rep, "argmax_flips": flips}
        if permute:
            shuffled = _shuffle_choices(r, rng)
            if shuffled is not None:
                perm_checked += 1
                p2 = _probs(serve(shuffled))
                d_perm = 0.0
                f_perm = 0
                for qid, dist in probs[0].items():
                    d_perm = max(d_perm, max(abs(dist[k] - p2[qid][k]) for k in dist))
                    if max(dist, key=dist.get) != max(p2[qid], key=p2[qid].get):
                        f_perm += 1
                perm_worst = max(perm_worst, d_perm)
                perm_flips += f_perm
                perm_identical += d_perm == 0.0
                entry["permutation"] = {"max_abs_diff": d_perm, "argmax_flips": f_perm}
        per_request.append(entry)
    lat = sorted(latencies)
    return {
        "requests": n_req,
        "repeats": repeats,
        "identical_bytes_fraction": identical / n_req if n_req else 1.0,
        "repeat_flip_rate": repeat_flips / max(1, n_req * max(0, repeats - 1)),
        "max_abs_diff_repeat": worst_repeat,
        "permutation": {
            "checked": perm_checked,
            "identical_fraction": perm_identical / perm_checked if perm_checked else None,
            "max_abs_diff": perm_worst,
            "argmax_flips": perm_flips,
            "order_flip_rate": perm_flips / perm_checked if perm_checked else None,
        },
        "latency_ms": {"p50": lat[len(lat) // 2] if lat else None, "p95": lat[int(len(lat) * 0.95)] if lat else None},
        "hardware": hardware_info(),
        "per_request": per_request,
    }


def _shuffle_choices(request: Mapping[str, Any], rng: random.Random) -> dict[str, Any] | None:
    qs = request.get("questions") or {}
    out = {}
    any_choice = False
    for qid, q in qs.items():
        if isinstance(q, Mapping) and q.get("type") == "choice" and isinstance(q.get("criteria"), Mapping) and len(q["criteria"]) > 1:
            items = list(q["criteria"].items())
            rng.shuffle(items)
            out[qid] = {**q, "criteria": dict(items)}
            any_choice = True
        else:
            out[qid] = q
    if not any_choice:
        return None
    return {**request, "questions": out}


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
