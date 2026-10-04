"""Local Jev-compatible HTTP API (docs/research/SPEC.md §5).

POST /v1/systemone   exactly the hosted wire schema, so the official SDKs work unchanged
GET  /v1/models      engines this process can serve, plus the auto aliases
GET  /healthz        engine status and memory (not a Jev route)

Every response carries `x-typesafe-request-id`; /v1/systemone also sets `x-jev-local-engine`
(fast | general | heuristic) and `x-jev-local-latency-ms` (server-side time for the request).
Engines are not thread-safe, so evaluations run one at a time in a worker thread; up to
`max_queue` requests may wait, beyond that the server answers 529 Overloaded (the SDK retries).
The server is meant for localhost only: `serve` refuses other interfaces unless told otherwise,
and a Host-header check blocks DNS-rebinding requests from web pages.

Benchmax (suite-reproduction-specs.md §1.2): `create_app(..., config=ServeConfig)` fixes the run's precision,
overflow mode, refusal status/wording and public model id once, at start-up (never per request). The server
counts served / refused / truncated requests (`/healthz` → "benchmax") and sets `x-jev-local-truncated` (state
tokens cut) on truncated answers, so a harness can write the disclosure counts into run.json.

    python -m jev_local.server.app --preset di --model-id meharsjev-68m --ckpt ~/jev/models/jev-local-fast-v2 \
        --calib runs/benchmax/calib/global.json --threads 16 --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import hmac
import json
import logging
import os
import resource
import secrets
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Iterable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from jev_local import __version__
from jev_local.api import PRESETS, ServeConfig, Served, hardware_info, refusal_kind, serve_request
from jev_local.engine.base import EngineError
from jev_local.engine.registry import Registry
from jev_local.schema import SystemOneRequest, SystemOneResponse

log = logging.getLogger("jev_local.server")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_QUEUE = 32
API_KEY_ENV = "JEV_LOCAL_API_KEY"
RELEASE_DATE = "2026-09-24"
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "testserver"})  # testserver: Starlette TestClient
REQUEST_ID_HEADER = "x-typesafe-request-id"
ENGINE_HEADER = "x-jev-local-engine"
LATENCY_HEADER = "x-jev-local-latency-ms"
TRUNCATED_HEADER = "x-jev-local-truncated"  # state tokens cut (overflow="truncate"); absent when untouched
_KNOWN_FIELDS = frozenset(SystemOneRequest.model_fields)
PUBLIC_DESCRIPTION = (
    "meharsjev: independent open reimplementation of a typed-decision (System One) model. Not affiliated with TypeSafe."
)


class Counters:
    """Run accounting for disclosure flags (PLAN §8: refused:k, trunc:k). Thread-safe, reset per process."""

    FIELDS = ("requests", "served", "truncated_requests", "truncated_tokens", "refused_context", "refused_options",
              "errors_4xx", "errors_5xx", "overloaded")

    def __init__(self) -> None:
        self._mu = threading.Lock()
        self._c = {k: 0 for k in self.FIELDS}
        self.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    def add(self, key: str, n: int = 1) -> None:
        with self._mu:
            self._c[key] += n

    def snapshot(self) -> dict[str, int]:
        with self._mu:
            return dict(self._c)

ENGINE_DESCRIPTIONS = {
    "fast": "Local encoder decision model (jev-local). Not affiliated with TypeSafe.",
    "general": "Local Qwen3-1.7B logit scorer on MLX (jev-local). Not affiliated with TypeSafe.",
    "heuristic": "Zero-shot fuzzy-matching fallback, no model (jev-local). Not affiliated with TypeSafe.",
}
ALIASES = {
    "jev-local": "Alias: auto-routes to the fast engine, else general, else heuristic.",
    "jev-latest": "Alias of jev-local, so clients that default to jev-latest work unchanged.",
}


def new_request_id() -> str:
    """req_ + 12 hex digits of milliseconds + 16 random hex digits (time-sortable, like a ULID)."""
    return f"req_{time.time_ns() // 1_000_000:012x}{secrets.token_hex(8)}"


def rss_mb() -> float:
    """Resident memory of this process in MB (macOS: phys_footprint, as Activity Monitor shows)."""
    if sys.platform == "darwin":
        try:
            # struct rusage_info_v0: 16-byte uuid, then uint64 counters; index 7 = ri_phys_footprint.
            buf = (ctypes.c_uint64 * 12)()
            libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
            if libc.proc_pid_rusage(os.getpid(), 0, ctypes.byref(buf)) == 0:
                return round(buf[2 + 7] / 1e6, 1)
        except Exception:
            pass
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)  # peak, bytes
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3, 1)  # peak, KB on Linux


class Overloaded(Exception):
    pass


class _Gate:
    """One evaluation at a time, FIFO, with a bounded number of waiters.

    The asyncio.Lock queues waiting requests as coroutines (not threads), so a burst cannot eat
    the threadpool. It is (re)created per event loop because test clients run their own loops.
    """

    def __init__(self, max_queue: int):
        self.max_queue = max_queue
        self.waiting = 0
        self._lock: asyncio.Lock | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._loop is not loop:
            self._lock, self._loop, self.waiting = asyncio.Lock(), loop, 0
        return self._lock

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        lock = self._get_lock()
        if lock.locked() and self.waiting >= self.max_queue:
            raise Overloaded
        self.waiting += 1
        try:
            await lock.acquire()
        finally:
            self.waiting -= 1
        try:
            yield
        finally:
            lock.release()


class _Guard:
    """Pure ASGI middleware: request id on every response, Host check, optional bearer auth.

    Pure ASGI (not BaseHTTPMiddleware) keeps per-request overhead in the tens of microseconds.
    """

    def __init__(self, app: Any, api_key: str | None, allowed_hosts: frozenset[str] | None):
        self.app = app
        self.api_key = api_key.encode() if api_key else None
        self.allowed_hosts = allowed_hosts

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        rid = new_request_id()
        state = scope.setdefault("state", {})
        state["request_id"] = rid
        state["t0"] = time.perf_counter()
        rid_header = (REQUEST_ID_HEADER.encode(), rid.encode())

        async def send_with_id(message: dict) -> None:
            if message["type"] == "http.response.start":
                message["headers"] = [*message.get("headers", []), rid_header]
            await send(message)

        headers = dict(scope.get("headers") or [])
        if self.allowed_hosts is not None and _host_name(headers.get(b"host", b"")) not in self.allowed_hosts:
            await _send_json(send_with_id, 400, {"detail": "Invalid host header"})
            return
        if self.api_key is not None and scope.get("path", "").startswith("/v1/"):
            auth = headers.get(b"authorization", b"")
            scheme, _, token = auth.partition(b" ")
            if not auth:
                await _send_json(send_with_id, 401, {"detail": "Missing API key"}, [(b"www-authenticate", b"Bearer")])
                return
            if scheme.lower() != b"bearer" or not hmac.compare_digest(token.strip(), self.api_key):
                await _send_json(send_with_id, 401, {"detail": "Invalid API key"}, [(b"www-authenticate", b"Bearer")])
                return
        await self.app(scope, receive, send_with_id)


def _host_name(host: bytes) -> str:
    h = host.decode("latin-1").strip().lower()
    if h.startswith("["):  # [::1]:8765
        return h[1 : h.find("]")] if "]" in h else h
    return h.rsplit(":", 1)[0] if h.count(":") == 1 else h


async def _send_json(send: Any, status: int, body: dict, extra: Iterable[tuple[bytes, bytes]] = ()) -> None:
    data = json.dumps(body).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(data)).encode()), *extra]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": data})


def _error_body(detail: object) -> dict[str, object]:
    # api.py raises EngineError(msg); tolerate engines that pass a ready-made {"detail": ...} body
    # (possibly with extra fields) instead of nesting it as {"detail": {"detail": ...}}.
    if isinstance(detail, dict) and "detail" in detail:
        return detail
    return {"detail": detail}


def create_app(
    registry: Registry,
    *,
    api_key: str | None = None,
    max_queue: int = MAX_QUEUE,
    allowed_hosts: Iterable[str] | None = LOCAL_HOSTS,
    warm_on_startup: bool = True,
    config: ServeConfig | None = None,
) -> FastAPI:
    """`api_key=None` reads $JEV_LOCAL_API_KEY (unset or empty: no auth). `allowed_hosts=None`
    disables the Host check (only for deliberately remote deployments). `config` is the benchmax run
    configuration (default: v1 behaviour, `ServeConfig()`)."""
    if api_key is None:
        api_key = os.environ.get(API_KEY_ENV, "").strip() or None
    config = config or ServeConfig()
    gate = _Gate(max_queue)
    eval_lock = threading.Lock()  # belt and braces: engines may be shared with other threads
    seen_extras: set[frozenset[str]] = set()
    counters = Counters()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if warm_on_startup:
            await run_in_threadpool(registry.warm)
        engines = await run_in_threadpool(registry.engines)
        log.info("jev-local engines: %s", {k: e.name for k, e in engines.items()})
        yield

    app = FastAPI(
        title="jev-local",
        version=__version__,
        description="Local System One decision API compatible with the Jev wire format. Not affiliated with TypeSafe.",
        lifespan=lifespan,
    )
    app.add_middleware(
        _Guard, api_key=api_key, allowed_hosts=frozenset(allowed_hosts) if allowed_hosts is not None else None
    )
    app.state.registry = registry
    app.state.gate = gate
    app.state.config = config
    app.state.counters = counters

    async def engines_async() -> dict:
        # Building engines can take seconds (checkpoint load); never do that on the event loop.
        return registry.engines() if registry.ready else await run_in_threadpool(registry.engines)

    def evaluate(req: SystemOneRequest, engines: dict) -> Served:
        with eval_lock:
            return serve_request(req, engines, config=config)

    def refused(e: EngineError) -> JSONResponse:
        kind = refusal_kind(e)
        if kind == "context":
            counters.add("refused_context")
        elif kind == "options":
            counters.add("refused_options")
        else:
            counters.add("errors_4xx" if e.status < 500 else "errors_5xx")
        return JSONResponse(_error_body(e.detail), status_code=e.status)

    @app.exception_handler(EngineError)
    async def engine_error(request: Request, exc: EngineError) -> JSONResponse:
        return refused(exc)

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    async def systemone(req: SystemOneRequest, request: Request) -> Response:
        counters.add("requests")
        try:
            _log_extra_fields(await request.json(), seen_extras)
            engines = await engines_async()
            async with gate.slot():
                served = await run_in_threadpool(evaluate, req, engines)
        except Overloaded:
            counters.add("overloaded")
            return JSONResponse({"detail": "Overloaded"}, status_code=529, headers={"retry-after": "1"})
        except EngineError as e:
            return refused(e)
        except Exception:
            # Handled here rather than by Starlette's outermost error middleware so the 500 still
            # passes through _Guard and carries a request id.
            counters.add("errors_5xx")
            log.exception("systemone failed (request %s)", request.state.request_id)
            return JSONResponse({"detail": "Internal server error"}, status_code=500)
        counters.add("served")
        headers = {ENGINE_HEADER: served.engine_key, LATENCY_HEADER: f"{(time.perf_counter() - request.state.t0) * 1000.0:.1f}"}
        if served.truncated_tokens:
            counters.add("truncated_requests")
            counters.add("truncated_tokens", served.truncated_tokens)
            headers[TRUNCATED_HEADER] = str(served.truncated_tokens)
        return Response(content=served.response.model_dump_json(), media_type="application/json", headers=headers)

    @app.get("/v1/models")
    async def models() -> dict[str, list[dict[str, str]]]:
        engines = await engines_async()
        out = []
        if config.model_id and "fast" in engines:
            out.append({"name": config.model_id, "description": PUBLIC_DESCRIPTION, "release_date": RELEASE_DATE})
        out += [
            {"name": e.name, "description": ENGINE_DESCRIPTIONS.get(k, "Local engine (jev-local)."), "release_date": RELEASE_DATE}
            for k, e in engines.items()
        ]
        out += [{"name": n, "description": d, "release_date": RELEASE_DATE} for n, d in ALIASES.items()]
        return {"models": out}

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        status = await run_in_threadpool(registry.status)
        return {
            "ok": True,
            "version": __version__,
            "engines": status,
            "queue": {"waiting": gate.waiting, "max": gate.max_queue},
            "rss_mb": rss_mb(),
            "benchmax": {"config": config.to_dict(), "counters": counters.snapshot(), "started_at": counters.started_at},
        }

    @app.get("/x/v1/run")
    async def run_facts() -> dict[str, Any]:
        """Everything a harness writes into run.json about the serving side (PLAN §6): config, counters,
        checkpoint sha256, calibration in force, engine limits, CPU threads and hardware. Not a Jev route."""
        desc = await run_in_threadpool(registry.describe)
        return {
            "config": config.to_dict(),
            "counters": counters.snapshot(),
            "started_at": counters.started_at,
            "registry": desc,
            "hardware": hardware_info(),
            "version": __version__,
        }

    return app


def _log_extra_fields(body: object, seen: set[frozenset[str]]) -> None:
    """Unknown top-level fields are ignored (extra="ignore"); say so once per distinct set."""
    if not isinstance(body, dict):
        return
    extra = frozenset(body) - _KNOWN_FIELDS
    if extra and extra not in seen:
        if len(seen) < 256:  # bounded: a client inventing new keys per request must not grow memory
            seen.add(extra)
        log.info("ignoring unknown request fields: %s", sorted(extra))


def serve(
    registry: Registry | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    allow_remote: bool = False,
    log_level: str = "info",
    access_log: bool = False,
    config: ServeConfig | None = None,
) -> None:
    """Run the API with uvicorn (single worker: one copy of the models)."""
    import uvicorn

    if host not in {"127.0.0.1", "localhost", "::1"} and not allow_remote:
        raise ValueError(f"refusing to bind {host}: jev-local serves localhost only (pass allow_remote=True)")
    app = create_app(registry or Registry(), allowed_hosts=None if allow_remote else LOCAL_HOSTS, config=config)
    uvicorn.run(app, host=host, port=port, workers=1, log_level=log_level, access_log=access_log)


# ---------------------------------------------------------------- benchmax command line


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m jev_local.server.app",
        description="Serve the local System One API with a fixed benchmax run configuration (module docstring).",
    )
    ap.add_argument("--preset", default="jev", choices=sorted(PRESETS), help="run preset (default: jev = v1 + DI markers)")
    ap.add_argument("--precision", default=None, help="round<N> | lr<N> | exact (overrides the preset)")
    ap.add_argument("--overflow", default=None, choices=["refuse", "truncate"])
    ap.add_argument("--refusal-status", default=None, type=int, choices=[400, 413, 422])
    ap.add_argument("--refusal-style", default=None, choices=["local", "jev"])
    ap.add_argument("--model-id", default=None, help="public id for the fast engine, e.g. meharsjev-68m (W10)")
    ap.add_argument("--no-label-alias", action="store_true", help="do not treat description == key as a bare label (W8)")
    ap.add_argument("--ckpt", type=Path, default=None, help="fast checkpoint dir (default: $JEV_LOCAL_FAST_CKPT or models/jev-local-fast)")
    ap.add_argument("--calib", type=Path, default=None, help="calibration.json chosen by this run (replaces the checkpoint's) (W9)")
    ap.add_argument("--global-calibration-only", action="store_true", help="drop per-header temperatures (DI: no benchmark detection)")
    ap.add_argument("--max-tokens", type=int, default=None, help="positions budget (default: checkpoint meta max_len)")
    ap.add_argument("--max-flat", type=int, default=None, help="tokens per forward pass (default: max(max_tokens, 8192))")
    ap.add_argument("--device", default=None, help="cpu | mps (default: mps if available, else cpu)")
    ap.add_argument("--threads", type=int, default=16, help="torch CPU threads (shared VM: leave ~16 cores free)")
    ap.add_argument("--no-general", action="store_true", help="disable the MLX decoder engine (default on Linux)")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--allow-remote", action="store_true", help="bind a non-loopback host (disables the Host check)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--max-queue", type=int, default=MAX_QUEUE)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--log-level", default="info")
    ap.add_argument("--print-config", action="store_true", help="print the resolved config as JSON and exit")
    return ap


def config_from_args(args: argparse.Namespace) -> ServeConfig:
    overrides = {
        k: v
        for k, v in {
            "precision": args.precision,
            "overflow": args.overflow,
            "refusal_status": args.refusal_status,
            "refusal_style": args.refusal_style,
            "model_id": args.model_id,
        }.items()
        if v is not None
    }
    if args.no_label_alias:
        overrides["label_alias"] = False
    return ServeConfig.preset(args.preset, **overrides)


def registry_from_args(args: argparse.Namespace) -> Registry:
    fast_options = {
        "calibration": str(args.calib) if args.calib else None,
        "drop_header_calibration": True if args.global_calibration_only else None,
        "max_tokens": args.max_tokens,
        "max_flat_tokens": args.max_flat,
        "device": args.device,
        "threads": args.threads,
    }
    load_general = "off" if (args.no_general or sys.platform != "darwin") else "lazy"
    return Registry(fast_ckpt=args.ckpt, load_general=load_general, fast_options=fast_options)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    if args.print_config:
        print(json.dumps({"config": config.to_dict(), "fast_options": registry_from_args(args).fast_options}, indent=2))
        return 0
    logging.basicConfig(level=args.log_level.upper())
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not args.allow_remote:
        raise SystemExit(f"refusing to bind {args.host}: jev-local serves localhost only (pass --allow-remote)")
    registry = registry_from_args(args)
    app = create_app(registry, api_key=args.api_key, max_queue=args.max_queue, config=config,
                     allowed_hosts=None if args.allow_remote else LOCAL_HOSTS)
    import uvicorn

    log.info("benchmax serve: %s", json.dumps(config.to_dict()))
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level=args.log_level, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
