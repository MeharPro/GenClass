"""DecisionClient: how the harness asks a System One engine (in process, local HTTP, or hosted Jev).

- `inproc` runs `jev_local.api.system_one` on a worker thread, so the event loop keeps receiving
  transcript events during the 15–80 ms forward pass. Engines are not thread-safe, so calls are
  serialized with a lock.
- `http` posts to a local `jev-local serve` (`/v1/systemone`), which binds to 127.0.0.1.
- `hosted` posts to TypeSafe's API with `TYPESAFE_API_KEY`. It sends the screen state off the
  machine, so it runs only when explicitly selected and never as a fallback.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Any, Literal, Mapping

import httpx

from jev_local.engine.base import Engine, EngineError
from jev_local.schema import Question, SystemOneRequest, SystemOneResponse

Backend = Literal["inproc", "http", "hosted"]
HOSTED_BASE_URL = "https://api.typesafe.ai"
SYSTEM_ONE_PATH = "/v1/systemone"
API_KEY_ENV = "TYPESAFE_API_KEY"


class DecisionError(RuntimeError):
    """A decision request failed (engine error, HTTP error, timeout). The controller logs and skips it."""

    def __init__(self, msg: str, status: int | None = None, detail: Any = None):
        super().__init__(msg)
        self.status = status
        self.detail = detail


class DecisionClient:
    def __init__(
        self,
        backend: Backend = "inproc",
        engines: Mapping[str, Engine] | None = None,
        base_url: str | None = None,
        model: str = "jev-local",
        timeout_s: float = 1.2,
        *,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,  # tests inject httpx.MockTransport
    ):
        if backend not in ("inproc", "http", "hosted"):
            raise ValueError(f"unknown backend {backend!r}")
        if backend == "inproc" and not engines:
            raise ValueError("the inproc backend needs engines (e.g. Registry(...).engines())")
        self.backend = backend
        self.engines = dict(engines or {})
        self.model = model
        self.timeout_s = timeout_s
        self.base_url = base_url or (HOSTED_BASE_URL if backend == "hosted" else "http://127.0.0.1:8765")
        self._api_key = api_key
        self._transport = transport
        self._http: httpx.AsyncClient | None = None
        self._engine_lock = threading.Lock()
        self.last_latency_ms = 0.0
        self.n_calls = 0

    # ------------------------------------------------------------------ public

    async def system_one(self, state: Any, questions: Mapping[str, Question]) -> SystemOneResponse:
        req = SystemOneRequest(state=state, model=self.model, questions=dict(questions))
        t0 = time.perf_counter()
        try:
            if self.backend == "inproc":
                return await asyncio.to_thread(self._inproc, req)
            return await self._post(req)
        finally:
            self.n_calls += 1
            self.last_latency_ms = (time.perf_counter() - t0) * 1000.0

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # ------------------------------------------------------------------ backends

    def _inproc(self, req: SystemOneRequest) -> SystemOneResponse:
        from jev_local.api import system_one  # imported lazily: api pulls in every engine module

        with self._engine_lock:
            try:
                return system_one(req, self.engines)
            except EngineError as e:
                raise DecisionError(f"engine error {e.status}: {e.detail}", e.status, e.detail) from e

    def _headers(self) -> dict[str, str]:
        h = {"content-type": "application/json"}
        if self.backend == "hosted":
            key = self._api_key or os.environ.get(API_KEY_ENV)
            if not key:
                raise DecisionError(f"the hosted backend needs {API_KEY_ENV}")
            h["authorization"] = f"Bearer {key}"
        elif self._api_key or os.environ.get("JEV_LOCAL_API_KEY"):
            h["authorization"] = f"Bearer {self._api_key or os.environ['JEV_LOCAL_API_KEY']}"
        return h

    async def _post(self, req: SystemOneRequest) -> SystemOneResponse:
        headers = self._headers()  # raises before any connection when the hosted key is missing
        if self._http is None:
            self._http = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_s, transport=self._transport)
        body = req.model_dump(mode="json", exclude_none=True)
        try:
            r = await self._http.post(SYSTEM_ONE_PATH, json=body, headers=headers)
        except httpx.TimeoutException as e:
            raise DecisionError(f"timeout after {self.timeout_s}s") from e
        except httpx.HTTPError as e:
            raise DecisionError(f"http error: {e!r}") from e
        if r.status_code != 200:
            try:
                detail = r.json().get("detail")
            except ValueError:
                detail = r.text[:200]
            raise DecisionError(f"HTTP {r.status_code}: {detail}", r.status_code, detail)
        return SystemOneResponse.model_validate(r.json())
