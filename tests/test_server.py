"""HTTP server through fastapi.testclient / httpx ASGI, with the heuristic engine and stubs.

Responses are checked against the SDK's generated wire models, which typesafe-sdk generates
from the hosted OpenAPI document (typesafe_sdk._schemas.models), plus the jsonschema they emit.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

jsonschema = pytest.importorskip("jsonschema", reason="jsonschema (dev extra) not installed")
wire = pytest.importorskip("typesafe_sdk._schemas.models", reason="typesafe-sdk (dev extra) not installed")

from jev_local.engine.base import EngineResult, RawDist
from jev_local.engine.registry import Registry
from jev_local.server.app import create_app

EXAMPLES = json.loads((Path(__file__).parent / "fixtures" / "official_examples.json").read_text())
EXAMPLE_IDS = [k for k in EXAMPLES if not k.startswith("_")]
RESPONSE_SCHEMA = wire.SystemOneResponse.model_json_schema()
REQ_ID_RE = re.compile(r"^req_[0-9a-f]{28}$")
ANSWER_KEYS = {
    "noul": {"type", "noul"},
    "choice": {"type", "choice", "confidence", "probabilities"},
    "score": {"type", "score", "confidence", "legend", "probabilities"},
}


class StubRegistry(Registry):
    """A real Registry whose engines are given instead of discovered."""

    def __init__(self, engines: dict):
        super().__init__(load_fast=False, load_general="off")
        self._given = engines

    def _build(self) -> dict:
        return dict(self._given)


def heuristic_registry() -> Registry:
    return Registry(load_fast=False, load_general="off")


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(heuristic_registry(), api_key="")) as c:  # runs lifespan (warm)
        yield c


# ---------------------------------------------------------------- happy path


@pytest.mark.parametrize("name", EXAMPLE_IDS)
def test_official_examples_have_the_documented_shape(client, name):
    ex = EXAMPLES[name]
    r = client.post("/v1/systemone", json=ex["request"])
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/json"
    assert REQ_ID_RE.match(r.headers["x-typesafe-request-id"])
    assert r.headers["x-jev-local-engine"] == "heuristic"
    assert float(r.headers["x-jev-local-latency-ms"]) >= 0.0

    body = r.json()
    wire.SystemOneResponse.model_validate(body)  # the vendor's own OpenAPI-generated model
    jsonschema.validate(body, RESPONSE_SCHEMA)
    assert set(body) == {"model", "answers", "usage"}
    assert body["model"] == "jev-local-heuristic-0.1.0"
    assert set(body["usage"]) == {"input_tokens", "output_tokens"}
    assert body["usage"]["input_tokens"] > 0 and body["usage"]["output_tokens"] > 0

    doc = ex["response"]["answers"]
    questions = ex["request"]["questions"]
    assert list(body["answers"]) == list(questions)
    for qid, ans in body["answers"].items():
        q = questions[qid]
        assert ans["type"] == q["type"] == doc[qid]["type"]
        assert set(ans) == ANSWER_KEYS[ans["type"]] == set(doc[qid])
        if ans["type"] == "noul":
            assert 0.0 <= ans["noul"] <= 1.0
            continue
        assert 0.0 <= ans["confidence"] <= 1.0
        probs = ans["probabilities"]
        assert abs(sum(probs.values()) - 1.0) <= 0.01 * len(probs)
        assert all(round(p, 2) == p for p in probs.values())
        if ans["type"] == "choice":
            assert list(probs) == list(q["criteria"])  # request order
            assert set(probs) == set(doc[qid]["probabilities"])
            assert ans["choice"] in q["criteria"]
        else:
            assert list(probs) == [str(i) for i in range(len(q["criteria"]))]
            assert ans["legend"] == {str(i): c for i, c in enumerate(q["criteria"])} == doc[qid]["legend"]
            assert 0.0 <= ans["score"] <= len(q["criteria"]) - 1


def test_request_ids_are_unique(client):
    req = EXAMPLES["A_choice"]["request"]
    ids = {client.post("/v1/systemone", json=req).headers["x-typesafe-request-id"] for _ in range(5)}
    assert len(ids) == 5


def test_unknown_top_level_fields_are_ignored_and_logged(client, caplog):
    body = dict(EXAMPLES["A_choice"]["request"], beam_width=4, stats={})
    with caplog.at_level(logging.INFO, logger="jev_local.server"):
        r = client.post("/v1/systemone", json=body)
    assert r.status_code == 200
    assert "beam_width" in caplog.text


def test_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200 and REQ_ID_RE.match(r.headers["x-typesafe-request-id"])
    body = r.json()
    wire.ModelMetadataList.model_validate(body)
    names = [m["name"] for m in body["models"]]
    assert "jev-local-heuristic-0.1.0" in names and "jev-local" in names and "jev-latest" in names
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", m["release_date"]) for m in body["models"])


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["engines"] == {"fast": "off", "general": "off", "heuristic": "loaded"}
    assert body["rss_mb"] > 10


def test_openapi_documents_systemone(client):
    spec = client.get("/openapi.json").json()
    assert "/v1/systemone" in spec["paths"]


# ---------------------------------------------------------------- errors


def post(client, body):
    return client.post("/v1/systemone", json=body)


def base_request(**questions):
    return {"state": "s", "model": "jev-local", "questions": questions or {"q": {"type": "noul"}}}


def test_422_missing_state(client):
    r = post(client, {"model": "jev-local", "questions": {"q": {"type": "noul"}}})
    assert r.status_code == 422 and REQ_ID_RE.match(r.headers["x-typesafe-request-id"])
    [err] = r.json()["detail"]
    assert err["loc"] == ["body", "state"] and err["type"] == "missing" and err["msg"] == "Field required"
    wire.HTTPValidationError.model_validate(r.json())


def test_422_null_state_empty_questions_bad_type_bad_json(client):
    r = post(client, dict(base_request(), state=None))
    assert r.status_code == 422 and r.json()["detail"][0]["loc"][:2] == ["body", "state"]
    r = post(client, {"state": "s", "model": "jev-local", "questions": {}})
    assert r.status_code == 422 and r.json()["detail"][0]["loc"] == ["body", "questions"]
    r = post(client, base_request(q={"type": "span"}))
    assert r.status_code == 422
    r = post(client, base_request(q={"type": "choice"}))  # criteria required
    assert r.status_code == 422 and r.json()["detail"][0]["loc"] == ["body", "questions", "q", "choice", "criteria"]
    r = client.post("/v1/systemone", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_400_limits(client):
    r = post(client, base_request(big={"type": "choice", "criteria": {f"o{i}": None for i in range(256)}}))
    assert r.status_code == 400
    assert r.json() == {"detail": "Choice question 'big' has 256 options; maximum is 255"}
    r = post(client, base_request(s={"type": "score", "criteria": [str(i) for i in range(11)]}))
    assert r.status_code == 400 and "maximum is 10" in r.json()["detail"]
    r = post(client, base_request(s={"type": "score", "criteria": []}))
    assert r.status_code == 422 and r.json()["detail"][0]["type"] == "too_short"


def test_404_unknown_model(client):
    r = post(client, dict(base_request(), model="gpt-4"))
    assert r.status_code == 404 and r.json() == {"detail": "Model not found: gpt-4"}
    r = post(client, dict(base_request(), model="jev-local-fast"))  # known id, engine not installed
    assert r.status_code == 404 and r.json()["detail"].startswith("Model not found: jev-local-fast")


def test_auth():
    app = create_app(heuristic_registry(), api_key="sekret")
    c = TestClient(app)
    body = EXAMPLES["A_choice"]["request"]
    r = c.post("/v1/systemone", json=body)
    assert r.status_code == 401 and r.json() == {"detail": "Missing API key"}
    assert REQ_ID_RE.match(r.headers["x-typesafe-request-id"])
    r = c.post("/v1/systemone", json=body, headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401 and r.json() == {"detail": "Invalid API key"}
    r = c.post("/v1/systemone", json={"garbage": 1}, headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401  # auth is checked before the body
    r = c.post("/v1/systemone", json=body, headers={"Authorization": "Bearer sekret"})
    assert r.status_code == 200
    assert c.get("/v1/models").status_code == 401
    assert c.get("/healthz").status_code == 200


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("JEV_LOCAL_API_KEY", "from-env")
    c = TestClient(create_app(heuristic_registry()))
    body = EXAMPLES["A_choice"]["request"]
    assert c.post("/v1/systemone", json=body, headers={"Authorization": "Bearer local"}).status_code == 401
    assert c.post("/v1/systemone", json=body, headers={"Authorization": "Bearer from-env"}).status_code == 200


def test_without_key_any_bearer_is_accepted(client):
    r = client.post("/v1/systemone", json=EXAMPLES["A_choice"]["request"], headers={"Authorization": "Bearer anything"})
    assert r.status_code == 200


def test_host_header_check_blocks_dns_rebinding(client):
    r = client.get("/healthz", headers={"host": "evil.example.com"})
    assert r.status_code == 400 and r.json() == {"detail": "Invalid host header"}
    for host in ("127.0.0.1:8765", "localhost", "[::1]:8765"):
        assert client.get("/healthz", headers={"host": host}).status_code == 200


# ---------------------------------------------------------------- stub engines


class SlowEngine:
    """Blocks inside evaluate until released; records overlapping calls."""

    name = "jev-local-fast-stub"
    max_tokens = 10_000

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.active = 0
        self.max_active = 0
        self._mu = threading.Lock()

    def supports(self, req):
        return True

    def count_tokens(self, req):
        return 7

    def evaluate(self, state, questions):
        with self._mu:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            self.release.wait(10)
            return EngineResult({qid: RawDist("noul", (0.9,)) for qid in questions}, 7, 0, self.name)
        finally:
            with self._mu:
                self.active -= 1


def test_engine_header_and_model_come_from_the_serving_engine():
    eng = SlowEngine()
    eng.release.set()
    c = TestClient(create_app(StubRegistry({"fast": eng})))
    r = c.post("/v1/systemone", json=base_request())
    assert r.status_code == 200
    assert r.headers["x-jev-local-engine"] == "fast"
    assert r.json() == {"model": "jev-local-fast-stub", "answers": {"q": {"type": "noul", "noul": 0.9}}, "usage": {"input_tokens": 7, "output_tokens": 3}}


def test_overload_returns_529_and_evaluations_never_overlap():
    eng = SlowEngine()
    app = create_app(StubRegistry({"fast": eng}), max_queue=1)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as c:
            first = asyncio.create_task(c.post("/v1/systemone", json=base_request()))
            for _ in range(200):
                if eng.started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert eng.started.is_set()
            second = asyncio.create_task(c.post("/v1/systemone", json=base_request()))
            await asyncio.sleep(0.1)  # second is now waiting for the engine
            third = await c.post("/v1/systemone", json=base_request())
            eng.release.set()
            return await first, await second, third

    r1, r2, r3 = asyncio.run(scenario())
    assert r3.status_code == 529 and r3.json() == {"detail": "Overloaded"}
    assert r3.headers["retry-after"] == "1" and REQ_ID_RE.match(r3.headers["x-typesafe-request-id"])
    assert r1.status_code == 200 and r2.status_code == 200
    assert eng.max_active == 1


def test_concurrent_requests_are_serialised():
    eng = SlowEngine()
    eng.release.set()
    app = create_app(StubRegistry({"fast": eng}))

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as c:
            return await asyncio.gather(*(c.post("/v1/systemone", json=base_request()) for _ in range(20)))

    rs = asyncio.run(scenario())
    assert all(r.status_code == 200 for r in rs)
    assert eng.max_active == 1


def test_engine_crash_is_a_500_with_request_id(caplog):
    class Broken(SlowEngine):
        def evaluate(self, state, questions):
            raise ZeroDivisionError("bug")

    c = TestClient(create_app(StubRegistry({"fast": Broken()})), raise_server_exceptions=False)
    with caplog.at_level(logging.ERROR, logger="jev_local.server"):
        r = c.post("/v1/systemone", json=base_request())
    assert r.status_code == 500 and r.json() == {"detail": "Internal server error"}
    assert REQ_ID_RE.match(r.headers["x-typesafe-request-id"])
    assert "ZeroDivisionError" in caplog.text


def test_latency_header_reflects_server_time():
    class Sleepy(SlowEngine):
        def evaluate(self, state, questions):
            time.sleep(0.05)
            return EngineResult({qid: RawDist("noul", (0.5,)) for qid in questions}, 1, 0, self.name)

    c = TestClient(create_app(StubRegistry({"fast": Sleepy()})))
    r = c.post("/v1/systemone", json=base_request())
    assert float(r.headers["x-jev-local-latency-ms"]) >= 50.0
