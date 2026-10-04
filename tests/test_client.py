"""DecisionClient: inproc through api.system_one, http via a mock transport (no sockets), hosted
refuses to send without an explicit key. Nothing here touches the network."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from jev_local.harness.client import DecisionClient, DecisionError
from jev_local.harness.fakes import RuleModel, ScriptedEngine, finder_snapshot
from jev_local.harness.questions import build_questions
from jev_local.harness.state import build_state
from jev_local.schema import SystemOneResponse


def request():
    snap = finder_snapshot()
    state = build_state("scroll down", snap)
    qs = build_questions(snap, ["Notes", "Safari"], [], [])
    return state, qs


def test_inproc_round_trip_on_worker_thread():
    eng = ScriptedEngine(RuleModel())
    c = DecisionClient("inproc", {"fast": eng})
    state, qs = request()
    resp = asyncio.run(c.system_one(state, qs))
    assert isinstance(resp, SystemOneResponse)
    assert resp.answers["intent"].choice == "scroll_down"
    assert resp.model == eng.name and set(resp.answers) == set(qs)
    assert c.n_calls == 1 and c.last_latency_ms > 0
    assert eng.calls == ["scroll down"]


def test_inproc_engine_error_becomes_decision_error():
    c = DecisionClient("inproc", {"fast": ScriptedEngine(RuleModel())}, model="no-such-model")
    state, qs = request()
    with pytest.raises(DecisionError) as ei:
        asyncio.run(c.system_one(state, qs))
    assert ei.value.status == 404


def test_constructor_validation():
    with pytest.raises(ValueError):
        DecisionClient("inproc", None)
    with pytest.raises(ValueError):
        DecisionClient("carrier-pigeon", {})  # type: ignore[arg-type]


def test_http_backend_posts_the_wire_schema():
    seen = {}
    eng = ScriptedEngine(RuleModel())
    ref = DecisionClient("inproc", {"fast": eng})
    state, qs = request()
    canned = asyncio.run(ref.system_one(state, qs)).model_dump(mode="json")

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["auth"] = req.headers.get("authorization")
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=canned)

    c = DecisionClient("http", base_url="http://127.0.0.1:9", transport=httpx.MockTransport(handler), api_key="k1")
    resp = asyncio.run(c.system_one(state, qs))
    asyncio.run(c.aclose())
    assert seen["url"] == "http://127.0.0.1:9/v1/systemone"
    assert seen["auth"] == "Bearer k1"
    assert seen["body"]["model"] == "jev-local" and seen["body"]["state"]["transcript"] == "scroll down"
    assert seen["body"]["questions"]["intent"]["type"] == "choice"
    assert resp.answers["intent"].choice == "scroll_down"


def test_http_errors_map_to_decision_error():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(529, json={"detail": "Overloaded"})

    c = DecisionClient("http", transport=httpx.MockTransport(handler))
    state, qs = request()
    with pytest.raises(DecisionError) as ei:
        asyncio.run(c.system_one(state, qs))
    assert ei.value.status == 529 and ei.value.detail == "Overloaded"

    def slow(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=req)

    c = DecisionClient("http", transport=httpx.MockTransport(slow), timeout_s=0.1)
    with pytest.raises(DecisionError, match="timeout"):
        asyncio.run(c.system_one(state, qs))


def test_hosted_needs_an_explicit_key_and_sends_nothing_without_it(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(500)

    c = DecisionClient("hosted", transport=httpx.MockTransport(handler))
    assert c.base_url == "https://api.typesafe.ai"
    state, qs = request()
    with pytest.raises(DecisionError, match="TYPESAFE_API_KEY"):
        asyncio.run(c.system_one(state, qs))
    assert calls == []  # refused before any request was built

    monkeypatch.setenv("TYPESAFE_API_KEY", "ts_test")
    seen = {}

    def ok(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers.get("authorization")
        seen["url"] = str(req.url)
        return httpx.Response(400, json={"detail": "mock"})

    c = DecisionClient("hosted", transport=httpx.MockTransport(ok))
    with pytest.raises(DecisionError):
        asyncio.run(c.system_one(state, qs))
    assert seen == {"auth": "Bearer ts_test", "url": "https://api.typesafe.ai/v1/systemone"}
