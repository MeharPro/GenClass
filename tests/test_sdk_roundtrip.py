"""typesafe-sdk (the official Python client) against a live uvicorn on an ephemeral port.

What the SDK needs, read from its source in .venv (typesafe_sdk 0.7.1):
- client: `typesafe_sdk.TypeSafeClient` (sync) / `AsyncTypeSafeClient`; call `client.system_one(state,
  questions, *, model=None, extra_body=None, ...)` and `client.models.list()`.
- base URL: the `base_url=` kwarg or env `TYPESAFE_BASE_URL` (constants.BASE_URL_ENV, default
  https://api.typesafe.ai); the client appends `/v1/systemone` and `/v1/models`.
- API key: `api_key=` or env `TYPESAFE_API_KEY`; required and non-empty even for a local server.
- default model: `TYPESAFE_DEFAULT_MODEL`, else "jev-latest" (an auto alias here).
- responses parse into frozen, *strict* pydantic models; score `legend`/`probabilities` are keyed by
  int in the SDK and by string on the wire; `.request_id` reads `x-typesafe-request-id`.
- errors: 400/401/404/422 map to TypeSafeBadRequestError / AuthenticationError / NotFoundError /
  UnprocessableEntityError; 5xx and 429 are retried (RetryPolicy), honouring Retry-After.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn

pytest.importorskip("typesafe_sdk", reason="typesafe-sdk (dev extra) not installed")
from typesafe_sdk import (  # noqa: E402
    AsyncTypeSafeClient,
    Choice,
    Noul,
    NoulCriteria,
    RetryPolicy,
    Score,
    TypeSafeAuthenticationError,
    TypeSafeBadRequestError,
    TypeSafeClient,
    TypeSafeNotFoundError,
    TypeSafeUnprocessableEntityError,
)
from typesafe_sdk.constants import API_KEY_ENV, BASE_URL_ENV

from jev_local.engine.registry import Registry
from jev_local.server.app import create_app

EXAMPLES = json.loads((Path(__file__).parent / "fixtures" / "official_examples.json").read_text())
EXAMPLE_IDS = [k for k in EXAMPLES if not k.startswith("_")]
NO_RETRY = RetryPolicy(max_retries=0)


class LiveServer:
    """uvicorn in a daemon thread on a pre-bound ephemeral port (no port race)."""

    def __init__(self, app, start_timeout_s: float = 20.0):
        self.start_timeout_s = start_timeout_s
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on"))
        self.thread = threading.Thread(target=self.server.run, kwargs={"sockets": [self.sock]}, daemon=True)

    def __enter__(self) -> "LiveServer":
        self.thread.start()
        deadline = time.monotonic() + self.start_timeout_s
        while not self.server.started:
            if time.monotonic() > deadline or not self.thread.is_alive():
                raise RuntimeError("uvicorn did not start")
            time.sleep(0.02)
        return self

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True
        self.thread.join(10)
        self.sock.close()


def registry() -> Registry:
    # Heuristic only: deterministic and model-free, whatever is on disk or in the HF cache.
    return Registry(load_fast=False, load_general="off")


@pytest.fixture(scope="module")
def live():
    with LiveServer(create_app(registry(), api_key="")) as srv:
        yield srv


@pytest.fixture(scope="module")
def live_auth():
    with LiveServer(create_app(registry(), api_key="local-secret")) as srv:
        yield srv


@pytest.fixture
def sdk(live, monkeypatch):
    # Configure exactly as a user would: environment variables only.
    monkeypatch.setenv(BASE_URL_ENV, live.url)
    monkeypatch.setenv(API_KEY_ENV, "local")
    with TypeSafeClient(retry=NO_RETRY) as c:
        yield c


@pytest.mark.parametrize("name", EXAMPLE_IDS)
def test_official_examples_round_trip(sdk, name):
    req = EXAMPLES[name]["request"]
    resp = sdk.system_one(req["state"], req["questions"], model=req["model"])
    assert resp.model == "jev-local-heuristic-0.1.0"
    assert resp.request_id.startswith("req_")
    assert resp.raw_http_response.headers["x-jev-local-engine"] == "heuristic"
    assert resp.usage.input_tokens > 0 and resp.usage.output_tokens > 0
    assert list(resp.answers) == list(req["questions"])
    for qid, q in req["questions"].items():
        a = resp.answers[qid]
        assert a.type == q["type"]
        if a.type == "noul":
            assert qid in resp.nouls and 0.0 <= a.noul <= 1.0
        elif a.type == "choice":
            assert qid in resp.choices
            assert a.choice in q["criteria"] and list(a.probabilities) == list(q["criteria"])
        else:
            assert qid in resp.scores
            assert list(a.probabilities) == list(range(len(q["criteria"])))  # int keys in the SDK
            assert a.legend == dict(enumerate(q["criteria"]))  # object levels echoed


def test_typed_questions_and_readme_example(sdk):
    resp = sdk.system_one(
        state="I was charged twice. Please help.",
        questions={
            "billing": Noul(instructions="Is this about billing?"),
            "spam": Noul(instructions="Is it spam?", criteria=NoulCriteria(true="advertising", false="a real request")),
            "tone": Choice(instructions="What is the tone?", criteria={"calm": None, "angry": None}),
            "structured": Choice(
                instructions={"question": "Which team?", "focus": "`state`"},
                criteria={"billing": {"what": "payments", "examples": ["charged twice"]}, "other": None},
            ),
            "urgency": Score(instructions="How urgent?", criteria=["Can wait", "This week", "Today"]),
        },
    )
    assert 0 <= resp.nouls["billing"].noul <= 1
    assert resp.choices["tone"].choice in {"calm", "angry"}
    assert resp.choices["structured"].choice in {"billing", "other"}
    assert set(resp.scores["urgency"].probabilities) == {0, 1, 2}
    assert 0 <= resp.scores["urgency"].score <= 2


def test_default_model_and_extra_body(sdk):
    # SDK default model is jev-latest (an auto alias); extra_body fields are ignored by the server.
    resp = sdk.system_one("hello", {"x": Noul()}, extra_body={"beam_width": 4})
    assert resp.model == "jev-local-heuristic-0.1.0"
    resp = sdk.system_one("hello", {"x": Noul()}, model="jev-local-heuristic")
    assert resp.model == "jev-local-heuristic-0.1.0"


def test_models_list(sdk):
    models = sdk.models.list()
    names = [m.name for m in models.models]
    assert "jev-local-heuristic-0.1.0" in names and "jev-local" in names
    assert models.request_id.startswith("req_")


def test_explicit_base_url_argument(live):
    with TypeSafeClient(api_key="local", base_url=live.url + "/", retry=NO_RETRY) as c:
        assert c.system_one("x", {"q": Noul()}).nouls["q"].noul == 0.5


def test_errors_map_to_sdk_exceptions(sdk):
    with pytest.raises(TypeSafeNotFoundError) as e:
        sdk.system_one("x", {"q": Noul()}, model="gpt-4")
    assert e.value.status == 404 and "Model not found: gpt-4" in str(e.value)
    assert e.value.request_id.startswith("req_")

    with pytest.raises(TypeSafeBadRequestError) as e:
        sdk.system_one("x", {"q": Choice(criteria={f"o{i}": None for i in range(256)})})
    assert "maximum is 255" in str(e.value)

    with pytest.raises(TypeSafeBadRequestError):
        sdk.system_one("x", {"q": Score(criteria=[str(i) for i in range(11)])})

    with pytest.raises(TypeSafeUnprocessableEntityError) as e:
        sdk.system_one("x", {"q": {"type": "noul", "instructions": 5}})
    assert "questions.q.noul.instructions" in str(e.value)


def test_auth_errors(live_auth, monkeypatch):
    monkeypatch.setenv(BASE_URL_ENV, live_auth.url)
    with TypeSafeClient(api_key="wrong", retry=NO_RETRY) as c:
        with pytest.raises(TypeSafeAuthenticationError) as e:
            c.system_one("x", {"q": Noul()})
        assert "Invalid API key" in str(e.value)
    with TypeSafeClient(api_key="local-secret", retry=NO_RETRY) as c:
        assert c.system_one("x", {"q": Noul()}).nouls["q"].noul == 0.5


def test_async_client(live):
    async def go():
        async with AsyncTypeSafeClient(api_key="local", base_url=live.url, retry=NO_RETRY) as c:
            rs = await asyncio.gather(*(c.system_one(f"message {i}", {"q": Noul(instructions="Is it urgent?")}) for i in range(8)))
            return rs

    rs = asyncio.run(go())
    assert len({r.request_id for r in rs}) == 8
    assert all(r.model == "jev-local-heuristic-0.1.0" for r in rs)
