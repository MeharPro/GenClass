"""benchmax W1–W11 on the real fast engine and the v2 checkpoint (VM only: loads torch).

    JEV_LOCAL_FAST_CKPT=~/jev/models/jev-local-fast-v2 ~/jev/.venv/bin/python -m pytest tests/test_bm_engine_vm.py -q

Every text here is synthetic (no benchmark items: PLAN §2.2 hygiene). What is checked:
8k inputs and 255 options through the HTTP server, empty states, object instructions, null == "" == key,
truncate vs refuse with the Decision Index markers, calibration selected by run config, and determinism.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jev_local.api import PRESETS, ServeConfig, determinism_report, system_one_json
from jev_local.engine.base import EngineError
from jev_local.engine.registry import Registry
from jev_local.schema import SystemOneRequest
from jev_local.server.app import create_app

CKPT = Path(os.environ.get("JEV_LOCAL_FAST_CKPT", Path.home() / "jev" / "models" / "jev-local-fast-v2"))
pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(sys.platform == "darwin", reason="loads torch: VM only (CONTRACT Mac safety)"),
    pytest.mark.skipif(not (CKPT / "heads.safetensors").exists(), reason=f"no checkpoint at {CKPT}"),
]

WORDS = ("payment card arrived late and the mobile app shows an error when I open the transfers tab after the update "
         "please tell me whether the refund for the duplicated charge is on its way because support said two days").split()


def long_text(n_words: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


class StubRegistry(Registry):
    def __init__(self, engines: dict):
        super().__init__(load_fast=False, load_general="off")
        self._given = engines

    def _build(self) -> dict:
        return dict(self._given)


@pytest.fixture(scope="module")
def engine():
    from jev_local.engine.encoder.engine import FastEngine

    return FastEngine(CKPT, device="cpu", threads=8)


@pytest.fixture(scope="module")
def client(engine):
    cfg = ServeConfig.preset("di", model_id="meharsjev-68m")
    with TestClient(create_app(StubRegistry({"fast": engine}), config=cfg, warm_on_startup=False)) as c:
        yield c


def text_of_tokens(engine, n_tokens: int, seed: int = 0) -> str:
    """Synthetic text that packs to about n_tokens tokens."""
    words = int(n_tokens * 0.8)
    for _ in range(6):
        t = long_text(words, seed)
        n = len(engine.packer.encode([t])[0])
        if abs(n - n_tokens) <= 32:
            return t
        words = int(words * n_tokens / max(n, 1))
    return t


QUESTIONS = {
    "intent": {"type": "choice", "instructions": "Which banking intent does the message express?",
               "criteria": {"card_arrival": None, "lost_card": None, "refund_status": None, "app_error": None}},
    "urgent": {"type": "noul", "instructions": "Is the customer blocked right now?",
               "criteria": {"true": "cannot use money or the card", "false": "can wait a few days"}},
    "anger": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "annoyed", "angry"]},
}


# ---------------------------------------------------------------------------- limits from the checkpoint


def test_max_tokens_and_calibration_come_from_the_checkpoint(engine):
    assert engine.max_tokens == min(int(engine.meta.get("max_len", 2048)), 8192)
    assert engine.max_tokens == 8192, "the v2 checkpoint is 8k-native; the server must not cap it at 2048"
    info = engine.calibration_info
    assert info["source"].endswith("calibration.json") and len(info["sha256"]) == 64
    assert set(info["per_kind"]) == {"noul", "choice", "score"}


def test_255_long_options_run_as_several_exact_passes_and_pass_the_router(engine, client):
    crit = {f"option_{i}": f"{long_text(28, i)} (topic {i})" for i in range(255)}
    req = {"state": {"query": "where is the card I ordered"}, "model": "meharsjev-68m",
           "questions": {"q": {"type": "choice", "instructions": "Which option matches the query?", "criteria": crit}}}
    r = SystemOneRequest.model_validate(req)
    total, need = engine.count_tokens(r), engine.positions_needed(r)
    assert total > engine.max_tokens > need, (total, need)  # whole request too long, one branch fits
    assert engine.supports(r)
    resp = client.post("/v1/systemone", json=req)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    p = body["answers"]["q"]["probabilities"]
    assert list(p) == list(crit) and abs(sum(p.values()) - 1.0) < 1e-9 and body["answers"]["q"]["choice"] in crit
    assert body["model"] == "meharsjev-68m" and body["usage"]["input_tokens"] == total
    assert engine.last_passes > 1


def test_8k_state_through_the_http_server(engine, client):
    text = text_of_tokens(engine, 7_900)
    req = {"state": {"document": text}, "model": "meharsjev-68m",
           "questions": {"q": {"type": "choice", "instructions": "What is the document about?",
                               "criteria": {"banking": None, "weather": None}},
                         "n": {"type": "noul", "instructions": "Does it mention a refund?"}}}
    r = SystemOneRequest.model_validate(req)
    assert 7_800 <= engine.positions_needed(r) <= 8_192, engine.positions_needed(r)
    resp = client.post("/v1/systemone", json=req)
    assert resp.status_code == 200, resp.text
    assert resp.headers["x-jev-local-engine"] == "fast" and "x-jev-local-truncated" not in resp.headers
    assert resp.json()["usage"]["input_tokens"] >= 7_800


# ---------------------------------------------------------------------------- W6 / W7 / W8


@pytest.mark.parametrize("state", [{}, "", []])
def test_empty_state_with_the_input_in_instructions(client, state):
    req = {"state": state, "model": "meharsjev-68m",
           "questions": {"q": {"type": "choice",
                               "instructions": "Classify the message. Message: my card never arrived. Options follow.",
                               "criteria": {"A": "card arrival", "B": "weather", "C": "sports"}},
                         "n": {"type": "noul", "instructions": "Is the message about a card?"}}}
    resp = client.post("/v1/systemone", json=req)
    assert resp.status_code == 200, resp.text
    a = resp.json()["answers"]
    assert abs(sum(a["q"]["probabilities"].values()) - 1.0) < 1e-9 and 0.0 <= a["n"]["noul"] <= 1.0
    assert max(a["q"]["probabilities"].values()) > 1 / 3  # not uniform: the instructions were read


def test_object_instructions_and_object_criteria(client):
    req = {"state": {"text": long_text(30, 3)}, "model": "meharsjev-68m",
           "questions": {
               "harassment": {"type": "noul",
                              "instructions": {"category": {"name": "harassment", "definition": "insults or threatens a person"},
                                               "question": "Does the text fall under the category?"}},
               "coherence": {"type": "score", "instructions": {"criterion": "coherence", "question": "Rate the summary"},
                             "criteria": [{"level": "incoherent"}, "mixed", ["coherent", "fully coherent"]]},
               "relevant": {"type": "noul", "instructions": {"task": "find documents about refunds", "candidate": long_text(12, 4)}},
           }}
    resp = client.post("/v1/systemone", json=req)
    assert resp.status_code == 200, resp.text
    a = resp.json()["answers"]
    assert a["coherence"]["legend"] == {"0": {"level": "incoherent"}, "1": "mixed", "2": ["coherent", "fully coherent"]}
    assert set(a) == {"harassment", "coherence", "relevant"}


def test_null_empty_string_and_key_descriptions_are_the_same_request(engine):
    es = {"fast": engine}
    cfg = ServeConfig.preset("di", model_id="meharsjev-68m")
    base = {"state": {"query": "I lost my card yesterday"}, "model": "meharsjev-68m",
            "questions": {"q": {"type": "choice", "instructions": "Which intent?", "criteria": {"lost_card": None, "card_arrival": None, "refund": None}}}}
    a = system_one_json(base, es, cfg)
    b = system_one_json({**base, "questions": {"q": {**base["questions"]["q"], "criteria": {k: "" for k in base["questions"]["q"]["criteria"]}}}}, es, cfg)
    c = system_one_json({**base, "questions": {"q": {**base["questions"]["q"], "criteria": {k: k for k in base["questions"]["q"]["criteria"]}}}}, es, cfg)
    assert a == b == c
    d = system_one_json({**base, "questions": {"q": {**base["questions"]["q"], "criteria": {k: k for k in base["questions"]["q"]["criteria"]}}}},
                        es, ServeConfig.preset("di", model_id="meharsjev-68m", label_alias=False))
    assert d != a  # without W8 the model would read "lost_card: lost_card"


# ---------------------------------------------------------------------------- W4 / W5


def test_refuse_vs_truncate_on_a_12k_state(engine):
    text = text_of_tokens(engine, 12_000, seed=7)
    req = {"state": {"document": text, "note": "short"}, "model": "meharsjev-68m", "questions": QUESTIONS}
    di = TestClient(create_app(StubRegistry({"fast": engine}), config=ServeConfig.preset("di", model_id="meharsjev-68m"), warm_on_startup=False))
    r = di.post("/v1/systemone", json=req)
    assert r.status_code == 400 and r.json()["detail"][0]["type"] == "max_tokens_exceeded"
    assert "maximum context length" in r.json()["detail"][0]["msg"]
    assert di.get("/healthz").json()["benchmax"]["counters"]["refused_context"] == 1
    jb = TestClient(create_app(StubRegistry({"fast": engine}), config=ServeConfig.preset("jevbench", model_id="meharsjev-68m"), warm_on_startup=False))
    assert jb.post("/v1/systemone", json=req).status_code == 422
    de = TestClient(create_app(StubRegistry({"fast": engine}), config=ServeConfig.preset("deusser", model_id="meharsjev-68m"), warm_on_startup=False))
    r = de.post("/v1/systemone", json=req)
    assert r.status_code == 200, r.text
    cut = int(r.headers["x-jev-local-truncated"])
    assert cut >= 12_000 - 8_192 - 64, cut
    assert r.json()["usage"]["input_tokens"] <= 8_192 + 64
    counters = de.get("/healthz").json()["benchmax"]["counters"]
    assert counters["truncated_requests"] == 1 and counters["truncated_tokens"] == cut and counters["served"] == 1


def test_fit_state_keeps_shape_cuts_only_the_longest_field_and_never_touches_questions(engine):
    from jev_local.schema import question_from_json

    qs = {k: question_from_json(v) for k, v in QUESTIONS.items()}
    short = {"document": "a short note", "tags": ["x", "y"]}
    assert engine.fit_state(short, qs) == (short, 0)
    big = {"document": text_of_tokens(engine, 9_000, seed=1), "tags": ["refund", "card"], "n": 3}
    state, cut = engine.fit_state(big, qs)
    assert cut > 0 and list(state) == ["document", "tags", "n"] and state["tags"] == ["refund", "card"] and state["n"] == 3
    assert big["document"].startswith(state["document"][:200])  # cut at the end
    r = SystemOneRequest(state=state, model="meharsjev-68m", questions=qs)
    assert engine.positions_needed(r) <= engine.max_tokens and engine.supports(r)
    # string and list states keep their shape too
    s, c = engine.fit_state(text_of_tokens(engine, 9_000, seed=2), qs)
    assert isinstance(s, str) and c > 0
    lst, c = engine.fit_state(["tiny", text_of_tokens(engine, 9_000, seed=3)], qs)
    assert isinstance(lst, list) and lst[0] == "tiny" and c > 0
    # a single option longer than the window can never fit: the engine refuses (W4 marker added by api)
    huge = {"q": question_from_json({"type": "choice", "criteria": {"a": text_of_tokens(engine, 8_300, seed=4), "b": None}})}
    with pytest.raises(EngineError):
        engine.fit_state({"query": "x"}, huge)


# ---------------------------------------------------------------------------- W9 / W11


def test_calibration_file_selected_by_run_config():
    from jev_local.engine.encoder.engine import FastEngine

    hot = FastEngine(CKPT, device="cpu", threads=8, calibration={"noul": 1.0, "choice": 4.0, "score": 1.0, "by_header": {"deadbeef0000": 9.0}})
    assert hot.calibration_info["source"] == "inline" and hot.calibration_info["n_by_header"] == 1
    base = FastEngine(CKPT, device="cpu", threads=8)
    qs = {k: __import__("jev_local.schema", fromlist=["question_from_json"]).question_from_json(v) for k, v in QUESTIONS.items()}
    st = {"query": "my card never arrived and I am furious"}
    assert max(hot.evaluate(st, qs).dists["intent"].probs) < max(base.evaluate(st, qs).dists["intent"].probs)
    g = FastEngine(CKPT, device="cpu", threads=8, calibration={"choice": 4.0, "by_header": {"deadbeef0000": 9.0}}, drop_header_calibration=True)
    assert g.calib["by_header"] == {} and g.calibration_info["by_header_dropped"] is True
    desc = Registry(fast_ckpt=CKPT, load_general="off", fast_options={"device": "cpu", "threads": 8}).describe()
    assert set(desc["fast"]["checkpoint_sha256"]) >= {"heads.safetensors", "backbone/model.safetensors"}
    assert desc["fast"]["max_tokens"] == 8192


def test_determinism_same_bytes_and_order_invariance(engine):
    es = {"fast": engine}
    cfg = ServeConfig.preset("di", model_id="meharsjev-68m")
    reqs = [
        {"state": {"query": long_text(40, 5)}, "model": "meharsjev-68m", "questions": QUESTIONS},
        {"state": {}, "model": "meharsjev-68m",
         "questions": {"q": {"type": "choice", "instructions": f"Message: {long_text(20, 6)}. Which option?",
                             "criteria": {f"option_{i}": f"{long_text(6, 10 + i)}" for i in range(40)}}}},
        {"state": text_of_tokens(engine, 3_000, seed=8), "model": "meharsjev-68m",
         "questions": {"s": {"type": "score", "instructions": "How formal is the text?", "criteria": ["casual", "neutral", "formal", "legal"]}}},
    ]
    rep = determinism_report(lambda r: system_one_json(r, es, cfg), reqs, repeats=5)
    assert rep["identical_bytes_fraction"] == 1.0, rep
    assert rep["repeat_flip_rate"] == 0.0 and rep["max_abs_diff_repeat"] == 0.0
    assert rep["permutation"]["argmax_flips"] == 0 and rep["permutation"]["max_abs_diff"] < 1e-5, rep["permutation"]
    assert rep["hardware"].get("torch_threads") == 8
    print(json.dumps({k: v for k, v in rep.items() if k != "per_request"}, indent=1))
