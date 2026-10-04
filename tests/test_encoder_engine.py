"""FastEngine: Engine protocol, answer shapes on a real harness request, and the exact invariances
the packing promises (CONTRACT "B" tests (b) and (c))."""

from __future__ import annotations

import json
import random

import pytest
import torch

from jev_local.engine.base import Engine, EngineError
from jev_local.engine.encoder.calibrate import header_key
from jev_local.engine.encoder.engine import FastEngine, init_model, load_checkpoint, write_checkpoint
from jev_local.harness.questions import build_questions, rank_apps
from jev_local.harness.spans import extract_text_candidates, extract_url_candidates
from jev_local.harness.state import build_state
from jev_local.schema import ChoiceQuestion, NoulCriteria, NoulQuestion, ScoreQuestion, SystemOneRequest
from jev_local.train.fixture import APPS, make_snapshot

pytestmark = pytest.mark.model  # needs the cached ettin weights

TOL = 1e-4


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    enc, heads, tok = init_model(seed=0)
    # Scale the last head layers up so random heads give clearly non-uniform answers; otherwise the
    # invariance checks below would pass trivially on near-uniform distributions.
    with torch.no_grad():
        for mlp in (heads.choice_mlp, heads.score_mlp, heads.noul_mlp):
            mlp[2].weight.normal_(std=1.0, generator=torch.Generator().manual_seed(1))
    return write_checkpoint(tmp_path_factory.mktemp("fast") / "ckpt", enc, heads, tok, meta={"test": True})


@pytest.fixture(scope="module")
def engine(ckpt):
    return FastEngine(ckpt, device="cpu", dtype=torch.float32)


def harness_request(n_elements: int = 60, text: str = "open notes and type hello world", seed: int = 0):
    snap = make_snapshot(random.Random(seed), n_elements)
    apps = rank_apps(text, APPS, running=[snap.app_name])
    qs = build_questions(snap, apps, extract_text_candidates(text), extract_url_candidates(text))
    return build_state(text, snap), qs


def generic_questions():
    return {
        "dept": ChoiceQuestion(instructions="Which department should handle the ticket?",
                               criteria={"billing": "payments, payouts, invoices", "tech": "bugs and outages",
                                         "sales": "new contracts", "other": None}),
        "urgent": NoulQuestion(instructions="Is the ticket urgent?",
                               criteria=NoulCriteria(true="customer is blocked", false="can wait")),
        "prio": ScoreQuestion(instructions="How important is it?", criteria=["low", "normal", "high", "critical"]),
    }


STATE = {"ticket": "Our payout failed twice this week and the dashboard shows an error.", "customer": "Acme"}


def test_protocol_and_checkpoint_layout(engine, ckpt):
    assert isinstance(engine, Engine)
    assert engine.name == "jev-local-fast-0.1.0"
    assert engine.max_tokens == 2048
    for f in ("backbone/config.json", "backbone/model.safetensors", "backbone/tokenizer.json", "heads.safetensors",
              "calibration.json", "meta.json"):
        assert (ckpt / f).is_file(), f
    cal = json.loads((ckpt / "calibration.json").read_text())
    assert {"noul", "choice", "score", "by_header"} <= set(cal)
    assert json.loads((ckpt / "meta.json").read_text())["test"] is True


def test_checkpoint_roundtrip_is_lossless(ckpt):
    enc, heads, tok, calib, meta = load_checkpoint(ckpt)
    enc2, heads2, _, _, _ = load_checkpoint(ckpt)
    for (k, a), (_, b) in zip(heads.state_dict().items(), heads2.state_dict().items()):
        assert torch.equal(a, b), k
    assert enc.backbone.get_input_embeddings().num_embeddings == len(tok) == 50373
    assert all(tok.convert_tokens_to_ids(m) >= 50368 for m in ("[Q]", "[O]", "[L]", "[T]", "[F]"))


def test_harness_request_shapes(engine):
    state, qs = harness_request()
    res = engine.evaluate(state, qs)
    req = SystemOneRequest(state=state, model="jev-local-fast", questions=qs)
    assert res.engine == engine.name
    assert res.input_tokens == engine.count_tokens(req)
    assert res.output_tokens == sum(len(q.criteria) + 1 if q.type != "noul" else 3 for q in qs.values())
    assert set(res.dists) == set(qs)
    for qid, q in qs.items():
        d = res.dists[qid]
        assert d.kind == q.type
        if q.type == "noul":
            assert d.labels == () and len(d.probs) == 1 and 0.0 <= d.probs[0] <= 1.0
        else:
            want = tuple(q.criteria) if q.type == "choice" else tuple(str(i) for i in range(len(q.criteria)))
            assert d.labels == want
            assert abs(sum(d.probs) - 1.0) < 1e-5
            assert all(p >= 0 for p in d.probs)
    assert {"serialize", "pack", "forward", "heads", "total"} <= set(res.timings_ms)
    # heads are random but scaled up: answers must not be uniform
    assert max(res.dists["intent"].probs) > 2.0 / len(qs["intent"].criteria)


def test_permuting_options_is_exact(engine):
    """(b): permuting choice options leaves each label's probability unchanged."""
    state, qs = harness_request()
    base = engine.evaluate(state, qs).dists
    rng = random.Random(3)
    perm = dict(qs)
    for qid in ("intent", "target", "app", "key"):
        items = list(qs[qid].criteria.items())
        rng.shuffle(items)
        perm[qid] = ChoiceQuestion(instructions=qs[qid].instructions, criteria=dict(items))
    got = engine.evaluate(state, perm).dists
    for qid in ("intent", "target", "app", "key"):
        a = dict(zip(base[qid].labels, base[qid].probs))
        b = dict(zip(got[qid].labels, got[qid].probs))
        assert got[qid].labels == tuple(perm[qid].criteria)  # answer follows the request's order
        assert max(abs(a[k] - b[k]) for k in a) <= TOL, qid


def test_unrelated_question_is_exact(engine):
    """(c): adding (or removing, or reordering) questions leaves every other answer unchanged."""
    qs = generic_questions()
    base = engine.evaluate(STATE, qs).dists
    extra = {"lang": ChoiceQuestion(instructions="Which language is the ticket in?",
                                    criteria={"english": None, "french": None, "german": None})}
    more = {**extra, **dict(reversed(list(qs.items())))}
    got = engine.evaluate(STATE, more).dists
    alone = engine.evaluate(STATE, {"urgent": qs["urgent"]}).dists
    for qid in qs:
        assert max(abs(x - y) for x, y in zip(base[qid].probs, got[qid].probs)) <= TOL, qid
    assert abs(base["urgent"].probs[0] - alone["urgent"].probs[0]) <= TOL


def test_harness_invariance_to_extra_elements_question(engine):
    """The target question (60 options, several item chunks) never leaks into the intent question."""
    state, qs = harness_request()
    base = engine.evaluate(state, qs).dists
    without = {k: v for k, v in qs.items() if k != "target"}
    got = engine.evaluate(state, without).dists
    for qid in without:
        assert max(abs(x - y) for x, y in zip(base[qid].probs, got[qid].probs)) <= TOL, qid


def test_tree_and_dense_engines_agree(ckpt, engine):
    dense = FastEngine(ckpt, device="cpu", dtype=torch.float32, attn="dense")
    state, qs = harness_request(n_elements=25)
    a = engine.evaluate(state, qs).dists
    b = dense.evaluate(state, qs).dists
    for qid in qs:
        assert max(abs(x - y) for x, y in zip(a[qid].probs, b[qid].probs)) <= TOL, qid


def test_state_matters(engine):
    qs = generic_questions()
    a = engine.evaluate(STATE, qs).dists["dept"].probs
    b = engine.evaluate({"ticket": "I would like to buy 400 more seats next quarter."}, qs).dists["dept"].probs
    assert max(abs(x - y) for x, y in zip(a, b)) > 1e-3


def test_max_tokens_exceeded(engine):
    big = {"doc": "lorem ipsum dolor sit amet " * 600}
    qs = generic_questions()
    req = SystemOneRequest(state=big, model="jev-local-fast", questions=qs)
    assert engine.count_tokens(req) > engine.max_tokens
    assert not engine.supports(req)
    with pytest.raises(EngineError) as ei:
        engine.evaluate(big, qs)
    assert ei.value.status == 400
    assert ei.value.detail["detail"] == "max_tokens_exceeded"


def test_too_many_options_not_supported(engine):
    q = ChoiceQuestion(instructions="pick", criteria={f"o{i}": None for i in range(256)})
    assert not engine.supports(SystemOneRequest(state="x", model="m", questions={"q": q}))


def test_calibration_temperatures(ckpt):
    eng = FastEngine(ckpt, device="cpu", dtype=torch.float32)
    qs = generic_questions()
    base = eng.evaluate(STATE, qs).dists
    eng.calib = {"noul": 1.0, "choice": 1.0, "score": 1.0,
                 "by_header": {header_key(qs["dept"].instructions): 4.0}}
    got = eng.evaluate(STATE, qs).dists
    assert max(got["dept"].probs) < max(base["dept"].probs)  # hotter -> flatter
    assert got["urgent"].probs == pytest.approx(base["urgent"].probs, abs=1e-6)
    eng.calib = {"noul": 0.5, "choice": 1.0, "score": 1.0, "by_header": {}}
    z = torch.logit(torch.tensor(base["urgent"].probs[0], dtype=torch.float64))
    want = torch.sigmoid(z / 0.5).item()
    assert eng.evaluate(STATE, qs).dists["urgent"].probs[0] == pytest.approx(want, abs=1e-5)


def test_api_end_to_end(engine):
    from jev_local.api import system_one

    req = SystemOneRequest(state=STATE, model="jev-local-fast", questions=generic_questions())
    resp = system_one(req, {"fast": engine})
    assert set(resp.answers) == {"dept", "urgent", "prio"}
    assert resp.answers["dept"].choice in {"billing", "tech", "sales", "other"}
    assert resp.usage.input_tokens == engine.count_tokens(req)


@pytest.mark.slow
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_fp16_close_to_cpu_fp32(ckpt, engine):
    mps = FastEngine(ckpt, device="mps", dtype=torch.float16)
    state, qs = harness_request()
    a = engine.evaluate(state, qs).dists
    b = mps.evaluate(state, qs).dists
    worst = max(abs(x - y) for qid in qs for x, y in zip(a[qid].probs, b[qid].probs))
    assert worst < 0.02, worst
