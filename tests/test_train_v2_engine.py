"""FastEngine v2: exact option batching (multi-pass == single pass, K up to 255), long states up to 8192,
raw logits, and (type, K-bucket) / Platt calibration incl. backward compatibility with v1 calibration.json."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from jev_local.engine.base import EngineError
from jev_local.engine.encoder import calibrate as cal
from jev_local.engine.encoder.calibrate import CalibRecord, bucket_key, fit_calibration, k_bucket, tau_for
from jev_local.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest

TOL = 1e-4


# ---------------------------------------------------------------------------- calibration (numpy only)


def test_k_buckets():
    assert [k_bucket(k) for k in (2, 3, 5, 6, 10, 11, 30, 31, 100, 101, 255, 400)] == \
        ["2", "3-5", "3-5", "6-10", "6-10", "11-30", "11-30", "31-100", "31-100", "101-255", "101-255", "101-255"]
    assert bucket_key("choice", 77) == "choice:31-100" and bucket_key("noul", 2) == "noul"


def _synthetic(kind: str, k: int, tau: float, n: int, seed: int) -> list[CalibRecord]:
    """Logits whose calibrated temperature is exactly `tau`: labels are drawn from softmax(z / tau)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        z = rng.normal(0, 3, size=k)
        p = np.exp(z / tau - (z / tau).max())
        p /= p.sum()
        y = rng.choice(k, p=p)
        t = np.zeros(k)
        t[y] = 1
        out.append(CalibRecord("q", kind, "h", z, t))
    return out


def test_fit_recovers_bucket_temperatures_and_platt():
    recs = _synthetic("choice", 2, 0.8, 3000, 0) + _synthetic("choice", 20, 2.0, 3000, 1) \
        + _synthetic("choice", 150, 3.0, 1500, 2)
    rng = np.random.default_rng(3)
    for _ in range(3000):
        z = rng.normal(0, 2)
        p = 1 / (1 + math.exp(-(0.5 * z - 0.7)))
        recs.append(CalibRecord("n", "noul", "hn", np.array([z]), np.array([float(rng.random() < p)])))
    c = fit_calibration(recs, min_per_header=10**9, min_per_bucket=100)
    assert c["by_bucket"]["choice:2"] == pytest.approx(0.8, rel=0.1)
    assert c["by_bucket"]["choice:11-30"] == pytest.approx(2.0, rel=0.1)
    assert c["by_bucket"]["choice:101-255"] == pytest.approx(3.0, rel=0.1)
    assert c["noul_platt"]["a"] == pytest.approx(0.5, rel=0.15) and c["noul_platt"]["b"] == pytest.approx(-0.7, abs=0.15)
    a, b = c["tau_k"]["choice"]
    assert b > 0  # temperature grows with K here
    assert c["_fit"]["choice:11-30"]["nll_after"] < c["_fit"]["choice:11-30"]["nll_before"]
    # lookup precedence: header > bucket > tau(K) > kind
    assert tau_for(c, "choice", "h", 20) == c["by_bucket"]["choice:11-30"]
    assert tau_for(c, "choice", "h", 7) == pytest.approx(min(max(a + b * math.log(7), 0.5), 5.0))
    c["by_header"]["h"] = 1.7
    assert tau_for(c, "choice", "h", 20) == 1.7
    assert tau_for(c, "choice", "h") == 1.7 and tau_for(c, "choice", "zz") == c["choice"]  # v1 lookups
    v1 = fit_calibration(recs, min_per_header=10**9, buckets=False)
    assert set(v1) == {"noul", "choice", "score", "by_header", "_fit"}


def test_bucket_clamp():
    recs = _synthetic("choice", 4, 0.1, 2000, 4)  # true tau 0.1: clamped to 0.5 in the bucket, not per kind
    c = fit_calibration(recs, min_per_header=10**9, min_per_bucket=50)
    assert c["by_bucket"]["choice:3-5"] == pytest.approx(0.5, abs=1e-3)
    assert c["choice"] < 0.2


# ---------------------------------------------------------------------------- engine (needs ettin-32m weights)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from jev_local.engine.encoder.engine import init_model, write_checkpoint

    enc, heads, tok = init_model(seed=0)
    with torch.no_grad():
        for mlp in (heads.choice_mlp, heads.score_mlp, heads.noul_mlp):
            mlp[2].weight.normal_(std=1.0, generator=torch.Generator().manual_seed(1))
    return write_checkpoint(tmp_path_factory.mktemp("fast") / "ckpt", enc, heads, tok, meta={"test": True})


def _engine(ckpt, **kw):
    from jev_local.engine.encoder.engine import FastEngine

    return FastEngine(ckpt, device="cpu", dtype=torch.float32, **kw)


def _many_options(k: int, desc_words: int = 12) -> ChoiceQuestion:
    words = "account card transfer refund payment balance limit pin fee exchange cash top-up".split()
    crit = {f"intent_{i}": " ".join(words[(i + j) % len(words)] for j in range(desc_words)) + f" case {i}"
            for i in range(k)}
    return ChoiceQuestion(instructions="Which banking intent does the message express?", criteria=crit)


QS_SMALL = {
    "urgent": NoulQuestion(instructions="Is the message urgent?"),
    "prio": ScoreQuestion(instructions="How important is it?", criteria=["low", "normal", "high"]),
}
STATE = {"message": "I was charged twice for the same coffee and my card limit is now reached."}


@pytest.mark.model
def test_option_batching_is_exact(ckpt):
    one = _engine(ckpt, max_tokens=2048, max_flat_tokens=16384)
    split = _engine(ckpt, max_tokens=2048, max_flat_tokens=600)
    qs = {"intent": _many_options(120), **QS_SMALL, "intent2": _many_options(40, 4)}
    a = one.evaluate(STATE, qs)
    b = split.evaluate(STATE, qs)
    assert a.timings_ms["passes"] == 1 and b.timings_ms["passes"] >= 4
    assert a.input_tokens == b.input_tokens > 2048  # total exceeds max_tokens; every branch fits
    for qid in qs:
        assert a.dists[qid].labels == b.dists[qid].labels
        assert max(abs(x - y) for x, y in zip(a.dists[qid].probs, b.dists[qid].probs)) <= TOL, qid
    assert abs(sum(a.dists["intent"].probs) - 1) < 1e-6
    # dense reference with passes agrees too
    dense = _engine(ckpt, max_tokens=2048, attn="dense")
    c = dense.evaluate(STATE, qs)
    assert c.timings_ms["passes"] > 1
    for qid in qs:
        assert max(abs(x - y) for x, y in zip(a.dists[qid].probs, c.dists[qid].probs)) <= TOL, qid


@pytest.mark.model
def test_255_options_supported_256_not(ckpt):
    eng = _engine(ckpt)
    q = _many_options(255, 8)
    req = SystemOneRequest(state=STATE, model="m", questions={"q": q})
    assert eng.count_tokens(req) > eng.max_tokens and eng.supports(req)
    res = eng.evaluate(STATE, {"q": q})
    assert len(res.dists["q"].probs) == 255 and abs(sum(res.dists["q"].probs) - 1) < 1e-6
    q256 = ChoiceQuestion(instructions="pick", criteria={f"o{i}": None for i in range(256)})
    assert not eng.supports(SystemOneRequest(state="x", model="m", questions={"q": q256}))


@pytest.mark.model
def test_long_state_up_to_8192(ckpt):
    long_state = {"doc": "the quarterly report shows revenue growth in every region " * 640}  # ~6.4k tokens
    qs = {**QS_SMALL, "intent": _many_options(30, 4)}
    short = _engine(ckpt)
    req = SystemOneRequest(state=long_state, model="m", questions=qs)
    assert not short.supports(req)
    with pytest.raises(EngineError) as ei:
        short.evaluate(long_state, qs)
    assert ei.value.detail["detail"] == "max_tokens_exceeded"
    eng = _engine(ckpt, max_tokens=8192)
    assert eng.supports(req)
    a = eng.evaluate(long_state, qs)
    assert 6000 < a.input_tokens and a.timings_ms["passes"] == 1
    # forcing passes on a long state is still exact
    b = _engine(ckpt, max_tokens=8192, max_flat_tokens=a.input_tokens - 200).evaluate(long_state, qs)
    assert b.timings_ms["passes"] >= 2
    for qid in qs:
        assert max(abs(x - y) for x, y in zip(a.dists[qid].probs, b.dists[qid].probs)) <= TOL, qid
    too_long = {"doc": "the quarterly report shows revenue growth in every region " * 1200}
    assert not eng.supports(SystemOneRequest(state=too_long, model="m", questions=qs))


@pytest.mark.model
def test_evaluate_logits_and_bucket_calibration(ckpt):
    eng = _engine(ckpt)
    qs = {"intent": _many_options(40, 3), "small": _many_options(3, 2), **QS_SMALL}
    raw = eng.evaluate_logits(STATE, qs)
    base = eng.evaluate(STATE, qs).dists
    for qid, r in raw.items():
        p = cal.calibrate_logits(r["kind"], r["header_key"], r["logits"], eng.calib)
        assert np.allclose(p, base[qid].probs, atol=1e-6), qid
    assert len(raw["intent"]["logits"]) == 40 and len(raw["urgent"]["logits"]) == 1
    # a K-bucket temperature applies only to questions in that bucket
    eng.calib = {"noul": 1.0, "choice": 1.0, "score": 1.0, "by_header": {}, "by_bucket": {"choice:31-100": 4.0},
                 "noul_platt": {"a": 0.5, "b": 1.0}}
    got = eng.evaluate(STATE, qs).dists
    assert max(got["intent"].probs) < max(base["intent"].probs)
    assert got["small"].probs == pytest.approx(base["small"].probs, abs=1e-6)
    z = raw["urgent"]["logits"][0]
    assert got["urgent"].probs[0] == pytest.approx(1 / (1 + math.exp(-(0.5 * z + 1.0))), abs=1e-6)
    # a v1 calibration.json (no buckets) behaves exactly as before
    eng.calib = {"noul": 2.0, "choice": 1.5, "score": 1.0, "by_header": {}}
    v1 = eng.evaluate(STATE, qs).dists
    zs = np.asarray(raw["intent"]["logits"]) / 1.5
    want = np.exp(zs - zs.max()) / np.exp(zs - zs.max()).sum()
    assert np.allclose(v1["intent"].probs, want, atol=1e-6)
    assert v1["urgent"].probs[0] == pytest.approx(1 / (1 + math.exp(-z / 2.0)), abs=1e-6)


@pytest.mark.model
def test_calibrate_cli_writes_v2_file(tmp_path, ckpt):
    import shutil

    from jev_local.train import fixture

    data = fixture.write_splits(tmp_path / "cu", n_train=4, n_dev=40, seed=0, n_elements=8)
    ck = tmp_path / "ck"
    shutil.copytree(ckpt, ck)
    cal.main(["--ckpt", str(ck), "--data", str(data), "--split", "dev", "--device", "cpu", "--min-per-header", "1000",
              "--min-per-bucket", "10"])
    c = json.loads((ck / "calibration.json").read_text())
    assert c["version"] == 2 and c["by_bucket"] and "noul_platt" in c
    for v in c["by_bucket"].values():
        assert 0.5 <= v <= 5.0
    from jev_local.train import eval as ev

    out = tmp_path / "e.json"
    ev.main(["--ckpt", str(ck), "--data", str(data), "--split", "dev", "--device", "cpu", "--out", str(out)])
    m = json.loads(out.read_text())
    assert set(m["by_kbucket"]) >= {"noul"} and any(k.startswith("choice:") for k in m["by_kbucket"])
    assert m["dropped_qids"] == 0
