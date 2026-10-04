"""Decision heads, losses, calibration math and eval metrics (no model weights needed)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from jev_local.engine.encoder.calibrate import CalibRecord, apply_tau, fit_calibration, fit_tau, header_key, nll
from jev_local.engine.encoder.heads import DecisionHeads, build_plan, probabilities
from jev_local.engine.encoder.tokenize_pack import Packed, QIndex
from jev_local.serialize import QBlock
from jev_local.train.eval import auroc, ece
from jev_local.train.losses import LossConfig, Targets, compute_loss, rps
from jev_local.train.train import lr_factor, make_batches, parse_target

D = 16


def _pack(n_tokens: int, questions: list[tuple[str, str, int]]) -> Packed:
    """Fake Packed: questions laid out one after another, 3 tokens per header and per item."""
    qidx, pos = {}, 2
    for qid, kind, k in questions:
        q_pos = pos
        pos += 3
        items = tuple(pos + 3 * j for j in range(k))
        pos += 3 * k
        labels = ("true", "false") if kind == "noul" else tuple(f"{qid}{j}" for j in range(k))
        qidx[qid] = QIndex(qid, kind, f"header {qid}", labels, q_pos, items)
    n = max(n_tokens, pos)
    return Packed(list(range(n)), list(range(n)), [0] * n, [0] * n, 2, qidx)


def _plan_and_h(seed: int = 0):
    packs = [
        _pack(40, [("a", "choice", 3), ("n", "noul", 2), ("s", "score", 4)]),
        _pack(60, [("b", "choice", 5), ("m", "noul", 2)]),
    ]
    L = 64
    plan = build_plan(packs, L)
    h = torch.randn(2, L, D, generator=torch.Generator().manual_seed(seed))
    return packs, plan, h


def test_plan_groups_and_shapes():
    packs, plan, h = _plan_and_h()
    heads = DecisionHeads(D)
    out = heads(h, plan)
    assert out.choice.shape == (2, 5) and out.score.shape == (1, 4) and out.noul.shape == (2,)
    assert torch.isinf(out.choice[0, 3:]).all() and torch.isfinite(out.choice[0, :3]).all()
    assert [r.qi.qid for r in plan.refs] == ["a", "n", "s", "b", "m"]
    assert plan.choice.q.tolist() == [packs[0].q_index["a"].q_pos, 64 + packs[1].q_index["b"].q_pos]


def test_probabilities_and_temperature():
    _, plan, h = _plan_and_h()
    heads = DecisionHeads(D)
    with torch.no_grad():
        heads.choice_mlp[2].weight.mul_(100)
    out = heads(h, plan)
    p1 = probabilities(out, plan, {"choice": 1.0, "score": 1.0, "noul": 1.0, "by_header": {}})
    assert torch.allclose(p1["choice"].sum(-1), torch.ones(2))
    assert (p1["choice"][0, 3:] == 0).all()
    hot = probabilities(out, plan, {"choice": 1.0, "score": 1.0, "noul": 1.0,
                                    "by_header": {header_key("header a"): 5.0}})
    assert hot["choice"][0].max() < p1["choice"][0].max()  # only question "a" flattened
    assert torch.allclose(hot["choice"][1], p1["choice"][1])


def test_choice_head_is_permutation_equivariant():
    """The head scores each option independently of its slot: permuting item positions permutes logits."""
    packs, plan, h = _plan_and_h()
    heads = DecisionHeads(D)
    z = heads(h, plan).choice[1]
    qi = packs[1].q_index["b"]
    perm = [4, 2, 0, 1, 3]
    packs[1].q_index["b"] = QIndex(qi.qid, qi.kind, qi.header, tuple(qi.labels[i] for i in perm), qi.q_pos,
                                   tuple(qi.item_pos[i] for i in perm))
    z2 = heads(h, build_plan(packs, 64)).choice[1]
    assert torch.allclose(z2, z[perm])


def test_losses_decrease_under_sgd_and_skip_unlabelled():
    _, plan, h = _plan_and_h()
    heads = DecisionHeads(D)
    tg = Targets(
        choice_idx=torch.tensor([1]), choice=torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]]), choice_hard=torch.tensor([True]),
        score_idx=torch.tensor([0]), score=torch.tensor([[0.0, 0.0, 0.3, 0.7]]), score_hard=torch.tensor([False]),
        noul_idx=torch.tensor([0, 1]), noul=torch.tensor([1.0, 0.0]),
    )
    opt = torch.optim.SGD(heads.parameters(), lr=0.1)
    losses = []
    for _ in range(30):
        loss, parts = compute_loss(heads(h, plan), tg, LossConfig(smoothing=0.02, brier=0.1, rps=0.25))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
    assert parts["choice"][1] == 1 and parts["score"][1] == 1 and parts["noul"][1] == 2
    none, parts = compute_loss(heads(h, plan), Targets(), LossConfig())
    assert none is None and parts == {}


def test_rps_is_ordinal():
    t = torch.tensor([[0.0, 0.0, 1.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    near = rps(torch.tensor([[0.0, 5.0, 0.0]]), t, mask)
    far = rps(torch.tensor([[5.0, 0.0, 0.0]]), t, mask)
    assert far > near


def test_parse_target():
    ch = QBlock("t", "choice", "h", ("a", "b", "none"), ("a", "b", "none"))
    assert parse_target(ch, {"type": "choice", "label": "b"}).dist == [0.0, 1.0, 0.0]
    assert parse_target(ch, {"type": "choice", "label": "zzz"}) is None
    soft = parse_target(ch, {"type": "choice", "dist": {"a": 0.7, "b": 0.3}})
    assert soft.dist == pytest.approx([0.7, 0.3, 0.0]) and not soft.hard
    sc = QBlock("s", "score", "h", ("x", "y", "z"), ("0", "1", "2"))
    assert parse_target(sc, {"type": "score", "level": 2}).dist == [0.0, 0.0, 1.0]
    assert parse_target(sc, {"type": "score", "level": 3}) is None
    assert parse_target(sc, {"type": "score", "dist": [1, 1, 2]}).dist == pytest.approx([0.25, 0.25, 0.5])
    nb = QBlock("n", "noul", "h", ("yes", "no"), ("true", "false"))
    assert parse_target(nb, {"type": "noul", "p": 0.8}).dist == [0.8]
    assert parse_target(nb, {"type": "choice", "label": "a"}) is None  # wrong type
    assert parse_target(nb, None) is None


def test_length_bucketed_batches():
    rng = np.random.default_rng(0)
    lengths = rng.integers(100, 5000, size=1000).tolist()
    batches = make_batches(lengths, 4, seed=1)
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(1000))
    spread = np.mean([max(lengths[i] for i in b) - min(lengths[i] for i in b) for b in batches])
    rand = np.mean([np.ptp(rng.choice(lengths, 4)) for _ in range(250)])
    assert spread < 0.1 * rand  # batches hold similar lengths
    assert make_batches(lengths, 4, seed=1) == batches  # deterministic (resume relies on it)
    assert make_batches(lengths, 4, seed=2) != batches


def test_lr_schedule():
    total = 1000
    f = [lr_factor(s, total) for s in range(total)]
    assert f[0] == pytest.approx(1 / 30) and max(f) == pytest.approx(1.0)
    assert f[29] == pytest.approx(1.0) and f[-1] < 0.01
    assert all(a >= b - 1e-12 for a, b in zip(f[30:], f[31:]))  # cosine decay is monotone


def _sample_records(kind: str, tau: float, n: int, k: int, seed: int) -> list[CalibRecord]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        if kind == "noul":
            z = rng.normal(0, 3, size=1)
            y = float(rng.random() < 1 / (1 + math.exp(-z[0] / tau)))
            out.append(CalibRecord("q", "noul", "h", z, np.array([y])))
        else:
            z = rng.normal(0, 3, size=k)
            p = np.exp(z / tau - (z / tau).max())
            p /= p.sum()
            t = np.zeros(k)
            t[rng.choice(k, p=p)] = 1
            out.append(CalibRecord("q", kind, "h", z, t))
    return out


@pytest.mark.parametrize("kind,tau", [("choice", 2.0), ("choice", 0.5), ("noul", 3.0), ("score", 1.5)])
def test_fit_tau_recovers_true_temperature(kind, tau):
    recs = _sample_records(kind, tau, 3000, 4, seed=1)
    got = fit_tau(recs)
    assert got == pytest.approx(tau, rel=0.15)
    assert nll(recs, got) <= nll(recs, 1.0) + 1e-12


def test_fit_calibration_by_header():
    recs = _sample_records("choice", 2.0, 400, 3, seed=2)
    recs += [CalibRecord("r", "choice", "other", r.logits, r.target) for r in _sample_records("choice", 0.5, 400, 3, 3)]
    cal = fit_calibration(recs, min_per_header=100)
    assert set(cal["by_header"]) == {"h", "other"}
    assert cal["by_header"]["h"] > 1.4 and cal["by_header"]["other"] < 0.7
    p = apply_tau(recs[0], 2.0)
    assert p.sum() == pytest.approx(1.0)


def test_ece_and_auroc():
    assert ece(np.array([0.9, 0.9]), np.array([1.0, 1.0])) == pytest.approx(0.1)
    assert ece(np.full(10, 0.7), np.array([1.0] * 7 + [0.0] * 3)) == pytest.approx(0.0, abs=1e-12)
    assert auroc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0.0, 0.0, 1.0, 1.0])) == pytest.approx(0.75)
    assert auroc(np.array([0.5, 0.5]), np.array([0.0, 1.0])) == pytest.approx(0.5)
    assert auroc(np.array([0.2, 0.3]), np.array([1.0, 1.0])) is None
