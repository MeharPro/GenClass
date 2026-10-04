"""losses.py v2 terms: the v1 default is bit-identical to the old implementation; KL/spherical/CORAL are
proper (zero gradient at p = t); logit noise and consistency pairs behave; padding never makes NaNs."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from jev_local.train.losses import LossConfig, Targets, compute_loss, consistency_terms, coral, entropy, spherical


class Out:
    def __init__(self, choice=None, score=None, noul=None):
        self.choice, self.score, self.noul = choice, score, noul


def _v1_reference(out, tg, smoothing=0.02, brier=0.0, rps_w=0.25):
    """Verbatim v1 compute_loss (before the v2 edit), for the reproducibility check."""
    def smooth(t, mask, hard, eps):
        k = mask.sum(-1, keepdim=True).clamp(min=1).to(t.dtype)
        sm = (1 - eps) * t + eps * mask.to(t.dtype) / k
        return torch.where(hard[:, None], sm, t)

    def ce(z, t, mask):
        logp = torch.log_softmax(z.float(), dim=-1)
        return -(torch.where(mask, logp, torch.zeros_like(logp)) * t).sum(-1)

    def rps(z, t, mask):
        p = torch.where(mask, torch.softmax(z.float(), dim=-1), torch.zeros_like(t))
        d = (p.cumsum(-1) - t.cumsum(-1)) ** 2
        k = mask.sum(-1).clamp(min=2).to(p.dtype)
        return torch.where(mask, d, torch.zeros_like(d)).sum(-1) / (k - 1)

    terms = []
    for kind in ("choice", "score"):
        z = getattr(out, kind)
        idx = getattr(tg, f"{kind}_idx")
        if z is None or idx is None:
            continue
        z, t, hard = z.index_select(0, idx), getattr(tg, kind), getattr(tg, f"{kind}_hard")
        mask = torch.isfinite(z)
        t = smooth(t, mask, hard, smoothing)
        loss = ce(z, t, mask)
        if kind == "score" and rps_w > 0:
            loss = loss + rps_w * rps(z, t, mask)
        terms.append(loss)
    if out.noul is not None and tg.noul_idx is not None:
        terms.append(F.binary_cross_entropy_with_logits(out.noul.index_select(0, tg.noul_idx).float(), tg.noul.float(),
                                                        reduction="none"))
    return torch.cat(terms).mean()


def _batch(seed=0, G=5, K=7):
    g = torch.Generator().manual_seed(seed)
    zc = torch.randn(G, K, generator=g)
    zc[1, 4:] = float("-inf")
    zc[3, 2:] = float("-inf")
    zs = torch.randn(3, 5, generator=g)
    zs[0, 3:] = float("-inf")
    zn = torch.randn(4, generator=g)
    tc = torch.zeros(4, K)
    tc[0, 2] = 1
    tc[1, :4] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    tc[2, 6] = 1
    tc[3, :2] = torch.tensor([0.5, 0.5])
    ts = torch.zeros(2, 5)
    ts[0, :3] = torch.tensor([0.2, 0.5, 0.3])
    ts[1, 4] = 1
    tg = Targets(choice_idx=torch.tensor([0, 1, 2, 3]), choice=tc, choice_hard=torch.tensor([True, False, True, False]),
                 score_idx=torch.tensor([0, 2]), score=ts, score_hard=torch.tensor([False, True]),
                 noul_idx=torch.tensor([0, 2, 3]), noul=torch.tensor([1.0, 0.3, 0.0]))
    return Out(zc.requires_grad_(), zs.requires_grad_(), zn.requires_grad_()), tg


def test_v1_default_is_bit_identical():
    out, tg = _batch()
    loss, parts = compute_loss(out, tg, LossConfig())
    ref = _v1_reference(out, tg)
    assert torch.equal(loss, ref)
    assert set(parts) == {"choice", "score", "noul"}
    g1 = torch.autograd.grad(loss, [out.choice, out.score, out.noul])
    g2 = torch.autograd.grad(ref, [out.choice, out.score, out.noul])
    for a, b in zip(g1, g2):
        assert torch.equal(torch.nan_to_num(a), torch.nan_to_num(b))


def test_kl_same_gradient_zero_at_optimum():
    out, tg = _batch(1)
    a, _ = compute_loss(out, tg, LossConfig(rps=0.0))
    b, _ = compute_loss(out, tg, LossConfig(rps=0.0, kl=True))
    ga = torch.autograd.grad(a, out.choice)[0]
    gb = torch.autograd.grad(b, out.choice)[0]
    assert torch.allclose(torch.nan_to_num(ga), torch.nan_to_num(gb), atol=1e-6)
    assert b < a  # minus the target entropy
    t = torch.tensor([[0.1, 0.2, 0.7]])
    z = torch.log(t).requires_grad_()
    tg2 = Targets(choice_idx=torch.tensor([0]), choice=t, choice_hard=torch.tensor([False]))
    loss, _ = compute_loss(Out(choice=z), tg2, LossConfig(kl=True, smoothing=0.0))
    assert loss.abs() < 1e-6


@pytest.mark.parametrize("term", ["spherical", "coral"])
def test_proper_scores_have_zero_gradient_at_target(term):
    t = torch.tensor([[0.1, 0.25, 0.4, 0.25], [0.6, 0.2, 0.15, 0.05]])
    z = torch.log(t).clone().requires_grad_()
    mask = torch.ones_like(t, dtype=torch.bool)
    if term == "spherical":
        val = spherical(torch.softmax(z, -1), t)
        assert val.abs().max() < 1e-6
    else:
        val = coral(z, t, mask)
    (g,) = torch.autograd.grad(val.sum(), z)
    assert g.abs().max() < 1e-5, g
    z2 = (z.detach() + torch.tensor([0.5, -0.3, 0.0, 0.2])).requires_grad_()
    v2 = spherical(torch.softmax(z2, -1), t) if term == "spherical" else coral(z2, t, mask)
    assert (v2 > val.detach()).all()


def test_coral_handles_padding_without_nan():
    z = torch.tensor([[0.3, -0.2, float("-inf"), float("-inf")], [1.0, 0.0, -1.0, 2.0]], requires_grad=True)
    t = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    v = coral(z, t, torch.isfinite(z))
    (g,) = torch.autograd.grad(v.sum(), z)
    assert torch.isfinite(v).all() and torch.isfinite(g[torch.isfinite(z)]).all()
    assert (g[~torch.isfinite(z)] == 0).all()


def test_v2_preset_finite_and_parts():
    out, tg = _batch(2)
    cfg = LossConfig.v2()
    assert (cfg.kl, cfg.spherical, cfg.rps, cfg.consistency) == (True, 0.5, 1.0, 0.1)
    loss, parts = compute_loss(out, tg, cfg)
    loss.backward()
    assert torch.isfinite(loss)
    for z in (out.choice, out.score, out.noul):
        assert torch.isfinite(z.grad[torch.isfinite(z)]).all()
    assert set(parts) == {"choice", "score", "noul"}


def test_logit_noise_schedule_and_term():
    out, tg = _batch(3)
    cfg = LossConfig(noise_sigma=(0.4, 0.1), noise_samples=4)
    assert cfg.sigma(0.0) == pytest.approx(0.4) and cfg.sigma(1.0) == pytest.approx(0.1)
    torch.manual_seed(0)
    loss, parts = compute_loss(out, tg, cfg, progress=0.5)
    base, _ = compute_loss(out, tg, LossConfig())
    assert "x_noise" in parts and loss > base
    loss.backward()
    assert torch.isfinite(out.choice.grad[torch.isfinite(out.choice)]).all()
    # sigma 0 -> no noise term at all
    _, p0 = compute_loss(out, tg, LossConfig(noise_sigma=(0.0, 0.0)))
    assert "x_noise" not in p0


def test_consistency_pairs():
    noul = torch.tensor([0.0, 0.0, 2.0, torch.logit(torch.tensor(0.3)).item()], requires_grad=True)
    choice = torch.tensor([[0.0, 0.0, float("-inf")], [2.0, 0.0, float("-inf")]], requires_grad=True)
    score = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)
    out = Out(choice=choice, score=score, noul=noul)
    tg = Targets(neg_a=torch.tensor([0, 2]), neg_b=torch.tensor([1, 3]),
                 nc_noul=torch.tensor([0]), nc_choice=torch.tensor([0]), nc_yes=torch.tensor([1]),
                 sc_score=torch.tensor([0]), sc_choice=torch.tensor([1]), sc_map=torch.tensor([[1, 0, 0]]),
                 sc_mask=torch.tensor([[True, True, False]]))
    c = consistency_terms(out, tg)
    cd = c.detach()
    p2 = torch.sigmoid(torch.tensor(2.0))
    assert float(cd[0]) == pytest.approx(0.0, abs=1e-7)  # 0.5 + 0.5 - 1
    assert float(cd[1]) == pytest.approx(float((p2 + 0.3 - 1) ** 2), abs=1e-6)
    assert float(cd[2]) == pytest.approx(0.0, abs=1e-7)  # p_noul 0.5 vs p(yes) 0.5
    ps = torch.softmax(score[0], 0)
    pc = torch.softmax(choice[1, :2], 0)
    want = (ps[0] - pc[1]) ** 2 + (ps[1] - pc[0]) ** 2
    assert float(cd[3]) == pytest.approx(float(want.detach()), abs=1e-6)
    # consistency alone (no supervised question) still produces a loss
    loss, parts = compute_loss(out, tg, LossConfig(consistency=0.1))
    assert float(loss) == pytest.approx(0.1 * float(cd.mean()), rel=1e-6) and parts["x_cons"][1] == 4
    loss.backward()
    assert noul.grad.abs().sum() > 0


def test_entropy_zero_safe():
    t = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    assert torch.allclose(entropy(t), torch.tensor([0.0, 0.6931472]), atol=1e-6)
