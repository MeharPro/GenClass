"""Training losses over `heads.HeadOut` logits. All are proper scoring rules (SPEC §3.5, PLAN §3.3).

Per question (defaults reproduce v1 exactly: CE + label smoothing 0.02, RPS 0.25 on score, no Brier):

    base      CE(t, p) = -sum t log p      (or, with kl=True, KL(t||p) = CE - H(t): same gradient, 0 at optimum)
    brier     sum (p - t)^2                                    choice/score/noul, weight `brier`
    spherical 1 - <p, t> / (|p| |t|)                           bounded proper score, weight `spherical`
    rps       sum_k (CDF_p - CDF_t)^2 / (K-1)                  score only, weight `rps`
    coral     mean_k BCE(P(y>k), T(y>k)) over the K-1 cuts     score only, weight `coral` (CORAL-style
              cumulative-link auxiliary computed from the same per-level logits: no extra parameters)
    noise     E_eps[base + spherical + rps on softmax(z + sigma*eps)]   pathwise logit-noise smoothing
              (Laya's RLCD objective without REINFORCE); eps ~ N(0,1) zero-mean over the options,
              sigma annealed linearly from noise_sigma[0] to noise_sigma[1] over training progress.

Per consistency pair (PLAN §2.3 rule 9; pairs come from example meta, labels not required):

    neg           (p(Q) + p(notQ) - 1)^2                            two nouls
    noul_choice   (p_noul(Q) - p_choice(yes))^2                     noul vs binary choice
    score_choice  sum_k (p_score[k] - p_choice[map[k]])^2           same ordinal question as score and choice

Total = mean over supervised questions + consistency * mean over pairs. Noul is a single logit (BCE),
treated as the two-point distribution (p, 1-p) for the spherical score.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

NEG_FILL = -1e4  # finite stand-in for -inf padding where gradients flow through cumulative sums


@dataclass
class Targets:
    """Aligned with HeadPlan groups, restricted to supervised questions (an example with no label for
    a qid simply has no row here). Index tensors instead of boolean masks: boolean indexing forces
    a device sync on MPS every step."""

    choice_idx: torch.Tensor | None = None  # [n] long: which choice groups are supervised
    choice: torch.Tensor | None = None  # [n, Kmax] target distribution (zeros at padding)
    choice_hard: torch.Tensor | None = None  # [n] bool: hard label (smoothing applies)
    score_idx: torch.Tensor | None = None
    score: torch.Tensor | None = None
    score_hard: torch.Tensor | None = None
    noul_idx: torch.Tensor | None = None  # [m] long
    noul: torch.Tensor | None = None  # [m] p(true)
    # consistency pairs (indices into the plan's per-kind groups; unsupervised questions allowed)
    neg_a: torch.Tensor | None = None  # [P] noul group
    neg_b: torch.Tensor | None = None  # [P] noul group
    nc_noul: torch.Tensor | None = None  # [P] noul group
    nc_choice: torch.Tensor | None = None  # [P] choice group
    nc_yes: torch.Tensor | None = None  # [P] option index of "yes" within the choice group
    sc_score: torch.Tensor | None = None  # [P] score group
    sc_choice: torch.Tensor | None = None  # [P] choice group
    sc_map: torch.Tensor | None = None  # [P, Kmax] choice option index per score level
    sc_mask: torch.Tensor | None = None  # [P, Kmax] bool, True for real levels

    @property
    def n_pairs(self) -> int:
        return sum(int(x.shape[0]) for x in (self.neg_a, self.nc_noul, self.sc_score) if x is not None)


def smooth(t: torch.Tensor, mask: torch.Tensor, hard: torch.Tensor, eps: float) -> torch.Tensor:
    if eps <= 0:
        return t
    k = mask.sum(-1, keepdim=True).clamp(min=1).to(t.dtype)
    sm = (1 - eps) * t + eps * mask.to(t.dtype) / k
    return torch.where(hard[:, None], sm, t)


def softmax_ce(z: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-group cross-entropy -sum t*log softmax(z); z is -inf at padding."""
    logp = torch.log_softmax(z.float(), dim=-1)
    return -(torch.where(mask, logp, torch.zeros_like(logp)) * t).sum(-1)


def entropy(t: torch.Tensor) -> torch.Tensor:
    """H(t) = -sum t log t with 0 log 0 = 0 (targets carry no gradient)."""
    return -(torch.where(t > 0, t * torch.log(t.clamp(min=1e-30)), torch.zeros_like(t))).sum(-1)


def softmax_brier(z: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(z.float(), dim=-1)
    return (torch.where(mask, (p - t) ** 2, torch.zeros_like(p))).sum(-1)


def rps(z: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Ranked probability score, normalised by K-1 so it lies in [0, 1]."""
    p = torch.where(mask, torch.softmax(z.float(), dim=-1), torch.zeros_like(t))
    d = (p.cumsum(-1) - t.cumsum(-1)) ** 2
    k = mask.sum(-1).clamp(min=2).to(p.dtype)
    return torch.where(mask, d, torch.zeros_like(d)).sum(-1) / (k - 1)


def spherical(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """1 - cosine(p, t): the spherical score sum t*p/|p| normalised by |t|, so it is 0 at p = t."""
    num = (p * t).sum(-1)
    den = p.norm(dim=-1).clamp(min=1e-12) * t.norm(dim=-1).clamp(min=1e-12)
    return 1.0 - num / den


def coral(z: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean BCE over the K-1 cumulative cuts P(y > k) of softmax(z) against T(y > k)."""
    zf = z.float().masked_fill(~mask, NEG_FILL)
    lse = torch.logsumexp(zf, dim=-1, keepdim=True)
    log_le = torch.logcumsumexp(zf, dim=-1) - lse  # log P(y <= k)
    log_gt = torch.logcumsumexp(zf.flip(-1), dim=-1).flip(-1)[:, 1:] - lse  # log P(y > k), k = 0..K-2
    log_le = log_le[:, :-1]
    t_gt = (t.sum(-1, keepdim=True) - t.cumsum(-1))[:, :-1].clamp(0, 1)
    valid = mask[:, 1:]  # cut k exists iff level k+1 exists
    bce = -(t_gt * log_gt + (1 - t_gt) * log_le)
    n = valid.sum(-1).clamp(min=1).to(bce.dtype)
    return torch.where(valid, bce, torch.zeros_like(bce)).sum(-1) / n


def noul_bce(z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(z.float(), p.float(), reduction="none")


def noul_entropy(p: torch.Tensor) -> torch.Tensor:
    return entropy(torch.stack([p, 1 - p], dim=-1))


def noul_brier(z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(z.float()) - p) ** 2


def _noul_pair(z: torch.Tensor) -> torch.Tensor:
    q = torch.sigmoid(z.float())
    return torch.stack([q, 1 - q], dim=-1)


@dataclass
class LossConfig:
    smoothing: float = 0.02
    brier: float = 0.0
    rps: float = 0.25
    kl: bool = False  # base term reported as KL(t||p) instead of CE (identical gradients)
    spherical: float = 0.0
    coral: float = 0.0
    noise_sigma: tuple[float, float] = (0.0, 0.0)  # (start, end) of the logit-noise sigma, linear in progress
    noise_samples: int = 1
    noise_weight: float = 1.0
    consistency: float = 0.0

    @classmethod
    def v2(cls, **over) -> "LossConfig":
        """PLAN §3.3 recipe: KL + 0.5 spherical + 1.0 RPS + CORAL aux + 0.1 consistency (noise is an ablation)."""
        base = dict(smoothing=0.02, rps=1.0, kl=True, spherical=0.5, coral=0.25, consistency=0.1)
        base.update({k: v for k, v in over.items() if v is not None})
        return cls(**base)

    def sigma(self, progress: float) -> float:
        s0, s1 = self.noise_sigma
        return float(s0 + (s1 - s0) * min(max(progress, 0.0), 1.0))


def _dist_terms(z: torch.Tensor, t: torch.Tensor, mask: torch.Tensor, kind: str, cfg: LossConfig,
                ent: torch.Tensor | None) -> torch.Tensor:
    """base (+ spherical + rps) for one set of logits; reused for the noisy copies."""
    loss = softmax_ce(z, t, mask)
    if ent is not None:
        loss = loss - ent
    if cfg.spherical > 0:
        p = torch.where(mask, torch.softmax(z.float(), dim=-1), torch.zeros_like(t))
        loss = loss + cfg.spherical * spherical(p, t)
    if kind == "score" and cfg.rps > 0:
        loss = loss + cfg.rps * rps(z, t, mask)
    return loss


def _noul_terms(z: torch.Tensor, p: torch.Tensor, cfg: LossConfig, ent: torch.Tensor | None) -> torch.Tensor:
    loss = noul_bce(z, p)
    if ent is not None:
        loss = loss - ent
    if cfg.spherical > 0:
        loss = loss + cfg.spherical * spherical(_noul_pair(z), torch.stack([p, 1 - p], dim=-1).float())
    return loss


def _noisy(z: torch.Tensor, mask: torch.Tensor | None, sigma: float) -> torch.Tensor:
    eps = torch.randn(z.shape, device=z.device, dtype=torch.float32) * sigma
    if mask is not None:  # zero-mean across the real options (softmax is shift-invariant anyway)
        m = mask.to(eps.dtype)
        eps = (eps - (eps * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp(min=1)) * m
    return z.float() + eps


def consistency_terms(out, tg: Targets) -> torch.Tensor | None:
    """Per-pair consistency penalties (concatenated), or None when the batch has no pairs."""
    terms = []
    if tg.neg_a is not None and out.noul is not None:
        pa = torch.sigmoid(out.noul.float().index_select(0, tg.neg_a))
        pb = torch.sigmoid(out.noul.float().index_select(0, tg.neg_b))
        terms.append((pa + pb - 1) ** 2)
    if tg.nc_noul is not None and out.noul is not None and out.choice is not None:
        pn = torch.sigmoid(out.noul.float().index_select(0, tg.nc_noul))
        pc = torch.softmax(out.choice.float().index_select(0, tg.nc_choice), dim=-1)
        py = pc.gather(1, tg.nc_yes[:, None])[:, 0]
        terms.append((pn - py) ** 2)
    if tg.sc_score is not None and out.score is not None and out.choice is not None:
        ps = torch.softmax(out.score.float().index_select(0, tg.sc_score), dim=-1)
        pc = torch.softmax(out.choice.float().index_select(0, tg.sc_choice), dim=-1)
        k = tg.sc_map.shape[1]
        ps = ps[:, :k] if ps.shape[1] >= k else F.pad(ps, (0, k - ps.shape[1]))
        pcm = pc.gather(1, tg.sc_map)
        d = torch.where(tg.sc_mask, (ps - pcm) ** 2, torch.zeros_like(ps))
        terms.append(d.sum(-1))
    return torch.cat(terms) if terms else None


def compute_loss(out, tg: Targets, cfg: LossConfig, progress: float = 0.0
                 ) -> tuple[torch.Tensor | None, dict[str, tuple[torch.Tensor, int]]]:
    """-> (loss or None if nothing is supervised/paired, parts). parts["choice"|"score"|"noul"] are
    (sum, count) of the per-question losses; extra diagnostics use "x_"-prefixed keys and are not part of
    the per-question average the trainer logs as `loss`."""
    terms: list[torch.Tensor] = []
    parts: dict[str, tuple[torch.Tensor, int]] = {}
    sigma = cfg.sigma(progress)
    noise_on = sigma > 0 and cfg.noise_weight > 0 and cfg.noise_samples > 0
    extra: dict[str, list[torch.Tensor]] = {}
    for kind in ("choice", "score"):
        z = getattr(out, kind)
        idx = getattr(tg, f"{kind}_idx")
        if z is None or idx is None:
            continue
        z, t, hard = z.index_select(0, idx), getattr(tg, kind), getattr(tg, f"{kind}_hard")
        mask = torch.isfinite(z)
        t = smooth(t, mask, hard, cfg.smoothing)
        ent = entropy(t) if cfg.kl else None
        if ent is None and cfg.spherical == 0:  # v1 path, bit-identical: CE (+ Brier) (+ RPS)
            loss = softmax_ce(z, t, mask)
            if cfg.brier > 0:
                loss = loss + cfg.brier * softmax_brier(z, t, mask)
            if kind == "score" and cfg.rps > 0:
                loss = loss + cfg.rps * rps(z, t, mask)
        else:
            loss = _dist_terms(z, t, mask, kind, cfg, ent)
            if cfg.brier > 0:
                loss = loss + cfg.brier * softmax_brier(z, t, mask)
        if kind == "score" and cfg.coral > 0:
            loss = loss + cfg.coral * coral(z, t, mask)
        if noise_on:
            zf = z.float().masked_fill(~mask, NEG_FILL)
            nz = sum(_dist_terms(_noisy(zf, mask, sigma).masked_fill(~mask, float("-inf")), t, mask, kind, cfg, ent)
                     for _ in range(cfg.noise_samples)) / cfg.noise_samples
            extra.setdefault("x_noise", []).append(nz.detach())
            loss = loss + cfg.noise_weight * nz
        terms.append(loss)
        parts[kind] = (loss.detach().sum(), int(loss.shape[0]))
    if out.noul is not None and tg.noul_idx is not None:
        z, p = out.noul.index_select(0, tg.noul_idx), tg.noul
        ent = noul_entropy(p.float()) if cfg.kl else None
        loss = noul_bce(z, p) if (ent is None and cfg.spherical == 0) else _noul_terms(z, p, cfg, ent)
        if cfg.brier > 0:
            loss = loss + cfg.brier * noul_brier(z, p)
        if noise_on:
            nz = sum(_noul_terms(_noisy(z, None, sigma), p, cfg, ent) for _ in range(cfg.noise_samples)) / cfg.noise_samples
            extra.setdefault("x_noise", []).append(nz.detach())
            loss = loss + cfg.noise_weight * nz
        terms.append(loss)
        parts["noul"] = (loss.detach().sum(), int(loss.shape[0]))
    total = torch.cat(terms).mean() if terms else None
    if cfg.consistency > 0 and tg.n_pairs:
        c = consistency_terms(out, tg)
        if c is not None and c.numel():
            parts["x_cons"] = (c.detach().sum(), int(c.shape[0]))
            total = cfg.consistency * c.mean() + (total if total is not None else 0.0)
    for k, v in extra.items():
        x = torch.cat(v)
        parts[k] = (x.sum(), int(x.shape[0]))
    if total is None:
        return None, {}
    return total, parts
