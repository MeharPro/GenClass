"""Decision heads over marker hidden states (docs/CONTRACT.md "B", SPEC §3.3).

    choice  z_i = MLP_c([h_Q; h_Oi; h_Q*h_Oi])            p = softmax(z / tau) over the question's options
    score   z_k = MLP_s([h_Q; h_Lk; h_Q*h_Lk])            p = softmax(z / tau) over levels
    noul    z   = MLP_n([h_Q; h_T; h_F; h_Q*h_T; h_Q*h_F])  p = sigmoid(z / tau)  (absolute, not a 2-way softmax)

A `HeadPlan` gathers every question of a batch into dense index tensors ([G, Kmax] for choice and
score, [M] for noul), so one forward computes all heads with no Python loop over questions and the
engine needs a single device sync at the end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from jev_local.engine.base import Kind
from jev_local.engine.encoder.tokenize_pack import Packed, QIndex


class MLP(nn.Sequential):
    def __init__(self, d_in: int, d: int):
        super().__init__(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, 1))
        # Small last layer: the heads start near-uniform instead of confidently random.
        nn.init.normal_(self[2].weight, std=0.02)
        nn.init.zeros_(self[2].bias)


@dataclass
class QRef:
    row: int  # batch row
    qi: QIndex
    group: int  # index within its kind's dense tensor


@dataclass
class GroupIdx:
    q: torch.Tensor  # [G] flat index (row*L + pos) of each question's [Q] marker
    items: torch.Tensor  # [G, Kmax] flat index of each item marker (padding repeats index 0)
    mask: torch.Tensor  # [G, Kmax] bool, True for real items


@dataclass
class NoulIdx:
    q: torch.Tensor  # [M]
    t: torch.Tensor  # [M]
    f: torch.Tensor  # [M]


@dataclass
class HeadPlan:
    refs: list[QRef]  # batch order: row-major, request order within a row
    choice: GroupIdx | None
    score: GroupIdx | None
    noul: NoulIdx | None

    def of_kind(self, kind: Kind) -> list[QRef]:
        return [r for r in self.refs if r.qi.kind == kind]


def _group_idx(items: list[tuple[int, QIndex]], offsets: Sequence[int], device: torch.device) -> GroupIdx | None:
    if not items:
        return None
    kmax = max(len(qi.item_pos) for _, qi in items)
    G = len(items)
    q = torch.empty(G, dtype=torch.long)
    idx = torch.zeros((G, kmax), dtype=torch.long)
    mask = torch.zeros((G, kmax), dtype=torch.bool)
    for g, (row, qi) in enumerate(items):
        q[g] = offsets[row] + qi.q_pos
        k = len(qi.item_pos)
        idx[g, :k] = torch.tensor(qi.item_pos) + offsets[row]
        mask[g, :k] = True
    return GroupIdx(q.to(device), idx.to(device), mask.to(device))


def build_plan(packs: Sequence[Packed], offsets: Sequence[int] | int, device: torch.device | str = "cpu") -> HeadPlan:
    """offsets: flat offset of each row in the hidden-state tensor (TreeBatch.row_offsets), or the
    padded row length L of a dense [B, L, d] batch."""
    device = torch.device(device)
    if isinstance(offsets, int):
        offsets = [r * offsets for r in range(len(packs))]
    refs: list[QRef] = []
    by_kind: dict[str, list[tuple[int, QIndex]]] = {"choice": [], "score": [], "noul": []}
    for row, p in enumerate(packs):
        for qi in p.q_index.values():
            lst = by_kind[qi.kind]
            refs.append(QRef(row, qi, len(lst)))
            lst.append((row, qi))
    noul = None
    if by_kind["noul"]:
        n = by_kind["noul"]
        noul = NoulIdx(
            torch.tensor([offsets[r] + qi.q_pos for r, qi in n], device=device),
            torch.tensor([offsets[r] + qi.item_pos[0] for r, qi in n], device=device),
            torch.tensor([offsets[r] + qi.item_pos[1] for r, qi in n], device=device),
        )
    return HeadPlan(
        refs,
        _group_idx(by_kind["choice"], offsets, device),
        _group_idx(by_kind["score"], offsets, device),
        noul,
    )


@dataclass
class HeadOut:
    choice: torch.Tensor | None  # [G, Kmax] logits, -inf at padding
    score: torch.Tensor | None  # [G, Kmax]
    noul: torch.Tensor | None  # [M]


class DecisionHeads(nn.Module):
    def __init__(self, d: int = 384):
        super().__init__()
        self.d = d
        # The backbone's final-norm outputs reach |x| ~ 40, so raw products h_Q*h_O would be ~1e3 and
        # random heads would start saturated. Normalising the gathered states keeps logits O(1).
        self.norm = nn.LayerNorm(d)
        self.choice_mlp = MLP(3 * d, d)
        self.score_mlp = MLP(3 * d, d)
        self.noul_mlp = MLP(5 * d, d)

    def _pairwise(self, mlp: MLP, flat: torch.Tensor, gi: GroupIdx) -> torch.Tensor:
        hq = self.norm(flat[gi.q])[:, None, :]  # [G, 1, d]
        ho = self.norm(flat[gi.items])  # [G, K, d]
        hq = hq.expand_as(ho)
        z = mlp(torch.cat([hq, ho, hq * ho], dim=-1)).squeeze(-1)
        return z.masked_fill(~gi.mask, float("-inf"))

    def forward(self, h: torch.Tensor, plan: HeadPlan) -> HeadOut:
        flat = h.reshape(-1, h.shape[-1])
        choice = self._pairwise(self.choice_mlp, flat, plan.choice) if plan.choice is not None else None
        score = self._pairwise(self.score_mlp, flat, plan.score) if plan.score is not None else None
        noul = None
        if plan.noul is not None:
            n = plan.noul
            hq, ht, hf = self.norm(flat[n.q]), self.norm(flat[n.t]), self.norm(flat[n.f])
            noul = self.noul_mlp(torch.cat([hq, ht, hf, hq * ht, hq * hf], dim=-1)).squeeze(-1)
        return HeadOut(choice, score, noul)


def temperature_vector(refs: list[QRef], kind: Kind, temps: dict, device: torch.device) -> torch.Tensor:
    """Per-question temperature (by_header overrides the per-kind value), in group order."""
    from jev_local.engine.encoder.calibrate import header_key

    by_header = temps.get("by_header", {})
    base = float(temps.get(kind, 1.0))
    vals = [float(by_header.get(header_key(r.qi.header), base)) for r in refs]
    return torch.tensor(vals, dtype=torch.float32, device=device)


def probabilities(out: HeadOut, plan: HeadPlan, temps: dict) -> dict[str, torch.Tensor | None]:
    """Apply temperatures and normalise: choice/score -> softmax per group, noul -> sigmoid. fp32."""
    res: dict[str, torch.Tensor | None] = {"choice": None, "score": None, "noul": None}
    for kind in ("choice", "score"):
        z = getattr(out, kind)
        if z is None:
            continue
        tau = temperature_vector(plan.of_kind(kind), kind, temps, z.device)
        res[kind] = torch.softmax(z.float() / tau[:, None], dim=-1)
    if out.noul is not None:
        tau = temperature_vector(plan.of_kind("noul"), "noul", temps, out.noul.device)
        res["noul"] = torch.sigmoid(out.noul.float() / tau)
    return res
