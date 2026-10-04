"""`FastEngine`: the trained ettin encoder + decision heads behind the `Engine` protocol.

Checkpoint directory layout (docs/CONTRACT.md "B"):

    backbone/            HF save_pretrained of the ModernBertModel + tokenizer (with marker tokens)
    heads.safetensors    DecisionHeads state dict
    calibration.json     {"noul": tau, "choice": tau, "score": tau, "by_header": {sha1(header)[:12]: tau}}
    meta.json            name, base model, markers, max_len, buckets, hidden size, training info

`write_checkpoint` is used both by the trainer and by `train --init-only` (random heads), so the
server can load a fast engine before any training has happened.

Long requests and many options (PLAN §3.2 items 2 and 4)
- `max_tokens` (default 2048, up to 8192: ModernBERT/ettin is 8k-native) bounds POSITIONS, i.e. what one
  question branch sees: state + header + its longest option. Options never attend to each other and
  restart their positions after the header, so the total request may be much longer than max_tokens.
- Exact option batching: when the whole request exceeds `max_flat_tokens` (one forward's token budget),
  questions are split into fragments (a subset of their options, header repeated) and fragments are packed
  into passes that each repeat the state. Raw logits of all fragments of a question are concatenated and
  calibrated with ONE softmax over all K options (temperature by K-bucket of the full K), so the answer is
  identical to a single pass. Choice questions may have up to 255 options.
- `evaluate_logits` returns the raw (tau = 1) logits per question for benchmark temperature refits.

Benchmax (suite-reproduction-specs.md §1.2)
- `max_tokens=None` reads the checkpoint's `meta.json["max_len"]` (8192 for the v2 checkpoints), so 8k inputs work
  without a flag; `MAX_LEN` (2048) stays the fallback for v1 checkpoints without the key.
- `calibration=` replaces the checkpoint's calibration.json for this run (W9: the calibration file is chosen by the
  run config, never by inspecting requests); `calibration_info` records its source and sha256 for run.json.
- `positions_needed(req)` is what the router compares with `max_tokens`: state + the longest question branch, not
  the whole request, because a 255-option question runs as several exact passes.
- `fit_state(state, questions)` is overflow="truncate" (W5): the longest state field is cut at its end on token
  boundaries until the request fits; questions are never touched; the number of tokens cut is returned.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.checkpoint import checkpoint

from jev_local.engine.base import EngineError, EngineResult, RawDist
from jev_local.engine.encoder.calibrate import (DEFAULT_CALIBRATION, calibrate_logits, describe_calibration, header_key,
                                                load_calibration)
from jev_local.engine.encoder.heads import DecisionHeads, build_plan
from jev_local.engine.encoder.model import BASE_MODEL, MaskedEncoder, init_marker_embeddings
from jev_local.engine.encoder.tokenize_pack import (BUCKETS, MARKERS, MAX_LEN, Packer, TreeBatch, add_marker_tokens,
                                                    collate_tree)
from jev_local.schema import Entry, Question, SystemOneRequest
from jev_local.serialize import MAX_ARRAY_SEGMENTS, QBlock, Segment, question_block, state_segments

ENGINE_NAME = "jev-local-fast-0.1.0"
MAX_CHOICE_OPTIONS = 255
MAX_POSITIONS = 8192  # ettin / ModernBERT context
ENGINE_BUCKETS = BUCKETS + (3072, 4096, 6144, 8192, 12288, 16384)  # flat-length buckets (stable MPS shapes)
# Seed words for the new marker embeddings (see model.init_marker_embeddings).
MARKER_SEEDS = {"[Q]": " question", "[O]": " option", "[L]": " level", "[T]": " true", "[F]": " false"}


def pick_device(device: str | None) -> torch.device:
    if device in (None, "auto"):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    return torch.device(device)


def load_tokenizer(path: str | Path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(path), clean_up_tokenization_spaces=False, **_local_kw(path))
    add_marker_tokens(tok)
    return tok


def _local_kw(path: str | Path) -> dict:
    # A bare repo id resolves from the HF cache only: training/serving never downloads (CONTRACT).
    return {} if Path(path).exists() else {"local_files_only": True}


class _SlotGather(torch.autograd.Function):
    """flat [N, F] -> slots [T, F] (TreeBatch slot order). Every real flat token sits in exactly one slot and
    padding slots never receive gradient (masked keys, discarded queries), so the backward is a plain gather
    by the inverse permutation instead of a serial scatter-add."""

    @staticmethod
    def forward(ctx, x, src0, inv, n_real):  # noqa: ANN001
        ctx.save_for_backward(inv)
        ctx.n_real = n_real
        return x.index_select(0, src0)

    @staticmethod
    def backward(ctx, g):  # noqa: ANN001
        (inv,) = ctx.saved_tensors
        gx = g.index_select(0, inv)
        gx[ctx.n_real :] = 0  # flat padding tokens feed no slot
        return gx, None, None, None


class _FlatGather(torch.autograd.Function):
    """slots [T, F] -> flat [N, F] (MaskedEncoder's `slots[tb.inv]`); backward gathers by slot source."""

    @staticmethod
    def forward(ctx, o, inv, src0, pad_slots):  # noqa: ANN001
        ctx.save_for_backward(src0, pad_slots)
        return o.index_select(0, inv)

    @staticmethod
    def backward(ctx, g):  # noqa: ANN001
        src0, pad = ctx.saved_tensors
        gs = g.index_select(0, src0)
        gs.masked_fill_(pad[:, None], 0)
        return gs, None, None, None


def _rope_halves(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """HF apply_rotary_pos_emb (x*cos + rotate_half(x)*sin, in fp32) written on the two halves; cos/sin are
    [T, D] with equal halves (ModernBERT duplicates the frequencies)."""
    dt = x.dtype
    xf = x.float()
    half = xf.shape[-1] // 2
    x1, x2 = xf[..., :half], xf[..., half:]
    c, s = cos[..., :half], sin[..., :half]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(dt)


class BandedEncoder(MaskedEncoder):
    """Faster, exact tree-layout forward (same masks and results as MaskedEncoder.forward_tree).

    1. Slot layout: q/k/v are moved into [trunk | headers | option chunks] slot order with ONE gather per layer
       whose backward is a gather by the inverse permutation (_SlotGather/_FlatGather), instead of nine
       advanced-index gathers whose backward is a serial index_put(accumulate) over zero-filled tensors.
    2. Banded local layers: in sliding-window layers a token only sees keys within +-window positions, so
       the trunk runs as (2w)-slot query blocks over (4w)-slot key windows (O(S) instead of O(S^2)), and
       headers/options use only the last `window` state slots of their row instead of the whole state.
    Masks are derived from the TreeBatch (validity, positions, the reference item mask), so every real token
    sees exactly the reference key set; padding query rows attend to themselves or key 0 so no softmax row
    is empty. Verified equal in float64 (outputs and gradients) by tests/test_train_v2_banded.py.
    JEV_ENCODER=reference selects the reference MaskedEncoder.
    """

    def _layout(self, tb: TreeBatch) -> dict:
        cache = getattr(tb, "_slot_cache", None)
        if cache is not None and cache.get("window") == self.window:
            return cache
        W = self.window
        N = tb.input_ids.shape[0]
        dev = tb.trunk_idx.device
        B, Sp = tb.trunk_idx.shape
        NH, Hp = tb.head_idx.shape
        NI, C = tb.item_idx.shape
        src = torch.cat([tb.trunk_idx.reshape(-1), tb.head_idx.reshape(-1), tb.item_idx.reshape(-1)])
        pad_slots = src >= N
        src0 = torch.where(pad_slots, torch.zeros_like(src), src)
        # ---- banded trunk (slot index == position inside the trunk)
        BQ = 2 * W
        nb = -(-Sp // BQ)
        t_valid = tb.trunk_idx < N
        n_state = t_valid.sum(1)
        ext_valid = torch.zeros((B, nb * BQ + 2 * W), dtype=torch.bool, device=dev)
        ext_valid[:, W : W + Sp] = t_valid
        a = torch.arange(BQ, device=dev)
        j = torch.arange(BQ + 2 * W, device=dev)
        blk = torch.arange(nb, device=dev)
        q_ext = W + blk[:, None] * BQ + a[None, :]
        k_ext = blk[:, None] * BQ + j[None, :]
        qv, kv = ext_valid[:, q_ext], ext_valid[:, k_ext]
        band = (q_ext[:, :, None] - k_ext[:, None, :]).abs() <= W
        diag = k_ext[:, None, :] == q_ext[:, :, None]
        m = (qv[..., None] & kv[:, :, None, :] & band[None]) | (~qv[..., None] & diag[None])
        t_mask = m.reshape(B * nb, 1, BQ, BQ + 2 * W)
        # ---- header / option keys in local layers: last W state slots of the row
        pos_pad = torch.cat([tb.position_ids, tb.position_ids.new_zeros(1)])
        h_valid, i_valid = tb.head_idx < N, tb.item_idx < N
        h_pos, i_pos = pos_pad[tb.head_idx], pos_pad[tb.item_idx]

        def window(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """(index into the trunk part of the slot tensor [B*Sp], valid, position) of the last W state slots."""
            slot = n_state[rows][:, None] - W + torch.arange(W, device=dev)[None, :]
            return (rows[:, None] * Sp + slot.clamp(min=0)).reshape(-1), slot >= 0, slot

        def near(qp: torch.Tensor, kp: torch.Tensor) -> torch.Tensor:
            return (qp[:, :, None] - kp[:, None, :]).abs() <= W

        hw_slot, hw_ok, hw_pos = window(tb.head_row)
        eye_h = torch.eye(Hp, dtype=torch.bool, device=dev)[None]
        m_w = hw_ok[:, None, :] & near(h_pos, hw_pos)
        m_h = h_valid[:, None, :] & near(h_pos, h_pos)
        m = torch.cat([m_w, m_h], 2) & h_valid[:, :, None]
        m = m | torch.cat([torch.zeros_like(m_w), eye_h.expand_as(m_h)], 2) & ~h_valid[:, :, None]
        h_mask = m[:, None]
        iw_slot, iw_ok, iw_pos = window(tb.item_row)
        own = tb.masks["item"]["full_attention"][:, 0, :, -C:] & i_valid[:, :, None]  # same option, valid key
        eye_c = torch.eye(C, dtype=torch.bool, device=dev)[None]
        m_w = iw_ok[:, None, :] & near(i_pos, iw_pos)
        m_h = h_valid[tb.item_head][:, None, :] & near(i_pos, h_pos[tb.item_head])
        m_o = own & near(i_pos, i_pos)
        m = torch.cat([m_w, m_h, m_o], 2) & i_valid[:, :, None]
        m = m | torch.cat([torch.zeros_like(m_w), torch.zeros_like(m_h), eye_c.expand_as(m_o)], 2) & ~i_valid[:, :, None]
        cache = {"window": W, "src0": src0, "pad_slots": pad_slots, "shape": (B, Sp, NH, Hp, NI, C), "nb": nb, "BQ": BQ,
                 "t_mask": t_mask, "h_win": hw_slot, "h_mask": h_mask, "i_win": iw_slot, "i_mask": m[:, None]}
        tb._slot_cache = cache  # type: ignore[attr-defined]
        return cache

    def forward_tree(self, tb: TreeBatch) -> torch.Tensor:
        bb = self.backbone
        L = self._layout(tb)
        h = bb.embeddings(input_ids=tb.input_ids[None])[0]  # [N, d]
        rope = {}
        for t in set(self.layer_types):
            cos, sin = bb.rotary_emb(h[None], tb.position_ids[None], t)
            rope[t] = (cos[0].float().index_select(0, L["src0"])[:, None, None, :],
                       sin[0].float().index_select(0, L["src0"])[:, None, None, :])
        for layer, t in zip(bb.layers, self.layer_types):
            cs, sn = rope[t]
            if self.grad_ckpt and self.training:
                h = checkpoint(self._slot_layer, layer, h, tb, t, cs, sn, use_reentrant=False)
            else:
                h = self._slot_layer(layer, h, tb, t, cs, sn)
        return bb.final_norm(h)

    def _slot_layer(self, layer, h: torch.Tensor, tb: TreeBatch, t: str, cos: torch.Tensor,
                    sin: torch.Tensor) -> torch.Tensor:
        L = self._layout(tb)
        B, Sp, NH, Hp, NI, C = L["shape"]
        H, D = self.n_heads, self.head_dim
        W = self.window
        qkv = layer.attn.Wqkv(layer.attn_norm(h))  # [N, 3HD]
        s = _SlotGather.apply(qkv, L["src0"], tb.inv, tb.n_tokens).view(-1, 3, H, D)  # [T, 3, H, D]
        qk = _rope_halves(s[:, :2], cos, sin)  # [T, 2, H, D]
        q, k, v = qk[:, 0], qk[:, 1], s[:, 2]
        BS, HS = B * Sp, NH * Hp

        def part(x: torch.Tensor, a: int, b: int, *shape: int) -> torch.Tensor:
            return x[a:b].reshape(*shape, H, D).transpose(-3, -2)  # [..., H, L, D]

        qt, kt, vt = (part(x, 0, BS, B, Sp) for x in (q, k, v))
        qh, kh, vh = (part(x, BS, BS + HS, NH, Hp) for x in (q, k, v))
        qi, ki, vi = (part(x, BS + HS, x.shape[0], NI, C) for x in (q, k, v))
        scale = D**-0.5
        if t == "sliding_attention":
            nb, BQ = L["nb"], L["BQ"]
            extra = nb * BQ - Sp
            qb = F.pad(qt, (0, 0, 0, extra)).reshape(B, H, nb, BQ, D).transpose(1, 2).reshape(B * nb, H, BQ, D)

            def windows(x: torch.Tensor) -> torch.Tensor:
                xp = F.pad(x, (0, 0, W, extra + W))  # [B, H, nb*BQ + 2W, D]
                return xp.unfold(2, BQ + 2 * W, BQ).permute(0, 2, 1, 4, 3).reshape(B * nb, H, BQ + 2 * W, D)

            ob = F.scaled_dot_product_attention(qb, windows(kt), windows(vt), attn_mask=L["t_mask"], scale=scale)
            ot = ob.reshape(B, nb, H, BQ, D).transpose(1, 2).reshape(B, H, nb * BQ, D)[:, :, :Sp]

            def win(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
                # x: slot tensor [T, H, D]; index_select (backward = vectorised index_add, not serial index_put)
                return x.index_select(0, idx).view(-1, W, H, D).transpose(1, 2)  # [n, H, W, D]

            oh = F.scaled_dot_product_attention(qh, torch.cat([win(k, L["h_win"]), kh], 2),
                                                torch.cat([win(v, L["h_win"]), vh], 2), attn_mask=L["h_mask"], scale=scale)
            oi = F.scaled_dot_product_attention(
                qi, torch.cat([win(k, L["i_win"]), kh.index_select(0, tb.item_head), ki], 2),
                torch.cat([win(v, L["i_win"]), vh.index_select(0, tb.item_head), vi], 2), attn_mask=L["i_mask"], scale=scale)
        else:
            ot = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=tb.masks["trunk"][t], scale=scale)
            oh = F.scaled_dot_product_attention(
                qh, torch.cat([kt.index_select(0, tb.head_row), kh], 2), torch.cat([vt.index_select(0, tb.head_row), vh], 2),
                attn_mask=tb.masks["head"][t], scale=scale)
            oi = F.scaled_dot_product_attention(
                qi, torch.cat([kt.index_select(0, tb.item_row), kh.index_select(0, tb.item_head), ki], 2),
                torch.cat([vt.index_select(0, tb.item_row), vh.index_select(0, tb.item_head), vi], 2),
                attn_mask=tb.masks["item"][t], scale=scale)
        o = torch.cat([x.transpose(-3, -2).reshape(-1, H * D) for x in (ot, oh, oi)])
        h = h + layer.attn.Wo(_FlatGather.apply(o, tb.inv, L["src0"], L["pad_slots"]))
        return h + layer.mlp(layer.mlp_norm(h))


ENCODERS = {"reference": MaskedEncoder, "banded": BandedEncoder}


def encoder_class(name: str = "reference") -> type[MaskedEncoder]:
    """`name` ("reference" | "banded"); the JEV_ENCODER environment variable overrides it (A/B checks).
    Low-level constructors (init_model, load_checkpoint) default to the v1 reference; FastEngine, the trainer
    and eval pick "banded" (identical results up to fp32 rounding, ~2x faster training on CPU)."""
    name = os.environ.get("JEV_ENCODER") or name
    if name not in ENCODERS:
        raise ValueError(f"unknown encoder {name!r} (choose {sorted(ENCODERS)})")
    return ENCODERS[name]


def init_model(base: str = BASE_MODEL, seed: int = 0, encoder: str = "reference"
               ) -> tuple[MaskedEncoder, DecisionHeads, object]:
    """Pretrained backbone + marker tokens + randomly initialised heads."""
    tok = load_tokenizer(base)
    enc = encoder_class(encoder).from_pretrained(base, dtype=torch.float32, **_local_kw(base))
    if enc.backbone.get_input_embeddings().num_embeddings < len(tok):
        enc.backbone.resize_token_embeddings(len(tok), mean_resizing=False)
        init_marker_embeddings(enc.backbone, tok, MARKER_SEEDS)
    g = torch.Generator().manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g)))
        heads = DecisionHeads(enc.hidden_size)
    return enc, heads, tok


def write_checkpoint(
    out_dir: str | Path,
    enc: MaskedEncoder,
    heads: DecisionHeads,
    tok,
    calibration: dict | None = None,
    meta: dict | None = None,
) -> Path:
    """Write atomically: build in a sibling temp dir, then swap, so a kill never leaves a half checkpoint."""
    out = Path(out_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / f".{out.name}.tmp-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "backbone").mkdir(parents=True)
    backbone = enc.backbone
    # Always store fp32 weights on CPU regardless of the training device/dtype.
    sd = {k: v.detach().to("cpu", torch.float32).contiguous() for k, v in backbone.state_dict().items()}
    backbone.save_pretrained(str(tmp / "backbone"), state_dict=sd)
    tok.save_pretrained(str(tmp / "backbone"))
    save_file({k: v.detach().to("cpu", torch.float32).contiguous() for k, v in heads.state_dict().items()},
              str(tmp / "heads.safetensors"))
    cal = dict(DEFAULT_CALIBRATION) if calibration is None else calibration
    (tmp / "calibration.json").write_text(json.dumps(cal, indent=2))
    m = {
        "name": ENGINE_NAME,
        "base": BASE_MODEL,
        "markers": list(MARKERS),
        "max_len": MAX_LEN,
        "buckets": list(BUCKETS),
        "hidden_size": enc.hidden_size,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    m.update(meta or {})
    (tmp / "meta.json").write_text(json.dumps(m, indent=2))
    old = out.parent / f".{out.name}.old-{os.getpid()}"
    if out.exists():
        os.replace(out, old)
    os.replace(tmp, out)
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)
    return out


def load_checkpoint(ckpt_dir: str | Path, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
                    encoder: str = "reference"):
    """-> (encoder, heads, tokenizer, calibration, meta), in eval mode on `device`."""
    ckpt = Path(ckpt_dir)
    if not (ckpt / "backbone").is_dir() or not (ckpt / "heads.safetensors").is_file():
        raise FileNotFoundError(f"not a fast-engine checkpoint: {ckpt}")
    meta = json.loads((ckpt / "meta.json").read_text()) if (ckpt / "meta.json").is_file() else {}
    calib = dict(DEFAULT_CALIBRATION)
    if (ckpt / "calibration.json").is_file():
        calib.update(json.loads((ckpt / "calibration.json").read_text()))
    tok = load_tokenizer(ckpt / "backbone")
    enc = encoder_class(encoder).from_pretrained(ckpt / "backbone", dtype=torch.float32)
    heads = DecisionHeads(enc.hidden_size)
    heads.load_state_dict(load_file(str(ckpt / "heads.safetensors")))
    enc.to(device=device, dtype=dtype).eval()
    heads.to(device=device, dtype=dtype).eval()
    return enc, heads, tok, calib, meta


@dataclass
class Fragment:
    """A question, or a contiguous slice of its options, placed in one pass."""

    block: QBlock  # items/labels restricted to the slice
    part: tuple[list[int], list[list[int]]]  # token ids: header, items
    qid: str
    lo: int  # index of the slice's first option in the full question

    @property
    def n_tokens(self) -> int:
        return len(self.part[0]) + sum(len(x) for x in self.part[1])


def plan_passes(n_state: int, blocks: list[QBlock], parts: list, max_tokens: int, max_flat: int,
                min_options: int = 1) -> list[list[Fragment]]:
    """Fragments and passes for one request (see module doc). Raises EngineError 400 if some question
    cannot fit even one option next to the state (positions) or the state alone exceeds max_flat."""
    frags: list[Fragment] = []
    cap = max_flat - n_state
    for b, (header, items) in zip(blocks, parts):
        longest = max((len(x) for x in items), default=0)
        need = n_state + len(header) + longest
        if need > max_tokens or len(header) + longest > cap:
            raise EngineError({"detail": "max_tokens_exceeded", "tokens": need, "max_tokens": max_tokens,
                               "qid": b.qid}, status=400)
        total = len(header) + sum(len(x) for x in items)
        if total <= cap or b.kind == "noul":
            frags.append(Fragment(b, (header, items), b.qid, 0))
            continue
        lo, used = 0, len(header)
        for j, it in enumerate(items):
            if used + len(it) > cap and j - lo >= min_options:
                frags.append(_slice(b, header, items, lo, j))
                lo, used = j, len(header)
            used += len(it)
        frags.append(_slice(b, header, items, lo, len(items)))
    passes: list[list[Fragment]] = []
    room: list[int] = []
    for f in frags:  # first fit in request order; never two fragments of one qid in a pass
        for pi, p in enumerate(passes):
            if f.n_tokens <= room[pi] and all(x.qid != f.qid for x in p):
                p.append(f)
                room[pi] -= f.n_tokens
                break
        else:
            passes.append([f])
            room.append(cap - f.n_tokens)
    return passes


def _slice(b: QBlock, header: list[int], items: list[list[int]], lo: int, hi: int) -> Fragment:
    blk = dataclasses.replace(b, items=b.items[lo:hi], labels=b.labels[lo:hi])
    return Fragment(blk, (header, items[lo:hi]), b.qid, lo)


def _replace_segment(state: Entry, i: int, text: str) -> Entry:
    """The state with the text of its i-th `state_segments` segment replaced (same top-level shape; the field
    becomes a plain string). Mirrors `serialize.state_segments`: str -> one segment, dict -> one per key, list ->
    one per item up to MAX_ARRAY_SEGMENTS plus one tail segment for the rest."""
    if isinstance(state, dict):
        key = list(state)[i]
        return {k: (text if k == key else v) for k, v in state.items()}
    if isinstance(state, list):
        if i < MAX_ARRAY_SEGMENTS:
            return [text if j == i else x for j, x in enumerate(state)]
        return [*state[:MAX_ARRAY_SEGMENTS], text]  # the "[64:]" tail segment
    return text


class FastEngine:
    """Implements engine.base.Engine. Not thread-safe (the server serialises calls)."""

    name = ENGINE_NAME

    def __init__(
        self,
        ckpt_dir: Path | str,
        device: str | None = "mps",
        dtype: torch.dtype = torch.float16,
        max_tokens: int | None = None,
        attn: str = "tree",
        max_flat_tokens: int | None = None,
        encoder: str = "banded",
        calibration: str | Path | Mapping | None = None,
        drop_header_calibration: bool = False,
        threads: int | None = None,
    ):
        """attn="tree" (default) runs the exact sparse layout; "dense" the [1,1,L,L]-mask reference.
        max_tokens: longest state + header + option one question may see (<= 8192); None = the checkpoint's
        meta.json "max_len" (MAX_LEN when absent).
        max_flat_tokens: token budget of one forward pass (tree: default max(max_tokens, 8192); dense:
        max_tokens); longer requests run as several passes with exact option batching.
        encoder: "banded" (default, fast exact tree path) or "reference" (v1 MaskedEncoder).
        calibration: a calibration.json path or dict that REPLACES the checkpoint's file for this run (W9);
        drop_header_calibration drops `by_header` entries (global (kind × K-bucket) temperatures only).
        threads: torch CPU threads for this process (None = leave torch's default)."""
        if threads is not None:
            torch.set_num_threads(int(threads))
        self.attn = attn
        self.device = pick_device(device)
        if self.device.type == "cpu" and dtype == torch.float16:
            dtype = torch.float32  # fp16 matmuls on CPU are slow and poorly supported
        self.dtype = dtype
        self.ckpt_dir = Path(ckpt_dir)
        self.enc, self.heads, self.tok, self.calib, self.meta = load_checkpoint(self.ckpt_dir, self.device, dtype,
                                                                                encoder=encoder)
        self.calibration_info = describe_calibration(self.calib, self.ckpt_dir / "calibration.json")
        if calibration is not None:
            self.calib, self.calibration_info = load_calibration(calibration)
        if drop_header_calibration and self.calib.get("by_header"):
            self.calib = {**self.calib, "by_header": {}}
            self.calibration_info = {**self.calibration_info, "by_header_dropped": True, "n_by_header": 0}
        if max_tokens is None:
            max_tokens = int(self.meta.get("max_len") or MAX_LEN)
            max_tokens = min(max_tokens, MAX_POSITIONS)
        if not 0 < max_tokens <= MAX_POSITIONS:
            raise ValueError(f"max_tokens must be in (0, {MAX_POSITIONS}]")
        self.max_tokens = max_tokens
        if attn == "dense":
            self.max_flat_tokens = max_tokens if max_flat_tokens is None else min(max_flat_tokens, max_tokens)
        else:
            self.max_flat_tokens = max_flat_tokens or max(max_tokens, 8192)
        self.packer = Packer(self.tok, max_len=max_tokens)
        self.last_passes = 0

    # ------------------------------------------------------------------ Engine protocol

    def supports(self, req: SystemOneRequest) -> bool:
        for q in req.questions.values():
            if getattr(q, "type", None) == "choice" and len(q.criteria) > MAX_CHOICE_OPTIONS:
                return False
        try:
            segs = state_segments(req.state)
            blocks = [question_block(qid, q) for qid, q in req.questions.items()]
            plan_passes(len(self.packer._state_ids(segs)), blocks, self.packer._block_parts(blocks), self.max_tokens,
                        self.max_flat_tokens)
            return True
        except Exception:
            return False

    def count_tokens(self, req: SystemOneRequest) -> int:
        segs = state_segments(req.state)
        blocks = [question_block(qid, q) for qid, q in req.questions.items()]
        return self.packer.count(segs, blocks)

    # ------------------------------------------------------------------ benchmax: capacity and overflow

    @staticmethod
    def _branch_need(parts: list[tuple[list[int], list[list[int]]]]) -> int:
        """Header + longest item over all questions: the tokens one question branch adds to the state."""
        return max((len(h) + max((len(x) for x in it), default=0) for h, it in parts), default=0)

    def positions_needed(self, req: SystemOneRequest) -> int:
        """Tokens the router compares with `max_tokens`: state + the longest question branch (module doc).
        The whole request may be far longer; it then runs as several exact passes."""
        segs = state_segments(req.state)
        blocks = [question_block(qid, q) for qid, q in req.questions.items()]
        return len(self.packer._state_ids(segs)) + self._branch_need(self.packer._block_parts(blocks))

    def fit_state(
        self,
        state: Entry,
        questions: Mapping[str, Question],
        *,
        margin: int = 8,
        min_keep: int = 16,
        max_rounds: int = 12,
    ) -> tuple[Entry, int]:
        """overflow="truncate" (W5): cut the longest state field at its end, on token boundaries, until the
        state plus the longest question branch fits `max_tokens`. Returns (state, state tokens cut); (state, 0)
        when nothing had to change. Questions are never touched (headers and options stay byte-identical).
        Raises the engine's too-long EngineError when even a `min_keep`-token state cannot host the longest
        question, i.e. a header or option alone is too long."""
        blocks = [question_block(qid, q) for qid, q in questions.items()]
        branch = self._branch_need(self.packer._block_parts(blocks))
        budget = self.max_tokens - branch  # state ids allowed, [CLS] and [SEP]s included
        segs = state_segments(state)
        n0 = n = len(self.packer._state_ids(segs))
        if n0 <= budget:
            return state, 0

        def too_long() -> EngineError:
            return EngineError({"detail": "max_tokens_exceeded", "tokens": n + branch, "max_tokens": self.max_tokens},
                               status=400)

        if budget < min_keep + 2:
            raise too_long()
        cur = state
        for _ in range(max_rounds):
            lens = [len(self.packer.encode([s.text])[0]) for s in segs]
            if not lens:
                break
            i = max(range(len(segs)), key=lambda k: lens[k])
            if lens[i] <= min_keep:
                break  # every field is already minimal
            ids = self.packer.encode([segs[i].text])[0]
            keep = max(min_keep, len(ids) - (n - budget + margin))
            text = self.tok.decode(list(ids[:keep]), skip_special_tokens=False, clean_up_tokenization_spaces=False)
            cur = _replace_segment(cur, i, text)
            segs = state_segments(cur)
            n = len(self.packer._state_ids(segs))
            if n <= budget:
                return cur, n0 - n
        raise too_long()

    @torch.inference_mode()
    def evaluate(self, state: Entry, questions: Mapping[str, Question]) -> EngineResult:
        t0 = time.perf_counter()
        segs = state_segments(state)
        blocks = [question_block(qid, q) for qid, q in questions.items()]
        t1 = time.perf_counter()
        raw, n_input, timings = self._forward(segs, blocks)
        t3 = time.perf_counter()
        dists: dict[str, RawDist] = {}
        for b in blocks:
            p = calibrate_logits(b.kind, header_key(b.header), raw[b.qid], self.calib)
            if b.kind == "noul":
                dists[b.qid] = RawDist("noul", (float(p[0]),), ())
            else:
                dists[b.qid] = RawDist(b.kind, tuple(float(x) for x in p), b.labels)
        t4 = time.perf_counter()
        return EngineResult(
            dists=dists,
            input_tokens=n_input,
            output_tokens=sum(len(b.items) + 1 for b in blocks),
            engine=self.name,
            timings_ms={
                "serialize": (t1 - t0) * 1e3,
                "pack": timings["pack"],
                "forward": timings["forward"],
                "heads": (t4 - t3) * 1e3,
                "total": (t4 - t0) * 1e3,
                "passes": float(timings["passes"]),
            },
        )

    @torch.inference_mode()
    def evaluate_logits(self, state: Entry, questions: Mapping[str, Question]) -> dict[str, dict]:
        """Raw (tau = 1) head logits per question: {qid: {"kind", "labels", "header_key", "logits"}}.
        choice/score logits cover all K options in request order (joined across passes); noul is [z]."""
        segs = state_segments(state)
        blocks = [question_block(qid, q) for qid, q in questions.items()]
        raw, _, _ = self._forward(segs, blocks)
        return {b.qid: {"kind": b.kind, "labels": b.labels, "header_key": header_key(b.header),
                        "logits": raw[b.qid].tolist()} for b in blocks}

    def _forward(self, segs, blocks: list[QBlock]) -> tuple[dict[str, np.ndarray], int, dict]:
        t1 = time.perf_counter()
        state = self.packer._state_ids(segs)
        parts = self.packer._block_parts(blocks)
        n_input = len(state) + sum(len(h) + sum(len(x) for x in it) for h, it in parts)
        passes = plan_passes(len(state), blocks, parts, self.max_tokens, self.max_flat_tokens)
        t_pack = (time.perf_counter() - t1) * 1e3
        t_fwd = 0.0
        chunks: dict[str, list[tuple[int, np.ndarray]]] = {}
        for frs in passes:
            t2 = time.perf_counter()
            packed = self.packer._layout(state, [f.block for f in frs], [f.part for f in frs])
            if self.attn == "tree":
                batch = collate_tree([packed], self.packer.pad_id, self.enc.window, self.device, buckets=ENGINE_BUCKETS)
                plan = build_plan([packed], batch.row_offsets, self.device)
                self.last_shape = batch.shape_key
            else:
                batch = self.packer.collate([packed], self.device)
                plan = build_plan([packed], batch.shape[1], self.device)
                self.last_shape = batch.shape
            t3 = time.perf_counter()
            t_pack += (t3 - t2) * 1e3
            out = self.heads(self.enc(batch), plan)
            host = {k: (getattr(out, k).float().cpu().numpy() if getattr(out, k) is not None else None)
                    for k in ("choice", "score", "noul")}  # one device sync per pass
            t_fwd += (time.perf_counter() - t3) * 1e3
            lo = {f.qid: f.lo for f in frs}
            for ref in plan.refs:
                qi = ref.qi
                if qi.kind == "noul":
                    z = host["noul"][ref.group : ref.group + 1]
                else:
                    z = host[qi.kind][ref.group, : len(qi.labels)]
                chunks.setdefault(qi.qid, []).append((lo[qi.qid], z.astype(np.float64)))
        raw = {q: np.concatenate([z for _, z in sorted(c, key=lambda x: x[0])]) for q, c in chunks.items()}
        self.last_passes = len(passes)
        return raw, n_input, {"pack": t_pack, "forward": t_fwd, "passes": len(passes)}

    # ------------------------------------------------------------------ extras

    def warmup(self, lengths: tuple[int, ...] = (256, 512, 768, 1024, 1536)) -> None:
        """Run one dummy pass per bucket so MPS compiles its kernels before the first real request."""
        from jev_local.schema import ChoiceQuestion

        for n in lengths:
            if n > self.max_tokens:
                continue
            words = " ".join(["word"] * max(1, n - 40))
            q = {"q": ChoiceQuestion(instructions="warmup", criteria={"a": None, "b": None})}
            try:
                self.evaluate(words, q)
            except EngineError:
                pass

    def unload(self) -> None:
        del self.enc, self.heads
        if self.device.type == "mps":
            torch.mps.empty_cache()
