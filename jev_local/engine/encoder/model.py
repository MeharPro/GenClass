"""ModernBERT (ettin) forward with an arbitrary block attention mask and explicit position ids.

HF's ModernBertModel only supports padding masks (it unpads for FlashAttention, or builds a
padding/sliding mask for SDPA). The segment-block layout needs a per-query mask, so this module
reuses the HF submodules and re-implements the per-layer math from
transformers/models/modernbert/modeling_modernbert.py:

    h = LayerNorm(tok_embeddings(ids))                        # embeddings (dropout is 0)
    for layer i:                                              # global iff layer_types[i] == "full_attention"
        x = attn_norm(h)                                      # Identity for layer 0
        q, k, v = Wqkv(x) split into heads; q, k = RoPE(q, k, position_ids)   (rotation in fp32)
        h = h + Wo(SDPA(q, k, v, mask_for_layer_type, scale=head_dim**-0.5))
        h = h + mlp(mlp_norm(h))                              # GLU: Wo(gelu(a) * gate)
    h = final_norm(h)

With an all-attend mask, ordinary positions and the |i-j| <= 64 window on local layers it matches
HF's forward to ~1e-6 (tests/test_encoder_model.py).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from jev_local.engine.encoder.tokenize_pack import Batch, TreeBatch, build_masks

BASE_MODEL = "jhu-clsp/ettin-encoder-32m"


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
          heads_dim: int | None = 1) -> tuple[torch.Tensor, torch.Tensor]:
    # Same as HF apply_rotary_pos_emb: rotate in fp32, cast back (keeps fp16 positions exact-ish).
    # heads_dim: where to broadcast cos/sin over heads (dense [B,H,L,D]); None when already shaped.
    dt = q.dtype
    if heads_dim is not None:
        cos, sin = cos.unsqueeze(heads_dim), sin.unsqueeze(heads_dim)
    cos, sin = cos.float(), sin.float()
    qf, kf = q.float(), k.float()
    return (qf * cos + _rotate_half(qf) * sin).to(dt), (kf * cos + _rotate_half(kf) * sin).to(dt)


class MaskedEncoder(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone  # transformers ModernBertModel
        cfg = backbone.config
        self.layer_types: list[str] = list(cfg.layer_types)
        self.window: int = int(cfg.sliding_window)  # half-window: local layers see |i-j| <= 64
        self.n_heads: int = cfg.num_attention_heads
        self.head_dim: int = cfg.hidden_size // cfg.num_attention_heads
        self.hidden_size: int = cfg.hidden_size
        self.grad_ckpt = False

    @classmethod
    def from_pretrained(cls, path: str | Path, dtype: torch.dtype = torch.float32, **kw) -> "MaskedEncoder":
        from transformers import AutoModel
        from transformers.utils import logging as hf_logging

        # The MLM head in the base checkpoint is intentionally dropped; keep that report out of logs.
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        kw.setdefault("attn_implementation", "sdpa")
        backbone = AutoModel.from_pretrained(str(path), dtype=dtype, **kw)
        return cls(backbone)

    @property
    def config(self):
        return self.backbone.config

    def _layer(self, layer: nn.Module, h: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, L, _ = h.shape
        qkv = layer.attn.Wqkv(layer.attn_norm(h)).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each [B, H, L, D]
        q, k = _rope(q, k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.head_dim**-0.5)
        h = h + layer.attn.Wo(o.transpose(1, 2).reshape(B, L, self.hidden_size))
        return h + layer.mlp(layer.mlp_norm(h))

    def forward_raw(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, masks: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """input_ids/position_ids [B, L]; masks: layer_type -> bool [B,1,L,L]. Returns [B, L, d]."""
        bb = self.backbone
        h = bb.embeddings(input_ids=input_ids)
        rope = {t: bb.rotary_emb(h, position_ids, t) for t in set(self.layer_types)}
        for layer, t in zip(bb.layers, self.layer_types):
            cos, sin = rope[t]
            if self.grad_ckpt and self.training:
                h = checkpoint(self._layer, layer, h, masks[t], cos, sin, use_reentrant=False)
            else:
                h = self._layer(layer, h, masks[t], cos, sin)
        return bb.final_norm(h)

    def forward(self, batch: Batch | TreeBatch) -> torch.Tensor:
        """Dense Batch -> [B, L, d]; TreeBatch -> flat [N, d] (same values at every real token)."""
        if isinstance(batch, TreeBatch):
            return self.forward_tree(batch)
        return self.forward_raw(batch.input_ids, batch.position_ids, build_masks(batch, self.window))

    # ------------------------------------------------------------------ tree layout (fast path)

    def _tree_layer(self, layer: nn.Module, h: torch.Tensor, tb: TreeBatch, t: str, cos: torch.Tensor,
                    sin: torch.Tensor) -> torch.Tensor:
        N = h.shape[0]
        H, D = self.n_heads, self.head_dim
        qkv = layer.attn.Wqkv(layer.attn_norm(h)).view(N, 3, H, D)
        q, k, v = qkv.unbind(1)  # [N, H, D]
        q, k = _rope(q, k, cos[0][:, None, :], sin[0][:, None, :], heads_dim=None)
        zero = h.new_zeros(1, H, D, dtype=q.dtype)
        q, k, v = (torch.cat([x, zero.to(x.dtype)]) for x in (q, k, v))  # row N = padding slot

        def lay(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            return x[idx].transpose(-3, -2)  # [..., slots, H, D] -> [..., H, slots, D]

        scale = D**-0.5
        qt, kt, vt = lay(q, tb.trunk_idx), lay(k, tb.trunk_idx), lay(v, tb.trunk_idx)
        ot = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=tb.masks["trunk"][t], scale=scale)
        qh, kh, vh = lay(q, tb.head_idx), lay(k, tb.head_idx), lay(v, tb.head_idx)
        oh = F.scaled_dot_product_attention(
            qh, torch.cat([kt[tb.head_row], kh], 2), torch.cat([vt[tb.head_row], vh], 2),
            attn_mask=tb.masks["head"][t], scale=scale)
        qi, ki, vi = lay(q, tb.item_idx), lay(k, tb.item_idx), lay(v, tb.item_idx)
        oi = F.scaled_dot_product_attention(
            qi, torch.cat([kt[tb.item_row], kh[tb.item_head], ki], 2),
            torch.cat([vt[tb.item_row], vh[tb.item_head], vi], 2),
            attn_mask=tb.masks["item"][t], scale=scale)
        slots = torch.cat([o.transpose(-3, -2).reshape(-1, H * D) for o in (ot, oh, oi)])
        h = h + layer.attn.Wo(slots[tb.inv])
        return h + layer.mlp(layer.mlp_norm(h))

    def forward_tree(self, tb: TreeBatch) -> torch.Tensor:
        bb = self.backbone
        h = bb.embeddings(input_ids=tb.input_ids[None])[0]  # [N, d]
        rope = {t: bb.rotary_emb(h[None], tb.position_ids[None], t) for t in set(self.layer_types)}
        for layer, t in zip(bb.layers, self.layer_types):
            cos, sin = rope[t]
            if self.grad_ckpt and self.training:
                h = checkpoint(self._tree_layer, layer, h, tb, t, cos, sin, use_reentrant=False)
            else:
                h = self._tree_layer(layer, h, tb, t, cos, sin)
        return bb.final_norm(h)


def init_marker_embeddings(backbone: nn.Module, tok, markers: dict[str, str]) -> None:
    """Initialise freshly added marker rows from related words, so they start as meaningful inputs
    rather than the resize default. markers: marker token -> seed text."""
    emb = backbone.get_input_embeddings().weight
    vocab = tok.get_vocab()
    backend = tok.backend_tokenizer
    with torch.no_grad():
        cls_vec = emb[tok.cls_token_id].clone()
        for m, seed_text in markers.items():
            ids = backend.encode(seed_text, add_special_tokens=False).ids
            vec = emb[ids].mean(0) if ids else cls_vec
            emb[vocab[m]] = 0.5 * (vec + cls_vec)
