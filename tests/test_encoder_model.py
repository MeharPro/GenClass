"""Custom masked ModernBERT forward (CONTRACT "B" test (a)) and the tree layout's exactness."""

from __future__ import annotations

import pytest
import torch

from jev_local.engine.encoder.model import BASE_MODEL, MaskedEncoder
from jev_local.engine.encoder.tokenize_pack import Packer, build_masks, collate_tree
from jev_local.schema import ChoiceQuestion, NoulCriteria, NoulQuestion, ScoreQuestion
from jev_local.serialize import question_block, state_segments

pytestmark = pytest.mark.model  # needs the cached ettin weights


@pytest.fixture(scope="module")
def hf():
    from transformers import AutoModel

    torch.manual_seed(0)
    return AutoModel.from_pretrained(BASE_MODEL, dtype=torch.float32, attn_implementation="sdpa",
                                     local_files_only=True).eval()


@pytest.fixture(scope="module")
def enc(hf):
    return MaskedEncoder(hf).eval()


@pytest.fixture(scope="module")
def menc_packer():
    """Encoder with the marker tokens added (resized embeddings), as the engine builds it."""
    from jev_local.engine.encoder.engine import init_model

    enc, _, tok = init_model(seed=0)
    return enc.eval(), Packer(tok)


def test_config_matches_contract(enc):
    cfg = enc.config
    assert (cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads) == (384, 10, 6)
    assert enc.window == 64
    assert enc.layer_types == ["full_attention" if i % 3 == 0 else "sliding_attention" for i in range(10)]


@pytest.mark.parametrize("L", [40, 300])  # 300 > 128, so the local window actually bites
@torch.inference_mode()
def test_all_attend_matches_hf(hf, enc, L):
    g = torch.Generator().manual_seed(L)
    ids = torch.randint(5, 50000, (2, L), generator=g)
    ids[:, 0] = hf.config.cls_token_id
    ref = hf(input_ids=ids).last_hidden_state
    pos = torch.arange(L)[None].expand(2, -1)
    full = torch.ones(2, 1, L, L, dtype=torch.bool)
    local = full & ((pos[:, None, :, None] - pos[:, None, None, :]).abs() <= enc.window)
    out = enc.forward_raw(ids, pos, {"full_attention": full, "sliding_attention": local})
    assert out.shape == ref.shape
    err = (out - ref).abs().max().item()
    assert err <= 1e-4, err


def _request():
    segs = state_segments({
        "screen": 'Mail, window "Inbox"',
        "focused": 'text field "Search"',
        "transcript": "click reply all and then type thanks so much for the update " * 6,
    })
    qs = {
        "intent": ChoiceQuestion(instructions="Which action?", criteria={"click": "click something", "type": None,
                                                                          "wait": "not enough words", "none": None}),
        "done": NoulQuestion(instructions="Is it complete?", criteria=NoulCriteria(true="complete", false="unfinished")),
        "amount": ScoreQuestion(instructions="How far?", criteria=["a little", "a page", "all the way"]),
        "target": ChoiceQuestion(instructions="Which element?",
                                 criteria={f"e{i:02d}": f'button "Label number {i}" in toolbar' for i in range(40)}),
    }
    return segs, [question_block(k, q) for k, q in qs.items()]


@torch.inference_mode()
def test_tree_layout_equals_dense_mask(menc_packer):
    enc, packer = menc_packer
    segs, blocks = _request()
    p1 = packer.pack(segs, blocks)
    p2 = packer.pack(state_segments("short state"), blocks[:2])
    packs = [p1, p2]
    dense = packer.collate(packs)
    hd = enc(dense)  # [B, L, d]
    tb = collate_tree(packs, packer.pad_id, enc.window)
    ht = enc(tb)  # [N, d]
    for r, p in enumerate(packs):
        o = tb.row_offsets[r]
        err = (ht[o : o + p.length] - hd[r, : p.length]).abs().max().item()
        assert err <= 1e-4, (r, err)
    assert torch.isfinite(ht).all() and torch.isfinite(hd).all()


@torch.inference_mode()
def test_padding_and_batching_do_not_change_outputs(menc_packer):
    enc, packer = menc_packer
    segs, blocks = _request()
    p = packer.pack(segs, blocks)
    alone = enc(packer.collate([p]))[0, : p.length]
    other = packer.pack(state_segments("another request entirely"), blocks[1:3])
    batched = enc(packer.collate([other, p], pad_to=1024))[1, : p.length]
    assert (alone - batched).abs().max().item() <= 1e-4


def test_tree_backward_and_grad_checkpoint_match(menc_packer):
    enc, packer = menc_packer
    segs, blocks = _request()
    packs = [packer.pack(segs, blocks)]
    tb = collate_tree(packs, packer.pad_id, enc.window)
    grads = []
    enc.train()
    try:
        for ckpt in (False, True):
            enc.grad_ckpt = ckpt
            enc.zero_grad(set_to_none=True)
            enc(tb).pow(2).mean().backward()
            w = enc.backbone.layers[1].attn.Wqkv.weight
            assert w.grad is not None and torch.isfinite(w.grad).all()
            grads.append(w.grad.clone())
    finally:
        enc.grad_ckpt = False
        enc.zero_grad(set_to_none=True)
        enc.eval()
    err = ((grads[0] - grads[1]).abs().max() / grads[0].abs().max()).item()
    assert err <= 1e-3, err
