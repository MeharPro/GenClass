"""BandedEncoder (engine.py) is exact: same hidden states and gradients as the reference MaskedEncoder on
multi-row training batches with padding, harness requests with many option chunks, and long states."""

from __future__ import annotations

import random

import pytest
import torch

from jev_local.engine.encoder.engine import BandedEncoder, load_tokenizer
from jev_local.engine.encoder.model import BASE_MODEL, MaskedEncoder
from jev_local.engine.encoder.tokenize_pack import Packer, collate_tree
from jev_local.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion
from jev_local.serialize import question_block, state_segments
from jev_local.train import fixture
from jev_local.train.train import encode_example

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def models():
    tok = load_tokenizer(BASE_MODEL)
    ref = MaskedEncoder.from_pretrained(BASE_MODEL, dtype=torch.float32, local_files_only=True)
    ban = BandedEncoder.from_pretrained(BASE_MODEL, dtype=torch.float32, local_files_only=True)
    for m in (ref, ban):
        m.backbone.resize_token_embeddings(len(tok), mean_resizing=False)
    ban.load_state_dict(ref.state_dict())
    return ref.eval(), ban.eval(), tok


def _compare(ref, ban, tb):
    """Exact in float64 (only zero-weight keys are skipped); fp32 differs by kernel rounding only."""
    with torch.no_grad():
        a = ref(tb)[: tb.n_tokens]
        b = ban(tb)[: tb.n_tokens]
        rel32 = ((a - b).abs().max() / a.abs().max()).item()
        ref.double(), ban.double()
        try:
            a64, b64 = ref(tb)[: tb.n_tokens], ban(tb)[: tb.n_tokens]
        finally:
            ref.float(), ban.float()
    rel64 = ((a64 - b64).abs().max() / a64.abs().max()).item()
    assert rel64 < 1e-8, rel64
    assert rel32 < 1e-4, rel32


def test_training_batch_with_padding(models):
    ref, ban, tok = models
    packer = Packer(tok, max_len=2048)
    rows = []
    for ex in fixture.make_examples(6, seed=3, n_elements=None):
        rows += encode_example(ex, packer, 2048)
    tb = collate_tree([r.pack for r in rows], packer.pad_id, ref.window)
    assert tb.trunk_idx.shape[0] == len(rows) >= 6
    _compare(ref, ban, tb)


def test_many_options_and_long_state(models):
    ref, ban, tok = models
    packer = Packer(tok, max_len=8192)
    state = {"doc": " ".join(random.Random(0).choice("alpha beta gamma delta report revenue team".split())
                             for _ in range(3000)), "msg": "please refund the duplicate charge"}
    qs = {"intent": ChoiceQuestion(instructions="Which intent?", criteria={f"i{k}": f"intent number {k} about refunds"
                                                                          for k in range(150)}),
          "urgent": NoulQuestion(instructions="Urgent?"),
          "prio": ScoreQuestion(instructions="Priority?", criteria=["low", "mid", "high"])}
    blocks = [question_block(q, v) for q, v in qs.items()]
    p = packer.pack(state_segments(state), blocks)
    short = packer.pack(state_segments({"msg": "hi"}), blocks)
    tb = collate_tree([p, short], packer.pad_id, ref.window)
    assert p.n_state > 2000 and tb.item_idx.shape[0] > 20
    _compare(ref, ban, tb)


def test_gradients_match(models):
    ref, ban, tok = models
    packer = Packer(tok, max_len=2048)
    rows = []
    for ex in fixture.make_examples(3, seed=7, n_elements=30):
        rows += encode_example(ex, packer, 2048)
    tb = collate_tree([r.pack for r in rows], packer.pad_id, ref.window)
    w = torch.randn(tb.n_tokens, ref.hidden_size, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    grads = []
    for m in (ref, ban):
        m.double()
        m.zero_grad(set_to_none=True)
        m.train()
        (m(tb)[: tb.n_tokens] * w).sum().backward()
        grads.append({n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None})
        m.zero_grad(set_to_none=True)
        m.eval().float()
    assert grads[0].keys() == grads[1].keys()
    for n in grads[0]:
        a, b = grads[0][n], grads[1][n]
        assert (a - b).abs().max() <= 1e-7 * max(a.abs().max().item(), 1.0), (n, (a - b).abs().max().item())


def test_gradients_long_state_many_options_with_checkpointing(models):
    ref, ban, tok = models
    packer = Packer(tok, max_len=4096)
    state = {"doc": " ".join(random.Random(1).choice("alpha beta gamma delta report revenue team".split())
                             for _ in range(1500))}
    qs = {"intent": ChoiceQuestion(instructions="Which intent?", criteria={f"i{k}": f"intent {k}" for k in range(90)}),
          "urgent": NoulQuestion(instructions="Urgent?")}
    blocks = [question_block(q, v) for q, v in qs.items()]
    packs = [packer.pack(state_segments(state), blocks), packer.pack(state_segments({"m": "short one"}), blocks[1:])]
    tb = collate_tree(packs, packer.pad_id, ref.window)
    w = torch.randn(tb.n_tokens, ref.hidden_size, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    grads = []
    for m, ckpt in ((ref, False), (ban, True)):
        m.double()
        m.grad_ckpt = ckpt
        m.zero_grad(set_to_none=True)
        m.train()
        (m(tb)[: tb.n_tokens] * w).sum().backward()
        grads.append({n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None})
        m.zero_grad(set_to_none=True)
        m.grad_ckpt = False
        m.eval().float()
    for n in grads[0]:
        a, b = grads[0][n], grads[1][n]
        assert (a - b).abs().max() <= 1e-7 * max(a.abs().max().item(), 1.0), (n, (a - b).abs().max().item())
