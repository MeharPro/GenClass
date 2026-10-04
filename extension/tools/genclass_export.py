"""Export the v1 computer-use model (jev-local-fast) to ONNX for GenClass (the browser extension).

RUN ON THE AZURE VM ONLY (imports torch). Usage, from ~/jev on the VM:

    .venv/bin/python scripts/genclass_export.py --ckpt models/jev-local-fast \
        --requests extension/genclass/test/fixtures/requests50.json --out ~/genclass_export

What it writes to --out:
    genclass-fp32.onnx   reference export (dense block mask built in-graph + typed heads, raw logits)
    genclass-fp16.onnx   fp16 weights/compute, fp32 inputs/outputs (WebGPU)
    genclass-q8.onnx     8-bit weight-only MatMulNBits (block 32) + fp16 embedding table, fp32 compute (WASM)
    tokenizer.json, calibration.json, meta.json (model card for the extension)
    parity.json          per-variant max |logit diff| and decision agreement vs PyTorch on the requests
    pack_fixtures.json   Python Packer output per request (token ids, positions, groups, marker positions)
    torch_fixtures.json  PyTorch raw logits + calibrated probabilities per request (JS end-to-end parity)

Graph inputs (all int64 except masks; batch is always 1, no padding):
    input_ids, position_ids, q_group, i_group      [1, L]
    choice_q [G], choice_items [G, K]               flat token index of each [Q] / [O] marker
    score_q [S], score_items [S, K2]                [Q] / [L] markers
    noul_q, noul_t, noul_f [M]                      [Q] / [T] / [F] markers
Outputs (fp32 raw logits, temperature 1; padding columns are garbage and must be ignored):
    choice_logits [G, K], score_logits [S, K2], noul_logits [M]

The attention layout is the reference one (tokenize_pack.build_masks): a token sees every state token,
plus, inside its own question, the header and its own item; local layers additionally need
|pos_q - pos_k| <= window. Padding never exists in the browser (batch 1, exact length).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from jev_local.engine.encoder.calibrate import calibrate_logits, header_key
from jev_local.engine.encoder.engine import FastEngine, load_checkpoint
from jev_local.engine.encoder.tokenize_pack import STATE, Packer
from jev_local.schema import question_from_json
from jev_local.serialize import question_block, state_segments

NEG = -1.0e4  # additive mask value: safe in fp16, exp() underflows to exactly 0


class ExportModel(nn.Module):
    def __init__(self, enc, heads):
        super().__init__()
        self.enc = enc
        self.heads = heads
        bb = enc.backbone
        self.layer_types = list(enc.layer_types)
        self.window = enc.window
        self.H, self.D, self.d = enc.n_heads, enc.head_dim, enc.hidden_size
        # RoPE inverse frequencies per layer type (HF rotary_emb; attention_scaling is 1 for "default").
        self.rope_types = sorted(set(self.layer_types))
        for t in self.rope_types:
            pos = torch.arange(8, dtype=torch.long)[None]
            cos, _ = bb.rotary_emb(torch.zeros(1, 8, self.d), pos, t)
            # Recover inv_freq from cos at position 1 is lossy; read the buffer instead.
            inv = getattr(bb.rotary_emb, f"{t}_inv_freq", None)
            if inv is None:
                inv = bb.rotary_emb.inv_freq
            # Plain floats, not a buffer: model.half() must not round the RoPE frequencies.
            setattr(self, f"inv_{t}", [float(x) for x in inv.detach().float()])
            scaling = getattr(bb.rotary_emb, f"{t}_attention_scaling", getattr(bb.rotary_emb, "attention_scaling", 1.0))
            assert float(scaling) == 1.0, scaling
            # sanity: our formula reproduces HF's cos
            f = pos[0, :, None].float() * inv[None, :].float()
            ref = torch.cat([f, f], -1).cos()
            assert torch.allclose(ref, cos[0].float(), atol=1e-5), "rope mismatch"

    def _rope(self, x, cos, sin):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        rot = torch.cat([-x2, x1], dim=-1)
        return x * cos + rot * sin

    def forward(self, input_ids, position_ids, q_group, i_group, choice_q, choice_items, score_q, score_items,
                noul_q, noul_t, noul_f):
        bb = self.enc.backbone
        h = bb.embeddings(input_ids=input_ids)  # [1, L, d]
        L = input_ids.shape[1]
        qg, ig, pos = q_group[0], i_group[0], position_ids[0]
        key_state = (qg == STATE)[None, :]
        same_q = (qg[:, None] == qg[None, :]) & (qg[:, None] >= 0)
        same_branch = same_q & ((ig[None, :] == STATE) | (ig[None, :] == ig[:, None]))
        full = key_state | same_branch
        near = (pos[:, None] - pos[None, :]).abs() <= self.window
        local = full & near
        zero = torch.zeros((), dtype=h.dtype)
        neg = torch.full((), NEG, dtype=h.dtype)
        add = {"full_attention": torch.where(full, zero, neg), "sliding_attention": torch.where(local, zero, neg)}
        posf = pos.float()
        rope = {}
        for t in self.rope_types:
            inv_t = torch.tensor(getattr(self, f"inv_{t}"), dtype=torch.float32)
            f = posf[:, None] * inv_t[None, :]
            emb = torch.cat([f, f], dim=-1)
            rope[t] = (emb.cos()[None, None], emb.sin()[None, None])  # [1,1,L,D]
        H, D, d = self.H, self.D, self.d
        scale = D ** -0.5
        for layer, t in zip(bb.layers, self.layer_types):
            x = layer.attn_norm(h)
            qkv = layer.attn.Wqkv(x).view(1, L, 3, H, D).permute(2, 0, 3, 1, 4)  # [3,1,H,L,D]
            q, k, v = qkv[0], qkv[1], qkv[2]
            cos, sin = rope[t]
            q = self._rope(q.float(), cos, sin).to(h.dtype)
            k = self._rope(k.float(), cos, sin).to(h.dtype)
            s = torch.matmul(q, k.transpose(-1, -2)) * scale + add[t]
            p = torch.softmax(s, dim=-1)
            o = torch.matmul(p, v).transpose(1, 2).reshape(1, L, d)
            h = h + layer.attn.Wo(o)
            h = h + layer.mlp(layer.mlp_norm(h))
        h = bb.final_norm(h)[0]  # [L, d]
        hd = self.heads
        n = hd.norm

        def pairwise(mlp, qi, items):
            hq = n(h[qi])[:, None, :]
            ho = n(h[items])
            hq = hq.expand_as(ho)
            return mlp(torch.cat([hq, ho, hq * ho], dim=-1)).squeeze(-1)

        choice = pairwise(hd.choice_mlp, choice_q, choice_items)
        score = pairwise(hd.score_mlp, score_q, score_items)
        hq, ht, hf = n(h[noul_q]), n(h[noul_t]), n(h[noul_f])
        noul = hd.noul_mlp(torch.cat([hq, ht, hf, hq * ht, hq * hf], dim=-1)).squeeze(-1)
        return choice.float(), score.float(), noul.float()


def make_q8(src: Path, dst: Path) -> None:
    """Weight-only 8-bit MatMuls + fp16 embedding table (Cast to fp32 after the Gather). Dynamic int8 (activations
    quantized too) lost 7% of argmax agreement on the parity set because of activation outliers; this keeps 99.8%."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer

    m = onnx.load(str(src))
    cfg = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, bits=8, op_types_to_quantize=("MatMul",),
                                       quant_axes=(("MatMul", 0),))
    q = MatMulNBitsQuantizer(m, algo_config=cfg)
    q.process()
    m = q.model.model
    inits = {i.name: i for i in m.graph.initializer}
    for n in list(m.graph.node):
        if n.op_type == "Gather" and n.input[0] in inits:
            w = numpy_helper.to_array(inits[n.input[0]])
            if w.ndim == 2 and w.shape[0] > 10000:
                new = numpy_helper.from_array(w.astype(np.float16), n.input[0] + "_fp16")
                m.graph.initializer.remove(inits[n.input[0]])
                m.graph.initializer.append(new)
                out_name = n.output[0]
                n.input[0], n.output[0] = new.name, out_name + "_fp16"
                cast = helper.make_node("Cast", [out_name + "_fp16"], [out_name], to=TensorProto.FLOAT, name=n.name + "_to_fp32")
                m.graph.node.insert(list(m.graph.node).index(n) + 1, cast)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))


def plan_inputs(p) -> dict[str, np.ndarray]:
    """Feeds for one Packed sequence (mirrors heads.build_plan with offset 0). Kinds with no question get one
    dummy row pointing at token 0, so every input is non-empty."""
    by = {"choice": [], "score": [], "noul": []}
    for qi in p.q_index.values():
        by[qi.kind].append(qi)

    def grp(lst):
        if not lst:
            return np.zeros(1, np.int64), np.zeros((1, 1), np.int64)
        k = max(len(q.item_pos) for q in lst)
        items = np.zeros((len(lst), k), np.int64)
        for g, q in enumerate(lst):
            items[g, : len(q.item_pos)] = q.item_pos
        return np.array([q.q_pos for q in lst], np.int64), items

    cq, ci = grp(by["choice"])
    sq, si = grp(by["score"])
    nl = by["noul"] or []
    nq = np.array([q.q_pos for q in nl] or [0], np.int64)
    nt = np.array([q.item_pos[0] for q in nl] or [0], np.int64)
    nf = np.array([q.item_pos[1] for q in nl] or [0], np.int64)
    a = lambda x: np.asarray(x, np.int64)[None]  # noqa: E731
    return {"input_ids": a(p.input_ids), "position_ids": a(p.position_ids), "q_group": a(p.q_group),
            "i_group": a(p.i_group), "choice_q": cq, "choice_items": ci, "score_q": sq, "score_items": si,
            "noul_q": nq, "noul_t": nt, "noul_f": nf}


def unpack(p, outs) -> dict[str, np.ndarray]:
    """Graph outputs -> raw logits per qid (request order), like FastEngine._forward."""
    choice, score, noul = (np.asarray(o, np.float64) for o in outs)
    idx = {"choice": 0, "score": 0, "noul": 0}
    res = {}
    for qid, qi in p.q_index.items():
        g = idx[qi.kind]
        idx[qi.kind] += 1
        if qi.kind == "noul":
            res[qid] = noul[g : g + 1]
        else:
            res[qid] = (choice if qi.kind == "choice" else score)[g, : len(qi.labels)]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("models/jev-local-fast"))
    ap.add_argument("--requests", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    enc, heads, tok, calib, meta = load_checkpoint(args.ckpt, "cpu", torch.float32, encoder="reference")
    model = ExportModel(enc, heads).eval()
    packer = Packer(tok, max_len=8192)  # total length; positions restart per branch (FastEngine semantics)
    reqs = json.loads(args.requests.read_text())

    packs, blocks_all = [], []
    for r in reqs:
        qs = {k: question_from_json(v) for k, v in r["questions"].items()}
        segs = state_segments(r["state"])
        blocks = [question_block(k, q) for k, q in qs.items()]
        packs.append(packer.pack(segs, blocks))
        blocks_all.append(blocks)

    # ---------------------------------------------------------------- export
    feeds0 = plan_inputs(packs[0])
    names = list(feeds0)
    dyn = {"input_ids": {1: "L"}, "position_ids": {1: "L"}, "q_group": {1: "L"}, "i_group": {1: "L"},
           "choice_q": {0: "G"}, "choice_items": {0: "G", 1: "K"}, "score_q": {0: "S"}, "score_items": {0: "S", 1: "K2"},
           "noul_q": {0: "M"}, "noul_t": {0: "M"}, "noul_f": {0: "M"},
           "choice_logits": {0: "G", 1: "K"}, "score_logits": {0: "S", 1: "K2"}, "noul_logits": {0: "M"}}
    fp32 = out / "genclass-fp32.onnx"
    t0 = time.time()
    with torch.no_grad():
        targs = tuple(torch.from_numpy(feeds0[n]) for n in names)
        try:
            torch.onnx.export(model, targs, str(fp32), input_names=names,
                              output_names=["choice_logits", "score_logits", "noul_logits"], dynamic_axes=dyn,
                              opset_version=args.opset, do_constant_folding=True, dynamo=False)
        except TypeError:
            torch.onnx.export(model, targs, str(fp32), input_names=names,
                              output_names=["choice_logits", "score_logits", "noul_logits"], dynamic_axes=dyn,
                              opset_version=args.opset, do_constant_folding=True)
    print(f"exported fp32 in {time.time() - t0:.1f}s", flush=True)

    import copy

    import onnx

    onnx.checker.check_model(onnx.load(str(fp32)))
    # fp16: export the half-precision module directly (RoPE angles and the outputs stay fp32; LayerNormalization
    # computes its statistics in fp32 inside ORT). Converter tools mangle this graph's Cast nodes.
    m16 = copy.deepcopy(model).half().eval()
    with torch.no_grad():
        torch.onnx.export(m16, targs, str(out / "genclass-fp16.onnx"), input_names=names,
                          output_names=["choice_logits", "score_logits", "noul_logits"], dynamic_axes=dyn,
                          opset_version=args.opset, do_constant_folding=True, dynamo=False)
    make_q8(fp32, out / "genclass-q8.onnx")
    print("converted fp16 + q8", flush=True)

    # ---------------------------------------------------------------- parity
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    sessions = {v: ort.InferenceSession(str(out / f"genclass-{v}.onnx"), so, providers=["CPUExecutionProvider"])
                for v in ("fp32", "fp16", "q8")}
    eng = FastEngine(args.ckpt, device="cpu", dtype=torch.float32, encoder="banded", threads=args.threads)

    pack_fx, torch_fx = [], []
    stats = {v: {"max_abs_logit": 0.0, "max_abs_prob": 0.0, "argmax_agree": 0, "argmax_total": 0,
                 "noul_side_agree": 0, "noul_total": 0, "ms": []} for v in sessions}
    stats["torch_dense_export_module"] = {"max_abs_logit": 0.0}
    torch_ms = []
    for r, p, blocks in zip(reqs, packs, blocks_all):
        qs = {k: question_from_json(v) for k, v in r["questions"].items()}
        t1 = time.perf_counter()
        ref = eng.evaluate_logits(r["state"], qs)
        torch_ms.append((time.perf_counter() - t1) * 1e3)
        ref_logits = {q: np.asarray(v["logits"], np.float64) for q, v in ref.items()}
        ref_probs = {b.qid: calibrate_logits(b.kind, header_key(b.header), ref_logits[b.qid], calib) for b in blocks}
        feeds = plan_inputs(p)
        with torch.no_grad():
            mo = model(*(torch.from_numpy(feeds[n]) for n in names))
        dm = unpack(p, [x.numpy() for x in mo])
        stats["torch_dense_export_module"]["max_abs_logit"] = max(
            stats["torch_dense_export_module"]["max_abs_logit"],
            max(float(np.abs(dm[q] - ref_logits[q]).max()) for q in ref_logits))
        for v, sess in sessions.items():
            t2 = time.perf_counter()
            o = sess.run(None, feeds)
            stats[v]["ms"].append((time.perf_counter() - t2) * 1e3)
            got = unpack(p, o)
            st = stats[v]
            for b in blocks:
                q = b.qid
                st["max_abs_logit"] = max(st["max_abs_logit"], float(np.abs(got[q] - ref_logits[q]).max()))
                gp = calibrate_logits(b.kind, header_key(b.header), got[q], calib)
                st["max_abs_prob"] = max(st["max_abs_prob"], float(np.abs(gp - ref_probs[q]).max()))
                if b.kind == "noul":
                    st["noul_total"] += 1
                    st["noul_side_agree"] += int((gp[0] >= 0.5) == (ref_probs[q][0] >= 0.5))
                else:
                    st["argmax_total"] += 1
                    st["argmax_agree"] += int(int(np.argmax(gp)) == int(np.argmax(ref_probs[q])))
        pack_fx.append({"id": r["id"], "input_ids": p.input_ids, "position_ids": p.position_ids, "q_group": p.q_group,
                        "i_group": p.i_group, "n_state": p.n_state,
                        "q_index": {q: {"kind": qi.kind, "header": qi.header, "labels": list(qi.labels),
                                        "q_pos": qi.q_pos, "item_pos": list(qi.item_pos)} for q, qi in p.q_index.items()}})
        torch_fx.append({"id": r["id"], "logits": {q: v.tolist() for q, v in ref_logits.items()},
                         "probs": {q: v.tolist() for q, v in ref_probs.items()},
                         "header_key": {b.qid: header_key(b.header) for b in blocks}})

    for v in sessions:
        ms = stats[v].pop("ms")
        stats[v]["ort_cpu_ms_p50"] = float(np.median(ms))
        stats[v]["file_mb"] = round((out / f"genclass-{v}.onnx").stat().st_size / 1e6, 1)
    stats["torch_banded_ms_p50"] = float(np.median(torch_ms))
    stats["n_requests"] = len(reqs)
    stats["tokens"] = {"min": min(p.length for p in packs), "median": int(np.median([p.length for p in packs])),
                       "max": max(p.length for p in packs)}
    stats["threads"] = args.threads
    (out / "parity.json").write_text(json.dumps(stats, indent=2))
    (out / "pack_fixtures.json").write_text(json.dumps(pack_fx))
    (out / "torch_fixtures.json").write_text(json.dumps(torch_fx))
    shutil.copy(args.ckpt / "backbone" / "tokenizer.json", out / "tokenizer.json")
    (out / "calibration.json").write_text(json.dumps(calib, indent=2))
    (out / "meta.json").write_text(json.dumps({
        "name": "genclass-model-0.1.0", "source_checkpoint": meta.get("name"), "max_len": meta.get("max_len"),
        "window": model.window, "hidden_size": model.d, "layer_types": model.layer_types,
        "markers": {m: tok.get_vocab()[m] for m in ("[Q]", "[O]", "[L]", "[T]", "[F]")},
        "cls_id": tok.cls_token_id, "sep_id": tok.sep_token_id, "pad_id": tok.pad_token_id,
        "inputs": names, "outputs": ["choice_logits", "score_logits", "noul_logits"],
    }, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
