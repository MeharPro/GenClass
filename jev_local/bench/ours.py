"""Run a jev-local FastEngine checkpoint on jevbench requests, with EXACT option / question chunking (VM only).

Why chunking is exact: the encoder isolates every option block (it sees the state, its own question header
and itself, never a sibling option or another question), state tokens never attend to question tokens, and
sibling options restart at the same position offset. So an option's logit does not depend on which other
options or questions share the sequence. A request longer than the engine's max_tokens is therefore split
into several passes that repeat the state, each carrying some questions or a contiguous slice of one
question's options; raw logits are concatenated and normalised once (softmax over all K / sigmoid) with the
checkpoint's temperature. If even the state plus the smallest unit does not fit, the longest state field
is cut at its end (recorded as `truncated`).

`OursRunner.answer(request) -> (answers, info)`; answers use the Jev wire shapes, unrounded.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jev_local.engine.encoder.calibrate import header_key
from jev_local.engine.encoder.engine import FastEngine
from jev_local.engine.encoder.heads import build_plan
from jev_local.engine.encoder.tokenize_pack import collate_tree
from jev_local.schema import question_from_json
from jev_local.serialize import QBlock, Segment, question_block, state_segments

SEP = "\x1f"  # sub-block id separator: "<qid>\x1f<chunk>"


class OursRunner:
    def __init__(self, ckpt: str | Path, threads: int = 2, max_tokens: int | None = None):
        torch.set_num_threads(threads)
        self.eng = FastEngine(ckpt, device="cpu", dtype=torch.float32)
        if max_tokens:
            self.eng.max_tokens = max_tokens
            self.eng.packer.max_len = max_tokens
        self.max = self.eng.max_tokens
        self.packer = self.eng.packer
        self.name = f"{self.eng.name}@{Path(ckpt).name}"

    # ------------------------------------------------------------------ token accounting

    def _state_len(self, segs: list[Segment]) -> int:
        return self.packer.count(segs, [])

    def _lens(self, b: QBlock) -> tuple[int, list[int]]:
        enc = self.packer.encode([b.header, *b.items])
        return len(enc[0]) + 1, [len(e) + 1 for e in enc[1:]]  # +1: the [Q] / item marker

    def _units(self, blocks: list[QBlock], budget: int) -> list[tuple[QBlock, int]] | None:
        """Split every block into sub-blocks of at most `budget` tokens. None if some unit cannot fit."""
        out: list[tuple[QBlock, int]] = []
        for b in blocks:
            hl, il = self._lens(b)
            if hl + sum(il) <= budget:
                out.append((b, hl + sum(il)))
                continue
            if b.kind == "noul" or hl + max(il) > budget:
                return None
            start, used, j = 0, hl, 0
            for i, n in enumerate(il + [None]):
                if n is None or used + n > budget:
                    out.append((replace(b, qid=f"{b.qid}{SEP}{j}", items=b.items[start:i], labels=b.labels[start:i]), used))
                    j += 1
                    start, used = i, hl + (n or 0)
                else:
                    used += n
        return out

    def _plan(self, segs: list[Segment], blocks: list[QBlock]) -> tuple[list[Segment], list[list[QBlock]], bool]:
        truncated = False
        for _ in range(8):
            s = self._state_len(segs)
            units = self._units(blocks, self.max - s)
            if units is not None:
                passes: list[list[QBlock]] = []
                cur: list[QBlock] = []
                used = s
                for b, n in units:
                    if cur and used + n > self.max:
                        passes.append(cur)
                        cur, used = [], s
                    cur.append(b)
                    used += n
                if cur:
                    passes.append(cur)
                return segs, passes, truncated
            # Cut the longest state field at its end: need room for the largest single unit.
            need = max(self._lens(b)[0] + (sum(self._lens(b)[1]) if b.kind == "noul" else max(self._lens(b)[1]))
                       for b in blocks)
            excess = s + need - self.max + 16
            i = max(range(len(segs)), key=lambda k: len(segs[k].text))
            ids = self.packer.encode([segs[i].text])[0]
            keep = max(16, len(ids) - excess)
            segs = list(segs)
            segs[i] = Segment(segs[i].key, self.eng.tok.decode(list(ids[:keep])))
            truncated = True
        raise ValueError("request cannot be fitted even after truncation")

    # ------------------------------------------------------------------ forward

    @torch.inference_mode()
    def _raw_logits(self, segs: list[Segment], passes: list[list[QBlock]]) -> dict[str, np.ndarray]:
        packs = [self.packer.pack(segs, p) for p in passes]
        batch = collate_tree(packs, self.packer.pad_id, self.eng.enc.window, self.eng.device)
        plan = build_plan(packs, batch.row_offsets, self.eng.device)
        h = self.eng.enc(batch)
        out = self.eng.heads(h, plan)
        res: dict[str, np.ndarray] = {}
        host = {k: (getattr(out, k).float().cpu().numpy() if getattr(out, k) is not None else None)
                for k in ("choice", "score", "noul")}
        for ref in plan.refs:
            qi = ref.qi
            if qi.kind == "noul":
                res[qi.qid] = np.array([host["noul"][ref.group]])
            else:
                res[qi.qid] = host[qi.kind][ref.group, : len(qi.labels)]
        return res

    def _tau(self, kind: str, header: str) -> float:
        cal = self.eng.calib
        return float(cal.get("by_header", {}).get(header_key(header), cal.get(kind, 1.0)))

    def answer(self, request: dict[str, Any]) -> tuple[dict[str, dict], dict[str, Any]]:
        t0 = time.perf_counter()
        segs = state_segments(request["state"])
        qs = {qid: question_from_json(q) for qid, q in request["questions"].items()}
        blocks = [question_block(qid, q) for qid, q in qs.items()]
        segs2, passes, truncated = self._plan(segs, blocks)
        raw = self._raw_logits(segs2, passes)
        answers: dict[str, dict] = {}
        for b in blocks:
            parts = [raw[b.qid]] if b.qid in raw else [raw[k] for k in sorted(
                (k for k in raw if k.startswith(b.qid + SEP)), key=lambda k: int(k.split(SEP)[1]))]
            z = np.concatenate(parts).astype(np.float64) / self._tau(b.kind, b.header)
            if b.kind == "noul":
                answers[b.qid] = {"type": "noul", "noul": float(1 / (1 + np.exp(-z[0])))}
                continue
            z = z - z.max()
            p = np.exp(z)
            p /= p.sum()
            if b.kind == "choice":
                answers[b.qid] = {"type": "choice", "probabilities": dict(zip(b.labels, map(float, p)))}
            else:
                answers[b.qid] = {"type": "score", "probabilities": {str(i): float(x) for i, x in enumerate(p)}}
        n_tok = sum(self.packer.count(segs2, p) for p in passes)
        return answers, {"passes": len(passes), "truncated": truncated, "input_tokens": n_tok,
                         "ms": (time.perf_counter() - t0) * 1e3}
