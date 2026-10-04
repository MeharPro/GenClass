"""Per-qid metrics for a fast-engine checkpoint on a data split.

    python -m jev_local.train.eval --ckpt models/jev-local-fast --data data/cu [--extra data/gen] \
        --split dev [--limit N] [--out runs/<name>/eval_dev.json] [--no-calib]

Metrics (all computed after the checkpoint's calibration.json temperatures unless --no-calib):
- choice/score: top-1 accuracy (vs the target argmax), NLL, Brier, ECE (15 bins, top-1 confidence);
  score also MAE of the expected level.
- noul: accuracy at 0.5, AUROC, ECE (15 bins over p), Brier; plus recall at 0.5 (destructive).
- text_span exact match (overall and when gold != none); target top-1 when gold != none and
  none-recall; intent accuracy split by meta.prefix (complete vs prefix examples).
- v2: by_kbucket (choice/score per K-bucket 2/3-5/6-10/11-30/31-100/101-255, noul) and by_source
  (bucket/source of v2 rows), so ECE <= 0.05 per bucket (PLAN §3.5) can be checked on dev.

v2 data: `--stream data/v2 --split dev_mix` reads the indexed raw/**/*.jsonl.zst rows of that split.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from jev_local.engine.encoder.calibrate import DEFAULT_CALIBRATION, CalibRecord, apply_calibration, bucket_key, header_key
from jev_local.engine.encoder.engine import load_checkpoint, pick_device
from jev_local.engine.encoder.heads import build_plan
from jev_local.engine.encoder.tokenize_pack import Packer, collate_tree
from jev_local.train.train import DataStats, JsonlIndex, Row, encode_example, make_batches, train_buckets

N_BINS = 15


@dataclass
class RecMeta:
    example_id: str
    prefix: bool
    labels: tuple[str, ...]
    source: str = ""  # "<bucket>/<source>" of v2 rows, "" for v1 data


@dataclass
class Collected:
    records: list[CalibRecord] = field(default_factory=list)
    metas: list[RecMeta] = field(default_factory=list)
    calib: dict = field(default_factory=lambda: dict(DEFAULT_CALIBRATION))
    n_examples: int = 0
    dropped_qids: int = 0  # questions whose state + block exceeds max_len


class StreamExamples:
    """v2 rows of one split as an indexable source (same interface as JsonlIndex)."""

    def __init__(self, roots: Sequence[str | Path], split: str, cache: str | Path = "runs/stream_cache",
                 limit: int | None = None):
        import numpy as np

        from jev_local.train.stream import SPLITS, Corpus

        self.corpus = Corpus.open(roots, cache)
        code = SPLITS.index(split) if split in SPLITS else len(SPLITS)
        ids = np.nonzero(self.corpus.split == code)[0]
        if limit is not None:  # per file, like JsonlIndex(limit=...)
            ids = np.concatenate([ids[self.corpus.file[ids] == f][:limit] for f in np.unique(self.corpus.file[ids])]) \
                if len(ids) else ids
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def length(self, i: int) -> int:
        return int(self.corpus.nbytes[self.ids[i]])

    def get(self, i: int) -> dict:
        return self.corpus.row(int(self.ids[i]))

    def close(self) -> None:
        self.corpus.close()


def example_source(data: Path | None, extra: Path | None, stream: Sequence[str] | None, split: str,
                   cache: str = "runs/stream_cache") -> list[Path] | tuple:
    if stream:
        return ("stream", tuple(stream), split, cache)
    paths = [d / f"{split}.jsonl" for d in [data, extra] if d is not None and (d / f"{split}.jsonl").is_file()]
    if not paths:
        raise SystemExit(f"no {split}.jsonl found")
    return paths


def _open_source(src, limit: int | None):
    if isinstance(src, tuple) and src and src[0] == "stream":
        return StreamExamples(src[1], src[2], src[3], limit=limit)
    return JsonlIndex(src, limit=limit)


@torch.inference_mode()
def collect_from_model(enc, heads, tok, paths, limit: int | None = None, batch: int = 8,
                       max_len: int = 2048, device: torch.device | str = "cpu") -> Collected:
    """Run the model over every example and keep raw (tau = 1) logits of every supervised question.
    `paths`: jsonl files (v1) or an `example_source(...)` stream spec."""
    device = torch.device(device)
    was_training = enc.training
    enc.eval()
    heads.eval()
    index = _open_source(paths, limit)
    packer = Packer(tok, max_len=max_len, buckets=train_buckets(max_len))
    res = Collected(n_examples=len(index))
    lengths = [index.length(i) for i in range(len(index))]
    stats = DataStats()
    try:
        for idxs in make_batches(lengths, batch, seed=0):
            rows: list[Row] = []
            srcs: list[str] = []
            for i in idxs:
                ex = index.get(i)
                got = encode_example(ex, packer, max_len, stats)
                rows += got
                srcs += [f"{ex.get('bucket', '')}/{ex.get('source', '')}" if ex.get("source") else ""] * len(got)
            keep = [j for j, r in enumerate(rows) if r.targets]
            rows, srcs = [rows[j] for j in keep], [srcs[j] for j in keep]
            if not rows:
                continue
            packs = [r.pack for r in rows]
            b = collate_tree(packs, packer.pad_id, enc.window, device)
            plan = build_plan(packs, b.row_offsets, device)
            out = heads(enc(b), plan)
            host = {k: (getattr(out, k).float().cpu().numpy() if getattr(out, k) is not None else None)
                    for k in ("choice", "score", "noul")}
            for ref in plan.refs:
                row = rows[ref.row]
                t = row.targets.get(ref.qi.qid)
                if t is None:
                    continue
                k = ref.qi.kind
                if k == "noul":
                    z = host["noul"][ref.group : ref.group + 1]
                else:
                    z = host[k][ref.group, : len(ref.qi.labels)]
                res.records.append(CalibRecord(ref.qi.qid, k, header_key(ref.qi.header), z.astype(np.float64),
                                               np.asarray(t.dist, dtype=np.float64)))
                res.metas.append(RecMeta(row.example_id, bool(row.meta.get("prefix", False)), ref.qi.labels,
                                         srcs[ref.row]))
    finally:
        res.dropped_qids = stats.dropped_qids
        index.close()
        if was_training:
            enc.train()
            heads.train()
    return res


def collect(ckpt: Path, paths, limit: int | None = None, batch: int = 8,
            device: str | None = None, max_len: int | None = None) -> Collected:
    dev = pick_device(device)
    enc, heads, tok, calib, meta = load_checkpoint(ckpt, dev, torch.float32, encoder="banded")
    res = collect_from_model(enc, heads, tok, paths, limit, batch, int(max_len or meta.get("max_len", 2048)), dev)
    res.calib = calib
    return res


# ---------------------------------------------------------------------------- metrics


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error: sum_b |B|/N * |mean(correct) - mean(conf)| over equal-width bins."""
    if len(conf) == 0:
        return float("nan")
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            tot += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(tot)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    pos, neg = labels.sum(), (1 - labels).sum()
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    s = scores[order]
    i = 0
    while i < len(s):  # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _choice_metrics(P: list[np.ndarray], T: list[np.ndarray]) -> dict:
    top = np.array([p.argmax() for p in P])
    gold = np.array([t.argmax() for t in T])
    conf = np.array([p.max() for p in P])
    correct = (top == gold).astype(np.float64)
    nll = np.array([-(t * np.log(np.clip(p, 1e-12, 1))).sum() for p, t in zip(P, T)])
    brier = np.array([((p - t) ** 2).sum() for p, t in zip(P, T)])
    return {"n": len(P), "acc": float(correct.mean()), "nll": float(nll.mean()), "brier": float(brier.mean()),
            "ece": ece(conf, correct)}


def _noul_metrics(p: np.ndarray, t: np.ndarray) -> dict:
    y = (t >= 0.5).astype(np.float64)
    pred = (p >= 0.5).astype(np.float64)
    m = {"n": int(len(p)), "acc": float((pred == y).mean()), "brier": float(((p - t) ** 2).mean()),
         "nll": float(-(t * np.log(np.clip(p, 1e-12, 1)) + (1 - t) * np.log(np.clip(1 - p, 1e-12, 1))).mean()),
         "ece": ece(p, y), "auroc": auroc(p, y), "pos_rate": float(y.mean())}
    if y.sum() > 0:
        m["recall@0.5"] = float(pred[y == 1].mean())
    return m


def metrics_from(res: Collected, calib: dict | None = None) -> dict:
    calib = res.calib if calib is None else calib
    probs = [apply_calibration(r, calib) for r in res.records]
    by_qid: dict[str, list[int]] = {}
    for i, r in enumerate(res.records):
        by_qid.setdefault(r.qid, []).append(i)
    out: dict = {"n_examples": res.n_examples, "n_questions": len(res.records), "by_qid": {}, "by_kind": {}, "harness": {}}
    for qid, ii in sorted(by_qid.items()):
        kind = res.records[ii[0]].kind
        if kind == "noul":
            m = _noul_metrics(np.array([probs[i][0] for i in ii]), np.array([res.records[i].target[0] for i in ii]))
        else:
            m = _choice_metrics([probs[i] for i in ii], [res.records[i].target for i in ii])
            if kind == "score":
                ev = np.array([(probs[i] * np.arange(len(probs[i]))).sum() for i in ii])
                gold = np.array([(res.records[i].target * np.arange(len(probs[i]))).sum() for i in ii])
                m["mae"] = float(np.abs(ev - gold).mean())
        m["kind"] = kind
        out["by_qid"][qid] = m
    for kind in ("choice", "score", "noul"):
        ii = [i for i, r in enumerate(res.records) if r.kind == kind]
        if not ii:
            continue
        if kind == "noul":
            out["by_kind"][kind] = _noul_metrics(np.array([probs[i][0] for i in ii]),
                                                 np.array([res.records[i].target[0] for i in ii]))
        else:
            out["by_kind"][kind] = _choice_metrics([probs[i] for i in ii], [res.records[i].target for i in ii])
    out["harness"] = _harness_metrics(res, probs)
    out["by_kbucket"] = _grouped(res, probs, lambda r, m: bucket_key(r.kind, len(r.logits)))
    if any(m.source for m in res.metas):
        out["by_source"] = _grouped(res, probs, lambda r, m: f"{m.source}:{r.kind}")
    out["dropped_qids"] = res.dropped_qids
    return out


def _grouped(res: Collected, probs: list[np.ndarray], key) -> dict:
    groups: dict[str, list[int]] = {}
    for i, (r, m) in enumerate(zip(res.records, res.metas)):
        groups.setdefault(key(r, m), []).append(i)
    out = {}
    for g, ii in sorted(groups.items()):
        kind = res.records[ii[0]].kind
        if kind == "noul":
            out[g] = _noul_metrics(np.array([probs[i][0] for i in ii]), np.array([res.records[i].target[0] for i in ii]))
        else:
            out[g] = _choice_metrics([probs[i] for i in ii], [res.records[i].target for i in ii])
    return out


def _gold_label(res: Collected, i: int) -> str:
    return res.metas[i].labels[int(res.records[i].target.argmax())]


def _harness_metrics(res: Collected, probs: list[np.ndarray]) -> dict:
    h: dict = {}

    def acc(ii: list[int]) -> float | None:
        if not ii:
            return None
        return float(np.mean([probs[i].argmax() == res.records[i].target.argmax() for i in ii]))

    q = lambda name: [i for i, r in enumerate(res.records) if r.qid == name]  # noqa: E731
    span = q("text_span")
    if span:
        h["text_span_em"] = acc(span)
        h["text_span_em_nonnone"] = acc([i for i in span if _gold_label(res, i) != "none"])
    tgt = q("target")
    if tgt:
        h["target_top1_nonnone"] = acc([i for i in tgt if _gold_label(res, i) != "none"])
        h["target_none_recall"] = acc([i for i in tgt if _gold_label(res, i) == "none"])
    it = q("intent")
    if it:
        h["intent_acc_complete"] = acc([i for i in it if not res.metas[i].prefix])
        h["intent_acc_prefix"] = acc([i for i in it if res.metas[i].prefix])
        h["n_intent_complete"] = sum(1 for i in it if not res.metas[i].prefix)
        h["n_intent_prefix"] = sum(1 for i in it if res.metas[i].prefix)
    return h


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m jev_local.train.eval")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--extra", type=Path, default=None)
    ap.add_argument("--stream", nargs="+", default=None, help="v2 data roots (rows filtered by --split)")
    ap.add_argument("--stream-cache", default="runs/stream_cache")
    ap.add_argument("--max-len", type=int, default=None, help="default: the checkpoint's training max_len")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=None, help="default: runs/<ckpt name>/eval_<split>.json")
    ap.add_argument("--no-calib", action="store_true")
    args = ap.parse_args(argv)

    if args.data is None and not args.stream:
        raise SystemExit("--data or --stream is required")
    src = example_source(args.data, args.extra, args.stream, args.split, args.stream_cache)
    res = collect(args.ckpt, src, args.limit, args.batch, args.device, args.max_len)
    m = metrics_from(res, dict(DEFAULT_CALIBRATION) if args.no_calib else None)
    files = [str(p) for p in src] if isinstance(src, list) else list(src[1])
    m.update(ckpt=str(args.ckpt), split=args.split, files=files, calibrated=not args.no_calib)
    out = args.out or Path("runs") / args.ckpt.name / f"eval_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=2))
    print(json.dumps({"out": str(out), "by_kind": m["by_kind"], "harness": m["harness"]}, indent=2))


if __name__ == "__main__":
    main()
