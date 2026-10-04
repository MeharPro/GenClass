"""Train the fast encoder (ettin + decision heads) on CONTRACT "D" jsonl examples.

v1 (index mode, unchanged):
    python -m jev_local.train.train --data data/cu --extra data/gen --out models/jev-local-fast \
        --epochs 1 --batch 4 --grad-accum 4 --max-len 1536 --lr 5e-5 --head-lr 1e-3 [--max-steps K] \
        [--limit N] [--resume]
    python -m jev_local.train.train --init-only --out models/jev-local-fast-init

v2 (stream mode over data/v2 raw/*.jsonl.zst or assembled shards; see jev_local/train/stream.py):
    python -m jev_local.train.train --stream data/v2 --mixture mix.json --batch-tokens 16384 --grad-accum 8 \
        --max-len 8192 --base models/base/ettin-encoder-68m --out models/jev-local-fast-v2 --loss v2 \
        --device cpu --amp --no-grad-ckpt --threads 32 [--ddp] [--resume]

Multi-node CPU (E2): launch the same command under torchrun on every node with --ddp (jev_local/train/ddp.py).

Design notes
- Index mode reads examples lazily through a byte-offset index; line byte length is the length proxy for
  length-bucketed batching. Stream mode reads an indexed mixture (bucket weights, per-source caps, sqrt(n)
  source sampling, length curriculum) with token-budget micro-batches.
- Questions are isolated by the mask, so an example longer than --max-len is split into several
  sequences that each repeat the state (identical per-question results, nothing dropped).
- The trainer state (weights + optimizer + data cursor) lives in runs/<name>/ckpt and is swapped in
  atomically; the servable export is written to --out the same way. SIGINT/SIGTERM finish the
  current optimizer step (single process) or the current micro-batch (DDP: all ranks agree, the partial
  step is dropped), checkpoint and exit 130; SIGKILL loses at most --ckpt-every steps.
- DDP: the data cursor counts global micro-batches, so it does not depend on the number of ranks; rank r
  takes micro-batches c+r of each group of `world`. One all-reduce per optimizer step (ddp.GradReducer).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import resource
import shutil
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import torch
from safetensors.torch import load_file, save_file

from jev_local.engine.encoder.engine import init_model, pick_device, write_checkpoint
from jev_local.engine.encoder.heads import DecisionHeads, HeadPlan, build_plan
from jev_local.engine.encoder.model import BASE_MODEL, MaskedEncoder
from jev_local.engine.encoder.tokenize_pack import BUCKETS, Batch, Packed, Packer, TreeBatch, collate_tree
from jev_local.schema import question_from_json
from jev_local.serialize import QBlock, question_block, state_segments
from jev_local.train import ddp
from jev_local.train.losses import LossConfig, Targets, compute_loss

LONG_BUCKETS = (3072, 4096, 6144, 8192)  # dense padding buckets beyond the v1 2048 (ModernBERT is 8k-native)
MAX_TRAIN_LEN = 8192

# ---------------------------------------------------------------------------- data


class JsonlIndex:
    """Byte offsets of every line of one or more jsonl files."""

    def __init__(self, paths: Sequence[Path], limit: int | None = None):
        self.paths = [Path(p) for p in paths]
        self.entries: list[tuple[int, int, int]] = []  # (file, offset, nbytes)
        for fi, p in enumerate(self.paths):
            n = 0
            with p.open("rb") as f:
                off = 0
                for line in f:
                    if line.strip():
                        self.entries.append((fi, off, len(line)))
                        n += 1
                        if limit is not None and n >= limit:
                            break
                    off += len(line)
        self._fh: dict[int, Any] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def length(self, i: int) -> int:
        return self.entries[i][2]

    def get(self, i: int) -> dict:
        fi, off, n = self.entries[i]
        fh = self._fh.get(fi)
        if fh is None:
            fh = self._fh[fi] = self.paths[fi].open("rb")
        fh.seek(off)
        return json.loads(fh.read(n))

    def close(self) -> None:
        for fh in self._fh.values():
            fh.close()
        self._fh.clear()


@dataclass
class QTarget:
    dist: list[float]  # choice/score: over labels; noul: [p]
    hard: bool


@dataclass
class Row:
    pack: Packed
    targets: dict[str, QTarget]  # supervised qids in this pack
    example_id: str
    meta: dict


@dataclass
class DataStats:
    examples: int = 0
    rows: int = 0
    dropped_qids: int = 0  # state + that question alone exceeded max_len
    bad_labels: int = 0  # label not among the question's labels / wrong type
    unsupervised_examples: int = 0


def parse_target(block: QBlock, lab: dict | None) -> QTarget | None:
    if not lab:
        return None
    kind = lab.get("type", block.kind)
    if kind != block.kind:
        return None
    K = len(block.labels)
    if kind == "noul":
        p = lab.get("p")
        return None if p is None else QTarget([float(p)], hard=float(p) in (0.0, 1.0))
    if kind == "choice":
        if "label" in lab:
            if lab["label"] not in block.labels:
                return None
            d = [0.0] * K
            d[block.labels.index(lab["label"])] = 1.0
            return QTarget(d, hard=True)
        if isinstance(lab.get("dist"), dict):
            d = [float(lab["dist"].get(x, 0.0)) for x in block.labels]
            s = sum(d)
            return QTarget([x / s for x in d], hard=False) if s > 0 else None
        return None
    if kind == "score":
        if "level" in lab:
            k = int(lab["level"])
            if not 0 <= k < K:
                return None
            d = [0.0] * K
            d[k] = 1.0
            return QTarget(d, hard=True)
        if isinstance(lab.get("dist"), list) and len(lab["dist"]) == K:
            d = [float(x) for x in lab["dist"]]
            s = sum(d)
            return QTarget([x / s for x in d], hard=False) if s > 0 else None
    return None


def encode_example(obj: dict, packer: Packer, max_len: int, stats: DataStats | None = None) -> list[Row]:
    questions = {qid: question_from_json(q) for qid, q in obj["questions"].items()}
    blocks = [question_block(qid, q) for qid, q in questions.items()]
    labels = obj.get("labels") or {}
    targets: dict[str, QTarget] = {}
    for b in blocks:
        t = parse_target(b, labels.get(b.qid))
        if t is not None:
            targets[b.qid] = t
        elif b.qid in labels and stats is not None:
            stats.bad_labels += 1
    packs, dropped = packer.pack_split(state_segments(obj["state"]), blocks, max_len)
    if stats is not None:
        stats.examples += 1
        stats.dropped_qids += len(dropped)
        stats.unsupervised_examples += int(not targets)
    rows = []
    meta = obj.get("meta") or {}
    for p in packs:
        tg = {q: targets[q] for q in p.q_index if q in targets}
        rows.append(Row(p, tg, obj.get("id", ""), meta))
    if stats is not None:
        stats.rows += len(rows)
    return rows


def example_tokens(obj: dict, packer: Packer, max_len: int) -> int:
    """Packed tokens of one example (all its rows), for stream length calibration."""
    try:
        return sum(r.pack.length for r in encode_example(obj, packer, max_len))
    except Exception:
        return 0


def make_batches(lengths: Sequence[int], batch: int, seed: int, pool: int = 64) -> list[list[int]]:
    """Length-bucketed batches: shuffle, sort within pools of `pool` batches, shuffle the batches."""
    rng = random.Random(seed)
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    out: list[list[int]] = []
    span = batch * pool
    for s in range(0, len(idx), span):
        chunk = sorted(idx[s : s + span], key=lambda i: lengths[i])
        out += [chunk[j : j + batch] for j in range(0, len(chunk), batch)]
    rng.shuffle(out)
    return out


def _pair_refs(rows: Sequence[Row], plan: HeadPlan) -> dict[str, list]:
    """Consistency pairs declared in example meta (`meta.pairs`, see losses.py) -> group index lists.
    Rows of one example share the same meta object, so a pair may span two packs of the same example."""
    loc: dict[tuple[int, str], Any] = {}
    for r in plan.refs:
        loc[(id(rows[r.row].meta), r.qi.qid)] = r
    out: dict[str, list] = {"neg": [], "nc": [], "sc": []}
    seen: set[int] = set()
    for row in rows:
        m = row.meta
        if id(m) in seen or not m.get("pairs"):
            continue
        seen.add(id(m))
        for pr in m["pairs"]:
            if not isinstance(pr, dict):
                continue
            kind, key = pr.get("kind"), id(m)
            try:
                if kind == "neg":
                    a, b = loc.get((key, pr["a"])), loc.get((key, pr["b"]))
                    if a and b and a.qi.kind == b.qi.kind == "noul":
                        out["neg"].append((a.group, b.group))
                elif kind == "noul_choice":
                    n, c = loc.get((key, pr["noul"])), loc.get((key, pr["choice"]))
                    if n and c and n.qi.kind == "noul" and c.qi.kind == "choice" and pr["yes"] in c.qi.labels:
                        out["nc"].append((n.group, c.group, c.qi.labels.index(pr["yes"])))
                elif kind == "score_choice":
                    s, c = loc.get((key, pr["score"])), loc.get((key, pr["choice"]))
                    if s and c and s.qi.kind == "score" and c.qi.kind == "choice" and len(pr["map"]) == len(s.qi.labels) \
                            and all(k in c.qi.labels for k in pr["map"]):
                        out["sc"].append((s.group, c.group, [c.qi.labels.index(k) for k in pr["map"]]))
            except (KeyError, TypeError):
                continue
    return out


def build_targets(rows: Sequence[Row], plan: HeadPlan, device: torch.device) -> tuple[Targets, int]:
    """Align per-question targets with the plan's groups. Returns (targets, n supervised questions)."""
    tg = Targets()
    n_sup = 0
    for kind in ("choice", "score"):
        gi = getattr(plan, kind)
        refs = plan.of_kind(kind)  # type: ignore[arg-type]
        idx, dists, hard = [], [], []
        for r in refs:
            t = rows[r.row].targets.get(r.qi.qid)
            if t is None:
                continue
            idx.append(r.group)
            dists.append(t.dist)
            hard.append(t.hard)
        if not idx or gi is None:
            continue
        kmax = gi.items.shape[1]
        T = torch.zeros((len(idx), kmax), dtype=torch.float32)
        for j, d in enumerate(dists):
            T[j, : len(d)] = torch.tensor(d)
        setattr(tg, f"{kind}_idx", torch.tensor(idx, device=device))
        setattr(tg, kind, T.to(device))
        setattr(tg, f"{kind}_hard", torch.tensor(hard, device=device))
        n_sup += len(idx)
    idx, ps = [], []
    for r in plan.of_kind("noul"):
        t = rows[r.row].targets.get(r.qi.qid)
        if t is not None:
            idx.append(r.group)
            ps.append(t.dist[0])
    if idx:
        tg.noul_idx = torch.tensor(idx, device=device)
        tg.noul = torch.tensor(ps, dtype=torch.float32, device=device)
        n_sup += len(idx)
    if any(r.meta.get("pairs") for r in rows):
        pr = _pair_refs(rows, plan)
        if pr["neg"]:
            tg.neg_a = torch.tensor([a for a, _ in pr["neg"]], device=device)
            tg.neg_b = torch.tensor([b for _, b in pr["neg"]], device=device)
        if pr["nc"]:
            tg.nc_noul = torch.tensor([x[0] for x in pr["nc"]], device=device)
            tg.nc_choice = torch.tensor([x[1] for x in pr["nc"]], device=device)
            tg.nc_yes = torch.tensor([x[2] for x in pr["nc"]], device=device)
        if pr["sc"]:
            k = max(len(x[2]) for x in pr["sc"])
            mp = torch.zeros((len(pr["sc"]), k), dtype=torch.long)
            mk = torch.zeros((len(pr["sc"]), k), dtype=torch.bool)
            for j, (_, _, m) in enumerate(pr["sc"]):
                mp[j, : len(m)] = torch.tensor(m)
                mk[j, : len(m)] = True
            tg.sc_score = torch.tensor([x[0] for x in pr["sc"]], device=device)
            tg.sc_choice = torch.tensor([x[1] for x in pr["sc"]], device=device)
            tg.sc_map, tg.sc_mask = mp.to(device), mk.to(device)
    return tg, n_sup


def train_buckets(max_len: int) -> tuple[int, ...]:
    b = tuple(x for x in BUCKETS + LONG_BUCKETS if x < max_len)
    return b + (max_len,)


def prepare_batch(
    rows: Sequence[Row], packer: Packer, device: torch.device, window: int, attn: str = "tree",
    chunk: int | None = None,
) -> tuple[Batch | TreeBatch, HeadPlan, Targets, int]:
    packs = [r.pack for r in rows]
    if attn == "tree":
        kw = {"chunk": chunk} if chunk else {}
        batch: Batch | TreeBatch = collate_tree(packs, packer.pad_id, window, device, **kw)
        plan = build_plan(packs, batch.row_offsets, device)
    else:
        batch = packer.collate(packs, device)
        plan = build_plan(packs, batch.shape[1], device)
    tg, n_sup = build_targets(rows, plan, device)
    return batch, plan, tg, n_sup


def padded_tokens(batch: Batch | TreeBatch) -> int:
    return int(batch.input_ids.numel())


# ---------------------------------------------------------------------------- micro-batch sources


@dataclass
class Micro:
    epoch: int  # epoch (index mode) or mixture pass (stream mode)
    cursor: int  # global micro-batch cursor after this micro-step (resume point if a step ends here)
    examples: list[dict]
    rows: list[Row] = field(default_factory=list)
    cost: float = 0.0  # stream mode: the mixture's estimated cost of this micro-batch (timing log only)


class IndexSource:
    """v1 data order (make_batches per epoch), shared by `world` ranks: rank r takes batch c+r of each group."""

    mode = "index"

    def __init__(self, index: JsonlIndex, batch: int, seed: int, d: ddp.Dist):
        self.index, self.batch, self.seed, self.d = index, batch, seed, d
        self.lengths = [index.length(i) for i in range(len(index))]

    def n_batches(self) -> int:
        return math.ceil(len(self.index) / self.batch)

    def total_steps(self, grad_accum: int, epochs: int) -> int:
        usable = self.n_batches() - (self.n_batches() % self.d.world if self.d.world > 1 else 0)
        return max(1, usable // (grad_accum * self.d.world)) * epochs

    def iterate(self, start_epoch: int, start_cursor: int) -> Iterator[Micro]:
        w, r = self.d.world, self.d.rank
        epoch = start_epoch
        while True:
            order = make_batches(self.lengths, self.batch, seed=self.seed * 100_003 + epoch)
            last = len(order) if w == 1 else len(order) - len(order) % w
            if last < w:
                return
            start = start_cursor if epoch == start_epoch else 0
            start -= start % w
            for bi in range(start, last, w):
                yield Micro(epoch, bi + w, [self.index.get(i) for i in order[bi + r]])
            epoch += 1

    def close(self) -> None:
        self.index.close()


class StreamSource:
    mode = "stream"

    def __init__(self, mix, d: ddp.Dist, grad_accum: int, cycle: bool):
        self.mix, self.d, self.grad_accum, self.cycle = mix, d, grad_accum, cycle

    def total_steps(self) -> int:
        return self.mix.total_steps(self.d.world, self.grad_accum)

    def iterate(self, start_epoch: int, start_cursor: int) -> Iterator[Micro]:
        corpus = self.mix.corpus
        for k, cur, ids in self.mix.iterate(self.d.rank, self.d.world, self.grad_accum, start_epoch, start_cursor,
                                             cycle=self.cycle):
            yield Micro(k, cur, [corpus.row(int(i)) for i in ids], cost=self.mix.batch_cost(ids))

    def close(self) -> None:
        self.mix.corpus.close()


class Prefetch:
    """Read + tokenize micro-batches in a background thread (same order, same results)."""

    _END = object()

    def __init__(self, it: Iterator[Micro], fn: Callable[[Micro], Micro], depth: int):
        self.q: queue.Queue = queue.Queue(maxsize=max(1, depth))
        self.stop = threading.Event()
        self.it, self.fn = it, fn
        self.t = threading.Thread(target=self._run, name="prefetch", daemon=True)
        self.t.start()

    def _put(self, x) -> bool:
        while not self.stop.is_set():
            try:
                self.q.put(x, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        try:
            for m in self.it:
                if not self._put(self.fn(m)):
                    return
        except BaseException as e:  # surfaced in the main thread
            self._put(("__error__", e))
            return
        self._put(self._END)

    def __iter__(self) -> Iterator[Micro]:
        while True:
            x = self.q.get()
            if x is self._END:
                return
            if isinstance(x, tuple) and len(x) == 2 and x[0] == "__error__":
                raise x[1]
            yield x

    def close(self) -> None:
        self.stop.set()


# ---------------------------------------------------------------------------- optimisation


def param_groups(enc: MaskedEncoder, heads: DecisionHeads, lr: float, head_lr: float, wd: float) -> list[dict]:
    decay, no_decay = [], []
    for n, p in enc.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "norm" in n or "embeddings" in n else decay).append(p)
    return [
        {"params": decay, "lr": lr, "weight_decay": wd, "base_lr": lr},
        {"params": no_decay, "lr": lr, "weight_decay": 0.0, "base_lr": lr},
        {"params": list(heads.parameters()), "lr": head_lr, "weight_decay": 0.0, "base_lr": head_lr},
    ]


def lr_factor(step: int, total: int, warmup_frac: float = 0.03, floor: float = 0.0) -> float:
    warm = max(1, int(round(total * warmup_frac)))
    if step < warm:
        return (step + 1) / warm
    prog = min(1.0, (step - warm) / max(1, total - warm))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))


def rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3  # bytes on macOS, KiB on Linux


def mps_mb() -> float:
    try:
        return torch.mps.driver_allocated_memory() / 1e6 if torch.backends.mps.is_available() else 0.0
    except Exception:
        return 0.0


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def amp_dtype(device: torch.device) -> torch.dtype | None:
    """Autocast dtype for --amp, or None when mixed precision isn't safe without a grad scaler."""
    if device.type == "mps":
        return torch.float16
    if device.type == "cpu":
        return torch.bfloat16  # AVX512-BF16 / AMX on server CPUs (e.g. Azure Dasv7)
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


# ---------------------------------------------------------------------------- checkpoints


def save_state(state_dir: Path, enc: MaskedEncoder, heads: DecisionHeads, opt: torch.optim.Optimizer, info: dict,
               rng: torch.Tensor | None = None) -> None:
    tmp = state_dir.parent / f".{state_dir.name}.tmp-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    sd = {f"enc.{k}": v.detach().cpu().contiguous() for k, v in enc.backbone.state_dict().items()}
    sd.update({f"heads.{k}": v.detach().cpu().contiguous() for k, v in heads.state_dict().items()})
    save_file(sd, str(tmp / "model.safetensors"))
    osd = opt.state_dict()
    blob = {"state": {k: {n: (t.cpu() if torch.is_tensor(t) else t) for n, t in v.items()} for k, v in osd["state"].items()},
            "param_groups": osd["param_groups"]}
    if rng is not None:
        blob["rng"] = rng
    torch.save(blob, tmp / "optim.pt")
    (tmp / "trainer.json").write_text(json.dumps(info, indent=2))
    old = state_dir.parent / f".{state_dir.name}.old-{os.getpid()}"
    if state_dir.exists():
        os.replace(state_dir, old)
    os.replace(tmp, state_dir)
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)


def load_state(state_dir: Path, enc: MaskedEncoder, heads: DecisionHeads) -> tuple[dict, dict]:
    sd = load_file(str(state_dir / "model.safetensors"))
    enc.backbone.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("enc.")})
    heads.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("heads.")})
    info = json.loads((state_dir / "trainer.json").read_text())
    osd = torch.load(state_dir / "optim.pt", map_location="cpu", weights_only=True)
    return info, osd


def load_weights(path: Path, enc: MaskedEncoder, heads: DecisionHeads) -> None:
    """--init-from: weights only, from a servable checkpoint (backbone/ + heads.safetensors) or a trainer
    state dir (model.safetensors with enc./heads. prefixes)."""
    path = Path(path)
    if (path / "model.safetensors").is_file() and not (path / "backbone").is_dir():
        sd = load_file(str(path / "model.safetensors"))
        enc.backbone.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("enc.")})
        heads.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("heads.")})
        return
    bb = path / "backbone"
    files = sorted(bb.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"--init-from {path}: no backbone/*.safetensors or model.safetensors")
    sd: dict = {}
    for f in files:
        sd.update(load_file(str(f)))
    missing, unexpected = enc.backbone.load_state_dict(sd, strict=False)
    bad = [k for k in missing if "num_batches_tracked" not in k]
    if bad:
        raise RuntimeError(f"--init-from {path}: backbone keys missing: {bad[:5]}")
    heads.load_state_dict(load_file(str(path / "heads.safetensors")))


# ---------------------------------------------------------------------------- main loop


class _Stop:
    """First SIGINT/SIGTERM: finish (or, under DDP, drop) the current step, checkpoint, exit 130. A second
    signal more than `grace` seconds later aborts immediately. Duplicates inside the grace window are
    ignored: `kill -TERM -<pgid>` / torchrun deliver the same SIGTERM to a worker twice."""

    def __init__(self, grace: float = 10.0) -> None:
        self.flag = False
        self.grace = grace
        self.t_first = 0.0

    def install(self) -> None:
        def handler(signum, frame):  # noqa: ARG001
            now = time.monotonic()
            if self.flag:
                if now - self.t_first < self.grace:
                    return
                raise KeyboardInterrupt  # deliberate second signal: give up immediately
            self.flag = True
            self.t_first = now
            print(f"[train] signal {signum}: finishing this step, checkpointing, exiting", file=sys.stderr)

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)


def data_paths(dirs: Sequence[Path | None], split: str) -> list[Path]:
    out = []
    for d in dirs:
        if d is None:
            continue
        p = Path(d) / f"{split}.jsonl"
        if p.is_file():
            out.append(p)
    return out


PART_KEYS = ("choice", "score", "noul", "x_cons", "x_noise")
# per-step counters summed across ranks in DDP (appended to the gradient all-reduce)
EXTRA_KEYS = ("tokens", "padded", "examples", "dropped_qids", "bad_labels", "stop",
              *(f"{k}_{x}" for k in PART_KEYS for x in ("s", "n")))


def loss_config(args: argparse.Namespace) -> LossConfig:
    if args.loss == "v2":
        cfg = LossConfig.v2()
    else:
        cfg = LossConfig()
    over = {"smoothing": args.smoothing, "brier": args.brier, "rps": args.rps, "kl": args.kl,
            "spherical": args.spherical, "coral": args.coral, "noise_samples": args.noise_samples,
            "noise_weight": args.noise_weight, "consistency": args.consistency}
    for k, v in over.items():
        if v is not None:
            setattr(cfg, k, v)
    if args.noise_sigma:
        vals = [float(x) for x in str(args.noise_sigma).split(",")]
        cfg.noise_sigma = (vals[0], vals[-1])
    return cfg


def parse_curriculum(s: str | None) -> tuple[tuple[float, int], ...] | None:
    if not s:
        return None
    out = []
    for part in s.split(","):
        p, v = part.split(":")
        out.append((float(p), int(v)))
    return tuple(out)


def build_stream(args: argparse.Namespace, packer: Packer, d: ddp.Dist, run_dir: Path):
    from jev_local.train.stream import Corpus, Mixture, MixConfig

    cache = Path(args.stream_cache)
    if d.is_local_main:
        Corpus.open(args.stream, cache, workers=args.index_workers)
    ddp.barrier(d)
    corpus = Corpus.open(args.stream, cache, workers=1, build=False)
    over = {"seed": args.seed, "source_cap": args.source_cap, "class_cap": args.class_cap, "alpha": args.alpha,
            "passes": args.passes, "pool": args.pool, "batch_tokens": args.batch_tokens, "batch_rows": args.batch_rows,
            "max_len": args.max_len, "curriculum": parse_curriculum(args.curriculum),
            "licenses": tuple(args.license.split(",")) if args.license else None,
            "buckets": tuple(args.buckets.split(",")) if args.buckets else None,
            "splits": tuple(args.splits.split(",")) if args.splits else None,
            "layout": args.batch_layout, "balance": args.balance}
    cfg = MixConfig.from_json(args.mixture or {}, **over)
    if cfg.batch_rows:
        cfg.batch_tokens = None
    if args.calibrate_lengths:  # deterministic sample + same tokenizer on every rank -> same batches
        corpus.calibrate_lengths(lambda ex: example_tokens(ex, packer, args.max_len), per_file=args.calibrate_lengths,
                                 seed=args.seed)
    mix = Mixture(corpus, cfg)
    if d.is_main:
        (run_dir / "mixture_plan.json").write_text(json.dumps({"corpus": corpus.summary(), "plan": mix.plan_summary(0)},
                                                             indent=1))
    return mix


def train(args: argparse.Namespace) -> dict:
    device = pick_device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    d = ddp.init_from_env(device, args.ddp_timeout) if args.ddp else ddp.NO_DIST
    torch.manual_seed(args.seed)
    out = Path(args.out)
    enc, heads, tok = init_model(args.base, seed=args.seed, encoder=args.encoder)
    if args.init_only:
        if d.is_main:
            write_checkpoint(out, enc, heads, tok, meta={"trained": False, "init_seed": args.seed, "base": args.base})
            print(json.dumps({"init_checkpoint": str(out)}))
        ddp.destroy(d)
        return {"init_checkpoint": str(out)}
    if d.rank > 0:
        torch.manual_seed(args.seed + 7919 * d.rank)  # independent logit-noise draws per rank

    run_dir = Path(args.runs_dir) / (args.run_name or out.name)
    state_dir = run_dir / "ckpt"
    if d.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
    if args.max_len > MAX_TRAIN_LEN:
        raise SystemExit(f"--max-len {args.max_len} > {MAX_TRAIN_LEN} (ModernBERT/ettin context)")
    packer = Packer(tok, max_len=args.max_len, buckets=train_buckets(args.max_len))

    if args.stream:
        mix = build_stream(args, packer, d, run_dir)
        source: IndexSource | StreamSource = StreamSource(mix, d, args.grad_accum, cycle=bool(args.max_steps))
    else:
        paths = data_paths([args.data, args.extra], "train")
        if not paths:
            raise SystemExit(f"no train.jsonl under {args.data} / {args.extra}")
        source = IndexSource(JsonlIndex(paths, limit=args.limit), args.batch, args.seed, d)

    info: dict = {"step": 0, "epoch": 0, "cursor": 0, "micro": 0}
    osd = None
    resumed = False
    if d.is_main:
        if args.resume and (state_dir / "trainer.json").is_file():
            info, osd = load_state(state_dir, enc, heads)
            resumed = True
            print(f"[train] resumed from {state_dir} at step {info['step']} (epoch {info['epoch']}, batch {info['cursor']})",
                  file=sys.stderr)
        elif args.init_from:
            load_weights(Path(args.init_from), enc, heads)
            print(f"[train] initialised weights from {args.init_from}", file=sys.stderr)
    info, resumed = ddp.broadcast_object(d, (info, resumed))
    if resumed and info.get("data_mode", "index") != source.mode:
        print(f"[train] checkpoint data mode {info.get('data_mode', 'index')!r} != {source.mode!r}: "
              "keeping weights/optimizer/step, restarting the data order", file=sys.stderr)
        info.update(epoch=0, cursor=0)

    enc.to(device).train()
    heads.to(device).train()
    enc.grad_ckpt = args.grad_ckpt
    opt = torch.optim.AdamW(param_groups(enc, heads, args.lr, args.head_lr, args.weight_decay), betas=(0.9, 0.98), eps=1e-6)
    if osd is not None:
        opt.load_state_dict({k: v for k, v in osd.items() if k != "rng"})
        if "rng" in osd and not d.enabled:
            torch.set_rng_state(osd["rng"])
        del osd
    reducer = None
    if d.enabled:
        ddp.broadcast_tensors(d, ddp.module_tensors([enc.backbone, heads]))
        if resumed:
            ddp.broadcast_optimizer(d, opt)
        reducer = ddp.GradReducer(d, list(enc.parameters()) + list(heads.parameters()), n_extra=len(EXTRA_KEYS),
                                  compress=args.ddp_compress)

    if isinstance(source, StreamSource):
        total = args.max_steps or source.total_steps()
    elif args.max_steps and not args.epochs_given:
        total = args.max_steps  # loop over the data as many times as needed
    else:
        total = source.total_steps(args.grad_accum, args.epochs)
        if args.max_steps:
            total = min(total, args.max_steps)
    loss_cfg = loss_config(args)
    amp_dt = amp_dtype(device) if args.amp else None
    use_amp = amp_dt is not None
    stop = _Stop()
    stop.install()
    stats = DataStats()
    log_f = (run_dir / "log.jsonl").open("a") if d.is_main else None
    tlog = None  # --timing-log: per-rank, per-step compute / wait / all-reduce seconds (DDP straggler analysis)
    if args.timing_log:
        (run_dir / "timing").mkdir(parents=True, exist_ok=True)
        tlog = (run_dir / "timing" / f"rank{d.rank}.jsonl").open("a")
    step_micro: list[list[float]] = []  # this step: [feed_wait_s, compute_s, tokens, padded, rows, est_cost] per micro
    step_flag_s = 0.0
    micro_stop = args.stop_check == "micro"
    lagged = ddp.LaggedFlag(d) if (d.enabled and not micro_stop) else None
    meta_common = {"base": args.base, "max_len": args.max_len, "run": str(run_dir)}
    params = list(enc.parameters()) + list(heads.parameters())

    def checkpoint(final: bool = False) -> None:
        if not d.is_main:
            return
        sync(device)
        save_state(state_dir, enc, heads, opt,
                   {**info, "data_mode": source.mode, "world": d.world, "grad_accum": args.grad_accum,
                    "args": vars_json(args), "time": time.time()},
                   rng=None if d.enabled else torch.get_rng_state())
        write_checkpoint(out, enc, heads, tok, meta={**meta_common, "trained": True, "step": info["step"], "final": final})

    def encode(m: Micro) -> Micro:
        for ex in m.examples:
            m.rows += encode_example(ex, packer, args.max_len, stats)
        return m

    step = info["step"]
    if step >= total:
        print(f"[train] checkpoint already at step {step} >= total {total}", file=sys.stderr)
    it_src = source.iterate(info["epoch"], info["cursor"])
    prefetch = args.prefetch if args.prefetch is not None else (2 if isinstance(source, StreamSource) else 0)
    feed: Iterator[Micro] | Prefetch = Prefetch(it_src, encode, prefetch) if prefetch > 0 else (encode(m) for m in it_src)

    acc: dict[str, list] = {}  # window: key -> [sum, count] (tensors in single-process mode: no per-step sync)
    step_acc: dict[str, list] = {}  # this optimizer step (DDP: reduced across ranks at the step)
    win = {"tokens": 0.0, "padded": 0.0, "examples": 0.0, "steps": 0, "tokens_rank": 0.0, "sync_s": 0.0}
    step_cnt = {"tokens": 0, "padded": 0, "examples": 0}
    last_stats = {"dropped_qids": 0, "bad_labels": 0}
    glob_stats = {"dropped_qids": 0, "bad_labels": 0}
    win_t0 = t_start = time.perf_counter()
    micro = 0
    cur_epoch, epoch_step0, epoch_from_zero = info["epoch"], step, info["cursor"] == 0
    abandoned = False
    opt.zero_grad(set_to_none=True)
    t_mark = time.perf_counter()
    try:
        for m in (feed if step < total else ()):
            t_got = time.perf_counter()
            if m.epoch != cur_epoch:  # v1 guard: a whole epoch produced no optimizer step -> nothing to learn
                if epoch_from_zero and step == epoch_step0:
                    break
                cur_epoch, epoch_step0, epoch_from_zero = m.epoch, step, True
            rows = m.rows
            if not rows and not d.enabled:
                continue  # v1: an example batch with no packable rows is not a micro-batch
            if rows:
                batch, plan, tg, n_sup = prepare_batch(rows, packer, device, enc.window, args.attn)
                if n_sup or tg.n_pairs:
                    with torch.autocast(device.type, dtype=amp_dt or torch.float32, enabled=use_amp):
                        h = enc(batch)
                        hout = heads(h, plan)
                        # inside autocast so softmax/CE/log-sigmoid run in fp32 even when the logits are bf16
                        loss, parts = compute_loss(hout, tg, loss_cfg, progress=step / max(total, 1))
                    if loss is not None:
                        (loss / args.grad_accum).backward()
                        for k, (s, n) in parts.items():
                            a = step_acc.setdefault(k, [0.0, 0])
                            a[0] = a[0] + s
                            a[1] += n
                step_cnt["tokens"] += batch.n_tokens
                step_cnt["padded"] += padded_tokens(batch)
            step_cnt["examples"] += len(m.examples)
            micro += 1
            if tlog is not None:
                t_done = time.perf_counter()
                step_micro.append([round(t_got - t_mark, 4), round(t_done - t_got, 4), batch.n_tokens if rows else 0,
                                   padded_tokens(batch) if rows else 0, len(rows), round(m.cost, 1)])
            if d.enabled and micro_stop:
                # --stop-check micro: a 1-float all-reduce after EVERY micro-batch. It is a barrier, so each
                # micro-step waits for the slowest rank (grad-accum then cannot average stragglers out).
                t_sync = time.perf_counter()
                stop_now = ddp.any_flag(d, stop.flag)
                dt_flag = time.perf_counter() - t_sync
                win["sync_s"] += dt_flag
                step_flag_s += dt_flag
            elif lagged is not None:
                t_sync = time.perf_counter()
                stop_now = lagged.push(stop.flag)  # result of the previous micro-batch's flag (all ranks agree)
                dt_flag = time.perf_counter() - t_sync
                win["sync_s"] += dt_flag
                step_flag_s += dt_flag
            else:
                stop_now = False
            if micro % args.grad_accum:
                t_mark = time.perf_counter()
                if stop_now:  # DDP: everyone drops the partial step, checkpoint = last completed step
                    opt.zero_grad(set_to_none=True)
                    abandoned = True
                    break
                continue
            # ---- optimizer step
            win["tokens_rank"] += step_cnt["tokens"]
            if d.enabled:
                vec = [float(step_cnt["tokens"]), float(step_cnt["padded"]), float(step_cnt["examples"]),
                       float(stats.dropped_qids - last_stats["dropped_qids"]), float(stats.bad_labels - last_stats["bad_labels"]),
                       float(bool(stop.flag))]
                for k in PART_KEYS:
                    s, n = step_acc.get(k, (0.0, 0))
                    vec += [float(s), float(n)]
                t_sync = time.perf_counter()
                red = dict(zip(EXTRA_KEYS, reducer.reduce(vec)))
                dt_red = time.perf_counter() - t_sync
                win["sync_s"] += dt_red
                if red["stop"] > 0:  # any rank signalled before the all-reduce -> all stop after this step
                    stop_now = True
                last_stats = {"dropped_qids": stats.dropped_qids, "bad_labels": stats.bad_labels}
                for k in ("tokens", "padded", "examples"):
                    win[k] += red[k]
                glob_stats["dropped_qids"] += int(red["dropped_qids"])
                glob_stats["bad_labels"] += int(red["bad_labels"])
                for k in PART_KEYS:
                    if red[f"{k}_n"] > 0:
                        a = acc.setdefault(k, [0.0, 0])
                        a[0] += red[f"{k}_s"]
                        a[1] += int(red[f"{k}_n"])
            else:
                for k in ("tokens", "padded", "examples"):
                    win[k] += step_cnt[k]
                glob_stats = {"dropped_qids": stats.dropped_qids, "bad_labels": stats.bad_labels}
                for k, (s, n) in step_acc.items():
                    a = acc.setdefault(k, [0.0, 0])
                    a[0] = a[0] + s
                    a[1] += n
            step_acc = {}
            step_cnt = {"tokens": 0, "padded": 0, "examples": 0}
            f = lr_factor(step, total, args.warmup)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
            torch.nn.utils.clip_grad_norm_(params, args.clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if tlog is not None:
                tlog.write(json.dumps({"step": step, "rank": d.rank, "micro": step_micro, "flag_s": round(step_flag_s, 4),
                                       "reduce_s": round(dt_red, 4) if d.enabled else 0.0,
                                       "t": round(time.perf_counter() - t_start, 3)}) + "\n")
                tlog.flush()
                step_micro, step_flag_s = [], 0.0
            win["steps"] += 1
            info.update(step=step, epoch=m.epoch, cursor=m.cursor)
            if (step % args.log_every == 0 or step == total) and d.is_main:
                sync(device)
                dt = time.perf_counter() - win_t0
                rec: dict[str, Any] = {"step": step, "epoch": m.epoch, "lr": opt.param_groups[0]["lr"],
                                       "head_lr": opt.param_groups[2]["lr"]}
                tot_s, tot_n = 0.0, 0
                for k, (s, n) in acc.items():
                    sv = float(s)
                    rec[f"loss_{k}"] = sv / max(n, 1)
                    rec[f"n_{k}"] = n
                    if not k.startswith("x_"):
                        tot_s += sv
                        tot_n += n
                rec["loss"] = tot_s / max(tot_n, 1)
                rec.update(steps_per_s=win["steps"] / dt, tokens_per_s=win["tokens"] / dt,
                           padded_tokens_per_s=win["padded"] / dt, examples_per_s=win["examples"] / dt,
                           mps_mb=round(mps_mb()), rss_peak_mb=round(rss_mb()),
                           elapsed_s=round(time.perf_counter() - t_start, 1), dropped_qids=glob_stats["dropped_qids"],
                           bad_labels=glob_stats["bad_labels"])
                if d.enabled:
                    # sync_s: rank-0 time inside collectives (straggler wait + all-reduce), per optimizer step
                    rec.update(world=d.world, tokens_per_s_rank0=win["tokens_rank"] / dt,
                               sync_s_per_step=win["sync_s"] / max(win["steps"], 1),
                               allreduce_mb=round(reducer.bytes_per_step / 2**20, 1))
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                print(json.dumps(rec), file=sys.stderr)
                acc = {}
                win = {"tokens": 0.0, "padded": 0.0, "examples": 0.0, "steps": 0, "tokens_rank": 0.0, "sync_s": 0.0}
                win_t0 = time.perf_counter()
            elif step % args.log_every == 0 or step == total:
                acc = {}
                win = {"tokens": 0.0, "padded": 0.0, "examples": 0.0, "steps": 0, "tokens_rank": 0.0, "sync_s": 0.0}
                win_t0 = time.perf_counter()
            if device.type == "mps" and step % args.empty_cache_every == 0:
                torch.mps.empty_cache()
            if step % args.ckpt_every == 0 and step < total:
                checkpoint()
            if args.stop_at and step >= args.stop_at:
                stop.flag = True  # simulated interruption (tests): same path as SIGINT/SIGTERM
            if step >= total or stop_now or (stop.flag and not d.enabled):
                break
            t_mark = time.perf_counter()
    finally:
        if lagged is not None:
            lagged.drain()  # every rank issued the same flag ops; complete the last one before the exit barrier
        if isinstance(feed, Prefetch):
            feed.close()
    interrupted = (stop.flag or abandoned) and step < total
    checkpoint(final=step >= total)
    if log_f:
        log_f.close()
    if tlog is not None:
        tlog.close()
    source.close()
    summary = {"steps": step, "total_steps": total, "out": str(out), "state": str(state_dir),
               "elapsed_s": round(time.perf_counter() - t_start, 1), "data": vars(stats), "interrupted": interrupted}
    if d.enabled:
        summary["rank"], summary["world"] = d.rank, d.world
    if d.is_main:
        print(json.dumps(summary), file=sys.stderr)
    ddp.barrier(d)  # nobody leaves before rank 0 has written the checkpoint
    ddp.destroy(d)
    if interrupted:
        raise SystemExit(130)
    return summary


def vars_json(args: argparse.Namespace) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m jev_local.train.train")
    ap.add_argument("--data", type=Path, default=Path("data/cu"))
    ap.add_argument("--extra", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("models/jev-local-fast"))
    ap.add_argument("--run-name", default=None, help="runs/<name>; defaults to the basename of --out")
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--init-from", type=Path, default=None,
                    help="start from these weights (servable ckpt or trainer state dir); fresh optimizer and schedule")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=4, help="examples per micro-batch (index mode)")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1536, help=f"packed row length, up to {MAX_TRAIN_LEN}")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--clip", type=float, default=1.0)
    # losses (defaults = v1: CE + smoothing 0.02 + RPS 0.25 on score)
    ap.add_argument("--loss", choices=("v1", "v2"), default="v1",
                    help="v2 = KL + 0.5 spherical + 1.0 RPS + 0.25 CORAL + 0.1 consistency (PLAN §3.3)")
    ap.add_argument("--smoothing", type=float, default=None)
    ap.add_argument("--brier", type=float, default=None)
    ap.add_argument("--rps", type=float, default=None)
    ap.add_argument("--kl", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--spherical", type=float, default=None)
    ap.add_argument("--coral", type=float, default=None)
    ap.add_argument("--noise-sigma", default=None, help="pathwise logit noise sigma 'start,end' (e.g. 0.4,0.1)")
    ap.add_argument("--noise-samples", type=int, default=None)
    ap.add_argument("--noise-weight", type=float, default=None)
    ap.add_argument("--consistency", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N examples of each input file")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-only", action="store_true", help="write a checkpoint with random heads and exit")
    ap.add_argument("--device", default=None, help="mps | cpu | cuda (default: mps when available)")
    ap.add_argument("--amp", action="store_true",
                    help="mixed precision: fp16 on MPS, bf16 on CPU and bf16-capable CUDA (weights stay fp32)")
    ap.add_argument("--threads", type=int, default=None, help="torch intra-op threads (CPU training)")
    ap.add_argument("--grad-ckpt", action=argparse.BooleanOptionalAction, default=True,
                    help="recompute layer activations in backward (default on: ~4x less activation memory)")
    ap.add_argument("--encoder", choices=("banded", "reference"), default="banded",
                    help="banded = exact fast tree forward (engine.BandedEncoder, ~2x on CPU); reference = v1 MaskedEncoder")
    ap.add_argument("--attn", choices=("tree", "dense"), default="tree",
                    help="tree = exact sparse layout (default); dense = [B,1,L,L] mask reference")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--empty-cache-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefetch", type=int, default=None,
                    help="micro-batches read+tokenized ahead in a thread (default 2 in stream mode, 0 in index mode)")
    # stream mode (v2 data)
    st = ap.add_argument_group("stream mode (jev_local/train/stream.py)")
    st.add_argument("--stream", nargs="+", default=None, help="data roots: data/v2 (uses raw/**) or shard dirs/files")
    st.add_argument("--stream-cache", default="runs/stream_cache", help="decompressed copies + row indices")
    st.add_argument("--mixture", default=None, help="MixConfig JSON (bucket_weights, caps, curriculum, ...)")
    st.add_argument("--batch-tokens", type=int, default=None, help="padded tokens per micro-batch (default 16384)")
    st.add_argument("--batch-rows", type=int, default=None, help="fixed rows per micro-batch instead of a token budget")
    st.add_argument("--source-cap", type=int, default=None, help="max supervised decisions per source group per pass")
    st.add_argument("--class-cap", type=int, default=None, help="max rows per (source group, class) per pass")
    st.add_argument("--alpha", type=float, default=None, help="within-bucket source sampling exponent (0.5 = sqrt)")
    st.add_argument("--passes", type=float, default=None, help="mixture passes (fractional ok)")
    st.add_argument("--pool", type=int, default=None, help="rows sorted together for length bucketing")
    st.add_argument("--batch-layout", choices=("tree", "dense"), default=None,
                    help="tree (default): group rows by option/header/state size, cut by padded tree-layout slots; "
                         "dense: v1 rows x longest")
    st.add_argument("--balance", action=argparse.BooleanOptionalAction, default=None,
                    help="DDP: deal each step's micro-batches to ranks by estimated cost")
    st.add_argument("--curriculum", default=None, help="'progress:max_tokens,...' e.g. '0:1024,0.2:8192'")
    st.add_argument("--license", default=None, help="comma list of license_use values to keep (D5: commercial)")
    st.add_argument("--buckets", default=None, help="comma list of buckets to keep")
    st.add_argument("--splits", default=None, help="comma list of splits to train on (default train)")
    st.add_argument("--calibrate-lengths", type=int, default=64,
                    help="rows per file tokenized to calibrate token-length estimates (0 = use stats/bytes)")
    st.add_argument("--index-workers", type=int, default=4)
    # distributed
    dd = ap.add_argument_group("distributed (jev_local/train/ddp.py)")
    dd.add_argument("--ddp", action="store_true", help="data parallel over torchrun ranks (gloo on CPU)")
    dd.add_argument("--ddp-timeout", type=float, default=1800.0, help="collective timeout in seconds")
    dd.add_argument("--ddp-compress", choices=("bf16",), default=None, help="all-reduce gradients in bf16")
    dd.add_argument("--stop-check", choices=("lagged", "micro"), default="lagged",
                    help="DDP stop flag (SIGTERM): 'lagged' = non-blocking all-reduce read one micro-batch later "
                         "(default; no per-micro barrier, latency ~2 micro-batches); 'micro' = old blocking check after "
                         "every micro-batch (a barrier per micro-batch)")
    dd.add_argument("--timing-log", action="store_true",
                    help="write <runs>/<run>/timing/rank<r>.jsonl: per step, per micro-batch feed-wait/compute seconds, "
                         "tokens and estimated cost, plus flag/all-reduce seconds")
    ap.add_argument("--stop-at", type=int, default=None, help=argparse.SUPPRESS)  # tests: act as if killed at step K
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.epochs_given = args.epochs is not None
    args.epochs = args.epochs or 1
    train(args)


if __name__ == "__main__":
    main()
