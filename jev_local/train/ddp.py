"""Data-parallel training helpers for `train.py --ddp` (PLAN §4.1 E2): torchrun + gloo on CPU (NCCL on CUDA).

Launch one process per node (CPU: intra-op threads do the in-node parallelism):

    OMP_NUM_THREADS=32 GLOO_SOCKET_IFNAME=eth0 torchrun --nnodes 2 --nproc-per-node 1 --node-rank $R \\
        --master-addr 10.0.0.4 --master-port 29500 -m jev_local.train.train --ddp --threads 32 ...

Design
- No DistributedDataParallel wrapper: the encoder + heads forward is two calls with a custom batch layout,
  and the trainer accumulates gradients over micro-batches. Instead `GradReducer` copies every gradient
  into ONE flat buffer and runs ONE all_reduce per optimizer step (plus a few float counters appended to
  the same buffer for global logging), then averages. Identical gradients on every rank -> identical clip,
  identical AdamW step -> replicas stay bit-identical without re-broadcasting weights.
- Stop flag (SIGTERM/SIGINT): `LaggedFlag` (default, `--stop-check lagged`) starts a 1-element MAX all_reduce
  after every micro-batch without waiting and reads it one micro-batch later, so all ranks learn of a signal
  at the same micro-batch, drop the partial accumulation, rank 0 checkpoints the last completed step, everyone
  meets at a barrier and exits. torchrun gives workers 30 s after forwarding a signal, so this must not wait
  for a whole optimizer step. The old blocking `any_flag` per micro-batch (`--stop-check micro`) made every
  micro-batch a barrier, so ranks waited for the slowest rank 2x per step and grad-accum could not average
  stragglers out (measured: 50-70% of each step at 32-48 ranks; docs/build/stage1-speed.md).
- Resume: only rank 0 reads the checkpoint (it may exist on rank 0's disk only); weights, AdamW state and
  the data cursor are broadcast to the other ranks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Iterable, Sequence

import torch
import torch.distributed as dist


@dataclass
class Dist:
    rank: int = 0
    world: int = 1
    local_rank: int = 0
    local_world: int = 1
    enabled: bool = False
    backend: str = ""

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def is_local_main(self) -> bool:
        return self.local_rank == 0


NO_DIST = Dist()


def init_from_env(device: torch.device, timeout_s: float = 1800.0) -> Dist:
    """Join the process group described by torchrun's env vars (RANK, WORLD_SIZE, MASTER_ADDR, ...)."""
    missing = [k for k in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT") if k not in os.environ]
    if missing:
        raise SystemExit(f"--ddp needs torchrun (or {', '.join(missing)} in the environment)")
    backend = "nccl" if device.type == "cuda" else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://", timeout=timedelta(seconds=timeout_s))
    d = Dist(rank=dist.get_rank(), world=dist.get_world_size(), local_rank=int(os.environ.get("LOCAL_RANK", 0)),
             local_world=int(os.environ.get("LOCAL_WORLD_SIZE", 1)), enabled=True, backend=backend)
    return d


def destroy(d: Dist) -> None:
    if d.enabled and dist.is_initialized():
        dist.destroy_process_group()


def barrier(d: Dist) -> None:
    if d.enabled:
        dist.barrier()


def any_flag(d: Dist, flag: bool) -> bool:
    """True on every rank iff `flag` is True on at least one rank."""
    if not d.enabled:
        return bool(flag)
    t = torch.tensor([1.0 if flag else 0.0])
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item() > 0)


class LaggedFlag:
    """Non-blocking `any_flag` with one micro-batch of slack: `push(flag)` starts a MAX all_reduce of this
    micro-batch's flag and returns the result of the PREVIOUS push (waiting for it only if some rank is still
    more than one micro-batch behind). Every rank reads result j at the same point (push j+1), so all ranks
    agree on when to stop, and the stop latency is ~2 micro-batches instead of a whole optimizer step, without
    a hard barrier after every micro-batch. `drain()` completes the outstanding op (before any other exit path)."""

    def __init__(self, d: Dist):
        self.d = d
        self.pending: tuple[Any, torch.Tensor] | None = None

    def _result(self) -> bool:
        if self.pending is None:
            return False
        work, t = self.pending
        self.pending = None
        work.wait()
        return bool(t.item() > 0)

    def push(self, flag: bool) -> bool:
        if not self.d.enabled:
            return bool(flag)
        prev = self._result()
        t = torch.tensor([1.0 if flag else 0.0])
        self.pending = (dist.all_reduce(t, op=dist.ReduceOp.MAX, async_op=True), t)
        return prev

    def drain(self) -> bool:
        return self._result()


def broadcast_object(d: Dist, obj: Any, src: int = 0) -> Any:
    if not d.enabled:
        return obj
    box = [obj if d.rank == src else None]
    dist.broadcast_object_list(box, src=src)
    return box[0]


def broadcast_tensors(d: Dist, tensors: Sequence[torch.Tensor], src: int = 0) -> None:
    """In-place broadcast, flattened per dtype so a model costs a handful of collectives."""
    if not d.enabled or not tensors:
        return
    by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
    for t in tensors:
        by_dtype.setdefault(t.dtype, []).append(t)
    for dt, ts in by_dtype.items():
        flat = torch.cat([t.detach().reshape(-1).to("cpu") for t in ts]) if d.rank == src else \
            torch.empty(sum(t.numel() for t in ts), dtype=dt)
        if d.backend == "nccl":
            flat = flat.cuda()
        dist.broadcast(flat, src=src)
        o = 0
        with torch.no_grad():
            for t in ts:
                n = t.numel()
                t.copy_(flat[o : o + n].view_as(t).to(t.device))
                o += n


def module_tensors(modules: Iterable[torch.nn.Module]) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for m in modules:
        out += [p.data for p in m.parameters()]
        out += [b for b in m.buffers()]
    return out


def broadcast_optimizer(d: Dist, opt: torch.optim.Optimizer, src: int = 0) -> None:
    """Make every rank's optimizer state equal to rank `src`'s (structure via pickle, tensors in bulk)."""
    if not d.enabled:
        return
    sd = opt.state_dict() if d.rank == src else None
    if d.rank == src:
        skel = {"param_groups": sd["param_groups"], "state": {
            k: {n: (("T", tuple(v.shape), str(v.dtype)) if torch.is_tensor(v) else v) for n, v in s.items()}
            for k, s in sd["state"].items()}}
    else:
        skel = None
    skel = broadcast_object(d, skel, src)
    if d.rank != src:
        state: dict = {}
        for k, s in skel["state"].items():
            state[k] = {n: (torch.empty(v[1], dtype=getattr(torch, v[2].split(".")[-1]))
                            if isinstance(v, tuple) and len(v) == 3 and v[0] == "T" else v) for n, v in s.items()}
        sd = {"state": state, "param_groups": skel["param_groups"]}
    tensors = [v for k in sorted(sd["state"]) for n, v in sorted(sd["state"][k].items()) if torch.is_tensor(v)]
    broadcast_tensors(d, tensors, src)
    if d.rank != src:
        opt.load_state_dict(sd)


class GradReducer:
    """One all_reduce per optimizer step over a flat copy of every gradient (+ float counters)."""

    def __init__(self, d: Dist, params: Sequence[torch.nn.Parameter], n_extra: int = 0, compress: str | None = None):
        self.d = d
        self.params = [p for p in params if p.requires_grad]
        self.numel = sum(p.numel() for p in self.params)
        self.n_extra = n_extra
        self.dtype = torch.bfloat16 if compress == "bf16" else torch.float32
        dev = self.params[0].device if self.params else torch.device("cpu")
        if d.backend == "gloo":
            dev = torch.device("cpu")
        # layout: [grads | extra counters | one has-grad flag per parameter]
        self.buf = torch.zeros(self.numel + n_extra + len(self.params), dtype=torch.float32, device=dev)
        self.bytes_per_step = self.buf.numel() * (2 if self.dtype == torch.bfloat16 else 4)

    def reduce(self, extra: Sequence[float] = ()) -> list[float]:
        """Average gradients across ranks in place; return the SUM of `extra` across ranks. A parameter
        whose gradient is None on every rank stays None (AdamW then skips it, as in single-process runs)."""
        if not self.d.enabled:
            return list(extra)
        buf = self.buf
        P = len(self.params)
        with torch.no_grad():
            o = 0
            for p in self.params:
                n = p.numel()
                if p.grad is None:
                    buf[o : o + n].zero_()
                else:
                    buf[o : o + n].copy_(p.grad.detach().reshape(-1))
                o += n
            tail = list(extra) + [0.0] * (self.n_extra - len(extra))
            tail += [0.0 if p.grad is None else 1.0 for p in self.params]
            buf[o:].copy_(torch.tensor(tail, dtype=torch.float32))
            if self.dtype == torch.float32:
                dist.all_reduce(buf)
                red = buf
            else:  # counters and flags stay exact enough in bf16 for logging (< 256 ranks)
                red = buf.to(self.dtype)
                dist.all_reduce(red)
                red = red.float()
            scale = 1.0 / self.d.world
            has = red[self.numel + self.n_extra :].tolist()
            o = 0
            for p, h in zip(self.params, has):
                n = p.numel()
                if h > 0:
                    g = red[o : o + n].view_as(p).to(p.device, p.dtype) * scale
                    if p.grad is None:
                        p.grad = g.clone()
                    else:
                        p.grad.copy_(g)
                else:
                    p.grad = None
                o += n
            return red[self.numel : self.numel + self.n_extra].tolist()
