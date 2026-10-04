"""Benchmark the fast encoder engine (`jev-local-fast`) and its trainer.

    .venv/bin/python scripts/bench_encoder.py                      # inference (mps fp16/fp32, cpu fp32) + training
    .venv/bin/python scripts/bench_encoder.py --only infer --runs 50 --configs mps:fp16,mps:fp32
    .venv/bin/python scripts/bench_encoder.py --only train --train-configs mps:fp32,mps:amp

Inference (CONTRACT B), for two harness requests built exactly as the runtime builds them
(`build_state` + `build_questions` + `rank_apps` + the span extractors) over the hand-written
screens in tests/fixtures/harness_cases.json:
  - "h60": the 60-element Safari screen, 11 questions (~1.5k tokens)
  - "h15": 15 elements of the Finder screen, 11 questions (~700 tokens)
and four regimes per request:
  - first:    the very first evaluate() after the engine is loaded (kernel compilation, cold caches)
  - uncached: a different transcript every run and the packer's token cache cleared first, i.e. a new
              screen every time (the tokenizer runs on every element/option)
  - partials: the transcript grows one word per run on a fixed screen (what the harness actually
              sends while someone talks: only the transcript segment is new)
  - warm:     the identical request repeated
Each config runs in its own subprocess, so load time and peak memory are per config.

Training: forward + backward + AdamW step at batch 4 on ~1k-token harness examples (the training
data format, via train.encode_example / prepare_batch), timed per step with a device sync; reports
examples/s, real tokens/s, padded tokens/s and peak RSS / MPS driver memory.

The machine is shared, so every result records the load average and swap in use while it ran.
Results go to runs/bench_encoder.json (and stdout).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "harness_cases.json"
sys.path.insert(0, str(ROOT))

from jev_local.harness.questions import build_questions, rank_apps  # noqa: E402
from jev_local.harness.spans import extract_text_candidates, extract_url_candidates  # noqa: E402
from jev_local.harness.state import build_state  # noqa: E402
from jev_local.harness.types import Element, Snapshot  # noqa: E402

DEFAULT_CKPT = ROOT / "models" / "jev-local-fast"
INIT_CKPT = ROOT / "models" / "jev-local-fast-init"

# Transcripts that produce both text and URL candidates, so the request carries all 11 questions.
TRANSCRIPTS = [
    "go to github dot com and type hello world in the search box",
    "open nytimes dot com and search for the latest market news",
    "visit the weather website and type toronto into the search field",
    "open youtube dot com and search for lofi hip hop radio",
    "go to amazon dot com and type usb c charger please",
    "navigate to wikipedia dot org and search for alan turing",
    "open reddit dot com and type best pasta recipe",
    "go to maps dot google dot com and search for coffee near me",
    "visit apple dot com and type macbook air in the search bar",
    "go to news dot ycombinator dot com and type show hn",
]
PARTIAL_TEXT = "go to github dot com and type hello world in the search box and press enter"


# ------------------------------------------------------------------ requests


def load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


def snapshot_of(screen: dict[str, Any], app: str, max_elements: int) -> Snapshot:
    els = tuple(
        Element(eid=e["eid"], role=e["role"], label=e.get("label", ""), value=e.get("value"),
                focused=bool(e.get("focused", False)), enabled=bool(e.get("enabled", True)), context=e.get("context"))
        for e in screen["elements"][:max_elements]
    )
    focused = next((e.eid for e in els if e.focused), None)
    return Snapshot(app_name=screen.get("app", app), bundle_id="com.example.bench", pid=1,
                    window_title=screen.get("window_title"), elements=els, taken_at=0.0, focused_eid=focused)


REQUESTS = {"h60": ("safari_news", 60), "h15": ("finder_downloads", 15)}


def harness_request(fx: dict[str, Any], which: str, transcript: str):
    screen_name, n = REQUESTS[which]
    screen = fx["screens"][screen_name]
    snap = snapshot_of(screen, screen_name, n)
    apps = rank_apps(transcript, fx["apps"], screen.get("running", ()), max_n=24)
    state = build_state(transcript, snap)
    qs = build_questions(snap, apps, extract_text_candidates(transcript), extract_url_candidates(transcript))
    return state, qs


# ------------------------------------------------------------------ measurement helpers


def conditions() -> dict[str, Any]:
    load = os.getloadavg()
    swap = None
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=2).stdout
        swap = out.split("used =")[1].split()[0] if "used =" in out else None
    except Exception:
        pass
    return {"load1": round(load[0], 1), "load5": round(load[1], 1), "swap_used": swap}


def rss_peak_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 1e6 if sys.platform == "darwin" else r / 1e3, 1)


def summarize(ms: list[float]) -> dict[str, float]:
    s = sorted(ms)

    def pct(q: float) -> float:
        k = (len(s) - 1) * q
        lo, hi = int(k), min(int(k) + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)

    return {"n": len(s), "p50": round(pct(0.5), 2), "p95": round(pct(0.95), 2), "mean": round(statistics.fmean(s), 2),
            "min": round(s[0], 2), "max": round(s[-1], 2)}


# ------------------------------------------------------------------ inference worker


def infer_worker(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from jev_local.engine.encoder.engine import FastEngine

    device, dt = args.config.split(":")
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[dt]
    fx = load_fixture()
    t0 = time.perf_counter()
    eng = FastEngine(args.ckpt, device=device, dtype=dtype)
    load_ms = (time.perf_counter() - t0) * 1e3
    out: dict[str, Any] = {"config": args.config, "device": str(eng.device), "dtype": str(eng.dtype),
                           "ckpt": str(args.ckpt), "load_ms": round(load_ms, 1)}

    def timed(state, qs) -> tuple[float, Any]:
        t = time.perf_counter()
        res = eng.evaluate(state, qs)
        return (time.perf_counter() - t) * 1e3, res

    ref = None
    for which in ("h60", "h15"):
        rows: dict[str, Any] = {}
        state, qs = harness_request(fx, which, TRANSCRIPTS[0])
        eng.packer._cache.clear()
        ms, res = timed(state, qs)
        rows["first"] = round(ms, 1)
        rows["input_tokens"] = res.input_tokens
        rows["n_questions"] = len(qs)
        rows["n_elements"] = len(qs["target"].criteria) - 1
        rows["tree_shape"] = list(eng.last_shape)
        if which == "h60":
            ref = {k: d.probs for k, d in res.dists.items()}
        rng = random.Random(0)
        # uncached: new transcript + cold token cache every run
        ms_l, toks = [], []
        for i in range(args.runs):
            text = TRANSCRIPTS[i % len(TRANSCRIPTS)] + f" {rng.choice(['now', 'thanks', 'ok', 'please'])} {i}"
            st, q = harness_request(fx, which, text)
            eng.packer._cache.clear()
            t, r = timed(st, q)
            ms_l.append(t)
            toks.append(r.input_tokens)
        rows["uncached"] = summarize(ms_l) | {"tokens_min": min(toks), "tokens_max": max(toks),
                                              "timings_ms_last": {k: round(v, 2) for k, v in r.timings_ms.items()}}
        # partials: transcript grows a word at a time over a fixed screen (token cache warm for the rest)
        words = PARTIAL_TEXT.split()
        ms_l = []
        for i in range(args.runs):
            n = 1 + i % len(words)
            st, q = harness_request(fx, which, " ".join(words[:n]))
            t, r = timed(st, q)
            ms_l.append(t)
        rows["partials"] = summarize(ms_l)
        # warm: identical request
        for _ in range(3):
            timed(state, qs)
        ms_l = []
        for _ in range(args.runs):
            t, r = timed(state, qs)
            ms_l.append(t)
        rows["warm"] = summarize(ms_l) | {"timings_ms_last": {k: round(v, 2) for k, v in r.timings_ms.items()}}
        rows["conditions"] = conditions()
        out[which] = rows
        print(f"[bench] {args.config} {which}: {json.dumps(rows)}", file=sys.stderr, flush=True)
    out["probs_h60"] = ref
    out["rss_peak_mb"] = rss_peak_mb()
    if eng.device.type == "mps":
        out["mps_driver_mb"] = round(torch.mps.driver_allocated_memory() / 1e6, 1)
    return out


# ------------------------------------------------------------------ training worker


def train_examples(n: int, target_tokens: int, seed: int = 0) -> list[dict[str, Any]]:
    """Fixture-format training examples whose packed length is close to target_tokens."""
    from jev_local.train.fixture import make_example

    rng = random.Random(seed)
    out = []
    i = 0
    # ~22 tokens per extra element on top of a ~560-token base (measured on the fixture generator)
    n_el = max(5, min(60, (target_tokens - 560) // 22))
    while len(out) < n:
        ex = make_example(rng, i, n_elements=n_el)
        i += 1
        if ex is not None:
            out.append(ex)
    return out


def train_worker(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from jev_local.engine.encoder.engine import init_model, pick_device
    from jev_local.train.losses import LossConfig, compute_loss
    from jev_local.train.train import Packer, encode_example, param_groups, prepare_batch

    device_s, mode = args.config.split(":")
    device = pick_device(device_s)
    amp = mode == "amp" and device.type == "mps"
    enc, heads, tok = init_model(seed=0)
    packer = Packer(tok, max_len=1536, buckets=(256, 384, 512, 768, 1024, 1536))
    enc.to(device).train()
    heads.to(device).train()
    enc.grad_ckpt = args.grad_ckpt
    opt = torch.optim.AdamW(param_groups(enc, heads, 5e-5, 1e-3, 0.01), betas=(0.9, 0.98), eps=1e-6)
    cfg = LossConfig()
    exs = train_examples(args.batch * (args.steps + args.warmup_steps), args.train_tokens)
    rows_all = [encode_example(e, packer, 1536) for e in exs]
    lens = [sum(r.pack.length for r in rows) for rows in rows_all]

    def sync() -> None:
        if device.type == "mps":
            torch.mps.synchronize()

    times, tokens, padded, losses = [], [], [], []
    for s in range(args.steps + args.warmup_steps):
        rows = [r for rs in rows_all[s * args.batch : (s + 1) * args.batch] for r in rs]
        sync()
        t = time.perf_counter()
        batch, plan, tg, _ = prepare_batch(rows, packer, device, enc.window, "tree")
        with torch.autocast("mps", dtype=torch.float16, enabled=amp):
            out = heads(enc(batch), plan)
        loss, _ = compute_loss(out, tg, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(heads.parameters()), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        lv = float(loss)  # sync
        sync()
        if s >= args.warmup_steps:
            times.append(time.perf_counter() - t)
            tokens.append(batch.n_tokens)
            padded.append(int(batch.input_ids.numel()))
        losses.append(lv)
    tot = sum(times)
    res = {
        "config": args.config, "amp": amp, "grad_ckpt": args.grad_ckpt, "batch": args.batch, "steps": args.steps,
        "tokens_per_example_mean": round(statistics.fmean(lens), 1), "tokens_per_example_max": max(lens),
        "step_s": summarize([x * 1e3 for x in times]),  # ms
        "examples_per_s": round(args.batch * len(times) / tot, 2),
        "tokens_per_s": round(sum(tokens) / tot, 1),
        "padded_tokens_per_s": round(sum(padded) / tot, 1),
        "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4),
        "finite": all(x == x and abs(x) != float("inf") for x in losses),
        "rss_peak_mb": rss_peak_mb(), "conditions": conditions(),
    }
    if device.type == "mps":
        res["mps_driver_mb"] = round(torch.mps.driver_allocated_memory() / 1e6, 1)
    print(f"[bench] train {args.config}: {json.dumps(res)}", file=sys.stderr, flush=True)
    return res


# ------------------------------------------------------------------ orchestration


def run_sub(kind: str, config: str, args: argparse.Namespace, extra: list[str] = ()) -> dict[str, Any]:
    cmd = [sys.executable, __file__, "--worker", kind, "--config", config, "--ckpt", str(args.ckpt),
           "--runs", str(args.runs), "--steps", str(args.steps), "--batch", str(args.batch),
           "--train-tokens", str(args.train_tokens), *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    sys.stderr.write(p.stderr[-4000:])
    if p.returncode != 0:
        return {"config": config, "error": p.stderr.strip().splitlines()[-1:] or ["failed"]}
    return json.loads(p.stdout.strip().splitlines()[-1])


def fp16_drift(results: list[dict[str, Any]]) -> dict[str, float]:
    """Max |p_fp16 - p_fp32| on the h60 request, per device (is fp16 numerically fine?)."""
    by = {r["config"]: r.get("probs_h60") for r in results if r.get("probs_h60")}
    out = {}
    base = by.get("cpu:fp32") or by.get("mps:fp32")
    for cfg, probs in by.items():
        if base is None or probs is base:
            continue
        out[cfg] = round(max(abs(a - b) for q in base for a, b in zip(base[q], probs[q])), 6)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None, help="default: models/jev-local-fast, else -fast-init")
    ap.add_argument("--only", choices=("infer", "train"), default=None)
    ap.add_argument("--configs", default="mps:fp16,mps:fp32,cpu:fp32")
    ap.add_argument("--train-configs", default="mps:fp32,mps:amp")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup-steps", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--train-tokens", type=int, default=1024)
    ap.add_argument("--grad-ckpt", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "bench_encoder.json")
    ap.add_argument("--worker", choices=("infer", "train"), default=None, help=argparse.SUPPRESS)
    ap.add_argument("--config", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.ckpt is None:
        args.ckpt = DEFAULT_CKPT if (DEFAULT_CKPT / "heads.safetensors").is_file() else INIT_CKPT

    if args.worker == "infer":
        print(json.dumps(infer_worker(args)))
        return 0
    if args.worker == "train":
        print(json.dumps(train_worker(args)))
        return 0

    report: dict[str, Any] = {"ckpt": str(args.ckpt), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                              "conditions_start": conditions()}
    if args.only in (None, "infer"):
        res = [run_sub("infer", c, args) for c in args.configs.split(",") if c]
        report["fp16_max_prob_drift"] = fp16_drift(res)
        for r in res:
            r.pop("probs_h60", None)
        report["infer"] = res
    if args.only in (None, "train"):
        extra = ["--grad-ckpt" if args.grad_ckpt else "--no-grad-ckpt", "--warmup-steps", str(args.warmup_steps)]
        report["train"] = [run_sub("train", c, args, extra) for c in args.train_configs.split(",") if c]
    report["conditions_end"] = conditions()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(args.out.read_text()) if args.out.is_file() else {}
    if args.only and prev:  # keep the other half of an earlier full run
        report = {**prev, **report}
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
