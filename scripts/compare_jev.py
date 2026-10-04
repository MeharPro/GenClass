"""Head-to-head: hosted Jev (via OpenRouter) vs jev-local-fast on the same held-out test examples.

Both models get byte-identical {state, questions}. Jev answers are cached in runs/compare/jev.jsonl
(re-runs only pay for missing ids); ours are computed locally on CPU.

Usage:
  .venv/bin/python scripts/compare_jev.py [--n-cu 1000] [--n-gen 300] [--workers 6] [--skip-local]
Key: ~/.jev-local/secrets/openrouter.key (never printed or logged).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "compare"
URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
KEY_PATH = Path.home() / ".jev-local" / "secrets" / "openrouter.key"


def load_sample(path: Path, n: int, seed: int) -> list[dict]:
    rows = [json.loads(line) for line in path.open()]
    random.Random(seed).shuffle(rows)
    return rows[:n]


# ---------------------------------------------------------------- Jev


def ask_jev(client: httpx.Client, key: str, ex: dict) -> dict:
    body = {"model": MODEL, "state": ex["state"], "questions": ex["questions"]}
    delay = 1.0
    for attempt in range(5):
        t = time.perf_counter()
        try:
            r = client.post(URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=60)
        except httpx.HTTPError as e:
            err = f"{type(e).__name__}"
        else:
            ms = (time.perf_counter() - t) * 1000
            if r.status_code == 200:
                d = r.json()
                return {"id": ex["id"], "ok": True, "ms": ms, "answers": d["answers"], "usage": d.get("usage", {}),
                        "model": d.get("model")}
            err = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code not in (408, 429, 500, 502, 503, 504, 529):
                return {"id": ex["id"], "ok": False, "error": err}
        time.sleep(delay)
        delay *= 2
    return {"id": ex["id"], "ok": False, "error": err}


def run_jev(examples: list[dict], workers: int) -> dict[str, dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    cache_path = OUT / "jev.jsonl"
    cache: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.open():
            rec = json.loads(line)
            if rec.get("ok"):
                cache[rec["id"]] = rec
    todo = [ex for ex in examples if ex["id"] not in cache]
    print(f"[jev] cached {len(cache)}, to query {len(todo)}", file=sys.stderr)
    if todo:
        key = KEY_PATH.read_text().strip()
        lock = threading.Lock()
        with httpx.Client(http2=False) as client, cache_path.open("a") as f, ThreadPoolExecutor(workers) as pool:
            futs = [pool.submit(ask_jev, client, key, ex) for ex in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                with lock:
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                if rec["ok"]:
                    cache[rec["id"]] = rec
                elif i <= 3 or i % 100 == 0:
                    print(f"[jev] error {rec['id']}: {rec['error'][:200]}", file=sys.stderr)
                if i % 100 == 0:
                    print(f"[jev] {i}/{len(todo)}", file=sys.stderr)
    return cache


# ---------------------------------------------------------------- ours


def run_local(examples: list[dict]) -> dict[str, dict]:
    import torch

    from jev_local.engine.encoder.engine import FastEngine
    from jev_local.schema import question_from_json

    torch.set_num_threads(4)
    eng = FastEngine(ROOT / "models" / "jev-local-fast", device="cpu", dtype=torch.float32)
    out: dict[str, dict] = {}
    for i, ex in enumerate(examples, 1):
        qs = {k: question_from_json(v) for k, v in ex["questions"].items()}
        t = time.perf_counter()
        res = eng.evaluate(ex["state"], qs)
        ms = (time.perf_counter() - t) * 1000
        ans = {}
        for qid, d in res.dists.items():
            if d.kind == "noul":
                ans[qid] = {"type": "noul", "noul": d.probs[0]}
            elif d.kind == "choice":
                ans[qid] = {"type": "choice", "probabilities": dict(zip(d.labels, d.probs))}
            else:
                ans[qid] = {"type": "score", "probabilities": {str(j): p for j, p in enumerate(d.probs)}}
        out[ex["id"]] = {"ok": True, "ms": ms, "answers": ans}
        if i % 200 == 0:
            print(f"[local] {i}/{len(examples)}", file=sys.stderr)
    return out


# ---------------------------------------------------------------- scoring


def gold_of(label: dict):
    if label["type"] == "noul":
        return label["p"] >= 0.5
    if label["type"] == "choice":
        return label["label"] if "label" in label else max(label["dist"], key=label["dist"].get)
    if "level" in label:
        return str(label["level"])
    dist = label["dist"]
    return str(max(range(len(dist)), key=dist.__getitem__))


def pred_of(ans: dict):
    if ans["type"] == "noul":
        return ans["noul"] >= 0.5, ans["noul"]
    probs = ans["probabilities"]
    best = max(probs, key=probs.get)
    return best, probs[best]


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.96, k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def score(examples, jev, ours):
    per: dict[str, dict] = {}
    groups: dict[str, dict] = {}

    def bump(table, key, j_ok, o_ok):
        s = table.setdefault(key, {"n": 0, "jev": 0, "ours": 0, "ours_only": 0, "jev_only": 0})
        s["n"] += 1
        s["jev"] += j_ok
        s["ours"] += o_ok
        s["ours_only"] += o_ok and not j_ok
        s["jev_only"] += j_ok and not o_ok

    for ex in examples:
        if ex["id"] not in jev or ex["id"] not in ours:
            continue
        ja, oa = jev[ex["id"]]["answers"], ours[ex["id"]]["answers"]
        corpus = "cu" if ex["id"].startswith("cu") else "gen"
        meta = ex.get("meta") or {}
        for qid, lab in ex["labels"].items():
            if qid not in ja or qid not in oa:
                continue
            g = gold_of(lab)
            j_ok = pred_of(ja[qid])[0] == g
            o_ok = pred_of(oa[qid])[0] == g
            bump(per, f"{corpus}:{qid}", j_ok, o_ok)
            if corpus == "cu":
                bump(groups, "CU all questions", j_ok, o_ok)
                if qid == "intent":
                    bump(groups, "CU intent, complete" if not meta.get("prefix") else "CU intent, mid-sentence prefix",
                         j_ok, o_ok)
                if qid == "target" and g != "none":
                    bump(groups, "CU target (real element)", j_ok, o_ok)
                if qid == "text_span" and g != "none":
                    bump(groups, "CU text span (real payload)", j_ok, o_ok)
            else:
                bump(groups, "GEN " + ("held-out task families" if meta.get("heldout") else "seen task families"),
                     j_ok, o_ok)
    return per, groups


def pct(k, n):
    return f"{100 * k / n:5.1f}%" if n else "  n/a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cu", type=int, default=1000)
    ap.add_argument("--n-gen", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-local", action="store_true", help="reuse runs/compare/ours.json")
    ap.add_argument("--jev-only", action="store_true", help="only query Jev (e.g. while the Mac is short on RAM)")
    ap.add_argument("--local-only", action="store_true", help="only compute ours.json (e.g. on the Azure VM)")
    args = ap.parse_args()

    examples = load_sample(ROOT / "data/cu/test.jsonl", args.n_cu, args.seed) + \
        load_sample(ROOT / "data/gen/test.jsonl", args.n_gen, args.seed)
    ours_path = OUT / "ours.json"
    if args.local_only:
        OUT.mkdir(parents=True, exist_ok=True)
        ours_path.write_text(json.dumps(run_local(examples)))
        print(f"wrote {ours_path}")
        return
    jev = run_jev(examples, args.workers)
    if args.jev_only:
        print(f"jev answers: {len(jev)}")
        return
    if args.skip_local and ours_path.exists():
        ours = json.loads(ours_path.read_text())
    else:
        ours = run_local(examples)
        ours_path.write_text(json.dumps(ours))

    per, groups = score(examples, jev, ours)
    jl = [r["ms"] for r in jev.values() if r.get("ok")]
    ol = [r["ms"] for r in ours.values() if r.get("ok")]
    cost = sum((r.get("usage") or {}).get("cost", 0) or 0 for r in jev.values())
    toks = sum((r.get("usage") or {}).get("input_tokens", 0) or 0 for r in jev.values())

    lines = [f"# Jev ({MODEL}) vs jev-local-fast — held-out test, same inputs", "",
             f"Examples scored: {sum(1 for e in examples if e['id'] in jev and e['id'] in ours)} "
             f"({args.n_cu} computer-use + {args.n_gen} generic sampled, seed {args.seed})", "",
             "| group | n | Jev | ours | ours right / Jev wrong | Jev right / ours wrong |", "|---|---|---|---|---|---|"]
    for k, s in sorted(groups.items()):
        lines.append(f"| {k} | {s['n']} | {pct(s['jev'], s['n'])} | {pct(s['ours'], s['n'])} | {s['ours_only']} | {s['jev_only']} |")
    lines += ["", "| question | n | Jev | ours |", "|---|---|---|---|"]
    for k, s in sorted(per.items()):
        if s["n"] >= 20:
            lines.append(f"| {k} | {s['n']} | {pct(s['jev'], s['n'])} | {pct(s['ours'], s['n'])} |")
    if jl and ol:
        q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]
        lines += ["", f"Latency per request (all questions in one call): Jev via OpenRouter p50 {statistics.median(jl):.0f} ms, "
                      f"p95 {q(jl, .95):.0f} ms (includes network) · ours on M1 CPU (4 threads) p50 {statistics.median(ol):.0f} ms, "
                      f"p95 {q(ol, .95):.0f} ms",
                  f"Jev usage: {toks:,} input tokens, ${cost:.4f}"]
    report = "\n".join(lines)
    (OUT / "report.md").write_text(report + "\n")
    (OUT / "scores.json").write_text(json.dumps({"groups": groups, "per_qid": per}, indent=1))
    print(report)


if __name__ == "__main__":
    main()
