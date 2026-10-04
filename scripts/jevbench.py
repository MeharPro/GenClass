"""jevbench v1 CLI: build | run-ours | run-jev | score | report   (docs/research/v2/PLAN.md §1).

Where each step runs (CONTRACT hard rules):
  build      VM train   python scripts/jevbench.py build [--keys a,b] [--roles test,dev,ref]
  run-ours   VM train   python scripts/jevbench.py run-ours --ckpt ~/jev/models/jev-local-fast --release m0 [--roles test,dev]
  run-jev    Mac        .venv/bin/python scripts/jevbench.py run-jev --release m0 [--variants main,choice,shuffle,bare]
                        (light httpx, <= 6 in flight, append-only cache, never prints the key; Jev is never run on dev)
  score      VM train   python scripts/jevbench.py score --release m0 --ours <file> [--jev jev.jsonl] [-B 2000]
  report     anywhere   python scripts/jevbench.py report --release m0   (renders REPORT.md from scores.json)

Layout: requests  bench/jevbench/<role>/<key>__<variant>.jsonl   (VM; the Mac copy lives in runs/jevbench/requests/)
        results   runs/jevbench/<release>/{jev.jsonl, ours__<name>.jsonl, scores.json, REPORT.md}
Jev output is evaluation-only (TypeSafe MCA §2.3(b)): never train, select, calibrate or filter on it.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import threading
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BENCH = ROOT / "bench" / "jevbench"
RUNS = ROOT / "runs" / "jevbench"
URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
KEY_PATH = Path.home() / ".jev-local" / "secrets" / "openrouter.key"
VARIANT_ORDER = ("main", "choice", "shuffle", "bare")
RETRY_STATUS = (408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529)
ACCOUNT_STATUS = (401, 402, 403)  # key / billing problems: abort the run, retry these ids on the next run


def iter_rows(path: Path) -> Iterator[dict]:
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def request_files(root: Path, roles: list[str], variants: list[str], keys: set[str] | None = None) -> list[Path]:
    from jev_local.bench.registry import ALL_DATASETS

    out = []
    for v in variants:  # variant-major, so the primary pass completes first
        for d in ALL_DATASETS:
            if d.role not in roles or (keys and d.key not in keys):
                continue
            for suf in (".jsonl", ".jsonl.gz"):
                p = root / d.role / f"{d.key}__{v}{suf}"
                if p.exists():
                    out.append(p)
                    break
    return out


def load_cache(path: Path, keep_failed_permanent: bool = True) -> dict[str, dict]:
    """id -> record. The newest record wins, but an ok record is never replaced by a later failure; transient
    failures are dropped (a re-run retries them), permanent ones (4xx) are kept unless asked otherwise."""
    cache: dict[str, dict] = {}
    if not path.exists():
        return cache
    for rec in iter_rows(path):
        prev = cache.get(rec["id"])
        if prev is None or not prev.get("ok"):
            cache[rec["id"]] = rec
    return {k: r for k, r in cache.items()
            if r.get("ok") or (keep_failed_permanent and r.get("permanent") and r.get("status") not in ACCOUNT_STATUS)}


# ------------------------------------------------------------------------------------------------ build


def cmd_build(a) -> None:
    from jev_local.bench import build

    argv = ["--out", str(a.out)] + (["--keys", a.keys] if a.keys else []) + ["--roles", a.roles]
    build.main(argv)


# ------------------------------------------------------------------------------------------------ ours


def _ours_worker(args) -> list[dict]:
    ckpt, threads, rows, max_tokens = args
    global _RUNNER  # one model per worker process
    if "_RUNNER" not in globals() or _RUNNER is None:
        from jev_local.bench.ours import OursRunner

        _RUNNER = OursRunner(ckpt, threads=threads, max_tokens=max_tokens)
    out = []
    for r in rows:
        try:
            ans, info = _RUNNER.answer(r["request"])
            out.append({"id": r["id"], "ok": True, "model": _RUNNER.name, "answers": ans, "ms": info["ms"],
                        "usage": {k: info[k] for k in ("input_tokens", "passes", "truncated")}})
        except Exception as e:  # count as failure, keep going
            out.append({"id": r["id"], "ok": False, "permanent": True, "error": f"{type(e).__name__}: {str(e)[:300]}"})
    return out


_RUNNER = None


def cmd_run_ours(a) -> None:
    import multiprocessing as mp

    rel = RUNS / a.release
    rel.mkdir(parents=True, exist_ok=True)
    name = a.name or Path(a.ckpt).name
    out_path = rel / f"ours__{name}.jsonl"
    cache = load_cache(out_path)
    files = request_files(Path(a.requests), a.roles.split(","), a.variants.split(","),
                          set(a.keys.split(",")) if a.keys else None)
    todo: list[dict] = []
    for p in files:
        todo += [r for r in iter_rows(p) if r["id"] not in cache]
    print(f"[ours] {name}: cached {len(cache)}, to run {len(todo)} from {len(files)} files", file=sys.stderr, flush=True)
    if not todo:
        return
    # Chunks of similar requests keep each worker's batches homogeneous; sort by dataset only (stable).
    chunks = [todo[i:i + a.chunk] for i in range(0, len(todo), a.chunk)]
    t0, done = time.time(), 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, maxtasksperchild=None) as pool, out_path.open("a") as f:
        for res in pool.imap_unordered(_ours_worker, [(a.ckpt, a.threads, c, a.max_tokens) for c in chunks]):
            for rec in res:
                f.write(json.dumps(rec) + "\n")
            f.flush()
            done += len(res)
            el = time.time() - t0
            print(f"[ours] {done}/{len(todo)} {done / el:.1f} req/s eta {(len(todo) - done) / max(done / el, 1e-9) / 60:.1f} min",
                  file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------ jev


def _ask_jev(client, key: str, row: dict, max_attempts: int) -> dict:
    body = {"model": MODEL, **row["request"]}
    delay, err, status = 2.0, "", None
    for attempt in range(max_attempts):
        t = time.perf_counter()
        try:
            r = client.post(URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=90)
        except Exception as e:  # network errors: retry
            err, status = type(e).__name__, None
        else:
            ms = (time.perf_counter() - t) * 1000
            status = r.status_code
            if status == 200:
                try:
                    d = r.json()
                    return {"id": row["id"], "ok": True, "ms": ms, "model": d.get("model"), "answers": d.get("answers"),
                            "usage": d.get("usage", {}), "t": time.time(), "attempts": attempt + 1}
                except ValueError:
                    err = "bad json"
            else:
                err = f"HTTP {status}: {r.text[:300]}"
                if status in ACCOUNT_STATUS:
                    return {"id": row["id"], "ok": False, "permanent": False, "account": True, "status": status,
                            "error": err, "t": time.time()}
                if status not in RETRY_STATUS:
                    return {"id": row["id"], "ok": False, "permanent": True, "status": status, "error": err, "t": time.time()}
        time.sleep(delay)
        delay = min(delay * 2, 60)
    return {"id": row["id"], "ok": False, "permanent": False, "status": status, "error": err, "t": time.time()}


def cmd_run_jev(a) -> None:
    import httpx

    rel = RUNS / a.release
    rel.mkdir(parents=True, exist_ok=True)
    out_path = rel / "jev.jsonl"
    cache = load_cache(out_path)
    done_ids = set(cache)
    del cache  # keep only ids in memory (the Mac has little RAM)
    roles = [r for r in a.roles.split(",") if r != "dev"]  # Jev is never run on dev (PLAN §1.2)
    files = request_files(Path(a.requests), roles, a.variants.split(","), set(a.keys.split(",")) if a.keys else None)
    key = KEY_PATH.read_text().strip()
    workers = min(a.workers, 6)
    stats = Counter()
    lock = threading.Lock()
    t0 = time.time()
    log = (rel / "jev.log").open("a")

    def note(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, file=sys.stderr, flush=True)
        log.write(line + "\n")
        log.flush()

    def todo_rows() -> Iterator[dict]:
        for p in files:
            n = 0
            for r in iter_rows(p):
                if a.limit_per_file and n >= a.limit_per_file:
                    break
                n += 1
                if r["id"] not in done_ids:
                    yield r

    note(f"[jev] start: {len(done_ids)} cached, files={len(files)}, workers={workers}, limit/file={a.limit_per_file}")
    with httpx.Client(http2=False, limits=httpx.Limits(max_connections=workers)) as client, \
            out_path.open("a") as f, ThreadPoolExecutor(workers) as pool:
        inflight: set = set()
        it = todo_rows()
        exhausted = False
        while True:
            while not exhausted and len(inflight) < workers * 2:
                r = next(it, None)
                if r is None:
                    exhausted = True
                    break
                inflight.add(pool.submit(_ask_jev, client, key, r, a.attempts))
            if not inflight:
                break
            fin, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in fin:
                rec = fut.result()
                with lock:
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                stats["ok" if rec["ok"] else ("perm" if rec.get("permanent") else "fail")] += 1
                stats["cost"] += float((rec.get("usage") or {}).get("cost", 0) or 0)
                stats["in_tok"] += int((rec.get("usage") or {}).get("input_tokens", 0) or 0)
                n = stats["ok"] + stats["perm"] + stats["fail"]
                if not rec["ok"] and (stats["perm"] + stats["fail"]) <= 20:
                    note(f"[jev] error {rec['id']}: {rec.get('error', '')[:200]}")
                if n % 500 == 0:
                    el = time.time() - t0
                    note(f"[jev] {n} done ({stats['ok']} ok, {stats['perm']} perm, {stats['fail']} transient) "
                         f"{n / el * 60:.0f} req/min, cost ${stats['cost']:.3f}, {stats['in_tok'] / 1e6:.1f}M tok")
                if rec.get("account") and not exhausted:
                    note(f"[jev] account-level HTTP {rec.get('status')} (key/billing): stopping; "
                         f"these ids stay pending. {rec.get('error', '')[:160]}")
                    exhausted = True
                if a.max_cost and stats["cost"] > a.max_cost and not exhausted:
                    note(f"[jev] cost cap ${a.max_cost} reached; draining {len(inflight)} in-flight requests, then stopping")
                    exhausted = True  # stop submitting; in-flight answers are still written (they are paid for)
    note(f"[jev] finished: {dict(stats)} in {(time.time() - t0) / 60:.1f} min")


# ------------------------------------------------------------------------------------------------ score


def _answers_for(path: Path, ids: set[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for rec in iter_rows(path):
        if rec["id"] in ids and (rec.get("ok") or rec["id"] not in out):
            out[rec["id"]] = rec
    return out


def cmd_score(a) -> None:
    import numpy as np

    from jev_local.bench import metrics as M
    from jev_local.bench.registry import TEST, REF, DEV, BY_KEY

    rel = RUNS / a.release
    req_root = Path(a.requests)
    systems: dict[str, Path] = {}
    if a.jev:
        systems["jev"] = rel / a.jev
    for o in a.ours.split(","):
        if o:
            systems[o.removeprefix("ours__").removesuffix(".jsonl")] = rel / o
    roles = a.roles.split(",")
    datasets = [d for d in (*TEST, *REF, *DEV) if d.role in roles]
    rows_by: dict[tuple[str, str], list[dict]] = {}
    for d in datasets:
        for v in VARIANT_ORDER:
            p = req_root / d.role / f"{d.key}__{v}.jsonl"
            if p.exists():
                rows_by[(d.key, v)] = list(iter_rows(p))
    all_ids = {r["id"] for rows in rows_by.values() for r in rows}
    answers = {s: _answers_for(p, all_ids) for s, p in systems.items()}
    rounding = {s: (None if s == "jev" else 2) for s in systems}  # Jev already returns 2 dp
    rng = np.random.default_rng(a.seed)
    out: dict = {"release": a.release, "systems": list(systems), "B": a.B, "seed": a.seed, "datasets": {},
                 "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    scores_by_role: dict[str, list] = {"test": [], "ref": [], "dev": []}
    for d in datasets:
        if (d.key, "main") not in rows_by:
            out["datasets"][d.key] = {"skipped": "not built"}
            continue
        head_key = (lambda r, q: f"{r['stratum']}/{q}") if d.key == "typed_decisions" else \
            ((lambda r, q: r["item"]) if d.key == "tasksource_heldout" else
             ((lambda r, q: f"K{len(r['request']['questions'][q]['criteria'])}") if d.key == "mmlu_pro" else None))
        present = [s for s in systems if any(r["id"] in answers[s] for r in rows_by[(d.key, "main")])]
        if not present:
            out["datasets"][d.key] = {"skipped": "no system answered"}
            continue
        preds = {s: M.collect(rows_by[(d.key, "main")], answers[s], rounding[s], head_key) for s in present}
        entry: dict = {"role": d.role, "area": d.area, "num": d.num, "title": d.title, "n_items": len(preds[present[0]].items)}
        prim = M.primary_for(d.key, d.primary_metric, d.question_kinds)
        if d.key != "tasksource_heldout":
            sc = M.score_dataset(d.key, d.area, prim, preds, a.B if d.role == "test" else min(a.B, 500), rng)
            scores_by_role[d.role].append(sc)
            entry.update(primary=prim.name, chance=sc.chance, metric=sc.metric, skill=sc.skill, ds=sc.ds,
                         metric_ci={s: M.ci(sc.boot_metric[s]) for s in present},
                         skill_ci={s: M.ci(sc.boot_skill[s]) for s in present},
                         ds_ci={s: M.ci(sc.boot_ds[s]) for s in present})
            if len(present) >= 2 and "jev" in present:
                for s in present:
                    if s != "jev":
                        entry.setdefault("diff_ci", {})[s] = M.ci(sc.boot_metric[s] - sc.boot_metric["jev"])
        else:
            entry.update(primary="acc", metric={s: M._pooled_acc(preds[s].heads) for s in present})
        entry["coverage"] = {s: {"requests": preds[s].n_requests, "failed": preds[s].n_failed_requests,
                                 "renormalised": preds[s].renormalised} for s in present}
        if d.key != "tasksource_heldout":
            entry["heads"] = {s: {k: M.head_metrics(h) for k, h in preds[s].heads.items()} for s in present}
        for v in ("shuffle", "bare", "choice"):
            if (d.key, v) not in rows_by:
                continue
            order = M.main_label_order(rows_by[(d.key, "main")]) if v != "choice" else None
            for s in present:
                pv = M.collect(rows_by[(d.key, v)], answers[s], rounding[s], head_key, order)
                if pv.n_requests == pv.n_failed_requests:
                    continue
                if v == "shuffle":
                    entry.setdefault("flip", {})[s] = M.flip_rate(preds[s], pv)
                else:
                    pr = prim if v == "bare" else M.PRIMARY["acc"]
                    entry.setdefault(v, {})[s] = {"metric": pr.fn(pv.heads), "failed": pv.n_failed_requests}
        lat = {}
        for s in present:
            ms = sorted(r["ms"] for i, r in answers[s].items() if r.get("ok") and i.startswith(d.key + "/main/"))
            if ms:
                lat[s] = {"p50": ms[len(ms) // 2], "p95": ms[min(len(ms) - 1, int(0.95 * len(ms)))]}
        entry["latency_ms"] = lat
        out["datasets"][d.key] = entry
        print(f"[score] {d.key}: " + ", ".join(f"{s}={entry.get('metric', {}).get(s, float('nan')):.3f}" for s in present),
              file=sys.stderr, flush=True)
    test_scores = scores_by_role["test"]
    # A system enters the indices / decision rule only with >= 99% valid main-variant requests on every counted
    # dataset (PLAN §5 M0 exit criterion); otherwise its incomplete datasets are listed instead.
    incomplete: dict[str, list[str]] = {}
    for s in systems:
        for sc in test_scores:
            cov = out["datasets"][sc.key].get("coverage", {}).get(s)
            if s not in sc.metric or not cov or cov["failed"] > 0.01 * cov["requests"]:
                incomplete.setdefault(s, []).append(sc.key)
    out["incomplete"] = incomplete
    sys_test = [s for s in systems if s not in incomplete] if test_scores else []
    if test_scores:
        out["index"] = {s: {"skill": M.index(test_scores, s, "skill"), "ds": M.index(test_scores, s, "ds"),
                            "skill_ci": M.ci(M.index(test_scores, s, "skill", True)),
                            "ds_ci": M.ci(M.index(test_scores, s, "ds", True))} for s in sys_test}
        out["areas_skill"] = M.area_table(test_scores, sys_test, "skill")
        out["areas_ds"] = M.area_table(test_scores, sys_test, "ds")
        if "jev" in sys_test:
            out["decision_rule"] = {s: M.decision_rule(test_scores, s, "jev") for s in sys_test if s != "jev"}
    meta = {}
    for s in systems:
        recs = answers[s].values()
        models = Counter(r.get("model") for r in recs if r.get("ok"))
        meta[s] = {"models": dict(models), "records": len(answers[s]), "ok": sum(1 for r in recs if r.get("ok")),
                   "cost": sum(float((r.get("usage") or {}).get("cost", 0) or 0) for r in recs),
                   "input_tokens": sum(int((r.get("usage") or {}).get("input_tokens", 0) or 0) for r in recs),
                   "truncated": sum(1 for r in recs if (r.get("usage") or {}).get("truncated")),
                   "multi_pass": sum(1 for r in recs if ((r.get("usage") or {}).get("passes") or 1) > 1),
                   "errors": dict(Counter(str(r.get("status") or r.get("error", "")[:40]) for r in recs if not r.get("ok")))}
    out["system_meta"] = meta
    (rel / a.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {rel / a.out}", file=sys.stderr)


# ------------------------------------------------------------------------------------------------ report


def _f(x, d=3):
    return "–" if x is None or (isinstance(x, float) and x != x) else f"{x:.{d}f}"


def _valid(v: dict, s: str) -> bool:
    c = v.get("coverage", {}).get(s)
    return bool(c) and c["failed"] <= 0.01 * c["requests"]


def _variant_valid(v: dict, var: str, s: str) -> bool:
    x = v.get(var, {}).get(s)
    if not x:
        return False
    n = v.get("n_items", 0)
    return x.get("n", n) >= 0.99 * n if var == "flip" else x.get("failed", 0) <= 0.01 * n


def _prereg_check() -> list[str]:
    """sha256 of PREREG.md and which pre-registered code files have changed since (by hash)."""
    import hashlib
    import re

    pre = BENCH / "PREREG.md"
    if not pre.exists():
        return ["No PREREG.md found."]
    txt = pre.read_text()
    out = [f"PREREG.md sha256 `{hashlib.sha256(pre.read_bytes()).hexdigest()}`."]
    files = {"registry.py": ROOT / "jev_local/bench/registry.py", "templates.py": ROOT / "jev_local/bench/templates.py",
             "build.py": ROOT / "jev_local/bench/build.py", "metrics.py": ROOT / "jev_local/bench/metrics.py",
             "ours.py": ROOT / "jev_local/bench/ours.py", "scripts/jevbench.py": ROOT / "scripts/jevbench.py"}
    changed = []
    for name, path in files.items():
        m = re.search(re.escape(f"`{name}` `") + r"([0-9a-f]{64})`", txt)
        if m and path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != m.group(1):
            changed.append(name)
    out.append("Code identical to the registered hashes." if not changed else
               f"Changed since registration (hash differs): {', '.join(changed)} — see the release notes for why; "
               "a release read must state that no registered primary definition changed.")
    return out


def cmd_report(a) -> None:
    rel = RUNS / a.release
    S = json.loads((rel / a.scores).read_text())
    systems = S["systems"]
    ours = [s for s in systems if s != "jev"]
    L = [f"# jevbench v1 — release read {a.release}", "",
         f"Generated {S['generated']} from `{a.scores}`; paired item-clustered, target-stratified bootstrap, "
         f"B = {S['B']}, seed {S['seed']}. Systems: {', '.join(systems)}. "
         "Pre-registration: `bench/jevbench/PREREG.md`. " + " ".join(_prereg_check()), ""]
    for s, ks in S.get("incomplete", {}).items():
        errs = S.get("system_meta", {}).get(s, {}).get("errors", {})
        why = f" Failure statuses: {', '.join(f'{k} × {n}' for k, n in errs.items())}." if errs else ""
        L += [f"> **{s} is incomplete** on {len(ks)} counted dataset(s) (< 99% valid responses): {', '.join(ks)}. "
              f"It is left out of the indices and the decision rule, and its cells are blank (with the valid share) "
              f"wherever it is below 99% valid, until its pass completes.{why}", ""]
    if "index" in S:
        L += ["## Indices (test, counted datasets)", "", "| system | skill index | 95% CI | Decision-Score index | 95% CI |",
              "|---|---|---|---|---|"]
        for s, v in S["index"].items():
            L.append(f"| {s} | {_f(v['skill'], 1)} | [{_f(v['skill_ci'][0], 1)}, {_f(v['skill_ci'][1], 1)}] | "
                     f"{_f(v['ds'], 1)} | [{_f(v['ds_ci'][0], 1)}, {_f(v['ds_ci'][1], 1)}] |")
        if "decision_rule" in S:
            L += ["", "### Decision rule (PLAN §0.1)", ""]
            for s, r in S["decision_rule"].items():
                L += [f"- **{s} vs Jev**: claim = **{r['claim']}**.",
                      f"  - (a) skill-index diff CI [{_f(r['skill_diff_ci'][0], 1)}, {_f(r['skill_diff_ci'][1], 1)}], "
                      f"DS-index diff CI [{_f(r['ds_diff_ci'][0], 1)}, {_f(r['ds_diff_ci'][1], 1)}] → {r['a_ci_excludes_0']}",
                      f"  - (b) dataset wins {r['b_wins']} (need ≥ 13) → {r['b_ok']}: {', '.join(r['wins']) or 'none'}",
                      f"  - (c) worst area gap {_f(r['c_worst_area_gap'], 1)} skill points (need ≥ −10) → {r['c_ok']}"]
        L += ["", "## Areas (skill × 100 / Decision Score × 100)", "", "| area | " + " | ".join(f"{s} skill | {s} DS" for s in S["index"]) + " |",
              "|---|" + "---|---|" * len(S["index"])]
        from jev_local.bench.registry import AREAS

        for ar in sorted(S["areas_skill"]):
            L.append(f"| {ar} {AREAS.get(ar, '')} | " + " | ".join(
                f"{_f(S['areas_skill'][ar][s], 1)} | {_f(S['areas_ds'][ar][s], 1)}" for s in S["index"]) + " |")
    for role, title in (("test", "Per dataset (test)"), ("ref", "Shown, not counted (knowledge reference)"), ("dev", "jevbench-dev (ours only)")):
        ds = [(k, v) for k, v in S["datasets"].items() if v.get("role") == role or (role == "test" and v.get("skipped") and k in _test_keys())]
        if not ds:
            continue
        L += ["", f"## {title}", "", "| # | dataset | metric | chance | " + " | ".join(f"{s}" for s in systems) +
              " | " + " | ".join(f"Δ {s}−jev [95% CI]" for s in ours if "jev" in systems) + " | DS " + " / ".join(systems) + " |",
              "|---|---|---|---|" + "---|" * len(systems) + "---|" * (len(ours) if "jev" in systems else 0) + "---|"]
        for k, v in ds:
            if v.get("skipped"):
                L.append(f"| | {k} | skipped: {v['skipped'][:80]} | | " + " | ".join("" for _ in systems) + " |")
                continue
            m = v.get("metric", {})
            cov = v.get("coverage", {})

            def cell(s):
                c = cov.get(s)
                if s not in m:
                    return "–"
                if not _valid(v, s):
                    return f"n/a ({100 * (1 - c['failed'] / c['requests']):.0f}% valid)"
                return _f(m[s])
            row = f"| {v.get('num') or ''} | {v['title']} | {v.get('primary', 'acc')} | {_f(v.get('chance'))} | " + \
                  " | ".join(cell(s) for s in systems)
            if "jev" in systems and role != "dev":
                row += " | " + " | ".join(
                    f"{_f(m[s] - m['jev'])} [{_f(v['diff_ci'][s][0])}, {_f(v['diff_ci'][s][1])}]"
                    if s in m and "jev" in m and _valid(v, s) and _valid(v, "jev") and s in v.get("diff_ci", {}) else "–"
                    for s in ours)
            row += " | " + " / ".join(_f(v.get("ds", {}).get(s), 2) if _valid(v, s) else "–" for s in systems) + " |"
            L.append(row)
    test = [(k, v) for k, v in S["datasets"].items() if v.get("role") == "test" and not v.get("skipped")]
    if test:
        L += ["", "## Calibration and robustness (test, main variant, pooled over heads)", "",
              "| dataset | " + " | ".join(f"{s} ECE15 / NLL / Brier" for s in systems) + " | " +
              " | ".join(f"{s} flip / TV" for s in systems) + " | " + " | ".join(f"{s} bare Δ" for s in systems) + " |",
              "|---|" + "---|" * (3 * len(systems))]
        for k, v in test:
            cells = []
            for s in systems:
                hs = v.get("heads", {}).get(s, {}) if _valid(v, s) else {}
                if hs:
                    n = sum(h["n"] for h in hs.values())
                    w = lambda f: sum(h[f] * h["n"] for h in hs.values() if h.get(f) == h.get(f)) / max(n, 1)
                    cells.append(f"{_f(w('ece15'))} / {_f(w('nll'), 2)} / {_f(w('brier'))}")
                else:
                    cells.append("–")
            for s in systems:
                fl = v.get("flip", {}).get(s)
                cells.append(f"{_f(fl['flip_rate'])} / {_f(fl['mean_tv'])}" if fl and _variant_valid(v, "flip", s) else "–")
            for s in systems:
                b = v.get("bare", {}).get(s)
                ok = b and _valid(v, s) and _variant_valid(v, "bare", s)
                cells.append(_f(b["metric"] - v["metric"][s]) if ok else "–")
            L.append(f"| {v['title']} | " + " | ".join(cells) + " |")
        L += ["", "## Coverage and latency", "", "| system | records | ok | resolved model ids | input tokens | cost $ | truncated | multi-pass |",
              "|---|---|---|---|---|---|---|---|"]
        for s, m in S.get("system_meta", {}).items():
            L.append(f"| {s} | {m['records']} | {m['ok']} | {', '.join(f'{k} ({n})' for k, n in m['models'].items())} | "
                     f"{m['input_tokens']:,} | {m['cost']:.2f} | {m['truncated']} | {m['multi_pass']} |")
        L += ["", "| dataset | " + " | ".join(f"{s} p50 / p95 ms" for s in systems) + " | " +
              " | ".join(f"{s} failed req" for s in systems) + " | SST-2 choice / other notes |", "|---|" + "---|" * (2 * len(systems) + 1)]
        for k, v in test:
            lat = v.get("latency_ms", {})
            cov = v.get("coverage", {})
            note = ""
            if "choice" in v:
                note = "choice-variant acc " + ", ".join(f"{s} {_f(x['metric'])}" for s, x in v["choice"].items()
                                                         if _variant_valid(v, "choice", s))
            L.append(f"| {v['title']} | " + " | ".join(f"{_f(lat[s]['p50'], 0)} / {_f(lat[s]['p95'], 0)}"
                                                         if s in lat and _valid(v, s) else "–" for s in systems)
                     + " | " + " | ".join(str(cov.get(s, {}).get("failed", "–")) for s in systems) + f" | {note} |")
    if a.note:
        L += ["", "## Release notes", ""] + [f"- {n}" for n in a.note]
    (rel / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"wrote {rel / 'REPORT.md'}", file=sys.stderr)


def _test_keys() -> set[str]:
    from jev_local.bench.registry import TEST

    return {d.key for d in TEST}


# ------------------------------------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, default=BENCH)
    b.add_argument("--keys", default="")
    b.add_argument("--roles", default="test,dev,ref")
    o = sub.add_parser("run-ours")
    o.add_argument("--ckpt", required=True)
    o.add_argument("--name", default="")
    o.add_argument("--release", required=True)
    o.add_argument("--requests", default=str(BENCH))
    o.add_argument("--roles", default="test,ref")
    o.add_argument("--variants", default="main,choice,shuffle,bare")
    o.add_argument("--keys", default="")
    o.add_argument("--workers", type=int, default=8)
    o.add_argument("--threads", type=int, default=2)
    o.add_argument("--chunk", type=int, default=64)
    o.add_argument("--max-tokens", type=int, default=0)
    j = sub.add_parser("run-jev")
    j.add_argument("--release", required=True)
    j.add_argument("--requests", default=str(RUNS / "requests"))
    j.add_argument("--roles", default="test")
    j.add_argument("--variants", default="main,choice,shuffle,bare")
    j.add_argument("--keys", default="")
    j.add_argument("--workers", type=int, default=6)
    j.add_argument("--attempts", type=int, default=6)
    j.add_argument("--limit-per-file", type=int, default=0)
    j.add_argument("--max-cost", type=float, default=8.0, help="stop when the session's Jev spend exceeds this ($)")
    s = sub.add_parser("score")
    s.add_argument("--release", required=True)
    s.add_argument("--requests", default=str(BENCH))
    s.add_argument("--ours", default="")
    s.add_argument("--jev", default="")
    s.add_argument("--roles", default="test,ref")
    s.add_argument("-B", type=int, default=2000)
    s.add_argument("--seed", type=int, default=20261001)
    s.add_argument("--out", default="scores.json")
    r = sub.add_parser("report")
    r.add_argument("--release", required=True)
    r.add_argument("--scores", default="scores.json")
    r.add_argument("--note", action="append", default=[], help="release note bullet (repeatable)")
    a = ap.parse_args()
    {"build": cmd_build, "run-ours": cmd_run_ours, "run-jev": cmd_run_jev, "score": cmd_score, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
