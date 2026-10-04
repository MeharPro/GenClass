"""CLI for the group-B adapters (VM for anything that downloads or loads a checkpoint):

  python -m jev_local.bench.benchmax.adapters_b.driver prepare   --spec SPEC --split validation
  python -m ...driver items     --spec SPEC --split validation --out items.jsonl [--limit N] [--tasks a,b]
  python -m ...driver run       --spec SPEC --split validation --engine local --ckpt ~/jev/models/jev-local-fast-v2
                                [--threads 8] [--limit N] [--tasks a,b] --out runs/benchmax-b/<spec>/validation
  python -m ...driver run       --spec SPEC --split test --engine http --url http://127.0.0.1:8765 --out ...
  python -m ...driver score     --spec SPEC --split validation --out <run dir> [--thresholds file.json]
  python -m ...driver fit       --spec SPEC --out <validation run dir>          (writes thresholds.json)
  python -m ...driver selfcheck --spec SPEC --ckpt ... [--limit 40] [--tasks ...]   (prepare, items, run, score on
                                the validation split; refuses the test split)

Every run directory gets items.jsonl, answers.jsonl, scores.json and run.json (PLAN §6 log: spec, split, tasks,
engine, checkpoint sha256, model id, counts truncated / failed, wall-clock, host).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

from jev_local.bench.benchmax.adapters_b.common import Item, read_jsonl, write_jsonl, work_dir


def _adapter(spec_id: str, work: str | None):
    from jev_local.bench.benchmax import specs_b

    cls = specs_b.adapter_class(spec_id)
    return cls(specs_b.spec(spec_id), argparse.Namespace(work=work))


def _ckpt_sha(ckpt: str | None) -> str | None:
    if not ckpt:
        return None
    p = Path(ckpt).expanduser()
    heads = p / "heads.safetensors"
    if not heads.exists():
        return None
    h = hashlib.sha256()
    with heads.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _answerer(a: argparse.Namespace):
    from jev_local.bench.benchmax.adapters_b import engine_client as ec

    if a.engine == "http":
        return ec.HttpAnswerer(a.url, model=a.model, api_key=os.environ.get("JEV_LOCAL_API_KEY"))
    if a.engine == "local":
        if sys.platform == "darwin":
            raise SystemExit("refusing to load torch on the Mac (CONTRACT hard rule 1); use --engine http or the VM")
        if not a.ckpt:
            raise SystemExit("--ckpt is required with --engine local")
        return ec.LocalAnswerer(a.ckpt, threads=a.threads, max_tokens=a.max_tokens, model=a.model)
    raise SystemExit(f"unknown engine {a.engine!r}")


def cmd_prepare(a: argparse.Namespace) -> None:
    ad = _adapter(a.spec, a.work)
    man = ad.prepare(a.split)
    print(json.dumps(man, indent=1, default=str)[:6000])


def cmd_items(a: argparse.Namespace) -> None:
    ad = _adapter(a.spec, a.work)
    items = ad.items(a.split, limit=a.limit, tasks=_tasks(a))
    n = write_jsonl(a.out, items)
    print(f"wrote {n} items to {a.out}")


def _tasks(a: argparse.Namespace) -> list[str] | None:
    return [t for t in a.tasks.split(",") if t] if getattr(a, "tasks", None) else None


def _guard_test(a: argparse.Namespace) -> None:
    if a.split == "test" and not getattr(a, "allow_test", False):
        raise SystemExit("split=test is the published evaluation split: pass --allow-test for a real release run "
                         "(PLAN §2.2: test is read once per release; adapters are debugged on validation)")


def cmd_run(a: argparse.Namespace) -> None:
    from jev_local.bench.benchmax.adapters_b import engine_client as ec

    _guard_test(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ad = _adapter(a.spec, a.work)
    t0 = time.time()
    items_path = out / "items.jsonl"
    if items_path.exists() and not a.rebuild_items:
        items = [Item.from_json(d) for d in read_jsonl(items_path)]
    else:
        ad.prepare(a.split)
        items = ad.items(a.split, limit=a.limit, tasks=_tasks(a))
        write_jsonl(items_path, items)
    answerer = _answerer(a)
    recs = ec.run_items(items, answerer, out / "answers.jsonl", resume=not a.no_resume)
    # multi-stage suites (zhuyansen banking77 group -> label): later stages depend on earlier answers
    for stage in range(2, getattr(ad, "STAGES", 1) + 1):
        extra = ad.stage_items(stage, items, recs)
        if not extra:
            break
        items = items + extra
        write_jsonl(items_path, items)
        recs = ec.run_items(items, answerer, out / "answers.jsonl", resume=True)
    thresholds = json.loads(Path(a.thresholds).read_text()) if a.thresholds else None
    scores = ad.score(items, recs, a.split, thresholds=thresholds)
    (out / "scores.json").write_text(json.dumps(scores, indent=1, default=str) + "\n")
    run = {
        "spec": a.spec, "split": a.split, "tasks": _tasks(a), "limit": a.limit, "n_items": len(items),
        "engine": a.engine, "engine_name": getattr(answerer, "name", None), "model": a.model, "ckpt": a.ckpt,
        "ckpt_heads_sha256": _ckpt_sha(a.ckpt), "threads": a.threads, "overflow": "truncate (counted)",
        "thresholds": a.thresholds, **ec.truncation_counts(recs.values()),
        "wall_s": round(time.time() - t0, 1), "host": platform.node(), "python": platform.python_version(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
    }
    (out / "run.json").write_text(json.dumps(run, indent=1) + "\n")
    print(json.dumps({k: v for k, v in scores.items() if k != "per_item"}, indent=1, default=str)[:8000])
    print(f"run: {json.dumps(run)}")


def cmd_score(a: argparse.Namespace) -> None:
    from jev_local.bench.benchmax.adapters_b import engine_client as ec

    out = Path(a.out)
    ad = _adapter(a.spec, a.work)
    items = [Item.from_json(d) for d in read_jsonl(out / "items.jsonl")]
    recs = ec.load_answers(out / "answers.jsonl")
    thresholds = json.loads(Path(a.thresholds).read_text()) if a.thresholds else None
    scores = ad.score(items, recs, a.split, thresholds=thresholds)
    (out / "scores.json").write_text(json.dumps(scores, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in scores.items() if k != "per_item"}, indent=1, default=str)[:8000])


def cmd_fit(a: argparse.Namespace) -> None:
    from jev_local.bench.benchmax.adapters_b import engine_client as ec

    out = Path(a.out)
    ad = _adapter(a.spec, a.work)
    items = [Item.from_json(d) for d in read_jsonl(out / "items.jsonl")]
    recs = ec.load_answers(out / "answers.jsonl")
    th = ad.fit(items, recs)
    if th is None:
        raise SystemExit(f"{a.spec} has no tuned variant")
    (out / "thresholds.json").write_text(json.dumps(th, indent=1) + "\n")
    print(json.dumps(th, indent=1)[:4000])


def cmd_selfcheck(a: argparse.Namespace) -> None:
    a.split = "validation" if a.split is None else a.split
    _guard_test(a)
    a.out = a.out or str(work_dir(a.work) / "selfcheck" / a.spec.replace("@", "_") / a.split)
    a.rebuild_items = True
    a.no_resume = False
    a.thresholds = None
    cmd_run(a)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser, split_default: str | None = "validation") -> None:
        p.add_argument("--spec", required=True)
        p.add_argument("--split", default=split_default)
        p.add_argument("--work", default=None, help="work dir (default $BENCHMAX_WORK or ~/bench_work_b)")
        p.add_argument("--tasks", default=None, help="comma-separated task subset")
        p.add_argument("--limit", type=int, default=None)

    def engine(p: argparse.ArgumentParser) -> None:
        p.add_argument("--engine", choices=("local", "http"), default="local")
        p.add_argument("--ckpt", default=os.environ.get("JEV_LOCAL_FAST_CKPT"))
        p.add_argument("--threads", type=int, default=8)
        p.add_argument("--max-tokens", type=int, default=None)
        p.add_argument("--url", default="http://127.0.0.1:8765")
        p.add_argument("--model", default=os.environ.get("BENCHMAX_MODEL", "genclass-68m"))
        p.add_argument("--allow-test", action="store_true")
        p.add_argument("--no-resume", action="store_true")
        p.add_argument("--rebuild-items", action="store_true")
        p.add_argument("--thresholds", default=None)

    p = sub.add_parser("prepare"); common(p); p.set_defaults(fn=cmd_prepare)
    p = sub.add_parser("items"); common(p); p.add_argument("--out", required=True); p.set_defaults(fn=cmd_items)
    p = sub.add_parser("run"); common(p); engine(p); p.add_argument("--out", required=True); p.set_defaults(fn=cmd_run)
    p = sub.add_parser("score"); common(p); p.add_argument("--out", required=True); p.add_argument("--thresholds", default=None)
    p.set_defaults(fn=cmd_score)
    p = sub.add_parser("fit"); common(p); p.add_argument("--out", required=True); p.set_defaults(fn=cmd_fit)
    p = sub.add_parser("selfcheck"); common(p, split_default=None); engine(p); p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_selfcheck)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
