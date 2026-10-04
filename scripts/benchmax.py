"""benchmax CLI: run meharsjev through a publisher's harness and log the run (PLAN §6).

  python scripts/benchmax.py list
  python scripts/benchmax.py verify --spec <id> [adapter options] [--out DIR]
  python scripts/benchmax.py run --spec <id> (--ckpt DIR | --base-url URL | --engine heuristic) [--calib FILE]
         --out runs/benchmax/<ckpt>/<spec>/ [--model-id meharsjev-68m] [--overflow refuse|truncate]
         [--precision unrounded|2dp] [--threads N] [--refusal-status 400|422] [--determinism N] [--trace]
         [--rerun-reason TEXT] [adapter options; see `run --spec <id> --help`]

Where it runs (CONTRACT hard rule 1): anything that loads a checkpoint or a dataset runs on VM train; the Mac
runs only `list`, `--help` and the pure-Python tests. Long runs: `setsid nohup nice -n 5 ... &` with a log.

Group-B specs (prepare / items / score contract) run through `jev_local.bench.benchmax.bridge_b`, so every one of the
19 specs writes the same run.json; their options are `--split/--limit/--tasks/--work/--thresholds/--fit/--allow-test`.

Hygiene: test splits are read once per release (PLAN §2.2.2). A second `run` into an `--out` that already holds
a complete run.json needs `--rerun-reason`; the earlier run.json is kept as run.<timestamp>.json and the reason
is recorded. Debug on `--split dev|validation|train` only.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_local.bench.benchmax import get_spec, load_specs, runner_adapter_class  # noqa: E402
from jev_local.bench.benchmax.runner import (  # noqa: E402
    OVERFLOWS, PRECISIONS, RunContext, Session, build_client, check_model_id, default_out, hardware_info, sha256_file, utc_now,
)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--spec", required=True, help="spec id (see `list`)")
    p.add_argument("--out", help="run directory (default runs/benchmax/<ckpt>/<spec>/)")
    p.add_argument("--model-id", default="genclass-68m", help="W10: genclass-<size>")
    p.add_argument("--threads", type=int, default=8, help="torch threads for the local engine (leave ~16 cores free on VM train)")


def _engine(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("engine")
    g.add_argument("--ckpt", help="FastEngine checkpoint dir (in-process; VM only)")
    g.add_argument("--base-url", help="a /v1/systemone server, e.g. http://127.0.0.1:8765")
    g.add_argument("--engine", choices=["local", "http", "heuristic"], help="default: local if --ckpt, http if --base-url")
    g.add_argument("--wire-model", help="model id put on the wire for --base-url when the server does not accept meharsjev-* yet (disclosed)")
    g.add_argument("--calib", help="W9: calibration file for this run (replaces the checkpoint's calibration.json in-process)")
    g.add_argument("--max-tokens", type=int, help="override the engine's context (positions) limit")
    g.add_argument("--overflow", choices=OVERFLOWS, help="W5 (default: the spec's rule)")
    g.add_argument("--precision", choices=PRECISIONS, default="unrounded", help="W3 (leaderboard mode = unrounded)")
    g.add_argument("--refusal-status", type=int, choices=[400, 422], default=400, help="W4 status of in-process refusals")
    g.add_argument("--determinism", type=int, default=20, help="W11: re-send the first N requests at the end")
    g.add_argument("--trace", action="store_true", help="write every request/response to <out>/trace.jsonl")
    g.add_argument("--rerun-reason", help="required when <out>/run.json already holds a complete run")
    g.add_argument("--track", choices=["Z", "S"], default="Z", help="recorded only (which scoreboard the run is for)")


def build_parser(argv: list[str]) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="benchmax", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="known specs")
    v = sub.add_parser("verify", help="hash / commit / count checks for one spec (no model)")
    _common(v)
    r = sub.add_parser("run", help="run one spec and write run.json")
    _common(r)
    _engine(r)
    # adapter-specific options: find --spec early and let its adapter extend the right subparser
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--spec")
    known, _ = pre.parse_known_args(argv[1:] if argv and argv[0] in ("run", "verify") else [])
    if known.spec:
        try:
            cls = runner_adapter_class(get_spec(known.spec))
        except KeyError as e:
            ap.error(str(e))
        for p in (v, r):
            cls.add_arguments(p.add_argument_group(f"{known.spec} options"))
    return ap


def cmd_list() -> int:
    specs = load_specs()
    for sid, s in sorted(specs.items()):
        print(f"{sid:40s} owner={s.owner:2s} rows={s.counted_rows:16s} {s.suite}")
        print(f"{'':40s} adapter={s.adapter}")
        if s.verification:
            print(f"{'':40s} verify: {s.verification}")
    return 0


def _args_record(args: argparse.Namespace) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in ("command",)}


def cmd_verify(args: argparse.Namespace) -> int:
    spec = get_spec(args.spec)
    cls = runner_adapter_class(spec)
    out = Path(args.out) if args.out else default_out(spec.id, args)
    ctx = RunContext(spec, out, check_model_id(args.model_id), _args_record(args))
    rep = cls(spec, args).verify(ctx)
    rep["dataset_revisions"] = ctx.dataset_revisions
    rep["checked_at"] = utc_now()
    print(json.dumps(rep, indent=1, default=str))
    if args.out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "verify.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    return 0 if rep.get("ok") else 1


def cmd_run(args: argparse.Namespace) -> int:
    spec = get_spec(args.spec)
    cls = runner_adapter_class(spec)
    if args.overflow is None:
        args.overflow = cls.default_overflow
    if args.overflow == "truncate" and not cls.allow_truncate:
        raise SystemExit(f"{spec.id}: truncation is forbidden by the suite; use --overflow refuse")
    model_id = check_model_id(args.model_id)
    out = Path(args.out) if args.out else default_out(spec.id, args)
    out.mkdir(parents=True, exist_ok=True)
    prev = out / "run.json"
    notes: list[str] = []
    if prev.exists():
        try:
            old = json.loads(prev.read_text())
        except ValueError:
            old = {}
        if old.get("status") == "complete":
            if not args.rerun_reason:
                raise SystemExit(f"{prev} holds a complete run; a rerun needs --rerun-reason (PLAN §2.2.2: first complete run is the one reported)")
            stamp = (old.get("finished") or utc_now()).replace(":", "")
            prev.rename(out / f"run.{stamp}.json")
            notes.append(f"RERUN of a complete run ({stamp}); reason: {args.rerun_reason}. The first complete run stays the reported one.")
        else:
            notes.append(f"resuming / replacing an incomplete run (status={old.get('status')!r})")
    ctx = RunContext(spec, out, model_id, _args_record(args), precision=args.precision)
    ctx.overflow = {"mode": args.overflow, "truncated": 0, "refused": 0}
    ctx.notes.extend(notes)
    ctx.hardware = hardware_info(args.threads)
    adapter = cls(spec, args)
    try:
        ctx.harness = adapter.verify(ctx)
    except Exception as e:  # verification problems are recorded, the adapter decides whether to proceed
        ctx.note(f"verify failed: {type(e).__name__}: {e}")
        ctx.harness = {"error": f"{type(e).__name__}: {e}"}
    client = build_client(args)
    ctx.engine = client.describe()
    if hasattr(client, "ckpt_hash"):
        ctx.engine["ckpt_sha256"] = client.ckpt_hash()
    calib = {"path": args.calib, "sha256": sha256_file(args.calib) if args.calib and Path(args.calib).exists() else None}
    calib["applied"] = "in-process (replaces checkpoint calibration.json)" if (args.calib and client.kind == "local") else (
        "server-side: the server must have been started with it (declared only)" if args.calib else "checkpoint default (one global file)")
    ctx.calibration = calib
    ctx.hardware = hardware_info(args.threads)  # torch threads are known only after the client loaded
    ctx.write()
    session = Session(client, ctx, keep_first=args.determinism, trace=(out / "trace.jsonl") if args.trace else None)
    try:
        ctx.results = adapter.run(session, ctx)
        if session.kept:
            session.determinism_check()
        session.close()
        ctx.finish("complete")
    except SystemExit as e:
        ctx.note(f"stopped: {e}")
        ctx.finish("failed")
        raise
    except BaseException:
        ctx.note("failed:\n" + traceback.format_exc())
        ctx.finish("failed")
        raise
    summary = {"run_json": str(ctx.path), "status": ctx.status, "requests": ctx.requests, "overflow": ctx.overflow,
               "determinism": ctx.determinism, "wall_clock_s": ctx.to_dict()["wall_clock_s"]}
    print(json.dumps(summary, indent=1, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser(argv)
    args = ap.parse_args(argv)
    if args.command == "list":
        return cmd_list()
    if args.command == "verify":
        return cmd_verify(args)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
