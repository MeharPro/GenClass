"""`decision_index_0.2.1@87d4650b`: the Decision Index reproduction kit, driven unchanged (specs §3.4; PLAN §6).

Route: `python -m decision_index pipeline --engine http --option base_url=… --option model=meharsjev-<size>
--edition 0.2.1` over the rebuilt, hash-verified suite, then its own `score --edition 0.2.1`. The kit sends
exactly `state` and `questions`; nothing is edited. The alternative `--di-engine inproc` passes
`--engine jev_local.bench.benchmax.adapters_a.decision_index:MeharsjevEngine`, a kit `Engine` subclass around
our in-process client that raises the kit's `Unsupported` on a refusal (so W4 semantics never depend on the
server's wording).

Rules encoded here (README "Rules", docs/engines.md):
- no truncation: overflow is forced to `refuse` and a W4 refusal is recorded `unsupported` (counts as wrong);
- unrounded probabilities (W3): `validate` needs |Σp − 1| ≤ 0.01 and `choice` among the options;
- one fixed rendering, the global calibration file only (W9), model id `meharsjev-*` (W10);
- complete untouched runs: `scores.json` must say `"complete": true` for a board-comparable index.

Hash checks (`verify`): rebuilt `selected-rows.jsonl` uncompressed sha256 b2b56d6f…, `added-rows.jsonl`
7429f3c9…, exclusions 331df32d…, kit commit 87d4650b…; the pipeline refuses to start on a mismatch unless
`--allow-unverified` (recorded). Jev's per-benchmark raw values come from the kit fixture
`tests/fixtures/board-0.2.1.json` (index 57.89; the Space says 57.91, PLAN §9).

The suite is evaluation data under 43 licences: never redistributed, never trained on, never inspected here.
Runs on VM train (`$BENCHMAX_WORK/venv-di` holds the kit's `[rebuild]` extras).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from jev_local.bench.benchmax.runner import DEFAULT_WORK, REPO_ROOT, Adapter, RunContext, Session, check_commit, sha256_file

REPO = "https://github.com/apolinario/decision-index"
PINNED_COMMIT = "87d4650b42b377c0291a89c1f1a879f9b31082bf"
LICENCE = "MIT (kit); suite texts under 43 upstream licences"
EDITION = "0.2.1"
ROWS_SHA256 = "b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5"  # selected-rows.jsonl, uncompressed
ADDED_SHA256 = "7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89"  # added-rows.jsonl, uncompressed
EXCLUSIONS_SHA256 = "331df32d4b719c7db43214d0e5d85859d39c3b2eb7d0b3812214cce150155e81"
LAB_ROWS_GZ_SHA256 = "25aac5e890a54a3172c7a0c184b4cc8b9a43f10b6ee89bbad8da923be423c656"
SCOREABLE, ADDED_REQUESTS = 119_898, 30_419
JEV_FIXTURE = "tests/fixtures/board-0.2.1.json"
JEV_INDEX_FIXTURE, JEV_INDEX_SPACE = 57.89, 57.91
REBUILT_REL = Path("artifacts") / "benchmark-suite" / "release-v2-rebuilt"


def sha256_gunzipped(path: str | Path) -> str:
    h = hashlib.sha256()
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:  # type: ignore[arg-type]
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jev_fixture_values(kit_dir: str | Path) -> dict[str, Any]:
    """Per-benchmark Jev raw / skill / coverage from the kit's board fixture (entrant named Jev)."""
    p = Path(kit_dir) / JEV_FIXTURE
    if not p.exists():
        return {}
    board = json.loads(p.read_text())
    ents = board.get("entrants") or []
    if isinstance(ents, dict):  # the fixture keys entrants by id
        ents = [dict(v, _id=k) if isinstance(v, dict) else {"name": k} for k, v in ents.items()]
    jev = next((e for e in ents if isinstance(e, dict) and str(e.get("name", "")).lower() == "jev"), None)
    if jev is None:
        return {}
    out = {"index": jev.get("scores", {}), "benchmarks": {}}
    for cid, b in (jev.get("benchmarks") or {}).items():
        out["benchmarks"][str(cid)] = dict(b.get("board") or {})
    return out


def unsupported_message(detail: Any) -> str:
    return json.dumps(detail, ensure_ascii=False) if not isinstance(detail, str) else detail


DI_SERVER_CONFIG = {"precision": "exact", "overflow": "refuse", "refusal_style": "jev"}


def config_problems(config: dict[str, Any] | None) -> list[str]:
    """Why a server's `/healthz` benchmax config would break the kit: 2-dp rounding fails |Σp − 1| ≤ 0.01 on
    wide questions, truncation is forbidden, and refusals without the capacity markers are recorded as errors."""
    if not isinstance(config, dict):
        return ["no benchmax config in /healthz (old server?)"]
    return [f"{k}={config.get(k)!r}, need {v!r}" for k, v in DI_SERVER_CONFIG.items() if config.get(k) != v]


def server_config_problems(base_url: str) -> list[str]:
    import httpx

    try:
        r = httpx.get(base_url.rstrip("/") + "/healthz", timeout=10.0)
        cfg = (r.json().get("benchmax") or {}).get("config") if r.status_code == 200 else None
    except Exception as e:  # unreachable server: the kit will report that itself
        return [f"/healthz unreachable: {type(e).__name__}"]
    return config_problems(cfg)


class DecisionIndexAdapter(Adapter):
    default_overflow = "refuse"
    allow_truncate = False
    default_split = "eval"

    @classmethod
    def add_arguments(cls, p: Any) -> None:
        p.add_argument("--kit-dir", help=f"pinned decision-index checkout (default {DEFAULT_WORK / 'decision-index'})")
        p.add_argument("--kit-python", help="interpreter with the kit's deps (default $BENCHMAX_WORK/venv-di/bin/python, else this one)")
        p.add_argument("--suite-dir", help="staged suite (default <kit>/suite-0.2)")
        p.add_argument("--work", help=f"rebuild work dir (default {DEFAULT_WORK / 'di-work'})")
        p.add_argument("--di-step", choices=["verify", "rebuild", "import", "pipeline", "run", "score"], default="pipeline",
                       help="run = the kit's `run` over --rows without scoring (engine smoke test; needs no staged suite)")
        p.add_argument("--di-engine", choices=["http", "inproc"], default="http", help="kit http engine against --base-url, or our in-process Engine subclass (--ckpt)")
        p.add_argument("--rows", help="run this rows file instead of the suite (smoke tests; never real test rows)")
        p.add_argument("--limit", type=int)
        p.add_argument("--compact", action="store_true")
        p.add_argument("--fresh", action="store_true")
        p.add_argument("--allow-unverified", action="store_true", help="run even if the suite hashes do not match (recorded)")

    # ------------------------------------------------------------------ paths

    @property
    def kit_dir(self) -> Path:
        return Path(self.args.kit_dir or DEFAULT_WORK / "decision-index").expanduser()

    @property
    def suite_dir(self) -> Path:
        return Path(self.args.suite_dir or self.kit_dir / "suite-0.2").expanduser()

    @property
    def work_dir(self) -> Path:
        return Path(self.args.work or DEFAULT_WORK / "di-work").expanduser()

    def _python(self) -> str:
        if self.args.kit_python:
            return self.args.kit_python
        if getattr(self.args, "di_engine", "http") == "inproc":
            return sys.executable  # needs torch: our venv, with the kit on PYTHONPATH
        venv = DEFAULT_WORK / "venv-di" / "bin" / "python"
        return str(venv) if venv.exists() else sys.executable

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["HF_HUB_DISABLE_XET"] = "1"
        parts = [str(self.kit_dir), str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        env["PYTHONPATH"] = os.pathsep.join(parts)
        return env

    def _kit(self, *argv: str, capture: bool = True) -> subprocess.CompletedProcess:
        cmd = [self._python(), "-m", "decision_index", *argv]
        return subprocess.run(cmd, cwd=self.kit_dir, env=self._env(), capture_output=capture, text=True)

    # ------------------------------------------------------------------ verify

    def verify(self, ctx: RunContext) -> dict[str, Any]:
        rep: dict[str, Any] = {"repo": REPO, "licence": LICENCE, "edition": EDITION, "commit": check_commit(self.kit_dir, PINNED_COMMIT),
                               "pinned": {"rows_sha256": ROWS_SHA256, "added_sha256": ADDED_SHA256, "exclusions_sha256": EXCLUSIONS_SHA256}}
        rebuilt = self.work_dir / REBUILT_REL
        rep["rebuilt"] = {}
        for name, pinned in (("selected-rows.jsonl.gz", ROWS_SHA256), ("added-rows.jsonl.gz", ADDED_SHA256)):
            p = rebuilt / name
            if p.exists():
                got = sha256_gunzipped(p)
                rep["rebuilt"][name] = {"uncompressed_sha256": got, "match": got == pinned, "bytes": p.stat().st_size}
            else:
                rep["rebuilt"][name] = {"present": False}
        ex = self.kit_dir / "hub" / "excluded-questions.json"
        if ex.exists():
            got = sha256_file(ex)
            rep["exclusions"] = {"sha256": got, "match": got == EXCLUSIONS_SHA256}
        if self.suite_dir.exists() and self.kit_dir.exists():
            v = self._kit("suite", "verify", "--dir", str(self.suite_dir), "--edition", EDITION)
            try:
                rep["suite_verify"] = json.loads(v.stdout)
            except ValueError:
                rep["suite_verify"] = {"error": (v.stderr or v.stdout)[-1500:]}
        else:
            rep["suite_verify"] = {"present": False, "note": "suite not staged: run --di-step rebuild, then import"}
        rep["jev_reference"] = jev_fixture_values(self.kit_dir)
        rep["jev_index"] = {"fixture": JEV_INDEX_FIXTURE, "space": JEV_INDEX_SPACE, "bar_uses": "stricter value per row (PLAN §9)"}
        sv = rep["suite_verify"]
        rep["ok"] = bool(rep["commit"]["match"]) and bool(isinstance(sv, dict) and sv.get("match"))
        return rep

    # ------------------------------------------------------------------ run

    def run(self, session: Session, ctx: RunContext) -> dict[str, Any]:
        if ctx.overflow.get("mode") != "refuse":
            raise SystemExit("Decision Index forbids truncation: --overflow refuse is mandatory")
        step = self.args.di_step
        if step == "verify":
            return {"verify": self.verify(ctx)}
        if step == "rebuild":
            ctx.note("suite rebuild: ~7 GB of downloads; HLE is gated (user's HF login). Prefer running it detached (see report).")
            r = self._kit("suite", "rebuild", "--work", str(self.work_dir), capture=False)
            return {"rebuild_returncode": r.returncode, "verify": self.verify(ctx)}
        if step == "import":
            rebuilt = self.work_dir / REBUILT_REL
            r = self._kit("suite", "import", "--dir", str(self.suite_dir), "--edition", EDITION, "--rows", str(rebuilt / "selected-rows.jsonl.gz"),
                          "--added-rows", str(rebuilt / "added-rows.jsonl.gz"))
            return {"import_stdout": r.stdout[-3000:], "import_stderr": r.stderr[-3000:], "returncode": r.returncode, "verify": self.verify(ctx)}
        out = ctx.out / "di"
        if step == "run":
            if not self.args.rows:
                raise SystemExit("--di-step run needs --rows (a smoke rows file; never real suite rows outside the pipeline)")
            argv = ["run", "--edition", EDITION, "--rows", self.args.rows, "--out", str(out), *self._engine_argv(ctx)]
            if self.args.limit:
                argv += ["--limit", str(self.args.limit)]
            if self.args.fresh:
                argv.append("--fresh")
            r = self._kit(*argv)
            (ctx.out / "di-run.log").write_text(" ".join(argv) + "\n\n" + r.stdout + r.stderr)
            res = self._collect(ctx, out, returncode=r.returncode, cmd=argv)
            res["smoke"] = self._summarize_results(out / "results.jsonl")
            return res
        if step == "score":
            r = self._kit("score", "--results", str(out / "results.jsonl"), "--suite-dir", str(self.suite_dir), "--edition", EDITION, "--engine", ctx.model_id, "--out", str(out))
            (ctx.out / "di-score.log").write_text(r.stdout + r.stderr)
            return self._collect(ctx, out, returncode=r.returncode)
        # pipeline
        ver = self.verify(ctx)
        ctx.harness = ver
        if not self.args.rows and not ver.get("ok"):
            if not self.args.allow_unverified:
                raise SystemExit("suite hash check failed or suite not staged; refusing to run (pass --allow-unverified to override, recorded)")
            ctx.note("RUNNING ON AN UNVERIFIED SUITE (--allow-unverified)")
        argv = ["pipeline", "--edition", EDITION, "--suite-dir", str(self.suite_dir), "--out", str(out), *self._engine_argv(ctx)]
        if self.args.rows:
            argv += ["--rows", self.args.rows]
        if self.args.limit:
            argv += ["--limit", str(self.args.limit)]
        if self.args.compact:
            argv.append("--compact")
        if self.args.fresh:
            argv.append("--fresh")
        ctx.results = {"pipeline_cmd": argv, "status": "running"}
        ctx.write()
        r = self._kit(*argv)
        (ctx.out / "di-pipeline.log").write_text(" ".join(argv) + "\n\n" + r.stdout + r.stderr)
        return self._collect(ctx, out, returncode=r.returncode, cmd=argv)

    def _engine_argv(self, ctx: RunContext) -> list[str]:
        if self.args.di_engine == "http":
            if not getattr(self.args, "base_url", None):
                raise SystemExit("--di-engine http needs --base-url")
            problems = server_config_problems(self.args.base_url)
            if problems and not self.args.allow_unverified:
                raise SystemExit("the server is not in Decision Index mode (start it with `python -m jev_local.server.app --preset di ...`): " + "; ".join(problems))
            for p in problems:
                ctx.note(f"server config: {p} (--allow-unverified)")
            wire = getattr(self.args, "wire_model", None) or ctx.model_id
            return ["--engine", "http", "--option", f"base_url={self.args.base_url}", "--option", f"model={wire}"]
        if not getattr(self.args, "ckpt", None):
            raise SystemExit("--di-engine inproc needs --ckpt")
        argv = ["--engine", "jev_local.bench.benchmax.adapters_a.decision_index:MeharsjevEngine",
                "--option", f"ckpt={self.args.ckpt}", "--option", f"model={ctx.model_id}", "--option", f"threads={self.args.threads}"]
        if getattr(self.args, "calib", None):
            argv += ["--option", f"calib={self.args.calib}"]
        if getattr(self.args, "max_tokens", None):
            argv += ["--option", f"max_tokens={self.args.max_tokens}"]
        return argv

    @staticmethod
    def _summarize_results(path: Path) -> dict[str, Any]:
        """Status counts and the engine's refusal texts from a kit results.jsonl (smoke runs)."""
        if not path.exists():
            return {"present": False}
        counts: dict[str, int] = {}
        errors: list[str] = []
        n = 0
        for line in path.read_text(encoding="utf-8").split("\n"):  # not splitlines(): U+2028 etc. inside texts
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            counts[r.get("status", "?")] = counts.get(r.get("status", "?"), 0) + 1
            if r.get("status") != "ok" and len(errors) < 5:
                errors.append(f"{r.get('run_id')}: {str(r.get('error'))[:300]}")
        return {"present": True, "rows": n, "counts": counts, "samples": errors}

    def _collect(self, ctx: RunContext, out: Path, *, returncode: int, cmd: list[str] | None = None) -> dict[str, Any]:
        res: dict[str, Any] = {"returncode": returncode, "pipeline_cmd": cmd, "run_dir": str(out), "edition": EDITION}
        scores_p = out / "scores.json"
        if not scores_p.exists():
            res["note"] = "no scores.json (pipeline failed or incomplete); see di-pipeline.log"
            return res
        s = json.loads(scores_p.read_text())
        jev = jev_fixture_values(self.kit_dir)
        res.update({
            "decision_index": s.get("decision_index"), "raw_index": s.get("raw_index"), "scores": s.get("scores"),
            "complete": s.get("complete"), "completed": s.get("completed"), "counts": s.get("counts"), "latency_ms": s.get("latency_ms"),
            "areas": s.get("areas"), "suite": s.get("suite"),
            "jev_index": {"fixture": JEV_INDEX_FIXTURE, "space": JEV_INDEX_SPACE},
        })
        counts = s.get("counts") or {}
        res["refused"] = counts.get("unsupported", 0)
        ctx.overflow["refused"] = res["refused"]
        bench: dict[str, Any] = {}
        for cid, b in (s.get("benchmarks") or {}).items():
            e = {k: b.get(k) for k in ("dataset", "requests", "answered", "unsupported", "errors", "metric", "score", "index_raw", "index_skill", "coverage", "chance", "in_index")}
            ref = (jev.get("benchmarks") or {}).get(str(cid))
            if ref:
                e["jev_raw"] = ref.get("raw")
                if isinstance(e.get("index_raw"), (int, float)) and isinstance(ref.get("raw"), (int, float)):
                    e["delta_vs_jev"] = e["index_raw"] - ref["raw"]
            bench[str(cid)] = e
        res["benchmarks"] = bench
        for name in ("environment.json", "status.json"):
            p = out / name
            if p.exists():
                try:
                    res[name.split(".")[0]] = json.loads(p.read_text())
                except ValueError:
                    pass
        return res


# ---------------------------------------------------------------------- kit Engine subclass (lazy)


def make_engine_class():
    """Built on first use so this module imports without the kit (Mac tests, registry listing)."""
    from decision_index.engines.base import Engine, Unsupported  # kit on PYTHONPATH

    class MeharsjevEngine(Engine):
        name = "meharsjev"
        latency = "In-process wall time of one /v1/systemone-equivalent evaluation (serialize + pack + forward + answers); excludes checkpoint loading."

        def __init__(self, ckpt=None, model="meharsjev-68m", threads=8, calib=None, max_tokens=None, base_url=None, precision="unrounded", **options):  # type: ignore[no-untyped-def]
            super().__init__(**options)
            from jev_local.bench.benchmax.runner import HttpClient, LocalClient

            if base_url:
                self.client = HttpClient(str(base_url), str(model))
            else:
                if not ckpt:
                    raise ValueError("MeharsjevEngine needs ckpt=<dir> or base_url=<url>")
                self.client = LocalClient(ckpt, str(model), threads=int(threads), calib_path=calib, max_tokens=int(max_tokens) if max_tokens else None,
                                          overflow="refuse", precision=precision)
            d = self.client.describe()
            self.provenance = {"kind": "meharsjev", "model": model, "ckpt": str(ckpt) if ckpt else None, "base_url": base_url,
                               "policy": "Unmodified state and questions; over-long or over-wide requests are refused (Unsupported), never truncated; one global calibration file; unrounded probabilities.",
                               **{k: v for k, v in d.items() if k in ("engine", "max_tokens", "calibration_sha256", "threads")}}

        def __call__(self, state, questions):  # type: ignore[no-untyped-def]
            o = self.client.system_one(state, questions)
            if o.status == "refused":
                raise Unsupported(unsupported_message(o.detail))
            if o.status != "ok":
                raise RuntimeError(unsupported_message(o.detail))
            return o.response, {"latency_s": o.latency_s, "engine": o.engine_name}

        def runtime(self):  # type: ignore[no-untyped-def]
            try:
                import torch

                return {"torch": torch.__version__, "threads": torch.get_num_threads(), "device": "cpu"}
            except ImportError:
                return {}

    return MeharsjevEngine


def __getattr__(name: str):
    if name == "MeharsjevEngine":
        return make_engine_class()
    raise AttributeError(name)
