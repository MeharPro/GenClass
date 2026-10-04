"""`deusser_exact@6bbdeb33`: Deußer, Sparrenberg & Sifa 2026 through their own code (specs §2.4; PLAN §6).

Route (byte-exact requests because the task code is theirs):
1. `jev_benchmarking` is imported from the pinned checkout (`--harness-dir`, default `$BENCHMAX_WORK/jev-benchmarking`,
   MIT). The commit must be 6bbdeb33… (`verify`).
2. For every `task in TASKS` and `e in task.examples(split, limit)`: `key = cache.request_key(model_id, e.state,
   e.questions)`, the request goes to our engine unchanged (W1–W3, W6, W8), and
   `{"key", "task", "response": {"model": model_id, "answers", "usage"}, "latency_s"}` is appended to
   `<out>/responses/shard-<i>.jsonl` (resumable; shardable by `int(key, 16) % num_shards`).
3. `scripts/import_responses.py` writes `cache/responses.<model_id>.db` (it refuses `jev-*` models: W10) and
   `scripts/evaluate.py <tasks> --split <split> --model <model_id>` scores with their metrics and CIs (bootstrap
   500, seed 0). The result files are copied under `<out>/results/`.

Overflow: Deußer scores answered rows only, so the default is `truncate` (100 % coverage, counted and disclosed,
PLAN §2.2.5); with `refuse` the run also reports a "missing = wrong" variant for accuracy-type primaries.
HF revisions: the harness does not pin them. `--revisions FILE` ({hf_id: sha}) pins `Task.load_split`; otherwise
the Hub sha at run time is recorded. `n` is asserted against targets.json for the eval split.

Debugging is done on `--split dev` only (train/validation items). Runs on VM train.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from jev_local.bench.benchmax.runner import (
    DEFAULT_WORK, Adapter, RunContext, Session, append_jsonl, check_commit, load_targets, read_jsonl, sha256_file,
)

REPO = "https://github.com/AppliedMachineLearning-Lab/jev-benchmarking"
PINNED_COMMIT = "6bbdeb33474849b6de2f0cccc9f5e19756abd67e"
LICENCE = "MIT"
N_BOOTSTRAP, BOOTSTRAP_SEED = 500, 0
# Ungated tasks with a dev split that exercise every head kind (choice, binary, score, multilabel): the self-check set.
SELF_CHECK_TASKS = ("sst2", "emotion", "paws", "stsb", "go_emotions")
ACCURACY_LIKE = {"accuracy", "argmax_accuracy"}


def task_of_source_url(url: str | None) -> str | None:
    """targets.json rows point at `results/eval/<task>.json`; that file stem is the harness task name."""
    if not url or "results/eval/" not in url:
        return None
    return url.rsplit("results/eval/", 1)[1].split(".json", 1)[0] or None


def expected_n_by_task(spec_id: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in load_targets(spec_id):
        t = task_of_source_url(r.get("source_url"))
        if t and r.get("role") == "headline" and isinstance(r.get("n"), int):
            out[t] = r["n"]
    return out


def shard_of(key: str, num_shards: int) -> int:
    return int(key[:16], 16) % max(1, num_shards)


def response_record(key: str, task: str, uid: str, model_id: str, response: dict[str, Any], latency_s: float) -> dict[str, Any]:
    """Exactly what `scripts/import_responses.py` reads (plus `uid` for us)."""
    if model_id.startswith("jev-"):
        raise ValueError("W10: a Jev model id may never be imported as ours")
    return {
        "key": key,
        "task": task,
        "uid": uid,
        "response": {"model": model_id, "answers": response["answers"], "usage": response.get("usage", {})},
        "latency_s": round(float(latency_s), 6),
    }


def missing_wrong(metric: str, value: float | None, n_answered: int, n_examples: int) -> float | None:
    """The "missing = wrong" variant of PLAN §2.2.5: exact for accuracy-type primaries, else not defined."""
    if value is None or metric not in ACCURACY_LIKE or n_examples <= 0:
        return None
    return value * n_answered / n_examples


class DeusserAdapter(Adapter):
    default_overflow = "truncate"
    allow_truncate = True
    default_split = "eval"

    @classmethod
    def add_arguments(cls, p: Any) -> None:
        p.add_argument("--harness-dir", help=f"pinned jev-benchmarking checkout (default $BENCHMAX_WORK/jev-benchmarking = {DEFAULT_WORK / 'jev-benchmarking'})")
        p.add_argument("--tasks", nargs="*", default=["all"], help="harness task names, 'all', or 'self-check'")
        p.add_argument("--split", choices=["eval", "dev"], default="eval", help="dev = train/validation items (debugging only)")
        p.add_argument("--limit", type=int, help="first N examples per task in the harness's hash order (same semantics as their --limit)")
        p.add_argument("--subsets", nargs="*", help="restrict to these configs (languages / subjects)")
        p.add_argument("--shard", type=int, default=0)
        p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--revisions", help="JSON {hf_id: revision} to pin Task.load_split")
        p.add_argument("--no-evaluate", action="store_true", help="only produce the response JSONL (another node imports/evaluates)")
        p.add_argument("--no-ci", action="store_true", help="skip their bootstrap CIs")
        p.add_argument("--harness-python", help="interpreter for import_responses.py / evaluate.py (default: this one)")

    # ------------------------------------------------------------------ paths

    @property
    def harness_dir(self) -> Path:
        return Path(self.args.harness_dir or DEFAULT_WORK / "jev-benchmarking").expanduser()

    def _python(self) -> str:
        return self.args.harness_python or sys.executable

    def _env(self, out: Path) -> dict[str, str]:
        env = dict(os.environ)
        env["JEV_CACHE_DB"] = str(out / "cache" / "responses.db")  # -> cache/responses.<model_id>.db next to it
        env["PYTHONPATH"] = str(self.harness_dir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        return env

    # ------------------------------------------------------------------ verify

    def verify(self, ctx: RunContext) -> dict[str, Any]:
        d = self.harness_dir
        rep: dict[str, Any] = {"repo": REPO, "licence": LICENCE, "commit": check_commit(d, PINNED_COMMIT)}
        for rel in ("scripts/import_responses.py", "scripts/evaluate.py", "jev_benchmarking/evaluate.py", "jev_benchmarking/cache.py", "LICENSE"):
            p = d / rel
            rep[rel] = sha256_file(p) if p.exists() else None
        rep["expected_n"] = expected_n_by_task(self.spec.id)
        rep["jev_reference"] = self._jev_reference()
        rep["ok"] = bool(rep["commit"]["match"]) and all(rep[k] for k in ("scripts/import_responses.py", "scripts/evaluate.py"))
        return rep

    def _jev_reference(self) -> dict[str, Any]:
        """Jev's published aggregates from the MIT repo (`results/eval/<task>.json`), never its raw outputs."""
        out: dict[str, Any] = {}
        for p in sorted((self.harness_dir / "results" / "eval").glob("*.json")):
            if p.name in ("thresholds.json", "probes.json"):
                continue
            try:
                r = json.loads(p.read_text())
                out[p.stem] = {"n": r.get("n_examples"), "primary": r.get("primary")}
            except (OSError, ValueError):
                continue
        return out

    # ------------------------------------------------------------------ run

    def _import_harness(self, out: Path):
        os.environ["JEV_CACHE_DB"] = str(out / "cache" / "responses.db")  # config.py reads it at import
        if str(self.harness_dir) not in sys.path:
            sys.path.insert(0, str(self.harness_dir))
        import jev_benchmarking.tasks as tasks_mod  # noqa: WPS433 (datasets import: VM only)
        from jev_benchmarking.cache import request_key
        from jev_benchmarking.tasks.base import Task

        return tasks_mod, request_key, Task

    def _pin_revisions(self, Task: Any, ctx: RunContext) -> dict[str, Any]:
        revisions: dict[str, str] = {}
        if self.args.revisions:
            revisions = json.loads(Path(self.args.revisions).read_text())
        seen: dict[str, Any] = ctx.dataset_revisions

        def own_loader(task_self):  # type: ignore[no-untyped-def]
            """The subclass's own `load_split`, when it defines one (ANLI does); else None."""
            for klass in type(task_self).__mro__:
                if klass is Task or klass is object:
                    continue
                if "load_split" in klass.__dict__:
                    return klass.__dict__["load_split"]
            return None

        def load_split(task_self, config, split):  # type: ignore[no-untyped-def]
            from datasets import load_dataset

            own = own_loader(task_self)
            if own is not None:  # the task defines its own loader: cannot pin here
                seen.setdefault(task_self.hf_path, {"pinned": None, "note": "task-defined loader"})
                return own(task_self, config, split)
            rev = revisions.get(task_self.hf_path)
            seen.setdefault(task_self.hf_path, {"pinned": rev})
            return load_dataset(task_self.hf_path, config, split=split, revision=rev)

        Task.load_split = load_split
        return revisions

    def _resolve_hub_shas(self, ctx: RunContext) -> None:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            for hf_id, rec in ctx.dataset_revisions.items():
                if isinstance(rec, dict) and rec.get("pinned") is None and "hub_sha_at_run" not in rec:
                    try:
                        rec["hub_sha_at_run"] = api.dataset_info(hf_id).sha
                    except Exception as e:  # network / gated: recorded, not fatal
                        rec["hub_sha_at_run"] = f"error: {type(e).__name__}"
        except Exception as e:
            ctx.note(f"could not resolve Hub shas: {type(e).__name__}: {e}")

    def _task_names(self, tasks_mod: Any) -> list[str]:
        names = list(self.args.tasks or ["all"])
        if names == ["self-check"]:
            names = list(SELF_CHECK_TASKS)
        return [t.name for t in tasks_mod.get_tasks(names)]

    def run(self, session: Session, ctx: RunContext) -> dict[str, Any]:
        out = ctx.out
        tasks_mod, request_key, Task = self._import_harness(out)
        self._pin_revisions(Task, ctx)
        names = self._task_names(tasks_mod)
        split, limit = self.args.split, self.args.limit
        expected = expected_n_by_task(self.spec.id) if split == "eval" else {}
        resp_dir = out / "responses"
        resp_dir.mkdir(parents=True, exist_ok=True)
        shard_path = resp_dir / f"shard-{self.args.shard}.jsonl"
        refused_path = out / "refused.jsonl"
        done: set[str] = set()
        for p in resp_dir.glob("*.jsonl"):
            done.update(r["key"] for r in read_jsonl(p))
        if done:
            ctx.note(f"resuming: {len(done)} responses already on disk")
        per_task: dict[str, Any] = {}
        for name in names:
            task = tasks_mod.ALL_TASKS[name]
            if task.splits.get(split) is None:
                ctx.note(f"{name}: no {split!r} split; skipped")
                continue
            t0 = time.time()
            examples = task.examples(split, limit=limit, subsets=self.args.subsets)
            n_exp = expected.get(name)
            stat = {"n_examples": len(examples), "expected_n": n_exp, "n_match": (n_exp == len(examples)) if (n_exp and limit is None) else None,
                    "sent": 0, "ok": 0, "refused": 0, "errors": 0, "truncated": 0, "skipped_done": 0, "skipped_shard": 0}
            if task.skipped_configs:
                stat["skipped_configs"] = dict(task.skipped_configs)
            if stat["n_match"] is False:
                ctx.note(f"{name}: n={len(examples)} differs from targets.json n={n_exp}: investigate before any claim (PLAN §3.1.7)")
            for e in examples:
                key = request_key(session.model_id, e.state, e.questions)
                if shard_of(key, self.args.num_shards) != self.args.shard:
                    stat["skipped_shard"] += 1
                    continue
                if key in done:
                    stat["skipped_done"] += 1
                    continue
                o = session.ask(e.state, e.questions, {"task": name, "uid": e.uid})
                stat["sent"] += 1
                if o.ok:
                    stat["ok"] += 1
                    stat["truncated"] += int(o.truncated)
                    append_jsonl(shard_path, response_record(key, name, e.uid, session.model_id, o.response, o.latency_s))
                    done.add(key)
                else:
                    stat["refused" if o.status == "refused" else "errors"] += 1
                    append_jsonl(refused_path, {"key": key, "task": name, "uid": e.uid, "status": o.status, "http_status": o.http_status, "detail": o.detail})
            stat["seconds"] = round(time.time() - t0, 1)
            per_task[name] = stat
            ctx.results = {"tasks": per_task}
            ctx.write()
        self._resolve_hub_shas(ctx)
        results: dict[str, Any] = {"harness_commit": PINNED_COMMIT, "split": split, "limit": limit, "tasks": per_task,
                                   "responses_dir": str(resp_dir), "shard": [self.args.shard, self.args.num_shards]}
        if self.args.no_evaluate:
            results["evaluated"] = False
            return results
        results.update(self.import_and_evaluate(ctx, names))
        return results

    # ------------------------------------------------------------------ their scorer

    def import_and_evaluate(self, ctx: RunContext, names: list[str]) -> dict[str, Any]:
        out = ctx.out
        env = self._env(out)
        files = sorted(str(p) for p in (out / "responses").glob("*.jsonl"))
        if not files:
            return {"evaluated": False, "note": "no responses to import"}
        imp = subprocess.run([self._python(), "scripts/import_responses.py", *files], cwd=self.harness_dir, env=env,
                             capture_output=True, text=True)
        (out / "import_responses.log").write_text(imp.stdout + imp.stderr)
        if imp.returncode != 0:
            ctx.note(f"import_responses.py failed: {imp.stderr[-800:]}")
            return {"evaluated": False, "import_error": imp.stderr[-2000:]}
        cmd = [self._python(), "scripts/evaluate.py", *names, "--split", self.args.split, "--model", ctx.model_id]
        if self.args.limit:
            cmd += ["--limit", str(self.args.limit)]
        if self.args.subsets:
            cmd += ["--subsets", *self.args.subsets]
        if self.args.no_ci:
            cmd.append("--no-ci")
        ev = subprocess.run(cmd, cwd=self.harness_dir, env=env, capture_output=True, text=True)
        (out / "evaluate.log").write_text(" ".join(cmd) + "\n\n" + ev.stdout + ev.stderr)
        if ev.returncode != 0:
            ctx.note(f"evaluate.py failed: {ev.stderr[-800:]}")
            return {"evaluated": False, "evaluate_error": ev.stderr[-2000:]}
        slug = ctx.model_id.replace("hf:", "").replace("/", "__")
        src = self.harness_dir / "results" / self.args.split / "open_models" / slug
        dst = out / "results"
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.glob("*"):
            if p.is_file():
                shutil.copyfile(p, dst / p.name)
        return {"evaluated": True, "results_dir": str(dst), "evaluate_cmd": cmd, "scores": self._collect(dst, names)}

    def _collect(self, dst: Path, names: list[str]) -> dict[str, Any]:
        jev = self._jev_reference()
        scores: dict[str, Any] = {}
        for name in names:
            p = dst / f"{name}.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            prim = r.get("primary", {})
            n_ex, n_ans = r.get("n_examples", 0), r.get("n_answered", 0)
            entry = {
                "n_examples": n_ex, "n_answered": n_ans, "coverage": (n_ans / n_ex) if n_ex else None,
                "metric": prim.get("metric"), "head": prim.get("head"), "value": prim.get("value"), "ci95": prim.get("ci95"),
                "ci_method": f"bootstrap {N_BOOTSTRAP}, seed {BOOTSTRAP_SEED} (theirs)",
                "missing_wrong_value": missing_wrong(prim.get("metric", ""), prim.get("value"), n_ans, n_ex),
            }
            ref = jev.get(name, {}).get("primary") if self.args.split == "eval" else None
            if ref:
                entry["jev_published"] = {"value": ref.get("value"), "ci95": ref.get("ci95")}
                if isinstance(entry["value"], (int, float)) and isinstance(ref.get("value"), (int, float)):
                    entry["delta_vs_jev"] = entry["value"] - ref["value"]
            scores[name] = entry
        return scores
