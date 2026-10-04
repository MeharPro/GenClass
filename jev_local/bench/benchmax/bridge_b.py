"""Bridge: drive a group-B adapter (`adapters_b.common.AdapterB`: prepare / items / score / fit) through the shared
benchmax runner (`runner.Adapter`: verify / run; run.json schema `benchmax-run/1`).

Why: `scripts/benchmax.py` is the one entry point PLAN §6 names ("every harness run logs ... into
`runs/benchmax/<ckpt>/<spec>/run.json`"), but the fifteen group-B specs were written against their own
`adapters_b.driver` CLI, whose run.json is a different, lighter record. Wrapping them here gives every one of
the 19 specs the same engine clients (W1–W5, W9, W10), the same refusal bookkeeping, the same determinism
replay (W11) and the same run.json, without touching the adapters themselves.

Mapping
- `verify(ctx)`  -> `inner.prepare(split)` (downloads / verifies the pinned sources; its manifest is recorded
  under `harness.prepare` and any revision-like leaves go to `dataset_revisions`) + `inner.expected_counts(split)`.
- `run(session, ctx)` -> `inner.items(split, limit, tasks)`; one `session.ask` per item; multi-stage suites
  (`STAGES > 1`, zhuyansen BANKING77) get their later stages through `inner.stage_items`; `inner.score(...)`
  (and `inner.fit(...)` with `--fit`, validation only). `items.jsonl`, `answers.jsonl` (the driver's record
  shape, so `adapters_b.driver score|fit` can re-read them) and `scores.json` land next to run.json.
- Test hygiene (PLAN §2.2.2): the evaluated split (`inner.EVAL_SPLIT`) is refused unless `--allow-test`; the
  default split is the first non-evaluated one. Thresholds are never fitted on the evaluated split.
- Overflow: group-B suites allow truncation (PLAN §4.2 item 5), counted and disclosed -> default `truncate`.

Pure Python at import time (the Mac may build the CLI and run the tests); torch loads only inside the runner's
`LocalClient` on the VM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from jev_local.bench.benchmax.runner import Adapter, Outcome, RunContext, Session, append_jsonl, load_targets, write_json

REVISION_KEYS = ("revision", "revisions", "commit", "commits", "sha", "resolved", "resolved_revision", "hub_sha", "pinned")
_HEX = set("0123456789abcdef")


def is_group_b_class(cls: type) -> bool:
    """A class speaking the group-B contract (items/score) but not the runner's (verify/run/add_arguments)."""
    has_runner = all(callable(getattr(cls, m, None)) for m in ("verify", "run", "add_arguments"))
    has_b = all(callable(getattr(cls, m, None)) for m in ("items", "score", "prepare"))
    return has_b and not has_runner


def collect_revisions(manifest: Any, prefix: str = "", depth: int = 0, out: dict[str, Any] | None = None) -> dict[str, Any]:
    """Revision-like leaves of a prepare() manifest ({path: value}); hex strings of 7..64 chars under keys that
    look like revisions / commits / shas. Best effort: adapters pin in slightly different shapes."""
    if out is None:
        out = {}
    if depth > 6:
        return out
    if isinstance(manifest, Mapping):
        for k, v in manifest.items():
            key = str(k)
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(v, str) and _looks_like_revision(key, v):
                out[path] = v
            elif isinstance(v, (Mapping, list, tuple)):
                collect_revisions(v, path, depth + 1, out)
    elif isinstance(manifest, (list, tuple)):
        for i, v in enumerate(manifest):
            if isinstance(v, (Mapping, list, tuple)):
                collect_revisions(v, f"{prefix}[{i}]", depth + 1, out)
    return out


def _looks_like_revision(key: str, value: str) -> bool:
    k = key.lower()
    named = any(x in k for x in REVISION_KEYS)
    hexish = 7 <= len(value) <= 64 and all(c in _HEX for c in value.lower())
    return (named and (hexish or len(value) <= 80)) or (k.endswith("sha256") and hexish)


def answer_record(item_id: str, o: Outcome) -> dict[str, Any]:
    """The driver's answers.jsonl record shape (`adapters_b.engine_client.run_items`), from a runner Outcome."""
    ms = round(o.latency_s * 1000.0, 3)
    if o.ok and o.response is not None:
        return {"id": item_id, "ok": True, "answers": o.response.get("answers", {}), "usage": o.response.get("usage", {}),
                "model": o.response.get("model"), "truncated": bool(o.truncated), "tokens_cut": int(o.tokens_cut), "ms": ms}
    return {"id": item_id, "ok": False, "permanent": True, "status": o.http_status, "refused": o.status == "refused",
            "error": json.dumps(o.detail, ensure_ascii=False, default=str)[:500], "ms": ms}


def answer_counts(records: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    recs = list(records.values())
    return {"requests": len(recs), "ok": sum(1 for r in recs if r.get("ok")),
            "truncated": sum(1 for r in recs if r.get("ok") and r.get("truncated")),
            "failed": sum(1 for r in recs if not r.get("ok")), "refused": sum(1 for r in recs if r.get("refused"))}


def _write_items(path: Path, items: Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.to_json(), ensure_ascii=False) + "\n")


def _load_answers(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("ok"):
                out[rec["id"]] = rec
    return out


class BridgedAdapter(Adapter):
    """Runner-facing view of one group-B adapter class (`inner_cls`); built per class by `bridge_class`."""

    inner_cls: type | None = None
    default_overflow = "truncate"  # group-B protocols allow (counted, disclosed) truncation: PLAN §4.2 item 5
    allow_truncate = True
    default_split = "validation"

    def __init__(self, spec: Any, args: Any):
        super().__init__(spec, args)
        if self.inner_cls is None:
            raise TypeError("BridgedAdapter needs an inner_cls (use bridge_class)")
        inner_args = argparse.Namespace(work=getattr(args, "work", None), limit=getattr(args, "limit", None),
                                        tasks=self._tasks(getattr(args, "tasks", None)))
        self.inner = self.inner_cls(spec, inner_args)

    # ------------------------------------------------------------------ CLI

    @classmethod
    def add_arguments(cls, p: Any) -> None:
        inner = cls.inner_cls
        splits = list(getattr(inner, "SPLITS", ("validation", "test")))
        eval_split = getattr(inner, "EVAL_SPLIT", "test")
        p.add_argument("--split", choices=splits, default=None,
                       help=f"default: the first non-evaluated split of {splits}; the evaluated split ({eval_split!r}) needs --allow-test")
        p.add_argument("--limit", type=int, help="first N items per task (self-check only)")
        p.add_argument("--tasks", help=f"comma-separated subset of {list(getattr(inner, 'TASKS', ()))}")
        p.add_argument("--work", help="group-B work dir (default $BENCHMAX_WORK or ~/bench_work_b)")
        p.add_argument("--thresholds", help="thresholds.json fitted earlier on validation (S variants)")
        p.add_argument("--fit", action="store_true", help="also fit thresholds on this run's answers (non-evaluated split only) -> <out>/thresholds.json")
        p.add_argument("--allow-test", action="store_true", help=f"answer the evaluated split ({eval_split!r}); PLAN §2.2.2: once per release")
        p.add_argument("--skip-prepare", action="store_true", help="do not re-run prepare() (sources already verified in this work dir)")
        p.add_argument("--resume", action="store_true", help="reuse ok answers already in <out>/answers.jsonl (crash recovery)")

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _tasks(raw: Any) -> list[str] | None:
        if not raw:
            return None
        if isinstance(raw, str):
            return [t for t in raw.split(",") if t]
        return list(raw)

    def split(self) -> str:
        inner = self.inner
        s = getattr(self.args, "split", None)
        if s is None:
            non_eval = [x for x in inner.SPLITS if x != inner.EVAL_SPLIT]
            s = non_eval[0] if non_eval else inner.EVAL_SPLIT
        if s not in inner.SPLITS:
            raise SystemExit(f"{self.spec.id}: unknown split {s!r}; known {list(inner.SPLITS)}")
        return s

    def guard_eval_split(self, split: str) -> None:
        if split == self.inner.EVAL_SPLIT and not getattr(self.args, "allow_test", False):
            raise SystemExit(f"{self.spec.id}: {split!r} is the evaluated split (the published Jev number); pass --allow-test for a "
                             "release run (PLAN §2.2.2: test is read once per release; adapters are debugged on validation)")

    def _target_rows(self) -> dict[str, dict[str, Any]]:
        return {r["id"]: r for r in load_targets()}

    def target_table(self, scores: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
        """task -> the targets.json rows its `target` names (one id, or a list of ids as DMB uses), with jev score,
        metric, n, bars and role, so a reader of run.json can set the task's number beside Jev's."""
        rows = self._target_rows()
        out: dict[str, list[dict[str, Any]]] = {}
        for task, r in (scores.get("tasks") or {}).items():
            tid = r.get("target") if isinstance(r, Mapping) else None
            if not tid:
                continue
            ids = [str(t) for t in tid] if isinstance(tid, (list, tuple, set)) else [str(tid)]
            table = []
            for one in ids:
                row = rows.get(one)
                if row is None:
                    table.append({"target": one, "error": "not a targets.json id"})
                    continue
                table.append({"target": one, "dataset": row.get("dataset"), "metric": row.get("metric"), "n": row.get("n"),
                              "jev_score": row.get("jev_score"), "higher_is_better": row.get("higher_is_better"),
                              "bar_z": row.get("bar_z"), "bar_s": row.get("bar_s"), "role": row.get("role"), "counted": row.get("counted")})
            out[task] = table
        return out

    # ------------------------------------------------------------------ verify / run

    def verify(self, ctx: RunContext) -> dict[str, Any]:
        inner = self.inner
        split = self.split()
        rep: dict[str, Any] = {
            "contract": "group-B (prepare / items / score) through bridge_b",
            "split": split, "eval_split": inner.EVAL_SPLIT, "splits": list(inner.SPLITS), "tasks": list(inner.TASKS),
            "targets": list(inner.TARGETS), "stages": int(getattr(inner, "STAGES", 1)), "work": str(inner.work),
        }
        if not getattr(self.args, "skip_prepare", False):
            man = inner.prepare(split)
            rep["prepare"] = man
            for k, v in collect_revisions(man).items():
                ctx.dataset_revisions.setdefault(k, v)
        rep["expected_counts"] = inner.expected_counts(split)
        rep["ok"] = True
        return rep

    def run(self, session: Session, ctx: RunContext) -> dict[str, Any]:
        inner = self.inner
        split = self.split()
        self.guard_eval_split(split)
        if getattr(self.args, "fit", False) and split == inner.EVAL_SPLIT:  # before any request is sent
            raise SystemExit(f"{self.spec.id}: thresholds are fitted on non-evaluated items only (PLAN §2.1), not on {split!r}")
        tasks = self._tasks(getattr(self.args, "tasks", None))
        limit = getattr(self.args, "limit", None)
        out = ctx.out
        out.mkdir(parents=True, exist_ok=True)
        items_path, answers_path, scores_path = out / "items.jsonl", out / "answers.jsonl", out / "scores.json"

        answers: dict[str, dict[str, Any]] = _load_answers(answers_path) if getattr(self.args, "resume", False) else {}
        resumed = len(answers)
        if not resumed and answers_path.exists():
            answers_path.unlink()
        if resumed:
            ctx.note(f"resumed {resumed} answered items from {answers_path}")

        items = list(inner.items(split, limit=limit, tasks=tasks))
        _write_items(items_path, items)

        def ask_all(batch: Sequence[Any]) -> None:
            for it in batch:
                if it.id in answers:
                    continue
                o = session.ask(it.request["state"], it.request["questions"], {"id": it.id, "task": it.task, "group": it.group})
                rec = answer_record(it.id, o)
                append_jsonl(answers_path, rec)
                answers[it.id] = rec

        ask_all(items)
        stages = int(getattr(inner, "STAGES", 1))
        for stage in range(2, stages + 1):
            extra = list(inner.stage_items(stage, items, answers))
            if not extra:
                break
            items = items + extra
            _write_items(items_path, items)
            ask_all(extra)

        thresholds = None
        if getattr(self.args, "thresholds", None):
            thresholds = json.loads(Path(self.args.thresholds).expanduser().read_text(encoding="utf-8"))
        scores = inner.score(items, answers, split, thresholds=thresholds)
        write_json(scores_path, scores)

        fitted_path: Path | None = None
        if getattr(self.args, "fit", False):
            th = inner.fit(items, answers)
            if th is None:
                ctx.note("--fit: this spec has no tuned variant")
            else:
                fitted_path = out / "thresholds.json"
                write_json(fitted_path, th)

        comparable = split == inner.EVAL_SPLIT and not limit and not tasks
        res: dict[str, Any] = {
            "split": split, "eval_split": inner.EVAL_SPLIT, "comparable": comparable,
            "tasks": tasks or list(inner.TASKS), "limit": limit, "stages": stages, "n_items": len(items),
            "coverage": answer_counts(answers), "resumed": resumed,
            "scores": {k: v for k, v in scores.items() if k != "per_item"},
            "targets": self.target_table(scores),
            "thresholds": getattr(self.args, "thresholds", None), "thresholds_fitted": str(fitted_path) if fitted_path else None,
            "files": {"items": str(items_path), "answers": str(answers_path), "scores": str(scores_path)},
        }
        if not comparable:
            res["note"] = "subset / non-evaluated split: self-check only, not comparable with the published Jev number"
        return res


def bridge_class(inner_cls: type) -> type[BridgedAdapter]:
    """A `BridgedAdapter` subclass bound to one group-B adapter class (cached per class)."""
    cached = getattr(inner_cls, "_bridged_runner_class", None)
    if cached is not None and getattr(cached, "inner_cls", None) is inner_cls:
        return cached
    attrs = {
        "inner_cls": inner_cls,
        "SPEC_ID": getattr(inner_cls, "SPEC_ID", ""),
        "TARGETS": tuple(getattr(inner_cls, "TARGETS", ())),
        "TASKS": tuple(getattr(inner_cls, "TASKS", ())),
        "SPLITS": tuple(getattr(inner_cls, "SPLITS", ())),
        "EVAL_SPLIT": getattr(inner_cls, "EVAL_SPLIT", "test"),
        "STAGES": int(getattr(inner_cls, "STAGES", 1)),
        "__doc__": f"runner bridge for {inner_cls.__module__}.{inner_cls.__qualname__}",
        "__module__": __name__,
    }
    cls = type(f"Bridged_{inner_cls.__module__.rsplit('.', 1)[-1]}", (BridgedAdapter,), attrs)
    try:
        inner_cls._bridged_runner_class = cls  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        pass
    return cls
