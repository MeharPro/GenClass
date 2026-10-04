"""stperic/jev-medhallu-benchmark (target T319): MedHallu, 1,000 MedHELM test items, accuracy at the dev-chosen threshold.

Protocol (@ 8a7f2f88, MIT; `bench/{run_bench,score,common}.py`, `datasets/medhallu/{task.json,PREREGISTRATION.md}`):
- items: the committed `items.test.jsonl` (the exact 1,000 MedHELM v4.0.0 instances; sha256 pinned) and the dev
  samples `items.dev.jsonl` / `items.dev2.jsonl` (500 each, pqa_artificial; our split "validation"); the pinned
  checkout is read, not re-downloaded;
- request (run 2, the published .929): state = {"knowledge", "question", "answer"}; one Noul `authors_reject` with
  the dev-chosen question and no criteria (`jev_questions` of task.json, combined by `mean`);
  secondary (run 1, .903): the MedHELM-style Noul `label` with the object instructions of task.json and criteria
  {"true": labels["1"], "false": labels["0"]} (objects), threshold 0.85;
- verdict: P(positive) >= threshold (0.65 frozen for run 2); accuracy with errors counted wrong, Wilson CI,
  AUROC, threshold sweep, slices by difficulty / category.
Track: Jev's number is dev-tuned (uses-train-data): compared on the S track only (targets.json). The test file is
read only to build requests and score; thresholds are chosen on dev.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from jev_local.bench.benchmax.adapters_b.common import (VENDOR, AdapterB, Item, answers_of, by_task, coverage, noul_p,
                                                        repos_dir, roc_auc, sha256_file, take, wilson_ci)

SPEC_ID = "stperic_medhallu"
TARGETS = ("T319",)
UPSTREAM = {"repo": "stperic/jev-medhallu-benchmark", "commit": "8a7f2f88eb258fd0bc859ed4aeca3b80643e5d94", "licence": "MIT"}
ITEM_FILES = {"test": ("items.test.jsonl", "10e4a36157a9550e399872544806524c1287a324dc12c83bb4bca9d7eae2dda7", 1000),
              "dev": ("items.dev.jsonl", "ebd0a02d0edc604c86b34022f7df62b4267c510a296821d2bcba51981776cdfd", 500),
              "dev2": ("items.dev2.jsonl", "3406c10cd7f454a04a3db23a1c86d2d3c73c0b7ba792e6f1d1a501cf701f9072", 500)}
THRESHOLDS = {"authors_question": 0.65, "medhelm_question": 0.85}
TASKS = ("authors_question", "medhelm_question")
SWEEP = (0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)


def task_json() -> dict[str, Any]:
    return json.loads((VENDOR / "stperic" / "task.json").read_text(encoding="utf-8"))


def build_request(state: dict[str, Any], task: str = "authors_question") -> dict[str, Any]:
    t = task_json()
    if task == "authors_question":
        qs = {qid: {"type": "noul", "instructions": q["instructions"], **({"criteria": q["criteria"]} if q.get("criteria") else {})}
              for qid, q in t["jev_questions"].items()}
    else:
        pos, neg = t["positive"], next(l for l in t["labels"] if l != t["positive"])
        qs = {"label": {"type": "noul", "instructions": t["instructions"],
                        "criteria": {"true": t["labels"][pos], "false": t["labels"][neg]}}}
    return {"state": state, "questions": qs}


def p_positive(answers: dict[str, Any], task: str) -> float | None:
    t = task_json()
    if task == "authors_question":
        ps = [noul_p(answers.get(qid)) for qid in t["jev_questions"]]
        if any(p is None for p in ps):
            return None
        rule = t.get("jev_combine", "mean")
        return sum(ps) / len(ps) if rule == "mean" else max(ps)
    return noul_p(answers.get("label"))


class Adapter(AdapterB):
    SPEC_ID = SPEC_ID
    TARGETS = TARGETS
    SPLITS = ("validation", "test")  # validation = dev + dev2
    TASKS = TASKS

    def _file(self, which: str) -> Path:
        name, sha, _ = ITEM_FILES[which]
        p = repos_dir() / "stperic" / "bench" / "datasets" / "medhallu" / "data" / name
        if not p.exists():
            raise FileNotFoundError(f"missing {p}: clone stperic/jev-medhallu-benchmark @ 8a7f2f88 under {repos_dir()}")
        if sha256_file(p) != sha:
            raise ValueError(f"{p} does not match the pinned sha256")
        return p

    def _rows(self, split: str) -> list[dict[str, Any]]:
        which = ("test",) if split == "test" else ("dev", "dev2")
        rows = []
        for w in which:
            p = self._file(w)
            got = [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l]
            if len(got) != ITEM_FILES[w][2]:
                raise ValueError(f"{w}: {len(got)} items, expected {ITEM_FILES[w][2]}")
            for r in got:
                r["_file"] = w
            rows.extend(got)
        return rows

    def prepare(self, split: str) -> dict[str, Any]:
        return {"spec": SPEC_ID, "upstream": UPSTREAM, "files": {w: {"sha256": s, "n": n, "path": str(self._file(w))}
                                                                for w, (_, s, n) in ITEM_FILES.items()},
                "thresholds": THRESHOLDS, "task_fingerprint_sha256": sha256_file(VENDOR / "stperic" / "task.json")}

    def expected_counts(self, split: str) -> dict[str, int]:
        n = 1000 if split == "test" else 1000  # dev + dev2
        return {t: n for t in TASKS}

    def items(self, split: str, limit: int | None = None, tasks: Sequence[str] | None = None) -> list[Item]:
        rows = self._rows(split)
        out = []
        for task in self._select_tasks(tasks):
            for r in take(rows, limit):
                out.append(Item(f"{task}/{r['id']}", task, build_request(r["state"], task), r["gold"],
                                meta={"difficulty": r.get("difficulty"), "category": r.get("category"), "file": r["_file"]}))
        return out

    def score(self, items: Sequence[Item], answers: dict[str, dict[str, Any]], split: str,
              thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
        t = task_json()
        pos = t["positive"]
        res: dict[str, Any] = {"spec": SPEC_ID, "split": split, "tasks": {}}
        for task, its in by_task(items).items():
            th = float((thresholds or {}).get(task, THRESHOLDS[task]))
            scored: list[tuple[float, bool, Item]] = []
            errors = 0
            for it in its:
                a = answers_of(answers.get(it.id))
                p = p_positive(a, task) if a else None
                if p is None:
                    errors += 1
                    continue
                scored.append((p, it.gold == pos, it))
            n = len(its)
            correct = sum((p >= th) == g for p, g, _ in scored)  # errors count as wrong
            r: dict[str, Any] = {**coverage(its, answers), "threshold": th, "errors": errors, "accuracy": correct / n if n else None,
                                 "accuracy_ci95": list(wilson_ci(correct, n)) if n else None, "target": "T319" if task == "authors_question" else None}
            if scored:
                r["auroc"] = roc_auc([p for p, _, _ in scored], [g for _, g, _ in scored])
                r["sweep"] = {str(s): sum((p >= s) == g for p, g, _ in scored) / n for s in SWEEP}
                for field in ("difficulty", "category"):
                    sl: dict[str, list[bool]] = {}
                    for p, g, it in scored:
                        v = it.meta.get(field)
                        if v is not None:
                            sl.setdefault(str(v), []).append((p >= th) == g)
                    r[f"by_{field}"] = {k: {"n": len(v), "accuracy": sum(v) / len(v)} for k, v in sorted(sl.items())}
            res["tasks"][task] = r
        return res

    def fit(self, items: Sequence[Item], answers: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
        """Threshold = the middle of the flattest high-accuracy region on dev+dev2 is a judgement call upstream;
        here: the sweep value with the highest validation accuracy (ties -> nearest 0.65 / 0.85)."""
        t = task_json()
        out: dict[str, Any] = {}
        for task, its in by_task(items).items():
            scored = []
            for it in its:
                a = answers_of(answers.get(it.id))
                p = p_positive(a, task) if a else None
                if p is not None:
                    scored.append((p, it.gold == t["positive"]))
            if not scored:
                continue
            acc = {s: sum((p >= s) == g for p, g in scored) / len(scored) for s in SWEEP}
            best = max(acc.values())
            out[task] = min((s for s, v in acc.items() if v == best), key=lambda s: abs(s - THRESHOLDS[task]))
        return out or None
