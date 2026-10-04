"""earino/zero-shot-complaint-benchmark (targets T521 bare, T522 form instruction + definitions): CFPB, 113 issue labels.

Protocol (@ 33d5ed74, MIT; `code/jev_benchmark.py`, `utils/data_utils.py`, `criteria/{v1,v1instr}.json` vendored):
- data: determined-ai/consumer_complaints_medium @ 4783de6e, train split rows at `data/val_indices.json` (6,430 =
  the course's canonical validation split), labels merged by `label_merge_mapping.json`, 113 canonical labels in
  `label_list.json` order; "Consumer Complaint" / "Issue" columns;
- request: state {"complaint": text}; one Choice `issue` over the 113 labels in order with criteria
  {label: None} (bare, T521) or v1 (T522: the form-aware instruction + structured {what, not_for} definitions for 34
  labels, None for the rest) / v1instr (instruction only);
- metrics: accuracy (`choice`), macro-F1 (sklearn default: union of observed gold/pred labels), zero-F1 class count,
  top-label ECE with 15 right-closed bins, accuracy by confidence band.
The Jev per-item responses in the repo (`jev_responses/`, licence "other") are never read.
Self-check split "validation": train-split rows outside the eval indices.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from jev_local.bench.benchmax.adapters_b.common import (VENDOR, AdapterB, Item, accuracy, answers_of, by_task,
                                                        choice_pick, choice_probs, coverage, ece_equal_width, macro_f1,
                                                        take)

SPEC_ID = "study:earino_zero_shot_complaint_benchmark_cfpb_113_cl"
TARGETS = ("T521", "T522")
UPSTREAM = {"repo": "earino/zero-shot-complaint-benchmark", "commit": "33d5ed7467eff8831ccb47730dc14a535395544f", "licence": "MIT"}
DATASET = "determined-ai/consumer_complaints_medium"
REVISION = "4783de6e089f8ef1dccb26e9f46d491ec0273c83"
TEXT_COLUMN, LABEL_COLUMN = "Consumer Complaint", "Issue"
INSTRUCTIONS_BARE = "Which issue category best describes the consumer financial complaint in `complaint`?"
TASKS = ("bare", "v1", "v1instr")
TARGET_OF = {"bare": "T521", "v1": "T522"}


def label_list() -> list[str]:
    return json.loads((VENDOR / "earino" / "label_list.json").read_text(encoding="utf-8"))


def merge_map() -> dict[str, str]:
    raw = json.loads((VENDOR / "earino" / "label_merge_mapping.json").read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def val_indices() -> list[int]:
    return json.loads((VENDOR / "earino" / "val_indices.json").read_text(encoding="utf-8"))


def criteria_variant(variant: str) -> tuple[str | dict, dict[str, Any]]:
    labels = label_list()
    if variant == "bare":
        return INSTRUCTIONS_BARE, {lab: None for lab in labels}
    spec = json.loads((VENDOR / "earino" / f"{variant}.json").read_text(encoding="utf-8"))
    defs = spec.get("criteria") or {}
    unknown = set(defs) - set(labels)
    if unknown:
        raise ValueError(f"{variant}: definitions for unknown labels {sorted(unknown)[:3]}")
    return spec["instructions"], {lab: defs.get(lab) for lab in labels}


def build_request(text: str, variant: str = "bare") -> dict[str, Any]:
    instr, crit = criteria_variant(variant)
    return {"state": {"complaint": text}, "questions": {"issue": {"type": "choice", "instructions": instr, "criteria": crit}}}


def ece15(probs: Sequence[Sequence[float]], correct: Sequence[bool]) -> float:
    return ece_equal_width([max(r) for r in probs], correct, 15, right_closed=True)


class Adapter(AdapterB):
    SPEC_ID = SPEC_ID
    TARGETS = TARGETS
    SPLITS = ("validation", "test")  # "test" = the 6,430 canonical validation rows (earino's eval split)
    TASKS = TASKS

    def _rows(self, split: str, limit: int | None) -> list[tuple[int, str, int]]:
        """(row index in the HF train split, text, label id) after merge + filter, in index order."""
        from datasets import load_dataset  # VM only

        ds = load_dataset(DATASET, split="train", revision=REVISION, cache_dir=str(self.work / "hf" / "datasets"))
        labels = label_list()
        lid = {lab: i for i, lab in enumerate(labels)}
        mm = merge_map()
        val = val_indices()
        if split == "test":
            idx = val
        else:
            vs = set(val)
            idx = [i for i in range(len(ds)) if i not in vs]
        out = []
        for i in idx:
            r = ds[int(i)]
            merged = mm.get(r[LABEL_COLUMN], r[LABEL_COLUMN])
            if merged not in lid:
                continue
            out.append((int(i), r[TEXT_COLUMN], lid[merged]))
            if limit is not None and len(out) >= limit:
                break
        if split == "test" and limit is None and len(out) != 6430:
            raise ValueError(f"cfpb: {len(out)} evaluation rows, expected 6430")
        return out

    def prepare(self, split: str) -> dict[str, Any]:
        return {"spec": SPEC_ID, "upstream": UPSTREAM, "dataset": DATASET, "revision": REVISION, "n_labels": len(label_list()),
                "n_eval": len(val_indices())}

    def expected_counts(self, split: str) -> dict[str, int]:
        return {t: 6430 for t in TASKS} if split == "test" else {}

    def items(self, split: str, limit: int | None = None, tasks: Sequence[str] | None = None) -> list[Item]:
        rows = self._rows(split, limit)
        out = []
        for variant in self._select_tasks(tasks):
            for pos, (i, text, y) in enumerate(take(rows, limit)):
                out.append(Item(f"{variant}/{i}", variant, build_request(text, variant), y, meta={"val_idx": pos, "row": i}))
        return out

    def score(self, items: Sequence[Item], answers: dict[str, dict[str, Any]], split: str,
              thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
        labels = label_list()
        res: dict[str, Any] = {"spec": SPEC_ID, "split": split, "tasks": {}}
        for variant, its in by_task(items).items():
            gold, pred, probs, failed = [], [], [], 0
            for it in its:
                a = answers_of(answers.get(it.id))
                pick = choice_pick(a.get("issue"), labels) if a else None
                pr = choice_probs(a.get("issue"), labels) if a else None
                if pick is None or pr is None:
                    failed += 1
                    continue
                gold.append(it.gold)
                pred.append(labels.index(pick))
                probs.append(pr)
            r: dict[str, Any] = {**coverage(its, answers), "failed": failed, "target": TARGET_OF.get(variant)}
            if gold:
                correct = [p == g for p, g in zip(pred, gold)]
                present = sorted(set(gold))
                per_class = [macro_f1([p for p, g in zip(pred, gold)], gold, [c]) for c in present]
                r.update({"n": len(gold), "accuracy": accuracy(pred, gold), "macro_f1": macro_f1(pred, gold),
                          "macro_f1_present_classes": sum(per_class) / len(per_class),
                          "zero_f1_classes": sum(1 for f in per_class if f == 0.0), "classes_present": len(present),
                          "ece_15": ece15(probs, correct)})
                conf = [max(p) for p in probs]
                r["by_confidence"] = {}
                for lo, hi in ((0.9, 1.01), (0.5, 0.9), (0.0, 0.5)):
                    m = [i for i, c in enumerate(conf) if lo <= c < hi]
                    if m:
                        r["by_confidence"][f"[{lo},{min(hi, 1.0)}]"] = {"share": len(m) / len(conf),
                                                                    "accuracy": sum(correct[i] for i in m) / len(m)}
                if failed:
                    r["accuracy_missing_as_wrong"] = sum(correct) / len(its)
            res["tasks"][variant] = r
        return res
