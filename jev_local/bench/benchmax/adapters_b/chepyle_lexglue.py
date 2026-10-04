"""chepyle/jev-test `lexglue-systemone-v1` (targets T269-T280): LexGLUE's 7 tasks, BANKING77 and CLINC150 OOS+.

Protocol (chepyle/jev-test @ eeb55e2c, Apache-2.0; `src/jev_test/{questions,data,metrics}.py`,
`analysis/tune_threshold.py`), rebuilt byte for byte:
- data: `coastalcph/lex_glue` @ c23fdff1a6bf74e0e1a71cb86f1e781d37da888c, parquet per task/split; label order checked
  against the pinned catalog (vendor/chepyle/labels.json, CC BY 4.0 / CC BY-SA 4.0 descriptors, see NOTICE);
  BANKING77 from the PolyAI CSVs @ 57ec275d (sha256-pinned), CLINC150 `clinc/clinc_oos` plus @ 155b9c71;
- request: state {"document": text}; multilabel tasks = one noul per label `label_{i}` with instructions
  "<task instruction> Does this label apply: <description>?" and true/false criteria; single-label tasks = one
  choice `label` with criteria {"0": option, ...} (CaseHOLD: the five holdings; intents: the readable intent names);
- long documents: head + tail within 48,000 characters around a fixed omission marker;
- prediction: multilabel p > threshold (0.5 by default), single = the chosen key;
- metrics: LexGLUE's micro/macro-F1 (synthetic 'none' column for ECtHR A/B and UNFAIR-ToS, not EUR-LEX; macro over
  all columns), exact match; arithmetic/harmonic mean over the 7 tasks; intents add accuracy, in-scope accuracy and
  OOS precision/recall (CLINC);
- tuned variant (T277, Jev given train data): per-label thresholds maximising validation F1 on the 0.05..0.95 grid,
  ties to the value nearest 0.5, labels without validation positives keep 0.5; a global per-task threshold
  maximising validation micro-F1 is reported beside it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from jev_local.bench.benchmax.adapters_b.common import (VENDOR, AdapterB, Item, answers_of, by_task, choice_pick,
                                                        coverage, fetch_verified, hf_parquet_split, noul_p,
                                                        parquet_feature_names, read_csv_rows, multilabel_f1,
                                                        macro_f1, accuracy, take)

SPEC_ID = "chepyle_lexglue_systemone_v1"
TARGETS = ("T269", "T270", "T271", "T272", "T273", "T274", "T275", "T276", "T277", "T278", "T279", "T280")
PROTOCOL_VERSION = "lexglue-systemone-v1"
UPSTREAM = {"repo": "chepyle/jev-test", "commit": "eeb55e2c79dd9a41ec19e8f5877f6a8c65532712", "licence": "Apache-2.0"}
LEXGLUE_TASKS = ("ecthr_a", "ecthr_b", "scotus", "eurlex", "ledgar", "unfair_tos", "case_hold")
INTENT_TASKS = ("banking77", "clinc150")
NONE_CLASS_TASKS = frozenset({"ecthr_a", "ecthr_b", "unfair_tos"})
MAX_CHARS = 48_000
THRESHOLD = 0.5
MARKER = "\n\n[... middle omitted by benchmark ...]\n\n"
GRID = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 .. 0.95

# questions.py INSTRUCTIONS, verbatim (Apache-2.0, chepyle 2026; see vendor/chepyle/NOTICE)
INSTRUCTIONS = {
    "ecthr_a": "Using the facts of this European Court of Human Rights case, predict which "
    "Convention provisions the court found to have been violated.",
    "ecthr_b": "Using the facts of this European Court of Human Rights case, identify which "
    "Convention provisions were allegedly violated and considered by the court. "
    "An allegation counts even if the court ultimately found no violation.",
    "scotus": "Classify this US Supreme Court opinion by the main issue area of the dispute.",
    "eurlex": "Classify this European Union legal document by its EuroVoc subject concepts. "
    "Several concepts may apply independently.",
    "ledgar": "Select the single main topic of this contractual provision.",
    "unfair_tos": "Identify potentially unfair contractual terms in this terms-of-service "
    "sentence under European consumer law. Select a category only if the "
    "sentence is potentially unfair in that respect; a neutral mention is insufficient.",
    "case_hold": "Select the holding that correctly fills the masked citation in this court "
    "opinion excerpt. Exactly one of the five candidate holdings is correct.",
    "banking77": "Classify this customer message to a bank by the customer's intent. "
    "Select the single intent that best matches the request.",
    "clinc150": "Classify this request to a virtual assistant by the user's intent. Select "
    "the single matching intent, or out of scope if the request fits none of the listed intents.",
}


def _catalogs() -> tuple[dict, dict]:
    lab = json.loads((VENDOR / "chepyle" / "labels.json").read_text(encoding="utf-8"))
    intents = json.loads((VENDOR / "chepyle" / "intents.json").read_text(encoding="utf-8"))
    return lab, intents


def task_entry(name: str) -> dict:
    lab, intents = _catalogs()
    if name in lab["tasks"]:
        return lab["tasks"][name]
    if name in intents["tasks"]:
        return intents["tasks"][name]
    raise KeyError(name)


def truncate(text: str, max_chars: int = MAX_CHARS) -> tuple[str, bool]:
    """chepyle `questions.truncate`: keep head and tail within max_chars around the omission marker."""
    if max_chars < 0 or 0 < max_chars < 100:
        raise ValueError("max_chars must be 0 (unlimited) or at least 100")
    if not max_chars or len(text) <= max_chars:
        return text, False
    available = max_chars - len(MARKER)
    head = (available + 1) // 2
    tail = available - head
    return text[:head] + MARKER + text[-tail:], True


def build_request(task: str, text: str, endings: Sequence[str] | None = None, max_chars: int = MAX_CHARS,
                  ) -> tuple[dict[str, Any], dict[str, Any]]:
    """chepyle `questions.build_request` without the model field (the runner adds it). Returns (request, info)."""
    entry = task_entry(task)
    sent, truncated = truncate(text, max_chars)
    instruction = INSTRUCTIONS[task]
    if entry["kind"] == "multilabel":
        questions = {
            f"label_{i}": {
                "type": "noul",
                "instructions": f"{instruction} Does this label apply: {description}?",
                "criteria": {
                    "true": f"The label '{description}' applies to this document.",
                    "false": f"The label '{description}' does not apply to this document.",
                },
            }
            for i, description in enumerate(entry["descriptions"])
        }
    else:
        options = list(endings) if task == "case_hold" else list(entry["descriptions"])
        if task == "case_hold" and len(options) != 5:
            raise ValueError("case_hold needs five holdings")
        questions = {"label": {"type": "choice", "instructions": instruction,
                               "criteria": {str(i): option for i, option in enumerate(options)}}}
    return ({"state": {"document": sent}, "questions": questions},
            {"original_chars": len(text), "sent_chars": len(sent), "truncated": truncated})


def normalize_row(task: str, index: int, row: dict[str, Any]) -> dict[str, Any]:
    """chepyle `data.normalize`: text (paragraph lists joined by blank lines), sorted gold ids, holdings."""
    entry = task_entry(task)
    source = entry.get("source")
    if source:
        label = row[source["label"]]
        if isinstance(label, str):
            if label not in entry["codes"]:
                raise ValueError(f"{task}/{index}: unknown label {label!r}")
            label = entry["codes"].index(label)
        row = {"text": row[source["text"]], "label": label}
    text = row["context"] if task == "case_hold" else row["text"]
    if isinstance(text, list) and all(isinstance(p, str) for p in text):
        text = "\n\n".join(text)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{task}/{index}: missing document text")
    labels = row["labels"] if entry["kind"] == "multilabel" else [row["label"]]
    labels = sorted(int(x) for x in labels)
    if any(not 0 <= x < len(entry["codes"]) for x in labels) or len(set(labels)) != len(labels):
        raise ValueError(f"{task}/{index}: invalid labels {labels}")
    if entry["kind"] != "multilabel" and len(labels) != 1:
        raise ValueError(f"{task}/{index}: single-label task with {len(labels)} labels")
    out = {"id": f"{task}/{index}", "task": task, "index": index, "text": text, "gold": labels}
    if task == "case_hold":
        endings = row["endings"]
        if len(endings) != 5 or any(not isinstance(x, str) or not x.strip() for x in endings):
            raise ValueError(f"{task}/{index}: expected five nonempty holdings")
        out["endings"] = list(endings)
    return out


def predictions_from_answers(task: str, answers: dict[str, Any] | None, thresholds: Sequence[float] | float = THRESHOLD,
                             ) -> tuple[list[int] | None, list[float] | None]:
    """chepyle `questions.parse_response`: (selected label ids, probabilities) or (None, None) on a bad answer."""
    entry = task_entry(task)
    k = len(entry["codes"])
    if answers is None:
        return None, None
    if entry["kind"] == "multilabel":
        probs = []
        for i in range(k):
            p = noul_p(answers.get(f"label_{i}"))
            if p is None or not 0 <= p <= 1:
                return None, None
            probs.append(p)
        th = [thresholds] * k if isinstance(thresholds, (int, float)) else list(thresholds)
        return [i for i, p in enumerate(probs) if p > th[i]], probs
    keys = [str(i) for i in range(k if task != "case_hold" else 5)]
    pick = choice_pick(answers.get("label"), keys)
    if pick is None:
        return None, None
    probs = answers["label"].get("probabilities") or {}
    return [int(pick)], [float(probs.get(kk, 0.0) or 0.0) for kk in keys]


def score_task(task: str, gold: Sequence[Sequence[int]], pred: Sequence[Sequence[int]]) -> dict[str, float]:
    """chepyle `metrics.score`: LexGLUE conventions (none column; macro over all columns / observed union)."""
    entry = task_entry(task)
    if entry["kind"] == "multilabel":
        return multilabel_f1(pred, gold, len(entry["codes"]), none_column=task in NONE_CLASS_TASKS)
    g = [x[0] for x in gold]
    p = [x[0] for x in pred]
    return {"micro_f1": accuracy(p, g), "macro_f1": macro_f1(p, g), "exact_match": accuracy(p, g)}


def nearest_half_best(values: dict[float, float]) -> float:
    top = max(values.values())
    return min((t for t, v in values.items() if v == top), key=lambda t: abs(t - 0.5))


def tune_thresholds(task: str, gold: Sequence[Sequence[int]], probs: Sequence[Sequence[float]]) -> dict[str, Any]:
    """analysis/tune_threshold.py `tune`: global (validation micro-F1) and per-label (per-label F1) thresholds."""
    entry = task_entry(task)
    k = len(entry["codes"])
    by_global = {t: score_task(task, gold, [[i for i, p in enumerate(row) if p > t] for row in probs])["micro_f1"]
                 for t in GRID}
    per_label = [0.5] * k
    for j in range(k):
        positives = sum(1 for g in gold if j in g)
        if not positives:
            continue
        f1 = {}
        for t in GRID:
            tp = sum(1 for g, row in zip(gold, probs) if row[j] > t and j in g)
            npred = sum(1 for row in probs if row[j] > t)
            d = npred + positives
            f1[t] = 2 * tp / d if d else 0.0
        per_label[j] = nearest_half_best(f1)
    return {"global": nearest_half_best(by_global), "per_label": per_label, "val_micro_by_global": by_global}


def aggregate(task_scores: dict[str, dict[str, float]]) -> dict[str, Any]:
    import math
    import statistics

    out: dict[str, Any] = {}
    for metric in ("micro_f1", "macro_f1"):
        vals = [task_scores[t][metric] for t in LEXGLUE_TASKS if t in task_scores]
        if not vals:
            continue
        out[metric] = {"arithmetic": statistics.mean(vals), "harmonic": statistics.harmonic_mean(vals) if all(vals) else 0.0,
                       "geometric": 0.0 if 0 in vals else math.exp(statistics.mean(map(math.log, vals))),
                       "n_tasks": len(vals)}
    return out


class Adapter(AdapterB):
    SPEC_ID = SPEC_ID
    TARGETS = TARGETS
    SPLITS = ("validation", "test")
    TASKS = LEXGLUE_TASKS + INTENT_TASKS

    # ------------------------------------------------------------------ data
    def _rows(self, task: str, split: str) -> list[dict[str, Any]]:
        entry = task_entry(task)
        if task in LEXGLUE_TASKS:
            lab, _ = _catalogs()
            rows, paths = hf_parquet_split(lab["dataset"], lab["revision"], split, config=task, work=self.work)
            col = "labels" if entry["kind"] == "multilabel" else "label"
            names = parquet_feature_names(paths[0], col)
            if names is not None and names != entry["codes"]:
                raise ValueError(f"{task}: dataset label order differs from the pinned catalog")
        else:
            src = entry["source"]
            if src["type"] == "csv":
                info = src["files"].get(split)
                if info is None:
                    raise KeyError(f"{task} has no {split!r} split (chepyle intents: train/test only)")
                p = fetch_verified(info["url"], self.raw_dir("chepyle") / f"{task}-{split}.csv", info["sha256"])
                rows = read_csv_rows(p, skipinitialspace=True)
            else:
                rows, paths = hf_parquet_split(src["repo"], src["revision"], split, config=src["config"], work=self.work)
                names = parquet_feature_names(paths[0], src["label"])
                if names is not None and names != entry["codes"]:
                    raise ValueError(f"{task}: dataset label order differs from the pinned catalog")
        expected = entry["split_sizes"].get(split)
        if expected is not None and len(rows) != expected:
            raise ValueError(f"{task}/{split}: {len(rows)} rows, expected {expected} (pinned revision drifted?)")
        return rows

    def prepare(self, split: str) -> dict[str, Any]:
        lab, intents = _catalogs()
        return {"spec": SPEC_ID, "protocol": PROTOCOL_VERSION, "upstream": UPSTREAM, "dataset": lab["dataset"],
                "revision": lab["revision"], "split": split,
                "counts": {t: self._count_split(t, split) for t in self.TASKS}}

    def _count_split(self, task: str, split: str) -> int | None:
        return task_entry(task)["split_sizes"].get(split)

    def expected_counts(self, split: str) -> dict[str, int]:
        return {t: n for t in self.TASKS if (n := self._count_split(t, split)) is not None}

    def items(self, split: str, limit: int | None = None, tasks: Sequence[str] | None = None) -> list[Item]:
        out: list[Item] = []
        for task in self._select_tasks(tasks):
            if self._count_split(task, split) is None:
                continue
            rows = self._rows(task, split)
            idx = range(len(rows)) if limit is None else range(min(limit, len(rows)))
            for i in idx:
                row = normalize_row(task, i, rows[i])
                req, info = build_request(task, row["text"], row.get("endings"))
                out.append(Item(row["id"], task, req, row["gold"], meta={**info, "index": i}))
        return out

    # ------------------------------------------------------------------ scoring
    def score(self, items: Sequence[Item], answers: dict[str, dict[str, Any]], split: str,
              thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {"spec": SPEC_ID, "protocol": PROTOCOL_VERSION, "split": split, "threshold": THRESHOLD,
                                  "tasks": {}}
        for task, its in by_task(items).items():
            entry = task_entry(task)
            variants: dict[str, Sequence[float] | float] = {"default_0.5": THRESHOLD}
            if thresholds and task in thresholds and entry["kind"] == "multilabel":
                variants["global"] = float(thresholds[task]["global"])
                variants["per_label"] = list(thresholds[task]["per_label"])
            gold_all = [it.gold for it in its]
            res: dict[str, Any] = {"n": len(its), **coverage(its, answers)}
            for name, th in variants.items():
                preds, golds, failed = [], [], 0
                for it in its:
                    pred, _ = predictions_from_answers(task, answers_of(answers.get(it.id)), th)
                    if pred is None:
                        failed += 1
                        continue
                    preds.append(pred)
                    golds.append(it.gold)
                sc = score_task(task, golds, preds) if golds else {}
                # chepyle scores answered rows only; "missing = wrong" counts a failed row as an empty set
                sc_all = score_task(task, gold_all, [p for p in preds] + [[] if entry["kind"] == "multilabel" else [-1]] * failed) \
                    if failed and golds else sc
                res[name] = {**sc, "failed": failed, "missing_as_wrong": sc_all if failed else None}
            if task == "clinc150":
                res["clinc"] = self._clinc_extras(its, answers)
            res["truncated"] = sum(1 for it in its if it.meta.get("truncated"))
            result["tasks"][task] = res
        result["lexglue_aggregate"] = aggregate({t: r["default_0.5"] for t, r in result["tasks"].items()
                                                if t in LEXGLUE_TASKS and r["default_0.5"]})
        if thresholds:
            result["lexglue_aggregate_per_label_thresholds"] = aggregate(
                {t: r.get("per_label", r["default_0.5"]) for t, r in result["tasks"].items() if t in LEXGLUE_TASKS})
        return result

    @staticmethod
    def _clinc_extras(its: Sequence[Item], answers: dict[str, dict[str, Any]]) -> dict[str, Any]:
        oos = task_entry("clinc150")["codes"].index("oos")
        tp_oos = pred_oos = gold_oos = in_ok = in_n = 0
        for it in its:
            pred, _ = predictions_from_answers("clinc150", answers_of(answers.get(it.id)))
            p = pred[0] if pred else -1
            g = it.gold[0]
            if g == oos:
                gold_oos += 1
                tp_oos += p == oos
            else:
                in_n += 1
                in_ok += p == g
            pred_oos += p == oos
        return {"in_scope_accuracy": in_ok / in_n if in_n else None, "in_scope_n": in_n,
                "oos_recall": tp_oos / gold_oos if gold_oos else None, "oos_n": gold_oos,
                "oos_precision": tp_oos / pred_oos if pred_oos else None}

    def fit(self, items: Sequence[Item], answers: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
        out: dict[str, Any] = {}
        for task, its in by_task(items).items():
            if task_entry(task)["kind"] != "multilabel":
                continue
            gold, probs = [], []
            for it in its:
                pred, pr = predictions_from_answers(task, answers_of(answers.get(it.id)))
                if pr is not None:
                    gold.append(it.gold)
                    probs.append(pr)
            if gold:
                out[task] = tune_thresholds(task, gold, probs)
        return out or None
