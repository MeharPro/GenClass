"""DMB expanded suites (nibzard/decision-model-benchmark @ eabd88b0; targets T109-T112, context T113-T119).

Reimplemented from the documented protocol (`docs/expanded-benchmarks.md`, `src/dmb/suites/expanded.py`,
`contenders/{jev,render}.py`, `expanded_metrics.py`, `thresholds.py`): the repo has NO licence, so no code is
copied. The one shared string every contender receives, `render.render_jev_instructions()`, is read at runtime
from the pinned checkout (sha256 of `src/dmb/contenders/render.py` recorded below) and never stored here.

Protocol:
- one request per decision, question id `decision`; state = the message text stripped (NLU++: a JSON string of
  {message, question: <ontology description>, instruction}); criteria {option: option} (key == value);
- options: Banking77 `sorted(train categories)` (77); CLINC `sorted(150 train intents)` + the explicit
  out-of-scope option; NLU++ ["no", "yes"], one request per (message, intent of its domain + general intents);
- splits: Banking77 test = all 3,080 PolyAI test rows, validation = 10 unique train texts per intent after removing
  normalised test texts (random.Random(f"{DMB_SEED}:banking77-validation") over labels in sorted order);
  CLINC test = test + oos_test (5,500), validation = val + oos_val minus texts also in test; NLU++ test = folds 18-19
  (302 messages, 13,712 decisions), validation = folds 16-17 (310 / 14,064);
- metrics: accuracy = correct / all requested decisions (failures wrong); CLINC OOS precision/recall + in-scope
  success; NLU++ micro/macro positive-class F1 (absent intents 0; missing answers = missed positives) and
  complete-message accuracy/coverage;
- thresholds (fit on validation, frozen on test): the largest validation coverage whose error rate <= 5% with >= 100
  accepted, tied scores kept together; unit = decision (Banking77, CLINC) or whole message with the minimum intent
  confidence (NLU++). `confidence` is the wire field DMB reads (W1).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Sequence

from jev_local.bench.benchmax.adapters_b.common import (VENDOR, AdapterB, Item, answers_of, by_task, choice_pick,
                                                        confidence_of, coverage, fetch_verified, repos_dir,
                                                        sha256_file, take)

SPEC_ID = "dmb_expanded@eabd88b0"
TARGETS = ("T109", "T110", "T111", "T112")
UPSTREAM = {"repo": "nibzard/decision-model-benchmark", "commit": "eabd88b04706bcc9b7769213d1ea644c09d8a7f8",
            "licence": "none (read-only; reimplemented from the documented protocol)"}
RENDER_PY_SHA256 = "17e31931a1e5160c5068cad21ab700628cf5e2dc937db645e18b04bd78d13d99"
DMB_SEED = 20260918
FAMILIES = ("s6_banking77", "s7_clinc150", "s8_nlupp")
CLINC_OOS_OPTION = "out_of_scope: none of the listed intents applies"
NLUPP_INSTRUCTION = "Does this intent apply to the message? Choose yes or no."
EXPECTED = {"s6_banking77": {"test": 3080, "validation": 770}, "s7_clinc150": {"test": 5500},
            "s8_nlupp": {"test": 13712, "validation": 14064}}
EXPECTED_MESSAGES = {"s8_nlupp": {"test": 302, "validation": 310}}
MAX_ERROR, MIN_ACCEPTED = 0.05, 100

_INSTRUCTIONS: str | None = None


def jev_instructions() -> str:
    """`render.render_jev_instructions()` from the pinned DMB checkout (not stored in this repo)."""
    global _INSTRUCTIONS
    if _INSTRUCTIONS is not None:
        return _INSTRUCTIONS
    env = os.environ.get("BENCHMAX_DMB_INSTRUCTIONS")
    if env:
        _INSTRUCTIONS = env
        return env
    path = repos_dir() / "dmb" / "src" / "dmb" / "contenders" / "render.py"
    if not path.exists():
        raise FileNotFoundError(f"DMB checkout missing: {path} (clone nibzard/decision-model-benchmark @ eabd88b0 "
                                f"under {repos_dir()} or set BENCHMAX_DMB_INSTRUCTIONS)")
    if sha256_file(path) != RENDER_PY_SHA256:
        raise ValueError("DMB render.py differs from the pinned commit; refusing to read instructions")
    ns: dict[str, Any] = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)  # pure stdlib module
    _INSTRUCTIONS = str(ns["render_jev_instructions"]())
    return _INSTRUCTIONS


def set_instructions_for_tests(text: str | None) -> None:
    global _INSTRUCTIONS
    _INSTRUCTIONS = text


def text_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()


def build_request(state: str, options: Sequence[str]) -> dict[str, Any]:
    return {"state": state.strip(), "questions": {"decision": {"type": "choice", "instructions": jev_instructions(),
                                                              "criteria": {o: o for o in options}}}}


# ---------------------------------------------------------------------------------------------- item builders


def banking_items(train: list[dict[str, str]], test: list[dict[str, str]]) -> dict[str, list[dict[str, Any]]]:
    labels = sorted({r["category"] for r in train})
    out = {"test": [{"id": f"s6_banking77_test-{i:05d}", "text": r["text"], "gold": r["category"], "options": labels,
                     "meta": {"source_split": "test", "source_text_sha256": text_hash(r["text"])}}
                    for i, r in enumerate(test)]}
    seen = {text_hash(r["text"]) for r in test}
    groups: dict[str, list[dict[str, str]]] = {lab: [] for lab in labels}
    for r in train:
        d = text_hash(r["text"])
        if d not in seen:
            groups[r["category"]].append(r)
            seen.add(d)
    rng = random.Random(f"{DMB_SEED}:banking77-validation")
    val: list[dict[str, str]] = []
    for lab, group in sorted(groups.items()):
        rng.shuffle(group)
        if len(group) < 10:
            raise ValueError(f"not enough disjoint validation examples for {lab}")
        val.extend(group[:10])
    out["validation"] = [{"id": f"s6_banking77_validation-{i:05d}", "text": r["text"], "gold": r["category"],
                          "options": labels, "meta": {"source_split": "train", "source_text_sha256": text_hash(r["text"])}}
                         for i, r in enumerate(val)]
    return out


def clinc_items(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    labels = sorted({lab for _, lab in data["train"]})
    if "oos" in labels or len(labels) != 150:
        raise ValueError("CLINC training intents must be the 150 in-scope intents")
    options = labels + [CLINC_OOS_OPTION]
    out: dict[str, list[dict[str, Any]]] = {}
    for split, src in (("validation", "val"), ("test", "test")):
        rows = data[src] + data[f"oos_{src}"]
        out[split] = [{"id": f"s7_clinc150_{split}-{i:05d}", "text": text, "gold": options[-1] if lab == "oos" else lab,
                       "options": options, "meta": {"out_of_scope": lab == "oos", "source_split": src,
                                                    "source_text_sha256": text_hash(text)}}
                      for i, (text, lab) in enumerate(rows)]
    test_hashes = {it["meta"]["source_text_sha256"] for it in out["test"]}
    out["validation"] = [it for it in out["validation"] if it["meta"]["source_text_sha256"] not in test_hashes]
    return out


def nlupp_items(ontology: dict[str, Any], folds: dict[tuple[str, int], list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {"validation": [], "test": []}
    for (domain, fold), rows in sorted(folds.items()):
        if fold not in (16, 17, 18, 19):
            raise ValueError("NLU++ evaluation uses folds 16 through 19 only")
        split = "validation" if fold in (16, 17) else "test"
        intents = {name: spec for name, spec in sorted(ontology["intents"].items())
                   if domain in spec["domain"] or "general" in spec["domain"]}
        for index, row in enumerate(rows):
            gold = set(row.get("intents", []))
            if not gold <= intents.keys():
                raise ValueError(f"NLU++ labels missing from {domain} ontology: {gold - intents.keys()}")
            base = f"nlupp-{domain}-{fold}-{index:04d}"
            for label, spec in intents.items():
                state = json.dumps({"message": row["text"], "question": spec["description"], "instruction": NLUPP_INSTRUCTION},
                                   ensure_ascii=False)
                out[split].append({"id": f"s8_nlupp_{split}-{len(out[split]):05d}", "text": state,
                                   "gold": "yes" if label in gold else "no", "options": ["no", "yes"], "group": base,
                                   "meta": {"domain": domain, "source_fold": fold, "intent": label, "group_size": len(intents),
                                            "source_text_sha256": text_hash(row["text"])}})
    test_hashes = {it["meta"]["source_text_sha256"] for it in out["test"]}
    out["validation"] = [it for it in out["validation"] if it["meta"]["source_text_sha256"] not in test_hashes]
    return out


# ---------------------------------------------------------------------------------------------- scoring


def decision_units(items: Sequence[Item], answers: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """One unit per decision, or per complete NLU++ message (min confidence; correct iff every intent correct)."""
    groups: dict[str, list[Item]] = {}
    for it in items:
        groups.setdefault(it.group or it.id, []).append(it)
    units = []
    for key, group in groups.items():
        expected = group[0].meta.get("group_size", 1)
        rows = [_decision(it, answers) for it in group]
        valid = len(group) == expected and all(r["ok"] for r in rows)
        units.append({"valid": valid, "correct": valid and all(r["correct"] for r in rows),
                      "score": min(r["confidence"] for r in rows) if valid else None, "group": key})
    return units


def _decision(it: Item, answers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    options = list(it.request["questions"]["decision"]["criteria"])
    a = answers_of(answers.get(it.id))
    pick = choice_pick(a.get("decision"), options) if a else None
    conf = confidence_of(a.get("decision")) if a else None
    ok = pick is not None and conf is not None and 0.0 <= conf <= 1.0
    return {"ok": ok, "pick": pick, "correct": ok and pick == it.gold, "confidence": conf if ok else None}


def choose_threshold(units: Sequence[dict[str, Any]], max_error: float = MAX_ERROR, min_accepted: int = MIN_ACCEPTED) -> dict[str, Any]:
    if not math.isfinite(max_error) or not 0 <= max_error < 1 or min_accepted < 1:
        raise ValueError("max_error must be in [0, 1); min_accepted must be positive")
    valid = sorted((u for u in units if u["valid"]), key=lambda u: -u["score"])
    best: dict[str, Any] = {"threshold": None, "accepted": 0, "error_rate": None}
    errors = 0
    for i, u in enumerate(valid):
        errors += not u["correct"]
        if i + 1 < len(valid) and valid[i + 1]["score"] == u["score"]:
            continue
        count = i + 1
        if count >= min_accepted and errors / count <= max_error:
            best = {"threshold": u["score"], "accepted": count, "error_rate": errors / count}
    return {**best, "expected": len(units), "coverage": best["accepted"] / len(units) if units else 0.0}


def apply_threshold(units: Sequence[dict[str, Any]], threshold: float | None) -> dict[str, Any]:
    acc = [u for u in units if u["valid"] and threshold is not None and u["score"] >= threshold]
    return {"expected": len(units), "accepted": len(acc), "coverage": len(acc) / len(units) if units else 0.0,
            "error_rate": sum(not u["correct"] for u in acc) / len(acc) if acc else None,
            "correct_all_requested": sum(u["correct"] for u in units) / len(units) if units else None}


def task_metrics(items: Sequence[Item], answers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = {it.id: _decision(it, answers) for it in items}
    family = items[0].task
    expected = len(items)
    res: dict[str, Any] = {"success_all_requested": sum(r["correct"] for r in rows.values()) / expected,
                           "answered": sum(r["ok"] for r in rows.values()), "expected": expected}
    if family == "s7_clinc150":
        predicted_oos = [it for it in items if rows[it.id]["ok"] and rows[it.id]["pick"] == CLINC_OOS_OPTION]
        correct_oos = sum(it.meta["out_of_scope"] for it in predicted_oos)
        expected_oos = sum(it.meta["out_of_scope"] for it in items)
        in_items = [it for it in items if not it.meta["out_of_scope"]]
        res.update({"out_of_scope_precision": correct_oos / len(predicted_oos) if predicted_oos else None,
                    "out_of_scope_recall": correct_oos / expected_oos if expected_oos else None,
                    "in_scope_success": sum(rows[it.id]["correct"] for it in in_items) / len(in_items) if in_items else None})
    if family == "s8_nlupp":
        positives = [it for it in items if rows[it.id]["ok"] and rows[it.id]["pick"] == "yes"]
        tp = sum(1 for it in positives if it.gold == "yes")
        expected_pos = sum(1 for it in items if it.gold == "yes")
        denom = expected_pos + len(positives)
        label_scores = []
        for label in sorted({it.meta["intent"] for it in items}):
            gold_count = sum(1 for it in items if it.meta["intent"] == label and it.gold == "yes")
            guesses = [it for it in positives if it.meta["intent"] == label]
            hits = sum(1 for it in guesses if it.gold == "yes")
            label_scores.append(2 * hits / (gold_count + len(guesses)) if gold_count + len(guesses) else 0.0)
        units = decision_units(items, answers)
        res.update({"micro_intent_f1": 2 * tp / denom if denom else 0.0,
                    "macro_intent_f1": sum(label_scores) / len(label_scores) if label_scores else 0.0,
                    "binary_accuracy": res["success_all_requested"],
                    "complete_message_accuracy": sum(u["correct"] for u in units) / len(units),
                    "complete_message_coverage": sum(u["valid"] for u in units) / len(units),
                    "expected_messages": len(units)})
    return res


class Adapter(AdapterB):
    SPEC_ID = SPEC_ID
    TARGETS = TARGETS
    SPLITS = ("validation", "test")
    TASKS = FAMILIES

    def _sources(self) -> dict[str, dict[str, str]]:
        return json.loads((VENDOR / "dmb" / "expanded_sources.json").read_text(encoding="utf-8"))

    def _path(self, name: str) -> Path:
        src = self._sources()[name]
        return fetch_verified(src["url"], self.raw_dir("dmb") / name, src["sha256"])

    def prepare(self, split: str) -> dict[str, Any]:
        paths = {name: str(self._path(name)) for name in self._sources()}
        counts = {fam: len(self._family(fam)[split]) for fam in FAMILIES}
        return {"spec": SPEC_ID, "upstream": UPSTREAM, "render_py_sha256": RENDER_PY_SHA256, "sources": paths,
                "split": split, "counts": counts, "instructions_sha256": hashlib.sha256(jev_instructions().encode()).hexdigest()}

    def _csv(self, name: str) -> list[dict[str, str]]:
        with self._path(name).open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _family(self, fam: str) -> dict[str, list[dict[str, Any]]]:
        if fam == "s6_banking77":
            return banking_items(self._csv("banking77-train.csv"), self._csv("banking77-test.csv"))
        if fam == "s7_clinc150":
            return clinc_items(json.loads(self._path("clinc150.json").read_text(encoding="utf-8")))
        if fam == "s8_nlupp":
            ontology = json.loads(self._path("nlupp-ontology.json").read_text(encoding="utf-8"))
            folds = {(d, f): json.loads(self._path(f"nlupp-{d}-{f}.json").read_text(encoding="utf-8"))
                     for d in ("banking", "hotels") for f in (16, 17, 18, 19)}
            return nlupp_items(ontology, folds)
        raise KeyError(fam)

    def expected_counts(self, split: str) -> dict[str, int]:
        return {fam: n for fam in FAMILIES if (n := EXPECTED[fam].get(split)) is not None}

    def items(self, split: str, limit: int | None = None, tasks: Sequence[str] | None = None) -> list[Item]:
        out: list[Item] = []
        for fam in self._select_tasks(tasks):
            rows = self._family(fam)[split]
            exp = EXPECTED[fam].get(split)
            if exp is not None and len(rows) != exp:
                raise ValueError(f"{fam}/{split}: {len(rows)} decisions, expected {exp}")
            if fam == "s8_nlupp" and limit is not None:
                # keep whole messages (threshold analysis rejects a cut label group)
                groups: list[str] = []
                for r in rows:
                    if r["group"] not in groups:
                        groups.append(r["group"])
                keep = set(groups[:max(1, limit // 50)])
                rows = [r for r in rows if r["group"] in keep]
            else:
                rows = take(list(rows), limit)
            for r in rows:
                out.append(Item(r["id"], fam, build_request(r["text"], r["options"]), r["gold"], r.get("group"), r["meta"]))
        return out

    def score(self, items: Sequence[Item], answers: dict[str, dict[str, Any]], split: str,
              thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
        res: dict[str, Any] = {"spec": SPEC_ID, "split": split, "tasks": {}}
        targets = {"s6_banking77": ["T109"], "s7_clinc150": ["T110", "T111"], "s8_nlupp": ["T112"]}
        for fam, its in by_task(items).items():
            r = {**coverage(its, answers), **task_metrics(its, answers), "target": targets.get(fam)}
            if thresholds and fam in thresholds:
                th = thresholds[fam].get("threshold")
                r["threshold"] = {"frozen": th, "unit": "message" if fam == "s8_nlupp" else "decision",
                                  **apply_threshold(decision_units(its, answers), th)}
            res["tasks"][fam] = r
        res["families"] = res["tasks"]  # DMB vocabulary (S6/S7/S8 families)
        return res

    def fit(self, items: Sequence[Item], answers: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
        out = {}
        for fam, its in by_task(items).items():
            units = decision_units(its, answers)
            if sum(u["valid"] for u in units) != len(units):
                out[fam] = {"threshold": None, "error": "threshold fitting requires a completed validation cell"}
                continue
            out[fam] = {**choose_threshold(units), "max_validation_error": MAX_ERROR, "min_accepted": MIN_ACCEPTED,
                        "unit": "message" if fam == "s8_nlupp" else "decision",
                        "source_text_hashes_sha256": hashlib.sha256(
                            "".join(sorted({it.meta["source_text_sha256"] for it in its})).encode()).hexdigest()}
        return out or None
