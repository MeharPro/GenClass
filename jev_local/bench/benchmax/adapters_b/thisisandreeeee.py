"""thisisandreeeee/jev-benchmarks (targets T254-T258): BANKING77, CLINC150 in-scope, HWU64, SST-2 (Noul), STS-B (Score).

Protocol (@ eaed9dd0, MIT; `benchmarks/{space2_intents,sst2,stsb}.py`, `providers/typesafe.py`):
- intents: state = the utterance; one Choice `answer` with the frozen instruction and criteria {label: None} for the
  full label set in the benchmark's canonical order (BANKING77: the dataset's ClassLabel names; CLINC150 / HWU64:
  the SPACE-2 release label mapping, vendored); CLINC150 drops the OOS rows (4,500); accuracy = `choice` == gold,
  plus mean confidence (max probability), NLL (floor 1e-15) and 10-bin ECE;
- SST-2: state = sentence; Noul "Does this movie-review sentence express positive sentiment?" with no criteria;
  hit = (p >= 0.5) == label; same calibration metrics;
- STS-B: state "Sentence 1: ...\\nSentence 2: ..."; Score with the 6-level rubric; Spearman (tie-aware) and Pearson of
  `score` against the 0-5 gold;
- datasets (revisions pinned upstream; row digests asserted): DeepPavlov/hwu64 @ 0dd289cc test 1,076;
  PolyAI/banking77 @ 1fb62b1b test 3,080; clinc/clinc_oos plus @ 155b9c71 test; nyu-mll/glue sst2 @ bcdcba79
  validation 872; mteb/stsbenchmark-sts @ 96943a16 test 1,379.
Self-check split "validation": the train (HWU64, BANKING77, SST-2) / validation (CLINC150, STS-B) rows.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Sequence

from jev_local.bench.benchmax.adapters_b.common import (VENDOR, AdapterB, Item, answers_of, by_task, choice_pick,
                                                        choice_probs, coverage, ece_equal_width, noul_p, pearson,
                                                        spearman, take)

SPEC_ID = "thisisandreeeee"
TARGETS = ("T254", "T255", "T256", "T257", "T258")
UPSTREAM = {"repo": "thisisandreeeee/jev-benchmarks", "commit": "eaed9dd0cfd6cfa085424e11f79cab8930847e16", "licence": "MIT"}
STSB_INSTRUCTION = "Rate the semantic similarity of the two sentences using the rubric."
STSB_RUBRIC = ("The sentences are completely dissimilar.",
               "The sentences are on the same topic but are not equivalent.",
               "The sentences share some details but are not equivalent.",
               "The sentences are roughly equivalent, but important information differs.",
               "The sentences are mostly equivalent, with only minor differences.",
               "The sentences are completely equivalent in meaning.")
SST2_INSTRUCTION = "Does this movie-review sentence express positive sentiment?"
BENCH: dict[str, dict[str, Any]] = {
    "banking77": {"dataset": "PolyAI/banking77", "config": None, "revision": "1fb62b1bb4635df59a8e1b2f2bc5e0643b2856c8",
                  "split": "test", "rows": 3080, "labels": 77, "smoke_split": "train",
                  "digest": "e331bc3fa83d880409f6044836f709ae70de4193a4186f297385bbd6e5527808",
                  "instruction": "Classify this banking customer request by choosing the most appropriate BANKING77 intent label.",
                  "target": "T254"},
    "clinc150": {"dataset": "clinc/clinc_oos", "config": "plus", "revision": "155b9c710419136e17307b80d0a13e68cd46b4ec",
                 "split": "test", "rows": 4500, "labels": 150, "smoke_split": "validation",
                 "digest": "b5f47620c581ba0b5934b422f6e14a7cdd32ae8aa8020e78d4ee67ca85e0ce6b",
                 "instruction": "Classify this request by choosing the most appropriate CLINC150 intent label.", "target": "T255"},
    "hwu64": {"dataset": "DeepPavlov/hwu64", "config": "default", "revision": "0dd289ccdeb185ec065d1ebcf5de1c443cd1620f",
              "split": "test", "rows": 1076, "labels": 64, "smoke_split": "train",
              "digest": "a18555d600b1a85d49013ca4bd12171c21361fa4756aaf4040785848c9d345ec",
              "instruction": "Classify this request by choosing the most appropriate HWU64 intent label.", "target": "T256"},
    "sst2": {"dataset": "nyu-mll/glue", "config": "sst2", "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
             "split": "validation", "rows": 872, "smoke_split": "train",
             "digest": "b85e9210b8507d7c79e2b9d5220d4c4152e6f9c19349504318523acc74ec8c3e", "target": "T257"},
    "stsb": {"dataset": "mteb/stsbenchmark-sts", "config": "default", "revision": "96943a16ea6a35129e253c659081cb59daf81b30",
             "split": "test", "rows": 1379, "smoke_split": "validation",
             "digest": "3e30964a29dd599b3c30fe108303e6858340a9fd82a4ac2a1da33757411c4060", "target": "T258"},
}
TASKS = tuple(BENCH)


def space2_labels(name: str) -> list[str]:
    return list(json.loads((VENDOR / "thisisandreeeee" / f"space2-{name}.labels.json").read_text(encoding="utf-8"))["label_mapping"])


def intent_request(bench: str, text: str, labels: Sequence[str]) -> dict[str, Any]:
    return {"state": text, "questions": {"answer": {"type": "choice", "instructions": BENCH[bench]["instruction"],
                                                    "criteria": {lab: None for lab in labels}}}}


def sst2_request(sentence: str) -> dict[str, Any]:
    return {"state": sentence, "questions": {"answer": {"type": "noul", "instructions": SST2_INSTRUCTION}}}


def stsb_request(s1: str, s2: str) -> dict[str, Any]:
    return {"state": f"Sentence 1: {s1}\nSentence 2: {s2}",
            "questions": {"answer": {"type": "score", "instructions": STSB_INSTRUCTION, "criteria": list(STSB_RUBRIC)}}}


def row_digest_intents(rows: Sequence[dict[str, str]]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps([r["text"], r["label"]], ensure_ascii=False, separators=(",", ":")).encode())
        h.update(b"\n")
    return h.hexdigest()


def row_digest_sst2(rows: Sequence[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps([r["sentence"], r["label"]], separators=(",", ":")).encode())
        h.update(b"\n")
    return h.hexdigest()


def row_digest_stsb(rows: Sequence[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps([r["sentence1"], r["sentence2"], f"{float(r['score']):.10f}"], ensure_ascii=False,
                            separators=(",", ":")).encode())
        h.update(b"\n")
    return h.hexdigest()


def classification_metrics(records: Sequence[tuple[str, list[float], str, Sequence[str]]]) -> dict[str, Any]:
    """thisisandreeeee `metrics.classification_metrics` on (expected, probs in label order, choice, labels)."""
    if not records:
        return {"accuracy": None}
    correct, confs, losses = 0, [], []
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(10)]
    for expected, probs, choice, labels in records:
        conf = max(probs)
        hit = choice == expected
        correct += hit
        confs.append(conf)
        losses.append(-math.log(max(probs[list(labels).index(expected)], 1e-15)))
        bins[min(int(conf * 10), 9)].append((conf, hit))
    ece = sum(len(b) / len(records) * abs(sum(c for c, _ in b) / len(b) - sum(h for _, h in b) / len(b)) for b in bins if b)
    return {"accuracy": correct / len(records), "mean_confidence": sum(confs) / len(records),
            "negative_log_loss": sum(losses) / len(records), "expected_calibration_error": ece, "n": len(records)}


def noul_metrics(records: Sequence[tuple[bool, float]]) -> dict[str, Any]:
    if not records:
        return {"accuracy": None}
    correct, confs, losses = 0, [], []
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(10)]
    for expected, p in records:
        conf = max(p, 1 - p)
        hit = (p >= 0.5) == expected
        correct += hit
        confs.append(conf)
        losses.append(-math.log(max(p if expected else 1 - p, 1e-15)))
        bins[min(int(conf * 10), 9)].append((conf, hit))
    ece = sum(len(b) / len(records) * abs(sum(c for c, _ in b) / len(b) - sum(h for _, h in b) / len(b)) for b in bins if b)
    return {"accuracy": correct / len(records), "mean_confidence": sum(confs) / len(records),
            "negative_log_loss": sum(losses) / len(records), "expected_calibration_error": ece, "n": len(records)}


class Adapter(AdapterB):
    SPEC_ID = SPEC_ID
    TARGETS = TARGETS
    SPLITS = ("validation", "test")
    TASKS = TASKS

    def _load(self, bench: str, split: str):
        from datasets import load_dataset  # VM only

        b = BENCH[bench]
        hf_split = b["split"] if split == "test" else b["smoke_split"]
        return load_dataset(b["dataset"], b["config"], revision=b["revision"], split=hf_split,
                            cache_dir=str(self.work / "hf" / "datasets"))

    def _intent_rows(self, bench: str, split: str) -> tuple[list[dict[str, str]], list[str]]:
        ds = self._load(bench, split)
        if bench == "banking77":
            labels = list(ds.features["label"].names)
            rows = [{"text": r["text"], "label": labels[r["label"]]} for r in ds]
        elif bench == "clinc150":
            canonical = list(ds.features["intent"].names)
            labels = space2_labels("clinc150")
            rows = [{"text": r["text"], "label": canonical[r["intent"]]} for r in ds if canonical[r["intent"]] != "oos"]
        else:
            labels = space2_labels("hwu64")
            rows = [{"text": r["utterance"], "label": labels[r["label"]]} for r in ds]
        if split == "test":
            if len(rows) != BENCH[bench]["rows"]:
                raise ValueError(f"{bench}: {len(rows)} rows, expected {BENCH[bench]['rows']}")
            if row_digest_intents(rows) != BENCH[bench]["digest"]:
                raise ValueError(f"{bench}: test row digest mismatch")
        if len(set(labels)) != BENCH[bench]["labels"] or any(r["label"] not in labels for r in rows):
            raise ValueError(f"{bench}: label set mismatch")
        return rows, labels

    def prepare(self, split: str) -> dict[str, Any]:
        return {"spec": SPEC_ID, "upstream": UPSTREAM,
                "benchmarks": {k: {kk: v for kk, v in b.items() if kk != "instruction"} for k, b in BENCH.items()}}

    def expected_counts(self, split: str) -> dict[str, int]:
        return {k: b["rows"] for k, b in BENCH.items()} if split == "test" else {}

    def items(self, split: str, limit: int | None = None, tasks: Sequence[str] | None = None) -> list[Item]:
        out: list[Item] = []
        for bench in self._select_tasks(tasks):
            if bench in ("banking77", "clinc150", "hwu64"):
                rows, labels = self._intent_rows(bench, split)
                for i, r in enumerate(take(rows, limit)):
                    out.append(Item(f"{bench}/{i}", bench, intent_request(bench, r["text"], labels), r["label"],
                                    meta={"dataset_id": i, "labels_sha256": hashlib.sha256("\n".join(labels).encode()).hexdigest()}))
            elif bench == "sst2":
                ds = self._load(bench, split)
                rows = [{"sentence": r["sentence"], "label": int(r["label"])} for r in ds]
                if split == "test" and (len(rows) != 872 or row_digest_sst2(rows) != BENCH[bench]["digest"]):
                    raise ValueError("sst2: validation row digest mismatch")
                for i, r in enumerate(take(rows, limit)):
                    out.append(Item(f"sst2/{i}", bench, sst2_request(r["sentence"]), r["label"], meta={"dataset_id": i}))
            else:
                ds = self._load(bench, split)
                rows = [{"sentence1": r["sentence1"], "sentence2": r["sentence2"], "score": float(r["score"])} for r in ds]
                if split == "test" and (len(rows) != 1379 or row_digest_stsb(rows) != BENCH[bench]["digest"]):
                    raise ValueError("stsb: test row digest mismatch")
                for i, r in enumerate(take(rows, limit)):
                    out.append(Item(f"stsb/{i}", bench, stsb_request(r["sentence1"], r["sentence2"]), r["score"],
                                    meta={"dataset_id": i}))
        return out

    def score(self, items: Sequence[Item], answers: dict[str, dict[str, Any]], split: str,
              thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
        res: dict[str, Any] = {"spec": SPEC_ID, "split": split, "tasks": {}}
        for bench, its in by_task(items).items():
            failed = 0
            r: dict[str, Any] = {**coverage(its, answers), "target": BENCH[bench]["target"]}
            if bench in ("banking77", "clinc150", "hwu64"):
                recs = []
                for it in its:
                    labels = list(it.request["questions"]["answer"]["criteria"])
                    a = answers_of(answers.get(it.id))
                    probs = choice_probs(a.get("answer"), labels) if a else None
                    pick = choice_pick(a.get("answer"), labels) if a else None
                    if probs is None or pick is None or sum(probs) <= 0:
                        failed += 1
                        continue
                    s = sum(probs)
                    recs.append((it.gold, [p / s for p in probs], pick, labels))
                r.update(classification_metrics(recs))
            elif bench == "sst2":
                recs = []
                for it in its:
                    a = answers_of(answers.get(it.id))
                    p = noul_p(a.get("answer")) if a else None
                    if p is None:
                        failed += 1
                        continue
                    recs.append((bool(it.gold), p))
                r.update(noul_metrics(recs))
            else:
                exp, pred = [], []
                for it in its:
                    a = answers_of(answers.get(it.id))
                    sc = a.get("answer", {}).get("score") if a else None
                    if not isinstance(sc, (int, float)) or not math.isfinite(sc) or not 0 <= sc <= 5:
                        failed += 1
                        continue
                    exp.append(float(it.gold))
                    pred.append(float(sc))
                r.update({"n": len(exp), "pearson": pearson(exp, pred) if len(exp) > 1 else None,
                          "spearman": spearman(exp, pred) if len(exp) > 1 else None})
            r["failed"] = failed
            if failed and r.get("accuracy") is not None:
                r["accuracy_missing_as_wrong"] = r["accuracy"] * r["n"] / len(its)
            res["tasks"][bench] = r
        return res
