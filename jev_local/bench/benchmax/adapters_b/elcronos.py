"""elcronos/jev-vs-open-decision-models `plain` protocol (targets T243-T247): DAIR emotion (+ `defined`),
TweetTopic single, Twitter financial news topic, DailyDialog utterance emotion.

The repo has NO licence, so this is a reimplementation from PROTOCOL.md / PROTOCOL_ADDENDUM_v2.md and the
`datasets_registry.py` facts (@ a1901bc3). Nothing is copied from `results/`.
- request: state = the raw text string (unmodified, DailyDialog's trailing space kept); one choice question
  (id `emotion` / `topic`) with the dataset's frozen instruction and criteria {label: ""} (`plain`), or the six
  one-line definitions (`defined`, emotion only);
- data: dair-ai/emotion `split` test @ cab853a1 (2,000); cardiffnlp/tweet_topic_single raw file
  dataset/split_temporal/test_2021.single.json @ 87b7a0d1 (1,693; labels underscores -> spaces, `&` kept);
  zeroshot/twitter-financial-news-topic validation @ acbc8af2 (4,117; 20 card labels verbatim); OpenRL/daily_dialog
  test @ 1668faf0 flattened to 7,740 utterances (7 labels, no dialogue context);
- metrics (metrics.py): accuracy, macro-F1 over ALL K classes (absent classes 0), NLL (eps 1e-6), multiclass Brier,
  ECE with 15 equal-width right-closed bins (c == 0 in bin 0), majority-class accuracy; prediction = argmax of the
  renormalised probabilities (fallback: one-hot on `choice`); `pred_mismatch` counts argmax != choice.
Smoke rows (never evaluated upstream): validation / train heads; our split "validation" returns them.
"""

from __future__ import annotations

import json
import math
from typing import Any, Sequence

from jev_local.bench.benchmax.adapters_b.common import (AdapterB, Item, accuracy, answers_of, by_task, choice_probs,
                                                        coverage, ece_equal_width, hf_download, macro_f1, take)

SPEC_ID = "elcronos_plain"
TARGETS = ("T243", "T244", "T245", "T246", "T247")
UPSTREAM = {"repo": "elcronos/jev-vs-open-decision-models", "commit": "a1901bc3d520e73936de8d4326545c0cdcf742fb",
            "licence": "none (reimplemented from PROTOCOL.md; raw predictions never read)"}
NLL_EPS = 1e-6
ECE_BINS = 15

EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOTION_DEFINITIONS = {
    "sadness": "feeling unhappy, down, grief, loss, disappointment or loneliness",
    "joy": "feeling happy, pleased, cheerful, content or excited",
    "love": "feeling affection, warmth, tenderness, caring or romantic attachment",
    "anger": "feeling mad, irritated, annoyed, resentful or hostile",
    "fear": "feeling afraid, scared, anxious, nervous or worried",
    "surprise": "feeling amazed, shocked, startled or caught off guard by something unexpected",
}
TWEET_TOPIC_RAW = ("arts_&_culture", "business_&_entrepreneurs", "pop_culture", "daily_life", "sports_&_gaming",
                   "science_&_technology")
FIN_TOPIC_LABELS = ["Analyst Update", "Fed | Central Banks", "Company | Product News", "Treasuries | Corporate Debt",
                    "Dividend", "Earnings", "Energy | Oil", "Financials", "Currencies", "General News | Opinion",
                    "Gold | Metals | Materials", "IPO", "Legal | Regulation", "M&A | Investments", "Macro", "Markets",
                    "Politics", "Personnel Change", "Stock Commentary", "Stock Movement"]
DAILY_DIALOG_LABELS = ["no emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]

DATASETS: dict[str, dict[str, Any]] = {
    "emotion": {"source": "dair-ai/emotion", "config": "split", "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe",
                "eval_split": "test", "smoke_split": "validation", "labels": EMOTION_LABELS, "qid": "emotion",
                "instruction": "Which single primary emotion is expressed in this text?", "n_eval": 2000,
                "definitions": EMOTION_DEFINITIONS, "targets": {"plain": "T243", "defined": "T244"}},
    "tweet_topic": {"source": "cardiffnlp/tweet_topic_single", "revision": "87b7a0d1c402dbb481db649569c556d9aa27ac05",
                    "eval_file": "dataset/split_temporal/test_2021.single.json",
                    "smoke_file": "dataset/split_temporal/validation_2021.single.json", "eval_split": "test_2021",
                    "labels": [n.replace("_", " ") for n in TWEET_TOPIC_RAW], "qid": "topic",
                    "instruction": "Which single topic does this tweet belong to?", "n_eval": 1693, "targets": {"plain": "T245"}},
    "fin_topic": {"source": "zeroshot/twitter-financial-news-topic", "revision": "acbc8af2a35ccf0916124efcbe9e6cf25f191012",
                  "eval_split": "validation", "smoke_split": "train", "labels": FIN_TOPIC_LABELS, "qid": "topic",
                  "instruction": "Which single topic does this financial news tweet belong to?", "n_eval": 4117,
                  "targets": {"plain": "T246"}},
    "daily_dialog": {"source": "OpenRL/daily_dialog", "revision": "1668faf0c0dc44664f108c489fd0666128db2c48",
                     "eval_split": "test", "smoke_split": "validation", "labels": DAILY_DIALOG_LABELS, "qid": "emotion",
                     "instruction": "Which single emotion is expressed in this utterance?", "n_eval": 7740,
                     "targets": {"plain": "T247"}},
}
TASKS = ("emotion", "emotion_defined", "tweet_topic", "fin_topic", "daily_dialog")


def build_request(dataset: str, text: str, variant: str = "plain") -> dict[str, Any]:
    spec = DATASETS[dataset]
    if variant == "defined":
        if "definitions" not in spec:
            raise ValueError(f"{dataset} has no definitions (addendum §3: plain only)")
        criteria = {lab: spec["definitions"][lab] for lab in spec["labels"]}
    else:
        criteria = {lab: "" for lab in spec["labels"]}
    return {"state": text, "questions": {spec["qid"]: {"type": "choice", "instructions": spec["instruction"],
                                                       "criteria": criteria}}}


def probs_from_answer(answer: dict[str, Any] | None, labels: Sequence[str]) -> list[float] | None:
    """Renormalised probabilities in label order; one-hot fallback on `choice`; None when unusable."""
    if not answer:
        return None
    probs = choice_probs(answer, labels)
    if probs is None:
        return None
    s = sum(probs)
    if s <= 0:
        c = answer.get("choice")
        if c in labels:
            return [1.0 if lab == c else 0.0 for lab in labels]
        return None
    return [p / s for p in probs]


def metrics(gold: Sequence[int], probs: Sequence[Sequence[float]], k: int) -> dict[str, Any]:
    pred = [max(range(k), key=lambda i: (row[i], -i)) for row in probs]  # numpy.argmax: first index on ties
    conf = [max(row) for row in probs]
    correct = [p == g for p, g in zip(pred, gold)]
    maj = max(range(k), key=lambda c: (sum(1 for g in gold if g == c), -c))
    return {"n": len(gold), "accuracy": accuracy(pred, gold), "macro_f1": macro_f1(pred, gold, list(range(k))),
            "majority_class_accuracy": sum(1 for g in gold if g == maj) / len(gold),
            "nll": -sum(math.log(min(1.0, max(NLL_EPS, row[g]))) for row, g in zip(probs, gold)) / len(gold),
            "brier": sum(sum((p - (1.0 if i == g else 0.0)) ** 2 for i, p in enumerate(row)) for row, g in zip(probs, gold)) / len(gold),
            "ece_15": ece_equal_width(conf, correct, ECE_BINS, right_closed=True),
            "mean_confidence": sum(conf) / len(conf), "frac_gold_prob_zero": sum(1 for row, g in zip(probs, gold) if row[g] == 0.0) / len(gold)}


class Adapter(AdapterB):
    SPEC_ID = SPEC_ID
    TARGETS = TARGETS
    SPLITS = ("validation", "test")
    TASKS = TASKS

    # ------------------------------------------------------------------ data (VM: `datasets`)
    def _rows(self, dataset: str, split: str, limit: int | None) -> list[tuple[str, int, dict[str, Any]]]:
        """(text, gold_id, extras) in file order for the evaluated split, or the smoke split for `validation`."""
        spec = DATASETS[dataset]
        if dataset == "tweet_topic":
            fname = spec["eval_file"] if split == "test" else spec["smoke_file"]
            p = hf_download(spec["source"], fname, spec["revision"], work=self.work)
            out = []
            with p.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("label_name") != TWEET_TOPIC_RAW[int(rec["label"])]:
                        raise RuntimeError(f"tweet_topic: label id {rec['label']} carries name {rec.get('label_name')!r}")
                    out.append((rec["text"], int(rec["label"]), {"id": int(rec["id"]), "date": str(rec["date"])}))
                    if limit is not None and len(out) >= limit:
                        break
            return out
        from datasets import load_dataset  # VM only

        hf_split = spec["eval_split"] if split == "test" else spec["smoke_split"]
        ds = load_dataset(spec["source"], spec.get("config"), split=hf_split, revision=spec["revision"],
                          cache_dir=str(self.work / "hf" / "datasets"))
        if dataset == "daily_dialog":
            names = list(ds.features["emotion"].feature.names)
            if names != DAILY_DIALOG_LABELS:
                raise RuntimeError(f"daily_dialog: ClassLabel names {names} != {DAILY_DIALOG_LABELS}")
            out = []
            for dialog_id, (utts, emos) in enumerate(zip(ds["dialog"], ds["emotion"])):
                if len(utts) != len(emos):
                    raise RuntimeError(f"daily_dialog: dialog {dialog_id} utterance/label mismatch")
                for turn_id, (u, e) in enumerate(zip(utts, emos)):
                    out.append((str(u), int(e), {"dialog_id": dialog_id, "turn_id": turn_id}))
                if limit is not None and len(out) >= limit:
                    break
            return out[:limit] if limit is not None else out
        n = len(ds) if limit is None else min(limit, len(ds))
        rows = ds.select(range(n))
        names = ds.features["label"].names if hasattr(ds.features["label"], "names") else None
        if names is not None and dataset == "emotion" and list(names) != EMOTION_LABELS:
            raise RuntimeError(f"emotion: ClassLabel names {list(names)} != {EMOTION_LABELS}")
        return [(str(t), int(y), {}) for t, y in zip(rows["text"], rows["label"])]

    def prepare(self, split: str) -> dict[str, Any]:
        return {"spec": SPEC_ID, "upstream": UPSTREAM, "datasets": {k: {kk: v for kk, v in s.items() if kk != "definitions"}
                                                                   for k, s in DATASETS.items()}}

    def expected_counts(self, split: str) -> dict[str, int]:
        if split != "test":
            return {}
        return {"emotion": 2000, "emotion_defined": 2000, "tweet_topic": 1693, "fin_topic": 4117, "daily_dialog": 7740}

    def items(self, split: str, limit: int | None = None, tasks: Sequence[str] | None = None) -> list[Item]:
        out: list[Item] = []
        cache: dict[str, list] = {}
        for task in self._select_tasks(tasks):
            dataset, variant = (task[:-8], "defined") if task.endswith("_defined") else (task, "plain")
            if dataset not in cache:
                cache[dataset] = self._rows(dataset, split, limit)
                exp = DATASETS[dataset]["n_eval"] if split == "test" else None
                if exp is not None and limit is None and len(cache[dataset]) != exp:
                    raise ValueError(f"{dataset}: evaluated split has {len(cache[dataset])} rows, expected {exp}")
            for i, (text, gold, extra) in enumerate(cache[dataset]):
                out.append(Item(f"{task}/{i}", task, build_request(dataset, text, variant), gold,
                                meta={"dataset_index": i, "variant": variant, **extra}))
        return out

    def score(self, items: Sequence[Item], answers: dict[str, dict[str, Any]], split: str,
              thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
        res: dict[str, Any] = {"spec": SPEC_ID, "split": split, "tasks": {}}
        for task, its in by_task(items).items():
            dataset = task[:-8] if task.endswith("_defined") else task
            spec = DATASETS[dataset]
            labels = spec["labels"]
            gold, probs, failed, mismatch = [], [], 0, 0
            for it in its:
                a = answers_of(answers.get(it.id))
                pr = probs_from_answer(a.get(spec["qid"]), labels) if a else None
                if pr is None:
                    failed += 1
                    continue
                gold.append(it.gold)
                probs.append(pr)
                pick = a[spec["qid"]].get("choice")
                mismatch += pick != labels[max(range(len(labels)), key=lambda i: (pr[i], -i))]
            r: dict[str, Any] = {**coverage(its, answers), "failed": failed, "pred_mismatch": mismatch,
                                 "target": spec["targets"].get("defined" if task.endswith("_defined") else "plain")}
            if gold:
                r.update(metrics(gold, probs, len(labels)))
                if failed:  # missing = wrong variant
                    r["accuracy_missing_as_wrong"] = r["accuracy"] * len(gold) / len(its)
            res["tasks"][task] = r
        return res
