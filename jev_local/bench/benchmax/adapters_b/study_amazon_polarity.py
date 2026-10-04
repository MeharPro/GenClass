"""Koa-action (arXiv 2609.36115; targets T511 SST-2 .961, T512 Amazon Reviews Polarity .968): binary sentiment, Jev row.

What the paper fixes (§4.4.2, Table 2, Appendix G): SST-2 = the 872 labelled examples (GLUE validation); Amazon
Reviews Polarity = "5k examples of the official test split"; every row zero-shot; labels "Negative"/"Positive";
the generative prompt reads "Based on the overall sentiment expressed in this review, respond with the relevant
control token" and shows the review body only (no title).
What it does NOT publish: the Jev request (state/criteria wording) and which 5,000 Amazon rows. This adapter is
therefore INDICATIVE for both rows: the first 5,000 rows of `fancyzhx/amazon_polarity` test (review `content`), one
Choice `sentiment` whose instruction mirrors the paper's prompt, criteria {Negative, Positive} without
descriptions; SST-2 from `stanfordnlp/sst2` validation (872). HF revisions are resolved and recorded at `prepare`.
Metrics: accuracy (argmax = `choice`), macro-F1, 10-bin ECE. Self-check split "validation": the train splits' heads.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from jev_local.bench.benchmax.adapters_b.common import (AdapterB, Item, accuracy, answers_of, by_task, choice_pick,
                                                        choice_probs, coverage, ece_equal_width, hf_resolve_revision,
                                                        macro_f1, take)

SPEC_ID = "study:koa_action_arxiv_2609_36115"
TARGETS = ("T511", "T512")
UPSTREAM = {"paper": "arXiv 2609.36115 (Koa-action)", "licence": "paper protocol only; Jev request wording not published",
            "indicative": True}
LABELS = ("Negative", "Positive")
INSTRUCTION = "Based on the overall sentiment expressed in this review, which label applies?"
SOURCES = {
    "sst2": {"dataset": "stanfordnlp/sst2", "split": "validation", "n": 872, "text": "sentence", "smoke_split": "train", "target": "T511"},
    "amazon_polarity": {"dataset": "fancyzhx/amazon_polarity", "split": "test", "n": 5000, "text": "content", "smoke_split": "train",
                        "target": "T512"},
}
TASKS = tuple(SOURCES)


def build_request(text: str) -> dict[str, Any]:
    return {"state": text, "questions": {"sentiment": {"type": "choice", "instructions": INSTRUCTION,
                                                       "criteria": {lab: None for lab in LABELS}}}}


class Adapter(AdapterB):
    SPEC_ID = SPEC_ID
    TARGETS = TARGETS
    SPLITS = ("validation", "test")
    TASKS = TASKS

    def _revisions(self) -> dict[str, str]:
        p = self.raw_dir("koa") / "revisions.json"
        revs = json.loads(p.read_text()) if p.exists() else {}
        for s in SOURCES.values():
            if s["dataset"] not in revs:
                revs[s["dataset"]] = hf_resolve_revision(s["dataset"])
        p.write_text(json.dumps(revs, indent=1))
        return revs

    def _rows(self, task: str, split: str, limit: int | None) -> list[tuple[int, str, int]]:
        from datasets import load_dataset  # VM only

        s = SOURCES[task]
        hf_split = s["split"] if split == "test" else s["smoke_split"]
        ds = load_dataset(s["dataset"], split=hf_split, revision=self._revisions()[s["dataset"]],
                          cache_dir=str(self.work / "hf" / "datasets"))
        n = s["n"] if split == "test" else (limit or 200)
        if limit is not None:
            n = min(n, limit)
        if split == "test" and task == "sst2" and len(ds) != 872:
            raise ValueError(f"sst2 validation has {len(ds)} rows, expected 872")
        rows = ds.select(range(min(n, len(ds))))
        return [(i, str(t), int(y)) for i, (t, y) in enumerate(zip(rows[s["text"]], rows["label"]))]

    def prepare(self, split: str) -> dict[str, Any]:
        return {"spec": SPEC_ID, "upstream": UPSTREAM, "revisions": self._revisions(), "sources": SOURCES,
                "note": "indicative: Jev request wording and the Amazon 5k sample are not published; first 5,000 test rows used"}

    def expected_counts(self, split: str) -> dict[str, int]:
        return {t: s["n"] for t, s in SOURCES.items()} if split == "test" else {}

    def items(self, split: str, limit: int | None = None, tasks: Sequence[str] | None = None) -> list[Item]:
        out = []
        for task in self._select_tasks(tasks):
            for i, text, y in self._rows(task, split, limit):
                out.append(Item(f"{task}/{i}", task, build_request(text), LABELS[y], meta={"row": i}))
        return out

    def score(self, items: Sequence[Item], answers: dict[str, dict[str, Any]], split: str,
              thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
        res: dict[str, Any] = {"spec": SPEC_ID, "split": split, "indicative": True, "tasks": {}}
        for task, its in by_task(items).items():
            gold, pred, conf, failed = [], [], [], 0
            for it in its:
                a = answers_of(answers.get(it.id))
                pick = choice_pick(a.get("sentiment"), LABELS) if a else None
                pr = choice_probs(a.get("sentiment"), LABELS) if a else None
                if pick is None:
                    failed += 1
                    continue
                gold.append(it.gold)
                pred.append(pick)
                conf.append(max(pr) / sum(pr) if pr and sum(pr) > 0 else 1.0)
            r: dict[str, Any] = {**coverage(its, answers), "failed": failed, "target": SOURCES[task]["target"]}
            if gold:
                correct = [p == g for p, g in zip(pred, gold)]
                r.update({"n": len(gold), "accuracy": accuracy(pred, gold), "macro_f1": macro_f1(pred, gold, list(LABELS)),
                          "ece_10": ece_equal_width(conf, correct, 10)})
                if failed:
                    r["accuracy_missing_as_wrong"] = sum(correct) / len(its)
            res["tasks"][task] = r
        return res
