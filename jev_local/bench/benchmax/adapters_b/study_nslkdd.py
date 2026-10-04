"""Jev-IDS (arXiv 2610.01079; jev-ids/jev-ids @ 6aa5ac45, MIT; target T492): NSL-KDD 2,000-flow paper split, k = 0.

Protocol (`prompts/nsl-kdd/jev.json`, `jev_ids/{detectors/jev,dataset,metrics}.py`, `docs/protocol.md`):
- flows: `data/nsl-kdd/splits/paper.csv` (committed in the repo, sha256 pinned; 2,000 KDDTest+ flows: 874 normal,
  642 dos, 248 r2l, 213 probe, 23 u2r; 300 novel attacks); features joined by commas in card order;
- request: the vendored template (model, state {instructions, columns, categories}, two questions) plus
  state.flows.under_test = the flow's feature values and, for k > 0, state.examples (not built here: the KDDTrain+
  pool is a Kaggle download; the headline row is k = 0); questions.category.criteria = state.categories;
- verdict: p_attack = is_attack noul >= 0.5; a failed call counts as normal (fail-open) and ranks lowest;
- metrics: attack-class F1, precision, recall, recall on novel / known attacks and per category, PR-AUC
  (average precision) and ROC-AUC of p_attack, error rate; category accuracy from the choice.
Self-check split "validation": the disjoint `internal.csv` split (50 flows).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from jev_local.bench.benchmax.adapters_b.common import (VENDOR, AdapterB, Item, answers_of, average_precision, by_task,
                                                        choice_pick, coverage, noul_p, repos_dir, roc_auc, sha256_file,
                                                        take)

SPEC_ID = "study:jev_ids_arxiv_2610_01079"
TARGETS = ("T492",)
UPSTREAM = {"repo": "jev-ids/jev-ids", "commit": "6aa5ac4570db1d6d9b88f769f7d16c1cd2c3784a", "licence": "MIT",
            "paper": "arXiv 2610.01079"}
SPLIT_FILES = {"test": ("paper.csv", "b035cb330df3e0dd5f732699e27e9ddb250c8f3f36b6c5f168528e05d41bf2e7", 2000),
               "validation": ("internal.csv", "93a2252f4b253e724689bb6210f7b32eb752e2587173ca5f14eeb965fbc86eae", 50),
               "smoke": ("smoke.csv", "d011c531d19d7d7b34545bb777fa8cbba57a1f43cdb476c7b6b7ecbd94a7249c", 5)}
THRESHOLD = 0.5
TASKS = ("k0",)


def template() -> dict[str, Any]:
    return json.loads((VENDOR / "jevids" / "jev.json").read_text(encoding="utf-8"))


def card() -> dict[str, Any]:
    return json.loads((VENDOR / "jevids" / "dataset.json").read_text(encoding="utf-8"))


def flow_from_row(row: dict[str, str], cfg: dict[str, Any]) -> dict[str, Any]:
    if row["category"] not in cfg["categories"]:
        raise ValueError(f"row {row['row_id']}: category {row['category']!r} is not in the card")
    return {"row_id": int(row["row_id"]), "attributes_csv": ",".join(row[n] for n in cfg["features"]),
            "category": row["category"], "is_attack": row["category"] != cfg["benign"], "novel_attack": row["novel_attack"] == "1"}


def request_body(flow: dict[str, Any], examples: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    """`jev_ids.detectors.jev.request_body` without the model field."""
    body = template()
    body["state"]["flows"] = {"under_test": flow["attributes_csv"]}
    if examples:
        body["state"]["examples"] = [{"record": e["attributes_csv"], "category": e["category"]} for e in examples]
    body["questions"]["category"]["criteria"] = body["state"]["categories"]
    return {"state": body["state"], "questions": body["questions"]}


class Adapter(AdapterB):
    SPEC_ID = SPEC_ID
    TARGETS = TARGETS
    SPLITS = ("validation", "test", "smoke")
    TASKS = TASKS

    def _split_path(self, split: str) -> Path:
        name, sha, _ = SPLIT_FILES[split]
        p = repos_dir() / "jevids" / "data" / "nsl-kdd" / "splits" / name
        if not p.exists():
            raise FileNotFoundError(f"missing {p}: clone jev-ids/jev-ids @ 6aa5ac45 under {repos_dir()}")
        if sha256_file(p) != sha:
            raise ValueError(f"{p} does not match the pinned sha256")
        return p

    def _flows(self, split: str) -> list[dict[str, Any]]:
        cfg = card()
        with self._split_path(split).open(newline="", encoding="utf-8") as f:
            flows = [flow_from_row(r, cfg) for r in csv.DictReader(f)]
        if len(flows) != SPLIT_FILES[split][2]:
            raise ValueError(f"{split}: {len(flows)} flows, expected {SPLIT_FILES[split][2]}")
        return flows

    def prepare(self, split: str) -> dict[str, Any]:
        flows = self._flows(split)
        counts: dict[str, int] = {}
        for fl in flows:
            counts[fl["category"]] = counts.get(fl["category"], 0) + 1
        return {"spec": SPEC_ID, "upstream": UPSTREAM, "split_file": SPLIT_FILES[split][0], "n": len(flows), "by_category": counts,
                "novel": sum(fl["novel_attack"] for fl in flows), "template_model_field": template()["model"],
                "note": "the template names jev-1.13.0 as upstream's pin; our runner sends meharsjev-* (W10)"}

    def expected_counts(self, split: str) -> dict[str, int]:
        return {"k0": SPLIT_FILES[split][2]}

    def items(self, split: str, limit: int | None = None, tasks: Sequence[str] | None = None) -> list[Item]:
        out = []
        for fl in take(self._flows(split), limit):
            out.append(Item(f"k0/{fl['row_id']}", "k0", request_body(fl), {"is_attack": fl["is_attack"], "category": fl["category"]},
                            meta={"novel_attack": fl["novel_attack"], "row_id": fl["row_id"]}))
        return out

    def score(self, items: Sequence[Item], answers: dict[str, dict[str, Any]], split: str,
              thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
        res: dict[str, Any] = {"spec": SPEC_ID, "split": split, "threshold": THRESHOLD, "tasks": {}}
        cats = card()["categories"]
        for task, its in by_task(items).items():
            rows = []
            for it in its:
                a = answers_of(answers.get(it.id))
                p = noul_p(a.get("is_attack")) if a else None
                cat = choice_pick(a.get("category"), cats) if a else None
                rows.append({"gold": it.gold["is_attack"], "category": it.gold["category"], "novel": it.meta["novel_attack"],
                             "p": p, "verdict": p is not None and p >= THRESHOLD, "cat_pred": cat, "error": p is None})
            attacks = [r for r in rows if r["gold"]]
            hits = sum(r["verdict"] for r in attacks)
            alerts = sum(r["verdict"] for r in rows)
            r: dict[str, Any] = {**coverage(its, answers), "target": "T492", "n": len(rows),
                                 "f1": 2 * hits / (alerts + len(attacks)) if alerts + len(attacks) else None,
                                 "precision": hits / alerts if alerts else None, "recall": hits / len(attacks) if attacks else None,
                                 "recall_novel": _rate([r for r in attacks if r["novel"]]), "recall_known": _rate([r for r in attacks if not r["novel"]]),
                                 "error_rate": sum(r["error"] for r in rows) / len(rows) if rows else None}
            for c in cats:
                members = [x for x in attacks if x["category"] == c]
                if members:
                    r[f"recall_{c}"] = _rate(members)
                    r[f"recall_novel_{c}"] = _rate([x for x in members if x["novel"]])
            scores = [x["p"] if x["p"] is not None else 0.0 for x in rows]
            labels = [x["gold"] for x in rows]
            if any(labels) and not all(labels):
                r["pr_auc"] = average_precision(scores, labels)
                r["roc_auc"] = roc_auc(scores, labels)
            r["category_accuracy"] = sum(x["cat_pred"] == x["category"] for x in rows) / len(rows) if rows else None
            res["tasks"][task] = r
        return res


def _rate(rows: Sequence[dict[str, Any]]) -> float | None:
    return sum(r["verdict"] for r in rows) / len(rows) if rows else None
