"""“Do System One Decisions Add Up?” (arXiv 2609.33971; target T458): TREC fine-grained question type, 50 labels, flat.

Protocol (paper §3-§4, Appendix C-D):
- data: all 500 TREC test questions from the CogComp Parquet conversion (`CogComp/trec`); the text is the state,
  unchanged (including the space before "?");
- flat question (the headline, acc .722 / macro-F1 .486): id `classification`, instructions "Classify the input
  into one of these specific categories.", criteria = the 50 fine labels, identifiers L000.. following the
  lexicographically sorted fine labels, descriptions = "<Expanded parent>: <fine suffix>" (e.g. "Entity: cremat");
  eight fine classes have no test instances but stay candidates;
- coarse question (context, acc .766): "Classify the input into one of these broad categories.", the six parents
  with identifiers over the sorted parents (Appendix D: L005 = Numeric);
- candidate order: NumPy shuffle seeded with the first 16 hex characters of sha256(serialised (42, example id,
  permutation index)); the serialisation is not published, so the order here is our reconstruction (our engine is
  order-invariant by construction, so only the request bytes, not the number, depend on it);
- metrics: flat accuracy (argmax), macro-F1 over ALL 50 declared classes (absent classes 0), multiclass Brier, 15-bin
  top-label ECE (bins include their lower endpoint; the last includes 1.0), stratified bootstrap not reproduced here.
The paper's examples are `trec:test:<row>`; the HF revision is not stated: `prepare` pins and records it.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Sequence

from jev_local.bench.benchmax.adapters_b.common import (AdapterB, Item, accuracy, answers_of, by_task, choice_probs,
                                                        coverage, ece_equal_width, hf_download, hf_list_files,
                                                        hf_resolve_revision, macro_f1, parquet_feature_names,
                                                        parquet_rows, take)

SPEC_ID = "study:do_system_one_decisions_add_up_arxiv_2609_33971"
TARGETS = ("T458",)
UPSTREAM = {"paper": "arXiv 2609.33971", "data": "CogComp/trec", "licence": "paper protocol; data per CogComp/trec card"}
DATASET = "CogComp/trec"
PARQUET_BRANCH = "refs/convert/parquet"
PARENT_FULL = {"ABBR": "Abbreviation", "DESC": "Description", "ENTY": "Entity", "HUM": "Human", "LOC": "Location", "NUM": "Numeric"}
FLAT_INSTRUCTION = "Classify the input into one of these specific categories."
COARSE_INSTRUCTION = "Classify the input into one of these broad categories."
SEED = 42
TASKS = ("flat", "coarse")


def identifiers(labels: Sequence[str]) -> dict[str, str]:
    """label -> Lnnn over the lexicographically sorted labels (Appendix C)."""
    return {lab: f"L{i:03d}" for i, lab in enumerate(sorted(labels))}


def fine_description(fine: str) -> str:
    parent, _, suffix = fine.partition(":")
    return f"{PARENT_FULL[parent]}: {suffix}"


def permutation_seed(example_id: str, perm_index: int = 0, seed: int = SEED) -> int:
    serialized = json.dumps([seed, example_id, perm_index])
    return int(hashlib.sha256(serialized.encode()).hexdigest()[:16], 16)


def shuffled(keys: Sequence[str], example_id: str, perm_index: int = 0) -> list[str]:
    import numpy as np

    arr = list(keys)
    np.random.default_rng(permutation_seed(example_id, perm_index)).shuffle(arr)
    return arr


def build_request(text: str, example_id: str, labels: Sequence[str], descriptions: dict[str, str], coarse: bool) -> dict[str, Any]:
    ids = identifiers(labels)
    order = shuffled([ids[l] for l in labels], example_id)
    by_id = {ids[l]: descriptions[l] for l in labels}
    return {"state": text, "questions": {"classification": {
        "type": "choice", "instructions": COARSE_INSTRUCTION if coarse else FLAT_INSTRUCTION,
        "criteria": {k: by_id[k] for k in order}}}}


class Adapter(AdapterB):
    SPEC_ID = SPEC_ID
    TARGETS = TARGETS
    SPLITS = ("validation", "test")  # validation = train-split rows (self-check)
    TASKS = TASKS

    def _rev(self) -> str:
        """The HF Parquet conversion (`refs/convert/parquet`) of CogComp/trec, pinned to its commit sha at first use
        (the paper used the Parquet conversion; the repo's own loading script is no longer runnable)."""
        p = self.raw_dir("trec") / "revision.json"
        if p.exists():
            cached = json.loads(p.read_text())
            if cached.get("branch") == PARQUET_BRANCH:
                return cached["revision"]
        rev = hf_resolve_revision(DATASET, revision=PARQUET_BRANCH)
        p.write_text(json.dumps({"revision": rev, "branch": PARQUET_BRANCH}))
        return rev

    def _rows(self, split: str, limit: int | None):
        hf_split = "test" if split == "test" else "train"
        rev = self._rev()
        files = [f for f in hf_list_files(DATASET, rev) if re.fullmatch(rf"default/{hf_split}/.*\.parquet", f)]
        if not files:
            raise FileNotFoundError(f"{DATASET}@{rev[:8]}: no default/{hf_split}/*.parquet shard")
        paths = [hf_download(DATASET, f, rev, work=self.work) for f in files]
        fine_names = parquet_feature_names(paths[0], "fine_label")
        coarse_names = parquet_feature_names(paths[0], "coarse_label")
        if not fine_names or not coarse_names or len(fine_names) != 50 or len(coarse_names) != 6:
            raise ValueError(f"trec: ClassLabel names missing or wrong size ({fine_names and len(fine_names)} fine / "
                             f"{coarse_names and len(coarse_names)} coarse)")
        recs = []
        for p in paths:
            recs.extend(parquet_rows(p, ["text", "fine_label", "coarse_label"]))
        if split == "test" and limit is None and len(recs) != 500:
            raise ValueError(f"trec: {len(recs)} test rows, expected 500")
        n = len(recs) if limit is None else min(limit, len(recs))
        rows = [(i, recs[i]["text"], fine_names[recs[i]["fine_label"]], coarse_names[recs[i]["coarse_label"]]) for i in range(n)]
        return rows, fine_names, coarse_names

    def prepare(self, split: str) -> dict[str, Any]:
        return {"spec": SPEC_ID, "upstream": UPSTREAM, "dataset": DATASET, "revision": self._rev(), "seed": SEED,
                "note": "candidate-order serialisation reconstructed (json list [42, id, perm]); order-invariant engine"}

    def expected_counts(self, split: str) -> dict[str, int]:
        return {"flat": 500, "coarse": 500} if split == "test" else {}

    def items(self, split: str, limit: int | None = None, tasks: Sequence[str] | None = None) -> list[Item]:
        rows, fine_names, coarse_names = self._rows(split, limit)
        fine_desc = {f: fine_description(f) for f in fine_names}
        coarse_desc = {c: PARENT_FULL[c] for c in coarse_names}
        out = []
        split_name = "test" if split == "test" else "train"
        for task in self._select_tasks(tasks):
            for i, text, fine, coarse in rows:
                eid = f"trec:{split_name}:{i}"
                if task == "flat":
                    req = build_request(text, eid, fine_names, fine_desc, coarse=False)
                    gold = identifiers(fine_names)[fine]
                else:
                    req = build_request(text, eid, coarse_names, coarse_desc, coarse=True)
                    gold = identifiers(coarse_names)[coarse]
                out.append(Item(f"{task}/{eid}", task, req, gold, meta={"fine": fine, "coarse": coarse, "row": i}))
        return out

    def score(self, items: Sequence[Item], answers: dict[str, dict[str, Any]], split: str,
              thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
        res: dict[str, Any] = {"spec": SPEC_ID, "split": split, "tasks": {}}
        for task, its in by_task(items).items():
            all_ids = sorted(its[0].request["questions"]["classification"]["criteria"])
            gold, pred, probs, failed = [], [], [], 0
            for it in its:
                a = answers_of(answers.get(it.id))
                pr = choice_probs(a.get("classification"), all_ids) if a else None
                if pr is None or sum(pr) <= 0:
                    failed += 1
                    continue
                s = sum(pr)
                pr = [p / s for p in pr]
                gold.append(it.gold)
                pred.append(all_ids[max(range(len(all_ids)), key=lambda i: (pr[i], -i))])
                probs.append(pr)
            r: dict[str, Any] = {**coverage(its, answers), "failed": failed, "target": "T458" if task == "flat" else None}
            if gold:
                correct = [p == g for p, g in zip(pred, gold)]
                r.update({"n": len(gold), "accuracy": accuracy(pred, gold), "macro_f1_all_declared": macro_f1(pred, gold, all_ids),
                          "brier": sum(sum((p - (1.0 if all_ids[i] == g else 0.0)) ** 2 for i, p in enumerate(row))
                                       for row, g in zip(probs, gold)) / len(gold),
                          "ece_15": ece_equal_width([max(r_) for r_ in probs], correct, 15)})
            res["tasks"][task] = r
        return res
