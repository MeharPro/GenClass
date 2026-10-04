"""zhuyansen/jev-zeroshot-vs-bert (targets T248-T253): 20 texts per call, label descriptions, seed-0 stratified samples.

Protocol (@ edbf0713, MIT; `src/jzb/{data,labels,players/jev,metrics}.py`, `labels/*.yaml` vendored):
- eval sets: a stratified 1,000-item sample (seed 0, `stratified_sample`: numpy permutation, proportional quotas,
  at least one per class) of each labelled test split; SST-2 = the whole 872-row validation split; arXiv 2026-09 =
  the author's own arXiv API crawl (NOT published: our crawl of the same window is INDICATIVE, flagged);
- request: BATCH = 20 texts per call; state = "Texts: <blurb>. Judge each one on its own.\\n\\nt1: <text[:600]>\\n..."
  (whitespace collapsed); question `t{i}` = choice with instructions "<question> (text t{i})" and criteria
  {key_of(name): description} (variant `desc`, the headline) or the bare names (`bare`);
- Banking77: two steps: a group choice over the 11 groups, then a label choice inside the chosen group; the label
  probabilities are the within-group probabilities times the group probability (stage 2 depends on stage 1);
- metrics: accuracy (argmax), macro-F1 over the observed gold classes, 15-bin ECE (floor bins), ROC-AUC for PAWS.
HF revisions are not pinned upstream; `prepare` resolves and records them.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from jev_local.bench.benchmax.adapters_b.common import (VENDOR, AdapterB, Item, accuracy, answers_of, by_task,
                                                        choice_probs, coverage, ece_equal_width, hf_resolve_revision,
                                                        macro_f1, roc_auc, take)

SPEC_ID = "zhuyansen_batch20"
TARGETS = ("T248", "T249", "T250", "T251", "T252", "T253")
UPSTREAM = {"repo": "zhuyansen/jev-zeroshot-vs-bert", "commit": "edbf0713583644bd3f47299fd6b104a8b2073219", "licence": "MIT"}
BATCH, TEXT_CHARS, EVAL_N, EVAL_SEED = 20, 600, 1000, 0
HF_SETS = {  # name -> (hf_id, config, train, eval, text, label, text_b)
    "agnews": ("fancyzhx/ag_news", None, "train", "test", "text", "label", None),
    "sst2": ("stanfordnlp/sst2", None, "train", "validation", "sentence", "label", None),
    "banking77": ("mteb/banking77", None, "train", "test", "text", "label", None),
    "tweet_emotion": ("cardiffnlp/tweet_eval", "emotion", "train", "test", "text", "label", None),
    "paws": ("google-research-datasets/paws", "labeled_final", "train", "test", "sentence1", "label", "sentence2"),
}
ARXIV_CATS = ["cs.CL", "cs.CV", "cs.LG", "cs.CR", "math.PR", "q-bio.NC", "astro-ph.GA", "econ.GN"]
ARXIV_SETS = {"arxiv2026": ("202609170000", "202612312359"), "arxiv2020": ("202003010000", "202006302359")}
ARXIV_PER_CAT = 125
_WS = re.compile(r"\s+")
TASKS = ("agnews", "sst2", "banking77", "tweet_emotion", "paws", "arxiv2026")
TARGET_OF = {"agnews": "T248", "sst2": "T249", "banking77": "T250", "tweet_emotion": "T251", "paws": "T252", "arxiv2026": "T253"}


def load_labels(ds: str) -> dict[str, Any]:
    raw = json.loads((VENDOR / "zhuyansen" / f"{ds}.json").read_text(encoding="utf-8"))
    return {"question": raw["question"], "blurb": raw["blurb"], "names": [x["name"] for x in raw["labels"]],
            "descs": [x["desc"] for x in raw["labels"]], "groups": raw.get("groups") or {}}


def label_texts(labels: dict[str, Any], variant: str) -> list[str]:
    if variant == "desc":
        return list(labels["descs"])
    return [n.replace("_", " ").rstrip("?") for n in labels["names"]]


def key_of(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def criteria(names: Sequence[str], texts: Sequence[str]) -> dict[str, str]:
    return {key_of(n): t for n, t in zip(names, texts)}


def build_request(labels: dict[str, Any], texts: Sequence[str], crits: Sequence[dict[str, str]], question: str) -> dict[str, Any]:
    lines = [f"Texts: {labels['blurb']}. Judge each one on its own.", ""]
    lines += [f"t{i}: {_WS.sub(' ', t)[:TEXT_CHARS]}" for i, t in enumerate(texts, 1)]
    qs = {f"t{i}": {"type": "choice", "instructions": f"{question} (text t{i})", "criteria": c} for i, c in enumerate(crits, 1)}
    return {"state": "\n".join(lines), "questions": qs}


def dist(answer: dict[str, Any] | None, keys: Sequence[str]) -> list[float] | None:
    """jzb `dist`: probabilities over keys, one-hot on `choice` when none returned, uniform when nothing usable."""
    if not answer:
        return None
    p = choice_probs(answer, keys) or [0.0] * len(keys)
    if sum(p) <= 0:
        p = [1.0 if k == answer.get("choice") else 0.0 for k in keys]
    s = sum(p)
    return [x / s for x in p] if s > 0 else [1.0 / len(keys)] * len(keys)


def stratified_sample(ys: Sequence[int], n: int = EVAL_N, seed: int = EVAL_SEED) -> list[int]:
    """jzb `data.stratified_sample` on label ids: returns the selected row indices in shuffled order."""
    import numpy as np

    rng = np.random.default_rng(seed)
    perm = [int(i) for i in rng.permutation(len(ys))]
    counts: dict[int, int] = {}
    for y in ys:
        counts[int(y)] = counts.get(int(y), 0) + 1
    counts = dict(sorted(counts.items()))
    target = max(n, len(counts))
    quota = {c: max(1, int(round(n * k / len(ys)))) for c, k in counts.items()}
    while sum(quota.values()) > target:
        quota[max(quota, key=lambda c: (quota[c], -c))] -= 1
    while sum(quota.values()) < target:
        quota[max(quota, key=lambda c: (counts[c] - quota[c], -c))] += 1
    picked: set[int] = set()
    for c in counts:
        taken = 0
        for i in perm:
            if int(ys[i]) == c and taken < quota[c]:
                picked.add(i)
                taken += 1
    return [i for i in perm if i in picked]


def ece15(probs: Sequence[Sequence[float]], gold: Sequence[int]) -> float:
    conf = [max(r) for r in probs]
    correct = [max(range(len(r)), key=lambda i: (r[i], -i)) == g for r, g in zip(probs, gold)]
    return round(ece_equal_width(conf, correct, 15), 12)


class Adapter(AdapterB):
    SPEC_ID = SPEC_ID
    TARGETS = TARGETS
    SPLITS = ("validation", "test")  # validation = the same sampler on the train split (self-check)
    TASKS = TASKS
    STAGES = 2  # banking77 label choice depends on the group choice

    def _revisions(self) -> dict[str, str]:
        p = self.raw_dir("zhuyansen") / "revisions.json"
        revs = json.loads(p.read_text()) if p.exists() else {}
        for name, (hf_id, *_rest) in HF_SETS.items():
            if hf_id not in revs:
                revs[hf_id] = hf_resolve_revision(hf_id)
        p.write_text(json.dumps(revs, indent=1))
        return revs

    def _eval_frame(self, name: str, split: str) -> list[tuple[str, str, int]]:
        """[(id, text, y)] of the frozen eval sample (or of the train-split sample for `validation`)."""
        cache = self.raw_dir("zhuyansen") / f"{name}.{split}.jsonl"
        if cache.exists():
            return [tuple(json.loads(l)) for l in cache.read_text(encoding="utf-8").split("\n") if l]
        if name.startswith("arxiv"):
            rows = self._arxiv(name)
        else:
            from datasets import load_dataset  # VM only

            hf_id, config, train, ev, text, label, text_b = HF_SETS[name]
            hf_split = ev if split == "test" else train
            ds = load_dataset(hf_id, config, split=hf_split, revision=self._revisions()[hf_id],
                              cache_dir=str(self.work / "hf" / "datasets"))
            ys = [int(y) for y in ds[label]]
            # the yaml label order must be the dataset's label-id order (jzb assumes it; we check it)
            expected = [key_of(n) for n in load_labels(name)["names"]]
            names = getattr(ds.features[label], "names", None)
            if names is not None:
                if [key_of(n) for n in names] != expected and not all(k.isdigit() for k in names):
                    raise RuntimeError(f"{name}: ClassLabel order {names} != labels yaml {expected}")
            elif "label_text" in ds.column_names:
                bad = [(y, t) for y, t in zip(ys, ds["label_text"]) if key_of(expected[y]) != key_of(str(t))]
                if bad:
                    raise RuntimeError(f"{name}: label ids do not match label_text, e.g. {bad[:3]}")
            if text_b:
                texts = [f"A: {a}\nB: {b}" for a, b in zip(ds[text], ds[text_b])]
            else:
                texts = [str(t) for t in ds[text]]
            ids = [f"{name}-{hf_split}-{i}" for i in range(len(ds))]
            if name == "sst2" and split == "test":
                sel = list(range(len(ds)))  # the whole labelled validation split
            else:
                sel = stratified_sample(ys)
            rows = [(ids[i], texts[i], ys[i]) for i in sel]
        cache.write_text("".join(json.dumps(list(r), ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        return rows

    def _arxiv(self, name: str) -> list[tuple[str, str, int]]:
        """jzb `_arxiv`: arXiv API crawl per category (indicative: the author's crawl is not published)."""
        import time
        import urllib.request
        import xml.etree.ElementTree as ET

        lo, hi = ARXIV_SETS[name]
        ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
        rows: list[tuple[str, str, int]] = []
        seen: set[str] = set()
        for y, cat in enumerate(ARXIV_CATS):
            got: list[dict[str, str]] = []
            for start in range(0, 2000, 200):
                q = urllib.parse.urlencode({"search_query": f"cat:{cat} AND submittedDate:[{lo} TO {hi}]", "start": start,
                                            "max_results": 200, "sortBy": "submittedDate", "sortOrder": "descending"})
                with urllib.request.urlopen(f"http://export.arxiv.org/api/query?{q}", timeout=60) as resp:
                    xml = resp.read().decode("utf-8")
                time.sleep(3.1)
                page = []
                for e in ET.fromstring(xml).findall("a:entry", ns):
                    prim = e.find("x:primary_category", ns)
                    text = re.sub(r"\s+", " ", f"{e.findtext('a:title', '', ns)}. {e.findtext('a:summary', '', ns)}").strip()
                    page.append({"id": e.findtext("a:id", "", ns), "primary": prim.get("term") if prim is not None else "", "text": text})
                got += [p for p in page if p["primary"] == cat]
                if len(got) >= ARXIV_PER_CAT or len(page) < 200:
                    break
            for it in got[:ARXIV_PER_CAT]:
                rid = f"{name}-{it['id'].rsplit('/', 1)[-1]}"
                if rid not in seen:
                    seen.add(rid)
                    rows.append((rid, it["text"], y))
        return rows

    def prepare(self, split: str) -> dict[str, Any]:
        return {"spec": SPEC_ID, "upstream": UPSTREAM, "revisions": self._revisions(), "batch": BATCH, "eval_n": EVAL_N,
                "seed": EVAL_SEED, "note": "arxiv2026 is an indicative re-crawl; the author's sample is not published"}

    def expected_counts(self, split: str) -> dict[str, int]:
        return {"agnews": 1000, "sst2": 872, "banking77": 1000, "tweet_emotion": 1000, "paws": 1000, "arxiv2026": 258} if split == "test" else {}

    def items(self, split: str, limit: int | None = None, tasks: Sequence[str] | None = None) -> list[Item]:
        out: list[Item] = []
        for name in self._select_tasks(tasks):
            labels = load_labels(name if name != "arxiv2026" else "arxiv2026")
            rows = take(self._eval_frame(name, split), limit)
            for b in range(0, len(rows), BATCH):
                batch = rows[b:b + BATCH]
                texts = [t for _, t, _ in batch]
                golds = [y for _, _, y in batch]
                ids = [i for i, _, _ in batch]
                if labels["groups"]:  # banking77 stage 1: group choice
                    gnames = list(labels["groups"])
                    gcrit = {g: labels["groups"][g]["desc"] for g in gnames}
                    req = build_request(labels, texts, [gcrit] * len(texts), "Which topic area is this about?")
                    out.append(Item(f"{name}/batch{b // BATCH:03d}/stage1", name, req, golds, group=f"{name}/batch{b // BATCH:03d}",
                                    meta={"stage": 1, "ids": ids, "texts": texts, "variant": "desc"}))
                else:
                    crit = criteria(labels["names"], label_texts(labels, "desc"))
                    req = build_request(labels, texts, [crit] * len(texts), labels["question"])
                    out.append(Item(f"{name}/batch{b // BATCH:03d}", name, req, golds, group=f"{name}/batch{b // BATCH:03d}",
                                    meta={"stage": 1, "ids": ids, "variant": "desc"}))
        return out

    def stage_items(self, stage: int, items: Sequence[Item], answers: dict[str, dict[str, Any]]) -> list[Item]:
        """Stage 2 (banking77 only): the label choice inside each text's argmax group."""
        if stage != 2:
            return []
        out: list[Item] = []
        for it in items:
            if it.meta.get("stage") != 1 or it.task != "banking77":
                continue
            labels = load_labels("banking77")
            gnames = list(labels["groups"])
            a = answers_of(answers.get(it.id))
            if a is None:
                continue
            gp = [dist(a.get(f"t{i}"), gnames) for i in range(1, len(it.meta["texts"]) + 1)]
            chosen = [gnames[max(range(len(gnames)), key=lambda j: (p[j], -j))] for p in gp]
            all_txt = dict(zip(labels["names"], label_texts(labels, "desc")))
            crits = [criteria(labels["groups"][g]["labels"], [all_txt[n] for n in labels["groups"][g]["labels"]]) for g in chosen]
            req = build_request(labels, it.meta["texts"], crits, labels["question"])
            out.append(Item(it.id.replace("/stage1", "/stage2"), it.task, req, it.gold, group=it.group,
                            meta={"stage": 2, "ids": it.meta["ids"], "chosen_groups": chosen, "group_probs": gp, "variant": "desc"}))
        return out

    def score(self, items: Sequence[Item], answers: dict[str, dict[str, Any]], split: str,
              thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
        res: dict[str, Any] = {"spec": SPEC_ID, "split": split, "tasks": {}}
        for name, its in by_task(items).items():
            labels = load_labels(name)
            keys = [key_of(n) for n in labels["names"]]
            gold, probs, failed_batches = [], [], 0
            if labels["groups"]:
                stage2 = {it.group: it for it in its if it.meta.get("stage") == 2}
                for it in (x for x in its if x.meta.get("stage") == 1):
                    s2 = stage2.get(it.group)
                    a2 = answers_of(answers.get(s2.id)) if s2 else None
                    if a2 is None:
                        failed_batches += 1
                        continue
                    gnames = list(labels["groups"])
                    col = {key_of(n): j for j, n in enumerate(labels["names"])}
                    for i, (g, crit_keys, gp) in enumerate(zip(s2.meta["chosen_groups"],
                                                              [list(c) for c in (s2.request["questions"][f"t{k}"]["criteria"] for k in range(1, len(it.gold) + 1))],
                                                              s2.meta["group_probs"]), start=1):
                        inner = dist(a2.get(f"t{i}"), crit_keys)
                        row = [0.0] * len(keys)
                        for k, v in zip(crit_keys, inner):
                            row[col[k]] = v * gp[gnames.index(g)]
                        s = sum(row)
                        probs.append([x / s for x in row] if s > 0 else [1.0 / len(keys)] * len(keys))
                        gold.append(it.gold[i - 1])
            else:
                for it in its:
                    a = answers_of(answers.get(it.id))
                    if a is None:
                        failed_batches += 1
                        continue
                    for i, g in enumerate(it.gold, start=1):
                        probs.append(dist(a.get(f"t{i}"), keys))
                        gold.append(g)
            r: dict[str, Any] = {**coverage(its, answers), "failed_batches": failed_batches, "n": len(gold),
                                 "target": TARGET_OF.get(name), "indicative": name.startswith("arxiv")}
            if gold:
                pred = [max(range(len(keys)), key=lambda i: (row[i], -i)) for row in probs]
                r["accuracy"] = accuracy(pred, gold)
                r["macro_f1"] = macro_f1(pred, gold, sorted(set(gold)))
                r["ece_15"] = ece15(probs, gold)
                if len(keys) == 2:
                    r["auc"] = roc_auc([row[1] for row in probs], [g == 1 for g in gold])
            res["tasks"][name] = r
        return res
