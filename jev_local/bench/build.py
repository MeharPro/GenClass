"""jevbench v1 builder: download, pin, sample and render every registry dataset (VM only).

    ~/jev/.venv/bin/python -m jev_local.bench.build --out ~/jev/bench/jevbench [--keys k1,k2] [--roles test,dev,ref]

Output (one JSONL per dataset and variant):
    <out>/<role>/<key>__<variant>.jsonl   rows {"id","dataset","variant","item","stratum","request","gold","meta"}
    <out>/manifest.json                   pinned revisions, counts, sha256 of every file, code hashes

Variants: "main" (described labels, dataset label order), "shuffle" (one seeded option-order rotation of every
choice question), "bare" (choice descriptions removed), plus mapper-specific extras ("choice" for SST-2).

`datasets` / `huggingface_hub` / `pyarrow` are imported inside functions so the module imports on the Mac.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from jev_local.bench import registry as R
from jev_local.bench import templates as T

TASKSOURCE_JEV = "tasksource/tasksource-jev-typed-decisions"
TASKSOURCE_JEV_REV = "8173a06c7bb640b6158c6a93519cb535196bef15"
DEV_PER_FAMILY = 200


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hashes() -> dict[str, str]:
    here = Path(__file__).resolve().parent
    return {f: sha256_file(here / f) for f in ("registry.py", "templates.py", "build.py")}


def rng_for(*parts: Any) -> random.Random:
    # str seeds hash with SHA-512 (random.seed version 2): stable across processes and Python builds.
    return random.Random(":".join(map(str, (R.SEED, *parts))))


# ---------------------------------------------------------------------------------------------- loading


def resolve_revision(d: R.BenchDataset) -> str:
    if d.revision:
        return d.revision
    from huggingface_hub import HfApi

    return HfApi().dataset_info(d.hf_id).sha


def hub_parquet(hf_id: str) -> tuple[str, Callable[[str], Any]]:
    """Script-only datasets (`datasets` >= 4 refuses their loader): read the Hub's parquet conversion
    (refs/convert/parquet), pinned by its own sha. -> (conversion sha, load(path within the conversion))."""
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    refs = HfApi().list_repo_refs(hf_id, repo_type="dataset")
    conv = next(r for r in refs.converts if r.name == "parquet").target_commit

    def load(path: str):
        p = hf_hub_download(hf_id, path, repo_type="dataset", revision=conv)
        return load_dataset("parquet", data_files=p, split="train")

    return conv, load


def load_rows(d: R.BenchDataset, rev: str) -> tuple[list[tuple[str, dict]], Any, dict]:
    """-> ([(item_id, row)], features of the first split, extra provenance)."""
    from datasets import load_dataset

    extra: dict = {}
    if d.key == "tweet_topic":  # script dataset: read the Hub's parquet conversion, pinned by its own sha
        conv, load = hub_parquet(d.hf_id)
        ds = load("tweet_topic_single/test_2021/0000.parquet")
        extra["parquet_conversion_sha"] = conv
        return [(f"test_2021:{i}", r) for i, r in enumerate(ds)], ds.features, extra
    if d.key == "scifact":  # script dataset; `claims` is one row per (claim, evidence abstract, rationale) -> one per claim
        conv, load = hub_parquet(d.hf_id)
        (sp,) = d.splits
        claims = load(f"{d.config}/{sp}/0000.parquet")
        corpus = load("corpus/train/0000.parquet")
        docs = {str(r["doc_id"]): {"doc_id": str(r["doc_id"]), "title": r["title"], "abstract": list(r["abstract"])}
                for r in corpus}
        groups = T.scifact_claim_rows(list(claims), docs)
        extra.update(parquet_conversion_sha=conv, claims_rows=len(claims),
                     corpus={"config": "corpus", "split": "train", "docs": len(docs)})
        return [(f"{sp}:{g['id']}", g) for g in groups], claims.features, extra
    rows: list[tuple[str, dict]] = []
    feats = None
    for sp in d.splits:
        ds = load_dataset(d.hf_id, d.config, split=sp, revision=rev)
        feats = feats or ds.features
        rows += [(f"{sp}:{i}", r) for i, r in enumerate(ds)]
    return rows, feats, extra


def class_names(d: R.BenchDataset, rows: list[tuple[str, dict]], feats) -> list[str]:
    def cl(col):
        f = feats.get(col) if feats is not None else None
        return list(getattr(f, "names", None) or getattr(getattr(f, "feature", None), "names", None) or [])

    k = d.key
    if k in ("ag_news", "sst2", "fin_phrasebank", "dair_emotion", "tweeteval_emotion", "rte", "tweeteval_sentiment",
             "cb", "tweeteval_offensive", "yelp5", "mrpc"):
        return cl("label")
    if k == "yahoo_topics":
        return cl("topic")
    if k == "clinc150":
        return cl("intent")
    if k == "climate_fever":
        return cl("claim_label")
    if k == "go_emotions":
        return cl("labels")
    if k == "banking77":
        return sorted({r["label_text"] for _, r in rows})  # Deußer: sorted(set(label_text))
    if k == "massive":
        return sorted({r["label"] for _, r in rows})
    if k == "snips":
        return sorted({r["category"] for _, r in rows})
    if k == "atis":
        return sorted({r["intent"] for _, r in rows})  # intents present in the split, joint intents ("a+b") included
    if k == "scifact":
        return list(T.SCIFACT)
    if k == "newsgroups20":
        return [t for _, t in sorted({(r["label"], r["label_text"]) for _, r in rows})]
    if k == "tweet_topic":
        names = cl("label")
        return names or ["arts_&_culture", "business_&_entrepreneurs", "pop_culture", "daily_life",
                         "sports_&_gaming", "science_&_technology"]
    if k == "hwu64":
        from datasets import load_dataset

        it = load_dataset(d.hf_id, "intents", split="intents", revision=resolve_revision(d))
        return [n for _, n in sorted(zip(it["id"], it["name"]))]
    return []


def btzsc_hypotheses(config: str) -> dict[str, str]:
    from datasets import load_dataset

    ds = load_dataset(T.BTZSC_ID, config, split="test", revision=T.BTZSC_REVISION)
    out: dict[str, str] = {}
    for r in ds:
        if str(r["labels"]) == "1":
            out.setdefault(T.norm_label(r["label_text"]), r["hypothesis"].strip())
    return out


# ---------------------------------------------------------------------------------------------- sampling


def proportional_quotas(counts: dict[str, int], n: int) -> dict[str, int]:
    """Largest-remainder allocation of n over strata in proportion to their sizes (ties: stratum name)."""
    total = sum(counts.values())
    if n >= total:
        return dict(counts)
    raw = {k: n * c / total for k, c in counts.items()}
    q = {k: min(counts[k], math.floor(v)) for k, v in raw.items()}
    left = n - sum(q.values())
    for k in sorted(counts, key=lambda k: (-(raw[k] - math.floor(raw[k])), k)):
        if left <= 0:
            break
        if q[k] < counts[k]:
            q[k] += 1
            left -= 1
    return q


def sample_items(items: list[T.Item], d: R.BenchDataset) -> list[T.Item]:
    n = d.n_used
    if d.sampling == "all" or len(items) <= n:
        return items
    rng = rng_for(d.key, "sample")
    if d.sampling == "random":
        keep = set(rng.sample(range(len(items)), n))
        return [it for i, it in enumerate(items) if i in keep]
    # "strat" (target) or "strat_by_source" (stratum carries "source|label"): proportional allocation
    by: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        by[it.stratum].append(i)
    quotas = proportional_quotas({k: len(v) for k, v in by.items()}, n)
    keep: set[int] = set()
    for k in sorted(by):
        keep.update(rng.sample(by[k], quotas[k]))
    return [it for i, it in enumerate(items) if i in keep]


# ---------------------------------------------------------------------------------------------- tasksource dev


def tasksource_dev_items() -> tuple[list[T.Item], dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    files = [s.rfilename for s in HfApi().dataset_info(TASKSOURCE_JEV, revision=TASKSOURCE_JEV_REV).siblings
             if s.rfilename.startswith("data/train-")]
    fams = R.DEV_TASKSOURCE_SOURCES
    pool: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(files):
        p = hf_hub_download(TASKSOURCE_JEV, f, repo_type="dataset", revision=TASKSOURCE_JEV_REV)
        t = pq.read_table(p)
        src = t.column("source").to_pylist()
        idx = [i for i, s in enumerate(src) if s.split("/")[0] in fams]
        if idx:
            for r in t.take(idx).to_pylist():
                pool[r["source"].split("/")[0]].append(r)
    items: list[T.Item] = []
    for fam in sorted(pool):
        rows = sorted(pool[fam], key=lambda r: r["id"])
        pick = rng_for("tasksource_heldout", fam).sample(rows, min(DEV_PER_FAMILY, len(rows)))
        for r in sorted(pick, key=lambda r: r["id"]):
            it = _tasksource_item(r)
            if it:
                items.append(it)
    return items, {"hf_id": TASKSOURCE_JEV, "revision": TASKSOURCE_JEV_REV, "families": {k: len(v) for k, v in pool.items()}}


def _tasksource_item(r: dict) -> T.Item | None:
    kind, opts, tgt = r["kind"], list(r["options"] or []), list(r["target"] or [])
    instr = (r["question"] or "").strip() or "Which option best fits?"
    if kind == "choice":
        if len(opts) < 2 or len(opts) > 255 or len(tgt) != len(opts):
            return None
        short = len(set(opts)) == len(opts) and all(0 < len(o) <= 60 and o.strip().lower() not in ("true", "false") for o in opts)
        keys = opts if short else [T._alpha(i) for i in range(len(opts))]
        crit = {k: (None if short else o) for k, o in zip(keys, opts)}
        dist = {k: float(p) for k, p in zip(keys, tgt)}
        gold = {"type": "choice", "label": max(dist, key=dist.get), "dist": dist}
        q = T.choice(instr, crit)
    elif kind == "noul":
        p = float(tgt[-1]) if tgt else None
        if p is None:
            return None
        q = {"type": "noul", "instructions": instr}
        gold = {"type": "noul", "label": p >= 0.5, "p": p}
    elif kind == "score":
        if not 2 <= len(opts) <= 10 or len(tgt) != len(opts):
            return None
        q = T.score(instr, opts)
        lv = max(range(len(tgt)), key=lambda j: tgt[j])
        gold = {"type": "score", "label": lv, "value": sum(j * p for j, p in enumerate(tgt)), "dist": [float(x) for x in tgt]}
    else:
        return None
    state = r["state"]
    try:  # many tasksource states are JSON objects serialised as strings
        js = json.loads(state)
        if isinstance(js, (dict, list)):
            state = js
    except (TypeError, ValueError):
        pass
    return T.Item(r["id"], state, {"decision": q}, {"decision": gold}, r["source"].split("/")[0],
                  f"native:tasksource-jev/{r['source']}", meta={"source": r["source"], "license_use": r.get("license_use")})


# ---------------------------------------------------------------------------------------------- writing


def row_json(d_key: str, variant: str, it: T.Item, questions: dict, gold: dict) -> dict:
    return {"id": f"{d_key}/{variant}/{it.item_id}", "dataset": d_key, "variant": variant, "item": it.item_id,
            "stratum": it.stratum, "request": {"state": it.state, "questions": questions}, "gold": gold,
            "meta": {"template": it.template, **it.meta}}


def write_jsonl(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n")
    tmp.replace(path)
    return {"path": str(path.relative_to(path.parents[1])), "rows": len(rows), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def build_dataset(d: R.BenchDataset, out: Path) -> dict:
    t0 = time.time()
    entry: dict[str, Any] = {"role": d.role, "num": d.num, "area": d.area, "title": d.title, "hf_id": d.hf_id,
                             "config": d.config, "splits": list(d.splits), "sampling": d.sampling,
                             "n_target": d.n_used, "licence": d.licence, "license_use": d.license_use,
                             "label_source": d.label_source}
    if d.key == "tasksource_heldout":
        items, prov = tasksource_dev_items()
        entry.update(revision=TASKSOURCE_JEV_REV, n_total=sum(prov["families"].values()), provenance=prov)
    else:
        if d.key not in T.MAPPERS:
            entry["skipped"] = "no mapper (not built in v1)"
            return entry
        try:
            rev = resolve_revision(d)
            rows, feats, extra = load_rows(d, rev)
        except Exception as e:  # gated / script-only datasets: record and skip (PLAN: failures are reported)
            entry["skipped"] = f"{type(e).__name__}: {str(e)[:300]}"
            return entry
        entry.update(revision=rev, n_total=len(rows), **extra)
        ctx = T.Ctx(names=class_names(d, rows, feats))
        if d.key in T.BTZSC_CONFIG:
            ctx.btzsc = btzsc_hypotheses(T.BTZSC_CONFIG[d.key])
        mapper = T.MAPPERS[d.key]
        items = []
        dropped = Counter()
        for iid, row in rows:
            try:
                it = mapper(row, iid, ctx)
            except (KeyError, IndexError, TypeError, ValueError) as e:
                dropped[type(e).__name__] += 1
                continue
            if it is None:
                dropped["none"] += 1
                continue
            items.append(it)
        entry["dropped_rows"] = dict(dropped)
        if ctx.btzsc or d.key in T.BTZSC_CONFIG:
            miss = sorted(set(ctx.btzsc_missing))
            entry["btzsc"] = {"config": T.BTZSC_CONFIG.get(d.key), "revision": T.BTZSC_REVISION,
                              "hypotheses": len(ctx.btzsc), "labels_without_hypothesis": miss}
        entry["class_names"] = ctx.names
    items = sample_items(items, d)
    entry["n_used"] = len(items)
    entry["strata"] = dict(sorted(Counter(it.stratum for it in items).items()))
    errs = Counter()
    variants: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        variants["main"].append(row_json(d.key, "main", it, it.questions, it.gold))
        if d.shuffle_rotation:
            sq = T.shuffled(it.questions, rng_for(d.key, it.item_id, "shuffle"))
            variants["shuffle"].append(row_json(d.key, "shuffle", it, sq, it.gold))
        if d.bare_variant:
            variants["bare"].append(row_json(d.key, "bare", it, T.bare(it.questions), it.gold))
        for v, (qs, gold) in it.extra.items():
            variants[v].append(row_json(d.key, v, it, qs, gold))
    for v, rows_v in variants.items():
        for r in rows_v:
            for e in T.validate_request(r["request"]):
                errs[e.split(":")[-1].strip()] += 1
    if errs:
        entry["validation_errors"] = dict(errs)
    ids = [r["id"] for rows_v in variants.values() for r in rows_v]
    assert len(ids) == len(set(ids)), f"{d.key}: duplicate request ids"
    entry["files"] = {v: write_jsonl(out / d.role / f"{d.key}__{v}.jsonl", rows_v) for v, rows_v in variants.items()}
    entry["seconds"] = round(time.time() - t0, 1)
    return entry


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path.home() / "jev" / "bench" / "jevbench")
    ap.add_argument("--keys", default="")
    ap.add_argument("--roles", default="test,dev,ref")
    a = ap.parse_args(argv)
    roles = set(a.roles.split(","))
    keys = set(filter(None, a.keys.split(",")))
    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {"datasets": {}}
    for d in R.ALL_DATASETS:
        if d.role not in roles or (keys and d.key not in keys):
            continue
        print(f"[build] {d.key} ...", file=sys.stderr, flush=True)
        entry = build_dataset(d, out)
        manifest["datasets"][d.key] = entry
        print(f"[build] {d.key}: {entry.get('n_used', 0)} items, {entry.get('skipped', 'ok')} "
              f"{entry.get('validation_errors', '')} ({entry.get('seconds', 0)} s)", file=sys.stderr, flush=True)
        # write after every dataset so a crash keeps progress
        manifest.update(version=R.REGISTRY_VERSION, template_version=T.TEMPLATE_VERSION, seed=R.SEED,
                        built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), code_sha256=code_hashes(),
                        label_sources={"deusser": {"repo": T.DEUSSER_REPO, "commit": T.DEUSSER_COMMIT},
                                       "btzsc": {"hf_id": T.BTZSC_ID, "revision": T.BTZSC_REVISION}})
        mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    tot = Counter()
    for k, e in manifest["datasets"].items():
        for v, f in (e.get("files") or {}).items():
            tot[f"{e['role']}:{v}"] += f["rows"]
    manifest["totals"] = dict(sorted(tot.items()))
    mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(json.dumps(manifest["totals"], indent=1))


if __name__ == "__main__":
    main()
