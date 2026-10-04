"""Shared pieces of the group-B adapters: the Item record, pinned fetching, HF parquet reading, answer readers
and the metrics the publishers use. Pure Python at import time (numpy / pyarrow / huggingface_hub lazy).

Hygiene (PLAN §2.2, §3.2): every upstream file is fetched by pinned commit / revision and, where the publisher
gives one, verified by sha256; nothing here ever reads a Jev per-item output file; test items are only ever
turned into requests and scored (the self-check CLI refuses split="test" unless asked for the real run).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

VENDOR = Path(__file__).resolve().parent / "vendor"
DEFAULT_WORK = Path(os.environ.get("BENCHMAX_WORK", str(Path.home() / "bench_work_b")))
MODEL_ID = os.environ.get("BENCHMAX_MODEL", "genclass-68m")  # W10: never jev-*


def work_dir(work: str | Path | None = None) -> Path:
    return Path(work).expanduser() if work else DEFAULT_WORK


def repos_dir(work: str | Path | None = None) -> Path:
    return Path(os.environ.get("BENCHMAX_REPOS", str(work_dir(work) / "repos")))


def hf_home(work: str | Path | None = None) -> Path:
    """Own HF cache under the work dir (several agents share the VM)."""
    return Path(os.environ.get("BENCHMAX_HF_HOME", str(work_dir(work) / "hf")))


# ---------------------------------------------------------------------------------------------- items


@dataclass
class Item:
    """One request to the engine. `gold` never enters `request`; `group` ties requests of one scoring unit."""

    id: str
    task: str
    request: dict[str, Any]  # {"state": ..., "questions": {...}}; the runner adds "model"
    gold: Any = None
    group: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "task": self.task, "request": self.request, "gold": self.gold, "group": self.group,
                "meta": self.meta}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Item":
        return cls(d["id"], d["task"], d["request"], d.get("gold"), d.get("group"), d.get("meta") or {})


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.to_json() if isinstance(r, Item) else r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------------------------- hashing / fetching


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def fetch_verified(url: str, dest: Path, sha256: str | None = None, timeout: float = 180.0) -> Path:
    """Download `url` to `dest` once; refuse (and never overwrite) when the sha256 does not match."""
    dest = Path(dest)
    if dest.exists():
        if sha256 and sha256_file(dest) != sha256:
            raise ValueError(f"cached {dest} does not match the pinned sha256 {sha256[:12]}")
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "meharsjev-benchmax/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if sha256 and sha256_bytes(data) != sha256:
        raise ValueError(f"{url}: sha256 {sha256_bytes(data)[:12]} != pinned {sha256[:12]}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return dest


def github_raw(owner_repo: str, commit: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{owner_repo}/{commit}/{path}"


# ---------------------------------------------------------------------------------------------- hugging face (lazy)


def _hf_api():
    from huggingface_hub import HfApi

    return HfApi()


def hf_resolve_revision(repo_id: str, revision: str | None = None, repo_type: str = "dataset") -> str:
    """The commit sha a dataset revision points at (records the pin when the publisher left it floating)."""
    if revision and re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    info = _hf_api().repo_info(repo_id, repo_type=repo_type, revision=revision)
    return info.sha


def hf_download(repo_id: str, filename: str, revision: str, repo_type: str = "dataset",
                work: str | Path | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, repo_type=repo_type,
                        cache_dir=str(hf_home(work) / "hub"))
    return Path(p)


def hf_list_files(repo_id: str, revision: str, repo_type: str = "dataset") -> list[str]:
    return sorted(_hf_api().list_repo_files(repo_id, revision=revision, repo_type=repo_type))


def parquet_rows(path: str | Path, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(str(path), columns=list(columns) if columns else None)
    return table.to_pylist()


def parquet_feature_names(path: str | Path, column: str) -> list[str] | None:
    """ClassLabel names of `column` from the HF metadata embedded in a parquet file (None if absent)."""
    import pyarrow.parquet as pq

    meta = pq.read_schema(str(path)).metadata or {}
    raw = meta.get(b"huggingface")
    if not raw:
        return None
    info = json.loads(raw.decode("utf-8"))
    feat = (info.get("info") or {}).get("features", {}).get(column)
    if not feat:
        return None
    if feat.get("_type") == "ClassLabel":
        return list(feat["names"])
    inner = feat.get("feature") if isinstance(feat, dict) else None
    if isinstance(inner, dict) and inner.get("_type") == "ClassLabel":
        return list(inner["names"])
    return None


def hf_parquet_split(repo_id: str, revision: str, split: str, config: str | None = None,
                     work: str | Path | None = None, columns: Sequence[str] | None = None,
                     ) -> tuple[list[dict[str, Any]], list[Path]]:
    """Rows of one parquet-backed split at a pinned revision, shards in name order. Returns (rows, shard paths)."""
    files = hf_list_files(repo_id, revision)
    prefix = f"{config}/" if config else ""
    cands = [f for f in files if f.startswith(prefix) and f.endswith(".parquet")
             and re.search(rf"(^|/){re.escape(split)}[-.]", f.rsplit("/", 1)[-1] if "/" in f else f)]
    if not cands and not config:  # datasets with a default config directory (e.g. "default/train-....parquet")
        cands = [f for f in files if f.endswith(".parquet") and re.search(rf"/{re.escape(split)}[-.]", f)]
    if not cands:
        raise FileNotFoundError(f"{repo_id}@{revision[:8]}: no parquet shard for config={config!r} split={split!r}")
    paths = [hf_download(repo_id, f, revision, work=work) for f in cands]
    rows: list[dict[str, Any]] = []
    for p in paths:
        rows.extend(parquet_rows(p, columns))
    return rows, paths


def read_csv_rows(path: str | Path, skipinitialspace: bool = False) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return [{k: ("" if v is None else v) for k, v in row.items()}
                for row in csv.DictReader(f, skipinitialspace=skipinitialspace)]


# ---------------------------------------------------------------------------------------------- answers


class BadAnswer(ValueError):
    pass


def answers_of(rec: dict[str, Any] | None) -> dict[str, Any] | None:
    """The `answers` map of one runner record ({"ok", "answers", ...}); None for a failed request."""
    if not rec or not rec.get("ok", True):
        return None
    a = rec.get("answers")
    return a if isinstance(a, dict) else None


def noul_p(answer: dict[str, Any] | None) -> float | None:
    if not answer or answer.get("type") not in (None, "noul"):
        return None
    p = answer.get("noul", answer.get("probability"))
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p):
        return None
    return float(p)


def choice_probs(answer: dict[str, Any] | None, labels: Sequence[str]) -> list[float] | None:
    """Probabilities in `labels` order (missing keys 0); None when the answer is unusable."""
    if not answer or answer.get("type") not in (None, "choice"):
        return None
    probs = answer.get("probabilities")
    if not isinstance(probs, dict):
        return None
    out = []
    for lab in labels:
        v = probs.get(lab, 0.0)
        out.append(float(v) if isinstance(v, (int, float)) and math.isfinite(v) else 0.0)
    return out


def choice_pick(answer: dict[str, Any] | None, labels: Sequence[str]) -> str | None:
    """The wire `choice` when it is one of `labels`; else the argmax of the probabilities (first label on ties)."""
    if not answer:
        return None
    c = answer.get("choice")
    if isinstance(c, str) and c in labels:
        return c
    probs = choice_probs(answer, labels)
    if probs is None or sum(probs) <= 0:
        return None
    best = max(range(len(labels)), key=lambda i: (probs[i], -i))
    return labels[best]


def renormalise(p: Sequence[float]) -> list[float]:
    s = float(sum(p))
    return [x / s for x in p] if s > 0 else list(p)


def score_expected(answer: dict[str, Any] | None, k: int) -> float | None:
    """Expected level Σ i·p_i from a score answer (its `probabilities` keyed "0".."K-1"), else its `score`."""
    if not answer or answer.get("type") not in (None, "score"):
        return None
    probs = answer.get("probabilities")
    if isinstance(probs, dict) and probs:
        try:
            vec = [float(probs.get(str(i), 0.0)) for i in range(k)]
        except (TypeError, ValueError):
            return None
        s = sum(vec)
        if s > 0:
            return sum(i * v for i, v in enumerate(vec)) / s
    sc = answer.get("score")
    return float(sc) if isinstance(sc, (int, float)) and math.isfinite(sc) else None


def confidence_of(answer: dict[str, Any] | None) -> float | None:
    if not answer:
        return None
    c = answer.get("confidence")
    return float(c) if isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) else None


# ---------------------------------------------------------------------------------------------- metrics (pure Python)


def accuracy(pred: Sequence[Any], gold: Sequence[Any]) -> float:
    return sum(p == g for p, g in zip(pred, gold)) / len(gold) if gold else float("nan")


def f1_counts(tp: int, fp: int, fn: int) -> float:
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


def binary_f1(pred: Sequence[bool], gold: Sequence[bool]) -> dict[str, Any]:
    tp = sum(1 for p, g in zip(pred, gold) if p and g)
    fp = sum(1 for p, g in zip(pred, gold) if p and not g)
    fn = sum(1 for p, g in zip(pred, gold) if not p and g)
    tn = len(gold) - tp - fp - fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "f1": f1_counts(tp, fp, fn),
            "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0,
            "accuracy": (tp + tn) / len(gold) if gold else float("nan")}


def macro_f1(pred: Sequence[Any], gold: Sequence[Any], labels: Sequence[Any] | None = None) -> float:
    """Unweighted mean of per-class F1. `labels=None` = sklearn's default (union of observed gold and pred),
    else exactly the given label set (absent classes score 0)."""
    labs = list(labels) if labels is not None else sorted(set(gold) | set(pred), key=str)
    if not labs:
        return float("nan")
    scores = []
    for lab in labs:
        tp = sum(1 for p, g in zip(pred, gold) if p == lab and g == lab)
        fp = sum(1 for p, g in zip(pred, gold) if p == lab and g != lab)
        fn = sum(1 for p, g in zip(pred, gold) if p != lab and g == lab)
        scores.append(f1_counts(tp, fp, fn))
    return sum(scores) / len(scores)


def multilabel_f1(pred_sets: Sequence[Iterable[int]], gold_sets: Sequence[Iterable[int]], width: int,
                  none_column: bool = False) -> dict[str, float]:
    """LexGLUE-style micro/macro-F1 over all label columns (+ an optional synthetic 'none' column for empty sets).
    Macro averages over every column (zero_division=0); exact_match is the share of identical sets."""
    w = width + (1 if none_column else 0)
    tp = [0] * w
    fp = [0] * w
    fn = [0] * w
    exact = 0
    for ps, gs in zip(pred_sets, gold_sets):
        p = set(ps)
        g = set(gs)
        if none_column:
            if not p:
                p = {width}
            if not g:
                g = {width}
        exact += p == g
        for j in p & g:
            tp[j] += 1
        for j in p - g:
            fp[j] += 1
        for j in g - p:
            fn[j] += 1
    micro = f1_counts(sum(tp), sum(fp), sum(fn))
    macro = sum(f1_counts(tp[j], fp[j], fn[j]) for j in range(w)) / w if w else float("nan")
    return {"micro_f1": micro, "macro_f1": macro, "exact_match": exact / len(gold_sets) if gold_sets else float("nan")}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def ece_equal_width(conf: Sequence[float], correct: Sequence[bool], bins: int = 10,
                    right_closed: bool = False) -> float:
    """ECE with equal-width bins. right_closed=False: bin = min(int(c*bins), bins-1) (floor; 1.0 in the last bin).
    right_closed=True: edge[b] < c <= edge[b+1] with c == 0 in bin 0 (elcronos / earino convention)."""
    n = len(conf)
    if n == 0:
        return float("nan")
    tot_c = [0.0] * bins
    tot_a = [0.0] * bins
    cnt = [0] * bins
    for c, ok in zip(conf, correct):
        if right_closed:
            b = 0 if c <= 0 else min(bins - 1, math.ceil(c * bins - 1e-12) - 1)
        else:
            b = min(int(c * bins), bins - 1)
        cnt[b] += 1
        tot_c[b] += c
        tot_a[b] += 1.0 if ok else 0.0
    return sum(cnt[b] / n * abs(tot_a[b] / cnt[b] - tot_c[b] / cnt[b]) for b in range(bins) if cnt[b])


def average_precision(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """sklearn's average_precision_score: Σ (R_n − R_{n−1}) P_n over thresholds, ties grouped."""
    pairs = sorted(zip(scores, labels), key=lambda t: -t[0])
    npos = sum(1 for _, y in pairs if y)
    if npos == 0 or not pairs:
        return float("nan")
    ap = 0.0
    tp = fp = 0
    prev_r = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            tp += 1 if pairs[j][1] else 0
            fp += 0 if pairs[j][1] else 1
            j += 1
        r = tp / npos
        p = tp / (tp + fp)
        ap += (r - prev_r) * p
        prev_r = r
        i = j
    return ap


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Mann–Whitney AUC with ties counting one half."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    order = sorted(zip(scores, labels))
    rank_sum = 0.0
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and order[j][0] == order[i][0]:
            j += 1
        mid = (i + 1 + j) / 2
        rank_sum += mid * sum(1 for k in range(i, j) if order[k][1])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def ndcg_at_k(ranked: Sequence[str], rels: dict[str, int], k: int = 10, exponential: bool = True) -> float:
    """nDCG@k; gains 2^rel−1 (BEIR / trec_eval, denser) or linear rel (anessbelbati); ideal over all qrels."""
    g = (lambda r: (2 ** r) - 1) if exponential else (lambda r: r)
    dcg = sum(g(rels.get(d, 0)) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted((g(r) for r in rels.values()), reverse=True)[:k]
    idcg = sum(x / math.log2(i + 2) for i, x in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(ranked: Sequence[str], rels: dict[str, int], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k]):
        if d in rels:
            return 1.0 / (i + 1)
    return 0.0


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        r = (i + j + 1) / 2
        for idx in order[i:j]:
            ranks[idx] = r
        i = j
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den else float("nan")


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(_ranks(x), _ranks(y))


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def bootstrap_ci(values_fn: Callable[[Sequence[int]], float], n: int, resamples: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap over item indices (numpy for speed when present, else `random`)."""
    try:
        import numpy as np

        rng = np.random.default_rng(seed)
        stats = sorted(values_fn(rng.integers(0, n, n).tolist()) for _ in range(resamples))
    except ImportError:  # pragma: no cover - numpy is always on the VM
        import random

        r = random.Random(seed)
        stats = sorted(values_fn([r.randrange(n) for _ in range(n)]) for _ in range(resamples))
    lo = stats[int(alpha / 2 * (len(stats) - 1))]
    hi = stats[int((1 - alpha / 2) * (len(stats) - 1))]
    return (lo, hi)


# ---------------------------------------------------------------------------------------------- adapter base


class AdapterB:
    """Base class: `Class(spec, args)` as `jev_local.bench.benchmax.make_adapter` builds it. `args` may carry
    `work` (work dir), `limit` and `tasks`; everything else is read from the environment."""

    SPEC_ID = ""
    TARGETS: tuple[str, ...] = ()
    SPLITS: tuple[str, ...] = ("validation", "test")
    TASKS: tuple[str, ...] = ()
    EVAL_SPLIT = "test"  # the split the published Jev number was measured on

    def __init__(self, spec: Any = None, args: Any = None):
        self.spec = spec
        self.args = args
        self.work = work_dir(getattr(args, "work", None))

    # subclasses implement:
    def prepare(self, split: str) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def items(self, split: str, limit: int | None = None, tasks: Sequence[str] | None = None) -> list[Item]:  # pragma: no cover
        raise NotImplementedError

    def score(self, items: Sequence[Item], answers: dict[str, dict[str, Any]], split: str,
              thresholds: dict[str, Any] | None = None) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def fit(self, items: Sequence[Item], answers: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
        """Thresholds chosen on validation items (None when the protocol has no tuned variant)."""
        return None

    def expected_counts(self, split: str) -> dict[str, int]:
        """Expected number of items per task for `split` (verification hook); {} when unknown."""
        return {}

    # helpers
    def raw_dir(self, name: str) -> Path:
        p = self.work / "raw" / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _select_tasks(self, tasks: Sequence[str] | None) -> list[str]:
        if not tasks:
            return list(self.TASKS)
        unknown = [t for t in tasks if t not in self.TASKS]
        if unknown:
            raise KeyError(f"{self.SPEC_ID}: unknown tasks {unknown}; known {list(self.TASKS)}")
        return list(tasks)


def by_task(items: Sequence[Item]) -> dict[str, list[Item]]:
    out: dict[str, list[Item]] = {}
    for it in items:
        out.setdefault(it.task, []).append(it)
    return out


def take(items: list[Item], limit: int | None) -> list[Item]:
    return items if limit is None else items[:limit]


def coverage(items: Sequence[Item], answers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    n_ok = sum(1 for it in items if answers_of(answers.get(it.id)) is not None)
    return {"n_items": len(items), "n_answered": n_ok, "coverage": n_ok / len(items) if items else float("nan")}
