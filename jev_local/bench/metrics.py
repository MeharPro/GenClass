"""jevbench v1 scoring: per-dataset metrics, the two indices, and the paired cluster bootstrap (PLAN §1.3, §0.1).

Everything here is numpy only (no torch / datasets), so it runs on the VM or in tests anywhere. The
definitions in this file are the pre-registered ones (bench/jevbench/PREREG.md quotes its sha256).

Answer math, identical for every system:
  * A response's probabilities are aligned to the request's canonical labels (the "main" variant's criteria
    order for choice, level index for score). Missing labels count 0, unknown labels are ignored, and the
    vector is renormalised. An all-zero vector, a missing question or a failed request is a FAILURE.
  * FAILURE (PLAN §1.4: failures count as wrong): the decision is wrong (argmax label -1; noul decision =
    not gold), and probability metrics use the uniform distribution (noul p = 0.5).
  * Argmax ties break by canonical label order (never by the order a shuffled request listed them), so an
    exactly order-invariant model shows 0 flips even after 2-dp rounding.
  * Our model's probabilities are rounded to 2 dp, as our API returns them and Jev does, before scoring.
  * NLL floors probabilities at 0.005 (Jev rounds to 0.01 and often returns exact zeros).
  * Brier is the multiclass sum over options for every kind (a noul is a 2-option question).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

NLL_FLOOR = 0.005
SUM_TOL = 0.02  # renormalise silently within this tolerance; count larger deviations


# ---------------------------------------------------------------------------------------------- containers


@dataclass
class Head:
    """All decisions of one question id (one head) of one dataset, for one system."""

    qid: str
    kind: str  # "choice" | "noul" | "score"
    labels: list[str]  # choice keys (canonical order) / score "0".."K-1" / noul ["false","true"]
    item: np.ndarray  # [n] item index of each decision (bootstrap cluster)
    gold: np.ndarray  # [n] int: class index / level / 0-1
    target: np.ndarray  # [n, K] target distribution (one-hot, or soft gold)
    value: np.ndarray | None  # [n] continuous gold for score (Spearman, MAE)
    prob: np.ndarray  # [n, K] predicted distribution (uniform where failed)
    failed: np.ndarray  # [n] bool

    @property
    def K(self) -> int:
        return self.prob.shape[1]

    def take(self, sel: np.ndarray) -> "Head":
        return Head(self.qid, self.kind, self.labels, self.item[sel], self.gold[sel], self.target[sel],
                    None if self.value is None else self.value[sel], self.prob[sel], self.failed[sel])

    def pred(self) -> np.ndarray:
        """Argmax label index (canonical tie-break); -1 where failed. noul: 1 if p_yes >= 0.5."""
        if self.kind == "noul":
            p = (self.prob[:, 1] >= 0.5).astype(int)
            return np.where(self.failed, 1 - self.gold, p)
        return np.where(self.failed, -1, self.prob.argmax(axis=1))

    def correct(self) -> np.ndarray:
        return self.pred() == self.gold


@dataclass
class DatasetPreds:
    key: str
    items: list[str]
    strata: list[str]
    heads: dict[str, Head]
    n_requests: int = 0
    n_failed_requests: int = 0
    renormalised: int = 0  # vectors whose raw sum was off by more than SUM_TOL
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------- parsing


def canonical_labels(question: dict) -> list[str]:
    t = question["type"]
    if t == "choice":
        return list(question["criteria"])
    if t == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return ["false", "true"]


def answer_vector(question: dict, labels: list[str], answer: dict | None, round_dp: int | None) -> tuple[np.ndarray | None, bool]:
    """-> (probability vector over `labels` or None for failure, renormalised-beyond-tolerance flag)."""
    if not answer or answer.get("type") != question["type"]:
        return None, False
    t = question["type"]
    if t == "noul":
        p = answer.get("noul")
        if p is None or not isinstance(p, (int, float)) or not math.isfinite(p):
            return None, False
        p = float(p)
        if round_dp is not None:
            p = round(p, round_dp)
        p = min(1.0, max(0.0, p))
        return np.array([1.0 - p, p]), False
    probs = answer.get("probabilities") or {}
    v = np.array([float(probs.get(lab, 0.0) or 0.0) for lab in labels], dtype=np.float64)
    if round_dp is not None:
        v = np.round(v, round_dp)
    v = np.clip(np.nan_to_num(v, nan=0.0), 0.0, None)
    s = v.sum()
    if s <= 0:
        return None, False
    return v / s, abs(s - 1.0) > SUM_TOL


def gold_arrays(question: dict, labels: list[str], gold: dict) -> tuple[int, np.ndarray, float | None]:
    K = len(labels)
    t = question["type"]
    tgt = np.zeros(K)
    if t == "noul":
        g = int(bool(gold["label"]))
        if "p" in gold:
            tgt[:] = [1 - gold["p"], gold["p"]]
        else:
            tgt[g] = 1.0
        return g, tgt, None
    if t == "choice":
        g = labels.index(gold["label"])
        if "dist" in gold:
            for k, p in gold["dist"].items():
                if k in labels:
                    tgt[labels.index(k)] = p
            tgt = tgt / tgt.sum() if tgt.sum() > 0 else np.eye(K)[g]
        else:
            tgt[g] = 1.0
        return g, tgt, None
    g = int(gold["label"])
    if "dist" in gold:
        tgt[: len(gold["dist"])] = gold["dist"]
        tgt = tgt / tgt.sum() if tgt.sum() > 0 else np.eye(K)[g]
    else:
        tgt[g] = 1.0
    return g, tgt, float(gold.get("value", g))


def main_label_order(main_rows: Iterable[dict]) -> dict[str, dict[str, list[str]]]:
    """item -> qid -> canonical labels of the main variant (shuffled variants are aligned to this order)."""
    return {r["item"]: {q: canonical_labels(v) for q, v in r["request"]["questions"].items()} for r in main_rows}


def collect(rows: Iterable[dict], answers: dict[str, dict], round_dp: int | None = 2,
            head_key: Callable[[dict, str], str] | None = None,
            label_order: dict[str, dict[str, list[str]]] | None = None) -> DatasetPreds:
    """rows: request rows of ONE dataset variant (as written by the builder); answers: id -> response record
    ({"ok": bool, "answers": {...}}). Heads group by qid (typed-decisions: by `stratum/qid`). Pass
    `label_order` (from `main_label_order`) when scoring a shuffled variant, so every variant shares the main
    variant's canonical label order and argmax tie-breaks."""
    acc: dict[str, dict[str, list]] = {}
    items: list[str] = []
    strata: list[str] = []
    item_ix: dict[str, int] = {}
    key = ""
    n_req = n_fail = renorm = 0
    for r in rows:
        key = r["dataset"]
        if r["item"] not in item_ix:
            item_ix[r["item"]] = len(items)
            items.append(r["item"])
            strata.append(r["stratum"])
        ii = item_ix[r["item"]]
        rec = answers.get(r["id"])
        n_req += 1
        ok = bool(rec and rec.get("ok"))
        n_fail += not ok
        ans = (rec or {}).get("answers") or {}
        for qid, q in r["request"]["questions"].items():
            if qid not in r["gold"]:
                continue
            hk = head_key(r, qid) if head_key else qid
            labels = label_order[r["item"]][qid] if label_order else canonical_labels(q)
            if set(labels) != set(canonical_labels(q)):
                raise ValueError(f"{key}/{qid}: variant options differ from the main variant")
            g, tgt, val = gold_arrays(q, labels, r["gold"][qid])
            vec, off = answer_vector(q, labels, ans.get(qid), round_dp) if ok else (None, False)
            renorm += off
            h = acc.setdefault(hk, {"kind": q["type"], "labels": labels, "item": [], "gold": [], "target": [],
                                    "value": [], "prob": [], "failed": []})
            if h["labels"] != labels:
                raise ValueError(f"{key}/{hk}: label set changed between items")
            h["item"].append(ii)
            h["gold"].append(g)
            h["target"].append(tgt)
            h["value"].append(val)
            h["prob"].append(vec if vec is not None else np.full(len(labels), 1.0 / len(labels)))
            h["failed"].append(vec is None)
    heads = {}
    for hk, h in acc.items():
        val = None if h["kind"] != "score" else np.array(h["value"], dtype=np.float64)
        heads[hk] = Head(hk, h["kind"], h["labels"], np.array(h["item"]), np.array(h["gold"]), np.array(h["target"]),
                         val, np.array(h["prob"]), np.array(h["failed"], dtype=bool))
    return DatasetPreds(key, items, strata, heads, n_req, n_fail, renorm)


# ---------------------------------------------------------------------------------------------- primitives


def rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties share the mean rank (as scipy.stats.rankdata 'average')."""
    a = np.asarray(a, dtype=np.float64)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), dtype=np.intp)
    inv[sorter] = np.arange(len(a))
    s = a[sorter]
    obs = np.r_[True, s[1:] != s[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / den) if den > 0 else 0.0


def f1_binary(pred: np.ndarray, gold: np.ndarray) -> float:
    tp = int(np.sum((pred == 1) & (gold == 1)))
    fp = int(np.sum((pred == 1) & (gold == 0)))
    fn = int(np.sum((pred == 0) & (gold == 1)))
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def average_precision(score: np.ndarray, gold: np.ndarray) -> float:
    """Step-wise AP over distinct score thresholds (sklearn's average_precision_score)."""
    P = int(gold.sum())
    if P == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    s, g = score[order], gold[order]
    tp = np.cumsum(g)
    fp = np.cumsum(1 - g)
    last = np.r_[s[1:] != s[:-1], True]  # end of each tie group
    tp, fp = tp[last], fp[last]
    prec = tp / (tp + fp)
    rec = tp / P
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def auroc(score: np.ndarray, gold: np.ndarray) -> float:
    P, N = int(gold.sum()), int(len(gold) - gold.sum())
    if P == 0 or N == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[gold == 1].sum() - P * (P + 1) / 2) / (P * N))


def brier(prob: np.ndarray, target: np.ndarray) -> np.ndarray:
    return ((prob - target) ** 2).sum(axis=1)


def nll(prob: np.ndarray, target: np.ndarray) -> np.ndarray:
    return -(target * np.log(np.maximum(prob, NLL_FLOOR))).sum(axis=1)


def kl(prob: np.ndarray, target: np.ndarray) -> np.ndarray:
    t = np.where(target > 0, target, 1.0)
    return (target * (np.log(t) - np.log(np.maximum(prob, NLL_FLOOR)))).sum(axis=1)


def ece(conf: np.ndarray, correct: np.ndarray, bins: int) -> float:
    if len(conf) == 0:
        return float("nan")
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.sum() * abs(conf[m].mean() - correct[m].mean())
    return float(tot / len(conf))


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    order = np.argsort(-conf, kind="mergesort")
    err = 1 - correct[order].astype(float)
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1))) if len(err) else float("nan")


def coverage_at_risk(conf: np.ndarray, correct: np.ndarray, max_err: float = 0.05) -> tuple[float, float | None]:
    """Largest coverage whose accepted set (conf >= t, ties accepted together) has error <= max_err.
    Thresholds are the realised confidences (test-oracle; Jev is never run on dev, so no frozen threshold)."""
    order = np.argsort(-conf, kind="mergesort")
    c, ok = conf[order], correct[order].astype(float)
    err = np.cumsum(1 - ok) / np.arange(1, len(ok) + 1)
    last = np.r_[c[1:] != c[:-1], True]
    best, thr = 0.0, None
    for i in np.nonzero(last)[0]:
        if err[i] <= max_err:
            best, thr = (i + 1) / len(c), float(c[i])
    return best, thr


def qwk(pred: np.ndarray, gold: np.ndarray, K: int) -> float:
    pred = np.where(pred < 0, 0, pred)  # failures sit at level 0 (still wrong unless gold is 0... count as is)
    O = np.zeros((K, K))
    np.add.at(O, (gold, pred), 1)
    W = np.array([[(i - j) ** 2 for j in range(K)] for i in range(K)]) / (K - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / max(O.sum(), 1)
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else 0.0


def rps(prob: np.ndarray, target: np.ndarray) -> np.ndarray:
    K = prob.shape[1]
    return ((np.cumsum(prob, 1) - np.cumsum(target, 1)) ** 2).sum(1) / (K - 1)


# ---------------------------------------------------------------------------------------------- per head


def head_metrics(h: Head) -> dict[str, float]:
    """Everything we report for one head (not resampled)."""
    out: dict[str, float] = {"n": int(len(h.gold)), "failed": int(h.failed.sum())}
    corr = h.correct()
    out["acc"] = float(corr.mean())
    out["brier"] = float(brier(h.prob, h.target).mean())
    out["nll"] = float(nll(h.prob, h.target).mean())
    if h.kind == "noul":
        p = h.prob[:, 1]
        pred = h.pred()
        conf = np.maximum(p, 1 - p)
        out["f1_pos"] = f1_binary(pred, h.gold)
        out["auroc"] = auroc(np.where(h.failed, 0.5, p), h.gold)
        out["auprc"] = average_precision(np.where(h.failed, 0.5, p), h.gold)
        band = np.where(p >= 0.8, 1, np.where(p <= 0.2, 0, -1))
        out["band_acc"] = float(np.mean(np.where(h.failed, False, band == h.gold)))
        out["prevalence"] = float(h.gold.mean())
    else:
        conf = h.prob.max(1)
        if h.kind == "choice":
            out["macro_f1"] = macro_f1(h.pred(), h.gold, h.K)
    out["ece15"] = ece(conf, corr, 15)
    out["ece10"] = ece(conf, corr, 10)
    out["aurc"] = aurc(conf, corr)
    out["cov_at_5pct_err"], _ = coverage_at_risk(conf, corr)
    if h.kind == "score":
        ev = (h.prob * np.arange(h.K)).sum(1)
        out["acc_mode"] = out["acc"]
        out["spearman"] = spearman(ev, h.value)
        out["mae"] = float(np.abs(ev - h.value).mean())
        out["rps"] = float(rps(h.prob, h.target).mean())
        out["qwk"] = qwk(h.pred(), h.gold, h.K)
    if not np.all((h.target == 0) | (h.target == 1)):
        out["kl"] = float(kl(h.prob, h.target).mean())
    return out


def macro_f1(pred: np.ndarray, gold: np.ndarray, K: int) -> float:
    f = []
    for k in range(K):
        if (gold == k).any() or (pred == k).any():
            f.append(f1_binary((pred == k).astype(int), (gold == k).astype(int)))
    return float(np.mean(f)) if f else 0.0


# ---------------------------------------------------------------------------------------------- primary metrics


@dataclass(frozen=True)
class Primary:
    name: str
    fn: Callable[[dict[str, Head]], float]  # heads (possibly resampled) -> metric
    chance: Callable[[dict[str, Head]], float]  # full-data heads -> chance level (computed once)


def _one(heads: dict[str, Head], qid: str = "answer") -> Head:
    return heads[qid] if qid in heads else next(iter(heads.values()))


def _acc(heads):
    return float(_one(heads).correct().mean())


def _chance_k(heads):
    return 1.0 / _one(heads).K


def _spearman1(heads):
    h = _one(heads)
    return spearman((h.prob * np.arange(h.K)).sum(1), h.value)


def _macro_f1_heads(heads):
    return float(np.mean([f1_binary(h.pred(), h.gold) for h in heads.values()]))


def _chance_f1_heads(heads):
    return float(np.mean([_trivial_f1(h.gold.mean()) for h in heads.values()]))


def _trivial_f1(pi: float) -> float:
    """F1 of the all-positive predictor, 2*pi/(1+pi): the best input-blind F1 (our 'prior-F1')."""
    return 2 * pi / (1 + pi) if pi > 0 else 0.0


def _f1_toxic(heads):
    h = heads["toxic"]
    return f1_binary(h.pred(), h.gold)


def _chance_f1_toxic(heads):
    return _trivial_f1(heads["toxic"].gold.mean())


def _mean_auprc(heads):
    v = [average_precision(np.where(h.failed, 0.5, h.prob[:, 1]), h.gold) for h in heads.values()]
    v = [x for x in v if not math.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def _chance_auprc(heads):
    v = [h.gold.mean() for h in heads.values() if h.gold.sum() > 0]
    return float(np.mean(v)) if v else 0.0


def _micro_f1(heads):
    pred = np.concatenate([h.pred() for h in heads.values()])
    gold = np.concatenate([h.gold for h in heads.values()])
    return f1_binary(pred, gold)


def _chance_micro_f1(heads):
    return _trivial_f1(float(np.concatenate([h.gold for h in heads.values()]).mean()))


def _mean_spearman(heads):
    return float(np.mean([spearman((h.prob * np.arange(h.K)).sum(1), h.value) for h in heads.values()]))


def _pooled_acc(heads):
    c = np.concatenate([h.correct() for h in heads.values()])
    return float(c.mean())


def _chance_pooled(heads):
    return float(np.mean(np.concatenate([np.full(len(h.gold), 1.0 / h.K) for h in heads.values()])))


def _balanced_acc(heads):
    h = _one(heads)
    pred = h.pred()
    return float(np.mean([np.mean(pred[h.gold == c] == c) for c in (0, 1) if (h.gold == c).any()]))


def _zero(heads):
    return 0.0


def _half(heads):
    return 0.5


PRIMARY: dict[str, Primary] = {
    "acc": Primary("acc", _acc, _chance_k),
    "acc_noul": Primary("acc", _acc, _half),
    "acc_mode": Primary("acc_mode", _acc, _chance_k),
    "spearman": Primary("spearman", _spearman1, _zero),
    "macro_f1": Primary("macro_f1", _macro_f1_heads, _chance_f1_heads),
    "f1_pos": Primary("f1_toxic", _f1_toxic, _chance_f1_toxic),
    "mean_auprc": Primary("mean_auprc", _mean_auprc, _chance_auprc),
    "micro_f1": Primary("micro_f1", _micro_f1, _chance_micro_f1),
    "mean_spearman": Primary("mean_spearman", _mean_spearman, _zero),
    "acc_pooled": Primary("acc", _pooled_acc, _chance_pooled),
    "balanced_acc": Primary("balanced_acc", _balanced_acc, _half),
}


def primary_for(key: str, metric: str, kinds: tuple[str, ...]) -> Primary:
    if key in ("typed_decisions", "mmlu_pro") or metric == "soft_acc":  # heads keyed by workflow / K / item: pool
        return PRIMARY["acc_pooled"]
    if metric == "acc" and kinds and kinds[0] == "noul":
        return PRIMARY["acc_noul"]
    return PRIMARY[metric]


def skill(m: float, chance: float) -> float:
    if math.isnan(m):
        return 0.0
    return float(min(1.0, max(0.0, (m - chance) / (1.0 - chance)))) if chance < 1 else 0.0


# ---------------------------------------------------------------------------------------------- decision score


def prior_brier(heads_full: dict[str, Head]) -> dict[str, float]:
    """Per head: the class-frequency (mean target) distribution of the full test set, as a fixed predictor.
    Returned as the head's prior vector; Brier_prior per decision is computed against it."""
    return {k: h.target.mean(0) for k, h in heads_full.items()}


def decision_score(heads: dict[str, Head], priors: dict[str, np.ndarray]) -> float:
    """1 - mean Brier / mean Brier(prior), pooled over all decisions of the dataset; clipped to [-1, 1]."""
    b = np.concatenate([brier(h.prob, h.target) for h in heads.values()])
    bp = np.concatenate([brier(np.broadcast_to(priors[k], h.target.shape), h.target) for k, h in heads.items()])
    den = bp.mean()
    return float(np.clip(1 - b.mean() / den, -1, 1)) if den > 0 else 0.0


# ---------------------------------------------------------------------------------------------- bootstrap


def resample_heads(heads: dict[str, Head], item_sample: np.ndarray, dec_of_item: dict[str, np.ndarray]) -> dict[str, Head]:
    out = {}
    for k, h in heads.items():
        d = dec_of_item[k][item_sample]
        out[k] = h.take(d[d >= 0])
    return out


def dec_index(heads: dict[str, Head], n_items: int) -> dict[str, np.ndarray]:
    """item -> decision row per head (-1 where the item has no decision for that head)."""
    out = {}
    for k, h in heads.items():
        a = np.full(n_items, -1, dtype=np.int64)
        a[h.item] = np.arange(len(h.item))
        out[k] = a
    return out


def strata_index(strata: list[str]) -> list[np.ndarray]:
    by: dict[str, list[int]] = {}
    for i, s in enumerate(strata):
        by.setdefault(s, []).append(i)
    return [np.array(v) for _, v in sorted(by.items())]


def stratified_sample(groups: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    return np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])


@dataclass
class DatasetScore:
    key: str
    area: str
    primary: str
    chance: float
    metric: dict[str, float]  # system -> primary metric (full data)
    skill: dict[str, float]
    ds: dict[str, float]  # decision score
    boot_metric: dict[str, np.ndarray] = field(default_factory=dict)
    boot_skill: dict[str, np.ndarray] = field(default_factory=dict)
    boot_ds: dict[str, np.ndarray] = field(default_factory=dict)


def score_dataset(key: str, area: str, prim: Primary, preds: dict[str, DatasetPreds], B: int,
                  rng: np.random.Generator) -> DatasetScore:
    """Paired, item-clustered, target-stratified bootstrap over the systems in `preds` (same item order)."""
    systems = list(preds)
    base = preds[systems[0]]
    for s in systems[1:]:
        if preds[s].items != base.items:
            raise ValueError(f"{key}: systems disagree on items")
    chance = prim.chance(base.heads)
    priors = prior_brier(base.heads)
    res = DatasetScore(key, area, prim.name, chance, {}, {}, {})
    for s in systems:
        m = prim.fn(preds[s].heads)
        res.metric[s], res.skill[s] = m, skill(m, chance)
        res.ds[s] = decision_score(preds[s].heads, priors)
        res.boot_metric[s], res.boot_skill[s], res.boot_ds[s] = (np.empty(B) for _ in range(3))
    groups = strata_index(base.strata)
    dix = {s: dec_index(preds[s].heads, len(base.items)) for s in systems}
    for b in range(B):
        samp = stratified_sample(groups, rng)
        for s in systems:
            hs = resample_heads(preds[s].heads, samp, dix[s])
            m = prim.fn(hs)
            res.boot_metric[s][b] = m
            res.boot_skill[s][b] = skill(m, chance)
            res.boot_ds[s][b] = decision_score(hs, priors)
    return res


def index(scores: list[DatasetScore], system: str, field_: str = "skill", boot: bool = False) -> Any:
    """100 x mean over areas of the mean per-dataset value (skill or ds). boot=True returns the [B] array."""
    by_area: dict[str, list] = {}
    for sc in scores:
        v = getattr(sc, ("boot_" if boot else "") + field_)[system]
        by_area.setdefault(sc.area, []).append(v)
    area_means = [np.mean(np.stack(v), axis=0) if boot else float(np.mean(v)) for v in by_area.values()]
    return 100 * (np.mean(np.stack(area_means), axis=0) if boot else float(np.mean(area_means)))


def area_table(scores: list[DatasetScore], systems: list[str], field_: str = "skill") -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for sc in scores:
        for s in systems:
            out.setdefault(sc.area, {}).setdefault(s, []).append(getattr(sc, field_)[s])  # type: ignore[union-attr]
    return {a: {s: 100 * float(np.mean(v)) for s, v in d.items()} for a, d in out.items()}


def ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    lo, hi = np.percentile(x, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


def decision_rule(scores: list[DatasetScore], ours: str, jev: str) -> dict[str, Any]:
    """PLAN §0.1: (a) both index-difference CIs exclude 0 (lower bound > 0), (b) >= 13 dataset wins on the
    primary metric (point estimate, strictly greater), (c) no area trails by more than 10 skill points."""
    d_skill = index(scores, ours, "skill", True) - index(scores, jev, "skill", True)
    d_ds = index(scores, ours, "ds", True) - index(scores, jev, "ds", True)
    wins = [sc.key for sc in scores if sc.metric[ours] > sc.metric[jev]]
    areas = area_table(scores, [ours, jev])
    worst = min(v[ours] - v[jev] for v in areas.values())
    a = ci(d_skill)[0] > 0 and ci(d_ds)[0] > 0
    b = len(wins) >= 13
    c = worst >= -10
    return {"a_ci_excludes_0": a, "b_wins": len(wins), "b_ok": b, "c_worst_area_gap": worst, "c_ok": c,
            "claim": bool(a and b and c), "skill_diff_ci": ci(d_skill), "ds_diff_ci": ci(d_ds), "wins": wins}


# ---------------------------------------------------------------------------------------------- robustness


def flip_rate(main: DatasetPreds, shuf: DatasetPreds) -> dict[str, float]:
    """Order robustness: share of items whose argmax label differs between the main order and the seeded
    shuffle (both aligned to canonical labels), plus the mean total-variation distance of the two dists."""
    flips, tv, n = 0, 0.0, 0
    for k, h in main.heads.items():
        if h.kind != "choice" or k not in shuf.heads:
            continue
        g = shuf.heads[k]
        pos = {it: j for j, it in enumerate(g.item)}
        for i, it in enumerate(h.item):
            j = pos.get(it)
            if j is None or h.failed[i] or g.failed[j]:
                continue
            n += 1
            flips += int(h.prob[i].argmax() != g.prob[j].argmax())
            tv += 0.5 * float(np.abs(h.prob[i] - g.prob[j]).sum())
    return {"n": n, "flip_rate": flips / n if n else float("nan"), "mean_tv": tv / n if n else float("nan")}
