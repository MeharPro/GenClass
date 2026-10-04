"""`typed_decisions_card@d0e2f0c4`: LocalLLaMA/typed-decisions, whole case per request (specs §8; PLAN §6).

Protocol: each row's `state` and `questions` (JSON strings) are exactly the request body; one request per
case carrying all five questions. Gold is the mean of three teacher samples (`gold`). The card's scorer is
not public, so `pin_scorer` enumerates the definitional variants (tie rule, noul threshold, KL epsilon,
Brier normalisation, ECE bins, decision- vs case-averaging, prior source) and keeps the one that reproduces
the card's **Prior** (acc .470, KL .347, Brier .189, ECE .088) and **Uniform** (.308, .444, .238, .169) rows to
3 decimals. Our numbers are only scored once a variant matches (or with `--allow-unpinned`, flagged).

Accuracy = argmax vs the gold label; KL = KL(gold ‖ model); Brier = multiclass squared error; ECE = top-label.
Overflow default `truncate` (allowed here, counted). Z: the model must never have seen these workflows; S: the
`train` split is allowed and the result goes in the card's fitted table (PLAN §2.5). An S checkpoint trained on
this `train` is barred from TypeSafe WorkflowEvals (same workflow names).
"""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from jev_local.bench.benchmax.runner import DEFAULT_WORK, Adapter, RunContext, Session, sha256_file

HF_ID = "LocalLLaMA/typed-decisions"
PINNED_REVISION = "d0e2f0c42fef86cc15d1688d25a19f5ba7c85b18"
LICENCE = "Apache-2.0"
WORKFLOWS = ("agent_trace_observability", "customer_service", "invoice_processing", "security_incidents")
EXPECTED_ROWS = {"train": 1200, "test": 400}
REFERENCE_ROWS = {
    "prior": {"accuracy": 0.470, "kl": 0.347, "brier": 0.189, "ece": 0.088},
    "uniform": {"accuracy": 0.308, "kl": 0.444, "brier": 0.238, "ece": 0.169},
}
JEV_CARD = {"accuracy": 0.727, "kl": 1.442, "brier": 0.148, "ece": 0.144, "by_type": {"noul": 0.775, "choice": 0.720, "score": 0.696},
            "p50_ms": 710, "measured": "2026-09-18 via jev-latest -> jev-1.13.0"}
CEILINGS = {"teacher_self_agreement": 0.735, "perfect_factor_recovery": 0.704}
METRICS = ("accuracy", "kl", "brier", "ece")


@dataclass(frozen=True)
class ScorerConfig:
    tie: str = "first"  # argmax ties: "first" | "last" option / level
    noul_rule: str = "ge"  # predicted true when p >= 0.5 ("ge") or p > 0.5 ("gt")
    kl_eps: float = 1e-6  # model probabilities are clipped below at eps inside the log
    brier: str = "sum"  # Σ_i (m_i − g_i)² ("sum") or its mean over classes ("mean")
    noul_brier: str = "two_class"  # noul as a 2-class distribution ("two_class") or (p − g)² ("binary")
    ece_bins: int = 10
    ece_mode: str = "floor"  # bin = min(B−1, floor(c·B)) ("floor") or (lo, hi] edges ("right_closed")
    ece_conf: str = "max"  # confidence = max probability ("max") or the Jev wire `confidence` formulas ("wire")
    average: str = "decision"  # mean over the 2,000 decisions ("decision") or over the 400 cases ("case")
    prior_source: str = "soft"  # Prior = train label frequencies ("hard") or mean train gold distribution ("soft")
    prior_key: str = "qid"  # Prior pooled per question id across workflows ("qid") or per (workflow, qid) ("wf_qid")

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "ScorerConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})  # type: ignore[attr-defined]


VARIANTS: dict[str, Sequence[Any]] = {
    "tie": ("first", "last"),
    "noul_rule": ("ge", "gt"),
    "kl_eps": (1e-3, 1e-6, 1e-12),
    "brier": ("sum", "mean"),
    "noul_brier": ("two_class", "binary"),
    "ece_bins": (10, 15),
    "ece_mode": ("floor", "right_closed"),
    "ece_conf": ("max", "wire"),
    "average": ("decision", "case"),
    "prior_source": ("hard", "soft"),
    "prior_key": ("qid", "wf_qid"),
}
# What the two reference rows can and cannot identify (measured on the real data, 2026-10-03, see the report):
# KL and Brier are tie-free and reproduce to 4 dp with prior_source="soft", prior_key="qid", brier="sum",
# noul_brier="two_class"; accuracy and ECE depend on how argmax ties are broken (the Uniform row has every
# option tied) and the card's values fall between every deterministic rule, consistent with random tie-breaking.
COMPARABLE_METRICS = ("kl", "brier")


@dataclass
class Case:
    id: str
    workflow: str
    state: Any
    questions: dict[str, dict]
    gold: dict[str, dict]


def parse_row(row: dict[str, Any]) -> Case:
    """A dataset row -> Case. `state`/`questions`/`gold` are JSON strings on the Hub."""
    load = lambda v: json.loads(v) if isinstance(v, str) else v  # noqa: E731
    return Case(str(row["id"]), str(row["workflow"]), load(row["state"]), load(row["questions"]), load(row["gold"]))


# ---------------------------------------------------------------------- distributions


def labels_of(q: dict[str, Any]) -> list[str]:
    t = q["type"]
    if t == "choice":
        return list(q["criteria"].keys())
    if t == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return ["false", "true"]


def gold_dist(q: dict[str, Any], g: dict[str, Any]) -> tuple[list[float], int]:
    """(gold distribution over labels_of(q), gold label index) from the row's `gold` entry."""
    labs = labels_of(q)
    t = q["type"]
    if t == "noul":
        p = float(g.get("noul", g.get("probability_true", 0.5)))
        lab = g.get("label")
        idx = 1 if (lab is True or str(lab).lower() == "true") else 0
        return [1.0 - p, p], idx
    probs = g.get("probabilities") or {}
    dist = [float(probs.get(lab, 0.0)) for lab in labs]
    s = sum(dist)
    if s > 0:
        dist = [x / s for x in dist]
    lab = g.get("label")
    idx = labs.index(str(lab)) if str(lab) in labs else max(range(len(dist)), key=lambda i: (dist[i], -i))
    return dist, idx


def model_dist(q: dict[str, Any], a: dict[str, Any] | None) -> list[float] | None:
    """Model distribution over labels_of(q) from a wire answer; None when there is no answer."""
    if a is None:
        return None
    labs = labels_of(q)
    if q["type"] == "noul":
        p = min(1.0, max(0.0, float(a["noul"])))
        return [1.0 - p, p]
    probs = a.get("probabilities") or {}
    dist = [max(0.0, float(probs.get(lab, 0.0))) for lab in labs]
    s = sum(dist)
    return [x / s for x in dist] if s > 0 else [1.0 / len(labs)] * len(labs)


def uniform_dist(q: dict[str, Any]) -> list[float]:
    k = len(labels_of(q))
    return [1.0 / k] * k


def predict(kind: str, m: Sequence[float], cfg: ScorerConfig) -> int:
    if kind == "noul":
        p = m[1]
        return 1 if (p >= 0.5 if cfg.noul_rule == "ge" else p > 0.5) else 0
    best = max(m)
    idx = [i for i, x in enumerate(m) if x == best]
    return idx[0] if cfg.tie == "first" else idx[-1]


def confidence_of(kind: str, m: Sequence[float], mode: str) -> float:
    """ECE confidence: the top probability, or the wire `confidence` a System One answer carries
    ((K·pmax − 1)/(K − 1) for choice, the spread formula for score, max(p, 1 − p) for noul)."""
    if mode == "max":
        return max(m)
    if kind == "noul":
        return max(m[1], 1.0 - m[1])
    if kind == "score":
        from jev_local.confidence import score_confidence

        return score_confidence(m, "mode")
    k = len(m)
    return (k * max(m) - 1.0) / (k - 1.0) if k > 1 else 1.0


def kl_div(g: Sequence[float], m: Sequence[float], eps: float) -> float:
    return sum(gi * math.log(gi / max(mi, eps)) for gi, mi in zip(g, m) if gi > 0)


def brier_score(kind: str, g: Sequence[float], m: Sequence[float], cfg: ScorerConfig) -> float:
    if kind == "noul" and cfg.noul_brier == "binary":
        return (m[1] - g[1]) ** 2
    s = sum((mi - gi) ** 2 for gi, mi in zip(g, m))
    return s / len(g) if cfg.brier == "mean" else s


def ece_score(confs: Sequence[float], corrects: Sequence[float], bins: int, mode: str) -> float:
    n = len(confs)
    if n == 0:
        return float("nan")
    sums = [0.0] * bins
    hits = [0.0] * bins
    counts = [0] * bins
    for c, y in zip(confs, corrects):
        if mode == "floor":
            b = min(bins - 1, int(math.floor(c * bins)))
        else:  # (lo, hi] with the first bin closed at 0
            b = min(bins - 1, max(0, int(math.ceil(c * bins)) - 1))
        sums[b] += c
        hits[b] += y
        counts[b] += 1
    return sum(abs(sums[b] / counts[b] - hits[b] / counts[b]) * counts[b] / n for b in range(bins) if counts[b])


# ---------------------------------------------------------------------- scoring


def score_cases(cases: Sequence[Case], answers: Sequence[dict[str, dict] | None], cfg: ScorerConfig,
                missing: str = "uniform") -> dict[str, Any]:
    """Metrics over cases; `answers[i]` maps qid -> wire answer (None = refused/error: wrong, distribution per
    `missing`: "uniform" substitutes the uniform distribution and is disclosed)."""
    per_dec: list[dict[str, float]] = []
    per_case: list[list[dict[str, float]]] = []
    by_type: dict[str, list[float]] = defaultdict(list)
    by_wf: dict[str, list[float]] = defaultdict(list)
    n_missing = 0
    for case, ans in zip(cases, answers):
        rows = []
        for qid, q in case.questions.items():
            g, gidx = gold_dist(q, case.gold[qid])
            a = (ans or {}).get(qid)
            m = model_dist(q, a)
            if m is None:
                n_missing += 1
                m = uniform_dist(q)
                correct = 0.0  # a missing answer is wrong whatever the substitute would have predicted
            else:
                correct = 1.0 if predict(q["type"], m, cfg) == gidx else 0.0
            rec = {"correct": correct, "kl": kl_div(g, m, cfg.kl_eps), "brier": brier_score(q["type"], g, m, cfg),
                   "conf": confidence_of(q["type"], m, cfg.ece_conf)}
            rows.append(rec)
            per_dec.append(rec)
            by_type[q["type"]].append(correct)
            by_wf[case.workflow].append(correct)
        per_case.append(rows)

    def mean(xs: Iterable[float]) -> float:
        xs = list(xs)
        return sum(xs) / len(xs) if xs else float("nan")

    if cfg.average == "decision":
        acc = mean(r["correct"] for r in per_dec)
        kl = mean(r["kl"] for r in per_dec)
        br = mean(r["brier"] for r in per_dec)
    else:
        acc = mean(mean(r["correct"] for r in rows) for rows in per_case)
        kl = mean(mean(r["kl"] for r in rows) for rows in per_case)
        br = mean(mean(r["brier"] for r in rows) for rows in per_case)
    ece = ece_score([r["conf"] for r in per_dec], [r["correct"] for r in per_dec], cfg.ece_bins, cfg.ece_mode)
    return {
        "accuracy": acc, "kl": kl, "brier": br, "ece": ece,
        "n_cases": len(cases), "n_decisions": len(per_dec), "n_missing_answers": n_missing,
        "by_type": {t: mean(v) for t, v in sorted(by_type.items())},
        "by_workflow": {w: mean(v) for w, v in sorted(by_wf.items())},
        "scorer": cfg.to_json(),
    }


def prior_key(case: Case, qid: str, cfg: ScorerConfig) -> tuple[str, str]:
    return (case.workflow, qid) if cfg.prior_key == "wf_qid" else ("*", qid)


def prior_table(train: Sequence[Case], cfg: ScorerConfig) -> dict[tuple[str, str], dict[str, float]]:
    """Prior per question: train label frequencies ("hard") or the mean gold distribution ("soft"), pooled per
    question id across workflows ("qid": `urgency` is one prior for all four workflows) or per (workflow, qid)."""
    acc: dict[tuple[str, str], Counter] = defaultdict(Counter)
    n: Counter = Counter()
    for c in train:
        for qid, q in c.questions.items():
            g, gidx = gold_dist(q, c.gold[qid])
            labs = labels_of(q)
            key = prior_key(c, qid, cfg)
            if cfg.prior_source == "hard":
                acc[key][labs[gidx]] += 1.0
            else:
                for lab, p in zip(labs, g):
                    acc[key][lab] += p
            n[key] += 1
    return {k: {lab: v / n[k] for lab, v in cnt.items()} for k, cnt in acc.items()}


def prior_answers(case: Case, table: dict[tuple[str, str], dict[str, float]], cfg: ScorerConfig | None = None) -> dict[str, dict]:
    cfg = cfg or ScorerConfig()
    out: dict[str, dict] = {}
    for qid, q in case.questions.items():
        labs = labels_of(q)
        pri = table.get(prior_key(case, qid, cfg), {})
        dist = [pri.get(lab, 0.0) for lab in labs]
        s = sum(dist)
        dist = [x / s for x in dist] if s > 0 else uniform_dist(q)
        out[qid] = _answer_from_dist(q, dist)
    return out


def uniform_answers(case: Case) -> dict[str, dict]:
    return {qid: _answer_from_dist(q, uniform_dist(q)) for qid, q in case.questions.items()}


def _answer_from_dist(q: dict[str, Any], dist: Sequence[float]) -> dict[str, Any]:
    labs = labels_of(q)
    if q["type"] == "noul":
        return {"type": "noul", "noul": float(dist[1])}
    best = max(range(len(dist)), key=lambda i: (dist[i], -i))
    if q["type"] == "choice":
        return {"type": "choice", "choice": labs[best], "probabilities": dict(zip(labs, map(float, dist)))}
    return {"type": "score", "score": sum(i * p for i, p in enumerate(dist)), "probabilities": dict(zip(labs, map(float, dist)))}


def matches_reference(metrics: dict[str, float], ref: dict[str, float], dp: int = 3, keys: Sequence[str] = METRICS) -> bool:
    return all(round(metrics[k], dp) == round(ref[k], dp) for k in keys)


def matched_metrics(rows: dict[str, dict[str, float]], reference: dict[str, dict[str, float]], dp: int = 3) -> list[str]:
    """Metrics that reproduce BOTH reference rows to `dp` decimals."""
    return [k for k in METRICS if all(round(rows[r][k], dp) == round(reference[r][k], dp) for r in ("prior", "uniform"))]


def pin_scorer(test: Sequence[Case], train: Sequence[Case], reference: dict[str, dict[str, float]] = REFERENCE_ROWS,
               variants: dict[str, Sequence[Any]] = VARIANTS, required: Sequence[str] = COMPARABLE_METRICS) -> dict[str, Any]:
    """Enumerate the scorer variants against the card's Prior and Uniform rows.

    `pinned` = some variant reproduces all four metrics of both rows to 3 dp. `pinned_metrics` = the metrics
    some variant reproduces for both rows (the chosen config maximises that set, then minimises the L1
    distance). `comparable` = every metric in `required` is pinned (KL and Brier by default): the ones our
    comparison with the card relies on when accuracy / ECE conventions cannot be identified."""
    keys = list(variants)
    full: list[ScorerConfig] = []
    best: tuple[tuple[int, float], ScorerConfig | None, dict[str, Any], list[str]] = ((-1, math.inf), None, {}, [])
    uniform = [uniform_answers(c) for c in test]
    n_tried = 0
    prior_cache: dict[tuple[str, str], list[dict[str, dict]]] = {}
    for combo in itertools.product(*(variants[k] for k in keys)):
        cfg = ScorerConfig(**dict(zip(keys, combo)))
        n_tried += 1
        pk = (cfg.prior_source, cfg.prior_key)
        if pk not in prior_cache:
            table = prior_table(train, cfg)
            prior_cache[pk] = [prior_answers(c, table, cfg) for c in test]
        u = score_cases(test, uniform, cfg)
        p = score_cases(test, prior_cache[pk], cfg)
        rows = {"uniform": {k: u[k] for k in METRICS}, "prior": {k: p[k] for k in METRICS}}
        got = matched_metrics(rows, reference)
        dist = sum(abs(rows[r][k] - reference[r][k]) for r in rows for k in METRICS)
        score = (len(got), -dist)
        if score > (best[0][0], -best[0][1]):
            best = ((len(got), dist), cfg, rows, got)
        if len(got) == len(METRICS):
            full.append(cfg)
    pinned_metrics = best[3]
    chosen = full[0] if full else best[1]
    free = sorted(k for k in keys if len({getattr(m, k) for m in full}) > 1) if full else []
    return {
        "pinned": bool(full), "comparable": all(k in pinned_metrics for k in required), "required_metrics": list(required),
        "pinned_metrics": pinned_metrics, "unpinned_metrics": [k for k in METRICS if k not in pinned_metrics],
        "n_variants": n_tried, "n_matching": len(full),
        "config": chosen.to_json() if chosen else None, "all_matching": [m.to_json() for m in full],
        "free_fields": free,
        "closest": {"config": best[1].to_json() if best[1] else None, "l1_distance": best[0][1], "rows": best[2]},
        "reference": reference,
    }


# ---------------------------------------------------------------------- adapter


class TypedDecisionsAdapter(Adapter):
    default_overflow = "truncate"
    allow_truncate = True
    default_split = "test"

    @classmethod
    def add_arguments(cls, p: Any) -> None:
        p.add_argument("--data-dir", help=f"snapshot of {HF_ID}@{PINNED_REVISION[:8]} (default {DEFAULT_WORK / 'hf-data' / 'typed-decisions'})")
        p.add_argument("--split", choices=["test", "train"], default="test", help="train = self-check only")
        p.add_argument("--workflows", nargs="*", help=f"subset of {WORKFLOWS}")
        p.add_argument("--limit", type=int, help="first N cases (self-check)")
        p.add_argument("--scorer-config", help="JSON of a pinned ScorerConfig (skips the enumeration)")
        p.add_argument("--allow-unpinned", action="store_true", help="score even if no scorer variant reproduces the card's rows (flagged)")
        p.add_argument("--pin-only", action="store_true", help="only reproduce the Prior/Uniform rows")

    @property
    def data_dir(self) -> Path:
        return Path(self.args.data_dir or DEFAULT_WORK / "hf-data" / "typed-decisions").expanduser()

    def _ensure_data(self, ctx: RunContext) -> Path:
        d = self.data_dir
        if not (d / "all" / "train-00000-of-00001.parquet").exists():
            from huggingface_hub import snapshot_download

            snapshot_download(HF_ID, repo_type="dataset", revision=PINNED_REVISION, local_dir=str(d))
            ctx.note(f"downloaded {HF_ID}@{PINNED_REVISION[:8]} to {d}")
        return d

    def load_cases(self, split: str) -> list[Case]:
        import pyarrow.parquet as pq

        p = self.data_dir / "all" / f"{split}-00000-of-00001.parquet"
        rows = pq.read_table(p).to_pylist()
        return [parse_row(r) for r in rows]

    def verify(self, ctx: RunContext) -> dict[str, Any]:
        rep: dict[str, Any] = {"hf_id": HF_ID, "revision": PINNED_REVISION, "licence": LICENCE, "files": {}}
        d = self.data_dir
        if not d.exists():
            rep["present"] = False
            return rep
        for split in ("train", "test"):
            p = d / "all" / f"{split}-00000-of-00001.parquet"
            rep["files"][p.relative_to(d).as_posix()] = sha256_file(p) if p.exists() else None
        try:
            train, test = self.load_cases("train"), self.load_cases("test")
        except Exception as e:
            rep["load_error"] = f"{type(e).__name__}: {e}"
            return rep
        rep["rows"] = {"train": len(train), "test": len(test), "expected": EXPECTED_ROWS,
                       "match": len(train) == EXPECTED_ROWS["train"] and len(test) == EXPECTED_ROWS["test"]}
        rep["decisions"] = {"test": sum(len(c.questions) for c in test)}
        if self.args.scorer_config:
            cfg = ScorerConfig.from_json(json.loads(Path(self.args.scorer_config).read_text()))
            table = prior_table(train, cfg)
            rows = {"prior": score_cases(test, [prior_answers(c, table, cfg) for c in test], cfg),
                    "uniform": score_cases(test, [uniform_answers(c) for c in test], cfg)}
            rows = {k: {m: v[m] for m in METRICS} for k, v in rows.items()}
            got = matched_metrics(rows, REFERENCE_ROWS)
            rep["pin"] = {"pinned": len(got) == len(METRICS), "comparable": all(k in got for k in COMPARABLE_METRICS),
                          "required_metrics": list(COMPARABLE_METRICS), "pinned_metrics": got, "unpinned_metrics": [k for k in METRICS if k not in got],
                          "config": cfg.to_json(), "closest": {"config": cfg.to_json(), "rows": rows}, "reference": REFERENCE_ROWS}
        else:
            rep["pin"] = pin_scorer(test, train)
        ctx.dataset_revisions[HF_ID] = {"pinned": PINNED_REVISION}
        rep["ok"] = bool(rep["rows"]["match"] and rep["pin"]["comparable"])
        rep["fully_pinned"] = bool(rep["pin"]["pinned"])
        return rep

    def run(self, session: Session, ctx: RunContext) -> dict[str, Any]:
        self._ensure_data(ctx)
        ver = self.verify(ctx)
        ctx.harness = ver
        if self.args.pin_only:
            return {"pin": ver.get("pin"), "rows": ver.get("rows")}
        pin = ver.get("pin") or {}
        if not pin.get("comparable"):
            if not self.args.allow_unpinned:
                raise SystemExit("no scorer variant reproduces the card's Prior/Uniform KL and Brier; refusing to score (see run.json harness.pin.closest, or pass --allow-unpinned)")
            ctx.note("SCORER NOT PINNED: numbers are not comparable with the card (--allow-unpinned)")
        elif not pin.get("pinned"):
            ctx.note(f"scorer pinned for {pin.get('pinned_metrics')} only; {pin.get('unpinned_metrics')} follow the closest variant "
                     "(the card's tie-breaking is not identifiable: those columns are indicative)")
        cfg = ScorerConfig.from_json(pin["config"]) if pin.get("config") else ScorerConfig.from_json(pin["closest"]["config"])
        cases = self.load_cases(self.args.split)
        if self.args.workflows:
            cases = [c for c in cases if c.workflow in set(self.args.workflows)]
        if self.args.limit:
            cases = cases[: self.args.limit]
        answers: list[dict[str, dict] | None] = []
        preds_path = ctx.out / "predictions.jsonl"
        for c in cases:
            o = session.ask(c.state, c.questions, {"id": c.id, "workflow": c.workflow})
            ans = o.response["answers"] if o.ok else None
            answers.append(ans)
            with preds_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"id": c.id, "workflow": c.workflow, "status": o.status, "answers": ans, "truncated": o.truncated,
                                    "latency_s": round(o.latency_s, 4)}, ensure_ascii=False) + "\n")
        ours = score_cases(cases, answers, cfg)
        res: dict[str, Any] = {"split": self.args.split, "n_cases": len(cases), "ours": ours, "pinned": pin.get("pinned"),
                               "comparable_metrics": pin.get("pinned_metrics"), "indicative_metrics": pin.get("unpinned_metrics"),
                               "scorer": cfg.to_json(), "jev_card": JEV_CARD, "reference_rows": REFERENCE_ROWS, "ceilings": CEILINGS}
        if self.args.split == "test" and not self.args.workflows and not self.args.limit:
            res["delta_vs_jev"] = {k: ours[k] - JEV_CARD[k] for k in METRICS}
            res["table"] = "fitted" if ctx.args.get("track") == "S" else "zero-shot"
        else:
            res["note"] = "subset / train split: self-check only, not comparable with the card"
        return res
