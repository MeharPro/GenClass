"""`jevbench_hf_praveenrajus_v0.1.1`: Praveenrajus/jev-bench (HF), one request per test record (PLAN §6).

Protocol, as the card and the Jevify scorer it cites (`uspraveen/Jevify@a1308666`, `jevify/bench/record.py`,
`jevify/runners/jev_api.py`, `jevify/bench/metrics.py`):
- a record is `(state, question, label, soft_label)` with `state`/`question`/`soft_label` stored as JSON strings;
- the request is `{"state": state, "model": …, "questions": {"q": question}}`, one per record;
- accuracy: choice/score = argmax over the option keys in criteria order (first max) vs the label's index,
  noul = (p ≥ 0.5) vs the 0/1 label; top-label ECE with 15 equal-width (lo, hi] bins; Brier (sum of squares;
  noul (p − y)²); NLL; TVD to the human distribution where `soft_label` exists;
- the headline is the macro accuracy over the 22 configs (Jev .733), with per-primitive means.

The dataset has no git tag for v0.1.1: `manifest.json` `version` is checked, and the revision is pinned to the
commit that carries it. Counts per config are asserted against the manifest and targets.json (1,000 / 2,000 /
1,599 / 800 / 687). Overflow is `refuse` (refusals count as wrong). Self-checks use the validation configs;
`chaosnli` has no validation split and is test-only.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from jev_local.bench.benchmax.runner import DEFAULT_WORK, Adapter, RunContext, Session, load_targets

HF_ID = "Praveenrajus/jev-bench"
PINNED_REVISION = "c37b0f6ab1687376ec9ae8dbd9f343163a68252b"  # main on 2026-10-03; manifest version 0.1.1
VERSION = "0.1.1"
LICENCE = "other (mixed, per manifest.json)"
JEVIFY_SCORER_COMMIT = "a1308666b5460291772161260f43ed95afa65815"
CONFIGS = (
    "banking77", "clinc150", "massive", "ledgar", "go_emotions", "mmlu", "arc_challenge", "mnli", "chaosnli",
    "sst5", "yelp5", "helpsteer2_helpfulness", "helpsteer2_verbosity", "stsb", "measuring_hate_speech",
    "boolq", "fever_evidence", "paws", "civil_comments", "sms_spam", "strategyqa_closed", "strategyqa_grounded",
)
TEST_ONLY = ("chaosnli",)
EXPECTED_TEST_N = {c: 1000 for c in CONFIGS}
EXPECTED_TEST_N.update({"civil_comments": 2000, "chaosnli": 1599, "sms_spam": 800, "strategyqa_closed": 687, "strategyqa_grounded": 687})
# Jev 1.13.0 per-config accuracy from the card's baseline table (evaluation-only reference; the card's numbers).
JEV_CARD_ACC = {
    "arc_challenge": 0.979, "banking77": 0.796, "boolq": 0.917, "chaosnli": 0.615, "civil_comments": 0.729, "clinc150": 0.893,
    "fever_evidence": 0.972, "go_emotions": 0.282, "helpsteer2_helpfulness": 0.363, "helpsteer2_verbosity": 0.341, "ledgar": 0.751,
    "massive": 0.808, "measuring_hate_speech": 0.527, "mmlu": 0.923, "mnli": 0.883, "paws": 0.846, "sms_spam": 0.965, "sst5": 0.565,
    "strategyqa_closed": 0.785, "strategyqa_grounded": 0.956, "stsb": 0.538, "yelp5": 0.685,
}
JEV_CARD_MACRO = {"macro_accuracy": 0.733, "macro_ece": 0.113, "macro_brier": 0.349, "choice_acc": 0.770, "score_acc": 0.503, "noul_acc": 0.881, "tvd_to_human": 0.432}
ECE_BINS = 15
EPS = 1e-12


@dataclass
class Record:
    id: str
    source: str
    primitive: str
    split: str
    state: Any
    question: dict[str, Any]
    label: str | int
    soft_label: Any
    meta: dict[str, Any]

    def option_keys(self) -> list[str]:
        if self.primitive == "choice":
            return list(self.question["criteria"].keys())
        if self.primitive == "score":
            return [str(i) for i in range(len(self.question["criteria"]))]
        return ["0", "1"]

    def label_index(self) -> int:
        return self.option_keys().index(str(self.label))


def parse_row(row: dict[str, Any]) -> Record:
    soft = row.get("soft_label")
    label: str | int = row["label"]
    if row["primitive"] in ("score", "noul"):
        label = int(label)
    return Record(
        id=row["id"], source=row["source"], primitive=row["primitive"], split=row["split"],
        state=json.loads(row["state"]) if isinstance(row["state"], str) else row["state"],
        question=json.loads(row["question"]) if isinstance(row["question"], str) else row["question"],
        label=label, soft_label=None if soft in (None, "") else (json.loads(soft) if isinstance(soft, str) else soft),
        meta=json.loads(row.get("meta") or "{}") if isinstance(row.get("meta"), str) else (row.get("meta") or {}),
    )


def request_body(rec: Record, model_id: str) -> dict[str, Any]:
    return {"state": rec.state, "model": model_id, "questions": {"q": rec.question}}


def probs_of(rec: Record, answer: dict[str, Any]) -> list[float]:
    """Distribution over `rec.option_keys()` from a wire answer."""
    if rec.primitive == "noul":
        p = min(1.0, max(0.0, float(answer["noul"])))
        return [1.0 - p, p]
    pr = answer.get("probabilities") or {}
    out = [max(0.0, float(pr.get(k, 0.0))) for k in rec.option_keys()]
    s = sum(out)
    return [x / s for x in out] if s > 0 else [1.0 / len(out)] * len(out)


def predicted_index(rec: Record, probs: Sequence[float]) -> int:
    if rec.primitive == "noul":
        return 1 if probs[1] >= 0.5 else 0
    best = max(probs)
    return next(i for i, p in enumerate(probs) if p == best)  # np.argmax: first maximum


def ece15(confs: Sequence[float], corrects: Sequence[float], n_bins: int = ECE_BINS) -> float:
    """Jevify's top-label ECE: equal-width bins, (lo, hi], the first bin closed at 0."""
    n = len(confs)
    if n == 0:
        return float("nan")
    edges = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        idx = [i for i, c in enumerate(confs) if (c > lo if lo > 0 else c >= lo) and c <= hi]
        if not idx:
            continue
        acc = sum(corrects[i] for i in idx) / len(idx)
        conf = sum(confs[i] for i in idx) / len(idx)
        ece += abs(acc - conf) * len(idx) / n
    return ece


def report(records: Sequence[Record], answers: Sequence[dict[str, Any] | None]) -> dict[str, Any]:
    """Per-config metrics. Missing answers (refused/error) are wrong and excluded from the calibration stats."""
    corrects: list[float] = []
    confs: list[float] = []
    briers: list[float] = []
    nlls: list[float] = []
    tvds: list[float] = []
    n_missing = 0
    for rec, a in zip(records, answers):
        if a is None:
            n_missing += 1
            corrects.append(0.0)
            continue
        p = probs_of(rec, a)
        y = rec.label_index()
        hit = 1.0 if predicted_index(rec, p) == y else 0.0
        corrects.append(hit)
        confs.append(max(p))
        if rec.primitive == "noul":
            briers.append((p[1] - y) ** 2)
            nlls.append(-math.log(max(EPS, p[1] if y == 1 else 1 - p[1])))
        else:
            briers.append(sum((pi - (1.0 if i == y else 0.0)) ** 2 for i, pi in enumerate(p)))
            nlls.append(-math.log(max(EPS, p[y])))
        if rec.soft_label is not None:
            keys = rec.option_keys()
            soft = rec.soft_label
            if rec.primitive == "noul":
                s = float(soft) if not isinstance(soft, dict) else float(soft.get("1", soft.get(1, 0.0)))
                tvds.append(abs(p[1] - s))
            else:
                if isinstance(soft, dict):
                    sv = [float(soft.get(k, 0.0)) for k in keys]
                else:
                    sv = [float(x) for x in soft]
                tvds.append(0.5 * sum(abs(pi - si) for pi, si in zip(p, sv)))
    n = len(records)
    answered = n - n_missing
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")  # noqa: E731
    conf_hits = [c for c, a in zip(corrects, answers) if a is not None]
    return {
        "n": n, "n_answered": answered, "n_missing": n_missing,
        "accuracy": mean(corrects), "accuracy_answered": mean(conf_hits) if conf_hits else float("nan"),
        "ece": ece15(confs, conf_hits), "brier": mean(briers), "nll": mean(nlls), "mean_confidence": mean(confs),
        "tvd_to_human": mean(tvds) if tvds else None,
    }


def macro(per_config: dict[str, dict[str, Any]], primitives: dict[str, str]) -> dict[str, Any]:
    accs = [r["accuracy"] for r in per_config.values()]
    by_prim: dict[str, list[float]] = defaultdict(list)
    for cfg, r in per_config.items():
        by_prim[primitives.get(cfg, "?")].append(r["accuracy"])
    out = {"macro_accuracy": sum(accs) / len(accs) if accs else float("nan"), "n_configs": len(accs),
           "macro_ece": _nanmean([r["ece"] for r in per_config.values()]), "macro_brier": _nanmean([r["brier"] for r in per_config.values()])}
    for prim, xs in sorted(by_prim.items()):
        out[f"{prim}_acc"] = sum(xs) / len(xs)
    tv = [r["tvd_to_human"] for r in per_config.values() if r.get("tvd_to_human") is not None]
    if tv:
        out["tvd_to_human"] = sum(tv) / len(tv)
    return out


def _nanmean(xs: Sequence[float]) -> float:
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def config_of_target(row: dict[str, Any]) -> str | None:
    ds = str(row.get("dataset") or "")
    cfg = ds.split(" ")[0].strip()
    return cfg if cfg in CONFIGS else None


class JevbenchHfAdapter(Adapter):
    default_overflow = "refuse"
    allow_truncate = False
    default_split = "test"

    @classmethod
    def add_arguments(cls, p: Any) -> None:
        p.add_argument("--data-dir", help=f"local mirror of {HF_ID} data files (default {DEFAULT_WORK / 'hf-data' / 'jev-bench'})")
        p.add_argument("--configs", nargs="*", help=f"subset of the 22 configs (default all)")
        p.add_argument("--split", choices=["test", "validation", "train"], default="test", help="validation/train = self-check only")
        p.add_argument("--limit", type=int, help="first N records per config (self-check)")
        p.add_argument("--revision", default=PINNED_REVISION)

    @property
    def data_dir(self) -> Path:
        return Path(self.args.data_dir or DEFAULT_WORK / "hf-data" / "jev-bench").expanduser()

    def _fetch(self, rel: str) -> Path:
        p = self.data_dir / rel
        if p.exists():
            return p
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(HF_ID, rel, repo_type="dataset", revision=self.args.revision, local_dir=str(self.data_dir)))

    def _manifest(self) -> dict[str, Any]:
        return json.loads(self._fetch("manifest.json").read_text())

    def verify(self, ctx: RunContext) -> dict[str, Any]:
        rep: dict[str, Any] = {"hf_id": HF_ID, "revision": self.args.revision, "licence": LICENCE, "scorer_reimplemented_from": f"uspraveen/Jevify@{JEVIFY_SCORER_COMMIT[:8]}"}
        try:
            m = self._manifest()
        except Exception as e:
            rep["manifest_error"] = f"{type(e).__name__}: {e}"
            rep["ok"] = False
            return rep
        rep["manifest"] = {"version": m.get("version"), "built_at": m.get("built_at"), "git_commit": m.get("git_commit"), "seed": m.get("seed")}
        rep["version_match"] = m.get("version") == VERSION
        counts = {cfg: (m.get("sources", {}).get(cfg, {}).get("counts") or {}) for cfg in CONFIGS}
        rep["test_counts"] = {cfg: {"manifest": c.get("test"), "expected": EXPECTED_TEST_N[cfg], "match": c.get("test") == EXPECTED_TEST_N[cfg]} for cfg, c in counts.items()}
        tgt = {}
        for r in load_targets(self.spec.id):
            cfg = config_of_target(r)
            if cfg and r.get("role") in ("headline", "secondary", "context") and isinstance(r.get("n"), int):
                tgt[cfg] = {"targets_n": r["n"], "match": r["n"] == counts[cfg].get("test"), "jev_score": r.get("jev_score"), "id": r.get("id")}
        rep["targets_json"] = tgt
        ctx.dataset_revisions[HF_ID] = {"pinned": self.args.revision, "manifest_version": m.get("version")}
        rep["ok"] = bool(rep["version_match"] and all(v["match"] for v in rep["test_counts"].values()))
        return rep

    def run(self, session: Session, ctx: RunContext) -> dict[str, Any]:
        ver = self.verify(ctx)
        ctx.harness = ver
        split = self.args.split
        configs = list(self.args.configs or CONFIGS)
        if split != "test":
            configs = [c for c in configs if c not in TEST_ONLY]
        per_config: dict[str, dict[str, Any]] = {}
        primitives: dict[str, str] = {}
        preds_dir = ctx.out / "predictions"
        preds_dir.mkdir(parents=True, exist_ok=True)
        for cfg in configs:
            path = self._fetch(f"data/{cfg}/{split}.jsonl")
            records = [parse_row(json.loads(l)) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]
            if self.args.limit:
                records = records[: self.args.limit]
            if records:
                primitives[cfg] = records[0].primitive
            answers: list[dict[str, Any] | None] = []
            with (preds_dir / f"{cfg}.jsonl").open("w", encoding="utf-8") as f:
                for rec in records:
                    o = session.ask(rec.state, {"q": rec.question}, {"id": rec.id})
                    a = o.response["answers"].get("q") if o.ok else None
                    answers.append(a)
                    f.write(json.dumps({"id": rec.id, "status": o.status, "answer": a, "latency_s": round(o.latency_s, 4), "error": None if o.ok else o.detail}, ensure_ascii=False) + "\n")
            r = report(records, answers)
            r["expected_n"] = EXPECTED_TEST_N[cfg] if split == "test" else None
            r["n_match"] = (r["n"] == EXPECTED_TEST_N[cfg]) if (split == "test" and not self.args.limit) else None
            r["jev_card_accuracy"] = JEV_CARD_ACC.get(cfg) if split == "test" else None
            if split == "test" and not self.args.limit and cfg in JEV_CARD_ACC:
                r["delta_vs_jev"] = r["accuracy"] - JEV_CARD_ACC[cfg]
            per_config[cfg] = r
            ctx.results = {"split": split, "configs": per_config}
            ctx.write()
        res: dict[str, Any] = {"split": split, "configs": per_config, "macro": macro(per_config, primitives), "primitives": primitives,
                               "jev_card": JEV_CARD_MACRO if split == "test" else None, "scorer": f"reimplemented from Jevify@{JEVIFY_SCORER_COMMIT[:8]} metrics.py"}
        if split != "test" or self.args.limit or set(configs) != set(CONFIGS):
            res["note"] = "subset / non-test split: self-check only, not comparable with the card"
        return res
