"""Post-hoc calibration for the fast engine (SPEC §3.5, PLAN §3.2 item 5).

calibration.json (v2 keys are additive; a v1 file still works and v1 readers ignore the new keys):

    {"noul": tau, "choice": tau, "score": tau,              # v1: one temperature per head kind
     "by_header": {sha1(header)[:12]: tau},                 # v1: per question header (harness qids)
     "version": 2,
     "by_bucket": {"choice:2": tau, "choice:3-5": tau, ..., "score:6-10": tau},   # (type, K-bucket)
     "tau_k": {"choice": [a, b], "score": [a, b]},          # parametric tau(K) = a + b ln K (fallback)
     "noul_platt": {"a": a, "b": b},                         # p = sigmoid(a z + b) for noul
     "bucket_clamp": [0.5, 5.0],
     "_fit": {...}}                                          # fit statistics (not used at inference)

Lookup for one question (kind, header, K = number of options):
    by_header[header] -> tau               (fixed-instruction harness questions keep their own temperature)
    noul:          noul_platt (a, b) -> per-kind tau
    choice/score:  by_bucket[kind:K-bucket] -> tau_k(K) -> per-kind tau
K-buckets: 2 / 3-5 / 6-10 / 11-30 / 31-100 / 101-255. Bucket temperatures and tau(K) are clamped to
[0.5, 5] (Laya shipped a 0.10 "11+" temperature, a 10x sharpener); per-kind/per-header keep v1's range.
All fits minimise NLL on a held-out dev slice (never training items or test). The NLL of a softmax or
sigmoid is convex in beta = 1/tau, so golden-section search on log beta is exact enough; Platt (a, b) is
fitted by Newton's method (a 2-D convex problem).

CLI:
    python -m jev_local.engine.encoder.calibrate --ckpt models/jev-local-fast --data data/cu \
        [--extra data/gen] [--split dev] [--limit N] [--min-per-header 50] [--min-per-bucket 50]
    python -m jev_local.engine.encoder.calibrate --ckpt ... --stream data/v2 --split dev_mix
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from jev_local.engine.base import Kind

DEFAULT_CALIBRATION: dict = {"noul": 1.0, "choice": 1.0, "score": 1.0, "by_header": {}}
TAU_MIN, TAU_MAX = 0.05, 20.0
BUCKET_CLAMP = (0.5, 5.0)
K_BUCKETS: tuple[tuple[int, int], ...] = ((2, 2), (3, 5), (6, 10), (11, 30), (31, 100), (101, 255))


def header_key(header: str) -> str:
    return hashlib.sha1(header.encode("utf-8")).hexdigest()[:12]


def k_bucket(k: int) -> str:
    for lo, hi in K_BUCKETS:
        if k <= hi:
            return str(lo) if lo == hi else f"{lo}-{hi}"
    lo, hi = K_BUCKETS[-1]
    return f"{lo}-{hi}"


def bucket_key(kind: str, k: int) -> str:
    return "noul" if kind == "noul" else f"{kind}:{k_bucket(k)}"


@dataclass
class CalibRecord:
    """One supervised question: raw head logits (tau = 1) and its target distribution."""

    qid: str
    kind: Kind
    header: str  # header_key(...) of the question's instructions
    logits: np.ndarray  # choice/score: [K]; noul: [1]
    target: np.ndarray  # choice/score: [K] distribution; noul: [1] = p(true)

    @property
    def k(self) -> int:
        return 2 if self.kind == "noul" else len(self.logits)


def _stack(records: Sequence[CalibRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kmax = max(len(r.logits) for r in records)
    z = np.full((len(records), kmax), -np.inf, dtype=np.float64)
    t = np.zeros((len(records), kmax), dtype=np.float64)
    for i, r in enumerate(records):
        z[i, : len(r.logits)] = r.logits
        t[i, : len(r.target)] = r.target
    return z, t, np.isfinite(z)


def _softmax_nll(z: np.ndarray, t: np.ndarray, m: np.ndarray, beta: np.ndarray | float) -> np.ndarray:
    b = np.asarray(beta, dtype=np.float64)
    zb = np.where(m, z * (b[:, None] if b.ndim else b), -np.inf)
    mx = zb.max(axis=1, keepdims=True)
    lse = mx[:, 0] + np.log(np.exp(zb - mx).sum(axis=1))
    logp = np.where(m, zb - lse[:, None], 0.0)
    return -(t * logp).sum(axis=1)


def _bce(z: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0) - z * p + np.log1p(np.exp(-np.abs(z)))


def nll(records: Sequence[CalibRecord], tau: float) -> float:
    if not records:
        return 0.0
    beta = 1.0 / tau
    if records[0].kind == "noul":
        z = np.array([r.logits[0] for r in records], dtype=np.float64) * beta
        p = np.array([r.target[0] for r in records], dtype=np.float64)
        return float(np.mean(_bce(z, p)))
    z, t, m = _stack(records)
    return float(np.mean(_softmax_nll(z, t, m, beta)))


def _golden(f, lo: float, hi: float, iters: int = 60) -> float:
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (a + b) / 2


def fit_tau(records: Sequence[CalibRecord], clamp: tuple[float, float] = (TAU_MIN, TAU_MAX)) -> float:
    if not records:
        return 1.0
    lo, hi = math.log(1 / clamp[1]), math.log(1 / clamp[0])  # search log(beta)
    return float(math.exp(-_golden(lambda x: nll(records, math.exp(-x)), lo, hi)))


def fit_tau_k(records: Sequence[CalibRecord], clamp: tuple[float, float] = BUCKET_CLAMP,
              max_n: int = 20000) -> tuple[float, float] | None:
    """tau(K) = a + b ln K by NLL (coordinate search over b, exact 1-D fit of a for each b)."""
    recs = [r for r in records if r.kind != "noul"]
    ks = {len(r.logits) for r in recs}
    if len(recs) < 20 or len(ks) < 2:
        return None
    if len(recs) > max_n:  # deterministic subsample keeps the 2-D search fast
        recs = [recs[i] for i in np.random.default_rng(0).choice(len(recs), size=max_n, replace=False)]
    z, t, m = _stack(recs)
    lnk = np.log(np.array([len(r.logits) for r in recs], dtype=np.float64))

    def loss(a: float, b: float) -> float:
        tau = np.clip(a + b * lnk, *clamp)
        return float(np.mean(_softmax_nll(z, t, m, 1.0 / tau)))

    def best_a(b: float) -> tuple[float, float]:
        a = _golden(lambda a: loss(a, b), clamp[0] - 3.0, clamp[1] + 3.0, iters=40)
        return a, loss(a, b)

    b = _golden(lambda b: best_a(b)[1], -2.0, 2.0, iters=30)
    a, _ = best_a(b)
    return round(a, 4), round(b, 4)


def fit_platt(records: Sequence[CalibRecord], a_range: tuple[float, float] = (1 / BUCKET_CLAMP[1], 1 / BUCKET_CLAMP[0]),
              b_max: float = 5.0) -> tuple[float, float]:
    """Noul Platt scaling p = sigmoid(a z + b), Newton on the mean BCE; a clamped like 1/tau."""
    z = np.array([r.logits[0] for r in records], dtype=np.float64)
    p = np.array([r.target[0] for r in records], dtype=np.float64)
    a, b = 1.0, 0.0
    for _ in range(50):
        s = np.exp(-np.logaddexp(0.0, -(a * z + b)))  # stable sigmoid
        r = s - p
        w = s * (1 - s) + 1e-9
        g = np.array([np.mean(r * z), np.mean(r)])
        H = np.array([[np.mean(w * z * z), np.mean(w * z)], [np.mean(w * z), np.mean(w)]]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return float(np.clip(a, *a_range)), float(np.clip(b, -b_max, b_max))


def _platt_nll(records: Sequence[CalibRecord], a: float, b: float) -> float:
    z = np.array([r.logits[0] for r in records], dtype=np.float64)
    p = np.array([r.target[0] for r in records], dtype=np.float64)
    return float(np.mean(_bce(a * z + b, p)))


def fit_calibration(records: Iterable[CalibRecord], min_per_header: int = 50, min_per_bucket: int = 50,
                    bucket_clamp: tuple[float, float] = BUCKET_CLAMP, platt: bool = True, buckets: bool = True) -> dict:
    """v1 keys (per kind, per header) plus, with buckets=True, (type, K-bucket) temperatures, tau(K) and
    noul Platt. buckets=False reproduces the v1 file exactly (plus "_fit")."""
    recs = list(records)
    out: dict = {"noul": 1.0, "choice": 1.0, "score": 1.0, "by_header": {}}
    stats: dict = {}
    for kind in ("noul", "choice", "score"):
        rk = [r for r in recs if r.kind == kind]
        tau = fit_tau(rk)
        out[kind] = round(tau, 4)
        stats[kind] = {"n": len(rk), "nll_before": nll(rk, 1.0), "nll_after": nll(rk, tau)}
        headers: dict[str, list[CalibRecord]] = {}
        for r in rk:
            headers.setdefault(r.header, []).append(r)
        for h, rh in headers.items():
            if len(rh) >= min_per_header:
                out["by_header"][h] = round(fit_tau(rh), 4)
    if buckets:
        out["version"] = 2
        out["bucket_clamp"] = list(bucket_clamp)
        out["by_bucket"] = {}
        out["tau_k"] = {}
        for kind in ("choice", "score"):
            rk = [r for r in recs if r.kind == kind]
            groups: dict[str, list[CalibRecord]] = {}
            for r in rk:
                groups.setdefault(bucket_key(kind, len(r.logits)), []).append(r)
            for key, rb in sorted(groups.items()):
                st = {"n": len(rb), "nll_before": nll(rb, 1.0)}
                if len(rb) >= min_per_bucket:
                    tau = fit_tau(rb, bucket_clamp)
                    out["by_bucket"][key] = round(tau, 4)
                    st["nll_after"] = nll(rb, tau)
                stats[key] = st
            tk = fit_tau_k(rk, bucket_clamp)
            if tk is not None:
                out["tau_k"][kind] = list(tk)
        rn = [r for r in recs if r.kind == "noul"]
        if platt and len(rn) >= min_per_bucket:
            a, b = fit_platt(rn)
            out["noul_platt"] = {"a": round(a, 4), "b": round(b, 4)}
            stats["noul"]["nll_platt"] = _platt_nll(rn, a, b)
    out["_fit"] = stats
    return out


# ---------------------------------------------------------------------------- run-config files (W9)


def describe_calibration(calib: Mapping, source: str | Path | None = None, raw: bytes | None = None) -> dict:
    """What a run.json should record about the calibration in force: source, sha256 and which lookups exist.
    The sha256 is of the file bytes when `raw` is given, else of the canonical JSON of `calib`."""
    data = raw if raw is not None else json.dumps(calib, sort_keys=True, separators=(",", ":")).encode()
    return {
        "source": str(source) if source is not None else "inline",
        "sha256": hashlib.sha256(data).hexdigest(),
        "version": calib.get("version", 1),
        "per_kind": {k: calib.get(k) for k in ("noul", "choice", "score")},
        "n_by_header": len(calib.get("by_header") or {}),
        "n_by_bucket": len(calib.get("by_bucket") or {}),
        "tau_k": sorted(calib.get("tau_k") or {}),
        "noul_platt": bool(calib.get("noul_platt")),
    }


def load_calibration(src: str | Path | Mapping, *, drop_by_header: bool = False) -> tuple[dict, dict]:
    """A calibration selected by the run config (a calibration.json path or an in-memory dict) -> (calib, info).
    Missing v1 keys take DEFAULT_CALIBRATION values; the result replaces, never merges with, the checkpoint's
    own file, so the run config alone decides every temperature (W9). Raises ValueError on a malformed file."""
    raw: bytes | None = None
    if isinstance(src, (str, Path)):
        path = Path(src).expanduser()
        raw = path.read_bytes()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"calibration file {path} is not JSON: {e}") from None
        source: str | Path | None = path
    else:
        data = dict(src)
        source = None
    if not isinstance(data, dict):
        raise ValueError("calibration must be a JSON object")
    calib = dict(DEFAULT_CALIBRATION)
    calib.update(data)
    for k in ("noul", "choice", "score"):
        v = calib.get(k)
        if not isinstance(v, (int, float)) or not v > 0:
            raise ValueError(f"calibration[{k!r}] must be a positive temperature, not {v!r}")
    if not isinstance(calib.get("by_header"), dict):
        raise ValueError("calibration['by_header'] must be an object")
    if drop_by_header:
        calib["by_header"] = {}
    info = describe_calibration(calib, source, raw)
    if drop_by_header:
        info["by_header_dropped"] = True
    return calib, info


# ---------------------------------------------------------------------------- lookup / application


def tau_for(calib: dict, kind: Kind, header: str, k: int | None = None) -> float:
    """Temperature for one question. Without k this is the v1 lookup (by_header, then per kind)."""
    bh = calib.get("by_header", {})
    if header in bh:
        return float(bh[header])
    if k is not None and kind != "noul":
        lo, hi = calib.get("bucket_clamp", BUCKET_CLAMP)
        bb = calib.get("by_bucket", {}).get(bucket_key(kind, k))
        if bb is not None:
            return float(bb)
        tk = calib.get("tau_k", {}).get(kind)
        if tk:
            return float(min(max(tk[0] + tk[1] * math.log(max(k, 2)), lo), hi))
    return float(calib.get(kind, 1.0))


def noul_affine(calib: dict, header: str) -> tuple[float, float]:
    """(a, b) with p = sigmoid(a z + b): per-header temperature, else Platt, else per-kind temperature."""
    bh = calib.get("by_header", {})
    if header in bh:
        return 1.0 / float(bh[header]), 0.0
    pl = calib.get("noul_platt")
    if pl:
        return float(pl["a"]), float(pl["b"])
    return 1.0 / float(calib.get("noul", 1.0)), 0.0


def calibrate_logits(kind: Kind, header: str, logits: np.ndarray | Sequence[float], calib: dict) -> np.ndarray:
    """Raw (tau = 1) logits of ONE question -> calibrated probabilities. Choice/score logits are the full
    option set (K = len(logits)), so option batches must be concatenated first: one joint softmax."""
    z = np.asarray(logits, dtype=np.float64)
    if kind == "noul":
        a, b = noul_affine(calib, header)
        return 1.0 / (1.0 + np.exp(-(a * z + b)))
    z = z / tau_for(calib, kind, header, len(z))
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def apply_calibration(r: CalibRecord, calib: dict) -> np.ndarray:
    return calibrate_logits(r.kind, r.header, r.logits, calib)


def apply_tau(r: CalibRecord, tau: float) -> np.ndarray:
    z = r.logits.astype(np.float64) / tau
    if r.kind == "noul":
        return 1.0 / (1.0 + np.exp(-z))
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def calibrated_probs(out, plan, calib: dict):
    """Torch version for a whole HeadOut: per-question temperatures by header / (kind, K-bucket) / kind and
    Platt for noul. Returns {"choice"|"score"|"noul": fp32 tensor | None} like heads.probabilities."""
    import torch

    res: dict = {"choice": None, "score": None, "noul": None}
    for kind in ("choice", "score"):
        z = getattr(out, kind)
        if z is None:
            continue
        refs = plan.of_kind(kind)
        tau = torch.tensor([tau_for(calib, kind, header_key(r.qi.header), len(r.qi.labels)) for r in refs],
                           dtype=torch.float32, device=z.device)
        res[kind] = torch.softmax(z.float() / tau[:, None], dim=-1)
    if out.noul is not None:
        ab = [noul_affine(calib, header_key(r.qi.header)) for r in plan.of_kind("noul")]
        a = torch.tensor([x[0] for x in ab], dtype=torch.float32, device=out.noul.device)
        b = torch.tensor([x[1] for x in ab], dtype=torch.float32, device=out.noul.device)
        res["noul"] = torch.sigmoid(out.noul.float() * a + b)
    return res


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m jev_local.engine.encoder.calibrate")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--extra", type=Path, default=None)
    ap.add_argument("--stream", nargs="+", default=None, help="v2 data roots (rows filtered by --split)")
    ap.add_argument("--stream-cache", default="runs/stream_cache")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--min-per-header", type=int, default=50)
    ap.add_argument("--min-per-bucket", type=int, default=50)
    ap.add_argument("--bucket-clamp", default=f"{BUCKET_CLAMP[0]},{BUCKET_CLAMP[1]}")
    ap.add_argument("--no-platt", action="store_true")
    ap.add_argument("--no-buckets", action="store_true", help="v1 file: per kind + per header only")
    args = ap.parse_args(argv)

    from jev_local.train.eval import collect, example_source, metrics_from

    src = example_source(args.data, args.extra, args.stream, args.split, args.stream_cache)
    res = collect(args.ckpt, src, limit=args.limit, batch=args.batch, device=args.device, max_len=args.max_len)
    before = metrics_from(res, dict(DEFAULT_CALIBRATION))
    lo, hi = (float(x) for x in args.bucket_clamp.split(","))
    calib = fit_calibration(res.records, min_per_header=args.min_per_header, min_per_bucket=args.min_per_bucket,
                            bucket_clamp=(lo, hi), platt=not args.no_platt, buckets=not args.no_buckets)
    after = metrics_from(res, calib)
    (args.ckpt / "calibration.json").write_text(json.dumps(calib, indent=2))
    summary = {
        kind: {"ece_before": before["by_kind"].get(kind, {}).get("ece"), "ece_after": after["by_kind"].get(kind, {}).get("ece"),
               "tau": calib[kind]}
        for kind in ("noul", "choice", "score")
    }
    kb = {k: {"ece_before": before["by_kbucket"][k]["ece"], "ece_after": after["by_kbucket"].get(k, {}).get("ece"),
              "tau": calib.get("by_bucket", {}).get(k)} for k in before.get("by_kbucket", {})}
    print(json.dumps({"calibration": str(args.ckpt / "calibration.json"), "summary": summary, "by_kbucket": kb,
                      "noul_platt": calib.get("noul_platt"), "tau_k": calib.get("tau_k"),
                      "by_header": calib["by_header"]}, indent=2))


if __name__ == "__main__":
    main()
