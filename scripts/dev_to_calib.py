"""Convert jevbench dev request files into training-format examples for engine/encoder/calibrate.py.

The calibration fitter reads {state, questions, labels} rows; the dev files hold {request, gold}. Only the clean
dev sets (registry v1.1) are used — never the retired public-Jev dev sets, never any test split.

  python scripts/dev_to_calib.py --dev bench/jevbench/dev --out data/calib/all [--datasets a,b,c]
Writes <out>/dev.jsonl (split "dev"), one row per dev request.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CLEAN_DEV = ["newsgroups20", "tweeteval_sentiment", "atis", "snips", "mrpc", "scitail", "cb",
             "tweeteval_offensive", "scifact", "tasksource_heldout"]


def label_of(g: dict) -> dict | None:
    t = g.get("type")
    if t == "noul":
        p = g.get("p")
        return {"type": "noul", "p": float(p) if p is not None else (1.0 if g.get("label") else 0.0)}
    if t == "choice":
        if isinstance(g.get("dist"), dict) and g["dist"]:
            return {"type": "choice", "dist": {str(k): float(v) for k, v in g["dist"].items()}}
        return {"type": "choice", "label": g["label"]}
    if t == "score":
        dist = g.get("dist")
        if isinstance(dist, dict) and dist:
            return {"type": "score", "dist": [float(dist[k]) for k in sorted(dist, key=lambda x: int(x))]}
        if isinstance(dist, list) and dist:
            return {"type": "score", "dist": [float(x) for x in dist]}
        return {"type": "score", "level": int(g["label"])}
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--datasets", default=",".join(CLEAN_DEV))
    a = ap.parse_args()
    names = [n for n in a.datasets.split(",") if n]
    bad = [n for n in names if n not in CLEAN_DEV]
    if bad:
        raise SystemExit(f"not in the clean dev set: {bad}")
    a.out.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with (a.out / "dev.jsonl").open("w") as f:
        for name in names:
            for line in (a.dev / f"{name}__main.jsonl").open():
                r = json.loads(line)
                req = r["request"]
                labels = {qid: lab for qid, g in r["gold"].items() if (lab := label_of(g)) is not None}
                if not labels:
                    continue
                f.write(json.dumps({"id": r["id"], "split": "dev", "family": f"calib/{name}", "state": req["state"],
                                    "questions": req["questions"], "labels": labels,
                                    "meta": {"dataset": name}}) + "\n")
                n_rows += 1
    print(json.dumps({"out": str(a.out), "datasets": names, "rows": n_rows}))


if __name__ == "__main__":
    main()
