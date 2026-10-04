"""Zero-shot Laya (convaiinnovations/laya) on the 20 harness cases, on this Mac.

Answers: is Laya a better drop-in general engine than Qwen3-1.7B, and how fast is it on M1?
Usage: .venv/bin/python scripts/bench_laya.py [--device mps|cpu] [--head-max-len 768] [--max-len 1024]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from jev_local.harness.questions import build_questions, rank_apps
from jev_local.harness.spans import extract_text_candidates, extract_url_candidates
from jev_local.harness.state import build_state
from jev_local.harness.types import Element, Snapshot
from jev_local.schema import question_to_json

ROOT = Path(__file__).resolve().parents[1]


def load_cases():
    d = json.loads((ROOT / "tests/fixtures/harness_cases.json").read_text())
    return d


def snapshot_for(screen: dict) -> Snapshot:
    els = tuple(
        Element(
            eid=e["eid"], role=e["role"], label=e.get("label", ""), value=e.get("value"),
            context=e.get("context"), focused=e.get("focused", False), enabled=e.get("enabled", True),
        )
        for e in screen["elements"]
    )
    return Snapshot(screen["app_name"], screen.get("bundle_id", ""), 0, screen.get("window_title"), els, time.monotonic())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--head-max-len", type=int, default=768)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--repo", default="convaiinnovations/laya")
    args = ap.parse_args()

    import laya

    t0 = time.perf_counter()
    agent = laya.load(args.repo, device=args.device)
    load_s = time.perf_counter() - t0

    d = load_cases()
    # doc example (Jev API reference)
    doc_q = {"department": {"type": "choice", "instructions": "Which department should handle this ticket?",
                            "criteria": {"billing": None, "technical": None, "sales": None}}}
    doc = agent.system_one("Help! My payouts have been failing for 3 days.", doc_q)

    rows, lat = [], []
    for case in d["cases"]:
        screen = d["screens"][case["screen"]]
        snap = snapshot_for(screen)
        t = case["transcript"]
        apps = rank_apps(t, d["apps"], screen.get("running", []))
        qs = build_questions(snap, apps, extract_text_candidates(t), extract_url_candidates(t))
        state = build_state(t, snap)
        qjson = {k: question_to_json(v) for k, v in qs.items()}
        agent.system_one(state, qjson, max_len=args.max_len, head_max_len=args.head_max_len)  # warm
        t1 = time.perf_counter()
        out = agent.system_one(state, qjson, max_len=args.max_len, head_max_len=args.head_max_len)
        lat.append((time.perf_counter() - t1) * 1000)
        ans = out["answers"] if "answers" in out else out
        exp = case["expect"]
        got = {
            "intent": ans["intent"]["choice"],
            "target": ans.get("target", {}).get("choice", "none"),
            "app": ans["app"]["choice"],
            "complete": ans["complete"]["noul"] >= 0.5,
            "is_command": ans["is_command"]["noul"] >= 0.5,
        }
        rows.append({"id": case["id"], "t": t, "n_el": len(snap.elements),
                     **{k: got[k] == exp[k] for k in got}, "got": got, "exp": exp})

    acc = {k: sum(r[k] for r in rows) / len(rows) for k in ("intent", "target", "app", "complete", "is_command")}
    report = {
        "repo": args.repo, "device": args.device, "load_s": round(load_s, 1),
        "head_max_len": args.head_max_len, "max_len": args.max_len,
        "doc_example": doc.get("answers", doc)["department"],
        "latency_ms": {"p50": round(statistics.median(lat)), "max": round(max(lat)), "min": round(min(lat))},
        "accuracy": {k: round(v, 2) for k, v in acc.items()},
        "misses": [{"id": r["id"], "t": r["t"], "got": r["got"], "exp": r["exp"]} for r in rows
                   if not all(r[k] for k in ("intent", "target", "app"))],
    }
    out = ROOT / "runs" / "bench_laya.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "misses"}, indent=1))
    print("misses:", len(report["misses"]))
    for m in report["misses"][:8]:
        print(" ", m["id"], repr(m["t"]), "got", {k: m["got"][k] for k in ("intent", "target", "app")},
              "exp", {k: m["exp"][k] for k in ("intent", "target", "app")})


if __name__ == "__main__":
    main()
