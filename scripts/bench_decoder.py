"""Benchmark and sanity-check the MLX decoder engine (`jev-local-general`).

    .venv/bin/python scripts/bench_decoder.py                 # everything, 5 timed runs each
    .venv/bin/python scripts/bench_decoder.py --only latency --runs 10
    .venv/bin/python scripts/bench_decoder.py --only accuracy --order-samples 2

Measures (CONTRACT C):
- the official doc example (payout ticket -> billing);
- latency of a 5-question, ~1k-token generic request, and of a 60-element harness request built
  with `harness.questions.build_questions`, each cold (no prefix reuse), warm (same request again:
  the whole prefix is reused) and, for the harness, per-partial (the transcript grows word by word,
  so only the tail of the state is re-prefilled);
- intent / target / app / complete / is_command accuracy on the 20 hand-written cases in
  tests/fixtures/harness_cases.json (argmax for choices, 0.5 for nouls).

The machine is shared (other agents, browsers), so every latency line also records the load
average and swap in use when it was measured. Results are merged into runs/bench_decoder.json as
each section finishes, with MLX memory use.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "harness_cases.json"
sys.path.insert(0, str(ROOT))

from jev_local.harness.questions import build_questions, rank_apps  # noqa: E402
from jev_local.harness.spans import extract_text_candidates, extract_url_candidates  # noqa: E402
from jev_local.harness.state import build_state  # noqa: E402
from jev_local.harness.types import Element, Snapshot  # noqa: E402
from jev_local.schema import Question, SystemOneRequest  # noqa: E402

# ------------------------------------------------------------------ requests


def load_fixture(path: Path = FIXTURE) -> dict[str, Any]:
    return json.loads(path.read_text())


def snapshot_of(screen: dict[str, Any]) -> Snapshot:
    els = tuple(
        Element(
            eid=e["eid"],
            role=e["role"],
            label=e.get("label", ""),
            value=e.get("value"),
            focused=bool(e.get("focused", False)),
            enabled=bool(e.get("enabled", True)),
            context=e.get("context"),
        )
        for e in screen["elements"]
    )
    focused = next((e.eid for e in els if e.focused), None)
    return Snapshot(
        app_name=screen["app_name"],
        bundle_id=screen["bundle_id"],
        pid=0,
        window_title=screen.get("window_title"),
        elements=els,
        taken_at=0.0,
        focused_eid=focused,
    )


def harness_request(fx: dict[str, Any], transcript: str, screen_name: str) -> tuple[dict[str, Any], dict[str, Question]]:
    """(state, questions) exactly as the runtime harness builds them."""
    screen = fx["screens"][screen_name]
    snap = snapshot_of(screen)
    apps = rank_apps(transcript, fx["apps"], screen.get("running", ()), max_n=24)
    state = build_state(transcript, snap)
    questions = build_questions(
        snap, apps, extract_text_candidates(transcript), extract_url_candidates(transcript)
    )
    return state, questions


DOC_EXAMPLE = {
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "jev-local-general",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this ticket?",
            "criteria": {"billing": None, "technical": None, "sales": None},
        }
    },
}


def generic_1k_request() -> SystemOneRequest:
    """A 5-question support-ticket request of about 1k input tokens (with the system template)."""
    ticket = (
        "Hi team, I run a small online shop and I've been a customer for about two years. Since Monday "
        "every payout to my bank account has failed. The dashboard shows the status 'failed - account "
        "verification required' next to each transfer, but I verified my account last month and uploaded "
        "the bank statement you asked for. I have three pending payouts now, totalling a little over "
        "4,200 dollars, and I need that money to pay my suppliers by Friday. I tried removing the bank "
        "account and adding it again, but the page just shows a spinner and then an error that says "
        "'something went wrong, please try again later'. I also tried a different browser and my phone. "
        "Nothing works. Your chat bot keeps sending me a link to an article about verification that "
        "does not help at all. Honestly I'm getting really frustrated, because this is the second time "
        "this year that payouts have stopped without any warning. If this can't be fixed by the end of "
        "the week I will have to move to another payment provider, which I'd rather not do. Could "
        "someone please look at my account today and tell me what is going on? I can jump on a call "
        "any time before 5pm Eastern. I'd also like to know whether you can refund the failed payout "
        "fees, since there were three of them. Thanks, Dana\n\n"
        "P.S. In case it helps: the first failure email arrived on Monday morning, a few hours after I "
        "changed the business address in my settings because we moved to a new warehouse. I don't know "
        "if that is related. My accountant also says the 1099 form on the documents page still shows the "
        "old address, so I'd like that corrected before tax season, but that part is not urgent."
    )
    state = {
        "ticket": ticket,
        "customer": {
            "name": "Dana Whitfield",
            "plan": "Growth (monthly)",
            "customer_since": "2024-08-14",
            "country": "United States",
            "lifetime_volume_usd": 184500,
            "open_tickets": 2,
        },
        "recent_events": [
            "2026-09-21 payout po_81f2 failed: account_verification_required",
            "2026-09-22 payout po_82a9 failed: account_verification_required",
            "2026-09-23 payout po_83c1 failed: account_verification_required",
            "2026-09-23 bank account update attempt failed: internal_error",
            "2026-08-19 identity verification completed",
            "2026-09-21 business address changed by account owner",
        ],
        "previous_tickets": [
            "2026-03-02 payouts delayed two days after a bank holiday; resolved by billing",
            "2026-05-17 asked how to add a second user to the dashboard; resolved by account",
            "2026-07-08 checkout showed a currency error for Canadian buyers; fixed by technical",
        ],
        "account_notes": (
            "Verified business, dispute rate 0.2%. Daily payout schedule to a checking account ending 4471, "
            "added 2025-11-03 and re-verified 2026-08-19."
        ),
        "channel": "email",
    }
    questions = {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "billing": "payments, payouts, invoices, refunds and fees",
                "technical": "bugs, errors, outages and integrations",
                "sales": "new plans, upgrades and pricing questions",
                "account": "login, identity verification and account settings",
            },
        },
        "urgent": {
            "type": "noul",
            "instructions": "Does this ticket need a reply today?",
            "criteria": {"true": "money is blocked or there is a hard deadline soon", "false": "it can wait"},
        },
        "sentiment": {
            "type": "choice",
            "instructions": "What is the customer's mood?",
            "criteria": {"positive": None, "neutral": None, "frustrated": None, "angry": None},
        },
        "churn_risk": {
            "type": "score",
            "instructions": "How likely is the customer to leave?",
            "criteria": ["very unlikely", "unlikely", "possible", "likely", "very likely"],
        },
        "refund_requested": {
            "type": "noul",
            "instructions": "Does the customer ask for money back?",
        },
    }
    return SystemOneRequest.model_validate({"state": state, "model": "jev-local-general", "questions": questions})


# ------------------------------------------------------------------ measuring


def conditions() -> dict[str, Any]:
    """Load average and swap in use: this M1 is shared, so latencies are only meaningful with these."""
    out: dict[str, Any] = {"loadavg": [round(x, 2) for x in os.getloadavg()]}
    try:
        swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5).stdout
        out["swap"] = " ".join(swap.split()[:6])
    except Exception:
        pass
    return out


def rss_mb() -> float:
    # ru_maxrss is bytes on macOS
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))
    return xs[k]


def summarize(ms: list[float]) -> dict[str, float]:
    return {
        "p50": round(statistics.median(ms), 1),
        "p90": round(pct(ms, 0.9), 1),
        "min": round(min(ms), 1),
        "max": round(max(ms), 1),
        "n": len(ms),
    }


def timed(engine, state, questions, cold: bool) -> tuple[float, Any]:
    if cold:
        engine._pcache = None  # no prefix reuse: the whole prompt is prefilled
    t0 = time.perf_counter()
    res = engine.evaluate(state, questions)
    return (time.perf_counter() - t0) * 1000.0, res


def bench_latency(engine, fx: dict[str, Any], runs: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    gen = generic_1k_request()
    h_state, h_q = harness_request(fx, "click the story about the markets", "safari_news")
    n_targets = len(h_q["target"].criteria) - 1  # minus "none"

    # warm-up: shader compilation and first-touch of the weights are not what we want to measure
    timed(engine, gen.state, gen.questions, cold=True)
    timed(engine, h_state, h_q, cold=True)

    for label, (state, qs) in {
        "generic_5q": (gen.state, gen.questions),
        f"harness_{n_targets}el": (h_state, h_q),
    }.items():
        rows: dict[str, Any] = {}
        for mode in ("cold", "warm"):
            ms, res = [], None
            for _ in range(runs):
                t, res = timed(engine, state, qs, cold=(mode == "cold"))
                ms.append(t)
            rows[mode] = summarize(ms) | {
                "input_tokens": res.input_tokens,
                "cached_tokens": res.cached_tokens,
                "timings_ms_last": {k: round(v, 1) for k, v in res.timings_ms.items()},
            }
        rows["n_questions"] = len(qs)
        rows["conditions"] = conditions()
        out[label] = rows
        print(f"{label}: {json.dumps(rows)}", flush=True)

    # Harness per-partial: the transcript grows a word at a time, as the controller would send it.
    words = "click the story about the markets".split()
    engine._pcache = None
    ms, cached = [], []
    for i in range(1, len(words) + 1):
        st, qs = harness_request(fx, " ".join(words[:i]), "safari_news")
        t, res = timed(engine, st, qs, cold=False)
        ms.append(t)
        cached.append(res.cached_tokens)
    out["harness_partials"] = summarize(ms) | {"cached_tokens": cached, "conditions": conditions()}
    print(f"harness_partials: {json.dumps(out['harness_partials'])}", flush=True)
    return out


def argmax_label(dist) -> str:
    best = max(range(len(dist.probs)), key=lambda i: dist.probs[i])
    return dist.labels[best]


def bench_accuracy(engine, fx: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for case in fx["cases"]:
        state, qs = harness_request(fx, case["transcript"], case["screen"])
        t0 = time.perf_counter()
        res = engine.evaluate(state, qs)
        ms = (time.perf_counter() - t0) * 1000.0
        exp = case["expect"]
        got = {
            "intent": argmax_label(res.dists["intent"]),
            "target": argmax_label(res.dists["target"]) if "target" in res.dists else "none",
            "app": argmax_label(res.dists["app"]),
        }
        p_intent = dict(zip(res.dists["intent"].labels, res.dists["intent"].probs))
        row = {
            "id": case["id"],
            "transcript": case["transcript"],
            "expect": exp,
            "got": got,
            "p_intent_gold": round(p_intent.get(exp["intent"], 0.0), 3),
            "p_intent_top": round(max(p_intent.values()), 3),
            "complete": round(res.dists["complete"].probs[0], 3),
            "is_command": round(res.dists["is_command"].probs[0], 3),
            "destructive": round(res.dists["destructive"].probs[0], 3),
            "ms": round(ms, 0),
            "input_tokens": res.input_tokens,
        }
        # noul answers are judged at 0.5, the natural cut for an absolute probability
        got["complete"] = row["complete"] >= 0.5
        got["is_command"] = row["is_command"] >= 0.5
        rows.append(row)
        ok_i = got["intent"] == exp["intent"]
        ok_t = got["target"] == exp["target"]
        ok_c = "complete" not in exp or got["complete"] == exp["complete"]
        print(
            f"{case['id']} {'OK ' if ok_i else 'BAD'} intent={got['intent']:<12} (gold {exp['intent']:<12} "
            f"p={row['p_intent_gold']:.2f})  {'OK ' if ok_t else 'BAD'} target={got['target']:<5} "
            f"(gold {exp['target']})  app={got['app']:<8} (gold {exp.get('app', '-')})  "
            f"{'OK ' if ok_c else 'BAD'} complete={row['complete']:.2f}  {ms:6.0f} ms  {case['transcript']!r}",
            flush=True,
        )
    n = len(rows)

    def acc(key: str, subset: list[dict[str, Any]] | None = None) -> str:
        rs = [r for r in (rows if subset is None else subset) if key in r["expect"]]
        return f"{sum(r['got'][key] == r['expect'][key] for r in rs)}/{len(rs)}"

    clicks = [r for r in rows if r["expect"]["target"] != "none"]
    named_apps = [r for r in rows if r["expect"].get("app", "none") != "none"]
    summary = {
        "intent_acc": acc("intent"),
        "target_acc_click_cases": acc("target", clicks),
        "target_acc_all_incl_none": acc("target"),
        "app_acc_named": acc("app", named_apps),
        "app_acc_all_incl_none": acc("app"),
        "complete_acc": acc("complete"),
        "is_command_acc": acc("is_command"),
        "ms_p50": round(statistics.median(r["ms"] for r in rows), 0),
        "n": n,
        "conditions": conditions(),
    }
    print(json.dumps(summary), flush=True)
    return {"summary": summary, "cases": rows}


def doc_example(engine) -> dict[str, Any]:
    req = SystemOneRequest.model_validate(DOC_EXAMPLE)
    res = engine.evaluate(req.state, req.questions)
    d = res.dists["department"]
    probs = {lab: round(p, 3) for lab, p in zip(d.labels, d.probs)}
    out = {"probabilities": probs, "choice": argmax_label(d), "input_tokens": res.input_tokens}
    print(f"doc example: {json.dumps(out)}", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="mlx-community/Qwen3-1.7B-4bit")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--order-samples", type=int, default=1)
    ap.add_argument("--batch-tokens", type=int, default=None, help="padded-token budget per batched forward")
    ap.add_argument("--only", choices=["latency", "accuracy", "doc"], default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "bench_decoder.json")
    args = ap.parse_args(argv)

    import mlx.core as mx

    from jev_local.engine.decoder.mlx_scorer import GeneralEngine

    kw: dict[str, Any] = {}
    if args.batch_tokens is not None:
        kw["batch_tokens"] = args.batch_tokens
    engine = GeneralEngine(args.repo, order_samples=args.order_samples, **kw)
    fx = load_fixture()
    t0 = time.perf_counter()
    engine.load()
    # Sections are merged into an existing report and saved as each one finishes, so a run that is
    # killed (or a later `--only accuracy`) keeps the sections measured before it.
    report: dict[str, Any] = {}
    if args.out.exists():
        try:
            report = json.loads(args.out.read_text())
        except ValueError:
            report = {}
    report.update(
        {
            "engine": engine.name,
            "repo": args.repo,
            "order_samples": args.order_samples,
            "batch_tokens": engine.batch_tokens,
            "cache_limit_mb": engine.cache_limit_mb,
            "load_ms": round((time.perf_counter() - t0) * 1000.0, 0),
            "weights_gb": round(mx.get_active_memory() / 1e9, 2),
            "conditions_start": conditions(),
        }
    )
    print(f"loaded in {report['load_ms']:.0f} ms", flush=True)
    mx.reset_peak_memory()

    def save(section: str, value: Any) -> None:
        report[section] = value
        report["memory"] = {
            "mlx_peak_gb": round(mx.get_peak_memory() / 1e9, 2),
            "mlx_active_gb": round(mx.get_active_memory() / 1e9, 2),
            "mlx_cache_gb": round(mx.get_cache_memory() / 1e9, 2),
            "max_rss_mb": round(rss_mb(), 0),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))

    if args.only in (None, "doc"):
        save("doc_example", doc_example(engine))
    if args.only in (None, "latency"):
        save("latency", bench_latency(engine, fx, args.runs))
    if args.only in (None, "accuracy"):
        save("accuracy", bench_accuracy(engine, fx))
    print(json.dumps(report["memory"]), flush=True)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
