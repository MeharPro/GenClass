"""W11 determinism report for a meharsjev checkpoint (VM only: loads torch).

    ~/jev/.venv/bin/python scripts/benchmax/engine_determinism.py --ckpt ~/jev/models/jev-local-fast-v2 \
        --preset di --model-id meharsjev-68m [--calib runs/benchmax/calib/global.json] --threads 16 \
        --repeats 5 --http --out runs/benchmax/jev-local-fast-v2/engine/determinism.json

Sends a fixed set of SYNTHETIC requests (every published request shape: Deußer null labels, DI empty state with
`option_i` keys, DMB key == value, object instructions, score levels, 255 options, an 8k state) `repeats` times
in process and, with --http, through a live uvicorn server, and records: identical-bytes fraction, repeat-flip
rate, order-flip rate and worst probability difference under option permutation, latency, the run config, the
checkpoint and calibration sha256, CPU model and thread count. No benchmark item is used (PLAN §2.2).
"""

from __future__ import annotations

import argparse
import json
import random
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jev_local.api import ServeConfig, determinism_report, hardware_info, system_one_json  # noqa: E402
from jev_local.engine.registry import Registry  # noqa: E402
from jev_local.server.app import LOCAL_HOSTS, create_app  # noqa: E402

WORDS = ("payment card arrived late and the mobile app shows an error when I open the transfers tab after the update "
         "please tell me whether the refund for the duplicated charge is on its way because support said two days "
         "the invoice total does not match the purchase order and the vendor address changed since last quarter").split()


def text(n_words: int, seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def build_requests(model: str) -> list[dict]:
    labels77 = [f"intent_{i}_{WORDS[i % len(WORDS)]}" for i in range(77)]
    return [
        # Deußer shape: state {query}, label names as keys, null values, sorted
        {"state": {"query": text(14, 1)}, "model": model,
         "questions": {"answer": {"type": "choice", "instructions": "Which banking intent does the query express?",
                                  "criteria": {k: None for k in sorted(labels77)}}}},
        # Decision Index shape: state {}, input in instructions, option_i keys with descriptions (151 options)
        {"state": {}, "model": model,
         "questions": {"answer": {"type": "choice", "instructions": f"Classify the utterance into one intent. Utterance: {text(12, 2)}",
                                  "criteria": {f"option_{i}": f"{text(3, 100 + i)}" for i in range(151)}}}},
        # DMB shape: raw string state, key == value
        {"state": text(12, 3), "model": model,
         "questions": {"decision": {"type": "choice", "instructions": "Pick the intent of the message.",
                                    "criteria": {k: k for k in sorted(labels77)}}}},
        # Deußer moderation shape: object instructions, several nouls
        {"state": {"text": text(25, 4)}, "model": model,
         "questions": {f"cat_{c}": {"type": "noul", "instructions": {"category": {"name": c, "definition": f"content that is {c}"},
                                                                     "question": "Does the text fall under the category?"}}
                       for c in ("harassment", "hate", "self-harm", "sexual", "violence")}},
        # score with described levels and an object level (SummEval / HelpSteer shape)
        {"state": {"source_document": text(120, 5), "summary": text(20, 6)}, "model": model,
         "questions": {"coherence": {"type": "score", "instructions": {"criterion": "coherence", "question": "Rate the summary"},
                                     "criteria": [{"level": "1", "meaning": "incoherent"}, "2", "3", "4", "5 fully coherent"]}}},
        # 255 options, long texts: several exact passes
        {"state": {"query": text(10, 7)}, "model": model,
         "questions": {"q": {"type": "choice", "instructions": "Which option matches the query?",
                             "criteria": {f"option_{i}": text(26, 200 + i) for i in range(255)}}}},
        # ~8k-token state (synthetic) with two questions
        {"state": {"document": text(6_100, 8)}, "model": model,
         "questions": {"topic": {"type": "choice", "instructions": "What is the document about?", "criteria": {"banking": None, "weather": None, "sports": None}},
                       "refund": {"type": "noul", "instructions": "Does it mention a refund?"}}},
        # empty string state, iSarcasm-track-C-like
        {"state": "", "model": model,
         "questions": {"sarcastic": {"type": "noul", "instructions": f"Is this tweet sarcastic? Tweet: {text(15, 9)}",
                                     "criteria": {"true": "Yes", "false": "No"}}}},
    ]


class LiveServer:
    def __init__(self, app):
        import uvicorn

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{self.sock.getsockname()[1]}"
        self.server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on"))
        self.thread = threading.Thread(target=self.server.run, kwargs={"sockets": [self.sock]}, daemon=True)

    def __enter__(self):
        self.thread.start()
        deadline = time.monotonic() + 120
        while not self.server.started:
            if time.monotonic() > deadline or not self.thread.is_alive():
                raise RuntimeError("uvicorn did not start")
            time.sleep(0.05)
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(10)
        self.sock.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--preset", default="di")
    ap.add_argument("--model-id", default="meharsjev-68m")
    ap.add_argument("--calib", type=Path, default=None)
    ap.add_argument("--global-calibration-only", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--http", action="store_true", help="also run through a live uvicorn server")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = ServeConfig.preset(args.preset, model_id=args.model_id)
    registry = Registry(
        fast_ckpt=args.ckpt, load_general="off",
        fast_options={"device": "cpu", "threads": args.threads, "calibration": str(args.calib) if args.calib else None,
                      "drop_header_calibration": True if args.global_calibration_only else None},
    )
    engines = registry.engines()
    if "fast" not in engines:
        print(json.dumps(registry.notes), file=sys.stderr)
        return 2
    requests = build_requests(args.model_id)
    t0 = time.perf_counter()
    inproc = determinism_report(lambda r: system_one_json(r, engines, cfg), requests, repeats=args.repeats)
    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": cfg.to_dict(),
        "registry": registry.describe(),
        "hardware": hardware_info(),
        "requests": [{"shape": i, "n_questions": len(r["questions"]),
                      "n_options": max(len(q.get("criteria") or []) if isinstance(q.get("criteria"), dict) else 0 for q in r["questions"].values())}
                     for i, r in enumerate(requests)],
        "inproc": inproc,
        "seconds_inproc": time.perf_counter() - t0,
    }
    if args.http:
        import httpx

        app = create_app(registry, config=cfg, allowed_hosts=LOCAL_HOSTS, warm_on_startup=False)
        with LiveServer(app) as srv, httpx.Client(base_url=srv.url, timeout=600) as client:
            def post(r):
                resp = client.post("/v1/systemone", json=r)
                resp.raise_for_status()
                return resp.json()

            t1 = time.perf_counter()
            report["http"] = determinism_report(post, requests, repeats=args.repeats)
            report["seconds_http"] = time.perf_counter() - t1
            report["http"]["run_facts"] = client.get("/x/v1/run").json()
            # in-process and HTTP must agree byte for byte
            a = [json.dumps(system_one_json(r, engines, cfg), sort_keys=True) for r in requests]
            b = [json.dumps(post(r), sort_keys=True) for r in requests]
            report["inproc_equals_http"] = a == b
    summary = {k: report["inproc"][k] for k in ("requests", "repeats", "identical_bytes_fraction", "repeat_flip_rate", "max_abs_diff_repeat", "latency_ms")}
    summary["permutation"] = {k: v for k, v in report["inproc"]["permutation"].items()}
    summary["inproc_equals_http"] = report.get("inproc_equals_http")
    summary["cpu"] = report["hardware"].get("cpu")
    summary["torch_threads"] = report["hardware"].get("torch_threads")
    print(json.dumps(summary, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
