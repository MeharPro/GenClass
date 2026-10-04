"""OursRunner equals FastEngine.evaluate, and option/question chunking is exact (needs weights: VM only)."""

import json
import os
import sys
from pathlib import Path

import pytest

CKPT = Path(os.environ.get("JEV_LOCAL_FAST_CKPT", Path.home() / "jev" / "models" / "jev-local-fast"))
pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(sys.platform == "darwin", reason="loads torch: VM only (CONTRACT Mac safety)"),
    pytest.mark.skipif(not (CKPT / "heads.safetensors").exists(), reason="no checkpoint"),
]

REQ = {
    "state": {"query": "I still have not received my new card, where is it?"},
    "questions": {
        "intent": {"type": "choice", "instructions": "Which intent best describes the bank customer's `query`?",
                   "criteria": {f"intent_{i}": f"This banking customer example message is about topic number {i}." for i in range(40)}},
        "urgent": {"type": "noul", "instructions": "Is `query` urgent?"},
        "sent": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "annoyed", "angry"]},
    },
}


@pytest.fixture(scope="module")
def runners():
    from jev_local.bench.ours import OursRunner

    return OursRunner(CKPT, threads=2), OursRunner(CKPT, threads=2, max_tokens=160)


def test_matches_engine(runners):
    from jev_local.schema import question_from_json

    full, _ = runners
    ans, info = full.answer(REQ)
    assert info["passes"] == 1 and not info["truncated"]
    res = full.eng.evaluate(REQ["state"], {k: question_from_json(v) for k, v in REQ["questions"].items()})
    for qid, d in res.dists.items():
        if d.kind == "noul":
            assert ans[qid]["noul"] == pytest.approx(d.probs[0], abs=1e-5)
        else:
            got = list(ans[qid]["probabilities"].values())
            assert got == pytest.approx(list(d.probs), abs=1e-5)


def test_chunking_is_exact(runners):
    full, small = runners
    a, _ = full.answer(REQ)
    b, info = small.answer(REQ)
    assert info["passes"] > 2 and not info["truncated"]
    for qid in a:
        if a[qid]["type"] == "noul":
            assert b[qid]["noul"] == pytest.approx(a[qid]["noul"], abs=1e-4)
        else:
            assert list(b[qid]["probabilities"]) == list(a[qid]["probabilities"])
            assert list(b[qid]["probabilities"].values()) == pytest.approx(list(a[qid]["probabilities"].values()), abs=1e-4)


def test_truncation_flags_long_state(runners):
    _, small = runners
    req = {"state": {"doc": "word " * 600, "q": "short"}, "questions": {"n": {"type": "noul", "instructions": "Is it?"}}}
    ans, info = small.answer(req)
    assert info["truncated"] and 0.0 <= ans["n"]["noul"] <= 1.0
    assert info["input_tokens"] <= 160


def test_bench_rows_run():
    """One row from each built test file runs (when the build output is present on this machine)."""
    root = Path.home() / "jev" / "bench" / "jevbench" / "test"
    if not root.exists():
        pytest.skip("no build output")
    from jev_local.bench.ours import OursRunner

    r = OursRunner(CKPT, threads=2)
    for p in sorted(root.glob("*__main.jsonl")):
        row = json.loads(p.open().readline())
        ans, info = r.answer(row["request"])
        assert set(ans) == set(row["request"]["questions"]), p.name
