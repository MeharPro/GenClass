"""scripts/jevbench.py plumbing: cache semantics and request-file order (pure Python, no network)."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("jevbench_cli", ROOT / "scripts" / "jevbench.py")
jb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jb)


def test_load_cache(tmp_path):
    p = tmp_path / "jev.jsonl"
    recs = [
        {"id": "a", "ok": True},
        {"id": "a", "ok": False, "permanent": False},  # a later failure never replaces an ok answer
        {"id": "b", "ok": False, "permanent": True, "status": 400},  # schema error: permanent, skipped on re-run
        {"id": "c", "ok": False, "permanent": True, "status": 402},  # old billing record: retried
        {"id": "d", "ok": False, "permanent": False, "account": True, "status": 402},
        {"id": "e", "ok": False, "permanent": False, "status": 429},
        {"id": "f", "ok": False, "permanent": False, "status": 503},
        {"id": "f", "ok": True},
    ]
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    c = jb.load_cache(p)
    assert set(c) == {"a", "b", "f"} and c["a"]["ok"] and c["f"]["ok"]
    assert set(jb.load_cache(p, keep_failed_permanent=False)) == {"a", "f"}


def test_request_files_variant_major(tmp_path):
    for role, key, v in [("test", "ag_news", "main"), ("test", "ag_news", "shuffle"), ("test", "sst2", "choice"),
                         ("test", "sst2", "main"), ("ref", "hellaswag", "main"), ("dev", "cb", "main")]:
        (tmp_path / role).mkdir(exist_ok=True)
        (tmp_path / role / f"{key}__{v}.jsonl.gz").write_bytes(b"")
    names = [p.name for p in jb.request_files(tmp_path, ["test", "ref"], ["main", "choice", "shuffle", "bare"])]
    assert names == ["ag_news__main.jsonl.gz", "sst2__main.jsonl.gz", "hellaswag__main.jsonl.gz",
                     "sst2__choice.jsonl.gz", "ag_news__shuffle.jsonl.gz"]
