"""Phase-2 assembly (jev_local/data/v2/assemble.py): mixture configs, statistics, the v1 GEN converter, the
binding-bucket pass size, and (VM only: forks a process pool) a small end-to-end build through the stream
reader. Light: pure Python + numpy, no torch."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from jev_local.data.v2 import assemble as A

np = pytest.importorskip("numpy")
stream = pytest.importorskip("jev_local.train.stream")


def test_plan_shares_and_configs_load():
    assert abs(sum(A.PLAN_SHARES.values()) - 0.99) < 1e-9 and "b6_synthetic" not in A.PLAN_SHARES
    cfg = stream.MixConfig.from_json(A.mix_config_json(A.base_mix(pass_tokens=123)))
    assert cfg.caps("b1_tasksource_jev") == (20_000, 2_000)
    assert cfg.caps("b3_nli") == (None, None)  # NLI sources are the planned units: no 20k cap
    assert cfg.repeat("b9_cu") == 1.0 and cfg.repeat("b5_open_jev") == 4.0 and cfg.pass_tokens == 123
    for name in A.ABLATIONS:
        d = A.ablation_mix(name)
        c = stream.MixConfig.from_json(A.mix_config_json(d))  # no unknown keys
        assert c.pass_tokens == A.ABLATION_TOKENS and set(c.buckets) == set(A.ABLATIONS[name][0])
    # cumulative: each ablation adds buckets to the previous one
    prev: set[str] = set()
    for name in sorted(A.ABLATIONS):
        cur = set(A.ABLATIONS[name][0])
        assert prev < cur
        prev = cur


def test_k_and_length_bins():
    assert [A.k_bin(k) for k in (2, 3, 5, 6, 20, 21, 255, 300)] == ["2", "3-5", "3-5", "6-20", "6-20", "21-255", "21-255", ">255"]
    assert [A.len_bucket(t) for t in (1, 512, 513, 1024, 5000, 99999)] == [512, 512, 1024, 1024, 8192, 8192]


def _row(i, bucket="b2_label_semantics", source="s", split="train", k=4, lic="commercial"):
    crit = {f"label{j}": None for j in range(k)}
    return {"id": f"{source}-{i}", "split": split, "family": f"{bucket}/{source}", "state": f"state text number {i} " * 3,
            "questions": {"c": {"type": "choice", "instructions": "Pick.", "criteria": crit},
                          "n": {"type": "noul", "instructions": "Is it?"}},
            "labels": {"c": {"type": "choice", "label": "label0"}, "n": {"type": "noul", "p": 0.3}},
            "meta": {}, "source": source, "bucket": bucket, "license_use": lic, "variant": "", "group_id": ""}


def test_mixstats():
    st = A.MixStats()
    st.add(_row(0), 100, 90)
    st.add(_row(1, bucket="b3_nli", k=30, lic="research"), 3000, 2900, repeat=1)
    js = st.to_json()
    assert js["total"] == {"rows": 2, "decisions": 4, "tokens": 3100, "tokens_chars4": 2990}
    assert js["kinds"] == {"choice": 2, "noul": 2, "score": 0}
    assert js["k_bins"]["3-5"] == 1 and js["k_bins"]["21-255"] == 1
    assert js["by_length_bucket"]["L4096"]["rows"] == 1 and js["by_license"]["research"]["tokens"] == 3000
    assert js["soft_label_decisions"] == 2 and js["repeat_index_rows"] == {"0": 1, "1": 1}
    other = A.MixStats()
    other.add(_row(2), 10, 10)
    st.merge(other)
    assert st.to_json()["total"]["rows"] == 3


def test_gen_v1_rows(tmp_path):
    src = tmp_path / "gen"
    src.mkdir()
    r = {"id": "gen-000001", "split": "train", "family": "gen/route/v0", "state": "s",
         "questions": {"q": {"type": "noul", "instructions": "x"}}, "labels": {"q": {"type": "noul", "p": 1.0}},
         "meta": {"task": "route", "heldout": False}}
    (src / "train.jsonl").write_text(json.dumps(r) + "\n")
    (src / "dev.jsonl").write_text(json.dumps({**r, "id": "gen-000002"}) + "\n")
    (src / "test.jsonl").write_text(json.dumps({**r, "id": "gen-000003"}) + "\n")
    rows = list(A.gen_v1_rows(src))
    assert [x["split"] for x in rows] == ["train", "dev_mix"]  # test stays out
    assert rows[0]["bucket"] == "gen_v1" and rows[0]["meta"]["cap_key"] == "gen_v1/route"


def test_auto_pass_tokens_binding_bucket():
    cap = {"a": {"max_pass_tokens": 1000}, "b": {"max_pass_tokens": 400}, "c": {"max_pass_tokens": None}}
    assert A.auto_pass_tokens(cap, slack=1.0) == (400, "b")


@pytest.mark.skipif(sys.platform == "darwin", reason="forks a process pool: VM only")
def test_build_end_to_end(tmp_path):
    """A small clean tree -> shards/{train,dev_mix,dev_family} + stats + passthrough config (VM: forks)."""
    from jev_local.data.v2.decontam import ZstWriter

    clean = tmp_path / "clean"
    specs = {("b1_tasksource_jev", "tsj"): 300, ("b2_label_semantics", "lab"): 200, ("b9_cu", "cu"): 200,
             ("s0_families", "s0_x"): 100}
    for (b, s), n in specs.items():
        w = ZstWriter(clean / b / f"{s}.jsonl.zst")
        for i in range(n):
            split = "dev_mix" if i % 50 == 0 else ("dev_family" if (s == "s0_x" and i % 7 == 0) else "train")
            w.write(_row(i, bucket=b, source=s, split=split))
        w.close()
    out = A.build(tmp_path, procs=2, cache=tmp_path / "cache", rows_per_part=150)
    tr = out["splits"]["train"]
    assert tr["total"]["rows"] > 0 and set(tr["by_bucket"]) <= set(A.PLAN_SHARES)
    assert out["pass_tokens"]["binding_bucket"] in A.PLAN_SHARES
    parts = sorted((tmp_path / "shards" / "train").glob("part-L*.jsonl.zst"))
    assert parts and all(p.with_name(p.name.replace(".jsonl.zst", ".stats.json")).is_file() for p in parts)
    assert (tmp_path / "shards" / "dev_mix").is_dir() and (tmp_path / "shards" / "dev_family").is_dir()
    # passthrough: the stream reader takes every shard row exactly once
    corpus = stream.Corpus.open([tmp_path / "shards" / "train"], tmp_path / "cache2", workers=1)
    cfg = stream.MixConfig.from_json(tmp_path / "shards" / "train" / "mix_passthrough.json")
    plan = stream.Mixture(corpus, cfg).plan(0)
    assert len(plan.rows) == len(corpus) == tr["total"]["rows"]
    assert len(set(plan.rows.tolist())) == len(corpus)
    # no dev rows in train shards, ids unique (repeats are suffixed)
    ids = [corpus.row(i)["id"] for i in range(len(corpus))]
    assert len(set(ids)) == len(ids) and all(corpus.row(i)["split"] == "train" for i in range(len(corpus)))
