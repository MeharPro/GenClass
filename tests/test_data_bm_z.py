"""bm-z-data clean Z mixture (jev_local/data/bm/{z_mixture,reference}.py): pure-Python parts.

The stage-1 (public_jev family) accounting over rows, the v2-vs-Z balance comparison, the mixture version patch,
the bundle manifest and the reference bench-dir assembly (symlinks over suite item files + jevbench request dirs).
No numpy, no network."""

from __future__ import annotations

import json
from pathlib import Path

from jev_local.data.bm import reference as R
from jev_local.data.bm import z_mixture as Z


def _row(i, source, hf_id=None, split="train", task=None):
    meta = {}
    if hf_id:
        meta["hf_id"] = hf_id
    if task:
        meta["task"] = task
    return {"id": f"{source}-{i}", "split": split, "bucket": "b1_tasksource_jev", "source": source,
            "state": f"state {i} with a few more words", "questions": {"q": {"type": "noul", "instructions": "ok?"}},
            "labels": {"q": {"type": "noul", "p": 1.0}}, "meta": meta}


def test_stage1_table_rows_counts_public_jev_family():
    rows = [
        _row(0, "tsj", hf_id="mteb/banking77"),  # jevbench test dataset -> test:banking77
        _row(1, "tsj", hf_id="cardiffnlp/tweet_topic_single"),  # retired dev -> public_jev:tweet_topic
        _row(2, "tsj", task="daily_dialog"),  # public_jev hand rule
        _row(3, "tsj", hf_id="nyu-mll/glue/mrpc", split="dev_mix"),  # clean dev -> dev:mrpc
        _row(4, "tsj", hf_id="some-org/an_unrelated_corpus"),
        _row(5, "tsj", hf_id="another/plain_source"),
    ]
    t = Z.stage1_table_rows(rows)
    assert t["kept"] == 2 and t["dropped"] == 4
    assert t["by_category"]["test"] == 1 and t["by_category"]["public_jev"] == 2 and t["by_category"]["dev"] == 1
    assert t["by_key"]["test:banking77"] == 1 and t["by_key"]["public_jev:tweet_topic"] == 1 and t["by_key"]["public_jev:daily_dialog"] == 1
    assert t["dropped_by_split"] == {"train": 3, "dev_mix": 1}
    assert "mteb/banking77" in t["top_names"]


def test_reason_parts():
    assert Z._reason_parts("public_jev:banking77 (mteb/banking77)") == ("public_jev", "banking77")
    assert Z._reason_parts("jev_labelled (x/y)") == ("jev_labelled", "")


def _stats(tokens_by_bucket, sources, rows=1000, version="v"):
    tot = sum(tokens_by_bucket.values())
    return {
        "version": version, "pass_tokens": {"value": tot, "binding_bucket": "b1"},
        "splits": {
            "train": {"total": {"rows": rows, "decisions": 2 * rows, "tokens": tot},
                      "by_bucket": {b: {"rows": rows // len(tokens_by_bucket), "tokens": t, "token_share": round(t / tot, 4)} for b, t in tokens_by_bucket.items()},
                      "by_source": {s: {"tokens": t} for s, t in sources.items()},
                      "kind_share": {"choice": 0.5, "noul": 0.4, "score": 0.1}, "k_bin_share": {"2": 0.3}},
            "dev_mix": {"total": {"rows": 10}}, "dev_family": {"total": {"rows": 5}},
        },
    }


def test_compare_stats_balance_and_lost_sources():
    v2 = _stats({"b1": 100, "b2": 50}, {"b1/a": 60, "b1/b": 40, "b2/c": 50}, rows=1000, version="mix-v2.0-no-b6")
    z = _stats({"b1": 80, "b2": 50}, {"b1/a": 15, "b2/c": 50, "b1/d": 65}, rows=900, version="mix-bmz-1.0-cleanZ")
    c = Z.compare_stats(z, v2, {"rows_kept": 1, "rows_dropped": 2, "dropped_by_category": {"public_jev": 2}, "dropped_by_key": {"public_jev:x": 2}})
    assert c["tokens"] == {"z": 130, "v2": 150, "ratio": round(130 / 150, 4)}
    assert c["by_bucket"]["b1"]["tokens_ratio"] == 0.8 and c["by_bucket"]["b2"]["tokens_ratio"] == 1.0
    assert c["sources"] == {"z": 3, "v2": 3, "lost": 1} and c["sources_lost_top"] == [{"source": "b1/b", "v2_tokens": 40}]
    assert c["sources_shrunk_top"][0]["source"] == "b1/a"
    assert c["public_jev_removed"]["rows_dropped"] == 2 and c["dev_mix_rows"] == {"z": 10, "v2": 10}


def test_patch_version():
    s = Z.patch_version({"version": "mix-v2.0-no-b6", "input": "/x/clean"}, extra={"reference": {"items": 3}})
    assert s["version"] == Z.MIX_VERSION and s["z_track"] == {"input": "/x/clean", "reference": {"items": 3}}


def test_bundle_manifest(tmp_path):
    (tmp_path / "data" / "bm" / "z" / "shards").mkdir(parents=True)
    (tmp_path / "data" / "bm" / "z" / "shards" / "a.zst").write_bytes(b"12345")
    (tmp_path / "data" / "bm" / "z" / "r.json").write_text("{}")
    m = Z.bundle_manifest([tmp_path / "data" / "bm" / "z" / "shards", tmp_path / "data" / "bm" / "z" / "r.json"], tmp_path)
    assert m["count"] == 2 and m["bytes"] == 7 and m["files"][0]["path"] == "data/bm/z/shards/a.zst"


def test_reference_collect_links_items_and_jevbench(tmp_path):
    ev = tmp_path / "eval"
    for suite in ("deusser", "lexglue"):
        d = ev / suite / "items"
        d.mkdir(parents=True)
        (d / f"{suite}__x.jsonl").write_text(json.dumps({"id": "a", "dataset": f"{suite}/x", "item": "0", "request": {"state": "s", "questions": {}}}) + "\n")
        (ev / suite / "manifest.json").write_text(json.dumps({"pins": {"hf:x": "abc"}}))
    jb1, jb2 = tmp_path / "jb1", tmp_path / "jb2"
    for jb in (jb1, jb2):
        for role in ("test", "dev"):
            (jb / role).mkdir(parents=True)
            (jb / role / f"{role}_ds__main.jsonl").write_text("{}\n")
    (jb2 / "dev" / "only_in_second__main.jsonl").write_text("{}\n")
    col = R.collect(ev, [jb1, jb2], tmp_path / "ref", suites=["deusser", "lexglue"])
    links = sorted(Path(f["link"]).relative_to(tmp_path / "ref" / "bench").as_posix() for f in col["files"])
    assert links == ["dev/jevbench__dev_ds__main.jsonl", "dev/jevbench__only_in_second__main.jsonl",
                     "test/deusser__deusser__x.jsonl", "test/jevbench__test_ds__main.jsonl", "test/lexglue__lexglue__x.jsonl"]
    assert all(Path(f["link"]).is_symlink() and Path(f["link"]).read_text() for f in col["files"])
    # a second collect rebuilds the bench dir from scratch (no stale links)
    col2 = R.collect(ev, [jb1], tmp_path / "ref", suites=["deusser"])
    assert len(col2["files"]) == 3
