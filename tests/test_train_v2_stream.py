"""stream.py: indexing (zst + plain, cache reuse), filters, bucket weights, sqrt sampling, caps, repeat
limits, determinism, DDP sharding / resume cursors and the length curriculum. Pure Python + numpy."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from jev_local.train.stream import Corpus, MixConfig, Mixture, build_file_index, discover


def _row(i: int, bucket: str, source: str, *, split: str = "train", lic: str = "commercial", label: str | None = None,
         nbytes: int = 0, n_q: int = 1, sub: str = "") -> dict:
    lab = label if label is not None else f"c{i % 4}"
    qs = {f"q{j}": {"type": "choice", "instructions": "pick", "criteria": {"c0": None, "c1": None, "c2": None, "c3": None}}
          for j in range(n_q)}
    return {"id": f"{source}-{i}", "split": split, "family": f"{bucket}/{source}", "state": "x" * nbytes,
            "questions": qs, "labels": {q: {"type": "choice", "label": lab} for q in qs},
            "meta": {"subsource": sub} if sub else {}, "source": source, "bucket": bucket, "license_use": lic,
            "variant": "", "group_id": ""}


def _write(path: Path, rows: list[dict], compress: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r) + "\n" for r in rows).encode()
    if not compress:
        path.write_bytes(data)
        return path
    try:
        import zstandard

        path.write_bytes(zstandard.ZstdCompressor(level=3).compress(data))
    except ImportError:
        if not shutil.which("zstd"):
            pytest.skip("needs zstandard or the zstd CLI")
        path.write_bytes(subprocess.run(["zstd", "-q", "-c"], input=data, stdout=subprocess.PIPE, check=True).stdout)
    return path


@pytest.fixture()
def root(tmp_path) -> Path:
    r = tmp_path / "v2"
    raw = r / "raw"
    # bucket A: two sources with 4:1 decisions -> sqrt sampling gives 2:1
    _write(raw / "bA" / "big.jsonl.zst", [_row(i, "bA", "big", nbytes=200) for i in range(800)]
           + [_row(i, "bA", "big", split="dev_mix", nbytes=200) for i in range(800, 820)])
    _write(raw / "bA" / "small.jsonl.zst", [_row(i, "bA", "small", nbytes=200) for i in range(200)])
    # bucket B: one source, research licence, long rows
    _write(raw / "bB" / "long.jsonl.zst", [_row(i, "bB", "long", lic="research", nbytes=6000) for i in range(100)])
    # bucket C: plain jsonl
    _write(raw / "bC" / "plain.jsonl", [_row(i, "bC", "plain", nbytes=400) for i in range(300)], compress=False)
    (raw / "bA" / "big.stats.json").write_text(json.dumps({"rows": 820}))
    return r


def _cfg(**kw) -> MixConfig:
    base = dict(seed=0, bucket_weights={"bA": 0.5, "bB": 0.25, "bC": 0.25}, max_repeat=100.0, batch_tokens=4096,
                pool=256, max_len=8192)
    base.update(kw)
    return MixConfig(**base)


def test_index_roundtrip_and_cache(root, tmp_path):
    cache = tmp_path / "cache"
    files = discover([root])
    assert [f.name for f in files] == ["big.jsonl.zst", "small.jsonl.zst", "long.jsonl.zst", "plain.jsonl"]
    c = Corpus.open([root], cache, workers=2)
    assert len(c) == 820 + 200 + 100 + 300
    assert {g[:2] for g in c.groups} == {("bA", "big"), ("bA", "small"), ("bB", "long"), ("bC", "plain")}
    for i in (0, 819, 1000, 1119, 1419):
        r = c.row(i)
        assert r["id"].split("-")[0] in ("big", "small", "long", "plain")
    assert c.row(0)["id"] == "big-0" and c.row(len(c) - 1)["id"] == "plain-299"
    # the plain file is indexed in place, the zst one decompressed into the cache
    assert c.files[3].data == str((root / "raw" / "bC" / "plain.jsonl").resolve())
    assert Path(c.files[0].data).parent == cache
    # second open reuses the index (same arrays); touching the source rebuilds
    fi = build_file_index(files[0], cache)
    assert np.array_equal(fi.offset, c.files[0].offset)
    _write(files[0], [_row(i, "bA", "big", nbytes=200) for i in range(10)])
    assert len(build_file_index(files[0], cache)) == 10
    c.close()


def test_filters(root, tmp_path):
    c = Corpus.open([root], tmp_path / "cache")
    m = Mixture(c, _cfg())
    assert len(m.eligible) == 800 + 200 + 100 + 300  # dev_mix rows excluded
    m2 = Mixture(c, _cfg(licenses=("commercial",)))
    assert len(m2.eligible) == 800 + 200 + 300
    m3 = Mixture(c, _cfg(splits=("dev_mix",)))
    assert len(m3.eligible) == 20
    m4 = Mixture(c, _cfg(buckets=("bC",)))
    assert len(m4.eligible) == 300
    for i in m2.plan(0).rows:
        assert c.row(int(i))["license_use"] == "commercial"


def test_bucket_token_shares_and_sqrt_sampling(root, tmp_path):
    c = Corpus.open([root], tmp_path / "cache")
    m = Mixture(c, _cfg(weight_unit="tokens"))
    s = m.plan_summary(0)
    sh = {b: v["token_share"] for b, v in s["by_bucket"].items()}
    assert sh["bA"] == pytest.approx(0.5, abs=0.02) and sh["bB"] == pytest.approx(0.25, abs=0.02)
    g = {x["source"]: x for x in s["groups"]}
    # within bA tokens follow n^0.5: big has 4x the decisions -> 2x the rows (equal row sizes)
    assert g["big"]["rows_taken"] / g["small"]["rows_taken"] == pytest.approx(2.0, rel=0.03)
    # alpha = 1 -> proportional
    m1 = Mixture(c, _cfg(alpha=1.0))
    g1 = {x["source"]: x for x in m1.plan_summary(0)["groups"]}
    assert g1["big"]["rows_taken"] / g1["small"]["rows_taken"] == pytest.approx(4.0, rel=0.03)


def test_caps_and_repeat_limit(root, tmp_path):
    c = Corpus.open([root], tmp_path / "cache")
    m = Mixture(c, _cfg(source_cap=100, class_cap=10, max_repeat=1.0))
    p = m.plan(0)
    for x in p.groups:
        assert x["rows_capped"] <= 40  # 4 classes x 10
        assert x["repeat"] <= 1.0 + 1e-9
    big = [c.row(int(i)) for i in p.rows if c.row(int(i))["source"] == "big"]
    labels = [r["labels"]["q0"]["label"] for r in big]
    assert max(labels.count(f"c{k}") for k in range(4)) <= 10
    # source cap counts decisions: 3-question rows hit a cap of 30 decisions after 10 rows
    _write(root / "raw" / "bD" / "multi.jsonl.zst", [_row(i, "bD", "multi", n_q=3, nbytes=100) for i in range(50)])
    c2 = Corpus.open([root], tmp_path / "cache2")
    m2 = Mixture(c2, _cfg(source_cap=30, buckets=("bD",)))
    assert m2.plan(0).groups[0]["rows_capped"] == 10
    # a capped group shows a different subset in the next pass
    m3 = Mixture(c2, _cfg(source_cap=30, buckets=("bD",), passes=2))
    a, b = ({c2.row(int(i))["id"] for i in m3.plan(k).rows} for k in (0, 1))
    assert a != b


def test_repeat_limit_waterfills(root, tmp_path):
    c = Corpus.open([root], tmp_path / "cache")
    # bB is tiny in tokens relative to its weight 0.9: limited to 2 repeats; nothing else in its bucket
    m = Mixture(c, _cfg(bucket_weights={"bA": 0.05, "bB": 0.9, "bC": 0.05}, max_repeat=2.0))
    g = {x["source"]: x for x in m.plan_summary(0)["groups"]}
    assert g["long"]["repeat"] <= 2.0 + 1e-9
    # within-bucket water-filling: small hits its limit, big takes the rest
    m2 = Mixture(c, _cfg(bucket_weights={"bA": 0.98, "bB": 0.01, "bC": 0.01}, max_repeat={"bA": 1.5, "*": 2.0}))
    g2 = {x["source"]: x for x in m2.plan_summary(0)["groups"]}
    assert g2["small"]["repeat"] <= 1.5 + 1e-9 and g2["big"]["repeat"] <= 1.5 + 1e-9
    assert g2["small"]["repeat"] == pytest.approx(1.5, abs=0.01)


def test_determinism_and_seed(root, tmp_path):
    c = Corpus.open([root], tmp_path / "cache")
    a, b = Mixture(c, _cfg()).plan(0), Mixture(c, _cfg()).plan(0)
    assert np.array_equal(a.order, b.order) and np.array_equal(a.starts, b.starts)
    d = Mixture(c, _cfg(seed=1)).plan(0)
    assert not np.array_equal(a.order, d.order)
    e = Mixture(c, _cfg(passes=2))
    assert not np.array_equal(e.plan(0).order, e.plan(1).order)


def test_token_budget_batches(root, tmp_path):
    c = Corpus.open([root], tmp_path / "cache")
    m = Mixture(c, _cfg(batch_tokens=4096))  # layout="tree": cut by estimated padded tree-layout slots
    p = m.plan(0)
    for j in range(p.n_micro):
        ids = p.micro(j)
        assert len(ids) == 1 or m.batch_cost(ids) <= 4096
    # every planned row appears exactly once in the micro-batch order
    assert np.array_equal(np.sort(p.order), np.sort(p.rows))
    # rows with long states are batched together (a pool's sort classes may meet in at most a few batches)
    big = c.st > 2000
    mixed = sum(1 for j in range(p.n_micro) if 0 < big[p.micro(j)].sum() < len(p.micro(j)))
    assert mixed <= -(-len(p.rows) // 256)
    assert m.plan_summary(0)["est_slots_per_token"] < 1.6
    md = Mixture(c, _cfg(batch_tokens=4096, layout="dense"))
    est = np.minimum(c.est, 8192)
    pd = md.plan(0)
    for j in range(pd.n_micro):
        ids = pd.micro(j)
        assert len(ids) == 1 or len(ids) * est[ids].max() <= 4096
    m2 = Mixture(c, _cfg(batch_rows=8, batch_tokens=None))
    assert set(np.diff(m2.plan(0).starts)[:-1]) <= {8}


def test_ddp_sharding_and_resume(root, tmp_path):
    c = Corpus.open([root], tmp_path / "cache")
    m = Mixture(c, _cfg(passes=2))
    full = list(m.iterate(0, 1, 1))
    for world, ga in ((2, 1), (3, 2)):
        per_rank = [list(m.iterate(r, world, ga)) for r in range(world)]
        n = min(len(x) for x in per_rank)
        assert all(len(x) == n for x in per_rank)  # equal micro-step counts on every rank
        steps = [[per_rank[r][i] for r in range(world)] for i in range(n)]
        for group in steps:
            ks, curs = {g[0] for g in group}, {g[1] for g in group}
            assert len(ks) == 1 and len(curs) == 1  # ranks agree on pass and cursor
        # disjoint and in global order: rank r of micro-step i holds global micro-batch cursor - world + r
        for group in steps:
            k, cur = group[0][0], group[0][1]
            for r, g in enumerate(group):
                assert np.array_equal(g[2], m.plan(k).micro(cur - world + r))
        assert m.total_steps(world, ga) == n // ga
    # resume from a cursor reproduces the tail of the uninterrupted stream (world-independent cursor)
    k, cur, _ = full[37]
    tail = list(m.iterate(0, 1, 1, start_pass=k, start_cursor=cur))
    assert [(a, b) for a, b, _ in tail] == [(a, b) for a, b, _ in full[38:]]
    two = list(m.iterate(1, 2, 1, start_pass=k, start_cursor=cur - cur % 2))
    assert np.array_equal(two[0][2], m.plan(k).micro(cur - cur % 2 + 1))


def test_cycle_past_passes(root, tmp_path):
    c = Corpus.open([root], tmp_path / "cache")
    m = Mixture(c, _cfg(passes=1))
    n = sum(1 for _ in m.iterate(0, 1, 1))
    it = m.iterate(0, 1, 1, cycle=True)
    ks = [next(it)[0] for _ in range(3 * n)]
    assert ks[n - 1] == 0 and ks[n] == 1 and max(ks) >= 2 and ks == sorted(ks)


def test_length_curriculum(root, tmp_path):
    c = Corpus.open([root], tmp_path / "cache")
    m = Mixture(c, _cfg(curriculum=((0.0, 1000), (0.5, 1000), (0.6, 8192)), pool=128))
    p = m.plan(0)
    assert np.array_equal(np.sort(p.order), np.sort(p.rows))  # deferred, never dropped
    first = p.order[: p.starts[p.n_micro // 4]]
    assert (c.est[first] <= 1000).all()
    assert (c.est[p.order] > 1000).any()
    s = m.plan_summary(0)
    long_rows = {x["source"]: x for x in s["groups"]}["long"]["rows_taken"]
    assert s["len_hist"]["1-4k"] == long_rows > 0  # the long rows (6000 bytes ~ 1.7k tokens)


def test_mixconfig_json(tmp_path):
    f = tmp_path / "m.json"
    f.write_text(json.dumps({"bucket_weights": {"b1_tasksource_jev": 0.5}, "curriculum": [[0, 1024], [0.2, 8192]],
                             "licenses": ["commercial"], "max_repeat": {"b1_tasksource_jev": 2, "*": 3}}))
    cfg = MixConfig.from_json(f, source_cap=20000)
    assert cfg.source_cap == 20000 and cfg.licenses == ("commercial",)
    assert cfg.weight("b1_tasksource_jev") == 0.5 and cfg.weight("b3_nli") == 0.05
    assert cfg.repeat("b1_tasksource_jev") == 2 and cfg.repeat("b9_cu") == 3
    assert cfg.cap_at(0.0) == 1024 and cfg.cap_at(0.1) == 1024 + (8192 - 1024) // 2 and cfg.cap_at(0.9) == 8192
    with pytest.raises(ValueError):
        MixConfig.from_json({"nope": 1})


def test_ddp_cost_balancing(root, tmp_path):
    c = Corpus.open([root], tmp_path / "cache")
    m = Mixture(c, _cfg(passes=1, batch_tokens=2048, balance=True))
    world, ga = 3, 2
    per_rank = [list(m.iterate(r, world, ga)) for r in range(world)]
    n = len(per_rank[0])
    assert all(len(x) == n for x in per_rank) and n % ga == 0
    plan = m.plan(0)
    cost = {}
    for s in range(0, n, ga):  # one optimizer step = ga micro-steps on every rank
        got = [[per_rank[r][s + j] for j in range(ga)] for r in range(world)]
        curs = {g[1] for rk in got for g in rk[-1:]}
        assert len(curs) == 1  # same resume cursor on every rank at the step boundary
        c0 = got[0][-1][1] - world * ga
        ids = sorted(np.concatenate([g[2] for rk in got for g in rk]).tolist())
        want = sorted(np.concatenate([plan.micro(i) for i in range(c0, c0 + world * ga)]).tolist())
        assert ids == want  # every micro-batch of the step used exactly once
        loads = [sum(m.batch_cost(g[2]) for g in rk) for rk in got]
        naive = [sum(m.batch_cost(plan.micro(c0 + j * world + r)) for j in range(ga)) for r in range(world)]
        cost.setdefault("bal", []).append(max(loads))
        cost.setdefault("naive", []).append(max(naive))
    assert sum(cost["bal"]) <= sum(cost["naive"])
    # resume from a step boundary reproduces the rest of the stream
    k, cur, _ = per_rank[1][2 * ga - 1]
    tail = list(m.iterate(1, world, ga, start_pass=k, start_cursor=cur))
    assert [x[1] for x in tail] == [x[1] for x in per_rank[1][2 * ga:]]
    assert all(np.array_equal(a[2], b[2]) for a, b in zip(tail, per_rank[1][2 * ga:]))
