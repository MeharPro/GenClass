"""S track, agent s-verify: the independent audit (jev_local.data.s.audit) and the assembly (jev_local.data.s.assemble_s).
Pure Python + numpy + the zstd CLI; no torch / datasets / pyarrow. No process pools (the VM runs those)."""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import pytest

from jev_local.data.s import assemble_s as AS
from jev_local.data.s import audit as AU
from jev_local.data.v2.decontam import ZstWriter, exact_hex, exact_key, iter_lines, normalize_tokens

np = pytest.importorskip("numpy")
pytestmark = pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd CLI needed")


# ------------------------------------------------------------------------------------------------ fixtures

def _row(i, state, bucket="s_intent", source="unit", split="train", label="a", qs=None, meta=None, gid=None):
    q = qs or {"answer": {"type": "choice", "instructions": "Which intent?", "criteria": {"a": None, "b": None, "c": None}}}
    labels = {k: ({"type": "choice", "label": label} if v["type"] == "choice" else {"type": "noul", "p": 1.0}) for k, v in q.items()}
    return {"id": f"{source}-{i}", "split": split, "family": f"{bucket}/{source}", "state": state, "questions": q, "labels": labels,
            "meta": meta or {}, "source": source, "bucket": bucket, "license_use": "commercial", "variant": "", "group_id": gid or ""}


LONG = ("the quick brown fox jumps over the lazy dog while the farmer watches from the old wooden porch of his "
        "weathered farmhouse near the river and the children count the geese flying south above the golden barley "
        "fields before the first frost of the year")


def _write_zst(path: Path, rows):
    w = ZstWriter(path)
    for r in rows:
        w.write(r)
    w.close()
    return path


@pytest.fixture
def ref(tmp_path):
    items = [
        AU.EvalItem("intent/test", "test", "wake me up"),  # 3 words, 9 chars: not an exact unit, but a whole test item
        AU.EvalItem("intent/test", "test", "please transfer money to my savings account tomorrow morning"),
        AU.EvalItem("review/test", "test", LONG),
        AU.EvalItem("nli/test", "test", ["a man is sleeping on a bench in the park today", "nobody is awake at all now"], judged=("nobody is awake at all now",)),
        AU.EvalItem("ifq:esci@test", "test", ["wireless mouse", "Logitech M185 wireless mouse grey compact design long battery life for laptops"], judged=("wireless mouse",)),
        AU.EvalItem("short/test", "test", "contains eight words exactly in this little item"),  # 8 words -> containment
        AU.EvalItem("race/dev", "dev", ["the article text goes on about a boy and his very old dog for a while", "what is the best title for the passage",
                                        "a boy", "a dog"], judged=("the article text goes on about a boy and his very old dog for a while \x1f what is the best title for the passage",)),
    ]
    legacy = [("legacy:it/old", AU.key64_text("an old legacy only utterance here"), AU.KIND_WHOLE),
              ("legacy:it/jevbench-test/x__main", AU.key64_text("a legacy jevbench leaf that is only a component"), AU.KIND_COMPONENT)]
    return AU.SRef.build(items, legacy, tmp_path / "ref", log=lambda s: None)


# ------------------------------------------------------------------------------------------------ keys

def test_key64_matches_every_sibling_hash_format():
    toks = normalize_tokens("Hello, World! Don't stop.")
    k = AU.key64(toks)
    assert k == int.from_bytes(exact_key(toks)[:8], "little")  # decontam.exact_key prefix
    assert k == AU.key64_from_hex(exact_hex(toks))  # common.EvalRef full hex
    assert k == AU.key64_from_hex(hashlib.sha1("".join(toks).encode()).hexdigest()[:24])  # core.unit_hex (24 chars)
    assert AU.key64_text("Hello, World! Don't stop.") == k


# ------------------------------------------------------------------------------------------------ row units

def test_row_fields_primary_rules():
    # single-field dict: primary at any length
    f, empty = AU.row_fields(_row(0, {"text": "stop"}))
    assert f == [("stop", True)] and not empty
    # meta.primary names the field; the other field is not primary
    f, _ = AU.row_fields(_row(0, {"title": "A", "content": "b c"}, meta={"primary": ["title"]}))
    assert f == [("A", True), ("b c", False)]
    # multi-field without meta.primary: nothing primary
    f, _ = AU.row_fields(_row(0, {"premise": "p q r", "hypothesis": "h"}))
    assert f == [("p q r", False), ("h", False)]
    # JSON-string state (DMB NLU++ shape): message is primary
    f, _ = AU.row_fields(_row(0, json.dumps({"message": "good bye", "question": "Is the intent to end?", "instruction": "yes/no"})))
    assert ("good bye", True) in f and ("Is the intent to end?", False) in f
    # empty state (Decision Index shape): the instruction body after the first line is primary
    qs = {"answer": {"type": "choice", "instructions": "Classify the banking intent of this user request:\n\ntext: wake me up",
                     "criteria": {"option_0": "card_arrival", "option_1": "alarm"}}}
    f, empty = AU.row_fields(_row(0, {}, qs=qs))
    assert empty and f == [("text: wake me up", True)]


def test_train_units_shapes():
    u = AU.s_train_units(_row(0, {"text": LONG + "\nsecond line with six words here\nshort"}))
    kinds = [(k, p) for k, _, p in u.exact]
    assert ("state", True) in kinds and ("line", False) in kinds
    assert any(k == "state" for k, _ in u.minhash) and any(k == "line" for k, _ in u.minhash)
    assert u.grams and not u.empty_state
    # long question text (> 20 words) becomes an exact + minhash unit
    qs = {"q": {"type": "noul", "instructions": " ".join(["word"] * 25)}}
    u = AU.s_train_units(_row(0, {"text": "hi"}, qs=qs))
    assert any(k == "qtext" for k, _, _ in u.exact)


# ------------------------------------------------------------------------------------------------ decide

def test_decide_precedence_short_and_common():
    hits = [("short", "x/test", "field", "ok", 0), ("ngram13", "y/test", "text", "g", 123), ("exact", "z/test", "state", "s", 0)]
    assert AU.decide(hits) == ("exact", hits[2], 1)
    assert AU.decide(hits[:2]) == ("ngram13", hits[1], 1)
    assert AU.decide(hits[:2], common={123}) == (None, None, 1)  # allow-listed boilerplate n-gram: kept
    assert AU.decide([hits[0]]) == (None, None, 1)


# ------------------------------------------------------------------------------------------------ reference

def test_reference_build_and_check(ref):
    assert ref.meta["eval_sets"] == 6 and ref.meta["legacy_sets"] == 2 and ref.meta["items"] == 7
    assert ref.meta["exact_keys_by_kind"]["judged"] >= 2 and ref.meta["exact_keys_by_kind"]["component"] >= 3
    # exact at any length on a primary text; short non-primary -> "short"
    hits = ref.check_row(AU.s_train_units(_row(0, {"text": "Wake me up!"})))
    assert [(h[0], h[1]) for h in hits] == [("exact", "intent/test")]
    hits = ref.check_row(AU.s_train_units(_row(0, {"title": "wake me up", "body": "something entirely different here"})))
    assert [h[0] for h in hits] == ["short"]
    # judged leaf of a multi-field eval item (the hypothesis) as a training field -> exact; the premise (component) -> declared only
    hits = ref.check_row(AU.s_train_units(_row(0, {"premise": "something else entirely about the weather", "hypothesis": "Nobody is awake at all now."})))
    assert ("exact", "nli/test") in [(h[0], h[1]) for h in hits] and not any(h[0].startswith("component") for h in hits)
    assert AU.decide(hits)[0] == "exact"
    hits = ref.check_row(AU.s_train_units(_row(0, {"premise": "A man is sleeping on a bench in the park today.", "hypothesis": "A man naps."})))
    assert hits and all(h[0].startswith("component") and h[1] == "nli/test" for h in hits)  # exact + contain + minhash, all declared
    assert AU.decide(hits)[0] is None and AU.decide(hits)[2] == len(hits)
    # a training whole state equal to a component is still only a component (the test item's judged text is absent)
    hits = ref.check_row(AU.s_train_units(_row(0, {"text": "A man is sleeping on a bench in the park today."})))
    assert hits and all(h[0].startswith("component") for h in hits)
    # the whole multi-field item (premise + hypothesis) as a training state -> exact (whole)
    hits = ref.check_row(AU.s_train_units(_row(0, {"premise": "A man is sleeping on a bench in the park today.", "hypothesis": "Nobody is awake at all now."})))
    assert ("exact", "nli/test") in [(h[0], h[1]) for h in hits]
    # composite judged text (RACE article + question): the template question alone is a component, article + question is judged
    hits = ref.check_row(AU.s_train_units(_row(0, {"article": "An unrelated article about trains and stations in the north", "question": "What is the best title for the passage?"})))
    assert hits and all(h[0].startswith("component") for h in hits)
    hits = ref.check_row(AU.s_train_units(_row(0, {"article": "The article text goes on about a boy and his very old dog for a while.", "question": "What is the best title for the passage?"})))
    assert ("exact", "race/dev") in [(h[0], h[1]) for h in hits]  # whole item; the composite judged key equals the whole here
    # legacy component-kind keys never drop
    hits = ref.check_row(AU.s_train_units(_row(0, {"text": "A legacy jevbench leaf that is only a component"})))
    assert [h[0] for h in hits] == ["component"]  # exact-only: legacy keys have no MinHash / n-gram units
    # MinHash near-duplicate (one word changed in a 24-word text)
    near = LONG.replace("lazy dog", "lazy cat")
    stages = {h[0] for h in ref.check_row(AU.s_train_units(_row(0, {"text": near})))}
    assert "minhash" in stages and "exact" not in stages
    # 13-gram shared inside a longer text; containment of an 8-word item
    stages = {h[0] for h in ref.check_row(AU.s_train_units(_row(0, {"text": "Intro words first. " + LONG[:90] + " and then it ends differently"})))}
    assert "ngram13" in stages
    stages = {h[0] for h in ref.check_row(AU.s_train_units(_row(0, {"text": "He said: contains eight words exactly in this little item, indeed."})))}
    assert "contain" in stages
    # legacy safety net
    hits = ref.check_row(AU.s_train_units(_row(0, {"text": "An old legacy-only utterance here"})))
    assert hits and hits[0][1] == "legacy:it/old"
    # clean row
    assert ref.check_row(AU.s_train_units(_row(0, {"text": "a completely unrelated training sentence about kittens"}))) == []


def test_esci_exception_indexes_pair_and_query_only(ref):
    # the product title alone is NOT a unit (products recur across split by design) ...
    hits = ref.check_row(AU.s_train_units(_row(0, {"query": "ergonomic keyboard", "product": "Logitech M185 wireless mouse grey compact design long battery life for laptops"})))
    assert hits == []
    # ... but the whole (query, product) pair and the query are
    hits = ref.check_row(AU.s_train_units(_row(0, {"query": "wireless mouse", "product": "Logitech M185 wireless mouse grey compact design long battery life for laptops"})))
    assert ("exact", "ifq:esci@test") in [(h[0], h[1]) for h in hits]
    assert ref.meta["set_opts"]["ifq:esci@test"]["minhash"] is False


# ------------------------------------------------------------------------------------------------ scan / apply / rescan

def test_scan_apply_rescan_roundtrip(tmp_path, ref, monkeypatch):
    raw = tmp_path / "raw" / "intent"
    rows = [_row(0, {"text": "wake me up"}), _row(1, {"text": "a clean training utterance"}, gid="g1"),
            _row(2, {"text": LONG.replace("lazy dog", "lazy cat")}), _row(3, {"text": "another clean one about cats"}, split="dev_mix"),
            _row(4, {"title": "wake me up", "body": "short non primary match stays"})]
    p = _write_zst(raw / "unit.jsonl.zst", rows)
    (raw / "unit.stats.json").write_text(json.dumps({"rows": 5, "license_use": "commercial"}))
    scan_dir = tmp_path / "scan"
    scan_dir.mkdir()
    monkeypatch.setattr(AU, "_SREF", ref)
    monkeypatch.setattr(AU, "AUDIT", tmp_path / "audit")
    (tmp_path / "audit").mkdir()
    res = AU.scan_task((str(p), str(tmp_path / "ref"), 0, 1, 512, str(scan_dir)))
    assert res["rows"] == 5 and res["hits"] >= 3
    (scan_dir / "tasks.json").write_text(json.dumps([res]))
    results = AU.load_scan(scan_dir)
    common, info = AU.common_grams(results)
    assert common == set()
    dec = AU.decisions(results, common)
    assert set(dec[str(p)]) == {0, 2}  # row 4's hit is short-only
    clean, quar = tmp_path / "clean", tmp_path / "quarantine"
    w = AU.write_split((str(p), str(clean), str(quar), dec[str(p)]))
    assert w["kept"] == 3 and w["quarantined"] == 2 and w["by_stage"] == {"exact": 1, "minhash": 1}
    kept = [json.loads(l) for l in iter_lines(clean / "s_intent" / "unit.jsonl.zst")]
    assert [r["id"] for r in kept] == ["unit-1", "unit-3", "unit-4"]
    q = [json.loads(l) for l in iter_lines(quar / "s_intent" / "unit.jsonl.zst")]
    assert {r["meta"]["audit"]["stage"] for r in q} == {"exact", "minhash"} and all(r["meta"]["audit"]["eval_set"] for r in q)
    st = json.loads((clean / "s_intent" / "unit.stats.json").read_text())
    assert st["rows"] == 3 and st["audit"]["quarantined"] == 2 and st["license_use"] == "commercial"
    rep = AU.build_report(results, dec, common, info, [w], ref.meta, 0.1)
    assert rep["rows_scanned"] == 5 and rep["rows_kept"] == 3 and rep["rows_quarantined"] == 2
    assert rep["rows_with_short_nonprimary_matches_kept"] == 1 and rep["quarantined_explained_only_by_legacy_sets"] == 0
    json.dumps(rep, default=AU._json_default)
    # exit test on the clean file: no hits -> nothing refused; a planted hit -> the clean file is removed
    res2 = pickle.load(open(AU.scan_task((str(clean / "s_intent" / "unit.jsonl.zst"), str(tmp_path / "ref"), 0, 1, 512, str(scan_dir)))["file"], "rb"))
    assert AU.decisions([res2], set()) == {}  # the kept rows have no drop-stage hit (row 4's short match is still there, kept)
    bad = _write_zst(clean / "s_intent" / "bad.jsonl.zst", [_row(9, {"text": "wake me up"})])
    (clean / "s_intent" / "bad.stats.json").write_text("{}")
    monkeypatch.setattr(AU, "run_scan", lambda files, ref_dir, work, procs, log=print: None)
    res3 = pickle.load(open(AU.scan_task((str(bad), str(tmp_path / "ref"), 0, 1, 512, str(scan_dir)))["file"], "rb"))
    monkeypatch.setattr(AU, "load_scan", lambda work: [res2, res3])
    out = AU.run_rescan(clean, tmp_path / "ref", tmp_path / "rw", 1, tmp_path / "rescan.json", None, report_path=None, log=lambda s: None)
    assert out["exit_test_pass"] is False and list(out["refused_files"]) == ["s_intent/bad"]
    assert not bad.exists() and not (clean / "s_intent" / "bad.stats.json").exists()
    assert (clean / "s_intent" / "unit.jsonl.zst").exists()


def test_registry_coverage_maps_every_key():
    from jev_local.bench.registry import DEV, REF, RETIRED_DEV, TEST

    keys = {d.key for d in (*TEST, *DEV, *REF, *RETIRED_DEV)}
    assert keys <= set(AU.REGISTRY_COVERAGE)
    cov = AU.registry_coverage(["ag_news/test", "ifq:boolq@test", "jevbench-dev/atis__main"])
    assert cov["unmapped_keys"] == [] and {"ag_news", "boolq", "atis"} <= set(cov["covered"])
    assert "llm_aggrefact" in cov["uncovered"]


# ------------------------------------------------------------------------------------------------ assemble

def test_class_key_and_pooling():
    assert AS.class_key(_row(0, "s", label="b")) == "answer=b"
    # opaque keys resolve to the label name behind them (Deusser letters with descriptions, DI option_i, BTZSC label_000)
    q = {"answer": {"type": "choice", "instructions": "x", "criteria": {"A": "World: news about politics", "B": "Sports: games"}}}
    assert AS.class_key(_row(0, "s", qs=q, label="B")) == "answer=Sports"
    q = {"answer": {"type": "choice", "instructions": "x", "criteria": {"option_0": "card_arrival", "option_1": "pin_blocked"}}}
    assert AS.class_key(_row(0, "s", qs=q, label="option_1")) == "answer=pin_blocked"
    q = {"answer": {"type": "choice", "instructions": "x", "criteria": {"label_000": "This example news text is about sports", "label_001": "x"}}}
    assert AS.class_key(_row(0, "s", qs=q, label="label_000")) == "answer=This example news text is about sports"
    two = {"c": {"type": "choice", "instructions": "x", "criteria": {"a": None, "b": None}}, "n": {"type": "noul", "instructions": "y"}}
    assert AS.class_key(_row(0, "s", qs=two, label="a")) == "c=a"  # the choice wins
    nouls = {f"n{i}": {"type": "noul", "instructions": "y"} for i in range(5)}
    assert AS.class_key(_row(0, "s", qs=nouls)) is None  # multi-noul: no class
    one = {"n": {"type": "noul", "instructions": "y"}}
    assert AS.class_key(_row(0, "s", qs=one)) == "n=p1"
    r = _row(0, "s")
    r["labels"] = {"answer": {"type": "choice", "dist": {"a": 0.5, "b": 0.5}}}
    assert AS.class_key(r) is None
    assert AS.pool_rare({"a": 500, "b": 500, "c": 1}) == {"a": "a", "b": "b", "c": "_rare"}


def test_waterfill_balances_and_respects_cap():
    q = AS.waterfill({"a": 1000, "b": 300, "c": 50}, 600)
    assert sum(q.values()) == 600 and q["c"] == 50 and q["b"] == 275 and q["a"] == 275
    q = AS.waterfill({"a": 10, "b": 10}, 100)
    assert q == {"a": 10, "b": 10}
    q = AS.waterfill({"a": 7, "b": 7, "c": 7}, 10)  # remainder goes to the classes with rows left
    assert sum(q.values()) == 10 and max(q.values()) - min(q.values()) <= 1


def test_plan_rows_cap_balance_and_group_carve():
    rows = []
    i = 0
    for cls, n in (("a", 3000), ("b", 600), ("c", 60)):
        for j in range(n):
            rows.append((i, "train", f"q={cls}", f"g{i // 3}"))  # 3 renderings per source item
            i += 1
    for j in range(20):
        rows.append((i, "dev_mix", "q=a", f"g{i}"))
        i += 1
    p = AS.plan_rows(rows, "unit", cap=1200, dev_frac=0.02, dev_floor=10)
    assert p["cap_applied"] and p["train_rows_in"] == 3660 and p["rows_cut_by_cap"] == 2460
    hist = p["class_hist_after"]
    assert hist["q=c"] == 60 and abs(hist["q=a"] - hist["q=b"]) <= 30  # equal quotas (570 each) minus the group carve
    assert p["skew_after"] < p["skew_before"]
    assert p["train_rows"] + p["dev_carve_rows"] == 1200 and p["dev_rows"] == 20 + p["dev_carve_rows"]
    assert p["dev_carve_rows"] >= 24
    # group integrity: every group is wholly in train or wholly in dev
    train, dev = set(p["train_idx"]), set(p["dev_idx"])
    assert not (train & dev)
    g_of = {idx: g for idx, _, _, g in rows}
    for g in {g_of[i] for i in p["carve_idx"]}:
        members = {idx for idx, _, _, gg in rows if gg == g}
        assert members <= dev
    # deterministic
    assert AS.plan_rows(rows, "unit", cap=1200, dev_frac=0.02, dev_floor=10)["train_idx"] == p["train_idx"]
    # groups sharing a normalised state are united: a state used by two group ids never straddles train / dev
    shared = [(k, "train", "q=a", f"g{k}", 1000 + (k % 7)) for k in range(70)]  # 7 states, 10 groups each
    ps2 = AS.plan_rows(shared, "shared", cap=1000, dev_frac=0.1)
    dev_states = {1000 + (k % 7) for k in ps2["dev_idx"]}
    train_states = {1000 + (k % 7) for k in ps2["train_idx"]}
    assert ps2["dev_idx"] and not (dev_states & train_states)
    # a converter dev_mix row pulls the rest of its (united) group into dev
    mixed = [(0, "dev_mix", "q=a", "g0", 7), (1, "train", "q=a", "g1", 7), (2, "train", "q=a", "g2", 8)]
    ps3 = AS.plan_rows(mixed, "mixed", cap=100, dev_frac=0.0, dev_floor=0)
    assert ps3["dev_idx"] == [0, 1] and ps3["train_idx"] == [2] and ps3["dev_mix_pulled_train_rows"] == 1
    assert AS.merge_groups([(0, "train", None, "a", 5), (1, "train", None, "b", 5), (2, "train", None, "c", 0)])["a"] == AS.merge_groups(
        [(0, "train", None, "a", 5), (1, "train", None, "b", 5), (2, "train", None, "c", 0)])["b"]
    # below the cap nothing is dropped; the floor gives 10 dev rows
    small = [(k, "train", "q=a" if k % 2 else "q=b", f"g{k}") for k in range(100)]
    ps = AS.plan_rows(small, "small", cap=1000)
    assert not ps["cap_applied"] and ps["rows_cut_by_cap"] == 0 and ps["dev_carve_rows"] == 10 and ps["train_rows"] == 90


def test_write_file_splits_and_stats(tmp_path):
    rows = [_row(i, {"text": f"utterance number {i}"}, split=("dev_mix" if i == 4 else "train"), gid=f"g{i // 2}") for i in range(6)]
    p = _write_zst(tmp_path / "clean" / "s_intent" / "unit.jsonl.zst", rows)
    out = AS.write_file((str(p), str(tmp_path / "shards"), "s_intent", "unit", [0, 1, 2, 3], [4, 5], [5]))
    tr = [json.loads(l) for l in iter_lines(tmp_path / "shards" / "train" / "s_intent" / "unit.jsonl.zst")]
    dv = [json.loads(l) for l in iter_lines(tmp_path / "shards" / "dev" / "s_intent" / "unit.jsonl.zst")]
    assert [r["id"] for r in tr] == ["unit-0", "unit-1", "unit-2", "unit-3"] and all(r["split"] == "train" for r in tr)
    assert [(r["id"], r["split"], r["meta"]["s_dev_origin"]) for r in dv] == [("unit-4", "dev", "dev_mix"), ("unit-5", "dev", "carve")]
    st = json.loads((tmp_path / "shards" / "train" / "s_intent" / "unit.stats.json").read_text())
    assert st["rows"] == 4 and st["decisions"] == 4 and st["tokens_est"] > 0 and st["split"] == "train"
    assert out["train"]["rows"] == 4 and len(out["keys"]["train"]) == 4


def test_mix_config_shares():
    mix = AS.mix_config({"s_intent": 100, "s_topic": 300, "b1_tasksource_jev": 2000, "b9_cu": 2000})
    w = mix["bucket_weights"]
    assert abs(w["s_intent"] + w["s_topic"] - 0.7) < 1e-6 and abs(w["b1_tasksource_jev"] + w["b9_cu"] - 0.3) < 1e-6
    assert abs(w["s_topic"] / w["s_intent"] - 3) < 1e-3 and mix["pass_tokens"] == round(400 / 0.7)
    assert mix["max_repeat"]["s_intent"] == AS.S_MAX_REPEAT and mix["max_repeat"]["b9_cu"] == 1.0
    only_s = AS.mix_config({"s_intent": 100, "s_topic": 300})
    assert abs(sum(only_s["bucket_weights"].values()) - 1.0) < 1e-6 and only_s["pass_tokens"] == 400
    stream = pytest.importorskip("jev_local.train.stream")
    cfg = stream.MixConfig.from_json(mix)
    assert cfg.repeat("s_topic") == AS.S_MAX_REPEAT and cfg.caps("s_topic") == (None, None) and cfg.pass_tokens == mix["pass_tokens"]


def test_disclosure_markdown_handles_all_manifest_styles():
    mans = {
        "s_sentiment/sst2": {"dataset": "sst2", "hf_id": "stanfordnlp/sst2", "revision": "8d51e7e4887a", "license": "unknown (SST)", "license_use": "unknown",
                             "rows_written": 66375, "train_splits_used": [{"hf_split": "train", "path": "x"}], "rows_by_hf_split": {"train": 66375},
                             "evaluated_split": [{"name": "sst2/validation", "evaluated_by": ["deusser", "jevbench-test"]}], "registry_stage1_reason": "test:sst2"},
        "s_intent/nlupp": {"dataset": "nlupp", "hf_id": "PolyAI", "revision": "57ec275d", "license": "CC BY 4.0", "license_use": "commercial", "rows_written": 3155,
                           "train_splits_used": [{"hf_split": f"fold{i}"} for i in range(18)], "rows_by_hf_split": {f"fold{i}": 170 for i in range(18)},
                           "evaluated_splits_never_trained_on": [{"name": "nlupp_banking/fold18", "evaluated_by": ["dmb"]}], "registry_stage1_reason": "public_jev:nlupp"},
        "s_inference/boolq": {"dataset": "boolq", "hf_id": "google/boolq", "revision": "35b264d0", "license": "cc-by-sa-3.0", "license_use": "commercial", "rows": 9421,
                              "train_splits_used": ["train"], "evaluated_splits_never_trained_on": [{"config": "default", "split": "validation", "role": "test",
                                                                                                    "evaluated_by": "Deusser + jevbench test #14"}],
                              "stage1_registry_reason": "test:boolq"},
    }
    audit = {"files": {"s_sentiment/sst2": {"quarantined": 12}, "s_inference/boolq": {"quarantined": 0}}, "reference": {"eval_sets": 3, "items": 10, "exact_keys": 10,
             "ngram13": 1, "minhash_units": 2, "legacy_sets": 1}, "rows_scanned": 100, "rows_quarantined": 12, "quarantined_by_stage": {"exact": 12},
             "rows_with_short_nonprimary_matches_kept": 3, "quarantined_explained_only_by_legacy_sets": 0, "exit_test": {"exit_test_pass": True, "refused_files": {}},
             "common_ngram_rule": {"allow_listed": 0}, "policy": {"exceptions": {"ifq:esci@test": "pair + query only"}}}
    assembled = {"datasets": {"s_sentiment/sst2": {"train_rows": 48000, "dev_rows": 1600}}}
    mixture = {"replay": {"root": "/x/v2/shards/train", "note": "FALLBACK", "share": 0.3}, "mix_config": {"pass_tokens": 1000},
               "totals": {"train": {"rows": 1, "decisions": 2, "tokens_chars4": 3}, "dev": {"rows": 4}}}
    md = AS.disclosure_markdown(mans, audit, assembled, mixture)
    assert "| sst2 | s_sentiment | stanfordnlp/sst2 @8d51e7e4 | train (66,375) | sst2/validation (deusser, jevbench-test) | 66,375 | 12 | 48,000 | 1,600 |" in md
    assert "fold0-17" in md and "default/validation (Deusser + jevbench test #14)" in md and "test:boolq (S override)" in md
    assert "| **total** | 3 datasets |" in md and "FALLBACK" in md and "Exception: `ifq:esci@test`" in md
    assert "atis, snips, newsgroups20" in md


def test_pick_replay_prefers_bm_z(tmp_path, monkeypatch):
    bm = tmp_path / "bm" / "z" / "shards" / "train"
    v2 = tmp_path / "v2" / "shards" / "train"
    monkeypatch.setattr(AS, "REPLAY_CANDIDATES", (bm, v2))
    assert AS.pick_replay()[0] is None
    v2.mkdir(parents=True)
    _write_zst(v2 / "part-00000.jsonl.zst", [_row(0, "x", bucket="b1_tasksource_jev")])
    root, note = AS.pick_replay()
    assert root == v2 and "FALLBACK" in note
    bm.mkdir(parents=True)
    _write_zst(bm / "part-00000.jsonl.zst", [_row(0, "x", bucket="b1_tasksource_jev")])
    assert AS.pick_replay()[0] == bm
    assert AS.pick_replay(str(v2)) == (v2, "explicit")
