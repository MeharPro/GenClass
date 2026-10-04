"""Phase-2 decontamination (jev_local/data/v2/decontam.py): normalisation, units, the reference index, the
per-row checks, the global decisions (stage 4 label sets, cross-source dedup) and the clean writer.

Light: pure Python plus numpy on a handful of synthetic items (no torch / datasets). The zstd round trip
uses python-zstandard or the zstd CLI and is skipped when neither exists."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from jev_local.data.v2 import decontam as D

np = pytest.importorskip("numpy")

LONG = ("The committee met on Tuesday to review the proposal for a new library in the northern district, "
        "and after a long debate the members voted to fund the first phase of construction next spring.")
SHORT_ITEM = "please wake me up at five am tomorrow morning"  # 9 words: containment stage
SST = "it 's a charming and often affecting journey ."
PREFACE = ("Before the main story, a long unrelated preface describes the bakery on the corner, its bread, "
           "its owners, the queue every morning and the smell of coffee. ")


# ------------------------------------------------------------------------------------------- normalisation
def test_normalize_ptb_and_raw_agree():
    assert D.exact_key(D.normalize_tokens(SST)) == D.exact_key(D.normalize_tokens("It's a charming, and often affecting journey!"))
    assert D.normalize_tokens("I do n't know") == D.normalize_tokens("I don't know") == ["i", "dont", "know"]
    assert D.normalize_tokens("ca n't stop") == ["cant", "stop"]
    assert D.normalize_tokens("Ｆｕｌｌ－width TEXT") == ["full", "width", "text"]  # NFKC
    assert D.exact_key(D.normalize_tokens("e-mail me")) == D.exact_key(D.normalize_tokens("email me"))
    assert D.normalize_tokens("") == []


def test_is_exact_unit_thresholds():
    assert not D.is_exact_unit(D.normalize_tokens("yes"))
    assert not D.is_exact_unit(D.normalize_tokens("go to bed"))  # 3 words but < 15 chars
    assert D.is_exact_unit(D.normalize_tokens("a gorgeous film indeed"))


def test_leaves_lines_and_units():
    state = {"claim": "Short claim here about bears", "evidence": "1. First evidence sentence is long enough.\n2. Second one."}
    assert D.leaves(state) == ["Short claim here about bears", state["evidence"]]
    assert D.lines_of(state["evidence"]) == ["First evidence sentence is long enough.", "Second one."]
    u = D.bench_units(state, {"q": {"type": "choice", "instructions": "x", "criteria": {"A": " ".join(["word"] * 25)}}},
                      line_units=True)
    kinds = [k for k, _ in u.exact]
    assert kinds.count("state") == 1 and kinds.count("field") == 2 and "line" in kinds and "option" in kinds
    row = {"state": "plain text state with several words", "questions": {"q": {"type": "noul", "instructions": "Is it?"}}}
    tu = D.train_units(row)
    assert [k for k, _ in tu.exact] == ["state"]
    assert ["is", "it"] in tu.grams  # question texts are scanned for n-grams


def test_choice_label_names_opaque_and_nota():
    q = {"type": "choice", "criteria": {"A": "Card Arrival: the card has not arrived", "opt_2": "no name here",
                                        "lost_or_stolen_card": None, "none of these": "none applies"}}
    assert D.choice_label_names(q) == ["card arrival", "lost or stolen card"]
    assert D.norm_label("Sci/Tech") == "sci tech"
    assert D.norm_label("AddToPlaylist") == "add to playlist"
    assert D.label_jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


# ------------------------------------------------------------------------------------------- reference
def _bench_dir(tmp: Path) -> Path:
    b = tmp / "jevbench"
    (b / "test").mkdir(parents=True)
    (b / "dev").mkdir()
    labels = ["joy", "anger", "fear", "sadness", "love", "surprise"]
    rows = []
    for i, text in enumerate([LONG, SST, SHORT_ITEM]):
        rows.append({"id": f"t{i}", "dataset": "toy", "variant": "main", "item": f"test:{i}",
                     "request": {"state": {"text": text}, "questions": {"answer": {
                         "type": "choice", "instructions": "Emotion?", "criteria": {k: None for k in labels}}}}})
    (b / "test" / "toy__main.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    dev = [{"id": "d0", "dataset": "toydev", "variant": "main", "item": "test:0",
            "request": {"state": "a completely different development item about planets and moons orbiting",
                        "questions": {"q": {"type": "noul", "instructions": "ok?"}}}}]
    (b / "dev" / "toydev__main.jsonl").write_text(json.dumps(dev[0]) + "\n")
    return b


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bench")
    return D.Reference.build(_bench_dir(tmp), tmp / "ref")


def _hits(ref, state, questions=None):
    return {(h[0], h[1]) for h in ref.check_row(D.train_units({"state": state, "questions": questions or {}}))}


def test_reference_meta_and_label_sets(ref):
    assert ref.meta["items"] == 4
    assert set(ref.label_sets["toy"]) == {D.norm_label(x) for x in ["joy", "anger", "fear", "sadness", "love", "surprise"]}
    assert ref.meta["ngram13"] > 0 and ref.meta["minhash_units"] > 0


def test_exact_hit_any_shape(ref):
    assert ("exact", "toy") in _hits(ref, "It's a charming and often affecting journey!")
    assert ("exact", "toy") in _hits(ref, {"review": SST, "meta": "other field"})  # field unit
    assert ("exact", "toy") in _hits(ref, "Header line of the doc\n" + SST + "\nfooter text")  # line unit


def test_ngram13_finds_embedded_long_text(ref):
    hits = _hits(ref, "Some unrelated preface. " + LONG[:-1] + " That was the news today in brief.")
    assert ("ngram13", "toy") in hits
    # in an option, too
    assert ("ngram13", "toy") in _hits(ref, "x", {"q": {"type": "choice", "instructions": "i",
                                                         "criteria": {"a": LONG, "b": None}}})


def test_containment_short_item(ref):
    assert ("contain", "toy") in _hits(ref, "User said: please wake me up at five am tomorrow morning thanks")


def test_minhash_near_duplicate(ref):
    near = LONG.replace("next spring", "next summer")  # one word: J ~ 0.94
    hits = _hits(ref, near)
    assert ("minhash", "toy") in hits
    assert ("exact", "toy") not in hits
    far = LONG.replace("Tuesday", "Wednesday").replace("spring", "summer")  # two edits far apart: J ~ 0.5
    assert ("minhash", "toy") not in _hits(ref, far)


def test_unrelated_text_has_no_hits(ref):
    assert _hits(ref, "The weather in the mountains was cold and windy, so the hikers turned back early.") == set()
    assert _hits(ref, "short") == set()


# ------------------------------------------------------------------------------------------- decisions
def _row(i, state, bucket, source, qs=None, labels=None, **kw):
    qs = qs or {"q": {"type": "noul", "instructions": "Is it about something?"}}
    labels = labels or {"q": {"type": "noul", "p": 1.0}}
    r = {"id": f"{source}-{i}", "split": "train", "family": f"{bucket}/{source}", "state": state, "questions": qs,
         "labels": labels, "meta": {}, "source": source, "bucket": bucket, "license_use": "commercial",
         "variant": "", "group_id": ""}
    r.update(kw)
    return r


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


def test_scan_decide_apply_end_to_end(tmp_path, ref):
    if not (shutil.which("zstd") or _has_zstandard()):
        pytest.skip("no zstd")
    raw = tmp_path / "raw"
    emo = ["joy", "anger", "fear", "sadness", "love", "surprise", "disgust"]
    choice = {"emotion": {"type": "choice", "instructions": "Emotion?", "criteria": {k: None for k in emo}}}
    shared = "A shared paragraph about the river festival that appears in two different training sources."
    _write(raw / "b2_label_semantics" / "emo.jsonl", [
        _row(i, f"diary entry number {i} about an ordinary afternoon at the park with friends", "b2_label_semantics",
             "emo", choice, {"emotion": {"type": "choice", "label": emo[i % 7]}}) for i in range(10)])
    _write(raw / "b1_tasksource_jev" / "tsj.jsonl", [
        _row(0, SST, "b1_tasksource_jev", "tsj/x"),  # exact
        _row(1, shared, "b1_tasksource_jev", "tsj/x"),  # dedup -> dropped (b2 copy wins)
        _row(2, "an innocent tasksource row about cooking pasta with garlic and olive oil", "b1_tasksource_jev", "tsj/x"),
        _row(3, "a row from an excluded source with enough words in it", "b1_tasksource_jev", "tsj/banking77_copy"),
        _row(4, PREFACE + LONG, "b1_tasksource_jev", "tsj/x"),  # 13-gram (J < 0.7: not a near-duplicate)
    ])
    _write(raw / "b2_label_semantics" / "other.jsonl", [_row(0, shared, "b2_label_semantics", "other")])
    _write(raw / "b9_cu" / "cu.jsonl", [_row(0, shared, "b9_cu", "cu"), _row(1, shared, "b9_cu", "cu")])
    D._REF = ref  # scan_task normally loads the pickled reference once per worker
    results = [D.scan_task((str(p), "", 0, 1, 512, "")) for p in D.discover_inputs([raw])]
    dec, summary = D.decide(results, ref.label_sets)
    tsj = str(raw / "b1_tasksource_jev" / "tsj.jsonl")
    assert dec.drop[tsj] == {0: "exact", 1: "dedup", 3: "stage1", 4: "ngram13"}
    assert str(raw / "b9_cu" / "cu.jsonl") not in dec.drop  # CU never deduplicated
    assert str(raw / "b2_label_semantics" / "other.jsonl") not in dec.drop
    flagged = list(dec.drop_q)
    assert flagged == [("b2_label_semantics", "emo", "", "emotion")]  # J = 6/7 with the toy set
    out = tmp_path / "clean"
    r = D.write_clean((str(raw / "b2_label_semantics" / "emo.jsonl"), str(out), {}, set(dec.drop_q), str(tmp_path / "eval")))
    assert r["rows"] == 0 and r["removed"] == {"labels": 10} and r["q_removed"] == 10
    r = D.write_clean((tsj, str(out), dec.drop[tsj], set(dec.drop_q), str(tmp_path / "eval")))
    assert r["rows"] == 1 and sum(r["removed"].values()) == 4
    kept = [json.loads(x) for x in D.iter_lines(out / "b1_tasksource_jev" / "tsj.jsonl.zst")]
    assert [k["id"] for k in kept] == ["tsj/x-2"]
    st = json.loads((out / "b1_tasksource_jev" / "tsj.stats.json").read_text())
    assert st["rows"] == 1 and st["decontam_removed"]["exact"] == 1


def test_strip_flagged_keeps_nouls():
    row = _row(0, "s", "b7_laurer", "laurer", {
        "c": {"type": "choice", "instructions": "x", "criteria": {"a": None, "b": None}},
        "n": {"type": "noul", "instructions": "y"}}, {"c": {"type": "choice", "label": "a"}, "n": {"type": "noul", "p": 0.0}},
        meta={"task": "t1"})
    out, removed = D.strip_flagged(row, {("b7_laurer", "laurer", "t1", "__meta_label_set__")})
    assert removed == ["c"] and list(out["questions"]) == ["n"] and list(out["labels"]) == ["n"]
    out, removed = D.strip_flagged(row, {("b7_laurer", "laurer", "t1", "other")})
    assert removed == [] and out is row


def test_stage1_recheck_names():
    row = _row(0, "s", "b2_label_semantics", "zsl_nli", meta={"task": "glue/rte"})
    assert D.stage1_reason(row).startswith("test:rte")
    assert D.stage1_reason(_row(0, "s", "b1_tasksource_jev", "tsj/sick/label")) is None
    # registry v1.1 (benchmax PLAN §2.3.2): public-Jev sources inside tasksource-jev are stage-1 exclusions on Z
    assert D.stage1_reason(_row(0, "s", "b1_tasksource_jev", "tsj/imdb")) == "public_jev:imdb (tsj/imdb)"


def _has_zstandard() -> bool:
    try:
        import zstandard  # noqa: F401

        return True
    except ImportError:
        return False


def test_common_ngram_rule_counts_distinct_documents(tmp_path, ref):
    """A benchmark 13-gram inside > 10 distinct training documents is boilerplate (GPT-3 rule) and kept;
    one contaminating document repeated in many rows still counts once and is dropped."""
    D._REF = ref
    phrase = " ".join(LONG.split()[:16])  # 16 words of the benchmark text -> 4 shared 13-grams
    rows = [_row(i, f"Distinct document {i} about topic {i * 7}: {phrase} and more words about item {i}.",
                 "b1_tasksource_jev", "tsj/x") for i in range(12)]
    rows += [_row(100 + i, "Repeated passage. " + phrase + " The end of this one passage.", "b1_tasksource_jev", "tsj/y")
             for i in range(15)]
    raw = tmp_path / "raw"
    _write(raw / "b1_tasksource_jev" / "t.jsonl", rows)
    results = [D.scan_task((str(p), "", 0, 1, 512, "")) for p in D.discover_inputs([raw])]
    dec, summary = D.decide(results, ref.label_sets)
    dropped = dec.drop.get(str(raw / "b1_tasksource_jev" / "t.jsonl"), {})
    # the shared grams occur in 12 + 1 = 13 distinct documents: all allow-listed, nothing dropped
    assert summary["common_grams"] and not dropped
    dec2, summary2 = D.decide([dict(results[0], hits=[h for h in results[0]["hits"] if h[0] >= 12])], ref.label_sets)
    assert set(dec2.drop[str(raw / "b1_tasksource_jev" / "t.jsonl")]) == set(range(12, 27))  # 1 document only


def test_line_units_only_for_list_states():
    state = {"evidence": "1. First independent evidence sentence here.\n2. Second independent evidence sentence."}
    assert any(k == "line" for k, _ in D.bench_units(state, {}, line_units=True).exact)
    assert not any(k == "line" for k, _ in D.bench_units(state, {}, line_units=False).exact)


def test_review_exclusions_and_source_priority(tmp_path, ref):
    D._REF = ref
    assert D.stage1_reason(_row(0, "s", "b1_tasksource_jev", "tsj/winodict")).startswith("review:sibling:winogrande")
    shared = "A passage that both tasksource-jev and zero-shot-label-nli carry, long enough to dedup."
    raw = tmp_path / "raw"
    _write(raw / "b1_tasksource_jev" / "tsj.jsonl", [_row(0, shared, "b1_tasksource_jev", "tsj/a")])
    _write(raw / "b2_label_semantics" / "zsl_nli.jsonl", [_row(0, shared, "b2_label_semantics", "zsl_nli")])
    # (registry v1.1 made `trec` a public_jev source, so the b2 original here is emocontext, which is still kept)
    _write(raw / "b2_label_semantics" / "emocontext.jsonl", [_row(0, "Another shared question text for the dedup test?",
                                                                  "b2_label_semantics", "emocontext")])
    _write(raw / "b1_tasksource_jev" / "tsj2.jsonl", [_row(0, "Another shared question text for the dedup test?",
                                                           "b1_tasksource_jev", "tsj/emocontext")])
    results = [D.scan_task((str(p), "", 0, 1, 512, "")) for p in D.discover_inputs([raw])]
    dec, _ = D.decide(results, ref.label_sets)
    assert dec.drop == {str(raw / "b2_label_semantics" / "zsl_nli.jsonl"): {0: "dedup"},
                        str(raw / "b1_tasksource_jev" / "tsj2.jsonl"): {0: "dedup"}}
