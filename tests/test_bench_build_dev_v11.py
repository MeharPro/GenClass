"""jevbench v1.1 dev mappers added for ATIS and SciFact: string-only requests, programmatic label text, per-claim
SciFact regrouping, class names and registry agreement (pure Python; no datasets / torch)."""

import json

from jev_local.bench import build as B
from jev_local.bench import registry as R
from jev_local.bench import templates as T


def _ok(item):
    assert T.validate_request({"state": item.state, "questions": item.questions}) == []
    assert set(item.gold) <= set(item.questions)
    json.dumps(item.state), json.dumps(item.questions), json.dumps(item.meta)  # JSON-serialisable
    return item


# ------------------------------------------------------------------------------------------------ coverage


def test_every_counted_dataset_has_a_mapper():
    assert set(R.test_keys()) <= set(T.MAPPERS)
    assert set(R.dev_keys()) - {"tasksource_heldout"} <= set(T.MAPPERS)  # tasksource_heldout is built natively
    assert "atis" in T.MAPPERS and "scifact" in T.MAPPERS


def test_registry_entries_match_the_mappers():
    a, s = R.BY_KEY["atis"], R.BY_KEY["scifact"]
    assert (a.hf_id, a.config, a.splits, a.n_used, a.sampling) == ("tuetschek/atis", None, ("test",), 893, "all")
    assert (s.hf_id, s.config, s.splits, s.n_used, s.sampling) == ("allenai/scifact", "claims", ("validation",), 300, "all")
    assert a.question_kinds == s.question_kinds == ("choice",) and a.label_source == s.label_source == "programmatic"
    assert not a.shuffle_rotation and not a.bare_variant and not s.shuffle_rotation and not s.bare_variant  # main only
    assert R.SEED == 20261001


# ------------------------------------------------------------------------------------------------ ATIS

ATIS_ROWS = [
    ("test:0", {"id": 0, "intent": "flight", "text": " i want a flight from boston to denver ", "slots": "O O O O O B-fromloc.city_name O B-toloc.city_name"}),
    ("test:1", {"id": 1, "intent": "flight+airfare", "text": "flights and fares from boston to denver", "slots": "O"}),
    ("test:2", {"id": 2, "intent": "ground_service", "text": "ground transportation in denver", "slots": "O"}),
    ("test:3", {"id": 3, "intent": "airfare+flight", "text": "fares and flights", "slots": "O"}),
    ("test:4", {"id": 4, "intent": "flight_no", "text": "flight numbers from boston", "slots": "O"}),
]


def test_atis_class_names_are_the_intents_present_including_joint_ones():
    names = B.class_names(R.BY_KEY["atis"], ATIS_ROWS, None)
    assert names == ["airfare+flight", "flight", "flight+airfare", "flight_no", "ground_service"]
    assert B.class_names(R.BY_KEY["atis"], ATIS_ROWS[:1], None) == ["flight"]  # whatever the split contains, nothing more


def test_atis_desc_is_programmatic():
    assert T.atis_desc("flight") == "The user's request is about flight."
    assert T.atis_desc("ground_service") == "The user's request is about ground service."
    assert T.atis_desc("flight+airfare") == "The user's request is about both flight and airfare."
    assert T.atis_desc("airfare+flight") == "The user's request is about both airfare and flight."
    assert T.atis_desc("aircraft+flight+flight_no") == "The user's request is about all of aircraft, flight and flight no."


def test_atis_mapper():
    ctx = T.Ctx(names=B.class_names(R.BY_KEY["atis"], ATIS_ROWS, None))
    items = [_ok(T.m_atis(r, iid, ctx)) for iid, r in ATIS_ROWS]
    it = items[0]
    assert it.state == {"utterance": "i want a flight from boston to denver"}  # stripped, no slots
    q = it.questions["answer"]
    assert q["type"] == "choice" and list(q["criteria"]) == ctx.names
    assert q["criteria"]["flight+airfare"] == "The user's request is about both flight and airfare."
    assert all(isinstance(v, str) and v for v in q["criteria"].values())
    assert "`utterance`" in q["instructions"]
    assert it.gold["answer"] == {"type": "choice", "label": "flight"} and it.stratum == "flight"
    assert items[1].gold["answer"]["label"] == "flight+airfare" and items[3].gold["answer"]["label"] == "airfare+flight"
    assert all(i.gold["answer"]["label"] in i.questions["answer"]["criteria"] for i in items)
    assert it.template == "programmatic:atis" and it.extra == {}
    # sampling "all": every row is kept and no RNG is consulted
    assert B.sample_items(items, R.BY_KEY["atis"]) == items
    row = B.row_json("atis", "main", it, it.questions, it.gold)
    assert row["id"] == "atis/main/test:0" and row["item"] == "test:0" and row["request"]["state"] == it.state


# ------------------------------------------------------------------------------------------------ SciFact

CORPUS = {
    "11": {"doc_id": "11", "title": " Doc eleven. ", "abstract": [" First sentence.", "Second sentence. "]},
    "22": {"doc_id": "22", "title": "Doc twenty-two.", "abstract": ["Only sentence."]},
    "33": {"doc_id": "33", "title": "Doc thirty-three.", "abstract": ["Alpha.", "Beta."]},
}
FLAT = [  # the HF `claims` shape: one row per (claim, evidence abstract, rationale); "" label = no evidence
    {"id": 1, "claim": "Claim one.", "evidence_doc_id": "", "evidence_label": "", "evidence_sentences": [], "cited_doc_ids": [11]},
    {"id": 3, "claim": "Claim three.", "evidence_doc_id": "22", "evidence_label": "SUPPORT", "evidence_sentences": [0], "cited_doc_ids": [22]},
    {"id": 3, "claim": "Claim three.", "evidence_doc_id": "22", "evidence_label": "SUPPORT", "evidence_sentences": [0], "cited_doc_ids": [22]},
    {"id": 5, "claim": "Claim five.", "evidence_doc_id": "33", "evidence_label": "CONTRADICT", "evidence_sentences": [1], "cited_doc_ids": [11, 33]},
    {"id": 7, "claim": "Claim seven.", "evidence_doc_id": "11", "evidence_label": "SUPPORT", "evidence_sentences": [0], "cited_doc_ids": [11, 22]},
    {"id": 7, "claim": "Claim seven.", "evidence_doc_id": "22", "evidence_label": "CONTRADICT", "evidence_sentences": [0], "cited_doc_ids": [11, 22]},
    {"id": 9, "claim": "Claim nine.", "evidence_doc_id": "44", "evidence_label": "SUPPORT", "evidence_sentences": [0], "cited_doc_ids": [99]},
]


def test_scifact_claim_rows_regroups_one_row_per_claim():
    groups = T.scifact_claim_rows(FLAT, CORPUS)
    assert [g["id"] for g in groups] == [1, 3, 5, 7, 9]  # first-appearance order, 7 flat rows -> 5 claims
    g = {g["id"]: g for g in groups}
    assert g[1]["evidence"] == [] and [d["doc_id"] for d in g[1]["docs"]] == ["11"]
    assert g[3]["evidence"] == [["22", "SUPPORT"]]  # duplicate rationale rows collapse
    assert g[5]["evidence"] == [["33", "CONTRADICT"]] and [d["doc_id"] for d in g[5]["docs"]] == ["11", "33"]  # cited order
    assert g[7]["evidence"] == [["11", "SUPPORT"], ["22", "CONTRADICT"]]
    assert g[9]["cited_doc_ids"] == ["99", "44"] and g[9]["docs"] == []  # evidence doc appended; nothing in the corpus
    assert FLAT[3]["cited_doc_ids"] == [11, 33]  # input untouched


def test_scifact_mapper():
    groups = {g["id"]: g for g in T.scifact_claim_rows(FLAT, CORPUS)}
    ctx = T.Ctx(names=B.class_names(R.BY_KEY["scifact"], [], None))
    assert ctx.names == ["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"]
    it1 = _ok(T.m_scifact(groups[1], "validation:1", ctx))
    assert it1.gold["answer"] == {"type": "choice", "label": "NOT_ENOUGH_INFO"} and it1.stratum == "NOT_ENOUGH_INFO"
    assert it1.state == {"claim": "Claim one.", "abstracts": "1. Doc eleven.\nFirst sentence. Second sentence."}
    q = it1.questions["answer"]
    assert q["type"] == "choice" and list(q["criteria"]) == ctx.names and q["criteria"] == T.SCIFACT
    assert "`abstracts`" in q["instructions"] and "`claim`" in q["instructions"]
    assert it1.meta == {"claim_id": 1, "doc_ids": ["11"], "evidence_doc_ids": []}
    it3 = _ok(T.m_scifact(groups[3], "validation:3", ctx))
    assert it3.gold["answer"]["label"] == "SUPPORT" and it3.meta["evidence_doc_ids"] == ["22"]
    it5 = _ok(T.m_scifact(groups[5], "validation:5", ctx))
    assert it5.gold["answer"]["label"] == "CONTRADICT"
    assert it5.state["abstracts"] == "1. Doc eleven.\nFirst sentence. Second sentence.\n\n2. Doc thirty-three.\nAlpha. Beta."
    assert it5.meta["doc_ids"] == ["11", "33"] and it5.meta["evidence_doc_ids"] == ["33"]
    assert T.m_scifact(groups[7], "validation:7", ctx) is None  # conflicting labels across abstracts: dropped
    assert T.m_scifact(groups[9], "validation:9", ctx) is None  # no cited abstract in the corpus: dropped
    assert it1.template == it3.template == "programmatic:scifact (per claim)"
    row = B.row_json("scifact", "main", it3, it3.questions, it3.gold)
    assert row["id"] == "scifact/main/validation:3" and row["meta"]["claim_id"] == 3
    assert B.sample_items([it1, it3, it5], R.BY_KEY["scifact"]) == [it1, it3, it5]


def test_scifact_labels_are_fixed_strings():
    assert list(T.SCIFACT) == ["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"]
    assert all(isinstance(v, str) and v.endswith(".") for v in T.SCIFACT.values())
    assert "NOT_ENOUGH_INFO" in T.CLIMATE_FEVER  # same NEI key as the Climate-FEVER mapper
