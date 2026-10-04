"""b8_extractive: span candidates, SQuAD/CoNLL renderers and synthetic records (pure Python)."""

from collections import Counter

from jev_local.data.v2 import extractive as X
from jev_local.data.v2.label_semantics import clean_qs
from jev_local.data.v2.render import NOTA_KEY, make_example, stable_rng
from jev_local.schema import SystemOneRequest
from jev_local.validate import validate_limits


def _check(state, qs):
    qs = clean_qs(qs)
    ex = make_example("x", state, qs, source="unit", bucket=X.BUCKET, license_use="commercial")
    validate_limits(SystemOneRequest.model_validate({"state": ex["state"], "model": "m", "questions": ex["questions"]}))
    for qid, lab in ex["labels"].items():
        if lab["type"] == "choice":
            assert lab["label"] in ex["questions"][qid]["criteria"]
    return ex


def test_candidate_spans_and_types():
    s = X.candidate_spans("Beyoncé Knowles (born September 4, 1981) sold 11 million copies, earning $1,200.50, "
                          "in Houston, Texas. The University of Notre Dame opened.")
    for want in ("Beyoncé Knowles", "September 4, 1981", "1981", "11 million", "$1,200.50", "Houston", "Texas",
                 "University of Notre Dame"):
        assert want in s, want
    assert "The" not in s and all(not x.endswith(",") for x in s)
    assert [X.span_type(x) for x in ("1981", "September 4, 1981", "11 million", "Houston")] == ["date", "date", "number", "name"]


def test_distractors_never_overlap_gold():
    pool = ["1981", "September 4, 1981", "Houston", "Texas", "2003", "11 million", "Houston, Texas"]
    for i in range(50):
        d = X.pick_distractors("Houston", pool, stable_rng("d", i), 4, forbid=["Houston, Texas"])
        assert all(not X.overlaps(x, "Houston") for x in d) and "Houston, Texas" not in d


SQUAD_ITEM = {"id": "s1", "title": "Beyoncé", "context": "Beyoncé Giselle Knowles-Carter (born September 4, 1981) is an "
              "American singer. Born and raised in Houston, Texas, she rose to fame in the late 1990s as lead singer of "
              "Destiny's Child. In 2003 she released Dangerously in Love.",
              "qs": [{"id": "a", "q": "When was Beyoncé born?", "answers": ["September 4, 1981"]},
                     {"id": "b", "q": "Where was she raised?", "answers": ["Houston, Texas"]},
                     {"id": "c", "q": "What did she release in 1999?", "answers": []}],
              "other_answers": ["September 4, 1981", "Houston, Texas"]}


def test_squad_renderer_nota_for_unanswerable():
    seen_nota = 0
    for i in range(40):
        state, qs, fam = X.r_squad(SQUAD_ITEM, stable_rng("sq", i))
        ex = _check(state, qs)
        for qid, q in ex["questions"].items():
            if q["instructions"].endswith(("released in 1999?", "1999? Pick the answer that appears in the passage.")):
                assert ex["labels"][qid]["label"] == NOTA_KEY
                seen_nota += 1
            if "born" in q["instructions"] and q["type"] == "choice" and ex["labels"][qid]["label"] != NOTA_KEY:
                assert ex["labels"][qid]["label"] == "September 4, 1981"
                assert "1981" not in q["criteria"]  # overlapping span is never a distractor
    assert seen_nota > 0


def test_conll_entities_and_renderer():
    toks = ["EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", ".", "Peter", "Blackburn", "said",
            "in", "Brussels", "."]
    tags = [1, 0, 2, 0, 0, 0, 2, 0, 0, 3, 4, 0, 0, 5, 0]
    ents = X.conll_entities(toks, tags)
    assert ents == [("EU", "ORG"), ("German", "MISC"), ("British", "MISC"), ("Peter Blackburn", "PER"), ("Brussels", "LOC")]
    assert X.detok(toks) == "EU rejects German call to boycott British lamb. Peter Blackburn said in Brussels."
    it = {"id": "c", "text": X.detok(toks), "entities": ents}
    kinds = Counter()
    for i in range(60):
        state, qs, _ = X.r_conll(it, stable_rng("c", i))
        ex = _check(state, qs)
        kinds.update(list(ex["questions"]))
        if "entity_type" in ex["labels"]:
            assert ex["labels"]["entity_type"]["label"] in ex["questions"]["entity_type"]["criteria"]
    assert kinds["entity"] > 0 and kinds["entity_type"] > 0


def test_records_labels_valid_and_dev_families():
    fams = Counter()
    nota = 0
    for it in X.gen_records(1500, seed=3):
        state, qs, fam = X.r_records(it, stable_rng("r", it["id"]))
        ex = _check(state, qs)
        fams[fam] += 1
        nota += sum(1 for l in ex["labels"].values() if l.get("label") == NOTA_KEY)
        # every gold person/amount/date option literally appears in the state (extractive by construction)
        s = str(state)
        for qid, lab in ex["labels"].items():
            if lab["type"] == "choice" and lab["label"] != NOTA_KEY:
                gold = clean_qs(qs)[qid].key_map[lab["label"]]
                assert gold in s, (fam, qid, gold)
    assert set(fams) == set(X.RECORD_FAMILIES) and nota > 20
    assert set(X.DEV_FAMILIES) <= set(X.RECORD_FAMILIES)


def test_records_deterministic():
    a = [X.r_records(it, stable_rng("d", it["id"]))[0] for it in X.gen_records(20, seed=9)]
    b = [X.r_records(it, stable_rng("d", it["id"]))[0] for it in X.gen_records(20, seed=9)]
    assert a == b
