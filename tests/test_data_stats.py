"""stats.py: the validator catches broken labels, and the corpus report runs end to end."""

from __future__ import annotations

import copy
import json

import pytest

from jev_local.data import stats as ST
from jev_local.data import synth_cu, synth_gen


@pytest.fixture(scope="module")
def corpora(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpora")
    synth_cu.generate(150, d / "cu", seed=2, log_every=0)
    synth_gen.generate(150, d / "gen", seed=2)
    return d / "cu", d / "gen"


def _first_cu(cu):
    return next(e for e in ST.iter_corpus(cu) if "target" in e["labels"] and "text_span" in e["questions"])


def test_valid_examples_pass(corpora):
    for d in corpora:
        for ex in ST.iter_corpus(d):
            assert ST.validate_example(ex) == [], ex["id"]


@pytest.mark.parametrize(
    "mutate, expect",
    [
        (lambda e: e["labels"]["intent"].update(label="dance"), "not an option"),
        (lambda e: e["labels"]["complete"].update(p=1.5), "bad noul p"),
        (lambda e: e["labels"].update(ghost={"type": "noul", "p": 0.0}), "label without question"),
        (lambda e: e["labels"]["app"].update(type="noul"), "label type"),
        (lambda e: e["labels"].update(scroll_amount={"type": "score", "level": 7}), "scroll_amount"),
        (lambda e: e["state"].update(transcript=e["state"]["transcript"] + " extra words here"), "options !="),
        (lambda e: e["labels"]["text_span"].update(label="words nobody said"), "text_span"),
        (lambda e: e.pop("family"), "missing key family"),
    ],
)
def test_validator_catches_corruption(corpora, mutate, expect):
    ex = copy.deepcopy(_first_cu(corpora[0]))
    mutate(ex)
    errs = ST.validate_example(ex)
    assert any(expect in e for e in errs), errs


def test_validator_checks_injected_gold(corpora):
    ex = copy.deepcopy(_first_cu(corpora[0]))
    eid = next(iter(ex["questions"]["target"]["criteria"]))
    ex["meta"]["injected"] = [eid]
    ex["meta"]["gold"].pop("ref", None)
    ex["labels"]["target"] = {"type": "choice", "label": eid}
    assert any("injected" in e for e in ST.validate_example(ex))


def test_packed_texts_follow_the_contract_layout(corpora):
    ex = _first_cu(corpora[0])
    texts, n_special = ST.packed_texts(ex)
    n_items = 0
    for q in ex["questions"].values():
        n_items += 2 if q["type"] == "noul" else len(q["criteria"])
    # [CLS] + one [SEP] per state segment + one [Q] per question + one marker per item
    assert n_special == 1 + len(ex["state"]) + len(ex["questions"]) + n_items
    assert texts[-1] == "a few lines" or isinstance(texts[-1], str)
    lens = ST.token_lengths([ex], None)
    assert lens[0] > n_special


def test_label_key_and_pct():
    assert ST.label_key("target", {"type": "choice", "label": "e03"}) == "<element>"
    assert ST.label_key("target", {"type": "choice", "label": "none"}) == "none"
    assert ST.label_key("intent", {"type": "choice", "label": "wait"}) == "wait"
    assert ST.label_key("x", {"type": "choice", "dist": {"a": 1.0}}) == "<soft>"
    assert ST.label_key("complete", {"type": "noul", "p": 1.0}) == "1"
    assert ST.label_key("scroll_amount", {"type": "score", "level": 2}) == "level2"
    p = ST.pct(list(range(101)))
    assert p["p50"] == 50 and p["p99"] == 99 and p["max"] == 100


def test_corpus_stats_and_cli(corpora, tmp_path, capsys):
    cu, gen = corpora
    s = ST.corpus_stats(cu, None, n_tok_sample=50)
    assert s["examples"] >= 150 and s["validation_errors"] == {}
    assert s["family_overlap"]["train&test"] == 0
    assert "intent" in s["labels"] and s["tokens"]["sampled"] == 50
    out = tmp_path / "stats.json"
    ST.main([str(cu), str(gen), "--tokens", "20", "--show", "2", "--json", str(out)])
    printed = capsys.readouterr().out
    assert "labels per qid" in printed and "### " in printed
    js = json.loads(out.read_text())
    assert set(js) == {str(cu), str(gen)}


def test_pretty_shows_transcript_labels_and_gold_first(corpora):
    ex = _first_cu(corpora[0])
    txt = ST.pretty(ex)
    assert ex["state"]["transcript"] in txt and "labels:" in txt
    gen_ex = next(ST.iter_corpus(corpora[1]))
    assert "state:" in ST.pretty(gen_ex)
