"""S-track converters (jev_local.data.s): pure-Python tests of the rendering core, the evaluated-split guard, the
refusing writer, the specs and the wording rules. No torch / datasets / pyarrow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev_local.bench.registry import exclusion_reason
from jev_local.data.s import build, commonsense, core, fact, index, inference, knowledge, typed
from jev_local.data.s.core import EvalGuard, Refused, SWriter, composite, render_mc, unit_hex, units_of
from jev_local.data.v2.render import FORBIDDEN_KEYS
from jev_local.schema import question_from_json

SPECS = build.SPECS


def check_row(row):
    for k in ("id", "split", "family", "state", "questions", "labels", "meta", "source", "bucket", "license_use", "variant", "group_id"):
        assert k in row, k
    assert row["bucket"].startswith("s_")
    assert row["license_use"] in ("commercial", "research", "unknown")
    assert row["split"] in ("train", "dev_mix")
    assert row["meta"]["s_track"] is True and row["meta"]["hf_split"] and row["meta"]["templates"]
    json.dumps(row)
    assert row["questions"]
    for qid, q in row["questions"].items():
        question_from_json(q)
        lab = row["labels"].get(qid)
        if lab is None:
            continue
        assert lab["type"] == q["type"]
        if q["type"] == "noul":
            assert 0.0 <= lab["p"] <= 1.0
        elif q["type"] == "choice":
            crit = q["criteria"]
            assert all(k.strip().lower() not in FORBIDDEN_KEYS for k in crit), crit
            if "label" in lab:
                assert lab["label"] in crit
            else:
                assert set(lab["dist"]) <= set(crit) and abs(sum(lab["dist"].values()) - 1) < 1e-6
        else:
            n = len(q["criteria"])
            if "level" in lab:
                assert 0 <= lab["level"] < n
            else:
                assert len(lab["dist"]) == n and abs(sum(lab["dist"]) - 1) < 1e-6
    assert set(row["labels"]) <= set(row["questions"])


# ------------------------------------------------------------------------------------------------ specs


def test_specs_consistent():
    assert len(SPECS) >= 25
    for n, s in SPECS.items():
        assert s.group in core.GROUPS
        assert s.license_use in ("commercial", "research", "unknown")
        assert s.evaluated, n  # every dataset names the split it is evaluated on
        assert s.train_splits and s.remotes
        for e in s.evaluated:
            assert (e.split not in s.train_splits) and all(not t.endswith("/" + e.split) or e.config not in t for t in s.train_splits), (n, e)
        # S-track: registry v1.1 waives train-split sources on track "S"; whatever it still excludes (dev sets,
        # test-only ids, context-row siblings such as RACE) must carry an explicit, disclosed `stage1` override
        reg = exclusion_reason(s.hf_id.split(" ")[0], track="S") or exclusion_reason(n, track="S")
        if reg and reg != "dev:tasksource_heldout":
            assert s.stage1, f"{n}: registry excludes it on S ({reg}) but the spec does not disclose the override"
            assert s.stage1 == reg, (n, s.stage1, reg)


def test_validation_only_when_eval_is_test():
    # hygiene rule 2: when the evaluated split is validation, train on train only
    for n in ("boolq", "rte", "hellaswag", "winogrande", "alphanli", "csqa", "mrpc", "cb", "sciq", "piqa", "socialiqa"):
        s = SPECS[n]
        if any(e.split == "validation" for e in s.evaluated):
            assert not any("validation" in t for t in s.train_splits), n
    for n in ("anli", "paws", "stsb", "arc", "openbookqa"):
        assert all(e.split.startswith("test") for e in SPECS[n].evaluated if "PAWS-X" not in e.note), n


def test_templates_at_least_five():
    tsets = [inference.NLI_CHOICE, inference.NOUL_ENTAIL, inference.NOUL_CONTRA, inference.NOUL_COMPAT, inference.PARA_NOUL, inference.PARA_CHOICE,
             inference.QQP_NOUL, inference.STS_INSTR, inference.BOOLQ_NOUL, inference.BOOLQ_CHOICE,
             commonsense.HELLA_T, commonsense.WINO_T, commonsense.ART_T, commonsense.PIQA_T, commonsense.SIQA_T, commonsense.CSQA_T,
             knowledge.MC_T, knowledge.MC_CTX_T, fact.CF_CHOICE, fact.HOVER_CHOICE, fact.HOVER_NOUL, fact.SCIFACT_CHOICE,
             index.VAST_T, index.ESCI_T, index.ACOS_NOUL, index.ACOS_SENT_CHOICE, index.ACOS_CAT_CHOICE, index.W2C_T, index.W2C_PAIR_T]
    for t in tsets:
        assert len(t) >= 5 and len(set(t)) == len(t)
    # suite wording present verbatim
    assert inference.NLI_CHOICE[0] == "What is the relationship between `{P}` and `{H}`?"
    assert inference.BOOLQ_NOUL[0] == "According to `{P}`, is the answer to `{Q}` yes?"
    assert inference.PARA_NOUL[0] == "Do `{A}` and `{B}` mean the same thing?"
    assert commonsense.HELLA_T[0] == "Which ending is the most plausible continuation of `context`?"
    assert commonsense.WINO_T[0] == "Which option correctly fills the blank `_` in `sentence`?"
    assert knowledge.MC_T[0] == "Which option correctly answers `question`?"
    assert fact.CF_CHOICE[0] == "Taken together, what does `evidence` say about `claim`?"
    assert inference.NLI_VOCABS[0][0][1] == "`{H}` is definitely true given `{P}`."


# ------------------------------------------------------------------------------------------------ units / guard


def test_units_and_guard():
    assert unit_hex("The cat sat.") == unit_hex("the  CAT sat")
    u = units_of(composite("A long premise sentence here.", "short hyp"), "short hyp", "a longer hypothesis with five tokens")
    assert len(u) == 2  # composite + the long primary; the 2-token primary is skipped
    g = EvalGuard()
    g.add_units("boolq", units_of(composite("P", "Is the sky blue today?"), "Is the sky blue today?"))
    g.add_units("mrpc", units_of(composite("first mrpc sentence", "second mrpc sentence"), "a sentence shared with a dev set"), role="dev")
    assert g.hits(units_of(composite("other passage", "is the sky blue today"), "Is the sky blue today?")) == ["boolq"]
    assert g.hit_detail(units_of(composite("q", "a sentence shared with a dev set"), "a sentence shared with a dev set")) == [(1, "mrpc", "dev", "p")]
    assert g.hit_detail(units_of(composite("first mrpc sentence", "second mrpc sentence"))) == [(0, "mrpc", "dev", "w")]
    assert g.hits(units_of(composite("x", "y"), "unrelated question text here")) == []
    # a test-role owner wins over a dev-role owner for the same hash
    g.add_units("mrpc", [unit_hex("a sentence shared with a dev set")], role="test")
    assert g.by_hex[unit_hex("a sentence shared with a dev set")] == "mrpc|w"  # a one-unit list is a whole item


def _boolq(ds, i, passage, question, ans=True):
    return inference.boolq_row(ds, f"b{i}", "train", passage, question, ans)


def test_writer_policy(tmp_path):
    ds = SPECS["boolq"]
    g = EvalGuard()
    g.add_units("boolq", units_of(composite("p", "Is water wet in the ocean?"), "Is water wet in the ocean?"))  # own, test role
    g.add_units("arc", units_of(composite("q"), "Which gas do plants absorb from the air?"))  # cross, test role
    g.add_units("sciq", units_of(composite("q2"), "What is the boiling point of water at sea level?"), role="dev")  # cross, dev role
    w = SWriter(ds, g, root=tmp_path)
    assert w.write(*_boolq(ds, 1, "plants", "which gas do plants absorb from the air")) is False  # cross test-role primary -> dropped
    assert w.write(*_boolq(ds, 2, "water", "What is the boiling point of water at sea level?")) is True  # cross dev-role primary -> kept, counted
    assert w.write(*_boolq(ds, 3, "x", "Is water wet in the ocean?")) is False  # own primary (test role) -> dropped
    assert w.write(*_boolq(ds, 4, "p", "Is water wet in the ocean?")) is False  # own whole item -> dropped (below the refusal threshold)
    assert w.write(*_boolq(ds, 5, "some passage", "Does the moon orbit the earth every month?")) is True
    man = w.close()
    h = man["hygiene"]
    assert man["rows"] == 2 and h["cross_eval_dropped"] == {"arc": 1} and h["cross_eval_shared_primary_kept_dev_role"] == {"sciq": 1}
    assert h["own_eval_whole_item_duplicates_dropped"] == 1 and h["own_eval_duplicate_row_ids"] == ["b4"] and h["own_eval_shared_primary_dropped"] == 1
    assert (tmp_path / "raw" / "inference" / "boolq.jsonl.zst").exists()
    m = json.loads((tmp_path / "raw" / "inference" / "boolq.manifest.json").read_text())
    assert m["evaluated_splits_never_trained_on"][0]["split"] == "validation" and m["evaluated_splits_never_trained_on"][0]["role"] == "test"


def test_writer_refuses_wrong_split(tmp_path):
    ds = SPECS["boolq"]
    g = EvalGuard()
    rows = [_boolq(ds, i, f"passage number {i} about a topic", f"Is question number {i} answered by passage {i}?") for i in range(250)]
    for row, units in rows:
        g.add_units("boolq", units)
    w = SWriter(ds, g, root=tmp_path)
    for row, units in rows:
        w.write(row, units)
    with pytest.raises(Refused):
        w.close()
    assert not (tmp_path / "raw" / "inference" / "boolq.jsonl.zst").exists()
    assert json.loads((tmp_path / "raw" / "inference" / "boolq.REFUSED.json").read_text())["n_hits"] == 250
    # strict mode refuses on any hit at all
    w2 = SWriter(ds, g, root=tmp_path / "strict", strict=True)
    w2.write(*_boolq(ds, 999, "p", "Is question number 3 answered by passage 3?"))
    with pytest.raises(Refused):
        w2.close()


# ------------------------------------------------------------------------------------------------ rendering


def test_render_mc_styles():
    import random

    seen = set()
    for i in range(200):
        rng = random.Random(i)
        r = render_mc("Which option correctly answers `question`?", ["Paris", "Rome", "Oslo", "Lima"], 2, rng, own_labels=["1", "2", "3", "4"])
        crit = r.question["criteria"]
        assert len(crit) == 4 and r.label["label"] in crit
        assert crit[r.label["label"]] in ("Oslo", None)
        if crit[r.label["label"]] is None:
            assert r.label["label"] == "Oslo"
        seen.add(tuple(crit)[0])
    assert {"A", "option_1", "Paris", "1"} & seen  # several key styles exercised
    r = render_mc("q", ["yes", "no"], 0, random.Random(1), style_probs={"text": 1.0})
    assert "yes" not in r.question["criteria"]  # bare yes/no keys are never used


def test_inference_rows():
    ds = SPECS["anli"]
    for i in range(60):
        row, units = inference.nli_row(ds, f"a{i}", str(i), "train_r1", "A man plays guitar on stage at a concert.", "A person is making music.",
                                       {"entailment": 1.0}, None)
        check_row(row)
        assert "relation" in row["questions"] and len(row["questions"]["relation"]["criteria"]) == 3
        assert units
    kinds = {k for i in range(60) for k in inference.nli_row(ds, f"a{i}", str(i), "x", "p one two three", "h one two three", {"neutral": 1.0}, None)[0]["questions"]}
    assert {"relation", "is_entailed", "is_contradicted", "is_compatible"} <= kinds
    row, _ = inference.nli_row(SPECS["rte"], "r1", "1", "train", "Dogs bark loudly at night.", "Dogs make noise.", None, 1.0)
    check_row(row)
    assert row["labels"]["is_entailed"]["p"] == 1.0
    row, _ = inference.para_row(SPECS["paws"], "p1", "labeled_final/train", "The cat chased the dog.", "The dog chased the cat.", False)
    check_row(row)
    assert row["labels"]["same_meaning"]["p"] == 0.0
    row, _ = inference.sts_row(SPECS["stsb"], "s1", "train", "A plane is taking off.", "An air plane is taking off.", 5.0)
    check_row(row)
    row, _ = inference.boolq_row(SPECS["boolq"], "b1", "train", "Water boils at 100 C at sea level.", "does water boil at 100 degrees", True)
    check_row(row)
    assert row["state"]["question" if "question" in row["state"] else "query"].endswith("?")


def test_commonsense_and_knowledge_rows():
    ds = SPECS["hellaswag"]
    for i in range(40):
        row, units = commonsense.mc_row(ds, f"h{i}", "train", {"context": "Removing ice from car: A man scrapes the windshield."}, commonsense.HELLA_T,
                                        ["he drives away.", "he adds wax.", "he puts on a coat.", "he keeps scraping."], 3, di_line=commonsense.HELLA_DI)
        check_row(row)
        assert len(row["questions"]["answer"]["criteria"]) == 4
    assert any(commonsense.mc_row(ds, f"h{i}", "train", {"context": "c"}, commonsense.HELLA_T, ["a", "b", "c", "d"], 0, di_line="L")[0]["state"] == ""
               for i in range(60))  # DI layout variant appears
    row, _ = knowledge.qa_row(SPECS["arc"], "arc-1", "ARC-Easy/train", "Which gas do plants absorb?", ["oxygen", "carbon dioxide", "helium"], 1,
                              own_labels=["A", "B", "C"], di_line=knowledge.ARC_DI)
    check_row(row)
    row, _ = knowledge.qa_row(SPECS["sciq"], "sq-1", "train", "What is H2O?", ["water", "salt", "air", "fire"], 0, context=("support", "H2O is water."), ctx_prob=1.0)
    check_row(row)
    assert row["state"]["support"] == "H2O is water." and row["variant"] == "with_context"
    assert commonsense.hellaswag_clean("A [title] B  C...") == "A. B C."


def test_fact_rows():
    row, units = fact.cf_row(SPECS["fever_cf"], "f1", "train", "Paris is the capital of France.",
                             [("Paris is the capital and most populous city of France.", "S"), ("Berlin is in Germany.", "N")], "fever")
    check_row(row)
    assert row["meta"]["claim_label"] == "SUPPORTS" and "1." in row["state"]["evidence"]
    row, _ = fact.cf_row(SPECS["fever_cf"], "f2", "train", "X is Y.", [("X is Y indeed.", "S"), ("X is not Y.", "R")], "vitaminc")
    assert row["meta"]["claim_label"] == "DISPUTED"
    row, _ = fact.hover_row(SPECS["hover"], "hv1", "train", "Claim text here.", [{"title": "T1", "text": "para one"}, {"title": "T2", "text": "para two"}], True, 2)
    check_row(row)
    row, _ = fact.scifact_row(SPECS["scifact"], "sf-1-2", "claims_train", "Drug X reduces Y.", "A trial", ["Sentence one.", "Sentence two."], "CONTRADICT")
    check_row(row)
    assert row["labels"]["verdict"]["label"] in row["questions"]["verdict"]["criteria"]


def test_index_rows():
    row, _ = index.vast_row(SPECS["vast"], "v1", "vast_train.csv", "Corporations have too much power over regulators.", "regulation", "pro")
    check_row(row)
    row, _ = index.esci_row(SPECS["esci"], "e-1-2", "train", "revent 80 cfm", {"title": "Panasonic fan", "brand": "Panasonic"}, "Irrelevant", "us")
    check_row(row)
    assert set(row["questions"]["relevance"]["criteria"]) >= {"Exact", "Substitute", "Complement", "Irrelevant"} or len(row["questions"]["relevance"]["criteria"]) == 4
    out = index.acos_row(SPECS["acos"], "ac1", "Generation/train", "the food was great but service slow",
                         [("food", "food quality", "positive", "great"), ("service", "service general", "negative", "slow")],
                         ["food quality", "service general", "ambience general", "restaurant prices", "drinks quality"])
    row, _ = out
    check_row(row)
    assert any(k.startswith("food_quality|positive") for k in row["questions"]) and row["variant"] == "restaurant"
    assert index.classify_response("<TOOLCALL>[...]</TOOLCALL>", None) == "tool_call"
    assert index.classify_response("Apologies, but I'm unable to provide real-time information.", None) == "cannot_answer"
    assert index.classify_response("Could you please provide the numbers in your dataset?", None) == "request_for_info"
    assert index.classify_response("The capital of France is Paris.", None) == "direct"
    row, _ = index.w2c_kind_row(SPECS["when2call"], "w1", "sft", [{"name": "get_weather"}], "What's the weather in Oslo?", "tool_call")
    check_row(row)
    row, _ = index.w2c_pair_row(SPECS["when2call"], "w2", "pref", [{"name": "t"}], "Show ICOs", "<TOOLCALL>...</TOOLCALL>", "Which language?")
    check_row(row)
    assert row["labels"]["better_response"]["label"] in row["questions"]["better_response"]["criteria"]


def test_typed_native():
    qs = {"action": {"type": "choice", "instructions": "What next?", "criteria": {"continue": "go on", "stop": "halt"}},
          "needs_review": {"type": "noul", "instructions": "Review?"},
          "severity": {"type": "score", "instructions": "How bad?", "criteria": ["low", "mid", "high"]}}
    gold = {"action": {"type": "choice", "label": "stop", "probabilities": {"continue": 0.3, "stop": 0.7}},
            "needs_review": {"type": "noul", "label": "true", "noul": 0.8},
            "severity": {"type": "score", "label": 2, "score": 1.7, "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8}}}
    state, rendered = typed.native_rows(json.dumps({"a": 1}), json.dumps(qs), json.dumps(gold))
    row = core.example(SPECS["typed_decisions"], "td-1", state, rendered, variant="wf", group_id="1", hf_split="all/train", templates=["native"])
    check_row(row)
    assert row["labels"]["action"]["dist"]["stop"] == pytest.approx(0.7) and row["labels"]["severity"]["dist"][2] == pytest.approx(0.8)
    assert row["questions"] == qs  # bodies untouched


def test_cli_list_runs(capsys):
    assert build.main(["list"]) == 0
    assert "boolq" in capsys.readouterr().out
