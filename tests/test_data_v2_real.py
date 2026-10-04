"""data-real converters (b1 tasksource-jev, b3 NLI, b5 Open-Jev, b7 Laurer) and the sources registry.

Pure Python: no torch / transformers / datasets / pyarrow. Rows are validated against the Jev schema and
the v1 example contract (labels refer to existing questions; choice labels are criteria keys; dists sum
to 1)."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter

import pytest

from jev_local.data.v2 import laurer, nli, open_jev, sources as S, tasksource_jev as T
from jev_local.data.v2.render import FORBIDDEN_KEYS, NOTA_KEY, stable_rng
from jev_local.schema import question_from_json


def check_row(row):
    for k in ("id", "split", "family", "state", "questions", "labels", "meta", "source", "bucket", "license_use", "variant", "group_id"):
        assert k in row, k
    assert row["license_use"] in S.LICENSE_USES
    assert row["split"] in ("train", "dev_mix")
    assert row["questions"], row
    json.dumps(row)  # serialisable
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


# ---------------------------------------------------------------------------------------------- sources

def test_registry_pinned_and_licensed():
    for s in S.SOURCES.values():
        assert re.fullmatch(r"[0-9a-f]{40}", s.revision), s
        assert s.license_use in (*S.LICENSE_USES, "per_row")
        assert s.allow_patterns
    assert S.SOURCES["tasksource_jev"].revision.startswith("8173a06c")
    assert S.SOURCES["open_jev"].revision.startswith("c67699e1")
    assert S.SOURCES["laurer_tasks_v4"].revision.startswith("73e6a196")
    assert not any(s.hf_id.lower() == "facebook/anli" for s in S.SOURCES.values() if s.bucket != "_ref")


@pytest.mark.parametrize("name,dataset,excluded", [
    ("banking77", "legacy-datasets/banking77", True),
    ("glue/rte", "nyu-mll/glue", True),
    ("anli/a1", "facebook/anli", True),
    ("toxic-chat/toxicchat0124/toxicity", "lmsys/toxic-chat", True),
    ("emotion", "dair-ai/emotion", True),
    ("IntentGrasp/all", "yuweiyin/IntentGrasp", True),
    ("tasksource_dpo_pairs", "tasksource/tasksource_dpo_pairs", True),
    ("auditor_review", "demo-org/auditor_review", True),
    ("multilingual/paws-x/de", "google-research-datasets/paws-x", True),
    ("sts-companion", "tasksource/sts-companion", True),
    ("crowdflower/text_emotion", "tasksource/crowdflower", False),
    ("cosmos_qa", "Samsoup/cosmos_qa", False),
    ("subjectivity", "tasksource/subjectivity", False),
    ("multilingual/americas_nli/all_languages", "nala-cub/americas_nli", False),
    ("procedural-typed-decisions/arithmetic", "tasksource/procedural-typed-decisions", False),
])
def test_stage1(name, dataset, excluded):
    assert bool(S.stage1_exclusion(name, dataset)) is excluded
    assert bool(S.stage1_exclusion(name, dataset, use_registry=False)) is excluded


def test_covered_elsewhere_and_helpers():
    assert S.covered_elsewhere("procedural-typed-decisions/arithmetic").startswith("covered_by_b4")
    assert S.covered_elsewhere("WANLI") == "covered_by_b3"
    assert S.covered_elsewhere("glue/mnli")
    assert S.covered_elsewhere("glue/qnli") is None
    assert S.chunk_sizes(5) == [5] and S.chunk_sizes(8) == [8]
    for n in (9, 13, 17, 256):
        cs = S.chunk_sizes(n)
        assert sum(cs) == n and max(cs) <= 6 and min(cs) >= 3
    assert S.most_restrictive(["commercial", "unknown"]) == "unknown"
    assert S.most_restrictive(["commercial", "research", "unknown"]) == "research"
    assert S.map_tasksource_license("non-commercial") == "research" and S.map_tasksource_license("unspecified") == "unknown"
    dev = sum(S.split_for(f"k{i}") == "dev_mix" for i in range(20000))
    assert 120 < dev < 290
    assert S.split_for("abc") == S.split_for("abc")


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd CLI missing")
def test_raw_writer_roundtrip(tmp_path):
    w = S.RawWriter("b_test", "unit", root=tmp_path)
    rows = [nli.render_pairs("unit", "nli", f"g{i}", [("A man sleeps.", "A person rests.", {"e": 1.0})],
                             license_use="commercial")[0] for i in range(20)]
    for r in rows:
        w.write(r)
    st = w.close(license_use="commercial", excluded_stage1=3)
    back = list(S.read_zst_jsonl(tmp_path / "raw" / "b_test" / "unit.jsonl.zst"))
    assert back == rows
    disk = json.loads((tmp_path / "raw" / "b_test" / "unit.stats.json").read_text())
    for k in ("rows", "decisions", "kinds", "k_hist", "tokens_est", "license_use", "excluded_stage1", "seconds"):
        assert k in disk
    assert disk["rows"] == 20 and disk["decisions"] == sum(len(r["labels"]) for r in rows) and st["excluded_stage1"] == 3
    # concatenated parts read back as one stream
    w2 = S.RawWriter("b_test", "cat", root=tmp_path, part="00")
    w2.write(rows[0])
    w2.close(license_use="commercial")
    w3 = S.RawWriter("b_test", "cat", root=tmp_path, part="01")
    w3.write(rows[1])
    w3.close(license_use="commercial")
    dst = S.concat_parts("b_test", "cat", [w2.path, w3.path], root=tmp_path)
    assert list(S.read_zst_jsonl(dst)) == rows[:2] and not w2.path.exists()


def test_stats_merge():
    rows = [nli.render_pairs("unit", "nli", f"g{i}", [("A.", "B.", {"e": 1.0})], license_use="commercial")[0] for i in range(10)]
    a, b, c = S.Stats(), S.Stats(), S.Stats()
    for i, r in enumerate(rows):
        (a if i % 2 else b).add(r)
        c.add(r)
    a.merge(b)
    assert a.to_json() == c.to_json()
    json.dumps(a.to_json())


def test_descriptions_bank(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"label": "joy", "context": "emo", "descriptions": ["happy", "glad", "cheerful"]}) + "\n")
    bank = S.load_descriptions(p)
    assert S.describe(bank, "joy", "emo") == ("happy", "glad", "cheerful")
    assert S.describe(bank, "joy", "other") == ("happy", "glad", "cheerful")
    assert S.describe(bank, "anger") == ()
    p.write_text(json.dumps({"label": "neutral", "context": "nli", "descriptions": ["neither follows nor contradicts"]}) + "\n")
    bank = S.load_descriptions(p)
    assert S.describe(bank, "neutral", "nli") and S.describe(bank, "neutral", "sentiment") == ()
    assert S.load_descriptions(tmp_path / "missing.jsonl") == {}


# ---------------------------------------------------------------------------------------------- b1

def _tsj(source, kind, q, options, target, gid="g1", state="Some text.", qid="decision", variant="direct"):
    return {"state": state, "kind": kind, "id": f"{gid}:{qid}", "options": options, "target": target, "question": q,
            "source": source, "variant": variant, "split": "train", "group_id": gid, "question_id": qid,
            "license": "cc-by-4.0", "license_use": "commercial"}


def _infos():
    yaml_like = {"sources": {
        "topic": {"dataset": "x/topic", "license": "cc-by-4.0", "license_use": "commercial", "revisions": {"x/topic": "a"}},
        "mcqa": {"dataset": "x/mcqa", "license": "unspecified", "license_use": "unspecified"},
        "votes": {"dataset": "x/votes", "license": "cc-by-nc-4.0", "license_use": "non-commercial"},
        "banking77": {"dataset": "legacy-datasets/banking77", "license": "cc-by-4.0", "license_use": "commercial",
                      "revisions": {"PolyAI/banking77": "b"}},
        "WANLI": {"dataset": "alisawuffles/WANLI", "license": "cc-by-4.0", "license_use": "commercial"},
    }}
    infos = T.source_infos(yaml_like)
    labels = tuple(sorted(f"topic_{i}" for i in range(12)))
    sets = {"topic": Counter({labels: 90, ("a", "b"): 10}), "mcqa": Counter({(f"o{i}", f"p{i}"): 1 for i in range(50)})}
    for s, ls in T.classify_labelsets(sets).items():
        infos[s].labelset = ls
    return infos, labels


def test_source_infos_and_labelsets():
    infos, labels = _infos()
    assert infos["banking77"].excluded == "registry:test:banking77" and not infos["topic"].excluded
    # registry v1.1: WANLI is a public-Jev source (Kev / bonzi rows), so stage 1 drops it before the covered-by-b3 rule
    assert infos["WANLI"].excluded == "registry:public_jev:wanli" and infos["WANLI"].covered is None
    assert infos["topic"].labelset == labels and infos["mcqa"].labelset is None
    assert infos["votes"].license_use == "research" and infos["mcqa"].license_use == "unknown"


def test_render_row_kinds():
    infos, labels = _infos()
    rng = stable_rng("r")
    # yes/no choice -> noul with p(yes)
    rend, qi, note = T.render_row(_tsj("mcqa", "choice", "Is it plausible?", ["no", "yes"], [0.25, 0.75]), infos["mcqa"], rng, {})
    assert rend.question["type"] == "noul" and rend.label["p"] == 0.75 and note == "choice_yesno->noul"
    assert rend.question["instructions"] == "Is it plausible?"
    rend, qi, note = T.render_row(_tsj("mcqa", "choice", "Choose the most appropriate category for the state.", ["False", "True"],
                                       [1.0, 0.0]), infos["mcqa"], rng, {})
    assert rend.label["p"] == 0.0 and '"True"' in rend.question["instructions"] and "category" not in rend.question["instructions"]
    for how, r in T.derive(qi, rng):
        if how == "neg":
            assert r.label["p"] == 1.0 and "category" not in r.question["instructions"]
    # soft votes kept as dist
    rend, qi, note = T.render_row(_tsj("votes", "choice", "Sentiment?", ["pos", "neg", "neu"], [0.6, 0.4, 0.0]), infos["votes"], rng, {})
    assert {rend.key_map[k] for k in rend.label["dist"]} == {"pos", "neg"}
    # score with soft dist
    rend, qi, _ = T.render_row(_tsj("votes", "score", "How good?", ["1", "2", "3"], [0.0, 0.5, 0.5]), infos["votes"], rng, {})
    assert rend.question["type"] == "score" and rend.label["dist"] == [0.0, 0.5, 0.5]
    # duplicate options merged; empty target dropped
    rend, qi, _ = T.render_row(_tsj("mcqa", "choice", "Q?", ["x", "x", "y"], [0.5, 0.5, 0.0]), infos["mcqa"], rng, {})
    assert set(rend.question["criteria"]) == {"x", "y"} and rend.key_map[rend.label["label"]] == "x"
    assert T.render_row(_tsj("mcqa", "choice", "Q?", ["x", "y"], [0.0, 0.0]), infos["mcqa"], rng, {}) is None


def test_labelset_rows_get_k_subsampling_and_nota_mc_rows_do_not():
    infos, labels = _infos()
    ks, nota, mc_k = Counter(), 0, set()
    for i in range(400):
        rng = stable_rng("ls", i)
        tgt = [1.0 if l == "topic_3" else 0.0 for l in labels]
        rend, _, note = T.render_row(_tsj("topic", "choice", "Topic?", list(labels), tgt), infos["topic"], rng, {})
        assert note == "labelset"
        crit = rend.question["criteria"]
        ks[len(crit)] += 1
        if rend.label.get("label") == NOTA_KEY:
            nota += 1
            assert "topic_3" not in rend.key_map.values()
        else:
            assert rend.key_map[rend.label["label"]] == "topic_3"
        opts = ["A cat.", "A dog.", "None of the above"]
        rend, _, note = T.render_row(_tsj("mcqa", "choice", "Which?", opts, [1, 0, 0]), infos["mcqa"], rng, {})
        assert note == "mc"
        mc_k.add(len(rend.question["criteria"]))
    assert min(ks) == 2 and max(ks) >= 12 and 15 <= nota <= 70
    assert mc_k == {3}


def test_pack_state_excludes_dedups_and_derives():
    infos, labels = _infos()
    ctr = T.Counters.new()
    rows = [
        _tsj("banking77", "choice", "Intent?", ["a", "b"], [1, 0], qid="b77"),
        _tsj("WANLI", "noul", "Entailed?", [], [1.0], qid="w"),
        _tsj("mcqa", "choice", "Which answer?", ["red", "green", "blue"], [0, 1, 0], qid="q1"),
        _tsj("mcqa", "choice", "Which answer?", ["blue", "red", "green"], [0, 0, 1], qid="q2", variant="criteria_permutation"),
    ]
    out = T.pack_state("g1", "Some text.", rows, infos, {}, ctr)
    assert len(out) == 1
    row = out[0]
    check_row(row)
    assert ctr.excluded["banking77"] == 1 and ctr.excluded["WANLI"] == 1 and ctr.covered["WANLI"] == 0 and ctr.dup_questions["mcqa"] == 1
    assert len(row["questions"]) == 3 and "q1" in row["questions"] and set(row["meta"]["derived"]) == set(row["questions"]) - {"q1"}
    # derived labels follow from the gold exactly
    for qid, how in row["meta"]["derived"].items():
        q, lab = row["questions"][qid], row["labels"][qid]
        if how == "lv":
            opt = re.search(r'"([^"]+)"', q["instructions"]) or re.search(r"Proposed: (\w+)", q["instructions"])
            assert "Which answer?" in q["instructions"]
            assert lab["p"] == (1.0 if opt.group(1) == "green" else 0.0)
        if how == "bin":
            assert set(q["criteria"]) >= {"green"} and lab["label"] == "green"
    assert row["source"] == "tsj/mcqa" and row["license_use"] == "unknown" and row["bucket"] == "b1_tasksource_jev"
    generic = T.pack_state("g9", "Q: 2+2?", [_tsj("mcqa", "choice", "Choose the most appropriate answer from the supplied options.",
                                                    ["3", "4", "5"], [0, 1, 0])], infos, {}, ctr)[0]
    for qid, how in generic["meta"]["derived"].items():
        if how == "lv":
            assert "supplied options" not in generic["questions"][qid]["instructions"]


def test_pack_state_score_and_noul_derivations_and_chunking():
    infos, _ = _infos()
    ctr = T.Counters.new()
    sc = _tsj("votes", "score", "How toxic is it?", ["none", "some", "a lot"], [0.2, 0.3, 0.5], qid="tox")
    out = T.pack_state("g2", "txt", [sc], infos, {}, ctr)
    row = out[0]
    check_row(row)
    for qid, how in row["meta"]["derived"].items():
        if how == "ge":
            lv = re.search(r'"(none|some|a lot)"', row["questions"][qid]["instructions"]).group(1)
            assert abs(row["labels"][qid]["p"] - {"some": 0.8, "a lot": 0.5}[lv]) < 1e-9
        if how == "sc":
            assert row["questions"][qid]["type"] == "choice" and row["labels"][qid]["dist"]["a lot"] == 0.5
    nl = _tsj("votes", "noul", "Does the speaker agree?", [], [0.7], qid="agree")
    row = T.pack_state("g3", "txt", [nl], infos, {}, ctr)[0]
    check_row(row)
    for qid, how in row["meta"]["derived"].items():
        if how == "neg":
            assert abs(row["labels"][qid]["p"] - 0.3) < 1e-9
        if how == "binc":
            assert abs(max(row["labels"][qid]["dist"].values()) - 0.7) < 1e-9
    prob = _tsj("votes", "noul", "What fraction of annotators rated it toxic?", [], [0.2], qid="frac")
    row = T.pack_state("g4", "txt", [prob], infos, {}, ctr)[0]
    assert "derived" not in row["meta"] and len(row["questions"]) == 1
    many = [_tsj("votes", "noul", f"Is it about thing {i}?", [], [i / 10], qid=f"n{i}") for i in range(10)]
    out = T.pack_state("g5", "txt", many, infos, {}, ctr)
    assert [len(r["questions"]) for r in out] == [5, 5] and len({r["id"] for r in out}) == 2
    for r in out:
        check_row(r)


def test_iter_groups_keeps_states_apart():
    rows = [_tsj("mcqa", "noul", "Q?", [], [1.0], gid="a", state="s1", qid="1"),
            _tsj("mcqa", "noul", "Q?", [], [1.0], gid="a", state="s2", qid="2"),
            _tsj("mcqa", "noul", "R?", [], [0.0], gid="a", state="s1", qid="3"),
            _tsj("mcqa", "noul", "Q?", [], [1.0], gid="b", state="s1", qid="4")]
    g = list(T.iter_groups(rows))
    assert [x[0] for x in g] == ["a", "b"]
    assert [(s, [r["question_id"] for r in rs]) for s, rs in g[0][1]] == [("s1", ["1", "3"]), ("s2", ["2"])]


# ---------------------------------------------------------------------------------------------- b3

def test_nli_pairs_both_layouts_and_exact_labels():
    seen_layouts = Counter()
    for i in range(200):
        prs = [("A man is sleeping on a couch.", "A person is resting.", {"e": 1.0}),
               ("A man is sleeping on a couch.", "A man is running.", {"c": 1.0}),
               ("A man is sleeping on a couch.", "The couch is red.", {"n": 1.0})]
        rows = nli.render_pairs("mnli", "nli", f"p{i}", prs, license_use="commercial")
        for r in rows:
            check_row(r)
            seen_layouts[r["meta"]["layout"]] += 1
            for qid, q in r["questions"].items():
                lab = r["labels"][qid]
                if q["type"] == "noul":
                    hyp = 0 if r["meta"]["layout"] == "pair" else int(qid[1])
                    gold = [("e"), ("c"), ("n")][hyp] if r["meta"]["layout"] == "inline" else None
                    if gold:
                        kind = r["meta"]["kinds"][qid].split("_", 1)[1]
                        expect = {"entail": gold == "e", "contradict": gold == "c", "compatible": gold != "c"}[kind]
                        assert lab["p"] == float(expect)
        if rows[0]["meta"]["layout"] == "inline":
            assert len(rows) == 1 and rows[0]["state"] == "A man is sleeping on a couch."
        else:
            assert len(rows) == 3 and rows[0]["state"].count("\n") == 1
    assert seen_layouts["inline"] > 40 and seen_layouts["pair"] > 100


def test_nli_two_way_docnli_and_chaos():
    for i in range(50):
        rows = nli.render_pairs("docnli", "doc", f"d{i}", [("Long document " * 50, "A claim.", {"e": 1.0})], two_way=True,
                                license_use="unknown")
        for r in rows:
            check_row(r)
            for qid, q in r["questions"].items():
                if q["type"] == "choice":
                    assert len(q["criteria"]) == 2 and NOTA_KEY not in q["criteria"]
                    assert "refuted" not in q["instructions"] and "neither" not in q["instructions"]
    item = {"uid": "u1", "label_dist": [0.3, 0.7, 0.0], "label_count": [30, 70, 0],
            "example": {"premise": "Kids wash hands.", "hypothesis": "Kids at a ballgame.", "source": "snli"}}
    row = nli.chaos_rows(item, "snli", "research")[0]
    check_row(row)
    assert row["labels"]["is_entail"]["p"] == 0.3 and row["labels"]["is_contradict"]["p"] == 0.0
    assert sorted(row["labels"]["relation"]["dist"].values()) == [0.3, 0.7]
    alpha = {"uid": "u2", "label_dist": [0.97, 0.03], "example": {"obs1": "Ron started a job.", "obs2": "Ron is fired.",
                                                                  "hyp1": "Ron insulted his boss.", "hyp2": "Ron's boss was nice."}}
    row = nli.chaos_rows(alpha, "alphanli", "research")[0]
    check_row(row)
    assert max(row["labels"]["explain"]["dist"].values()) == 0.97


# ---------------------------------------------------------------------------------------------- b5

def _oj(kind, q, options, target, gid="customer-control-v1:42:1", qid="x", lic="CC0-1.0"):
    meta = {"question_id": qid, "target_basis": "explicit_control_rule", "provenance": {"license": lic, "type": "synthetic"}}
    return {"id": f"{gid}:{qid}", "group_id": gid, "split": "train", "source": gid.split(":")[0], "kind": kind,
            "question": q, "options": options, "target": target, "state_json": json.dumps({"board": "..."}),
            "metadata_json": json.dumps(meta), "record_json": "", "original_line_number": 1}


def test_open_jev_rendering_and_licences():
    rng = stable_rng("oj")
    r = open_jev.render_decision(_oj("noul", "Approve refund?", ["no", "yes"], [0.25, 0.75]), rng)
    assert r.question == {"type": "noul", "instructions": "Approve refund?"} and r.label["p"] == 0.75
    r = open_jev.render_decision(_oj("choice", "Category?", ["billing: Charges, refunds", "account: Login, profile"], [0, 1]), rng)
    assert r.question["criteria"] == {"billing": "Charges, refunds", "account": "Login, profile"} or \
        r.question["criteria"] == {"account": "Login, profile", "billing": "Charges, refunds"}
    assert r.label == {"type": "choice", "label": "account"}
    r = open_jev.render_decision(_oj("choice", "Square?", ["0", "1", "2"], [0.5, 0.5, 0.0]), rng)
    assert r.label["dist"] == {"0": 0.5, "1": 0.5}
    r = open_jev.render_decision(_oj("score", "Severity?", ["low", "mid", "high"], [0, 0.5, 0.5]), rng)
    assert r.label == {"type": "score", "dist": [0, 0.5, 0.5]}
    assert open_jev.row_license({"provenance": {"license": "CC0-1.0"}}) == ("commercial", "CC0-1.0")
    lic = "Generated conversations: CC0-1.0; upstream question descriptions: TypeSafe documentation, license not verified"
    assert open_jev.row_license({"provenance": {"license": lic}})[0] == "unknown"
    assert open_jev.is_jev_output({"target_basis": "Jev 1.13 output via OpenRouter"})
    assert not open_jev.is_jev_output({"target_basis": "exact minimax", "provenance": {"type": "synthetic"}})
    ctr = Counter()
    rows = [_oj("noul", f"Action {i}?", ["no", "yes"], [1, 0], qid=f"a{i}") for i in range(10)]
    rows.append(_oj("noul", "Mixed?", ["no", "yes"], [0, 1], qid="m", lic=lic))
    out = open_jev.pack_group("customer-control-v1:42:1", rows[0]["state_json"], rows, ctr)
    assert [len(r["questions"]) for r in out] == [6, 5] and out[1]["license_use"] == "unknown" and out[0]["license_use"] == "commercial"
    assert out[0]["state"] == {"board": "..."}
    for r in out:
        check_row(r)


# ---------------------------------------------------------------------------------------------- b7

def _v4(text, hyps, pos_i, td="Classify texts by genre. The classes are: prose, poetry, script."):
    return [{"hypothesis": h, "text": text, "labels": 0 if i == pos_i else 1, "text_type": "t", "text_style": "s",
             "profession": "artist", "task_description": td, "task_hypotheses": hyps} for i, h in enumerate(hyps)]


def test_laurer_v4_and_refinedweb():
    hyps = ["The text is prose.", "The text is poetry.", "The text is a script."]
    assert laurer.parse_classes("Do X. The classes are: prose, poetry, script.", 3) == ["prose", "poetry", "script"]
    assert laurer.parse_classes("Do X. The classes are: a, b and c.", 3) == ["a", "b", "c"]
    assert laurer.parse_classes("Classify texts by genre.", 3) is None
    golds = Counter()
    for i in range(100):
        rows = laurer.v4_rows(_v4(f"Roses are red {i}", hyps, 1)[0]["task_description"], f"Roses are red {i}",
                              _v4(f"Roses are red {i}", hyps, 1), "commercial")
        row = rows[0]
        check_row(row)
        assert row["labels"]["verify_pos"]["p"] == 1.0 and row["labels"]["verify_neg"]["p"] == 0.0
        assert "poetry" in row["questions"]["verify_pos"]["instructions"]
        lab = row["labels"]["label"]["label"]
        golds[lab if lab == NOTA_KEY else "gold"] += 1
        if lab != NOTA_KEY:
            crit = row["questions"]["label"]["criteria"]
            assert lab == "poetry" or (crit[lab] and "poetry" in crit[lab])
    assert golds["gold"] > 80
    # no class names -> hypotheses as keys or opaque keys
    td = "Classify texts by genre."
    opaque = 0
    for i in range(100):
        row = laurer.v4_rows(td, f"t{i}", _v4(f"t{i}", hyps, 2, td), "commercial")[0]
        check_row(row)
        q, lab = row["questions"]["label"], row["labels"]["label"]["label"]
        if lab != NOTA_KEY:
            assert lab == "The text is a script." or q["criteria"][lab] == "The text is a script."
        opaque += lab != NOTA_KEY and lab != "The text is a script."
    assert 10 < opaque < 60
    rw = [{"text": "Recipe for soup", "category": c, "labels": l} for c, l in
          [("cooking", 0), ("food", 0), ("politics", 1), ("music", 1)]]
    row = laurer.refinedweb_rows("Recipe for soup", rw, "commercial")[0]
    check_row(row)
    crit = row["questions"]["topic"]["criteria"]
    assert not ({"cooking", "food"} <= set(crit))
    assert row["labels"]["is_topic"]["p"] == 1.0 and row["labels"]["not_topic"]["p"] == 0.0
    for i in range(60):
        rw_i = [{"text": f"t{i}", "category": c, "labels": l} for c, l in [("cooking", 0), ("politics", 1), ("music", 1), ("art", 1)]]
        assert laurer.refinedweb_rows(f"t{i}", rw_i, "commercial")[0]["labels"]["topic"]["label"] != NOTA_KEY
    assert laurer.refinedweb_rows("x", [{"text": "x", "category": "a", "labels": 0}], "commercial") == []
