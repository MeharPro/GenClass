"""S track, agent s-intent-topic: specs, wording, harness shapes, renderers and the hygiene writer (pure Python, Mac-safe)."""

from __future__ import annotations

import json
import random
from collections import Counter

import pytest

from jev_local.bench.templates import FIN_TOPIC_NAMES
from jev_local.data.s import common as C
from jev_local.data.s import intent_topic as S
from jev_local.data.v2.render import FORBIDDEN_KEYS, make_example, stable_rng
from jev_local.schema import SystemOneRequest
from jev_local.validate import validate_limits


# ------------------------------------------------------------------------------------------ fixtures

def _names(prefix: str, n: int) -> list[str]:
    return sorted(f"{prefix}_{i:03d}" for i in range(n))


AG = ["World", "Sports", "Business", "Sci/Tech"]
YAHOO = ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet", "Sports",
         "Business & Finance", "Entertainment & Music", "Family & Relationships", "Politics & Government"]
TT = ["arts_&_culture", "business_&_entrepreneurs", "pop_culture", "daily_life", "sports_&_gaming", "science_&_technology"]
ONTO = {
    "affirm": {"description": "is the intent to affirm something?", "domain": ["general"]},
    "deny": {"description": "is the intent to deny something?", "domain": ["general"]},
    "request_info": {"description": "is the intent to ask or request some information?", "domain": ["general"]},
    "how_long": {"description": "is the intent asking about how long something takes?", "domain": ["general"]},
    "pin": {"description": "is the intent asking about the pin?", "domain": ["banking"]},
    "arrival": {"description": "is the intent asking about the arrival of something?", "domain": ["banking"]},
    "transfer_payment_deposit": {"description": "is the intent asking about a transfer, payment or deposit?", "domain": ["banking"]},
    "booking": {"description": "is the intent asking about a booking?", "domain": ["hotels"]},
    "room": {"description": "is the intent asking about a room?", "domain": ["hotels"]},
}
CTX = {
    "banking77": {"names": _names("intent", 77), "hyps": {}, "di_order": list(reversed(_names("intent", 77)))},
    "clinc150": {"names": _names("clinc", 150)[:42] + ["oos"] + _names("clinc", 150)[42:]},
    "massive": {"names": _names("massive", 60), "hyps": {}, "scenarios": _names("scenario", 18)},
    "hwu64": {"names": _names("hwu", 64)},
    "snips": {"names": ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]},
    "atis": {"names": ["abbreviation", "airfare", "flight", "flight+airfare", "ground_service"]},
    "nlupp": {"ontology": ONTO, "domain_intents": {d: sorted(i for i, v in ONTO.items() if d in v["domain"] or "general" in v["domain"]) for d in ("banking", "hotels")}},
    "mtop": {"names": _names("INTENT", 113), "domains": _names("domain", 11), "bank": {}},
    "ag_news": {"names": AG, "hyps": {"world": "This example news text is about world news", "sports": "This example news text is about sports",
                                       "business": "This example news text is about business news", "sci tech": "This example news text is about science and technology"}},
    "yahoo_topics": {"names": YAHOO, "hyps": {}},
    "newsgroups20": {"names": _names("rec.group", 20)},
    "dbpedia_14": {"names": _names("Type", 14), "bank": {}},
    "tweet_topic": {"names": TT},
    "fin_topic": {"names": list(FIN_TOPIC_NAMES)},
    "trec": {"names": _names("fine", 50), "coarse": _names("coarse", 6), "bank": {}, "bank_coarse": {}},
    "sib200_en": {"names": list(S.SIB_LABELS)},
}


def _row(name: str, i: int) -> dict:
    t = f"example utterance number {i} about things"
    return {
        "banking77": {"text": t, "label_text": CTX["banking77"]["names"][i % 77]},
        "clinc150": {"text": t, "intent": i % 151},
        "massive": {"text": t, "label": CTX["massive"]["names"][i % 60], "_scenario": CTX["massive"]["scenarios"][i % 18]},
        "hwu64": {"utterance": t, "label": i % 64},
        "snips": {"text": t, "category": CTX["snips"]["names"][i % 7]},
        "atis": {"text": t, "intent": CTX["atis"]["names"][i % 5]},
        "nlupp": {"text": t, "intents": (["affirm", "pin"] if i % 3 else ["booking"]) if i % 5 else [], "domain": "banking" if i % 3 else "hotels", "fold": i % 18},
        "mtop": {"text": t, "label_text": CTX["mtop"]["names"][i % 113], "_domain": CTX["mtop"]["domains"][i % 11]},
        "ag_news": {"text": t, "label": i % 4},
        "yahoo_topics": {"question_title": f"Why is example {i} like this?", "question_content": "" if i % 2 else f"body of question {i} here", "best_answer": f"answer {i}", "topic": i % 10},
        "newsgroups20": {"text": t, "label_text": CTX["newsgroups20"]["names"][i % 20]},
        "dbpedia_14": {"title": f"Entity {i}", "content": t, "label": i % 14},
        "tweet_topic": {"text": t, "label": i % 6},
        "fin_topic": {"text": t, "label": str(i % 20)},
        "trec": {"text": f"How many examples are there in set {i} ?", "label_text": CTX["trec"]["names"][i % 50], "label_coarse_text": CTX["trec"]["coarse"][i % 6]},
        "sib200_en": {"text": t, "category": S.SIB_LABELS[i % 7]},
    }[name]


def _valid(ex: dict) -> None:
    req = SystemOneRequest.model_validate({"state": ex["state"], "model": "m", "questions": ex["questions"]})
    validate_limits(req)
    assert ex["labels"], "no supervision"
    for qid, lab in ex["labels"].items():
        q = ex["questions"][qid]
        assert lab["type"] == q["type"]
        if q["type"] == "choice":
            if "label" in lab:
                assert lab["label"] in q["criteria"]
            else:
                assert set(lab["dist"]) <= set(q["criteria"]) and abs(sum(lab["dist"].values()) - 1) < 1e-6
        else:
            assert 0.0 <= lab["p"] <= 1.0
    assert ex["meta"]["primary"] and ex["meta"]["template"] and ex["meta"]["shape"]


def _render(name: str, i: int) -> list[dict]:
    spec = S.SPECS[name]
    out = []
    rng = stable_rng("S", name, "train", i)
    for state, qs, variant, meta, texts in S.RENDERERS[name](_row(name, i), "train", i, rng, CTX[name]):
        qs = {k: v for k, v in qs.items() if v.label is not None}
        ex = make_example(f"{name}-{i}", state, qs, source=name, bucket=f"s_{spec.group}", license_use=spec.license_use, variant=variant, meta=meta)
        _valid(ex)
        assert texts and all(isinstance(t, str) and t for t in texts)
        out.append(ex)
    return out


# ------------------------------------------------------------------------------------------ specs and wording

def test_templates_at_least_five_with_harness_wording_first():
    for key, tset in S.TEMPLATE_SETS.items():
        assert len(tset) >= 5, key
        assert len(set(tset)) == len(tset), key
    # Deusser (MIT) / jevbench wording, verbatim
    assert S.CH_BANKING[0] == "Which intent best describes the bank customer's `query`?"
    assert S.CH_CLINC[0] == "Which intent does the user's `utterance` to a virtual assistant express?"
    assert S.CH_MASSIVE[0] == "Which intent does the user's `utterance` to a voice assistant express?"
    assert S.CH_HWU[0] == "Which intent does the user's `utterance` to a home assistant express?"
    assert S.CH_AG[0] == "What is the topic of the news `article`?"
    assert S.CH_YAHOO[0] == "Which topic category of the Yahoo Answers forum does `title` belong to?"
    assert S.CH_20NG[0] == "Which newsgroup was `post` posted to?"
    assert S.CH_TT[0] == "What is the topic of `tweet`?"
    assert S.CH_FIN[0] == "What is the topic of the financial news `tweet`?"
    assert S.CH_SIB[0] == "What is the topic of `text`?"
    # Decision Index kit (MIT) task lines, verbatim
    assert S.DI_LINE_BANKING77 == "Classify the banking intent of this user request:"
    assert S.DI_LINE_CLINC == "Classify the intent of this user request, or choose out of scope if none applies:"
    assert S.DI_CLINC_OOS == "out of scope: none of the listed intents"
    assert S.CLINC_OOS == "Out of scope: the request matches none of the other intents."
    for n, shapes in S.HARNESS_SHAPES.items():
        assert n in S.SPECS and shapes


def test_specs_consistent():
    assert set(S.RENDERERS) == set(S.SPECS)
    assert {s.group for s in S.SPECS.values()} == {"intent", "topic"}
    for s in S.SPECS.values():
        assert s.license_use in ("commercial", "research", "unknown"), s.name
        assert s.train_files, s.name
        for split, path in s.train_files:
            assert "test" not in path.lower() and "valid" not in path.lower() or split.startswith("validation"), (s.name, split, path)
            assert split.startswith(("train", "validation", "fold")), (s.name, split)
            if split.startswith("fold"):
                assert int(split[4:]) not in S.NLUPP_EVAL_FOLDS
        assert s.eval_srcs, s.name
        for e in s.eval_srcs:
            p = (e.hf[1] if e.hf else e.url).lower()
            assert any(x in p for x in ("test", "valid", "fold18", "fold19")), (s.name, p)
    # rule 2: evaluated on validation -> train only
    assert [x for x, _ in S.SPECS["fin_topic"].train_files] == ["train"]
    # validation joins training only where test is evaluated
    for n in ("clinc150", "massive", "mtop", "sib200_en"):
        assert "validation" in [x for x, _ in S.SPECS[n].train_files], n
    assert {x for x, _ in S.SPECS["tweet_topic"].train_files} == {"train_2020", "train_2021", "validation_2020", "validation_2021"}
    assert {e.name for e in S.SPECS["tweet_topic"].eval_srcs} == {"tweet_topic/test_2021", "tweet_topic/test_2020"}
    assert {e.name for e in S.SPECS["banking77"].eval_srcs} == {"banking77/test", "banking77_polyai/test"}
    assert len([e for e in S.SPECS["nlupp"].eval_srcs]) == 4 and len(S.SPECS["nlupp"].train_files) == 36
    # registry stage-1 verdicts are recorded (and overridden on purpose for official train splits)
    assert C.registry_reason("mteb/banking77") == "test:banking77"
    assert C.registry_reason("DeepPavlov/hwu64") == "public_jev:hwu64"  # retired dev set (registry v1.1)
    assert C.registry_reason("tuetschek/atis") == "dev:atis" and C.registry_reason("benayas/snips") == "dev:snips"
    assert C.registry_reason("mteb/mtop_intent") == "public_jev:label_pressure"
    assert C.registry_reason(None, None, "nlupp") == "public_jev:nlupp"


# ------------------------------------------------------------------------------------------ choice shapes

def test_label_choice_keeps_all_labels_in_every_style():
    names = _names("intent", 77)
    rng = random.Random(0)
    for style in ("bare", "desc", "opaque"):
        q = S.label_choice("Which?", names, names[5], rng, desc={n: f"About {n}." for n in names}, style=style, order="canonical")
        assert len(q.question["criteria"]) == 77 and set(q.key_map.values()) == set(names)
        assert q.question["criteria"][q.label["label"]] in (None, "About intent_005.", "intent_005: About intent_005.", "intent_005") or style == "opaque"
        if style == "bare":
            assert list(q.question["criteria"]) == names and all(v is None for v in q.question["criteria"].values())
        if style == "opaque":
            assert list(q.question["criteria"]) == [f"option_{i + 1}" for i in range(77)]
    q = S.label_choice("Which?", AG, "Sports", rng, style="opaque", order="canonical")
    assert list(q.question["criteria"]) == ["A", "B", "C", "D"] and q.question["criteria"]["B"] == "Sports"
    sh = S.label_choice("Which?", names, names[0], random.Random(1), style="bare", order="shuffle")
    assert list(sh.question["criteria"]) != names and set(sh.question["criteria"]) == set(names)
    soft = S.label_choice("Which?", ["a", "b", "c"], {"a": 0.5, "b": 0.5}, rng, style="desc", desc={})
    assert abs(sum(soft.label["dist"].values()) - 1) < 1e-9 and set(soft.label["dist"]) <= set(soft.question["criteria"])
    forb = S.label_choice("Which?", ["yes", "no", "maybe"], "yes", rng, style="bare", order="canonical")
    assert set(forb.question["criteria"]) == {"yes (label)", "no (label)", "maybe"} and forb.label["label"] == "yes (label)"


def test_harness_shapes():
    names = _names("intent", 77)
    di = S.di_choice(S.DI_LINE_BANKING77, "how do i get a new card", names, names, 3)
    assert list(di.question["criteria"]) == [f"option_{i}" for i in range(77)]  # 0-indexed above 26 options
    assert di.question["instructions"] == "Classify the banking intent of this user request:\nhow do i get a new card"
    assert di.label["label"] == "option_3" and di.question["criteria"]["option_3"] == names[3]
    di4 = S.di_choice("line:", "text", AG, AG, 2)
    assert list(di4.question["criteria"]) == ["A", "B", "C", "D"] and di4.label["label"] == "C"
    dmb = S.dmb_choice(S.DMB_LINE, names, names[9])
    assert all(k == v for k, v in dmb.question["criteria"].items()) and dmb.label["label"] == names[9]
    bt = S.btzsc_choice(S.BTZSC_LINE, AG, {n: f"hyp {n}" for n in AG}, "Sports")
    assert list(bt.question["criteria"]) == ["label_000", "label_001", "label_002", "label_003"] and bt.label["label"] == "label_001"
    el = S.elcronos_choice("Which topic?", TT, "daily_life")
    assert all(v == "" for v in el.question["criteria"].values()) and el.label["label"] == "daily_life"
    with_f, plain, lower = S.nlupp_question("is the intent to affirm something?", "`message`")
    assert with_f == "Is the intent of `message` to affirm something?" and plain == "Is the intent to affirm something?" and lower == "is the intent to affirm something?"


# ------------------------------------------------------------------------------------------ renderers

def test_every_renderer_is_valid_and_keeps_full_label_sets():
    K = {"banking77": 77, "clinc150": 151, "massive": 60, "hwu64": 64, "snips": 7, "atis": 5, "mtop": 113, "ag_news": 4, "yahoo_topics": 10,
         "newsgroups20": 20, "dbpedia_14": 14, "tweet_topic": 6, "fin_topic": 20, "sib200_en": 7}
    shapes: dict[str, Counter] = {}
    for name in S.SPECS:
        c = Counter()
        for i in range(120):
            for ex in _render(name, i):
                c[ex["meta"]["shape"]] += 1
                q = ex["questions"].get("answer")
                if name in K and q is not None:
                    assert len(q["criteria"]) == K[name], (name, ex["meta"]["shape"], len(q["criteria"]))
                if name == "trec":
                    for qid, qq in ex["questions"].items():
                        assert len(qq["criteria"]) == (50 if qid == "answer_type" else 6)
                for qq in ex["questions"].values():
                    if qq["type"] == "choice" and ex["meta"]["shape"] != "dmb":
                        assert all(k.strip().lower() not in FORBIDDEN_KEYS for k in qq["criteria"]), (name, list(qq["criteria"])[:3])
        shapes[name] = c
        assert set(c) >= set(S.HARNESS_SHAPES[name]) - {"nouls_subset"}, (name, c)
    # harness-shape specifics
    for ex in [e for i in range(200) for e in _render("banking77", i)]:
        sh = ex["meta"]["shape"]
        q = ex["questions"]["answer"] if "answer" in ex["questions"] else ex["questions"]["label"]
        if sh == "deusser":
            assert ex["state"].keys() == {"query"} and all(v is None for v in q["criteria"].values()) and q["instructions"] == S.CH_BANKING[0]
        elif sh == "di":
            assert ex["state"] == {} and q["instructions"].startswith(S.DI_LINE_BANKING77 + "\n") and list(q["criteria"])[0] == "option_0"
            assert ex["meta"]["primary"] == ["_instructions"]
            if ex["meta"]["order"] if "order" in ex["meta"] else True:
                assert list(q["criteria"].values()) == CTX["banking77"]["di_order"]  # categories.json order, raw names
        elif sh == "dmb":
            assert isinstance(ex["state"], str) and all(k == v for k, v in q["criteria"].items())
        elif sh == "btzsc":
            assert ex["state"].keys() == {"text"} and list(q["criteria"])[0] == "label_000" and "label" in ex["questions"]
    for ex in [e for i in range(200) for e in _render("clinc150", i)]:
        q = ex["questions"]["answer"]
        if ex["meta"]["shape"] == "deusser":
            vals = q["criteria"]
            assert vals["oos"] == S.CLINC_OOS and sum(v is None for v in vals.values()) == 150
            if ex["meta"]["order"] == "canonical":
                assert list(vals)[-1] == "oos"
        elif ex["meta"]["shape"] == "di":
            assert ex["state"] == {} and S.DI_CLINC_OOS in q["criteria"].values() and len(q["criteria"]) == 151
        elif ex["meta"]["shape"] == "dmb":
            assert "out of scope" in q["criteria"] and len(q["criteria"]) == 151
    # massive carries a scenario question on a share of rows; mtop a domain question
    ms = [e for i in range(200) for e in _render("massive", i)]
    assert sum("scenario" in e["questions"] for e in ms) > 30 and all(len(e["questions"]["scenario"]["criteria"]) == 18 for e in ms if "scenario" in e["questions"])
    mt = [e for i in range(200) for e in _render("mtop", i)]
    assert sum("domain" in e["questions"] for e in mt) > 30
    # nlupp: nouls with ontology questions, DMB yes/no rows with a JSON-string state, soft choice
    nl = [e for i in range(200) for e in _render("nlupp", i)]
    kinds = Counter(e["meta"]["shape"] for e in nl)
    assert kinds["nouls_all"] > 20 and kinds["nouls_subset"] > 10 and kinds["dmb"] > 20 and kinds["choice_soft"] > 5
    for e in nl:
        if e["meta"]["shape"] == "dmb":
            st = json.loads(e["state"])
            assert set(st) == {"message", "question", "instruction"} and e["questions"]["decision"]["criteria"] == {"no": "no", "yes": "yes"}
            assert e["labels"]["decision"]["label"] in ("yes", "no")
        elif e["meta"]["shape"] == "nouls_all":
            dom = e["meta"]["domain"]
            assert set(e["questions"]) == set(CTX["nlupp"]["domain_intents"][dom])
            golds = {k for k, l in e["labels"].items() if l["p"] == 1.0}
            assert golds == set(_row("nlupp", int(e["id"].split("-")[-1]))["intents"]) & set(ONTO)
        elif e["meta"]["shape"] == "choice_soft":
            assert "dist" in e["labels"]["answer"]
    # tweets: elcronos raw-string shape with "" criteria
    tw = [e for i in range(100) for e in _render("tweet_topic", i)]
    el = [e for e in tw if e["meta"]["shape"] == "elcronos"]
    assert el and all(isinstance(e["state"], str) and all(v == "" for v in e["questions"]["answer"]["criteria"].values()) for e in el)
    # yahoo jevbench shape drops empty fields
    ya = [e for i in range(60) for e in _render("yahoo_topics", i) if e["meta"]["shape"] == "jevbench"]
    assert ya and all("question" not in e["state"] for e in ya if int(e["id"].split("-")[-1]) % 2 == 1)
    assert all(S.SIB_LABELS[0] in e["questions"]["answer"]["criteria"] or e["meta"]["style"] == "opaque" for e in [x for i in range(20) for x in _render("sib200_en", i)])


def test_renderers_are_deterministic_and_use_many_templates():
    a = [e for i in range(30) for e in _render("hwu64", i)]
    b = [e for i in range(30) for e in _render("hwu64", i)]
    assert a == b
    used = {e["meta"]["template"] for i in range(300) for e in _render("snips", i)}
    assert len(used) >= 5 and "CH_SNIPS:0" in used


# ------------------------------------------------------------------------------------------ hygiene

def _ref(texts, name="snips/test"):
    ref = C.EvalRef()
    ref.add(name, texts)
    return ref


def _ex(i, state, primary, instr="Which?", crit=None):
    q = C.Rendered({"type": "choice", "instructions": instr, "criteria": crit or {"a": None, "b": None}}, {"type": "choice", "label": "a"})
    return make_example(f"x-{i}", state, {"answer": q}, source="snips", bucket="s_intent", license_use="unknown", meta={"primary": primary, "template": "t", "shape": "s", "hf_split": "train"})


def test_guarded_writer_drops_whole_text_at_any_length_and_checks_instructions(tmp_path):
    ref = _ref(["wake me up", "the quick brown fox jumps over the lazy dog"], "massive/test")
    ref.add("snips/test", ["play some jazz"])
    w = S.GuardedWriter(S.SPECS["snips"], ref, root=tmp_path)
    assert w.write(_ex(0, {"utterance": "Wake me up!"}, ["utterance"]), ["Wake me up!"]) is False  # 3 words / 10 chars: still a duplicate
    assert w.write(_ex(1, {"utterance": "play some jazz"}, ["utterance"]), ["play some jazz"]) is False and w.own_whole == 1
    assert w.write(_ex(2, {}, ["_instructions"], instr="Classify:\nwake me up"), ["wake me up"]) is False  # DI shape: text in the instructions
    assert w.write(_ex(3, {"utterance": "something else entirely", "note": "intro\nthe quick brown fox jumps over the lazy dog\nend"}, ["utterance"]),
                   ["something else entirely"]) is False  # a non-primary field line equal to an eval text (exact unit)
    assert w.write(_ex(4, {"utterance": "turn the kitchen lights on", "note": "ok"}, ["utterance"]), ["turn the kitchen lights on"]) is True
    assert w.dropped_whole == 3 and w.dropped_unit == 1
    stats = w.close({"hf_id": "x"})
    assert stats["rows"] == 1
    man = json.loads((tmp_path / "raw" / "intent" / "snips.manifest.json").read_text())
    h = man["hygiene"]
    assert h["dropped_whole_text"] == 3 and h["dropped_exact_unit"] == 1 and h["own_eval_whole_text_duplicates"] == 1
    assert h["by_eval_set"]["cross:massive/test"] == 2 and h["by_eval_set"]["own:snips/test"] == 1 and man["eval_overlap"]["exit_rescan_hits"] == 0
    # verify's probe: primary texts are recovered from meta.primary for dict, raw, DI and DMB-JSON states
    assert S._probe_texts(_ex(5, {"utterance": "abc"}, ["utterance"]))[0] == ["abc"]
    assert S._probe_texts(_ex(6, "raw text", ["_state"]))[0] == ["raw text"]
    assert S._probe_texts(_ex(7, {}, ["_instructions"], instr="line:\nthe text"))[0] == ["the text"]
    assert S._probe_texts(_ex(8, json.dumps({"message": "msg", "question": "q", "instruction": "i"}), ["_json_message"]))[0] == ["msg"]


def test_guarded_writer_refuses_wrong_split(tmp_path):
    texts = [f"evaluated utterance number {i} for the test" for i in range(300)]
    ref = _ref(texts, "snips/test")
    w = S.GuardedWriter(S.SPECS["snips"], ref, root=tmp_path)
    for i, t in enumerate(texts):
        w.write(_ex(i, {"utterance": t}, ["utterance"]), [t])
    for i in range(50):
        w.write(_ex(1000 + i, {"utterance": f"clean training utterance {i}"}, ["utterance"]), [f"clean training utterance {i}"])
    assert w.own_whole == 300
    with pytest.raises(S.Refused):
        w.close({})
    assert not (tmp_path / "raw" / "intent" / "snips.jsonl.zst").exists()
    assert (tmp_path / "raw" / "intent" / "snips.REFUSED.json").exists()


def test_rows_any_reads_json_lines_arrays_and_tsv(tmp_path):
    p = tmp_path / "x.single.json"
    p.write_text('{"text": "a", "label": 1}\n{"text": "b", "label": 2}\n')
    assert [r["text"] for r in S.rows_any(p)] == ["a", "b"]
    p2 = tmp_path / "fold0.json"
    p2.write_text(json.dumps([{"text": "m1", "intents": ["affirm"]}, {"text": "m2"}]))
    assert [r["text"] for r in S.rows_any(p2)] == ["m1", "m2"]
    p3 = tmp_path / "train.tsv"
    p3.write_text("index_id\tcategory\ttext\n1\tsports\tHe \"ran\" fast.\n")
    rows = list(S.rows_any(p3))
    assert rows[0]["category"] == "sports" and rows[0]["text"] == 'He "ran" fast.'


def test_cli_list_runs(capsys):
    assert S.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "banking77" in out and "sib200_en" in out and "test" in out
