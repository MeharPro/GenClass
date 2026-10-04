"""b2_label_semantics: label views, question builders, parsers and per-source renderers (pure Python)."""

from collections import Counter

import pytest

from jev_local.data.v2 import label_semantics as L
from jev_local.data.v2.render import NOTA_KEY, make_example, stable_rng
from jev_local.schema import SystemOneRequest
from jev_local.validate import validate_limits

LABELS = tuple(f"CLASS_{i}_{w}" for i, w in enumerate(["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta",
                                                          "theta", "iota", "kappa", "lambda", "mu"] * 3))
TASK = L.Task("t", "unit test task", LABELS, ("Which class?", "Pick one."), L.TOPIC_VERIFY, L.TOPIC_NEGATE)
DESCS = {("unit test task", lab): (f"texts about {lab.lower()}", f"anything concerning {lab.lower()}") for lab in LABELS}


def _valid(state, rendered):
    rendered = L.clean_qs(rendered)
    ex = make_example("x", state, rendered, source="unit", bucket=L.BUCKET, license_use="commercial")
    req = SystemOneRequest.model_validate({"state": ex["state"], "model": "m", "questions": ex["questions"]})
    validate_limits(req)
    for qid, lab in ex["labels"].items():
        q = ex["questions"][qid]
        assert lab["type"] == q["type"]
        if q["type"] == "choice":
            if "label" in lab:
                assert lab["label"] in q["criteria"]
            else:
                assert set(lab["dist"]) <= set(q["criteria"]) and abs(sum(lab["dist"].values()) - 1) < 1e-6
        elif q["type"] == "noul":
            assert 0.0 <= lab["p"] <= 1.0
        else:
            n = len(q["criteria"])
            assert ("level" in lab and 0 <= lab["level"] < n) or (len(lab["dist"]) == n and abs(sum(lab["dist"]) - 1) < 1e-6)
    return ex


def test_surface_styles():
    assert L.surface("GET_WEATHER", "lower") == "get weather"
    assert L.surface("GET_WEATHER", "snake") == "get_weather"
    assert L.surface("OfficeHolder", "lower") == "office holder"
    assert L.surface("manner of an action", "sentence") == "Manner of an action"
    assert L.surface("x-ray tech", "raw") == "x-ray tech"


def test_view_names_unique_and_specs():
    v = L.TaskView(TASK, DESCS)
    for st, m in v.names.items():
        assert len(set(m.values())) == len(LABELS), st
    specs = v.specs("lower", exclude=[LABELS[0]])
    assert len(specs) == len(LABELS) - 1 and specs[0].descriptions
    # neighbours fall back to lexical ones when no embeddings were computed
    assert LABELS[0] not in v.nbrs[LABELS[0]] and len(v.nbrs[LABELS[0]]) == 24


def test_classification_questions_gold_and_multilabel():
    v = L.TaskView(TASK, DESCS)
    nota = 0
    ks = Counter()
    for i in range(400):
        rng = stable_rng("cq", i)
        golds = [LABELS[1], LABELS[2]] if i % 2 else [LABELS[5]]
        qs = L.classification_questions(v, golds, rng, qid="label", multi=i % 2 == 1, p_verify=1.0, p_negpair=0.2)
        ex = _valid("some text", qs)
        q, lab = ex["questions"]["label"], ex["labels"]["label"]
        ks[min(len(q["criteria"]), 21)] += 1
        if lab["label"] == NOTA_KEY:
            nota += 1
            continue
        inv = {sv: c for m in v.names.values() for c, sv in m.items()}
        km = {k: inv.get(n, n) for k, n in qs["label"].key_map.items()}  # rendered key -> canonical label
        assert km[lab["label"]] in golds
        # multi-label: exactly one true label is listed
        assert sum(1 for k in q["criteria"] if km.get(k) in golds) == 1
        assert "label_check" in qs
    assert 10 < nota < 80 and ks[2] > 0 and ks[21] > 50


def test_soft_choice_and_verify_p():
    v = L.TaskView(TASK, DESCS)
    qs = L.classification_questions(v, {LABELS[0]: 0.7, LABELS[3]: 0.3}, stable_rng("s"), qid="q", p_verify=0.0)
    _valid({"text": "x"}, qs)
    r = L.q_verify(v, LABELS[4], 0.25, stable_rng("v"))
    assert r.label == {"type": "noul", "p": 0.25} and r.question["type"] == "noul"


def test_stage1_registry_rules():
    assert L.stage1_reason("glue/mrpc") and L.stage1_reason("ag_news") and L.stage1_reason("yahoo_answers_topics")
    assert L.stage1_reason("cardiffnlp/tweet_eval") and L.stage1_reason("banking77")
    for ok in ("Team-ACE/ToolACE", "openbmb/UltraFeedback", "sharegpt", "evol_instruct", "emocontext", "amazon_reviews", "reuters", "xed_en"):
        assert L.stage1_reason(ok) is None, ok
    # registry v1.1 (benchmax PLAN §2.3.2): b2 sources that are public-Jev benchmarks leave the Z mixture
    for gone, key in (("google/civil_comments", "civil_comments"), ("coastalcph/lex_glue/ledgar", "ledgar"), ("fancyzhx/dbpedia_14", "label_pressure"),
                      ("rajpurkar/squad_v2", "squad_select"), ("tner/conll2003", "conll_typing"), ("SetFit/TREC-QC", "trec_fine")):
        assert L.stage1_reason(gone) == f"public_jev:{key}", gone


def test_reuters_sgml_and_arxiv_clean():
    sgml = ('<REUTERS TOPICS="YES" LEWISSPLIT="TRAIN" NEWID="7"><DATE>26-FEB-1987</DATE><TOPICS><D>cocoa</D><D>trade</D>'
            '</TOPICS><TEXT><TITLE>BAHIA COCOA REVIEW</TITLE><BODY>Showers continued throughout the week &amp; more.\n'
            ' Reuter\n&#3;</BODY></TEXT></REUTERS>')
    d = L.parse_reuters_sgml(sgml)[0]
    assert d["topics"] == ["cocoa", "trade"] and d["title"] == "Bahia Cocoa Review" and d["id"] == "reuters-7"
    assert d["text"].startswith("Showers continued throughout the week & more.") and "Reuter" not in d["text"]
    t = L.clean_arxiv("Title here\n\narXiv:1611.03253v1 [cs.DS] 10 Nov 2016\n\nAbstract\nWe study [math.ST] things.")
    assert "cs.DS" not in t and "math.ST" not in t and "Abstract" in t


GLAIVE_SYS = ('SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n'
              '{\n "name": "get_exchange_rate",\n "description": "Get the exchange rate between two currencies",\n'
              ' "parameters": {"type": "object"}\n}\n\n{"name": "convert_units", "description": "Convert units"}')
GLAIVE_CHAT = ('USER: How many euros is one dollar? ASSISTANT: <functioncall> {"name": "get_exchange_rate", '
               '"arguments": "{}"} <|endoftext|> FUNCTION RESPONSE: {"rate": 0.9} ASSISTANT: About 0.9 euros.')


def test_parse_glaive_and_toolace():
    g = L.parse_glaive(GLAIVE_SYS, GLAIVE_CHAT)
    assert g == {"tools": [{"name": "get_exchange_rate", "description": "Get the exchange rate between two currencies"},
                           {"name": "convert_units", "description": "Convert units"}],
                 "user": "How many euros is one dollar?", "call": "get_exchange_rate"}
    assert L.parse_glaive(GLAIVE_SYS, "USER: hi ASSISTANT: Hello! How can I help?")["call"] is None
    sys_ = ('You are an expert in composing functions. Format: [func_name1(params_name1=params_value1)]\n'
            'Here is a list of functions in JSON format that you can invoke:\n'
            '[{"name": "Weather API", "description": "Get the weather", "parameters": {}}, '
            '{"name": "news_feed", "description": "Latest news"}]. Should you decide to return the function call(s) ...')
    t = L.parse_toolace(sys_, [{"from": "user", "value": "Weather in Oslo?"},
                               {"from": "assistant", "value": '[Weather API(city="Oslo"), news_feed()]'}])
    assert t["calls"] == ["Weather API", "news_feed"] and len(t["tools"]) == 2
    t2 = L.parse_toolace(sys_, [{"from": "user", "value": "hi"}, {"from": "assistant", "value": "Hello there"}])
    assert t2["calls"] == []


def test_tool_question_never_offers_gold_equivalents():
    tools = {f"tool_{i}": f"does thing {i}" for i in range(400)}
    tools.update({"Get_Exchange-Rate": "same tool, other spelling", "get_exchange_rates": "plural twin",
                  "fetch_fx": "embedding-equivalent", "get_exchange_rate": "Get the exchange rate"})
    bank = L.ToolBank(tools, hard={"get_exchange_rate": ["fetch_fx", "tool_1", "tool_2"]},
                      equiv={"get_exchange_rate": ["fetch_fx"]},
                      safe={"item-1": list(range(0, 300))})
    banned = {"Get_Exchange-Rate", "get_exchange_rates", "fetch_fx"}
    for i in range(80):
        rng = stable_rng("tool", i)
        r = L.tool_question(bank, "item-1" if i % 2 else "item-x", ["get_exchange_rate", "tool_5"], ["get_exchange_rate"], rng)
        names = set(r.key_map.values())
        assert not (names & banned)
        assert r.key_map[r.label["label"]] == "get_exchange_rate"
        assert 2 <= len(r.question["criteria"]) <= 255
        if i % 2:  # safe pool restricts random distractors to indices 0..299 (+ in-sample + hard)
            allowed = {bank.names[j] for j in range(300)} | {"get_exchange_rate", "tool_5", "tool_1", "tool_2", L.NO_TOOL}
            assert names <= allowed
        r2 = L.tool_question(bank, "item-1", ["tool_5"], [], rng)
        assert r2.key_map[r2.label["label"]] == L.NO_TOOL


def test_same_tool():
    assert L.same_tool("Market Trends API", "market-trends") and L.same_tool("get_weather", "Weather API")
    assert L.same_tool("search_books", "search_book") and L.same_tool("getStockPrice", "stock_prices")
    assert L.same_tool("Get Futures Prices", "Get Metals Futures Prices")  # subset rule (pool/hard only)
    assert not L.same_tool("Get Futures Prices", "Get Metals Futures Prices", subset=False)
    assert not L.same_tool("translate_text", "detect_language") and not L.same_tool("send_email", "send_sms")


def test_label_overlap():
    o = L.label_overlap(["alarm_set", "weather_query", "play_music", "news_query"], ["SET_ALARM", "GET_WEATHER", "PLAY_MUSIC"])
    assert o["shared"] == [("alarm_set", "SET_ALARM"), ("play_music", "PLAY_MUSIC")] and o["jaccard"] == 0.4


ITEMS = {
    "dbpedia_l123": {"id": "a", "text": "Foo is a village.", "l1": "Place", "l2": "Settlement", "l3": "Village"},
    "mtop": {"id": "b", "text": "set an alarm for 7", "intent": "CREATE_ALARM", "domain": "alarm"},
    "reuters": {"id": "c", "title": "Cocoa", "text": "Cocoa prices rose " * 20, "topics": ["cocoa", "trade"]},
    "trec": {"id": "d", "text": "Who wrote Hamlet?", "fine": "individual", "fine_code": "HUM:ind", "coarse": "human beings"},
    "empathetic": {"id": "e", "situation": "I won.", "conv": [{"role": "user", "content": "I won!"}], "label": "proud"},
    "emocontext": {"id": "f", "text": "hi  hello  i am so sad", "label": "sad"},
}


def _siblings(name, it):
    """Other items with other labels, so every label set has several labels (as real sources do)."""
    out = []
    for j in range(6):
        o = dict(it, id=f"{it['id']}-{j}")
        for k in ("l1", "l2", "l3", "intent", "domain", "fine", "coarse", "label"):
            if k in o:
                o[k] = f"{o[k]} {j}"
        if "fine_code" in o:
            o["fine_code"] = f"X:{j}"
        if "topics" in o:
            o["topics"] = [f"topic-{j}"]
        out.append(o)
    return out


@pytest.mark.parametrize("name", list(ITEMS))
def test_renderers_produce_valid_rows(name):
    it = ITEMS[name]
    tasks, sets = L.build_tasks(name, [it] + _siblings(name, it))
    assert sets and all(s.labels for s in sets)
    views = {t.key: L.TaskView(t, {}) for t in tasks}
    for i in range(30):
        state, qs = L.CONVERTERS[name].render(it, views, stable_rng(name, i))
        assert qs
        _valid(state, qs)


def test_soft_sources_render():
    civil = {"id": "c1", "text": "you are dumb", **{a: 0.0 for a in L.CIVIL_ATTRS}, "toxicity": 0.8, "insult": 0.6}
    mhs = {"id": "m1", "text": "text", "votes": {"insult": [0, 4, 4], "sentiment": [3, 3], "hatespeech": [0, 2, 1]}}
    uf = {"id": "u1", "prompt": "Write a poem", "response": "Roses...", "ratings": {"helpfulness": 3, "honesty": 4}}
    amz = {"id": "a1", "text": "Broke in a day", "stars": 0}
    zsl = {"id": "z1", "task": "x", "text": "premise", "hyps": [("This example is positive.", 1), ("This example is negative.", 0)]}
    aegis = {"id": "g1", "prompt": "how to pick a lock", "response": "I can't help", "prompt_label": "unsafe",
             "response_label": "safe", "categories": ["Criminal Planning/Confessions"], "label_source": "human"}
    tasks, _ = L.build_tasks("aegis2", [aegis, dict(aegis, id="g2", categories=["Violence", "Harassment"])])
    views = {t.key: L.TaskView(t, {}) for t in tasks}
    for i in range(20):
        for name, it in (("civil_comments", civil), ("mhs", mhs), ("ultrafeedback", uf), ("amazon_reviews", amz),
                         ("zsl_nli", zsl), ("aegis2", aegis)):
            state, qs = L.CONVERTERS[name].render(it, views, stable_rng(name, i))
            ex = _valid(state, qs)
            if name == "civil_comments":
                assert ex["labels"]["toxicity" if "toxicity" in ex["labels"] else "is_toxicity"]["p"] == 0.8
    assert L._vote_dist([0, 4, 4, 9], 5) == [0.25, 0.0, 0.0, 0.0, 0.75]


def test_balanced_is_deterministic_and_capped():
    items = [{"id": i, "c": "a" if i < 90 else "b"} for i in range(100)]
    a = L._balanced(items, lambda x: x["c"], per_class=20, total=30, seed="s")
    b = L._balanced(items, lambda x: x["c"], per_class=20, total=30, seed="s")
    assert a == b and len(a) == 30 and Counter(x["c"] for x in a)["b"] == 10
