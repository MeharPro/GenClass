"""label_desc: prompt building, response parsing/cleaning and the cache (pure Python, no server)."""

import json

from jev_local.data.v2 import label_desc as ld


def test_clean_descriptions_filters_and_dedups():
    raw = ["- The customer asks when their new card will arrive.", "card_arrival", "When will my card come?",
           "the customer asks when their new card will arrive", "too short", "line one\nline two continues here",
           "1. Questions about the delivery time of a payment card"]
    got = ld.clean_descriptions(raw, "card_arrival")
    assert got == ["The customer asks when their new card will arrive.", "When will my card come?",
                   "Questions about the delivery time of a payment card"]
    assert ld.clean_descriptions(["a b c d"] * 5, "x") == ["a b c d"]


def test_parse_response_shapes():
    assert ld.parse_response('{"descriptions": ["a b c", "d e f"]}') == ["a b c", "d e f"]
    assert ld.parse_response('noise ["x y z", "u v w"] noise') == ["x y z", "u v w"]
    assert ld.parse_response("1. first one here\n2. second one here") == ["1. first one here", "2. second one here"]


def test_messages_share_prefix_within_chunk():
    labels = tuple(f"label_{i}" for i in range(150))
    ls = ld.LabelSet("Some task", labels, examples={"label_3": ["an example text"]})
    m3 = ld.build_messages(ls, "label_3")[1]["content"]
    m7 = ld.build_messages(ls, "label_7")[1]["content"]
    m70 = ld.build_messages(ls, "label_70")[1]["content"]
    pre = lambda s: s.split("\n\nLabel to describe")[0]  # noqa: E731
    assert pre(m3) == pre(m7) != pre(m70)  # same fixed chunk -> same prompt prefix (llama.cpp prompt cache)
    assert '"label_3"' in m3 and "an example text" in m3 and "150 labels in total" in m3
    assert ld.build_messages(ld.LabelSet("t", ("a", "b")), "a")[0]["role"] == "system"


def test_cache_roundtrip_and_describe_noop(tmp_path):
    p = tmp_path / "descriptions.jsonl"
    w = ld._CacheWriter(p)
    w.add("ctx", "alpha", ["first description here", "second description here"])
    w.add("ctx", "alpha", ["newer description wins", "another newer one"])
    with open(p, "a") as f:
        f.write('{"label": "torn"')  # a torn last line must not break readers
    cache = ld.load_descriptions(p)
    assert cache[("ctx", "alpha")] == ("newer description wins", "another newer one")
    rows = [json.loads(l) for l in p.read_text().splitlines()[:2]]
    assert set(rows[0]) >= {"label", "context", "descriptions", "model"}
    # everything cached -> no server needed, no requests
    out, stats = ld.describe([ld.LabelSet("ctx", ("alpha",))], cache_path=p, autostart=False, progress=False)
    assert stats.requests == 0 and out[("ctx", "alpha")][0] == "newer description wins"


def test_label_sets_io(tmp_path):
    sets = [ld.LabelSet("c1", ("a", "b"), {"a": "alpha"}, {"a": ["ex"]}), ld.LabelSet("c2", ("x",))]
    ld.write_label_sets(sets, tmp_path / "ls.jsonl")
    back = ld.read_label_sets(tmp_path / "ls.jsonl")
    assert [(s.context, s.labels, dict(s.display)) for s in back] == [("c1", ("a", "b"), {"a": "alpha"}), ("c2", ("x",), {})]
    assert ld._stable_seed("c1", "a") == ld._stable_seed("c1", "a") != ld._stable_seed("c1", "b")
