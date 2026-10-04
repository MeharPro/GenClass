"""Open-baseline adapter contract (open-baselines.md §4.0): option mapping and answer wrapping (pure Python)."""

import pytest

from jev_local.bench import baselines as B


def test_option_mapping():
    c = {"type": "choice", "instructions": "i", "criteria": {"a": "desc a", "b": None}}
    assert B.options_for(c) == [("a", "desc a"), ("b", "b")]
    n = {"type": "noul", "instructions": "i"}
    assert B.options_for(n) == [("no", "No"), ("yes", "Yes")]
    n2 = {"type": "noul", "instructions": "i", "criteria": {"true": "T", "false": "F"}}
    assert B.options_for(n2) == [("no", "F"), ("yes", "T")]
    s = {"type": "score", "instructions": "i", "criteria": ["lo", "hi"]}
    assert B.options_for(s) == [("0", "level 0: lo"), ("1", "level 1: hi")]


def test_answers_and_buckets():
    assert B.to_answer({"type": "noul"}, {"no": 1, "yes": 3}) == {"type": "noul", "noul": 0.75}
    a = B.to_answer({"type": "choice"}, {"a": 2, "b": 2})
    assert a["probabilities"] == {"a": 0.5, "b": 0.5}
    assert B.kbucket("choice", 77) == "choice:31-100" and B.kbucket("score", 5) == "score:3-5" and B.kbucket("noul", 2) == "noul"
    assert B.render_state({"x": "é"}) == '{"x": "é"}'


def test_stubs_and_registry():
    assert {b.adapter for b in B.BASELINES} <= set(B.ADAPTERS)
    assert any(b.key == "deberta-v3-large-zs-c" for b in B.BASELINES)
    assert not any("zeroshot-v2.0\"" in b.hf_id or b.hf_id.endswith("zeroshot-v2.0") for b in B.BASELINES)  # never non -c
    with pytest.raises(NotImplementedError):
        B.run_rows(B.NLIZeroShotAdapter(), [{"id": "x", "request": {"state": "s", "questions": {}}}])
    assert isinstance(B.SystemOneHTTP("http://127.0.0.1:1"), B.Decider)
