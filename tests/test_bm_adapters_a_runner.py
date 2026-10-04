"""benchmax runner plumbing (pure Python: registry, W1–W5 wire logic, run.json, determinism, HTTP client)."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from jev_local.bench.benchmax import SPECS, Spec, get_spec, load_specs, register
from jev_local.bench.benchmax import runner as R
from jev_local.engine.base import RawDist

ROOT = Path(__file__).resolve().parents[1]

CHOICE = {"type": "choice", "instructions": "Which?", "criteria": {"x": None, "y": "why", "z": None}}
SCORE = {"type": "score", "instructions": "How?", "criteria": ["low", "mid", "high"]}
NOUL = {"type": "noul", "instructions": "Is it?"}


# ------------------------------------------------------------------ registry


def test_registry_has_the_four_specs_and_matches_targets_json():
    specs = load_specs()
    mine = {"deusser_exact@6bbdeb33", "decision_index_0.2.1@87d4650b", "typed_decisions_card@d0e2f0c4", "jevbench_hf_praveenrajus_v0.1.1"}
    assert mine <= set(specs)
    rows = json.loads((ROOT / "bench/public/targets.json").read_text())["targets"]
    spec_ids = {r["spec_id"] for r in rows}
    assert mine <= spec_ids, "every registered spec id is a targets.json spec_id"
    for sid in mine:
        assert specs[sid].owner == "a" and ":" in specs[sid].adapter


def test_registry_tolerates_missing_specs_b_but_not_a_broken_one(monkeypatch):
    assert importlib.util.find_spec("jev_local.bench.benchmax.specs_b") is None or True  # either way load_specs works
    load_specs()
    # a plugin whose own import is broken must raise, not be silently skipped
    import jev_local.bench.benchmax as bm

    monkeypatch.setattr(bm, "PLUGIN_MODULES", ("jev_local.bench.benchmax.specs_a", "tests._bm_broken_plugin"))
    monkeypatch.setattr(bm, "_LOADED", set())
    sys.modules.pop("tests._bm_broken_plugin", None)
    (ROOT / "tests" / "_bm_broken_plugin.py").write_text("import this_module_does_not_exist_xyz\n")
    try:
        with pytest.raises(ModuleNotFoundError):
            bm.load_specs()
    finally:
        (ROOT / "tests" / "_bm_broken_plugin.py").unlink()
        sys.modules.pop("tests._bm_broken_plugin", None)


def test_register_rejects_conflicting_duplicate():
    s = Spec(id="zz_test@1", suite="t", adapter="a.b:C", owner="a")
    register(s)
    register(s)  # identical is fine
    with pytest.raises(ValueError):
        register(Spec(id="zz_test@1", suite="other", adapter="a.b:C"))
    SPECS.pop("zz_test@1")
    with pytest.raises(KeyError):
        get_spec("nope@0")


# ------------------------------------------------------------------ W10 model id


@pytest.mark.parametrize("bad", ["jev-1.13.0", "jev-latest", "typesafe/jev-1.13", "Jev", "mehars", ""])
def test_model_id_rejects_jev_and_malformed(bad):
    with pytest.raises(ValueError):
        R.check_model_id(bad)


def test_model_id_accepts_ours():
    assert R.check_model_id("meharsjev-68m") == "meharsjev-68m"
    assert R.check_model_id("meharsjev-400m-s1") == "meharsjev-400m-s1"


# ------------------------------------------------------------------ W3 precision


def test_largest_remainder_sums_to_exactly_one():
    for probs in ([1 / 151] * 151, [0.333, 0.333, 0.334], [0.005, 0.995], [0.2, 0.2, 0.2, 0.2, 0.2], [0.0, 1.0]):
        r = R.largest_remainder_2dp(probs)
        assert len(r) == len(probs)
        assert abs(sum(r) - 1.0) < 1e-12, (probs, r)
        assert all(round(x, 2) == x for x in r)
        assert sum(abs(a - b) for a, b in zip(r, [p / sum(probs) for p in probs])) <= 0.01 * len(probs)
    assert R.largest_remainder_2dp([]) == []


def test_wire_answers_choice_unrounded_argmax_and_first_tie():
    qs = {"q": CHOICE}
    d = {"q": RawDist("choice", (0.4, 0.4, 0.2), ("x", "y", "z"))}
    a = R.wire_answers(qs, d)["q"]
    assert a["choice"] == "x", "ties go to the first key in request order"
    assert a["probabilities"] == {"x": 0.4, "y": 0.4, "z": 0.2}
    assert a["confidence"] == pytest.approx((3 * 0.4 - 1) / 2)
    # engine labels in another order are re-ordered to the request order
    d2 = {"q": RawDist("choice", (0.2, 0.5, 0.3), ("z", "y", "x"))}
    a2 = R.wire_answers(qs, d2)["q"]
    assert list(a2["probabilities"]) == ["x", "y", "z"] and a2["probabilities"]["x"] == 0.3 and a2["choice"] == "y"


def test_wire_answers_2dp_mode_sums_exactly_and_keeps_unrounded_argmax():
    labels = tuple(f"o{i}" for i in range(151))
    probs = [1.0] * 151
    probs[7] = 1.6  # tiny edge: argmax must come from the unrounded distribution
    s = sum(probs)
    d = {"q": RawDist("choice", tuple(p / s for p in probs), labels)}
    q = {"q": {"type": "choice", "criteria": {l: None for l in labels}}}
    a = R.wire_answers(q, d, "2dp")["q"]
    assert a["choice"] == "o7"
    assert abs(sum(a["probabilities"].values()) - 1.0) < 1e-12
    assert all(round(v, 2) == v for v in a["probabilities"].values())
    u = R.wire_answers(q, d, "unrounded")["q"]
    assert abs(sum(u["probabilities"].values()) - 1.0) < 1e-9 and u["probabilities"]["o7"] == pytest.approx(1.6 / s)


def test_wire_answers_score_and_noul():
    d = {"s": RawDist("score", (0.1, 0.2, 0.7), ("0", "1", "2")), "n": RawDist("noul", (0.73,), ())}
    a = R.wire_answers({"s": SCORE, "n": NOUL}, d)
    assert a["s"]["score"] == pytest.approx(0.2 + 1.4) and a["s"]["legend"] == {"0": "low", "1": "mid", "2": "high"}
    assert list(a["s"]["probabilities"]) == ["0", "1", "2"]
    assert a["n"] == {"type": "noul", "noul": 0.73}
    a2 = R.wire_answers({"s": SCORE, "n": NOUL}, d, "2dp")
    assert a2["s"]["score"] == 1.6 and a2["n"]["noul"] == 0.73


def test_wire_answers_rejects_bad_precision():
    with pytest.raises(ValueError):
        R.wire_answers({"n": NOUL}, {"n": RawDist("noul", (0.5,), ())}, "3dp")


# ------------------------------------------------------------------ W4 refusals


def test_refusal_bodies_carry_the_markers():
    ctx = R.refusal_body("context", model_id="meharsjev-68m", need=9000, limit=8192)
    assert ctx["detail"][0]["type"] == R.MAX_TOKENS_EXCEEDED  # Deußer's permanent-error type
    assert R.CONTEXT_MARKER in ctx["detail"][0]["msg"]  # DI CAPACITY_MARKERS
    opt = R.refusal_body("options", qid="q", n=300)
    assert R.OPTION_CAP_MARKER in opt["detail"][0]["msg"]
    assert R.is_refusal_text(json.dumps(ctx)) and R.is_refusal_text(json.dumps(opt))
    assert not R.is_refusal_text('{"detail": "Model not found: x"}')
    for kind in ("too_few_options", "levels"):
        assert R.is_refusal_text(json.dumps(R.refusal_body(kind, qid="q", n=1)))
    with pytest.raises(ValueError):
        R.refusal_body("nope")


class _FakeEngine:
    """Deterministic fake: uniform-ish distributions; records calls."""

    name = "fake-0"
    max_tokens = 100

    def __init__(self):
        self.calls = 0

    def evaluate(self, state, questions):
        from jev_local.engine.base import EngineResult
        from jev_local.schema import ChoiceQuestion, NoulQuestion

        self.calls += 1
        dists = {}
        for qid, q in questions.items():
            if isinstance(q, NoulQuestion):
                dists[qid] = RawDist("noul", (0.25,), ())
            elif isinstance(q, ChoiceQuestion):
                labs = tuple(q.criteria)
                k = len(labs)
                dists[qid] = RawDist("choice", tuple((2.0 if i == 0 else 1.0) / (k + 1) for i in range(k)), labs)
            else:
                k = len(q.criteria)
                dists[qid] = RawDist("score", tuple(1 / k for _ in range(k)), tuple(str(i) for i in range(k)))
        return EngineResult(dists=dists, input_tokens=7, output_tokens=3, engine=self.name)


def test_inprocess_client_counts_and_refuses_option_cap():
    c = R.InProcessClient(_FakeEngine(), "meharsjev-68m", overflow="refuse", refusal_status=422)
    o = c.system_one({"t": "x"}, {"q": CHOICE, "n": NOUL, "s": SCORE})
    assert o.ok and o.response["model"] == "meharsjev-68m" and set(o.response["answers"]) == {"q", "n", "s"}
    assert o.response["usage"] == {"input_tokens": 7, "output_tokens": 3}
    big = {"q": {"type": "choice", "criteria": {f"o{i}": None for i in range(256)}}}
    o2 = c.system_one("x", big)
    assert o2.status == "refused" and o2.http_status == 422 and R.OPTION_CAP_MARKER in json.dumps(o2.detail)
    o3 = c.system_one("x", {"q": {"type": "choice", "criteria": {"only": None}}})
    assert o3.status == "refused" and R.TOO_FEW_OPTIONS_MARKER in json.dumps(o3.detail)
    o4 = c.system_one("x", {"s": {"type": "score", "criteria": ["a"] * 11}})
    assert o4.status == "refused" and R.SCORE_LEVELS_MARKER in json.dumps(o4.detail)
    with pytest.raises(ValueError):
        R.InProcessClient(_FakeEngine(), "meharsjev-68m", overflow="maybe")


def test_inprocess_client_maps_engine_too_long_to_w4_refusal():
    from jev_local.engine.base import EngineError

    class TooLong(_FakeEngine):
        def evaluate(self, state, questions):
            raise EngineError({"detail": "max_tokens_exceeded", "tokens": 9001, "max_tokens": 100, "qid": "q"}, status=400)

    c = R.InProcessClient(TooLong(), "meharsjev-68m")
    o = c.system_one("x", {"q": CHOICE})
    assert o.status == "refused" and o.http_status == 400
    msg = o.detail["detail"][0]["msg"]
    assert "9,001" in msg and "100" in msg and R.CONTEXT_MARKER in msg

    class Broken(_FakeEngine):
        def evaluate(self, state, questions):
            raise RuntimeError("boom")

    o2 = R.InProcessClient(Broken(), "meharsjev-68m").system_one("x", {"q": CHOICE})
    assert o2.status == "error" and "boom" in json.dumps(o2.detail)


def test_rebuild_state_after_truncation_round_trips_segments():
    from jev_local.serialize import Segment, state_segments

    st = {"a": "hello world", "b": ["x", "y"]}
    segs = state_segments(st)
    segs[0] = Segment("a", "hello")
    st2 = R._rebuild_state(st, segs)
    assert st2 == {"a": "hello", "b": "x\ny"}
    assert [s.text for s in state_segments(st2)] == [s.text for s in segs]
    assert R._rebuild_state("long text", [Segment("", "long")]) == "long"
    assert R._rebuild_state(["p", "q"], [Segment("[0]", "p"), Segment("[1]", "q")]) == ["p", "q"]


# ------------------------------------------------------------------ run.json / session / determinism


def _spec():
    return get_spec("jevbench_hf_praveenrajus_v0.1.1")


def test_run_context_writes_plan_section_6_fields(tmp_path):
    ctx = R.RunContext(_spec(), tmp_path / "run", "meharsjev-68m", {"split": "validation"}, precision="unrounded")
    ctx.harness = {"commit": "abc"}
    ctx.dataset_revisions["x/y"] = {"pinned": "deadbeef"}
    ctx.engine = {"kind": "local", "ckpt_sha256": {"sha256": "00"}}
    ctx.calibration = {"path": None, "sha256": None}
    ctx.hardware = R.hardware_info(4)
    p = ctx.write()
    d = json.loads(p.read_text())
    for key in ("schema", "spec", "model_id", "engine", "calibration", "overflow", "precision", "harness", "dataset_revisions",
                "hardware", "started", "finished", "wall_clock_s", "requests", "determinism", "results", "notes", "status"):
        assert key in d, key
    assert d["schema"] == R.RUN_SCHEMA and d["spec"]["id"] == _spec().id and d["status"] == "running"
    assert d["hardware"]["threads"] == 4 and "cpu_count" in d["hardware"]
    ctx.finish("complete")
    d = json.loads(p.read_text())
    assert d["status"] == "complete" and d["finished"] and d["wall_clock_s"] >= 0


def test_session_counts_truncation_refusals_and_replays_for_w11(tmp_path):
    eng = _FakeEngine()
    client = R.InProcessClient(eng, "meharsjev-68m")
    ctx = R.RunContext(_spec(), tmp_path, "meharsjev-68m", {})
    s = R.Session(client, ctx, keep_first=3, trace=tmp_path / "trace.jsonl")
    for i in range(5):
        o = s.ask({"t": f"s{i}"}, {"q": CHOICE}, {"i": i})
        assert o.ok
    s.ask("x", {"q": {"type": "choice", "criteria": {f"o{i}": None for i in range(300)}}})
    assert ctx.requests == {"sent": 6, "ok": 5, "refused": 1, "errors": 0, "truncated": 0}
    assert ctx.overflow["refused"] == 1
    rep = s.determinism_check()
    assert rep == {"n_resent": 3, "mismatches": 0, "identical": True, "engine": "inprocess"}
    assert eng.calls == 5 + 3
    s.close()
    assert ctx.results["latency"]["n"] == 5
    rows = [json.loads(l) for l in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert len(rows) == 6 and rows[0]["meta"] == {"i": 0} and rows[-1]["status"] == "refused"


def test_determinism_detects_a_flaky_engine(tmp_path):
    class Flaky(_FakeEngine):
        def evaluate(self, state, questions):
            r = super().evaluate(state, questions)
            if self.calls > 2:  # changes its mind on the replay
                r.dists["q"] = RawDist("choice", (0.1, 0.1, 0.8), ("x", "y", "z"))
            return r

    ctx = R.RunContext(_spec(), tmp_path, "meharsjev-68m", {})
    s = R.Session(R.InProcessClient(Flaky(), "meharsjev-68m"), ctx, keep_first=5)
    s.ask("a", {"q": CHOICE})
    s.ask("b", {"q": CHOICE})
    rep = s.determinism_check()
    assert rep["n_resent"] == 2 and rep["mismatches"] == 2 and not rep["identical"]


# ------------------------------------------------------------------ HTTP client classification


def _http_client(handler):
    import httpx

    c = R.HttpClient("http://server.test", "meharsjev-68m", wire_model="jev-local-fast", retries=1)
    c._client = httpx.Client(base_url="http://server.test", transport=httpx.MockTransport(handler))
    return c


def test_http_client_ok_refused_error(monkeypatch):
    import httpx

    seen = []

    def handler(req: httpx.Request):
        body = json.loads(req.content)
        seen.append(body)
        text = json.dumps(body["state"])
        if "long" in text:
            return httpx.Response(400, json={"detail": [{"type": "max_tokens_exceeded", "msg": "state+questions need 9000 tokens; maximum context length is 8192"}]})
        if "wide" in text:
            return httpx.Response(422, json={"detail": "Choice question has 300 options; maximum is 255 options per choice"})
        if "boom" in text:
            return httpx.Response(500, text="oops")
        if "missing" in text:
            return httpx.Response(404, json={"detail": "Model not found: x"})
        return httpx.Response(200, json={"model": "jev-local-fast-0.1.0", "answers": {"q": {"type": "noul", "noul": 0.6}}, "usage": {"input_tokens": 5, "output_tokens": 2}})

    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    c = _http_client(handler)
    o = c.system_one("fine", {"q": NOUL})
    assert o.ok and o.response["model"] == "meharsjev-68m" and o.engine_name == "jev-local-fast-0.1.0"
    assert seen[0]["model"] == "jev-local-fast", "wire model override is what goes on the wire"
    assert c.system_one("long", {"q": NOUL}).status == "refused"
    o2 = c.system_one("wide", {"q": NOUL})
    assert o2.status == "refused" and o2.http_status == 422
    o3 = c.system_one("boom", {"q": NOUL})
    assert o3.status == "error" and o3.http_status == 500
    o4 = c.system_one("missing", {"q": NOUL})
    assert o4.status == "error" and o4.http_status == 404


def test_default_out_and_targets_loader():
    class A:
        ckpt = "/x/models/meharsjev-68m-z"
        base_url = None
        engine = None

    p = R.default_out("deusser_exact@6bbdeb33", A())
    assert p.parts[-3:] == ("benchmax", "meharsjev-68m-z", "deusser_exact_6bbdeb33")
    rows = R.load_targets("deusser_exact@6bbdeb33")
    assert len(rows) == 54 and all(r["spec_id"] == "deusser_exact@6bbdeb33" for r in rows)


def test_sha256_dir_is_deterministic_and_order_independent(tmp_path):
    (tmp_path / "b.json").write_text("{}")
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "heads.safetensors").write_bytes(b"\x00\x01")
    d1 = R.sha256_dir(tmp_path)
    d2 = R.sha256_dir(tmp_path)
    assert d1 == d2 and d1["files"] == 3 and set(d1["per_file"]) == {"b.json", "a.txt", "sub/heads.safetensors"}
    (tmp_path / "a.txt").write_text("y")
    assert R.sha256_dir(tmp_path)["sha256"] != d1["sha256"]


def test_git_head_reads_ref_and_packed(tmp_path):
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "refs" / "heads" / "main").write_text("a" * 40 + "\n")
    assert R.git_head(tmp_path) == "a" * 40
    (git / "refs" / "heads" / "main").unlink()
    (git / "packed-refs").write_text("# pack\n" + "b" * 40 + " refs/heads/main\n")
    assert R.git_head(tmp_path) == "b" * 40
    (git / "HEAD").write_text("c" * 40 + "\n")
    assert R.git_head(tmp_path) == "c" * 40
    assert R.check_commit(tmp_path, "c" * 8)["match"] and not R.check_commit(tmp_path / "nope", "c")["present"]


def test_hardware_info_has_threads_and_platform():
    h = R.hardware_info(3)
    assert h["threads"] == 3 and h["platform"] and h["python"] and h["hostname"]
    assert math.isfinite(float(h["cpu_count"] or 1))
