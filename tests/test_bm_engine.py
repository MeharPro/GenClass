"""benchmax engine requirements W1–W11 (suite-reproduction-specs.md §1.2) on stub engines and the heuristic
engine: pure Python, no torch. The checkpoint-level checks live in tests/test_bm_engine_vm.py (VM only)."""

from __future__ import annotations

import json
import math
from typing import Mapping

import pytest
from fastapi.testclient import TestClient

from jev_local import api
from jev_local.api import (
    PRESETS,
    ServeConfig,
    determinism_report,
    hardware_info,
    normalize_request,
    parse_precision,
    public_model_id,
    refusal_kind,
    resolve_model_id,
    serve_request,
    system_one,
    system_one_json,
)
from jev_local.confidence import build_answer, largest_remainder, round_probs
from jev_local.engine.base import EngineError, EngineResult, RawDist
from jev_local.engine.encoder.calibrate import describe_calibration, load_calibration
from jev_local.engine.heuristic import HeuristicEngine
from jev_local.engine.registry import Registry
from jev_local.schema import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion, SystemOneRequest
from jev_local.server import app as server
from jev_local.server.app import create_app
from jev_local.validate import OptionCapExceeded, validate_limits

# ---------------------------------------------------------------------------- fixtures


class Stub:
    """Deterministic engine with optional benchmax hooks (positions_needed, fit_state)."""

    def __init__(
        self,
        name: str = "jev-local-fast-0.1.0",
        max_tokens: int = 10_000,
        tokens: int = 100,
        supports: bool = True,
        positions: int | None = None,
        fit_cut: int | None = None,
        fit_raises: bool = False,
        probs: Mapping[str, tuple[float, ...]] | None = None,
    ):
        self.name = name
        self.max_tokens = max_tokens
        self.tokens = tokens
        self._supports = supports
        self._positions = positions
        self._fit_cut = fit_cut
        self._fit_raises = fit_raises
        self._probs = dict(probs or {})
        self.seen: list[tuple[object, dict]] = []
        if positions is not None:
            self.positions_needed = self._positions_needed  # type: ignore[method-assign]
        if fit_cut is not None or fit_raises:
            self.fit_state = self._fit_state  # type: ignore[method-assign]

    def supports(self, req: SystemOneRequest) -> bool:
        return self._supports

    def count_tokens(self, req: SystemOneRequest) -> int:
        return self.tokens

    def _positions_needed(self, req: SystemOneRequest) -> int:
        return int(self._positions)

    def _fit_state(self, state, questions):
        if self._fit_raises:
            raise EngineError({"detail": "max_tokens_exceeded", "tokens": 99_999, "max_tokens": self.max_tokens}, status=400)
        if self._fit_cut:
            return "TRUNCATED", self._fit_cut
        return state, 0

    def evaluate(self, state, questions: Mapping[str, Question]) -> EngineResult:
        self.seen.append((state, dict(questions)))
        dists = {}
        for qid, q in questions.items():
            if isinstance(q, ChoiceQuestion):
                labels = tuple(q.criteria)
                if qid in self._probs:
                    dists[qid] = RawDist("choice", self._probs[qid], labels)
                else:
                    rest = 0.3 / (len(labels) - 1)
                    dists[qid] = RawDist("choice", (0.7,) + (rest,) * (len(labels) - 1), labels)
            elif isinstance(q, ScoreQuestion):
                n = len(q.criteria)
                p = self._probs.get(qid) or tuple([0.0, 1.0] + [0.0] * (n - 2))
                dists[qid] = RawDist("score", tuple(p), tuple(str(i) for i in range(n)))
            else:
                dists[qid] = RawDist("noul", self._probs.get(qid, (0.8,)))
        return EngineResult(dists, self.tokens, 0, self.name)


class StubRegistry(Registry):
    def __init__(self, engines: dict):
        super().__init__(load_fast=False, load_general="off")
        self._given = engines

    def _build(self) -> dict:
        return dict(self._given)


def make_req(model: str = "jev-local", state="My payouts fail.", **extra) -> SystemOneRequest:
    questions = {
        "department": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "pay", "technical": None, "sales": None}},
        "frustration": {"type": "score", "criteria": ["calm", "annoyed", "angry"]},
        "urgent": {"type": "noul", "instructions": "Is it urgent?"},
    }
    questions.update(extra)
    return SystemOneRequest.model_validate({"state": state, "model": model, "questions": questions})


def body(model: str = "jev-local", state="s", **questions) -> dict:
    return {"state": state, "model": model, "questions": questions or {"q": {"type": "noul"}}}


# ---------------------------------------------------------------------------- W3 precision


def test_largest_remainder_sums_to_exactly_one_and_is_deterministic():
    p = [1 / 3, 1 / 3, 1 / 3]
    r = largest_remainder(p, 2)
    assert r == [0.34, 0.33, 0.33]  # ties: earlier index first
    assert round(sum(r), 10) == 1.0
    assert largest_remainder([0.125, 0.125, 0.75], 2) == [0.13, 0.12, 0.75]
    assert largest_remainder([1.0], 2) == [1.0]
    assert largest_remainder([], 2) == []


def test_independent_rounding_can_break_the_di_sum_check_but_lr_cannot():
    # 151 options (CLINC plus): 150 share 0.5 and the first one holds the rest; each 0.5/150 = 0.00333 rounds to
    # 0.00, so independent 2-dp rounding loses 0.5 of mass (|Σp − 1| = 0.5 > 0.01); largest remainder keeps 1.00.
    k = 151
    p = [0.5] + [0.5 / (k - 1)] * (k - 1)
    rounded = round_probs(p, "round", 2)
    assert abs(sum(rounded) - 1.0) > 0.01
    lr = round_probs(p, "lr", 2)
    assert round(sum(lr), 10) == 1.0 and all(round(x, 2) == x for x in lr)
    exact = round_probs(p, "exact", 2)
    assert exact == p and abs(sum(exact) - 1.0) < 1e-9


def test_build_answer_precision_modes_choice_and_score():
    q = ChoiceQuestion(criteria={"a": None, "b": None, "c": None})
    d = RawDist("choice", (0.335, 0.335, 0.33), ("a", "b", "c"))
    exact = build_answer(q, d, precision="exact")
    assert exact.probabilities == {"a": 0.335, "b": 0.335, "c": 0.33}
    assert exact.choice == "a"  # unrounded tie -> first key in request order (W1)
    assert exact.confidence == pytest.approx((3 * 0.335 - 1) / 2)
    lr = build_answer(q, d, precision="lr")
    assert lr.probabilities == {"a": 0.34, "b": 0.33, "c": 0.33} and lr.choice == "a"
    rnd = build_answer(q, d, precision="round")
    assert rnd.probabilities == {"a": 0.34, "b": 0.34, "c": 0.33}  # Σ = 1.01: the v1 drift
    s = ScoreQuestion(criteria=["lo", "mid", "hi"])
    sd = RawDist("score", (0.1, 0.6, 0.3), ("0", "1", "2"))
    se = build_answer(s, sd, precision="exact")
    assert se.score == pytest.approx(0.6 + 0.6) and se.legend == {"0": "lo", "1": "mid", "2": "hi"}  # Σ i·pᵢ unrounded (W2)
    assert list(se.probabilities) == ["0", "1", "2"]
    n = build_answer(NoulQuestion(), RawDist("noul", (0.123456,)), precision="exact")
    assert n.noul == 0.123456
    assert build_answer(NoulQuestion(), RawDist("noul", (0.123456,)), precision="lr").noul == 0.12


def test_parse_precision_and_config_validation():
    assert parse_precision("round2") == ("round", 2) and parse_precision("lr2") == ("lr", 2)
    assert parse_precision("exact") == ("exact", 0) and parse_precision("round") == ("round", 2)
    for bad in ("", "round-1", "lr99x", "float"):
        with pytest.raises(ValueError):
            parse_precision(bad)
    with pytest.raises(ValueError):
        ServeConfig(overflow="drop")
    with pytest.raises(ValueError):
        ServeConfig(refusal_status=500)
    with pytest.raises(ValueError):
        ServeConfig(model_id="jev-68m")  # never jev-*
    with pytest.raises(ValueError):
        ServeConfig.preset("nope")
    assert ServeConfig().rounding == ("round", 2) and ServeConfig().name == "compat"


def test_presets_match_the_spec_table():
    di, jb, de, pp = PRESETS["di"], PRESETS["jevbench"], PRESETS["deusser"], PRESETS["paired2dp"]
    assert (di.precision, di.overflow, di.refusal_status, di.refusal_style) == ("exact", "refuse", 400, "jev")
    assert (jb.precision, jb.overflow, jb.refusal_status) == ("exact", "refuse", 422)
    assert (de.precision, de.overflow, de.refusal_status) == ("exact", "truncate", 400)
    assert (pp.precision, pp.overflow) == ("lr2", "truncate")
    assert PRESETS["compat"] == ServeConfig()
    c = ServeConfig.preset("di", model_id="meharsjev-68m", precision="lr2")
    assert c.model_id == "meharsjev-68m" and c.precision == "lr2" and c.name == "di"
    assert json.loads(json.dumps(c.to_dict()))["overflow"] == "refuse"


def test_serve_request_round_digits_knob_still_works():
    served = serve_request(make_req(), {"fast": Stub()}, round_digits=3)
    assert served.response.answers["department"].probabilities == {"billing": 0.7, "technical": 0.15, "sales": 0.15}
    exact = serve_request(make_req(), {"fast": Stub(probs={"department": (0.123456, 0.5, 0.376544)})}, config=PRESETS["di"])
    assert exact.response.answers["department"].probabilities["billing"] == 0.123456


# ---------------------------------------------------------------------------- W10 model ids


@pytest.mark.parametrize("mid", ["meharsjev-68m", "meharsjev-400m", "typesafe/meharsjev-68m"])
def test_public_ids_route_to_fast_without_a_configured_id(mid):
    assert resolve_model_id(mid) == "fast"
    resp = system_one(make_req(mid), {"fast": Stub(), "heuristic": HeuristicEngine()})
    assert resp.model == mid.removeprefix("typesafe/")  # echoed public id, never jev-local-fast-*


def test_configured_public_id_is_strict_and_reported():
    cfg = ServeConfig(model_id="meharsjev-68m")
    assert resolve_model_id("meharsjev-68m", cfg) == "fast"
    with pytest.raises(EngineError) as e:
        resolve_model_id("meharsjev-400m", cfg)
    assert e.value.status == 404 and e.value.detail == "Model not found: meharsjev-400m"
    es = {"fast": Stub(), "heuristic": HeuristicEngine()}
    assert system_one(make_req("meharsjev-68m"), es, cfg).model == "meharsjev-68m"
    assert system_one(make_req("jev-local"), es, cfg).model == "meharsjev-68m"  # local alias, public name
    assert system_one(make_req("jev-local-heuristic"), es, cfg).model == "jev-local-heuristic-0.1.0"
    assert public_model_id("jev-local", "general", "g-0.1", cfg) == "g-0.1"


def test_local_aliases_are_kept():
    for mid in ("jev-local", "jev-latest", "jev", "jev-1.13.0", "typesafe/jev-1.13", "jev-local-fast", "jev-local-heuristic"):
        resolve_model_id(mid, ServeConfig(model_id="meharsjev-68m"))
    with pytest.raises(EngineError):
        resolve_model_id("meharsjev", ServeConfig())  # no size: not a public id


def test_models_endpoint_lists_the_public_id():
    c = TestClient(create_app(StubRegistry({"fast": Stub(), "heuristic": HeuristicEngine()}), config=ServeConfig(model_id="meharsjev-68m")))
    names = [m["name"] for m in c.get("/v1/models").json()["models"]]
    assert names[0] == "meharsjev-68m" and "jev-local-fast-0.1.0" in names and "jev-latest" in names
    r = c.post("/v1/systemone", json=body("meharsjev-68m"))
    assert r.status_code == 200 and r.json()["model"] == "meharsjev-68m"
    assert c.post("/v1/systemone", json=body("meharsjev-1b")).status_code == 404


# ---------------------------------------------------------------------------- W4 refusals


def test_context_refusal_carries_the_di_marker_in_jev_style():
    es = {"fast": Stub(max_tokens=50, tokens=5_000)}
    with pytest.raises(EngineError) as e:
        serve_request(make_req("jev-local-fast"), es, config=PRESETS["di"])
    assert e.value.status == 400
    [d] = e.value.detail
    assert d["type"] == "max_tokens_exceeded" and "maximum context length" in d["msg"] and "5,000" in d["msg"]
    assert refusal_kind(e.value) == "context"
    with pytest.raises(EngineError) as e:
        serve_request(make_req("jev-local-fast"), es, config=PRESETS["jevbench"])
    assert e.value.status == 422 and "maximum context length" in e.value.detail[0]["msg"]
    # v1 wording is unchanged under the default config
    with pytest.raises(EngineError) as e:
        serve_request(make_req("jev-local-fast"), es)
    assert e.value.status == 400 and "(local limit)" in e.value.detail[0]["msg"]


def test_engine_reported_too_long_is_canonicalised_with_the_run_config():
    raw = EngineError({"detail": "max_tokens_exceeded", "tokens": 9_000, "max_tokens": 8_192}, 400)

    class Raises(Stub):
        def evaluate(self, state, questions):
            raise raw

    with pytest.raises(EngineError) as e:
        serve_request(make_req("jev-local-fast"), {"fast": Raises()}, config=PRESETS["jevbench"])
    assert e.value.status == 422
    assert e.value.detail == [{"type": "max_tokens_exceeded", "msg": "This model's maximum context length is 8,192 tokens; the state plus its longest question need 9,000 tokens"}]


def test_option_cap_refusal_marker_and_status():
    big = {"type": "choice", "criteria": {f"o{i}": None for i in range(256)}}
    req = make_req(big=big)
    with pytest.raises(OptionCapExceeded) as e:
        validate_limits(req, status=422, style="jev")
    assert e.value.status == 422 and e.value.detail == "Choice question 'big' has 256 options; the maximum is 255 options per choice"
    assert refusal_kind(e.value) == "options"
    with pytest.raises(EngineError) as e:
        validate_limits(req)
    assert e.value.status == 400 and e.value.detail == "Choice question 'big' has 256 options; maximum is 255"
    # other limits keep their status whatever the run config
    with pytest.raises(EngineError) as e:
        validate_limits(make_req(s={"type": "score", "criteria": ["a"] * 11}), status=422, style="jev")
    assert e.value.status == 400 and refusal_kind(e.value) is None


def test_server_refusals_and_counters():
    eng = Stub(max_tokens=50, tokens=5_000)
    c = TestClient(create_app(StubRegistry({"fast": eng}), config=ServeConfig.preset("jevbench", model_id="meharsjev-68m")))
    r = c.post("/v1/systemone", json=body("meharsjev-68m"))
    assert r.status_code == 422 and "maximum context length" in r.json()["detail"][0]["msg"]
    r = c.post("/v1/systemone", json=body("meharsjev-68m", big={"type": "choice", "criteria": {f"o{i}": None for i in range(300)}}))
    assert r.status_code == 422 and "options per choice" in r.json()["detail"]
    assert c.post("/v1/systemone", json=body("gpt-4")).status_code == 404
    h = c.get("/healthz").json()["benchmax"]
    assert h["config"]["refusal_status"] == 422 and h["config"]["model_id"] == "meharsjev-68m"
    assert h["counters"]["requests"] == 3 and h["counters"]["refused_context"] == 1 and h["counters"]["refused_options"] == 1
    assert h["counters"]["errors_4xx"] == 1 and h["counters"]["served"] == 0
    run = c.get("/x/v1/run").json()
    assert run["config"] == h["config"] and "hardware" in run and run["registry"]["fast_options"] == {}


def test_length_check_runs_before_supports():
    # A too-long request must be a capacity refusal, never "does not support this request" (DI retries the latter).
    es = {"fast": Stub(max_tokens=50, tokens=5_000, supports=False)}
    with pytest.raises(EngineError) as e:
        serve_request(make_req("jev-local-fast"), es, config=PRESETS["di"])
    assert e.value.detail[0]["type"] == "max_tokens_exceeded"


def test_positions_needed_lets_multi_pass_requests_through():
    # 255 options x long texts: the whole request exceeds max_tokens but one branch (state + header + longest option)
    # fits, which is what the per-option-isolated encoder needs. The router must ask the engine, not count everything.
    eng = Stub(max_tokens=8_192, tokens=11_000, positions=1_200)
    served = serve_request(make_req("jev-local-fast"), {"fast": eng}, config=PRESETS["di"])
    assert served.response.usage.input_tokens == 11_000
    with pytest.raises(EngineError):
        serve_request(make_req("jev-local-fast"), {"fast": Stub(max_tokens=8_192, tokens=11_000)}, config=PRESETS["di"])


# ---------------------------------------------------------------------------- W5 overflow


def test_truncate_mode_uses_the_engine_fit_state_and_reports_the_cut():
    eng = Stub(fit_cut=37)
    served = serve_request(make_req("jev-local-fast"), {"fast": eng}, config=PRESETS["deusser"])
    assert served.truncated_tokens == 37
    assert eng.seen[-1][0] == "TRUNCATED"  # the engine evaluated the fitted state
    served = serve_request(make_req("jev-local-fast"), {"fast": eng}, config=PRESETS["di"])  # refuse mode: untouched
    assert served.truncated_tokens == 0 and eng.seen[-1][0] == "My payouts fail."


def test_truncate_mode_refuses_when_even_the_fitted_state_cannot_host_a_question():
    eng = Stub(fit_raises=True)
    with pytest.raises(EngineError) as e:
        serve_request(make_req("jev-local-fast"), {"fast": eng}, config=ServeConfig.preset("deusser", refusal_status=422))
    assert e.value.status == 422 and "maximum context length" in e.value.detail[0]["msg"]
    # auto mode: fall through to the next engine
    served = serve_request(make_req(), {"fast": eng, "heuristic": HeuristicEngine()}, config=PRESETS["deusser"])
    assert served.engine_key == "heuristic"


def test_truncation_header_and_counters_over_http():
    eng = Stub(fit_cut=120)
    c = TestClient(create_app(StubRegistry({"fast": eng}), config=PRESETS["deusser"]))
    r = c.post("/v1/systemone", json=body("jev-local-fast"))
    assert r.status_code == 200 and r.headers["x-jev-local-truncated"] == "120"
    r2 = c.post("/v1/systemone", json=body("jev-local-fast"))
    assert r2.status_code == 200
    counters = c.get("/healthz").json()["benchmax"]["counters"]
    assert counters["truncated_requests"] == 2 and counters["truncated_tokens"] == 240 and counters["served"] == 2
    c2 = TestClient(create_app(StubRegistry({"fast": Stub()}), config=PRESETS["deusser"]))
    assert "x-jev-local-truncated" not in c2.post("/v1/systemone", json=body("jev-local-fast")).headers


# ---------------------------------------------------------------------------- W8 null == "" == key


def test_normalize_request_treats_value_equal_to_key_as_bare_label():
    req = SystemOneRequest.model_validate(
        {"state": "x", "model": "jev-local", "questions": {
            "q": {"type": "choice", "criteria": {"Positive": "Positive", "Neutral": " Neutral ", "Negative": "", "Other": None, "Mixed": "both"}},
            "n": {"type": "noul"},
        }}
    )
    out = normalize_request(req)
    assert out.questions["q"].criteria == {"Positive": None, "Neutral": None, "Negative": "", "Other": None, "Mixed": "both"}
    assert list(out.questions["q"].criteria) == ["Positive", "Neutral", "Negative", "Other", "Mixed"]
    assert out.questions["n"] is req.questions["n"]
    assert normalize_request(make_req()) is make_req() or normalize_request(make_req()).questions == make_req().questions


def test_label_alias_is_applied_to_the_engine_and_can_be_switched_off():
    eng = Stub()
    req = make_req(dmb={"type": "choice", "criteria": {"card_arrival": "card_arrival", "lost_card": "lost_card"}})
    serve_request(req, {"fast": eng})
    assert eng.seen[-1][1]["dmb"].criteria == {"card_arrival": None, "lost_card": None}
    serve_request(req, {"fast": eng}, config=ServeConfig(label_alias=False))
    assert eng.seen[-1][1]["dmb"].criteria == {"card_arrival": "card_arrival", "lost_card": "lost_card"}


def test_null_empty_and_key_descriptions_give_identical_heuristic_answers():
    es = {"heuristic": HeuristicEngine()}
    cfg = PRESETS["di"]
    base = {"state": {"query": "I lost my card yesterday"}, "model": "jev-local",
            "questions": {"q": {"type": "choice", "instructions": "Which intent?", "criteria": {"lost_card": None, "card_arrival": None}}}}
    a = system_one_json(base, es, cfg)
    b = system_one_json({**base, "questions": {"q": {**base["questions"]["q"], "criteria": {"lost_card": "", "card_arrival": ""}}}}, es, cfg)
    c = system_one_json({**base, "questions": {"q": {**base["questions"]["q"], "criteria": {"lost_card": "lost_card", "card_arrival": "card_arrival"}}}}, es, cfg)
    assert a == b == c and a["answers"]["q"]["choice"] == "lost_card"


# ---------------------------------------------------------------------------- W6 / W7 through the server


@pytest.fixture(scope="module")
def heuristic_client():
    with TestClient(create_app(Registry(load_fast=False, load_general="off"), config=PRESETS["di"])) as c:
        yield c


@pytest.mark.parametrize("state", [{}, "", [], {"text": ""}])
def test_empty_state_with_everything_in_instructions(heuristic_client, state):
    r = heuristic_client.post("/v1/systemone", json={
        "state": state, "model": "jev-local",
        "questions": {"q": {"type": "choice", "instructions": "Question: 2 + 2 = ? Options below.",
                            "criteria": {"A": "3", "B": "4", "C": "5"}},
                      "n": {"type": "noul", "instructions": "Is the answer even?"}},
    })
    assert r.status_code == 200, r.text
    a = r.json()["answers"]
    assert a["q"]["choice"] in {"A", "B", "C"} and all(math.isfinite(v) for v in a["q"]["probabilities"].values())
    assert abs(sum(a["q"]["probabilities"].values()) - 1) < 1e-9 and 0 <= a["n"]["noul"] <= 1


def test_object_instructions_and_criteria(heuristic_client):
    r = heuristic_client.post("/v1/systemone", json={
        "state": {"text": "You are all idiots"}, "model": "jev-local",
        "questions": {
            "harassment": {"type": "noul", "instructions": {"category": {"name": "harassment", "definition": "insults a person"},
                                                             "question": "Does the text fall under the category?"},
                           "criteria": {"true": {"meaning": "yes"}, "false": ["no", "not at all"]}},
            "coherence": {"type": "score", "instructions": {"criterion": "coherence", "question": "Rate it"},
                          "criteria": [{"level": "bad"}, "ok", ["good", "very good"]]},
            "relevance": {"type": "noul", "instructions": {"task": "find docs", "candidate": "a doc"}},
        },
    })
    assert r.status_code == 200, r.text
    a = r.json()["answers"]
    assert a["coherence"]["legend"] == {"0": {"level": "bad"}, "1": "ok", "2": ["good", "very good"]}
    assert set(a) == {"harassment", "coherence", "relevance"}


def test_255_options_through_the_server(heuristic_client):
    crit = {f"option_{i}": f"Intent number {i} about banking topic {i % 7}" for i in range(255)}
    r = heuristic_client.post("/v1/systemone", json={"state": {}, "model": "jev-local",
                                                     "questions": {"q": {"type": "choice", "instructions": "Text: where is my card? Pick.", "criteria": crit}}})
    assert r.status_code == 200
    p = r.json()["answers"]["q"]["probabilities"]
    assert list(p) == list(crit) and abs(sum(p.values()) - 1) < 1e-9  # exact precision: Σp == 1 even at K = 255
    r = heuristic_client.post("/v1/systemone", json={"state": "x", "model": "jev-local",
                                                     "questions": {"q": {"type": "choice", "criteria": {f"o{i}": None for i in range(256)}}}})
    assert r.status_code == 400 and "options per choice" in r.json()["detail"]


# ---------------------------------------------------------------------------- in-process adapter path and W11


def test_system_one_json_equals_the_http_body():
    es = {"fast": Stub(probs={"department": (0.123456, 0.5, 0.376544)}), "heuristic": HeuristicEngine()}
    cfg = ServeConfig.preset("di", model_id="meharsjev-68m")
    req = {"state": "My payouts fail.", "questions": {k: v for k, v in make_req().model_dump(mode="json")["questions"].items()}}
    inproc = system_one_json(req, es, cfg, model="meharsjev-68m")
    c = TestClient(create_app(StubRegistry(es), config=cfg))
    http = c.post("/v1/systemone", json={**req, "model": "meharsjev-68m"}).json()
    assert inproc == http and http["model"] == "meharsjev-68m"
    assert http["answers"]["department"]["probabilities"]["billing"] == 0.123456
    assert set(http["answers"]["department"]) == {"type", "choice", "confidence", "probabilities"}
    assert set(http["answers"]["frustration"]) == {"type", "score", "confidence", "legend", "probabilities"}


def test_serve_json_returns_the_body_and_the_run_accounting_meta():
    from jev_local.api import serve_json

    eng = Stub(fit_cut=42, tokens=321)
    resp, meta = serve_json(make_req().model_dump(mode="json"), {"fast": eng}, ServeConfig.preset("deusser", model_id="meharsjev-68m"))
    assert resp["model"] == "meharsjev-68m" and set(resp) == {"model", "answers", "usage"}
    assert meta["engine"] == "fast" and meta["truncated_tokens"] == 42 and meta["input_tokens"] == 321 and meta["latency_ms"] >= 0
    resp2, meta2 = serve_json({k: v for k, v in make_req().model_dump(mode="json").items() if k != "model"}, {"fast": Stub()}, model="meharsjev-68m")
    assert resp2["model"] == "meharsjev-68m" and meta2["truncated_tokens"] == 0


def test_determinism_report_on_the_heuristic_engine():
    es = {"heuristic": HeuristicEngine()}
    reqs = [
        {"state": {"query": "where is my new card"}, "model": "jev-local",
         "questions": {"intent": {"type": "choice", "instructions": "Which intent?", "criteria": {"card_arrival": None, "lost_card": None, "pin": None}},
                       "urgent": {"type": "noul", "instructions": "Is it urgent?"}}},
        {"state": "", "model": "jev-local", "questions": {"s": {"type": "score", "instructions": "angry?", "criteria": ["calm", "angry"]}}},
    ]
    rep = determinism_report(lambda r: system_one_json(r, es, PRESETS["di"]), reqs, repeats=3)
    assert rep["requests"] == 2 and rep["repeats"] == 3
    assert rep["identical_bytes_fraction"] == 1.0 and rep["repeat_flip_rate"] == 0.0 and rep["max_abs_diff_repeat"] == 0.0
    assert rep["permutation"]["checked"] == 1 and rep["permutation"]["argmax_flips"] == 0 and rep["permutation"]["max_abs_diff"] < 1e-12
    assert rep["latency_ms"]["p50"] >= 0 and "cpu_count" in rep["hardware"]
    assert len(rep["per_request"]) == 2 and rep["per_request"][0]["permutation"]["argmax_flips"] == 0


def test_determinism_report_detects_a_flaky_engine():
    calls = {"n": 0}

    class Flaky(Stub):
        def evaluate(self, state, questions):
            calls["n"] += 1
            self._probs = {"department": (0.7, 0.2, 0.1) if calls["n"] % 2 else (0.2, 0.7, 0.1)}
            return super().evaluate(state, questions)

    es = {"fast": Flaky()}
    req = make_req().model_dump(mode="json")
    rep = determinism_report(lambda r: system_one_json(r, es, PRESETS["di"]), [req], repeats=4, permute=False)
    assert rep["identical_bytes_fraction"] == 0.0 and rep["repeat_flip_rate"] > 0 and rep["max_abs_diff_repeat"] == pytest.approx(0.5)


def test_hardware_info_is_pure_python():
    info = hardware_info()
    assert {"platform", "python", "cpu_count"} <= set(info)
    assert "torch" not in info or isinstance(info["torch"], str)


# ---------------------------------------------------------------------------- W9 calibration files


def test_load_calibration_replaces_and_describes(tmp_path):
    f = tmp_path / "global.json"
    f.write_text(json.dumps({"version": 2, "choice": 1.4, "by_bucket": {"choice:2": 1.1}, "by_header": {"abc": 0.7}}))
    calib, info = load_calibration(f)
    assert calib["choice"] == 1.4 and calib["noul"] == 1.0 and calib["score"] == 1.0  # defaults for missing v1 keys
    assert info["source"] == str(f) and len(info["sha256"]) == 64 and info["n_by_header"] == 1 and info["n_by_bucket"] == 1
    calib2, info2 = load_calibration(f, drop_by_header=True)
    assert calib2["by_header"] == {} and info2["by_header_dropped"] is True and info2["n_by_header"] == 0
    calib3, info3 = load_calibration({"noul": 0.8})
    assert calib3["noul"] == 0.8 and info3["source"] == "inline"
    with pytest.raises(ValueError):
        load_calibration({"choice": -1})
    f.write_text("{not json")
    with pytest.raises(ValueError):
        load_calibration(f)
    d = describe_calibration({"noul": 1.0, "choice": 1.0, "score": 1.0, "by_header": {}})
    assert d["version"] == 1 and d["noul_platt"] is False


def test_registry_forwards_fast_options_only_when_set(monkeypatch, tmp_path):
    import sys
    import types

    built: list[tuple] = []

    class FakeFast:
        name = "jev-local-fast-0.1.0"
        max_tokens = 8192

        def __init__(self, ckpt, **kw):
            built.append((ckpt, kw))
            self.calibration_info = {"source": kw.get("calibration", "checkpoint")}
            self.meta = {"max_len": 8192}

        def supports(self, req):
            return True

        def count_tokens(self, req):
            return 1

        def evaluate(self, state, questions):
            raise NotImplementedError

    from jev_local.engine import registry as reg

    mod = types.ModuleType("jev_bm_stub")
    mod.FakeFast = FakeFast
    monkeypatch.setitem(sys.modules, "jev_bm_stub", mod)
    monkeypatch.setattr(reg, "FAST_IMPORT", ("jev_bm_stub", "FakeFast"))
    ckpt = tmp_path / "ck"
    (ckpt / "backbone").mkdir(parents=True)
    (ckpt / "heads.safetensors").write_bytes(b"x")
    r = Registry(fast_ckpt=ckpt, load_general="off", fast_options={"calibration": None, "threads": None})
    r.engines()
    assert built[-1] == (ckpt, {}) and r.fast_options == {}
    r2 = Registry(fast_ckpt=ckpt, load_general="off", fast_options={"calibration": "/tmp/g.json", "threads": 4, "max_tokens": None})
    r2.engines()
    assert built[-1][1] == {"calibration": "/tmp/g.json", "threads": 4}
    d = r2.describe()
    assert d["fast"]["calibration"] == {"source": "/tmp/g.json"} and d["fast"]["checkpoint_sha256"]["heads.safetensors"]
    assert "with ['calibration', 'threads']" in r2.notes["fast"]


# ---------------------------------------------------------------------------- command line


def test_server_cli_print_config(capsys):
    rc = server.main(["--preset", "di", "--model-id", "meharsjev-68m", "--calib", "/tmp/g.json", "--threads", "8",
                      "--global-calibration-only", "--print-config"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["config"]["precision"] == "exact" and out["config"]["model_id"] == "meharsjev-68m" and out["config"]["refusal_style"] == "jev"
    assert out["fast_options"] == {"calibration": "/tmp/g.json", "drop_header_calibration": True, "threads": 8}
    rc = server.main(["--preset", "jevbench", "--precision", "lr2", "--overflow", "truncate", "--print-config"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["config"] == {**PRESETS["jevbench"].to_dict(), "precision": "lr2", "overflow": "truncate"}
    assert out["fast_options"]["threads"] == 16  # default: leave ~16 of the VM's cores free


def test_importing_the_server_does_not_import_torch():
    import subprocess
    import sys

    code = "import sys, jev_local.api, jev_local.server.app, jev_local.engine.registry; print(sorted(m for m in ('torch','transformers') if m in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "[]"
