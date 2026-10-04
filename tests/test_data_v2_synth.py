"""S0 programmatic families, the shortcut probe, and the b4/b9 converters (pure Python, no torch)."""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from jev_local.data.stats import validate_example
from jev_local.data.v2 import cu as CU
from jev_local.data.v2 import probe as P
from jev_local.data.v2 import procedural as PROC
from jev_local.data.v2.families import REGISTRY
from jev_local.data.v2.families.build import build_family, check_row, make_row, plan_targets
from jev_local.data.v2.families.io import SourceStats, ZstJsonlWriter, read_jsonl

N_PER_FAMILY = 60


@pytest.fixture(scope="module")
def rows() -> dict[str, list[dict]]:
    return {name: [make_row(fam, i, 0, 0.12) for i in range(N_PER_FAMILY)] for name, fam in REGISTRY.items()}


def test_registry_size_and_holdouts():
    assert len(REGISTRY) >= 40
    held = [f for f in REGISTRY.values() if f.heldout]
    assert 4 <= len(held) <= 8
    assert "rec_user_access" in {f.name for f in held}  # v1's failure case is the polarity-fix measurement
    skills = Counter(f.skill for f in REGISTRY.values())
    for s in ("rule", "route", "extract", "record", "long", "score", "multilabel"):
        assert skills[s] >= 2, s


def test_rows_valid_supervised_and_safe(rows):
    for name, rs in rows.items():
        for r in rs:
            assert not validate_example(r), (name, validate_example(r))
            assert check_row(r) is None, (name, check_row(r))
            assert set(r["labels"]) == set(r["questions"]), name  # every question is supervised
            assert r["bucket"] == "s0_families" and r["license_use"] == "commercial" and r["source"] == f"s0_{name}"
            for q in r["questions"].values():
                if q["type"] == "choice":
                    assert all(k.strip().lower() not in {"yes", "no", "true", "false"} for k in q["criteria"])
                    for v in q["criteria"].values():  # string criteria only (OpenRouter rejects objects)
                        assert v is None or isinstance(v, str)
            json.dumps(r)


def test_multi_question_states(rows):
    for name, rs in rows.items():
        mean_q = sum(len(r["questions"]) for r in rs) / len(rs)
        assert mean_q >= 1.9, (name, mean_q)
    allq = [len(r["questions"]) for rs in rows.values() for r in rs]
    assert sum(allq) / len(allq) >= 3.0


def test_deterministic():
    fam = REGISTRY["rule_eligibility"]
    a = json.dumps(make_row(fam, 17, 3, 0.12), sort_keys=True)
    b = json.dumps(make_row(fam, 17, 3, 0.12), sort_keys=True)
    c = json.dumps(make_row(fam, 18, 3, 0.12), sort_keys=True)
    assert a == b and a != c


def test_polarity_balance():
    """Hard nouls are ~50/50 in every family (the probe's q_only view checks wording-level leaks)."""
    for name, fam in REGISTRY.items():
        ps = []
        for i in range(400):
            r = make_row(fam, i, 1, 0.0)
            for qid, q in r["questions"].items():
                lab = r["labels"][qid]
                if q["type"] == "noul" and lab["p"] in (0.0, 1.0):
                    ps.append(lab["p"])
        if len(ps) < 100:
            continue
        rate = sum(ps) / len(ps)
        assert 0.40 <= rate <= 0.60, (name, rate)


def test_negation_pairs_complementary(rows):
    n = 0
    for rs in rows.values():
        for r in rs:
            for qid, lab in r["labels"].items():
                if "_pq" in qid and qid.replace("_pq", "_pnq", 1) in r["labels"]:
                    assert lab["p"] + r["labels"][qid.replace("_pq", "_pnq", 1)]["p"] == pytest.approx(1.0)
                    n += 1
    assert n > 50


def test_padding_keeps_labels():
    fam = REGISTRY["rule_fee_schedule"]
    for i in range(40):
        a = make_row(fam, i, 0, 0.0)
        b = make_row(fam, i, 0, 1.0)
        assert a["labels"] == b["labels"]
        if b["meta"]["padded"]:
            assert len(json.dumps(b["state"])) > len(json.dumps(a["state"]))


def test_long_family_lengths():
    lens = [len(json.dumps(make_row(REGISTRY["long_needle_facts"], i, 0, 0)["state"])) for i in range(60)]
    assert min(lens) > 2000 and max(lens) > 16000


def test_record_needle_label_rederived():
    for i in range(30):
        r = make_row(REGISTRY["long_record_needle"], i, 0, 0)
        q = r["questions"]["status"]
        target = re.search(r"record (\d{5})", q["instructions"] if isinstance(q["instructions"], str) else json.dumps(q["instructions"]), re.I).group(1)
        line = next(line for line in r["state"]["records"].split("\n") if line.startswith(target))
        status = re.search(r"status=([a-z ]+?)(?: \||;|,)", line).group(1)
        assert r["labels"]["status"]["label"] == status


def test_vote_distribution_exact():
    for i in range(30):
        r = make_row(REGISTRY["score_vote_distribution"], i, 0, 0)
        lab = r["labels"]["random_vote"]
        d = lab["dist"]
        vals = list(d.values()) if isinstance(d, dict) else d
        assert sum(vals) == pytest.approx(1.0)
        votes = r["state"]["votes"] if isinstance(r["state"], dict) else None
        if votes and isinstance(d, list):
            tot = sum(votes.values())
            assert [v / tot for v in votes.values()] == pytest.approx(d)


def test_build_family_and_io_roundtrip(tmp_path: Path):
    st = build_family(("cmp_superlative", 300, 0, 0.1, str(tmp_path)))
    for k in ("rows", "decisions", "kinds", "k_hist", "tokens_est", "license_use", "excluded_stage1", "seconds"):
        assert k in st
    assert st["decisions"] >= 300 and st["license_use"] == "commercial"
    back = list(read_jsonl(tmp_path / "s0_cmp_superlative.jsonl.zst"))
    assert len(back) == st["rows"]
    assert json.loads((tmp_path / "s0_cmp_superlative.stats.json").read_text())["rows"] == st["rows"]
    assert {r["split"] for r in back} <= {"train", "dev_mix"}


def test_plan_targets_sum():
    t = plan_targets(300_000, 3000)
    train = sum(v for k, v in t.items() if not REGISTRY[k].heldout)
    assert abs(train - 300_000) < 100
    assert all(t[k] == 3000 for k in t if REGISTRY[k].heldout)


# ============================================================================ probe


def _toy_rows(leaky: bool, n: int = 400) -> list[dict]:
    rng = random.Random(0)
    out = []
    for i in range(n):
        y = rng.random() < 0.5
        word = ("crimson" if y else "azure") if leaky else rng.choice(["crimson", "azure"])
        state = {"note": f"the {word} file has {rng.randint(1, 99)} lines"}
        q = {"type": "noul", "instructions": "Is the flag set?"}
        out.append({"id": f"t{i}", "state": state, "questions": {"q": q}, "labels": {"q": {"type": "noul", "p": float(y)}}})
    return out


def test_probe_detects_leak_and_not_noise():
    leaky = P.probe_rows(_toy_rows(True))["noul"]
    noise = P.probe_rows(_toy_rows(False))["noul"]
    assert leaky["full"] >= 0.95 and leaky["solved"] and leaky["state_leak"]
    assert noise["full"] <= 0.65 and not noise["solved"]


def test_probe_choice_position_shortcut():
    rng = random.Random(1)
    rows = []
    for i in range(300):
        names = [f"opt{j}{rng.randint(0, 999)}" for j in range(4)]
        q = {"type": "choice", "instructions": "Pick one.", "criteria": {n: None for n in names}}
        rows.append({"id": f"c{i}", "state": "x", "questions": {"q": q}, "labels": {"q": {"type": "choice", "label": names[0]}}})
    res = P.probe_rows(rows)["choice"]
    assert res["full"] >= 0.95  # gold always first: position feature catches it


# ============================================================================ converters


def test_procedural_convert_row():
    row = {"id": "table_lookup:train:5", "task": "table_lookup", "level": "2",
           "state": json.dumps({"people": [{"name": "A", "city": "Oslo"}]}),
           "questions": json.dumps({"who": {"type": "choice", "instructions": "Who?", "criteria": {"A": "A", "yes": "x"}},
                                    "n": {"type": "score", "instructions": "How many?", "criteria": ["0", "1", "2"]},
                                    "ok": {"type": "noul", "instructions": "Listed?"},
                                    "post": {"type": "noul", "instructions": "Real?"}}),
           "answers": json.dumps({"who": {"type": "choice", "choice": "A", "probabilities": {"A": 1.0, "yes": 0.0}},
                                  "n": {"type": "score", "score": 1.0, "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0}},
                                  "ok": {"type": "noul", "noul": 1.0}, "post": {"type": "noul", "noul": 0.37}})}
    ex, why = PROC.convert_row(row)
    assert why is None and not validate_example(ex) and check_row(ex) is None
    assert isinstance(ex["state"], dict) and ex["bucket"] == "b4_procedural" and ex["license_use"] == "commercial"
    assert ex["labels"]["who"] == {"type": "choice", "label": "A"} and "yes (label)" in ex["questions"]["who"]["criteria"]
    assert ex["labels"]["n"] == {"type": "score", "level": 1}
    assert ex["labels"]["post"]["p"] == pytest.approx(0.37)
    assert PROC.convert_row(dict(row, task="boolq_recast"))[1] == "stage1"
    assert PROC.convert_row(dict(row, task="state_perturbation"))[1] is None  # no false stage-1 hit


def test_cu_convert_keeps_format():
    path = Path("data/cu/dev.jsonl")
    if not path.exists():
        pytest.skip("no local data/cu")
    with open(path) as f:
        ex = json.loads(f.readline())
    a = CU.convert(ex, "train", 0)
    b = CU.convert(ex, "train", 1)
    for k in ("state", "questions", "labels"):
        assert a[k] == ex[k] and b[k] == ex[k]
    assert a["id"] == ex["id"] and b["id"] == ex["id"] + "~dup1" and b["meta"]["upsample_copy"] == 1
    assert a["meta"]["family_v1"] == ex["family"] and a["bucket"] == "b9_cu" and a["group_id"] == re.sub(r"-p\d+$", "", ex["id"])


def test_stats_merge():
    a, b = SourceStats(), SourceStats()
    r = make_row(REGISTRY["count_filtered"], 0, 0, 0)
    a.add(r)
    b.add(r)
    a.merge(b)
    assert a.rows == 2 and a.decisions == 2 * len(r["labels"])


def test_zst_writer_abort(tmp_path: Path):
    p = tmp_path / "x.jsonl.zst"
    with pytest.raises(RuntimeError):
        with ZstJsonlWriter(p) as w:
            w.write({"a": 1})
            raise RuntimeError("boom")
    assert not p.exists()
