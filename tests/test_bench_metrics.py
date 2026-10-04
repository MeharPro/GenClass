"""jevbench scorer: primitives, answer parsing, failure handling, indices and the bootstrap (numpy only)."""

import math

import numpy as np
import pytest

from jev_local.bench import metrics as M


def test_rank_spearman_ap_auroc():
    assert list(M.rankdata(np.array([3.0, 1.0, 2.0, 2.0]))) == [4.0, 1.0, 2.5, 2.5]
    assert M.spearman(np.array([1, 2, 3, 4.0]), np.array([10, 20, 30, 40.0])) == pytest.approx(1.0)
    assert M.spearman(np.array([1, 2, 3, 4.0]), np.array([4, 3, 2, 1.0])) == pytest.approx(-1.0)
    s, g = np.array([0.9, 0.8, 0.7, 0.6]), np.array([1, 0, 1, 0])
    assert M.average_precision(s, g) == pytest.approx((1 + 2 / 3) / 2)
    # a tie group counts as one threshold: precision 0.5 at recall 1
    assert M.average_precision(np.array([0.5, 0.5]), np.array([1, 0])) == pytest.approx(0.5)
    assert M.auroc(s, g) == pytest.approx(0.75)
    assert M.f1_binary(np.array([1, 1, 0]), np.array([1, 0, 1])) == pytest.approx(0.5)


def test_answer_vector_rounding_and_failures():
    q = {"type": "choice", "criteria": {"a": None, "b": None, "c": None}}
    labels = M.canonical_labels(q)
    v, off = M.answer_vector(q, labels, {"type": "choice", "probabilities": {"b": 0.5, "a": 0.49, "zzz": 0.3}}, None)
    assert np.allclose(v, [0.49 / 0.99, 0.5 / 0.99, 0]) and not off
    assert M.answer_vector(q, labels, {"type": "choice", "probabilities": {"a": 0, "b": 0}}, None)[0] is None
    assert M.answer_vector(q, labels, None, None)[0] is None
    v, _ = M.answer_vector(q, labels, {"type": "choice", "probabilities": {"a": 0.334, "b": 0.333, "c": 0.333}}, 2)
    assert np.allclose(v, [1 / 3] * 3)  # rounding to 2 dp ties the three; argmax breaks by canonical order
    n = {"type": "noul"}
    assert np.allclose(M.answer_vector(n, ["false", "true"], {"type": "noul", "noul": 0.874}, 2)[0], [0.13, 0.87])


def _rows(n=40, K=3):
    rows = []
    for i in range(n):
        g = ["a", "b", "c"][i % K]
        rows.append({"id": f"d/main/{i}", "dataset": "d", "variant": "main", "item": str(i), "stratum": g,
                     "request": {"state": "s", "questions": {"answer": {"type": "choice", "instructions": "i",
                                                                         "criteria": {"a": None, "b": None, "c": None}}}},
                     "gold": {"answer": {"type": "choice", "label": g}}})
    return rows


def _answers(rows, mode):
    out = {}
    for r in rows:
        g = r["gold"]["answer"]["label"]
        if mode == "perfect":
            p = {k: (0.98 if k == g else 0.01) for k in "abc"}
        elif mode == "uniform":
            p = {k: 1 / 3 for k in "abc"}
        else:
            out[r["id"]] = {"id": r["id"], "ok": False}
            continue
        out[r["id"]] = {"id": r["id"], "ok": True, "answers": {"answer": {"type": "choice", "probabilities": p}}}
    return out


def test_collect_score_index_and_rule():
    rows = _rows()
    preds = {s: M.collect(rows, _answers(rows, s), None) for s in ("perfect", "uniform", "failed")}
    prim = M.primary_for("d", "acc", ("choice",))
    sc = M.score_dataset("d", "A", prim, preds, B=200, rng=np.random.default_rng(0))
    assert sc.chance == pytest.approx(1 / 3)
    assert sc.metric["perfect"] == 1.0 and sc.skill["perfect"] == 1.0
    # uniform: ties break to the first canonical label, so accuracy is the share of gold "a"
    assert sc.metric["uniform"] == pytest.approx(14 / 40)
    assert sc.metric["failed"] == 0.0 and preds["failed"].n_failed_requests == 40
    assert sc.ds["uniform"] == pytest.approx(0.0, abs=1e-2)  # uniform ~ the (near-balanced) prior
    assert sc.ds["failed"] == sc.ds["uniform"]  # failures are scored as uniform
    assert sc.ds["perfect"] > 0.99
    lo, hi = M.ci(sc.boot_metric["perfect"] - sc.boot_metric["uniform"])
    assert lo > 0
    sc2 = M.score_dataset("e", "B", prim, {"perfect": preds["perfect"], "jev": preds["uniform"]}, 100, np.random.default_rng(1))
    rule = M.decision_rule([sc2], "perfect", "jev")
    assert rule["a_ci_excludes_0"] and rule["b_wins"] == 1 and not rule["b_ok"] and rule["c_ok"]
    assert M.index([sc2], "perfect") == pytest.approx(100.0)


def test_flip_rate_zero_for_order_invariant():
    rows = _rows(12)
    main = M.collect(rows, _answers(rows, "perfect"), 2)
    shuf_rows = [dict(r, id=r["id"].replace("/main/", "/shuffle/"), variant="shuffle",
                      request={"state": "s", "questions": {"answer": {"type": "choice", "instructions": "i",
                                                                       "criteria": {"c": None, "a": None, "b": None}}}})
                 for r in rows]
    shuf_ans = {k.replace("/main/", "/shuffle/"): v for k, v in _answers(rows, "uniform").items()}
    shuf = M.collect(shuf_rows, shuf_ans, 2, label_order=M.main_label_order(rows))
    assert shuf.heads["answer"].labels == ["a", "b", "c"]
    # canonical labels come from each variant's own criteria order, so uniform predictions all break to
    # the first listed label of THAT variant; aligning by item compares apples to apples
    fr = M.flip_rate(main, M.collect(rows, _answers(rows, "perfect"), 2))
    assert fr["flip_rate"] == 0.0 and fr["n"] == 12
    assert M.flip_rate(main, shuf)["n"] == 12


def test_multi_head_primaries():
    rows = []
    for i in range(20):
        labs = {"x": i % 2 == 0, "y": i % 5 == 0}
        rows.append({"id": f"m/main/{i}", "dataset": "m", "variant": "main", "item": str(i), "stratum": "s",
                     "request": {"state": "s", "questions": {k: {"type": "noul", "instructions": k} for k in labs}},
                     "gold": {k: {"type": "noul", "label": v} for k, v in labs.items()}})
    ans = {r["id"]: {"ok": True, "answers": {k: {"type": "noul", "noul": 0.9 if g["label"] else 0.1}
                                             for k, g in r["gold"].items()}} for r in rows}
    p = M.collect(rows, ans, 2)
    assert M.PRIMARY["macro_f1"].fn(p.heads) == 1.0
    assert M.PRIMARY["micro_f1"].fn(p.heads) == 1.0
    assert M.PRIMARY["mean_auprc"].fn(p.heads) == 1.0
    assert M.PRIMARY["macro_f1"].chance(p.heads) == pytest.approx(np.mean([2 * .5 / 1.5, 2 * .2 / 1.2]))
    hm = M.head_metrics(p.heads["x"])
    assert hm["band_acc"] == 1.0 and hm["auroc"] == 1.0


def test_score_heads_and_soft_gold():
    rows = []
    for i in range(10):
        lv = i % 3
        rows.append({"id": f"s/main/{i}", "dataset": "s", "variant": "main", "item": str(i), "stratum": str(lv),
                     "request": {"state": "s", "questions": {"answer": {"type": "score", "instructions": "i",
                                                                         "criteria": ["lo", "mid", "hi"]}}},
                     "gold": {"answer": {"type": "score", "label": lv, "value": lv + 0.1, "dist": [0.8 if j == lv else 0.1 for j in range(3)]}}})
    ans = {r["id"]: {"ok": True, "answers": {"answer": {"type": "score", "probabilities":
                                                        {str(j): (0.8 if j == r["gold"]["answer"]["label"] else 0.1) for j in range(3)}}}}
           for r in rows}
    p = M.collect(rows, ans, 2)
    h = p.heads["answer"]
    m = M.head_metrics(h)
    assert m["acc_mode"] == 1.0 and m["spearman"] > 0.9 and m["kl"] == pytest.approx(0.0, abs=1e-9)
    assert not math.isnan(m["qwk"])
