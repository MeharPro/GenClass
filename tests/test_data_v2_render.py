"""Rendering rules shared by every v2 converter (pure Python, no torch)."""

from jev_local.data.v2.render import (NOTA_KEY, LabelSpec, label_verification, make_example, negation_pair,
                                      render_choice, render_noul, render_score, stable_rng)

LABELS = [LabelSpec(f"label_{i}", (f"description of label {i}",)) for i in range(40)] + [LabelSpec("yes", ("affirmative",))]


def test_choice_gold_present_or_nota_and_keys_safe():
    for i in range(500):
        rng = stable_rng("t", i)
        r = render_choice("Which label?", LABELS, "label_3", rng)
        crit, lab = r.question["criteria"], r.label
        assert 2 <= len(crit) <= len(LABELS) + 1
        assert all(k.lower() not in {"true", "false", "yes", "no"} for k in crit)
        if lab["label"] == NOTA_KEY:
            assert "label_3" not in r.key_map.values()
        else:
            assert r.key_map[lab["label"]] == "label_3" and lab["label"] in crit


def test_choice_soft_dist_normalized_and_mapped():
    r = render_choice("Which?", LABELS, {"label_1": 0.6, "label_2": 0.4}, stable_rng("s"))
    assert abs(sum(r.label["dist"].values()) - 1) < 1e-9
    assert {r.key_map[k] for k in r.label["dist"]} == {"label_1", "label_2"}


def test_k_distribution_has_small_and_full():
    ks = [len(render_choice("q", LABELS, "label_0", stable_rng("k", i), nota_prob=0).question["criteria"]) for i in range(400)]
    assert min(ks) <= 3 and max(ks) >= len(LABELS)


def test_score_noul_pairs_and_example():
    s = render_score("How good?", ["bad", "ok", "good"], 2, stable_rng("x"))
    assert s.question["type"] == "score" and s.label == {"type": "score", "level": 2}
    c = render_score("How good?", ["bad", "ok", "good"], [0.1, 0.2, 0.7], stable_rng("x"), as_choice_prob=1.0, row_key="r")
    assert c.question["type"] == "choice" and abs(sum(c.label["dist"].values()) - 1) < 1e-9
    a, b = negation_pair("Is it raining?", "Is it dry?", 0.8)
    assert a.label["p"] + b.label["p"] == 1.0
    pos, neg = label_verification("Is this about {label}? ({description})", LABELS[0], LABELS[1], stable_rng("v"))
    assert pos.label["p"] == 1.0 and neg.label["p"] == 0.0
    ex = make_example("x1", {"text": "hi"}, {"q1": render_noul("Is it a greeting?", 1.0)}, source="unit", bucket="b0",
                      license_use="commercial")
    assert ex["labels"]["q1"]["p"] == 1.0 and ex["questions"]["q1"]["type"] == "noul" and ex["bucket"] == "b0"
