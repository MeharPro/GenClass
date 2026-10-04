"""Specs owned by bm-adapters-a (PLAN §6 rows): Deußer exact, Decision Index 0.2.1, typed-decisions, jev-bench.

Spec ids are exactly the `spec_id` strings of bench/public/targets.json.
"""

from __future__ import annotations

from jev_local.bench.benchmax import Spec, register

DEUSSER = register(Spec(
    id="deusser_exact@6bbdeb33",
    suite="Deußer, Sparrenberg & Sifa 2026 (arXiv 2609.37647)",
    adapter="jev_local.bench.benchmax.adapters_a.deusser:DeusserAdapter",
    harness={"repo": "https://github.com/AppliedMachineLearning-Lab/jev-benchmarking", "commit": "6bbdeb33474849b6de2f0cccc9f5e19756abd67e",
             "licence": "MIT", "route": "import their task classes; engine.system_one -> JSONL -> scripts/import_responses.py + scripts/evaluate.py",
             "ci": "bootstrap 500, seed 0", "jev_raw_outputs": "do-not-use (Jev Responses License §3.2)"},
    verification="n per task equals Jev's n (targets.json); 100% coverage (truncate mode, disclosed) or a missing=wrong variant",
    counted_rows="37 (+6 S-bar/secondary)",
    description="One request per example with every question of the task; letter keys for MC, label names (null) for classification.",
    owner="a",
))

DECISION_INDEX = register(Spec(
    id="decision_index_0.2.1@87d4650b",
    suite="Decision Index 0.2.1 (apolinario/decision-index kit; board multimodalart/jev-decision-index)",
    adapter="jev_local.bench.benchmax.adapters_a.decision_index:DecisionIndexAdapter",
    harness={"repo": "https://github.com/apolinario/decision-index", "commit": "87d4650b42b377c0291a89c1f1a879f9b31082bf", "licence": "MIT (kit)",
             "edition": "0.2.1", "route": "suite rebuild -> pipeline --engine http -> score --edition 0.2.1",
             "rows_sha256": "b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5",
             "added_sha256": "7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89",
             "exclusions_sha256": "331df32d4b719c7db43214d0e5d85859d39c3b2eb7d0b3812214cce150155e81",
             "jev_index": {"fixture": 57.89, "space": 57.91}},
    verification="rebuilt sha256 b2b56d6f… / 7429f3c9…; overflow = refuse (unsupported counts as wrong); unrounded probabilities",
    counted_rows="41 (+10)",
    description="The kit sends exactly state and questions; no truncation, no option filtering, one fixed rendering, global calibration only.",
    owner="a",
))

TYPED_DECISIONS = register(Spec(
    id="typed_decisions_card@d0e2f0c4",
    suite="LocalLLaMA/typed-decisions (HF card leaderboard)",
    adapter="jev_local.bench.benchmax.adapters_a.typed_decisions:TypedDecisionsAdapter",
    harness={"hf_id": "LocalLLaMA/typed-decisions", "revision": "d0e2f0c42fef86cc15d1688d25a19f5ba7c85b18", "licence": "Apache-2.0",
             "route": "whole case (state + 5 questions) per request; scorer reconstructed and pinned by the Prior/Uniform rows"},
    verification="Prior .470/.347/.189/.088 and Uniform .308/.444/.238/.169 reproduced to 3 dp before scoring us",
    counted_rows="1 (+4)",
    description="Accuracy (argmax vs gold), KL(gold||model), Brier, top-label ECE over 400 test cases / 2,000 decisions.",
    owner="a",
))

JEVBENCH_HF = register(Spec(
    id="jevbench_hf_praveenrajus_v0.1.1",
    suite="Praveenrajus/jev-bench v0.1.1 (HF)",
    adapter="jev_local.bench.benchmax.adapters_a.jevbench_hf:JevbenchHfAdapter",
    harness={"hf_id": "Praveenrajus/jev-bench", "revision": "c37b0f6ab1687376ec9ae8dbd9f343163a68252b", "manifest_version": "0.1.1",
             "licence": "other (mixed; per manifest)", "scorer": "uspraveen/Jevify@a1308666b5460291772161260f43ed95afa65815 jevify/bench/metrics.py (reimplemented)",
             "route": "one request per test record: {state, model, questions: {q: question}}"},
    verification="n per config (1,000 / 2,000 / 1,599 / 800 / 687); manifest version 0.1.1",
    counted_rows="13 (+10 ctx)",
    description="Accuracy per config (argmax / p>=0.5), macro over 22 configs; ECE-15, Brier, NLL, TVD to human labels.",
    owner="a",
))

__all__ = ["DEUSSER", "DECISION_INDEX", "TYPED_DECISIONS", "JEVBENCH_HF"]
