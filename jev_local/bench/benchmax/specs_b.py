"""Group-B spec registry (bm-adapters-b): the suites of PLAN §6 owned by this agent, registered into the shared
`jev_local.bench.benchmax.SPECS` through `register()` so adapters-a's runner (`load_specs()`) sees them.

Contract every group-B adapter honours (see `adapters_b/__init__.py`): `Class(spec, args)` where `args.work` is the
work directory (default `$BENCHMAX_WORK` or ~/bench_work_b); `prepare(split)`, `items(split, limit, tasks)` ->
list[Item] with `item.request` = the wire body minus `model`, `score(items, answers, split, thresholds)`,
`fit(items, answers)` on validation items only, `expected_counts(split)`. Multi-stage suites (zhuyansen banking77)
expose `STAGES` and `stage_items(stage, items, answers)`. `EVAL_SPLIT` is the split the published Jev number was
measured on; validation items are for self-checks and thresholds only.

Spec ids are the `spec_id` strings of bench/public/targets.json byte for byte, including the truncated
`study:jev_for_network_traffic_classification_arxiv_261`.
"""

from __future__ import annotations

from jev_local.bench.benchmax import SPECS, Spec, register
from jev_local.bench.benchmax import adapter_class as _adapter_class

_B = "jev_local.bench.benchmax.adapters_b."

SPECS_B: tuple[Spec, ...] = (
    Spec(id="chepyle_lexglue_systemone_v1", suite="chepyle/lexglue-systemone (LexGLUE 7 tasks + BANKING77 + CLINC150 OOS+)",
         adapter=_B + "chepyle_lexglue:Adapter",
         harness={"repo": "chepyle/lexglue-systemone", "commit": "eeb55e2c79dd9a41ec19e8f5877f6a8c65532712", "licence": "Apache-2.0",
                  "data": {"coastalcph/lex_glue": "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c", "clinc/clinc_oos": "155b9c710419136e17307b80d0a13e68cd46b4ec",
                           "PolyAI banking77": "csv sha256 in vendor/chepyle/intents.json"}},
         verification="per-task test counts (1000/1000/1400/5000/10000/1607/3600; 3080; 5500) and LexGLUE revision pin",
         counted_rows="T269-T280 (T277 S-only: thresholds fit on validation)", owner="b",
         description="Verbatim instructions and criteria from the repo; multilabel nouls thresholded at 0.5 (per-label tuned variant on validation); micro-F1 with the none column."),
    Spec(id="mbburabak_safety", suite="mbburabak/jev-safety-bench (HateCheck, ToxicChat, Aegis v1/v2, WildGuardTest, HarmBench)",
         adapter=_B + "mbburabak_safety:Adapter",
         harness={"repo": "mbburabak/jev-safety-bench", "commit": "1bf1eacfc37e01cb0dd80a45d838f00c36cc1f1d", "licence": "MIT",
                  "policies": "vendor/mbburabak/policy (question sets + policy markdown)"},
         verification="case counts 3728/2853/359/1928/1928/1725/239+602 and manifest sha256 of every source file",
         counted_rows="T297-T304 (T297, T303, T304 Z only)", owner="b",
         description="Policy text in state, boolean question sets as nouls, verdict-forming question thresholded at 0.5; F1 on the harmful class with Wilson CIs."),
    Spec(id="dmb_expanded@eabd88b0", suite="DMB expanded (nibzard/decision-model-benchmark): BANKING77, CLINC150+OOS, NLU++",
         adapter=_B + "dmb_expanded:Adapter",
         harness={"repo": "nibzard/decision-model-benchmark", "commit": "eabd88b04706bcc9b7769213d1ea644c09d8a7f8", "licence": "none (instructions read at runtime from the pinned checkout)",
                  "raw_archives": "never downloaded (PLAN §3.2)", "sources": "vendor/dmb/expanded_sources.json (PolyAI / clinc / NLU++ sha256)"},
         verification="test counts 3080 / 5500 (1000 OOS) / 13712 decisions over 302 messages; render.py sha256 pin",
         counted_rows="T109-T112", owner="b",
         description="Reimplemented from the documented protocol: one choice per utterance (CLINC with the OOS option), NLU++ yes/no per intent; thresholds (max error 5%, min 100 accepted) frozen on validation."),
    Spec(id="rerank_scripts", suite="rerank scripts: denser-org (BEIR SciFact/NFCorpus), hev (SciFact/NFCorpus/FiQA), anessbelbati (8 sets + NevIR)",
         adapter=_B + "rerank_scripts:Adapter",
         harness={"denser": "41fe2570d3d0391c65895431dc2c3e40f329054c", "hev": "1eb47266270b32b2a3667f9fb89646378ca9c9d6", "anessbelbati": "fecba75a443c580a8979ce2587e2d452d4d8e511",
                  "candidates": "denser + anessbelbati committed lists (sha256 pinned); hev lists rebuilt with BM25 (indicative)"},
         verification="query counts 300/323/300/1383 and candidate-file sha256",
         counted_rows="T387-T393 (T389-T391 indicative: hev candidate lists unpublished)", owner="b",
         description="Per-passage noul relevance (denser object instructions; hev doc-id nouls; anessbelbati 4-level score), re-ranked by probability; nDCG@10 and NevIR paired accuracy."),
    Spec(id="study:earino_zero_shot_complaint_benchmark_cfpb_113_cl", suite="earino/zero-shot-complaint-benchmark (CFPB, 113 issues)",
         adapter=_B + "study_cfpb:Adapter",
         harness={"repo": "earino/zero-shot-complaint-benchmark", "commit": "33d5ed7467eff8831ccb47730dc14a535395544f", "licence": "MIT",
                  "data": {"determined-ai/consumer_complaints_medium": "4783de6e089f8ef1dccb26e9f46d491ec0273c83"}},
         verification="6,430 evaluation rows from the committed val_indices; 113 labels", counted_rows="T521 (bare), T522 (v1)", owner="b",
         description="One 113-way choice per complaint; bare labels or the form instruction plus structured definitions; accuracy, macro-F1, ECE15."),
    Spec(id="study:do_system_one_decisions_add_up_arxiv_2609_33971", suite="Do System One Decisions Add Up? (TREC-50 flat)",
         adapter=_B + "study_trec50:Adapter",
         harness={"paper": "arXiv 2609.33971", "data": "CogComp/trec (revision pinned at prepare)", "seed": 42},
         verification="500 test questions, 50 fine labels, 6 coarse", counted_rows="T458", owner="b",
         description="Flat 50-way choice with Lnnn identifiers and 'Parent: suffix' descriptions in a seeded order; accuracy and macro-F1 over all declared classes."),
    Spec(id="study:jev_for_network_traffic_classification_arxiv_261", suite="Jev for network traffic classification (CESNET-QUICEXT-25)",
         adapter=_B + "study_cesnet:Adapter",
         harness={"paper": "arXiv 2610.00376", "data": "Lystea/CESNET-QUICEXT25-PARQUET@4180d7d97e440e7fdf9e43693e052509074999a4", "seed": 2026092305,
                  "indicative": "hash, wording and selected IDs unpublished"},
         verification="2,000 records per week x 26 test weeks = 52,000; 40 examples from weeks 1-4", counted_rows="T490 (k=0), T491 (k=40, S only)", owner="b",
         description="Ten-packet triples as packet-major JSON, one 10-way choice, optional 40 fixed labelled examples; accuracy (indicative sample)."),
    Spec(id="study:koa_action_arxiv_2609_36115", suite="Koa-action (SST-2, Amazon Reviews Polarity)",
         adapter=_B + "study_amazon_polarity:Adapter",
         harness={"paper": "arXiv 2609.36115", "data": "stanfordnlp/sst2 validation; fancyzhx/amazon_polarity test[:5000] (revisions pinned at prepare)",
                  "indicative": "Jev request wording and the 5k sample unpublished"},
         verification="872 and 5,000 rows", counted_rows="T511, T512", owner="b",
         description="Binary Negative/Positive choice mirroring the paper's prompt; accuracy (indicative)."),
    Spec(id="study:jev_ids_arxiv_2610_01079", suite="Jev-IDS (NSL-KDD paper split, k=0)",
         adapter=_B + "study_nslkdd:Adapter",
         harness={"repo": "jev-ids/jev-ids", "commit": "6aa5ac4570db1d6d9b88f769f7d16c1cd2c3784a", "licence": "MIT", "paper": "arXiv 2610.01079",
                  "split_file": "data/nsl-kdd/splits/paper.csv sha256 b035cb33..."},
         verification="2,000 flows (874 normal / 1,126 attack, 300 novel); template sha256", counted_rows="T492", owner="b",
         description="The repo's request template with the flow's 41 comma-joined features; is_attack noul >= 0.5 (fail-open); attack-class F1, PR-AUC."),
    Spec(id="elcronos_plain", suite="elcronos/jev-vs-open-decision-models `plain` (emotion, TweetTopic, fin topic, DailyDialog)",
         adapter=_B + "elcronos:Adapter",
         harness={"repo": "elcronos/jev-vs-open-decision-models", "commit": "a1901bc3d520e73936de8d4326545c0cdcf742fb", "licence": "none (reimplemented from PROTOCOL.md)",
                  "data": {"dair-ai/emotion": "cab853a1dbdf4c42c2b3ef2173804746df8825fe", "cardiffnlp/tweet_topic_single": "87b7a0d1c402dbb481db649569c556d9aa27ac05",
                           "zeroshot/twitter-financial-news-topic": "acbc8af2a35ccf0916124efcbe9e6cf25f191012", "OpenRL/daily_dialog": "1668faf0c0dc44664f108c489fd0666128db2c48"}},
         verification="2000 / 1693 / 4117 / 7740 rows and ClassLabel name checks", counted_rows="T243-T247", owner="b",
         description="Raw text state, one choice with criteria {label: ''} (or one-line definitions); accuracy, macro-F1 over all classes, ECE15."),
    Spec(id="zhuyansen_batch20", suite="zhuyansen/jev-zeroshot-vs-bert (20 texts per call; AG News, SST-2, BANKING77 two-step, TweetEval emotion, PAWS, arXiv)",
         adapter=_B + "zhuyansen:Adapter",
         harness={"repo": "zhuyansen/jev-zeroshot-vs-bert", "commit": "edbf0713583644bd3f47299fd6b104a8b2073219", "licence": "MIT",
                  "labels": "vendor/zhuyansen/*.json (yaml.safe_load of labels/*.yaml)", "revisions": "resolved at prepare (unpinned upstream)",
                  "indicative": "arxiv2026 (author's crawl unpublished)"},
         verification="1000 / 872 / 1000 / 1000 / 1000 stratified rows (seed 0)", counted_rows="T248-T253 (T253 indicative)", owner="b",
         description="Batched 20-text state with t{i} choices and described criteria; banking77 group-then-label; accuracy."),
    Spec(id="thisisandreeeee", suite="thisisandreeeee/jev-benchmarks (BANKING77, CLINC150 in-scope, HWU64, SST-2 Noul, STS-B Score)",
         adapter=_B + "thisisandreeeee:Adapter",
         harness={"repo": "thisisandreeeee/jev-benchmarks", "commit": "eaed9dd0cfd6cfa085424e11f79cab8930847e16", "licence": "MIT",
                  "data": {"PolyAI/banking77": "1fb62b1bb4635df59a8e1b2f2bc5e0643b2856c8", "clinc/clinc_oos plus": "155b9c710419136e17307b80d0a13e68cd46b4ec",
                           "DeepPavlov/hwu64": "0dd289ccdeb185ec065d1ebcf5de1c443cd1620f", "nyu-mll/glue sst2": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
                           "mteb/stsbenchmark-sts": "96943a16ea6a35129e253c659081cb59daf81b30"}},
         verification="row digests (sha256 of the serialised rows) asserted for every evaluated split", counted_rows="T254-T258", owner="b",
         description="Choice over the canonical label set (criteria None), SST-2 noul at 0.5, STS-B 6-level score rubric; accuracy / Spearman."),
    Spec(id="asevlad_injection", suite="ASEVlad/jev-injection-bench (combined prompt-injection corpus)",
         adapter=_B + "asevlad_injection:Adapter",
         harness={"repo": "ASEVlad/jev-injection-bench", "commit": "c0d0f25d75f7f21908be968ae7cbe9da7d889287", "licence": "MIT",
                  "revisions": "resolved at prepare (unpinned upstream)"},
         verification="11,900 rows (3,464 attacks, 8,436 benign, 339 NotInject) after sha256 dedup", counted_rows="T308 (Z only)", owner="b",
         description="is_attack noul with the v2 wording and fixed criteria; AUPRC, ROC-AUC, catch/false-alarm/panic rates."),
    Spec(id="stperic_medhallu", suite="stperic/jev-medhallu-benchmark (MedHallu 1,000 MedHELM items)",
         adapter=_B + "stperic_medhallu:Adapter",
         harness={"repo": "stperic/jev-medhallu-benchmark", "commit": "8a7f2f88eb258fd0bc859ed4aeca3b80643e5d94", "licence": "MIT",
                  "items": "bench/datasets/medhallu/data/items.{test,dev,dev2}.jsonl (sha256 pinned)"},
         verification="1,000 test / 500 + 500 dev items by sha256", counted_rows="T319 (S only)", owner="b",
         description="One noul (authors_reject question) at the dev-chosen 0.65 threshold; accuracy with errors counted wrong."),
    Spec(id="goya_rm_eval", suite="goya/jev-rm-eval (RewardBench 1/2, RM-Bench, RubricBench, PPE, ProcessBench, PRMBench)",
         adapter=_B + "goya_rm:Adapter",
         harness={"repo": "goya/jev-rm-eval", "commit": "d594fd4fdce39d805120a510c3804b960560acae", "licence": "see checkout",
                  "data": {"RM-Bench": "THU-KEG/RM-Bench@73c52d7b total_dataset.json", "Qwen/ProcessBench": "3bdcd5371ed567559a78f559c01c13a6deee7604",
                           "hitsmy/PRMBench_Preview": "5cc7683d0ae5797f84d7aeac0607966f277c39e1", "others": "resolved at prepare (unpinned upstream)"}},
         verification="example counts 2985 / 1865 / 1327 / 1147 / 16038 / 3400 / 6216", counted_rows="T311-T318 (low priority)", owner="b",
         description="Ported request builders (pairwise with deterministic swap, listwise, ratings, RM grid, first-error, PRM steps, ordinal) and the repo's official aggregations."),
)

for _s in SPECS_B:
    register(_s)

SPEC_IDS: tuple[str, ...] = tuple(s.id for s in SPECS_B)


def spec(spec_id: str) -> Spec:
    """The group-B spec for `spec_id` (KeyError with the known ids otherwise)."""
    for s in SPECS_B:
        if s.id == spec_id:
            return s
    raise KeyError(f"unknown group-B spec {spec_id!r}; known: {', '.join(SPEC_IDS)}")


def adapter_class(spec_id: str):
    return _adapter_class(spec(spec_id))


def targets_of(spec_id: str) -> tuple[str, ...]:
    return tuple(adapter_class(spec_id).TARGETS)
