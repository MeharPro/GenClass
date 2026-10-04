# jevbench v1 — release read m0

Generated 2026-10-02T01:22:27Z from `scores.json`; paired item-clustered, target-stratified bootstrap, B = 2000, seed 20261001. Systems: jev, jev-local-fast. Pre-registration: `bench/jevbench/PREREG.md`. PREREG.md sha256 `a35004c6bb383fb5527951f99c465299f569a98c03a658b5067462370b6fe457`. Changed since registration (hash differs): metrics.py, scripts/jevbench.py — see the release notes for why; a release read must state that no registered primary definition changed.

> **jev is incomplete** on 22 counted dataset(s) (< 99% valid responses): fin_topic, sst2, fin_phrasebank, sst5, yelp5, dair_emotion, tweeteval_emotion, go_emotions, banking77, clinc150, massive, boolq, rte, anli, paws, stsb, climate_fever, toxic_chat, openai_moderation, typed_decisions, unfair_tos, helpsteer2. It is left out of the indices and the decision rule, and its cells are blank (with the valid share) wherever it is below 99% valid, until its pass completes. Failure statuses: 402 × 20818.

## Indices (test, counted datasets)

| system | skill index | 95% CI | Decision-Score index | 95% CI |
|---|---|---|---|---|
| jev-local-fast | 10.1 | [9.7, 10.7] | -27.9 | [-28.6, -27.1] |

## Areas (skill × 100 / Decision Score × 100)

| area | jev-local-fast skill | jev-local-fast DS |
|---|---|---|
| A Topic | 9.6 | -4.9 |
| B Sentiment | 3.9 | -40.0 |
| C Emotion | 15.0 | -7.8 |
| D Intent | 31.8 | 16.2 |
| E Inference | 2.7 | -43.4 |
| F Fact & safety | 5.1 | -31.1 |
| G Multi-question | 2.6 | -84.2 |

## Per dataset (test)

| # | dataset | metric | chance | jev | jev-local-fast | Δ jev-local-fast−jev [95% CI] | DS jev / jev-local-fast |
|---|---|---|---|---|---|---|---|
| 1 | AG News | acc | 0.250 | 0.882 | 0.315 | -0.567 [-0.585, -0.549] | 0.74 / -0.12 |
| 2 | Yahoo Answers Topics | acc | 0.100 | 0.743 | 0.211 | -0.532 [-0.555, -0.509] | 0.56 / 0.02 |
| 3 | Twitter Financial News Topic | acc | 0.050 | n/a (7% valid) | 0.124 | – | – / -0.05 |
| 4 | SST-2 | acc | 0.500 | n/a (2% valid) | 0.491 | – | – / -0.96 |
| 5 | Financial PhraseBank | acc | 0.333 | n/a (2% valid) | 0.322 | – | – / -0.62 |
| 6 | SST-5 | acc_mode | 0.200 | n/a (0% valid) | 0.274 | – | – / 0.01 |
| 7 | Yelp Review Full | acc_mode | 0.200 | n/a (0% valid) | 0.250 | – | – / -0.02 |
| 8 | DAIR Emotion | acc | 0.167 | n/a (0% valid) | 0.338 | – | – / -0.05 |
| 9 | TweetEval emotion | acc | 0.250 | n/a (0% valid) | 0.426 | – | – / 0.02 |
| 10 | GoEmotions | macro_f1 | 0.075 | n/a (0% valid) | 0.086 | – | – / -0.20 |
| 11 | Banking77 | acc | 0.013 | n/a (0% valid) | 0.411 | – | – / 0.23 |
| 12 | CLINC150 plus OOS | acc | 0.007 | n/a (0% valid) | 0.343 | – | – / 0.17 |
| 13 | MASSIVE en-US intent | acc | 0.017 | n/a (0% valid) | 0.225 | – | – / 0.08 |
| 14 | BoolQ | acc | 0.500 | n/a (0% valid) | 0.384 | – | – / -1.00 |
| 15 | RTE | acc | 0.500 | n/a (1% valid) | 0.527 | – | – / -0.35 |
| 16 | ANLI r1-r3 | acc | 0.333 | n/a (0% valid) | 0.321 | – | – / -0.27 |
| 17 | PAWS | acc | 0.500 | n/a (0% valid) | 0.465 | – | – / -0.53 |
| 18 | STS-B | spearman | 0.000 | n/a (0% valid) | 0.079 | – | – / -0.02 |
| | llm_aggrefact | skipped: not built | |  |  |
| 20 | Climate-FEVER | acc | 0.250 | n/a (0% valid) | 0.309 | – | – / -0.11 |
| 21 | ToxicChat | f1_toxic | 0.133 | n/a (0% valid) | 0.126 | – | – / -0.14 |
| 22 | OpenAI moderation eval | mean_auprc | 0.095 | n/a (0% valid) | 0.161 | – | – / -0.68 |
| 23 | typed-decisions | acc | 0.318 | n/a (1% valid) | 0.321 | – | – / -1.00 |
| 24 | UNFAIR-ToS | micro_f1 | 0.029 | n/a (0% valid) | 0.028 | – | – / -1.00 |
| 25 | HelpSteer2 | mean_spearman | 0.000 | n/a (0% valid) | 0.073 | – | – / -0.52 |

## Shown, not counted (knowledge reference)

| # | dataset | metric | chance | jev | jev-local-fast | Δ jev-local-fast−jev [95% CI] | DS jev / jev-local-fast |
|---|---|---|---|---|---|---|---|
|  | HellaSwag | acc | 0.250 | n/a (0% valid) | 0.265 | – | – / -0.21 |
|  | WinoGrande | acc | 0.500 | n/a (0% valid) | 0.506 | – | – / -0.51 |
|  | MMLU-Pro sample | acc | 0.110 | n/a (0% valid) | 0.098 | – | – / -0.14 |

## jevbench-dev (ours only)

| # | dataset | metric | chance | jev | jev-local-fast | Δ jev-local-fast−jev [95% CI] | DS jev / jev-local-fast |
|---|---|---|---|---|---|---|---|
|  | 20 Newsgroups | acc | 0.050 | – | 0.070 | – / -0.05 |
|  | tweet_topic_single | acc | 0.167 | – | 0.076 | – / -0.54 |
|  | TweetEval sentiment | acc | 0.333 | – | 0.442 | – / -0.08 |
|  | app_reviews stars | acc_mode | 0.200 | – | 0.069 | – / -0.43 |
|  | HWU64 | acc | 0.016 | – | 0.295 | – / 0.15 |
|  | SNIPS | acc | 0.143 | – | 0.537 | – / 0.29 |
|  | MRPC | acc | 0.500 | – | 0.493 | – / -0.55 |
|  | SciTail | acc | 0.500 | – | 0.420 | – / -0.86 |
|  | CommitmentBank | acc | 0.333 | – | 0.286 | – / -0.47 |
|  | ToxiGen annotated | acc | 0.500 | – | 0.567 | – / -0.33 |
|  | TweetEval offensive | acc | 0.500 | – | 0.717 | – / -0.35 |
|  | deepset prompt-injections | acc | 0.500 | – | 0.517 | – / -0.83 |
|  | held-out tasksource-jev sources | acc | – | – | 0.374 | – / – |

## Calibration and robustness (test, main variant, pooled over heads)

| dataset | jev ECE15 / NLL / Brier | jev-local-fast ECE15 / NLL / Brier | jev flip / TV | jev-local-fast flip / TV | jev bare Δ | jev-local-fast bare Δ |
|---|---|---|---|---|---|---|
| AG News | 0.081 / 0.42 / 0.194 | 0.280 / 1.53 / 0.838 | – | 0.000 / 0.000 | – | 0.024 |
| Yahoo Answers Topics | 0.139 / 0.87 / 0.395 | 0.050 / 2.24 / 0.878 | – | 0.000 / 0.000 | – | 0.065 |
| Twitter Financial News Topic | – | 0.083 / 3.14 / 0.953 | – | 0.000 / 0.000 | – | 0.003 |
| SST-2 | – | 0.493 / 2.32 / 0.981 | – | – | – | – |
| Financial PhraseBank | – | 0.324 / 1.70 / 0.909 | – | 0.000 / 0.000 | – | 0.026 |
| SST-5 | – | 0.101 / 1.54 / 0.779 | – | – | – | – |
| Yelp Review Full | – | 0.083 / 1.68 / 0.819 | – | – | – | – |
| DAIR Emotion | – | 0.048 / 1.70 / 0.793 | – | 0.000 / 0.000 | – | -0.138 |
| TweetEval emotion | – | 0.068 / 1.28 / 0.691 | – | 0.000 / 0.000 | – | -0.163 |
| GoEmotions | – | 0.041 / 0.22 / 0.088 | – | – | – | – |
| Banking77 | – | 0.040 / 2.26 / 0.757 | – | 0.000 / 0.000 | – | -0.007 |
| CLINC150 plus OOS | – | 0.051 / 3.02 / 0.798 | – | 0.000 / 0.000 | – | -0.045 |
| MASSIVE en-US intent | – | 0.082 / 3.04 / 0.889 | – | 0.000 / 0.000 | – | 0.049 |
| BoolQ | – | 0.573 / 2.50 / 1.142 | – | – | – | – |
| RTE | – | 0.279 / 1.00 / 0.671 | – | – | – | – |
| ANLI r1-r3 | – | 0.297 / 1.41 / 0.850 | – | 0.000 / 0.000 | – | – |
| PAWS | – | 0.325 / 1.09 / 0.753 | – | – | – | – |
| STS-B | – | 0.058 / 1.82 / 0.842 | – | – | – | – |
| Climate-FEVER | – | 0.092 / 1.43 / 0.764 | – | 0.000 / 0.000 | – | – |
| ToxicChat | – | 0.038 / 0.22 / 0.096 | – | – | – | – |
| OpenAI moderation eval | – | 0.068 / 0.41 / 0.237 | – | – | – | – |
| typed-decisions | – | 0.335 / 1.61 / 0.408 | – | – | – | – |
| UNFAIR-ToS | – | 0.408 / 1.33 / 0.857 | – | – | – | – |
| HelpSteer2 | – | 0.215 / 1.93 / 0.908 | – | – | – | – |

## Coverage and latency

| system | records | ok | resolved model ids | input tokens | cost $ | truncated | multi-pass |
|---|---|---|---|---|---|---|---|
| jev | 25238 | 4420 | typesafe/jev-1.13-20260917 (4420) | 2,657,719 | 0.11 | 0 | 0 |
| jev-local-fast | 136190 | 136190 | jev-local-fast-0.1.0@jev-local-fast (136190) | 69,493,851 | 0.00 | 49 | 11040 |

| dataset | jev p50 / p95 ms | jev-local-fast p50 / p95 ms | jev failed req | jev-local-fast failed req | SST-2 choice / other notes |
|---|---|---|---|---|---|
| AG News | 162 / 253 | 56 / 67 | 0 | 0 |  |
| Yahoo Answers Topics | 163 / 315 | 84 / 217 | 0 | 0 |  |
| Twitter Financial News Topic | – | 70 / 80 | 3818 | 0 |  |
| SST-2 | – | 51 / 61 | 857 | 0 | choice-variant acc jev-local-fast 0.735 |
| Financial PhraseBank | – | 55 / 65 | 950 | 0 |  |
| SST-5 | – | 52 / 61 | 2208 | 0 |  |
| Yelp Review Full | – | 68 / 225 | 1998 | 0 |  |
| DAIR Emotion | – | 50 / 61 | 1998 | 0 |  |
| TweetEval emotion | – | 52 / 62 | 1419 | 0 |  |
| GoEmotions | – | 129 / 163 | 1998 | 0 |  |
| Banking77 | – | 359 / 386 | 3074 | 0 |  |
| CLINC150 plus OOS | – | 438 / 470 | 5498 | 0 |  |
| MASSIVE en-US intent | – | 287 / 393 | 2972 | 0 |  |
| BoolQ | – | 77 / 133 | 3268 | 0 |  |
| RTE | – | 70 / 112 | 275 | 0 |  |
| ANLI r1-r3 | – | 75 / 108 | 3198 | 0 |  |
| PAWS | – | 72 / 104 | 1998 | 0 |  |
| STS-B | – | 66 / 94 | 1377 | 0 |  |
| Climate-FEVER | – | 113 / 164 | 1533 | 0 |  |
| ToxicChat | – | 66 / 116 | 5081 | 0 |  |
| OpenAI moderation eval | – | 155 / 361 | 1678 | 0 |  |
| typed-decisions | – | 134 / 246 | 398 | 0 |  |
| UNFAIR-ToS | – | 98 / 150 | 1605 | 0 |  |
| HelpSteer2 | – | 241 / 870 | 1036 | 0 |  |

## Release notes

- **Jev pass blocked by billing.** OpenRouter returned HTTP 402 'Insufficient credits' after 4,420 valid Jev responses ($0.11, resolved id typesafe/jev-1.13-20260917). GET /api/v1/credits shows the key is a free-tier account with $0 total credits (lifetime usage $0.217). Adding about $5 of credit is a user action. Jev is complete only on AG News and Yahoo (main variant), so the indices and the decision rule cover our model alone and no claim is evaluated.
- **To finish M0 after credits are added** (Mac): `.venv/bin/python scripts/jevbench.py run-jev --release m0 --roles test,ref` (the cache skips the 4,420 done ids and retries the 402 ones; ~115k requests, about $4.5, ~60-70 min at ~1,900 req/min), copy `runs/jevbench/m0/jev.jsonl` to VM train `~/jev/runs/jevbench/m0/`, then `scripts/jevbench.py score --release m0 --ours ours__jev-local-fast.jsonl --jev jev.jsonl --roles test,ref,dev` and `report --release m0` there. Our model's numbers will not change (same seed and data).
- **Harness sanity check against published Jev numbers:** AG News 0.882 on our 2,000-item stratified sample vs Deußer 0.885 on all 7,600; Jev ECE15 0.081 vs Deußer's 0.077. Within the ±3-point M0 criterion.
- **Code changed after registration, with no change to any registered number:** `metrics.collect` aligns shuffled and bare variants to the main variant's label order (it crashed before); `primary_for` pools MMLU-Pro heads by option count and the dev soft_acc row. `scripts/jevbench.py` gains a 401/402/403 abort, a ≥99%-coverage gate, error counts and report fixes. Re-scoring with the final code reproduced every main-variant number and CI of the first scoring run exactly.
- **jev-local-fast v1 (32M; trained on computer-use and GEN data only)** sits near chance on general classification: skill index 10.1 [9.7, 10.7]. Its Decision-Score index is −27.9: probabilities are worse than the class prior because its temperatures were fitted for harness questions. Noul polarity is broken on general yes/no: SST-2 noul 0.491 vs 0.735 for the same items as a 2-option choice; BoolQ 0.384, below chance. It is exactly order-invariant (0 flips on all 11 choice datasets; total variation 0.000). Described labels do not consistently help it: bare names are −0.16 on TweetEval emotion and −0.14 on DAIR, +0.065 on Yahoo.
- LLM-AggreFact (#19) was not built because it is gated and VM train has no HF credentials. The index runs over 24 datasets, as pre-registered.
