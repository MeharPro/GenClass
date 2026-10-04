# Every published Jev result on a public benchmark

- **Date:** 2026-10-02, completed 2026-10-03 (resume pass). **Scope:** every Jev number on a public dataset or public benchmark I could find and read: arXiv papers, leaderboards, HF cards, GitHub studies and blogs, Sep 16 – Oct 3, 2026.
- **Method:** read-only. I pulled small result files only (JSON, markdown, CSV under about 1 MB) and arXiv HTML pages with `gh api`, `curl` and web fetches. No dataset, model or raw-response download. No Jev, Azure or OpenRouter calls, so $0 spent.
- **Outputs:**
  - `bench/public/jev_published.json` holds **549 rows** (397 from the first pass, 152 added on 2026-10-03). Fields: `suite, dataset, hf_id, config, split, n, metric, jev_score, jev_model, date, source_url, raw_outputs_url, notes`.
  - This note: the summary, a per-dataset target table, conflicts, raw-output licences, and the full per-suite tables.
- **How to read `notes`:** every row starts `tier=…; protocol=…`.
  - **tier=A**: full official eval split, or a large frozen run (≥ about 1,000 items over an official set).
  - **tier=B**: documented sample of ≥ about 200 items, or a public board run by its maintainer.
  - **tier=C**: fewer than 200 items, partial coverage, or a non-standard protocol.
  - **tier=X**: not a usable bar. Covers LLM-consensus or synthetic gold, diagnostics, and numbers quoted second-hand.
  - **protocol=zero-shot**: Jev got no benchmark training data. This is the Z-track bar.
  - **protocol=uses-train-data**: Jev was given train examples, or thresholds/weights fitted on train/dev. This is the S-track bar.
  - Rows added in the resume pass end with `added=2026-10-03 resume pass`. Where a source's date field carried extra detail, it is kept as `date_detail=…` and `date` is now always ISO (`YYYY-MM-DD`, or `YYYY-MM` / `YYYY` when the source gives no day).
- **Scales:** proportions are on 0–1. Composite indices keep their native 0–100 scale (DI index, JevBench score, Jevals Decision Score). Error metrics (RMSE, WER, Brier, KL, false-negative rate) say "lower is better" in `metric`.
- **Verification level:**
  - Most rows were read from the source's own result file (JSON, CSV, README table, or the arXiv HTML table).
  - About 40 first-pass rows say "number via Yifan-Lan/awesome-jev-robustness … not re-verified at source". That is a CC0 community index of Jev studies, and I did not open those repos.
  - Blog rows (Lightfield, AY Automate, SOTAAZ) were checked against the post's own HTML table text.
- **Generator:** the first-pass scripts lived in an earlier session's scratchpad and are gone. The resume pass rebuilt §5 from the JSON, and added §0.1, §2b, §3.3 and new rows in §1, §3.1, §4 and §6. The first-pass §2 and §3.2 tables are unchanged and do not include the new rows; §2b and §3.3 cover them.

---

## 0. Bottom line

1. **Size of the corpus.**
   - 549 Jev result rows from 139 suites, papers or studies (397 rows / 85 suites in the first pass; the 2026-10-03 pass added 152 rows from 54 more, 32 of them arXiv papers).
   - Tiers: 202 A, 249 B, 68 C, 30 X. 16 rows are S-track (`uses-train-data`).
   - 213 rows point at publicly posted per-item Jev outputs.
   - Model: everything is `jev-1.13.0`. Some rows say `jev-latest` resolving to 1.13.0, or the OpenRouter snapshot `typesafe/jev-1.13-20260917`. No other Jev version has public numbers yet.
2. **Two sources give the bars to beat. They are the only large, frozen, full-split sources:**
   - **Deußer et al. (37 datasets, 346k requests).** I read the per-dataset `results/eval/*.json` files in the MIT repo.
   - **Decision Index 0.2.1.** 48 non-interactive Jev scores plus the index, from `data/index.json`.
   - Newer large runs add bars with no equivalent above:
     - DMB expanded (BANKING77, CLINC150, NLU++)
     - DecisionBench: Jev 0.720 on 23,900 rows; binary view 0.592; ordinal view 0.451
     - chepyle LexGLUE, all 7 tasks, full test
     - Praveenrajus/jev-bench: 22 configs, and it ships train splits
     - TMMLU+, GAOKAO, MedHallu, RewardBench and others, the safety suite, and the reranking suites
3. **🚩 Licence flag: Deußer's raw Jev responses must not be used by this project, not even for evaluation.**
   - The Zenodo `responses.db` is under "Jev Responses License 1.0".
   - §2 permits "comparing other models' outputs with the Data on the same requests".
   - **§3.2 forbids using the Data "to develop, or to facilitate the development of, a product or service that is similar to or competes with TypeSafe's models or services."** meharsjev is a reimplementation of Jev, so item-level use for selection, evaluation-driven development or paired tests falls under 3.2.
   - This contradicts `docs/research/v2/classifier-benchmarks.md` §2, which says the licence "gives item-level paired tests with no re-query".
   - Recommendation:
     - Cite only Deußer's published **aggregate** numbers (facts in a paper).
     - The MIT code and templates may still be used to rebuild the same requests for our own model.
     - Do not download `responses.db`.
   - The same caution applies to two other kinds of raw-output source:
     - Raw outputs with **no licence**, which means all rights reserved: DMB, elcronos, Alexander-Ollman, simonmesmith, switchboard, Janus, jev-phishing-bench. Read aggregates only.
     - TypeSafe's MCA §2.3(b), already in the leaderboard plan: Jev outputs never feed training, selection or calibration.
   - `autotrust/JEV-9B/27B` are distilled from 25,376 Jev-labelled rows (their card says so). Never use them or their outputs as a teacher. They are a contamination vector for any training data they touch.
4. **S-track bars exist, and they are much higher than the zero-shot numbers.** Jev given train data reaches:

   | Dataset | Jev with train data | Jev zero-shot (full test) |
   |---|---|---|
   | BANKING77 | **0.924** (24 BM25-retrieved train examples + definitions; simonmesmith) | 0.79–0.81 |
   | BANKING77 | 0.853 (2 examples per label; manojlds) | |
   | UNFAIR-ToS micro-F1 | **0.748** (per-label thresholds tuned on dev; Deußer) | 0.499 |
   | GoEmotions macro-F1 | 0.353 (tuned) | 0.243 |
   | ToxicChat F1 | 0.793 (tuned) | 0.786 |
   | LexGLUE 7-task mean micro-F1 | **0.742** (validation-tuned; chepyle) | 0.699 |
   | MedHallu | 0.929 (question and threshold chosen on 1,000 dev items) | |

   For the S track, beat the **"Jev with train data" number where one exists**, not only the zero-shot number. The target table (§2) lists both.
5. **Where Jev's public numbers are weakest.** These are the realistic wins for an encoder, and supervised encoders already beat Jev on most of them:
   - **Intent:**
     - BANKING77 0.80 zero-shot, against 0.94–0.95 for supervised SPACE-2 or fine-tuned BERT.
     - HWU64 0.831, against 0.942 supervised.
     - NLU++ micro-F1 0.483.
     - CLINC OOS recall 0.776 (Deußer) to 0.881 (chepyle).
   - **Multi-label and legal:**
     - GoEmotions macro-F1 0.243.
     - UNFAIR-ToS 0.499 (positives only), or 0.764 under the LexGLUE convention, against 0.952 for BERT.
     - EUR-LEX 0.391 against 0.716 for BERT.
     - LEDGAR 0.753 against 0.876 for BERT.
     - AGB-DE F1 0.204.
     - CUAD F1 0.519.
   - **Topic and emotion:**
     - Financial-tweet topic 0.670 (LR on TF-IDF gets 0.828).
     - TweetTopic 0.793 (LR on MiniLM gets 0.848).
     - DAIR emotion 0.585.
     - DailyDialog macro-F1 0.385.
     - CARER emotion macro-F1 0.484.
   - **Ordinal and judging:**
     - SST-5 0.579.
     - HelpSteer2 mean ρ 0.412, and helpfulness accuracy 0.36–0.41 (Jevals: "no model beats guessing").
     - DecisionBench ordinal 0.451.
     - Measuring-hate-speech 0.527.
   - **Safety:**
     - PhishNChips 0.626 (an untrained Gemma-4-12B gets 0.825).
     - ToxicChat F1 0.757–0.786.
     - DecisionBench guardrails family 0.391.
     - Implicit hate macro-F1 0.439.
     - iSarcasmEval F1 0.505.
   - **Distribution quality:**
     - typed-decisions KL 1.442. Prior gets 0.347 and uniform gets 0.444.
     - ChaosNLI: accuracy 0.615 and TVD 0.332; confidence is flat across agreement bands.
6. **Where a 68M encoder will not beat Jev, on either track.** Knowledge and reasoning sets, with Jev's score in brackets:

   | Benchmark | Jev |
   |---|---|
   | MMLU | 0.918 |
   | MMLU-Pro | 0.827 |
   | HellaSwag | 0.955 |
   | ARC | 0.988 |
   | WinoGrande | 0.914 |
   | BBH | 0.929 |
   | GPQA-D | 0.783 |
   | HLE | 0.204 |
   | C-Eval | 0.839 |
   | TMMLU+ | 0.776 |
   | GAOKAO | 0.909 |
   | JMedQA | 0.886 |
   | Belebele | 0.867 |
   | RewardBench | 0.926 |
   | TruthfulQA MC1 | 0.95 |

   - Also near ceiling: language ID 0.996, IMDB 0.965, SST-2 0.964.
   - Even with train splits (MMLU auxiliary_train, HellaSwag train), small encoders sit near chance on the knowledge sets (DI: every model at 0.5B or less scores ≤ 11.2 index).
   - "Better everywhere" is not reachable on these with a 68M model. Report them as "shown, not counted", or they need a much larger backbone. This is a scale limit, not a data limit.
7. **Corrections to earlier notes** (`docs/research/v2/classifier-benchmarks.md`):

   | Item | Earlier note | Correct |
   |---|---|---|
   | LLM-AggreFact 0.786 | pooled balanced accuracy | the **mean of 11 per-source balanced accuracies** (leaderboard convention); pooled is 0.831 |
   | DI iSarcasmEval | 4,600 cases | scored on **1,400** track-A English cases (4,600 requests) |
   | DI Jev index | 57.89 (kit tests) | 57.91 in the current `index.json` |
   | Deußer responses licence | item-level paired tests allowed | §3.2 blocks this project (point 3) |
   | UNFAIR-ToS 0.499 vs 0.764 | – | a protocol difference, not noise (§3) |
   | LangWatch typed-decisions | – | 0.739 on 1,965 decisions, not 2,000 |
   | WebJev typed-decisions | – | 0.7405 |
   | typed-decisions card | – | 0.727 |

### 0.1 Resume pass, 2026-10-03: what changed

1. **The first pass missed almost all of arXiv.** It cited two papers (Deußer; Ibrahim & Zaki). An arXiv API search (`all:Jev`, `all:TypeSafe`, `abs:"typed decision"`, `abs:"System One model"`) returns about 75 Jev papers posted 2026-09-19 … 10-01. I read the HTML tables of every one that reports Jev on a public dataset and added **90 rows from 32 papers**. Papers with no public-benchmark Jev number (systems, agents, surveys, Jev-style open models with no Jev row) are listed in §1.
2. **Other additions:** 19 GitHub/HF studies the robustness index or awesome lists point at but the first pass had not read (QuicqDev, cuad-jev-bench, jev-edge, CompleteDotTech, rag-jev, jev-rag-benchmark, jevr, JevForge, KoBBQ audit, vector-graph-rag, statsguysam, CFPB complaints, morrenhale, CanITrustYou-Jev, jev-ar-bench, WANLI-256, poorjev, jev-skill, trifleen) and three blogs with their own runs (Lightfield, AY Automate, SOTAAZ).
3. **Fixes to existing rows:** 121 non-ISO `date` values normalised (detail moved to `date_detail=`), 93 rows given the missing `protocol=` tag, 39 empty `split` fields filled with "sample (per-task; see source)", MedHallu re-tagged as S-track (question and threshold chosen on dev).
4. **New Z-track bars worth targeting** (tier A/B, zero-shot, Jev clearly beatable by a trained encoder):

   | Dataset (n) | Jev | Note |
   |---|---|---|
   | CFPB complaints, 113 issue labels (6,430) | acc 0.344, macro-F1 0.132 (0.465 / 0.166 with label definitions) | fine-tuned ModernBERT-base 0.611 / 0.264, ModernBERT-large 0.617 / 0.308; TF-IDF + LR 0.545 |
   | TREC fine, 50 labels (all 500) | acc 0.722, macro-F1 0.486 | TREC train 5,452 |
   | CyberSecEval CWE, 50-way (1,916) | top-1 0.458, macro-F1 0.269 | |
   | DiagnosisArena-MCQ (915) | 0.598 | knowledge-heavy |
   | MCPHunt agent traces (3,615) | positive F1 0.611 | |
   | Amazon QA answerability (500) | 0.628, below the 0.730 majority | |
   | Persuasion-for-Good strategy, 18-way (500) | 0.442 (supervised SOTA 0.795) | |
   | ELLIPSE essay traits, 5 levels (1,548 pairs) | exact 0.134 | |
   | CESNET-QUICEXT-25 traffic (52,000) | 0.098; 0.284 with 40 examples | RF 0.700 |
   | UCI tabular: Bank Marketing / Online Shoppers (1,000 each) | balanced acc 0.534 / 0.514 | |
   | WISDM / UCI HAR / PAMAP2 windows (600 each) | macro-F1 0.038 / 0.118 / 0.089 | |

5. **New S-track bars** (Jev given train data): BANKING77 + 8 train examples per intent **0.909** (770-item sample; with names only 0.821); CLINC150 in-scope + 8 examples **0.981** (1,500 sample); TREC coarse + 4 per class 0.855 (200); BANKING77 + 1 per class 0.819 balanced acc (1,500); DBLP-ACM + 6 demonstrations macro-F1 0.9859; SemEval-2026 DimABSA (Jev answers + 488 fitted coefficients) RMSE 1.0645, best of all participants; brain-to-text rescoring WER 0.0746.
6. **New strong spots for Jev** (a 68M encoder should not expect to win): MMLU full test 0.891 (third measurement, see §3.1), ContractNLI 0.774, RewardBench sample 0.925, HaluEval QA 0.873, TriviaQA-as-MC 0.955, RACE-H 0.95 (n=100), CRT 0.99, SST-2 0.961, Amazon Polarity 0.968, XQuAD-EN rerank nDCG@10 0.989.
7. **Raw outputs found in this pass** (eval-only): morrenhale (CC-BY-4.0, full probability vectors on AG News / DAIR / BANKING77 / MASSIVE / typed-decisions), earino CFPB (licence "other"), jev-edge (Apache-2.0), cuad-jev-bench cache (MIT), KoBBQ audit (MIT), vector-graph-rag (MIT), DimABSA code. See §4.

---

## 1. Sources covered (and what was not found)

| Source | What it is | Jev rows | Verified at |
|---|---|---:|---|
| Deußer, Sparrenberg & Sifa, arXiv 2609.37647 (v1 2026-09-29) | 37 datasets, full eval splits, frozen templates; plus memorization probes and threshold-tuned rows | 54 | `results/eval/*.json`, `thresholds.json`, `probes.json`, `docs/datasets.md` |
| Decision Index 0.2.1 (board generated 2026-09-28) | 38-benchmark panel plus shown-not-counted and dropped interactive; chance-corrected index | 55 | Space `data/index.json`; kit `docs/suite.md` for sources |
| DMB (nibzard), runs 09-18…09-30 | Expanded full-test BANKING77, CLINC150, NLU++; pilot; controls; v3 historical S1–S5 | 11 | `results/*/*.md` |
| Jevals release 2026-09-18 | BANKING77, PubMedQA, HelpSteer2; 300 items × 5 repeats | 6 | `releases/2026-09-18/board.json` |
| typed-decisions card (edited 2026-10-01) | 400 cases, 2,000 decisions | 5 | HF card |
| TypeSafe workflow evals | 4 synthetic workflows, LLM-consensus gold | 5 | evals.typesafe.ai (no data download) |
| JevBench v1.5.5 | Composite plus axes; 1,624 rows (601 public) | 3 | `benchmarkheaven.com/api/jevbench/v1.5.5` |
| DecisionBench (Hanno Labs) | 23,900 rows, 43 tasks | 5 | `decision-bench-results/results/typesafe__jev-1.13/…/DecisionBench.json` |
| DecideBench v1.1 | 400 contrastive decisions | 1 | README |
| LangWatch Jev benchmark (2026-09-23.1) | 15 tasks over public HF sets | 15 | web page |
| jev-bench (Praveenrajus) | 22 configs with train/val/test, raw predictions | 23 | HF card plus `results/jev-1.13.0` |
| jev-decision-bench (OmarMujahid) | 49 tasks; 39 rows on public sets kept | 39 | `report.json` |
| Ibrahim & Zaki, arXiv 2609.24574 | 18 CSS annotation tasks | 18 | `analysis/cell_metrics.csv` |
| AbdelStark, elcronos, zhuyansen, thisisandreeeee, manojlds, simonmesmith, Alexander-Ollman, chepyle (LexGLUE + intents + Kev suites), Kev README, WebJev, mbburabak (safety), Gaurav-Gosain, switchboard, ASEVlad, Red Hat (2026-10-02), goya4140 (reward models), MedHallu, TMMLU+, GAOKAO, JMedQA, Cohen-2006 screening, 4esv, heswy, dhruvmehra, onlyoneaman, mugenkyou, open-system-one, bakeoff, scienthoon, ChaosNLI, jev-phishing-bench, frontier-bench, rerankers (denser, hev, anessbelbati), gazelle93, Feishu (Laya repo), and about 25 smaller studies | | 157 | READMEs / result files; robustness index for the rest |

**Asked for but no Jev number exists:**
- **OpenRouter Jev Lab** (`openrouter.ai/labs/jev`) has latency and throughput only ("475 answers in 1.2 s" and the like). It has no accuracy claims.
- **Laya README / BENCHMARKS.md "Laya vs Jev"** contains no Jev measurements of its own. Every Jev cell is quoted from third parties: typed-decisions 0.727, AG News 0.910, DAIR 0.480, BANKING77 0.870 (BTZSC 72 labels), "ECE 0.246" (DMB S5 synthetic), and 236–276 ms latency. The only new values are typed-decisions soft-accuracy 0.580 and score MAE 0.391, attributed to the "published" Jev run without a source file. They are in the typed-decisions row notes.
- **The Feishu benchmark** in Laya's repo (`research/benchmarks/feishu_zh`, mirrored from Adkid-Zephyr) is a 64-case synthetic diagnostic. Jev scored 64/64 single Choice and 63/64 with four Nouls; Laya-multilingual scored 20/64. Raw Jev responses are MIT. It is tier X.
- **BTZSC official board** has no Jev row.
- **TypeSafe evals** have no downloadable items. The case counts (240/111/150/204) come from press summaries.
- **No published Jev number** on these of our jevbench-test sets: **#2 Yahoo Answers Topics, #15 RTE, #20 Climate-FEVER.** The match is weak for #13 MASSIVE: only jev-bench's 1,000-item en config (0.808) and laya-ft's MASSIVE en-US macro-F1 (0.799), not our `mteb/amazon_massive_intent` protocol.
- **Not found:** Jev numbers on full GLUE/SuperGLUE, XNLI full, DBpedia full, Yahoo, 20 Newsgroups (only yodablocks pass/fail), SNIPS, ATIS, MultiNLI full, Jigsaw, HateXplain, ETHOS, the MTEB classification suite, or BEIR beyond SciFact, NFCorpus, FiQA, NQ, TREC-COVID and BRIGHT reranking.

**Added 2026-10-03 (resume pass)**

| Source | What it is | Jev rows | Verified at |
|---|---|---:|---|
| QuicqDev/Jev-vs-ML (protocol V3.0.1) | AG News; BANKING77; Breast Cancer Wisconsin diagnostic; IMDb; … | 9 | README / result files |
| maybern-tripp-smith/cuad-jev-bench | CUAD clause retrieval; CUAD hard pairs | 2 | README / result files |
| kiwi0719/jev-edge | deepset prompt-injections | 2 | README / result files |
| CompleteDotTech/paper-package (run-20260918) | DBLP-ACM entity matching; SciFact claim-evidence relation | 3 | README / result files |
| EmreKaplaner/rag-jev | BEIR SciFact rerank | 1 | README / result files |
| emretheus/jev-rag-benchmark | BEIR SciFact rerank; XQuAD-EN passage rerank | 2 | README / result files |
| romeromarcelo/jev-retrieval (jevr) | BEIR NFCorpus; BEIR SciFact; HAKARI-Bench NanoRTEB | 3 | README / result files |
| zwliJay/jev-forge (JevForge) | JevForge-Mind2Web action choice | 1 | README / result files |
| trifleen/jev-vs-luna-phishing | Phishing emails | 1 | README / result files |
| rupeshpoojary9/poorjev (crossbench) | BANKING77 | 1 | README / result files |
| abhisheksharma001/jev-skill | BANKING77 | 1 | README / result files |
| Jev in Medicine (arXiv 2609.34024) | DiagnosisArena-MCQ; MetaMedQA; NEJM Case Challenges; PubMedQA | 4 | arXiv HTML tables |
| JEV as a Judge for Agent Trace Security (arXiv 2609.34862) | ATBench500 agent-trajectory safety; Four agent-trace benchmarks; MCPHunt agent-trajectory safety; R-Judge agent-trajectory safety; … | 5 | arXiv HTML tables |
| JEV vs LLMs as Rubric Judges (arXiv 2609.29769) | ELLIPSE essay traits; FED-Dialogue quality; FED-Turn dialogue quality; HealthBench rubric criteria; … | 9 | arXiv HTML tables |
| JEV-as-a-Judge (arXiv 2609.26550) | HaluEval QA; HaluEval general; HaluEval summarization; JudgeBench; … | 8 | arXiv HTML tables |
| JevVibe (arXiv 2609.34963) | CyberSecEval Instruct | 1 | arXiv HTML tables |
| Ordinal-Scale Bias in JEV-like Models (arXiv 2609.38827) | 36 ordinal rating datasets; ANLI R1-R3 | 2 | arXiv HTML tables |
| Benchmarking System One models vs trained classifiers (arXiv 2610.00346) | CLINC150; Conversations Gone Awry; Twitter emotion; Wikipedia politeness; … | 6 | arXiv HTML tables |
| Do System One Decisions Add Up? (arXiv 2609.33971) | CLINC150 in-scope; MASSIVE en-US intents; TREC fine-grained question type | 3 | arXiv HTML tables |
| Same Scores, Different Decisions (arXiv 2609.27678) | ContractNLI | 1 | arXiv HTML tables |
| Beyond Answer Confidence (arXiv 2610.01006) | BANKING77; CLINC150 in-scope; Daily Oracle yes/no news forecasting; MMLU-CF; … | 10 | arXiv HTML tables |
| Chinese-Jev (arXiv 2609.36965) | CJ-Bench general subset | 1 | arXiv HTML tables |
| Evaluating System One Models for Agent Security (arXiv 2609.33401) | AgentHarm harmful-request classification; R-Judge interaction-risk classification; WAInjectBench-text prompt-injection detection | 3 | arXiv HTML tables |
| Just Ask Jev / RLCDAlignBench (arXiv 2609.29429) | RLCDAlignBench | 1 | arXiv HTML tables |
| Beyond Calibration: probability axioms (arXiv 2609.33209) | Negation coherence on ChaosNLI + PubMedQA items | 1 | arXiv HTML tables |
| Can Jev Judge Radiology Reports? (arXiv 2609.27607) | RadEvalExpert report-pair error counts; RadEvalX report-pair error counts; ReXErr sentence error detection | 3 | arXiv HTML tables |
| Decide, Don't Generate: DimABSA (arXiv 2609.35293) | SemEval-2026 Task 3 Track A DimABSA | 2 | arXiv HTML tables |
| Jev for Speech-Neuroprosthesis Rescoring (arXiv 2609.33538) | Brain-to-text candidate-sentence rescoring | 1 | arXiv HTML tables |
| Decision-Oriented Recommendation Reranking (arXiv 2609.40241) | Amazon Reviews 2023 Books; Amazon Reviews 2023 Movies and TV; Amazon Reviews 2023 Video Games | 3 | arXiv HTML tables |
| Training-free HAR with Jev (arXiv 2609.36154) | PAMAP2 accelerometer windows; UCI HAR; WISDM accelerometer windows | 3 | arXiv HTML tables |
| Jev for Network Traffic Classification (arXiv 2610.00376) | CESNET-QUICEXT-25; CESNET-QUICEXT-25 with 40 fixed labelled… | 2 | arXiv HTML tables |
| Jev-IDS (arXiv 2610.01079) | NSL-KDD flows | 1 | arXiv HTML tables |
| JEVQA video quality (arXiv 2609.24395) | AOM CTC encodes; AVT-VQDB-UHD-1 subjective quality | 2 | arXiv HTML tables |
| Decision Hijacking (arXiv 2609.28613) | InjecAgent tool-response injection | 1 | arXiv HTML tables |
| JET: Justification Evaluation in Transformer (arXiv 2609.33874) | MMLU | 1 | arXiv HTML tables |
| LLM2Jev (arXiv 2610.02076) | JevBench v1.4.2.2 public set | 1 | arXiv HTML tables |
| this-that-model-1.0 (arXiv 2609.23886) | this-that released benchmark | 1 | arXiv HTML tables |
| JevOut: Natural Context Can Flip Decision Models (arXiv 2609.30243) | BFCL V4; LAR-ECHR; MMLU-Pro; MuSR; … | 7 | arXiv HTML tables |
| Sys1Cal-v1 (arXiv 2609.35342) | Sys1Cal-v1 known-probability True/False items | 1 | arXiv HTML tables |
| Code Owns the Simulation, Jev Owns the Evaluation (arXiv 2610.01834) | ALFWorld unseen games; Cognitive Reflection Test items | 2 | arXiv HTML tables |
| Decision Readouts for Video Anomaly Detection (arXiv 2609.34180) | XD-Violence caption anchors | 1 | arXiv HTML tables |
| HydroJEV (arXiv 2610.02048) | C-Town EPANET cause attribution | 1 | arXiv HTML tables |
| Koa-action (arXiv 2609.36115) | Amazon Reviews Polarity; SST-2 | 2 | arXiv HTML tables |
| jujumilk3/jev-calibration-audit | KoBBQ ambiguous | 1 | README / result files |
| zilliztech/vector-graph-rag (Jev reranker evaluation) | HotpotQA multi-hop retrieval; MuSiQue multi-hop retrieval | 2 | README / result files |
| statsguysam/jev-classification-benchmark | Breast Cancer Wisconsin diagnostic; SST-2; TREC coarse question type; Wine | 5 | README / result files |
| earino/zero-shot-complaint-benchmark (CFPB 113-class) | CFPB consumer complaints | 2 | README / result files |
| morrenhale/decision-benchmark-jev-laya-julia (2026-09-27) | AG News; BANKING77; DAIR Emotion; MASSIVE scenario; … | 5 | README / result files |
| KikiNLP/CanITrustYou-Jev | Agent execution decisions adapted from 10… | 1 | README / result files |
| atmaneayoub/jev-ar-bench | Gulf-Arabic government-service routing | 1 | README / result files |
| yehor-oleksiuk/bonzi-vs-jev-wanli256 | WANLI NLI | 1 | README / result files |
| Lightfield blog (Testing TypeSafe Jev on Text Understanding) | Amazon QA answerability yes/no; CommonsenseQA; CraigslistBargain; MMLU-CF; … | 7 | post HTML table |
| AY Automate blog (Jev vs GPT and Claude) | BANKING77; deepset prompt-injections | 3 | post HTML table |
| SOTAAZ blog (Kev vs Jev) | AG News; BANKING77; CLINC150; MASSIVE en-US; … | 8 | post HTML table |

**Read in the resume pass, no public-benchmark Jev number to add:**
- arXiv: 2610.02046 JEVDB (SemBench system), 2610.00437 JevSpawn, 2609.30186 Jev-Mobile (AndroidWorld 79% is the VLM+Jev system), 2609.26532 REFLEX (BFCL/τ system results), 2609.34227 and 2609.36059 / 2609.23986 (memory systems on LoCoMo/LongMemEval), 2609.30922 JevSoup, 2609.34969 NavJev, 2609.27331 JEV-Star, 2609.28940 pentest case study, 2609.22753 / 2609.23136 (6G edge orchestration, authors' own requests), 2609.24052 / 2610.00213 (Texas crash narratives, audited against coded fields), 2609.36399 (cultural-values survey), 2609.24965 (20 scientific choices), 2609.39496 (synthetic arithmetic rejection), 2609.37470 (synthetic probability contracts), 2609.31142 JevAdvBench (self-consistency, no labels), 2609.26758 (option renaming on 1,200 workflow decisions), 2609.32160 (evidence audit of 28 papers; useful as a cross-check), 2609.30216 (ecosystem census), 2610.01231 (essay), and the Jev-style open-model papers with no Jev row (LLM2Jev External table, AnyJev, Bongard, Dyad, PACT, LAVOIR, Visual Jev, PixelJev, OmniMed-Jev, OpenJev-RLCD, NumericJev, JevSoup).
- HF dataset cards: tasksource-jev-typed-decisions, jevalt-bench, JEVal, rsi-jev-benchmarks, Darwin-27B DI run, autotrust DI results (Jev teacher distributions; contamination vector, do not use), JevJudge-Public, jev-as-a-guardrails, bfcl-v4/v1-jev-routing (no Jev row), jev-ai-benchmark (billing/latency only), kishida/jev-bench, jev-bench-tr (quotes the English jev-bench numbers).
- Blogs: jock.pl (40 private tickets), InfiniSynapse (Apple 10-K founder test, rest quoted), devxlabs (100 private tickets; reranking corpora not named), Kingy, DataCamp, InfoQ, FourWeekMBA, greennode, beri.net (all quote others or the vendor).
- **Still no Jev number** on full GLUE/SuperGLUE, XNLI full, DBpedia full, Yahoo Answers, 20 Newsgroups, SNIPS, ATIS, RTE, Climate-FEVER, Jigsaw, HateXplain or ETHOS.

---

## 2. Target table: one bar per (dataset, metric)

How the columns are built:
- **Canonical Jev**: the zero-shot row with the best tier, then the largest n. This is the number to cite.
- **Best tier-A / B(n≥500) zero-shot**: the highest published Jev number on the same metric from a comparable-size run.
  - It is the hardest honest Z bar.
  - Samples under 500 items are excluded from this column to keep noise out.
- **Jev with train data**: the S-track bar, where one exists.
- **Public train split**: what Track S may train on. Use the upstream train split and never the eval rows.
- Keys are internal grouping ids. The same key across suites means the same upstream data and metric family.
- Tier-X rows are excluded here.

**Intent / routing**

| Key | Dataset (canonical row) | n | Metric | Canonical Jev (suite, tier) | Best tier-A / B(n>=500) zero-shot Jev, same metric (suite, n) | Jev with train data (S bar) | Public train split for S |
|---|---|---:|---|---|---|---|---|
| banking77 | BANKING77 (S6, official test) | 3080 | accuracy | 0.792 (DMB, A) | 0.832 (saurabhkumar8112/jev-gpt5-routing-study, n=500) | 0.924 (simonmesmith/jev-banking77-experiment: accuracy) | yes: PolyAI train 10,003 |
| banking77_20 | Routing (20 intents) | 1000 | accuracy | 0.891 (LangWatch Jev benchmark, B) | same |  |  |
| banking77_btzsc72 | BANKING77 (BTZSC, 72 labels, no-positive rows dropped) | 100 | accuracy | 0.87 (AbdelStark/jev-benchmarks, C) | same |  |  |
| banking77_decision_score | Banking77 (choice) | 300 | Decision Score (0-100; 0 = label-prior | 67.78 (Jevals, B) | same |  |  |
| banking77_macrof1 | BANKING77 | 3080 | macro-F1 | 0.7974 (Decision Index 0.2.1, A) | same |  | yes: PolyAI train 10,003 |
| banking77_order | S4 order stability (BANKING77-derived, 100 bases x 3 orders  | 900 | accuracy | 0.767 (DMB, C) | same |  |  |
| clinc150 | CLINC150 (plus, incl. OOS) | 5500 | accuracy | 0.8945 (Deusser et al. 2026, A) | same |  | yes: clinc_oos plus train 15,250 (incl. 250 oos) |
| clinc150_inscope | CLINC150 (plus, incl. OOS) | 5500 | in_scope_accuracy | 0.9209 (Deusser et al. 2026, A) | same |  | yes |
| clinc150_inscope150 | CLINC150 (in-scope only, OOS excluded) | 4500 | accuracy | 0.9196 (thisisandreeeee/jev-benchmarks, A) | same |  |  |
| clinc150_macrof1 | CLINC150+OOS | 5500 | macro-F1 | 0.8927 (Decision Index 0.2.1, A) | same |  | yes |
| clinc150_oos_recall | CLINC150 (plus, incl. OOS) | 5500 | oos_recall | 0.776 (Deusser et al. 2026, A) | 0.881 (chepyle/jev-test, n=1000) |  | yes: plus train has 250 oos |
| darija | Darija reviews (Moroccan Arabic) | 171 | accuracy | 0.795 (mouadse/jev-vs-laya, C) | same |  |  |
| hwu64 | HWU64 | 1076 | accuracy | 0.8309 (thisisandreeeee/jev-benchmarks, A) | same |  | yes |
| massive_intent_en | massive (choice) | 1000 | accuracy | 0.808 (jev-bench, B) | same |  | yes: 11,514 |
| massive_intent_en_macrof1 | Assistant request routing (MASSIVE en-US, 60 intents) | 1000 | macro-F1 | 0.799 (Alexander-Ollman/laya-ft, B) | same |  |  |
| massive_sv | MASSIVE Swedish routing | 360 | accuracy | 0.889 (oluies/jev-vs-spacy, B) | same |  |  |
| nlupp | NLU++ (S8, folds 18-19, banking+hotel) | 13712 | micro intent F1 | 0.483 (DMB, A) | same |  | yes (other folds) |
| tanaos_intent | tanaos synthetic intent classifier | 3447 | accuracy | 0.8784 (xxkuboxx/jev-eval, B) | same |  |  |

**Topic / sentiment / emotion**

| Key | Dataset (canonical row) | n | Metric | Canonical Jev (suite, tier) | Best tier-A / B(n>=500) zero-shot Jev, same metric (suite, n) | Jev with train data (S bar) | Public train split for S |
|---|---|---:|---|---|---|---|---|
| ag_news | AG News | 7600 | accuracy | 0.8851 (Deusser et al. 2026, A) | same |  | yes: 120,000 |
| ag_news_paraphrase | Topic classification under reworded instructions (AG News) | 200 | accuracy | 0.845 (jev-decision-bench, B) | same |  |  |
| amazon_counterfactual | Product-review counterfactuals | 670 | macro-F1 | 0.865 (Alexander-Ollman/laya-ft, B) | same |  |  |
| app_reviews | Android app reviews rating | 1000 | Spearman | 0.803 (goodrahstar/jev-column-race, B) | same |  |  |
| css_emotion | Emotion (CARER) | 498 | macro-F1 | 0.4836 (Ibrahim & Zaki 2026, B) | same |  |  |
| daily_dialog | DailyDialog emotion (utterances, 7 classes) | 7740 | macro-F1 | 0.385 (elcronos/jev-vs-open-decision-models, A) | same |  | yes: 87,170 utterances |
| dair_emotion | DAIR Emotion | 2000 | accuracy | 0.585 (Deusser et al. 2026, A) | 0.599 (elcronos/jev-vs-open-decision-models, n=2000) |  | yes: 16,000 |
| dair_emotion_macrof1 | DAIR Emotion | 2000 | macro_f1 | 0.4987 (Deusser et al. 2026, A) | same |  |  |
| fin_topic | Twitter financial news topic (20 classes) | 4117 | accuracy | 0.67 (elcronos/jev-vs-open-decision-models, A) | same |  | yes: 16,990 |
| financial_phrasebank | Financial PhraseBank | 970 | accuracy | 0.7299 (Deusser et al. 2026, A) | same |  | yes (atrost train) |
| go_emotions_multilabel | GoEmotions (28 labels, multi-label) | 5427 | macro_f1 | 0.2434 (Deusser et al. 2026, A) | same | 0.3529 (Deusser et al. 2026: macro_f1 (per-label thresholds tuned on ) | yes: 43,410 |
| go_emotions_singlelabel | go_emotions (choice) | 1000 | accuracy | 0.282 (jev-bench, B) | same |  |  |
| imdb | IMDB | 25000 | accuracy | 0.9652 (Deusser et al. 2026, A) | same |  | yes: 25,000 |
| language_id | Language identification (20 langs) | 10000 | accuracy | 0.9964 (Deusser et al. 2026, A) | same |  | yes: 70,000 |
| rotten_tomatoes | Rotten Tomatoes | 1066 | accuracy | 0.9334 (Deusser et al. 2026, A) | same |  | yes: 8,530 |
| sib200 | SIB-200 (205 langs) | 41820 | accuracy | 0.8152 (Deusser et al. 2026, A) | same |  | yes: 701 per language |
| sst2 | SST-2 | 872 | accuracy | 0.9644 (Deusser et al. 2026, A) | same |  | yes: 67,349 |
| sst2_auroc | Movie review sentiment (SST-2) | 200 | auroc | 0.9808 (jev-decision-bench, B) | same |  |  |
| sst2_injection_attacked | Sentiment under prompt injection - attacked | 150 | accuracy | 0.9467 (jev-decision-bench, B) | same |  |  |
| sst2_injection_clean | Sentiment under prompt injection - clean | 150 | accuracy | 0.96 (jev-decision-bench, B) | same |  |  |
| sst5 | SST-5 | 2210 | argmax_accuracy | 0.5792 (Deusser et al. 2026, A) | same |  | yes: 8,544 |
| sst5_spearman | SST-5 | 2210 | spearman | 0.851 (Deusser et al. 2026, A) | same |  |  |
| trec | Question type (TREC coarse) | 200 | accuracy | 0.925 (jev-decision-bench, B) | same |  | yes: 5,452 |
| tweet_topic | TweetTopic single (6 classes) | 1693 | accuracy | 0.793 (elcronos/jev-vs-open-decision-models, A) | same |  | yes: train_2020 + train_2021 |
| tweeteval_emotion | TweetEval emotion (4 classes) | 1000 | accuracy | 0.827 (zhuyansen/jev-zeroshot-vs-bert, B) | same |  | yes |
| yelp5 | yelp5 (score) | 1000 | accuracy | 0.685 (jev-bench, B) | same |  | yes: 650,000 |
| yelp5_spearman | Star rating from review text (Yelp Review Full) | 200 | spearman | 0.9267 (jev-decision-bench, B) | same |  |  |

**NLI / grounding / RC**

| Key | Dataset (canonical row) | n | Metric | Canonical Jev (suite, tier) | Best tier-A / B(n>=500) zero-shot Jev, same metric (suite, n) | Jev with train data (S bar) | Public train split for S |
|---|---|---:|---|---|---|---|---|
| afrixnli | AfriXNLI (18 langs) | 10800 | accuracy | 0.6402 (Deusser et al. 2026, A) | same |  | no (val/test only; use XNLI/MNLI) |
| anli | ANLI R1-R3 | 3200 | accuracy | 0.7394 (Deusser et al. 2026, A) | same |  | yes: 162,865 |
| anli_macrof1 | ANLI | 3200 | macro-F1 | 0.7479 (Decision Index 0.2.1, A) | same |  | yes: 162,865 |
| anli_r3 | Adversarial natural language inference (ANLI R3) | 200 | accuracy | 0.66 (jev-decision-bench, B) | same |  |  |
| art | alphaNLI (ART) | 1532 | accuracy | 0.8388 (Deusser et al. 2026, A) | same |  | yes: 169,654 |
| belebele | Belebele (122 langs) | 109800 | accuracy | 0.8673 (Deusser et al. 2026, A) | same |  | no |
| boolq | BoolQ | 3270 | accuracy | 0.9131 (Deusser et al. 2026, A) | 0.917 (jev-bench, n=1000) |  | yes: 9,427 |
| boolq_auroc | Yes/no reading comprehension (BoolQ) | 200 | auroc | 0.9692 (jev-decision-bench, B) | same |  |  |
| boolq_negation | Yes/no questions and their explicit negations (BoolQ) | 200 | accuracy | 0.805 (jev-decision-bench, B) | same |  |  |
| chaosnli | chaosnli (choice) | 1599 | accuracy | 0.615 (jev-bench, B) | same |  | no (MNLI train) |
| contractnli | ContractNLI | 123 | macro-F1 | 0.7169 (Decision Index 0.2.1, A) | same |  | yes |
| fever | fever_evidence (noul) | 1000 | accuracy | 0.972 (jev-bench, B) | same |  | yes |
| halueval | RAG faithfulness | 666 | balanced accuracy | 0.803 (LangWatch Jev benchmark, B) | same |  |  |
| hover | HoVer | 4000 | accuracy | 0.7285 (Decision Index 0.2.1, A) | same |  | yes |
| kobest_boolq | KoBEST BoolQ | 80 | accuracy | 0.988 (NomaDamas/kojev, C) | same |  |  |
| llm_aggrefact | LLM-AggreFact (11 sources) | 29320 | mean_balanced_accuracy | 0.7858 (Deusser et al. 2026, A) | same |  | dev split only (no train); MiniCheck synthetic train is public |
| llm_aggrefact_pooled | LLM-AggreFact (11 sources) | 29320 | pooled_balanced_accuracy | 0.8306 (Deusser et al. 2026, A) | same |  | dev only |
| medhallu | MedHallu | 1000 | accuracy | 0.929 (stperic/jev-medhallu-benchmark, A) | same |  | yes (pqa_artificial-derived train) |
| mnli | mnli (choice) | 1000 | accuracy | 0.883 (jev-bench, B) | same |  | yes: 392,702 |
| nli4ct | NLI4CT | 5500 | macro-F1 | 0.8406 (Decision Index 0.2.1, A) | same |  | yes |
| paws | PAWS | 8000 | accuracy | 0.8499 (Deusser et al. 2026, A) | 0.855 (zhuyansen/jev-zeroshot-vs-bert, n=1000) |  | yes: 49,401 |
| paws_auroc | Adversarial paraphrase detection (PAWS) | 200 | auroc | 0.9159 (jev-decision-bench, B) | same |  |  |
| pubmedqa_3way | PubMedQA (yes/no/maybe) | 1000 | accuracy | 0.787 (Deusser et al. 2026, A) | same |  | pqa_artificial 211k (labelled set is the eval) |
| pubmedqa_yesno | PubMedQA (noul) | 300 | accuracy | 0.9127 (Jevals, B) | same |  | pqa_artificial |
| pubmedqa_yesno_decision_score | PubMedQA (noul) | 300 | Decision Score (0-100; 0 = label-prior | 69.03 (Jevals, B) | same |  |  |
| ragtruth | RAGTruth | 2700 | F1 on hallucinated class | 0.7653 (Decision Index 0.2.1, A) | same |  | yes |
| squad_sentence | Answer-sentence selection (SQuAD v1.1) | 200 | accuracy | 0.945 (jev-decision-bench, B) | same |  |  |
| squad_value | Value selection from pre-extracted spans (SQuAD v1.1) | 155 | accuracy | 0.929 (jev-decision-bench, B) | same |  |  |
| strategyqa_closed | strategyqa_closed (noul) | 687 | accuracy | 0.785 (jev-bench, B) | same |  | yes |
| strategyqa_grounded | strategyqa_grounded (noul) | 687 | accuracy | 0.956 (jev-bench, B) | same |  | yes |
| tabfact | TabFact (120 tables/claims) | 120 | accuracy | 0.917 (slavadubrov/sgr-judge-bench, C) | same |  |  |
| vast | VAST | 3006 | macro-F1 | 0.6463 (Decision Index 0.2.1, A) | same |  | yes |
| wanli | WANLI | 256 | accuracy | 0.758 (Kev, C) | same |  |  |

**Safety / moderation / injection**

| Key | Dataset (canonical row) | n | Metric | Canonical Jev (suite, tier) | Best tier-A / B(n>=500) zero-shot Jev, same metric (suite, n) | Jev with train data (S bar) | Public train split for S |
|---|---|---:|---|---|---|---|---|
| aegis_v1 | Aegis v1 unsafe prompt | 359 | F1 (harmful class) | 0.8909 (mbburabak/jev-safety-benchmark, A) | same |  |  |
| aegis_v2_prompt | Aegis v2 unsafe prompt | 1928 | F1 (harmful class) | 0.8357 (mbburabak/jev-safety-benchmark, A) | same |  | yes |
| aegis_v2_response | Aegis v2 unsafe response | 1928 | F1 (harmful class) | 0.8017 (mbburabak/jev-safety-benchmark, A) | same |  | yes |
| agb_de | AGB-DE (German T&C voidness) | 755 | f1 | 0.2038 (Deusser et al. 2026, A) | same |  | yes |
| civil_comments | civil_comments (noul) | 2000 | accuracy | 0.729 (jev-bench, B) | same |  | yes: 1.8M |
| civil_comments_auroc | Toxic comment detection (Civil Comments) | 200 | auroc | 0.8716 (jev-decision-bench, B) | same |  |  |
| css_implicit_hate | Implicit/latent hate | 498 | macro-F1 | 0.4392 (Ibrahim & Zaki 2026, B) | same |  |  |
| deepset_prompt_injections | deepset prompt-injections | 116 | accuracy | 0.7414 (Deusser et al. 2026, A) | same |  | yes: 546 |
| deepset_prompt_injections_all | deepset prompt-injections (all 662, with deployment context) | 662 | accuracy | 0.965 (Gaurav-Gosain/jev-sec-bench, A) | same |  |  |
| deepset_prompt_injections_f1 | Prompt-injection detection | 116 | macro-F1 | 0.787 (Alexander-Ollman/laya-ft, B) | same |  |  |
| deepset_prompt_injections_fnr | deepset prompt-injections (546) false-negative rate | 546 | false-negative rate (lower is better) | 0.473 (ca7ai/jev-prompt-sentry, B) | same |  |  |
| email_spam | Email spam (18,514 emails) | 18514 | accuracy | 0.983 (Arize blog, B) | same |  |  |
| enron_spam | Enron spam | 300 | accuracy | 0.987 (onlyoneaman/jev-eval, B) | same |  |  |
| harmbench_prompt | HarmBench harmful request (detection rate) | 239 | detection rate | 0.9916 (mbburabak/jev-safety-benchmark, A) | same |  |  |
| harmbench_response | HarmBench harmful response | 596 | F1 (harmful class) | 0.8746 (mbburabak/jev-safety-benchmark, A) | same |  |  |
| hatecheck | HateCheck (hateful) | 3728 | F1 (harmful class) | 0.9923 (mbburabak/jev-safety-benchmark, A) | same |  | no |
| injection_combined | Combined injection corpus (4 HF datasets, dedup) | 11900 | AUPRC | 0.98 (ASEVlad/jev-injection-bench, A) | same |  |  |
| isarcasmeval | iSarcasmEval | 1400 | Sarcasm F1 · track A, English | 0.5051 (Decision Index 0.2.1, A) | same |  | yes |
| lw_injection | Prompt injection | 1000 | catch rate @ 5% false alarms | 0.946 (LangWatch Jev benchmark, B) | same |  |  |
| lw_moderation | Moderation | 667 | AUROC | 0.903 (LangWatch Jev benchmark, B) | same |  |  |
| lw_offtopic | Off-topic | 1000 | balanced accuracy | 0.934 (LangWatch Jev benchmark, B) | same |  |  |
| lw_pii | PII | 1000 | catch rate @ 5% false alarms | 0.908 (LangWatch Jev benchmark, B) | same |  |  |
| lw_tool_routing | Tool routing | 1000 | accuracy | 0.783 (LangWatch Jev benchmark, B) | same |  |  |
| measuring_hate_speech | measuring_hate_speech (score) | 1000 | accuracy | 0.527 (jev-bench, B) | same |  | yes |
| openai_moderation | OpenAI moderation eval (8 categories) | 1680 | mean_auprc | 0.7173 (Deusser et al. 2026, A) | same |  | no |
| phishnchips | PhishNChips phishing decisions | 2000 | accuracy | 0.6255 (Decision Index 0.2.1, A) | 0.626 (anisselbd/jev-phishing-bench, n=2000) |  | no |
| redhat_injection | Prompt injection (EvalHub NeMo Guardrails benchmark) | ? | accuracy | 0.8635 (Red Hat Developer, C) | same |  |  |
| sms_spam | SMS Spam | 5574 | f1 | 0.9381 (Deusser et al. 2026, A) | same |  | no (single split; Deusser scores all rows) |
| sms_spam_auroc | SMS spam detection | 200 | auroc | 0.9996 (jev-decision-bench, B) | same |  |  |
| sms_spam_macrof1 | SMS spam | 1000 | macro-F1 | 0.908 (Alexander-Ollman/laya-ft, B) | same |  |  |
| toxic_chat | ToxicChat | 5083 | f1 | 0.7862 (Deusser et al. 2026, A) | same | 0.7933 (Deusser et al. 2026: f1 toxic (threshold tuned on 1,000 dev e) | yes: 5,082 |
| toxic_chat_jailbreak | ToxicChat | 5083 | f1 (jailbreak head) | 0.7173 (Deusser et al. 2026, A) | same |  | yes |
| toxigen | ToxiGen (annotated) | 940 | accuracy | 0.8777 (Deusser et al. 2026, A) | same |  | yes (annotated train 8,960) |
| wildguardtest_prompt | WildGuardTest prompt harmful | 1699 | F1 (harmful class) | 0.8843 (mbburabak/jev-safety-benchmark, A) | same |  | wildguardmix train (gated) |
| xstest | XSTest harmless prompts (false-alarm rate, lower is better) | 250 | false-alarm rate | 0.076 (Alexander-Ollman/laya-ft, B) | same |  | no |

**Legal**

| Key | Dataset (canonical row) | n | Metric | Canonical Jev (suite, tier) | Best tier-A / B(n>=500) zero-shot Jev, same metric (suite, n) | Jev with train data (S bar) | Public train split for S |
|---|---|---:|---|---|---|---|---|
| cuad | CUAD clause detection (41 clauses x 500 pages) | 20500 | F1 | 0.519 (matu79go/jev-hanko, B) | same |  |  |
| ledgar | LexGLUE LEDGAR | 10000 | micro-F1 | 0.753 (chepyle/jev-test, A) | same |  | yes: 60,000 |
| lexglue_case_hold | LexGLUE CaseHOLD | 3600 | micro-F1 | 0.773 (chepyle/jev-test, A) | same |  | yes |
| lexglue_ecthr_a | LexGLUE ECtHR A | 1000 | micro-F1 | 0.73 (chepyle/jev-test, A) | same |  | yes |
| lexglue_ecthr_b | LexGLUE ECtHR B | 1000 | micro-F1 | 0.754 (chepyle/jev-test, A) | same |  | yes |
| lexglue_eurlex | LexGLUE EUR-LEX | 5000 | micro-F1 | 0.391 (chepyle/jev-test, A) | same |  | yes |
| lexglue_mean | LexGLUE 7-task mean | 23607 | arithmetic mean micro-F1 | 0.699 (chepyle/jev-test, A) | same | 0.742 (chepyle/jev-test: arithmetic mean micro-F1) | yes |
| lexglue_scotus | LexGLUE SCOTUS | 1400 | micro-F1 | 0.726 (chepyle/jev-test, A) | same |  | yes |
| unfair_tos | LexGLUE UNFAIR-ToS (8 labels) | 1607 | micro_f1 | 0.4993 (Deusser et al. 2026, A) | same | 0.7482 (Deusser et al. 2026: micro_f1 (per-label thresholds tuned on ) | yes: 5,532 |
| unfair_tos_lexglue | LexGLUE UNFAIR-ToS | 1607 | micro-F1 | 0.764 (chepyle/jev-test, A) | same |  | yes: 5,532 |

**Ordinal / scoring**

| Key | Dataset (canonical row) | n | Metric | Canonical Jev (suite, tier) | Best tier-A / B(n>=500) zero-shot Jev, same metric (suite, n) | Jev with train data (S bar) | Public train split for S |
|---|---|---:|---|---|---|---|---|
| helpsteer2_5attr | HelpSteer2 (5 attributes) | 1038 | mean_spearman | 0.4122 (Deusser et al. 2026, A) | same |  | yes: 20,324 |
| helpsteer2_helpfulness | helpsteer2_helpfulness (score) | 1000 | accuracy | 0.363 (jev-bench, B) | same |  | yes: 20,324 |
| helpsteer2_helpfulness_decision_score | HelpSteer2 helpfulness (score) | 300 | Decision Score (0-100; 0 = label-prior | 9.2 (Jevals, B) | same |  |  |
| helpsteer2_helpfulness_spearman | Response helpfulness rating (HelpSteer2) | 200 | spearman | 0.4902 (jev-decision-bench, B) | same |  |  |
| helpsteer2_verbosity | helpsteer2_verbosity (score) | 1000 | accuracy | 0.341 (jev-bench, B) | same |  |  |
| stsb | STS-B | 1379 | spearman | 0.8902 (Deusser et al. 2026, A) | same |  | yes: 5,749 |
| stsb_6level_acc | stsb (score) | 1000 | accuracy | 0.538 (jev-bench, B) | same |  |  |
| stsb_spearman | Semantic textual similarity (STS-Benchmark) | 200 | spearman | 0.9311 (jev-decision-bench, B) | same |  |  |
| summeval | SummEval (4 dims) | 1600 | mean_group_spearman | 0.5538 (Deusser et al. 2026, A) | same |  | no |

**Typed-decision suites**

| Key | Dataset (canonical row) | n | Metric | Canonical Jev (suite, tier) | Best tier-A / B(n>=500) zero-shot Jev, same metric (suite, n) | Jev with train data (S bar) | Public train split for S |
|---|---|---:|---|---|---|---|---|
| decidebench | DecideBench (8 task families, contrastive pairs) | 400 | accuracy | 0.98 (DecideBench v1.1, B) | same |  |  |
| decision_index | Decision Index (aggregate of 38 panel benchmarks) | 119898 | index (skill, 0-100) | 57.91 (Decision Index 0.2.1, A) | same |  |  |
| decisionbench | DecisionBench overall | 23900 | primary accuracy | 0.7203 (DecisionBench, A) | same |  | no |
| decisionbench_binary | DecisionBench view primitive:binary_classification | 6388 | accuracy | 0.5922 (DecisionBench, A) | same |  |  |
| decisionbench_choice | DecisionBench view primitive:candidate_selection | 15814 | accuracy | 0.8009 (DecisionBench, A) | same |  |  |
| decisionbench_eng | DecisionBench view suite:DecisionBench(eng, v1) | 22700 | accuracy | 0.719 (DecisionBench, A) | same |  |  |
| decisionbench_ordinal | DecisionBench view primitive:ordinal_scoring | 1698 | accuracy | 0.4511 (DecisionBench, A) | same |  |  |
| jevbench_calibration | JevBench Calibration axis | 1624 | Calibration (0-100) | 88.03 (JevBench v1.5.5, B) | same |  |  |
| jevbench_composite | JevBench composite score | 1624 | JevBench score (0-100) | 72.13 (JevBench v1.5.5, B) | same |  | 601 public items (disclosure + gap penalty) |
| jevbench_intelligence | JevBench Intelligence axis | 1624 | Intelligence (0-100) | 72.00 (JevBench v1.5.5, B) | same |  |  |
| jevbench_praveen | jev-bench macro (22 configs) | 22773 | macro accuracy | 0.733 (jev-bench, B) | same |  |  |
| jevbench_public | JevBench public | 231 | accuracy | 0.857 (LangWatch Jev benchmark, B) | same |  |  |
| kev_decision-v7 | Kev decision-v7 | ? | accuracy | 0.8331 (WebJev, B) | same |  |  |
| kev_decision-v7_dev | Kev decision-v7 dev | 1264 | accuracy | 0.845 (Kev suites, B) | same |  |  |
| kev_decision-v7_test | Kev decision-v7 test | 1200 | accuracy | 0.835 (Kev suites, B) | same |  |  |
| kev_transfer-v4 | Kev transfer-v4 | ? | accuracy | 0.8521 (WebJev, B) | same |  |  |
| kev_transfer-v4_dev | Kev transfer-v4 dev (new sources) | 764 | accuracy | 0.857 (Kev, B) | same |  |  |
| kev_transfer-v4_test | Kev transfer-v4 test | 656 | accuracy | 0.877 (Kev suites, B) | same |  |  |
| typed_decisions | typed-decisions (4 workflows) | 400 | accuracy (argmax vs gold argmax, 2,000 | 0.727 (typed-decisions, A) | same |  | yes: 1,200 cases (moves row to 'fitted' table) |
| typed_decisions_choice | typed-decisions choice subset | ? | accuracy (choice questions) | 0.72 (typed-decisions, A) | same |  |  |
| typed_decisions_customer_service | typed-decisions customer_service (100 rows) | 100 | accuracy | 0.78 (JoeSlain/jev-gliclass-bench, C) | same |  |  |
| typed_decisions_kl | typed-decisions KL | 400 | KL from gold (lower is better) | 1.442 (typed-decisions, A) | same |  |  |
| typed_decisions_noul | typed-decisions noul subset | ? | accuracy (noul questions) | 0.775 (typed-decisions, A) | same |  |  |
| typed_decisions_score | typed-decisions score subset | ? | accuracy (score questions) | 0.696 (typed-decisions, A) | same |  |  |

**Knowledge / reasoning (shown, small models not expected to win)**

| Key | Dataset (canonical row) | n | Metric | Canonical Jev (suite, tier) | Best tier-A / B(n>=500) zero-shot Jev, same metric (suite, n) | Jev with train data (S bar) | Public train split for S |
|---|---|---:|---|---|---|---|---|
| arc_challenge | ARC-Challenge | 1172 | accuracy | 0.9778 (Decision Index 0.2.1, B) | 0.979 (jev-bench, n=1000) |  | yes: 1,119 |
| arc_e+c | ARC Easy+Challenge | 3548 | accuracy | 0.9876 (Deusser et al. 2026, A) | same |  | yes: 3,370 |
| arc_easy | ARC-Easy | 2376 | accuracy | 0.9933 (Decision Index 0.2.1, B) | same |  | yes |
| bbh | BBH | 5507 | accuracy | 0.9292 (Decision Index 0.2.1, A) | same |  | no |
| bigbench_mc | BIG-bench MC subset (93 tasks) | 13228 | accuracy | 0.8139 (Deusser et al. 2026, A) | same |  | per-task train in tasksource/bigbench |
| ceval | C-Eval (52 subjects) | 12342 | accuracy | 0.8392 (Deusser et al. 2026, A) | same |  | dev/val only |
| cladder | CLadder | 5000 | accuracy | 0.7264 (Decision Index 0.2.1, A) | same |  |  |
| commonsense_qa | CommonsenseQA | 1221 | accuracy | 0.8821 (Deusser et al. 2026, A) | same |  | yes: 9,741 |
| cruxeval | CRUXEval | 570 | accuracy | 0.7298 (Decision Index 0.2.1, A) | same |  |  |
| gaokao | GAOKAO-Bench objective single-choice | 1497 | accuracy | 0.9085 (kuaitoukuai/jev-gaokao-eval, A) | same |  |  |
| gpqa_diamond | GPQA Diamond | 198 | accuracy | 0.7828 (Decision Index 0.2.1, A) | same |  | no |
| gsm8k_mc | GSM8K | 1319 | accuracy | 0.7987 (Decision Index 0.2.1, A) | same |  |  |
| hellaswag | HellaSwag | 10042 | accuracy | 0.9549 (Deusser et al. 2026, A) | same |  | yes: 39,905 |
| hle | HLE | 501 | accuracy | 0.2036 (Decision Index 0.2.1, A) | same |  | no |
| jmedqa | JMedQA (Japanese medical licensing, 2018-2026) | 3556 | exact-set accuracy | 0.8858 (kokuren333/jev-jmle-benchmark, A) | same |  |  |
| jmmlu_accounting | JMMLU professional_accounting | 150 | accuracy | 0.7467 (shunmoridev/jev-jp-accounting-bench, C) | same |  |  |
| kobest_copa | KoBEST COPA | 80 | accuracy | 0.988 (NomaDamas/kojev, C) | same |  |  |
| kobest_hellaswag | KoBEST HellaSwag | 80 | accuracy | 0.775 (NomaDamas/kojev, C) | same |  |  |
| kobest_sentineg | KoBEST SentiNeg | 80 | accuracy | 0.938 (NomaDamas/kojev, C) | same |  |  |
| kobest_wic | KoBEST WiC | 80 | accuracy | 0.888 (NomaDamas/kojev, C) | same |  |  |
| logiqa | Logical reasoning over a passage (LogiQA) | 200 | accuracy | 0.765 (jev-decision-bench, B) | same |  | yes |
| mmlu | MMLU (57 subjects) | 14042 | accuracy | 0.9181 (Deusser et al. 2026, A) | 0.923 (jev-bench, n=1000) |  | auxiliary_train 99,842 |
| mmlu_pro | MMLU-Pro | 12032 | accuracy | 0.827 (Decision Index 0.2.1, A) | 0.834 (WebJev, n=1000) |  | validation 70 only |
| msmarco_rerank | Passage reranking (MS MARCO v1.1) | 100 | mrr | 0.4987 (jev-decision-bench, B) | same |  |  |
| musr | MuSR | 752 | accuracy | 0.6609 (Decision Index 0.2.1, A) | same |  |  |
| openbookqa | OpenBookQA | 500 | accuracy | 0.942 (scienthoon/jev-ood-calibration, A) | same |  |  |
| sat | SAT practice tests (11) | 1009 | accuracy | 0.912 (drvegabermudez/jev-sat5-text-only-evaluation, B) | same |  |  |
| sata_bench | SATA-Bench | 1650 | case exact accuracy | 0.2642 (Decision Index 0.2.1, A) | same |  |  |
| tmmluplus | TMMLU+ v1.1 (66 subjects) | 19646 | category macro accuracy | 0.7757 (lianghsun/jev-tmmluplus-eval, A) | same |  | dev/val only |
| truthfulqa_mc1 | Truthful answers to misleading questions (TruthfulQA MC1) | 200 | accuracy | 0.95 (jev-decision-bench, B) | same |  | no |
| winogrande | WinoGrande | 1267 | accuracy | 0.914 (Deusser et al. 2026, A) | 0.9195 (Decision Index 0.2.1, n=1267) |  | yes: train_xl 40,398 |
| xnli_ar | Natural language inference - XNLI (Arabic) | 150 | accuracy | 0.74 (jev-decision-bench, B) | same |  |  |
| xnli_en | Natural language inference - XNLI (English) | 150 | accuracy | 0.86 (jev-decision-bench, B) | same |  |  |
| xnli_fr | Natural language inference - XNLI (French) | 150 | accuracy | 0.7733 (jev-decision-bench, B) | same |  |  |
| xnli_hi | Natural language inference - XNLI (Hindi) | 150 | accuracy | 0.7067 (jev-decision-bench, B) | same |  |  |
| xnli_sw | Natural language inference - XNLI (Swahili) | 150 | accuracy | 0.7 (jev-decision-bench, B) | same |  |  |
| xnli_zh | Natural language inference - XNLI (Chinese) | 150 | accuracy | 0.7933 (jev-decision-bench, B) | same |  |  |

**Other public sets**

| Key | Dataset (canonical row) | n | Metric | Canonical Jev (suite, tier) | Best tier-A / B(n>=500) zero-shot Jev, same metric (suite, n) | Jev with train data (S bar) | Public train split for S |
|---|---|---:|---|---|---|---|---|
| acos | ACOS | 400 | per-review F1 | 0.2952 (Decision Index 0.2.1, A) | same |  |  |
| aita | Reddit AITA verdicts | 770 | accuracy | 0.754 (dchristopoulos/jev-bench, B) | same |  |  |
| ajgt | Arabic tweet sentiment (AJGT) | 200 | auroc | 0.9539 (jev-decision-bench, B) | same |  |  |
| amazon_esci | Amazon ESCI | 5000 | macro-F1 | 0.5521 (Decision Index 0.2.1, A) | same |  | yes |
| api_bank | API-Bank | 508 | accuracy | 0.8819 (Decision Index 0.2.1, A) | same |  |  |
| arxiv_category | arXiv primary category, papers submitted >= 2026-09-17 | 258 | accuracy | 0.891 (zhuyansen/jev-zeroshot-vs-bert, C) | same |  |  |
| beir_fiqa | BEIR FiQA rerank | 300 | nDCG@10 | 0.376 (hev/reranker, B) | same |  |  |
| beir_nfcorpus | BEIR NFCorpus rerank | 323 | nDCG@10 | 0.3623 (denser-org/rerank-bench-jev, A) | same |  |  |
| beir_scifact | BEIR SciFact rerank (BM25 top-100) | 300 | nDCG@10 | 0.7699 (denser-org/rerank-bench-jev, A) | 0.772 (hev/reranker, n=300) |  |  |
| bfcl | BFCL | 1694 | case exact accuracy | 0.9575 (Decision Index 0.2.1, A) | same |  | no |
| bpomp | BPoMP | 811 | accuracy | 0.906 (Decision Index 0.2.1, A) | same |  |  |
| bright | BRIGHT | 220 | nDCG@10 | 0.4752 (Decision Index 0.2.1, A) | same |  |  |
| cfcolor | cfcolor | 5000 | accuracy | 0.6474 (Decision Index 0.2.1, A) | same |  |  |
| cfpb_complaints | Complaint routing | 1000 | accuracy | 0.787 (LangWatch Jev benchmark, B) | same |  |  |
| chessbench | ChessBench | 5000 | accuracy | 0.1722 (Decision Index 0.2.1, A) | same |  |  |
| cohen2006 | Cohen 2006 systematic-review screening (15 reviews) | 16015 | recall (include) | 0.9 (Saeedabdf/jev-screening-benchmark, A) | same |  | no |
| cohen2006_adhd | Cohen 2006 ADHD abstracts | 851 | accuracy | 0.946 (PistachioAIHQ/jev-synergy-screening, B) | same |  |  |
| commit_type | Commit type | 1000 | accuracy | 0.683 (LangWatch Jev benchmark, B) | same |  |  |
| commonlit | CommonLit readability (300 excerpts) | 300 | Pearson r | 0.824 (keltokhy/jsort, B) | same |  |  |
| conll2003_typing | Entity typing (CoNLL-2003) | 200 | accuracy | 0.895 (jev-decision-bench, B) | same |  | yes |
| css_conv_go_awry | Conversations Gone Awry (toxicity forecasting) | 500 | macro-F1 | 0.5015 (Ibrahim & Zaki 2026, B) | same |  |  |
| css_discourse | Discourse acts | 497 | macro-F1 | 0.5952 (Ibrahim & Zaki 2026, B) | same |  |  |
| css_flute | FLUTE figurative language | 500 | macro-F1 | 0.8633 (Ibrahim & Zaki 2026, B) | same |  |  |
| css_ibc | Ideological Books Corpus | 498 | macro-F1 | 0.6325 (Ibrahim & Zaki 2026, B) | same |  |  |
| css_indian_english_dialect | Indian English dialect features | 266 | macro-F1 | 0.6401 (Ibrahim & Zaki 2026, B) | same |  |  |
| css_media_ideology | Media ideology (document) | 498 | macro-F1 | 0.654 (Ibrahim & Zaki 2026, B) | same |  |  |
| css_mrf | Misinfo Reaction Frames | 500 | macro-F1 | 0.7955 (Ibrahim & Zaki 2026, B) | same |  |  |
| css_persuasion | Persuasion strategies (conversation) | 434 | macro-F1 | 0.5605 (Ibrahim & Zaki 2026, B) | same |  |  |
| css_raop | Random Acts of Pizza (persuasion) | 399 | macro-F1 | 0.6074 (Ibrahim & Zaki 2026, B) | same |  |  |
| css_reddit_humor | Reddit humor | 500 | macro-F1 | 0.5633 (Ibrahim & Zaki 2026, B) | same |  |  |
| css_semeval_stance | SemEval-2016 T6 stance | 435 | macro-F1 | 0.7341 (Ibrahim & Zaki 2026, B) | same |  |  |
| css_talklife | TalkLife empathy | 498 | macro-F1 | 0.2688 (Ibrahim & Zaki 2026, B) | same |  |  |
| css_tempowic | TempoWiC semantic change | 344 | macro-F1 | 0.6826 (Ibrahim & Zaki 2026, B) | same |  |  |
| css_tropes | Character tropes | 114 | macro-F1 | 0.1905 (Ibrahim & Zaki 2026, C) | same |  |  |
| css_wiki_corpus | Wikipedia power/talk | 500 | macro-F1 | 0.5813 (Ibrahim & Zaki 2026, B) | same |  |  |
| css_wiki_politeness | Wikipedia politeness | 498 | macro-F1 | 0.573 (Ibrahim & Zaki 2026, B) | same |  |  |
| eclipse_severity | Eclipse bug-title severity | 2000 | accuracy | 0.491 (actuallyrizzn/decision-systems-bakeoff, B) | same |  |  |
| enem2025 | ENEM 2025 | 182 | accuracy | 0.566 (patryckalves/jev-no-enem, C) | same |  |  |
| fake_jobs | Real/Fake job postings | 1000 | PR-AUC (fraud) | 0.276 (geckguy/job-posting-triage, B) | same |  |  |
| finentity | FinEntity | 979 | macro-F1 | 0.8698 (Decision Index 0.2.1, A) | same |  | no official split |
| forecastbench | ForecastBench | 10139 | Brier (lower is better) | 0.1736 (Decision Index 0.2.1, A) | same |  |  |
| frontier_pooled | Pooled 4 tasks (BANKING77, BoolQ, Yelp-5, ChaosNLI; 50 each) | 200 | accuracy | 0.725 (manjunathshiva/jev-frontier-bench, C) | same |  |  |
| habermas | Habermas Machine | 1676 | accuracy | 0.4594 (Decision Index 0.2.1, A) | same |  |  |
| heswy_mean | 5-dataset mean (BANKING77 300, CLINC150+OOS 200, SST-5 200,  | 1050 | equal-weight mean accuracy | 0.7937 (heswy/Jev-Benchmark, B) | same |  |  |
| home_appliance | Home appliance simulator | 88 | case exact accuracy | 0.5227 (Decision Index 0.2.1, A) | same |  |  |
| humicroedit | Humicroedit | 2628 | accuracy | 0.6187 (Decision Index 0.2.1, A) | same |  |  |
| jfinqa | jfinqa | 1000 | accuracy | 0.731 (shunmoridev/jev-jp-accounting-bench, B) | same |  |  |
| label_pressure | Label-set pressure (CLINC/MTOP/GoEmotions/DBpedia/fin tweets | ? | accuracy at 128 candidates | 0.6 (gazelle93/decision-models-under-pressure, C) | same |  |  |
| mind2web | Mind2Web train_6 shard (planning top-1) | 706 | top-1 accuracy | 0.521 (hosamsh/jev-mind2web, B) | same |  |  |
| mind2web_jevforge | Web-agent actions | 2329 | accuracy | 0.708 (LangWatch Jev benchmark, B) | same |  |  |
| nevir | NevIR negation pairs | 1383 | pairwise accuracy | 0.71 (anessbelbati/jev-rerank-bench, A) | same |  |  |
| newyorker | New Yorker | 528 | accuracy | 0.7008 (Decision Index 0.2.1, A) | same |  |  |
| nfcorpus | Biomedical document reranking (BEIR NFCorpus) | 60 | ndcg10 | 0.7337 (jev-decision-bench, B) | same |  |  |
| nimble | Nimble holdout | ? | accuracy | 0.9259 (WebJev, B) | same |  |  |
| nsl_kdd | NSL-KDD flows (pilot) | 900 | F1 (attack) | 0.859 (jev-ids/jev-ids, C) | same |  |  |
| osob_macro | 4-dataset macro (SST-2, AG News, Emotion, BANKING77; 2,500 e | 10000 | macro accuracy | 0.793 (dylantom2012 / zhlei07 open-system-one, B) | same |  |  |
| pop909_cl | POP909-CL | 2000 | accuracy | 0.181 (Decision Index 0.2.1, A) | same |  |  |
| ppe_human | PPE Human Preference V1 | 16038 | no-tie pairwise accuracy | 0.644 (goya4140/jev-reward-model-evaluation, A) | same |  |  |
| prmbench | PRMBench Preview | 6216 | official PRM score | 0.6638 (goya4140/jev-reward-model-evaluation, A) | same |  |  |
| processbench | ProcessBench | 3400 | official mean F1 | 0.6951 (goya4140/jev-reward-model-evaluation, A) | same |  |  |
| redhat_safety | Content safety (toxicity-profanity-safety) | ? | accuracy | 0.862 (Red Hat Developer, C) | same |  |  |
| rerank_mean | BEIR+BRIGHT+CodeSearchNet rerank (8 sets, dataset mean) | 1617 | mean nDCG@10 | 0.692 (anessbelbati/jev-rerank-bench, A) | same |  |  |
| rewardbench | RewardBench v1 | 2985 | official 4-section macro | 0.9258 (goya4140/jev-reward-model-evaluation, A) | same |  | no (use preference data) |
| rewardbench2 | RewardBench 2 | 1865 | official 6-domain macro | 0.8115 (goya4140/jev-reward-model-evaluation, A) | same |  |  |
| rmbench_pairwise | RM-Bench structured pairwise | 1327 | official 4-domain macro | 0.8129 (goya4140/jev-reward-model-evaluation, A) | same |  |  |
| rmbench_pointwise | RM-Bench pointwise | 7962 | official 4-domain macro | 0.8379 (goya4140/jev-reward-model-evaluation, A) | same |  |  |
| routerbench | RouterBench | 5001 | selected quality (quality objective) | 0.7992 (Decision Index 0.2.1, B) | same |  |  |
| rubricbench | RubricBench (human rubric given) | 1147 | pairwise accuracy | 0.7602 (goya4140/jev-reward-model-evaluation, A) | same |  |  |
| scienthoon_tickets | scienthoon tickets | ? | accuracy | 0.7491 (WebJev, B) | same |  |  |
| semif | SemIf | ? | accuracy | 0.9841 (WebJev, B) | same |  |  |
| sgd | SGD/SGD-X | 2500 | macro-F1 | 0.4295 (Decision Index 0.2.1, B) | same |  |  |
| thaiexam | ThaiExam | 567 | accuracy | 0.707 (vehas/thaiexam-jev-charts, C) | same |  |  |
| toolret | ToolRet | 685 | nDCG@10 | 0.6528 (Decision Index 0.2.1, A) | same |  |  |
| upworthy | Upworthy headline A/B (confirmatory pairs) | 10984 | accuracy | 0.645 (Gaurav-Gosain/jev-headline-bench, A) | same |  |  |
| when2call | When2Call | 3652 | accuracy | 0.8097 (Decision Index 0.2.1, A) | same |  | yes (train jsonl) |
| whowhen | Who&When Pro agent-failure attribution | 6257 | 'All' score | 0.313 (TokenTrim/jev-agent-failure-benchmark, B) | same |  |  |

### 2b. Bars added in the 2026-10-03 resume pass

Tier-X rows are left out. "Earlier best" is the highest first-pass Jev number (lowest for lower-is-better metrics) with the same upstream dataset (`hf_id`), metric family and track; "new" means the first pass had no comparable row. Lower-is-better metrics are marked in the metric column.

**Intent / topic / sentiment / complaints**

| Dataset | n | Metric | Jev (suite, tier, track) | Earlier best, same hf_id + metric (suite, n) |
|---|---:|---|---|---|
| AG News | 1,000 | balanced accuracy (raw decision) | 0.875 (QuicqDev/Jev-vs-ML (protocol V3.0.1), B, Z) | new |
| BANKING77 | 1,500 | balanced accuracy (raw decision) | 0.789 (QuicqDev/Jev-vs-ML (protocol V3.0.1), B, Z) | new |
| SMS Spam | 1,000 | balanced accuracy (raw decision) | 0.961 (QuicqDev/Jev-vs-ML (protocol V3.0.1), B, Z) | 0.961 (mugenkyou/JEV-VS-ML, n=1,000) |
| IMDb | 1,000 | balanced accuracy (raw decision) | 0.963 (QuicqDev/Jev-vs-ML (protocol V3.0.1), B, Z) | 0.963 (mugenkyou/JEV-VS-ML, n=1,000) |
| BANKING77 (1 example per class in prompt) | 1,500 | balanced accuracy | 0.819 (QuicqDev/Jev-vs-ML (protocol V3.0.1), B, S) | new |
| BANKING77 (77-way) | 154 | accuracy | 0.812 (rupeshpoojary9/poorjev (crossbench), C, Z) | 0.891 (LangWatch Jev benchmark (release 2026-09, n=1,000) |
| BANKING77 (77-way routing) | 134 | accuracy (first-draft question) | 0.761 (abhisheksharma001/jev-skill, C, Z) | 0.891 (LangWatch Jev benchmark (release 2026-09, n=1,000) |
| CLINC150 (600 in-scope + 200 OOS) | 800 | accuracy | 0.913 (Benchmarking System One models vs trained cla, B, Z) | 0.93 (nikkoxgonzales/jev-certify, n=400) |
| Twitter emotion (hashtag labels, 6 classes) | 498 | accuracy | 0.504 (Benchmarking System One models vs trained cla, B, Z) | 0.599 (elcronos/jev-vs-open-decision-models, n=2,000) |
| TREC fine-grained question type (50 labels) | 500 | accuracy (flat fine-label) | 0.722 (Do System One Decisions Add Up? (arXiv 2609.3, A, Z) | new |
| CLINC150 in-scope (label-balanced sample) | 1,000 | accuracy (flat fine-label) | 0.91 (Do System One Decisions Add Up? (arXiv 2609.3, B, Z) | 0.93 (nikkoxgonzales/jev-certify, n=400) |
| MASSIVE en-US intents (label-balanced sample) | 1,000 | accuracy (flat fine-label) | 0.833 (Do System One Decisions Add Up? (arXiv 2609.3, B, Z) | 0.889 (oluies/jev-vs-spacy, n=360) |
| BANKING77, intent names only | 770 | accuracy | 0.821 (Beyond Answer Confidence (arXiv 2610.01006), B, Z) | 0.891 (LangWatch Jev benchmark (release 2026-09, n=1,000) |
| BANKING77, intent names + 8 train examples per intent | 770 | accuracy | 0.909 (Beyond Answer Confidence (arXiv 2610.01006), B, S) | 0.924 (simonmesmith/jev-banking77-experiment, n=3,080) |
| CLINC150 in-scope, intent names only | 1,500 | accuracy | 0.926 (Beyond Answer Confidence (arXiv 2610.01006), B, Z) | 0.93 (nikkoxgonzales/jev-certify, n=400) |
| CLINC150 in-scope, intent names + 8 train examples per intent | 1,500 | accuracy | 0.981 (Beyond Answer Confidence (arXiv 2610.01006), B, S) | new |
| SST-2 | 872 | accuracy | 0.961 (Koa-action (arXiv 2609.36115), A, Z) | 0.9644 (Deusser et al. 2026 (arXiv 2609.37647), n=872) |
| Amazon Reviews Polarity | 5,000 | accuracy | 0.968 (Koa-action (arXiv 2609.36115), A, Z) | new |
| SST-2 (held-out from labelled validation) | 200 | accuracy | 0.935 (statsguysam/jev-classification-benchmark, B, Z) | 0.9644 (Deusser et al. 2026 (arXiv 2609.37647), n=872) |
| TREC coarse question type (6 classes) | 200 | accuracy | 0.335 (statsguysam/jev-classification-benchmark, B, Z) | 0.925 (jev-decision-bench (OmarMujahid), n=200) |
| TREC coarse question type, 4 train examples per class | 200 | accuracy | 0.855 (statsguysam/jev-classification-benchmark, B, S) | new |
| CFPB consumer complaints, 113 issue labels (bare labels) | 6,430 | accuracy | 0.3442 (earino/zero-shot-complaint-benchmark (CFPB 11, A, Z) | new |
| CFPB consumer complaints, 113 labels with form instruction + label definitions | 6,430 | accuracy | 0.4649 (earino/zero-shot-complaint-benchmark (CFPB 11, A, Z) | new |
| AG News | 7,600 | accuracy | 0.8924 (morrenhale/decision-benchmark-jev-laya-julia , A, Z) | 0.913 (onlyoneaman/jev-eval, n=300) |
| DAIR Emotion (6 classes) | 2,000 | accuracy | 0.5885 (morrenhale/decision-benchmark-jev-laya-julia , A, Z) | 0.599 (elcronos/jev-vs-open-decision-models, n=2,000) |
| BANKING77 (77 options per call) | 3,076 | accuracy | 0.8001 (morrenhale/decision-benchmark-jev-laya-julia , A, Z) | 0.891 (LangWatch Jev benchmark (release 2026-09, n=1,000) |
| MASSIVE scenario (18 classes, 51 locales x 100) | 5,100 | accuracy | 0.6767 (morrenhale/decision-benchmark-jev-laya-julia , A, Z) | 0.889 (oluies/jev-vs-spacy, n=360) |
| Gulf-Arabic government-service routing (UAE + KSA), 60 routes | 4,974 | accuracy | 0.862 (atmaneayoub/jev-ar-bench, B, Z) | new |
| BANKING77, 8-intent subset | 160 | accuracy | 0.838 (AY Automate blog (Jev vs GPT and Claude), C, Z) | 0.891 (LangWatch Jev benchmark (release 2026-09, n=1,000) |
| BANKING77, 77 intents | 231 | accuracy | 0.788 (AY Automate blog (Jev vs GPT and Claude), B, Z) | 0.891 (LangWatch Jev benchmark (release 2026-09, n=1,000) |
| BANKING77, 77 intents | 154 | accuracy | 0.76 (SOTAAZ blog (Kev vs Jev), C, Z) | 0.891 (LangWatch Jev benchmark (release 2026-09, n=1,000) |
| TREC coarse, label names | 500 | accuracy | 0.89 (SOTAAZ blog (Kev vs Jev), B, Z) | 0.925 (jev-decision-bench (OmarMujahid), n=200) |
| TREC coarse, label names + one-line descriptions | 500 | accuracy | 0.936 (SOTAAZ blog (Kev vs Jev), B, Z) | 0.925 (jev-decision-bench (OmarMujahid), n=200) |
| AG News, label names | 200 | accuracy | 0.89 (SOTAAZ blog (Kev vs Jev), B, Z) | 0.913 (onlyoneaman/jev-eval, n=300) |
| AG News, label names + descriptions | 200 | accuracy | 0.9 (SOTAAZ blog (Kev vs Jev), B, Z) | 0.913 (onlyoneaman/jev-eval, n=300) |
| CLINC150, 150 intents + out-of-scope | 400 | accuracy | 0.685 (SOTAAZ blog (Kev vs Jev), B, Z) | 0.93 (nikkoxgonzales/jev-certify, n=400) |
| MASSIVE en-US, 60 intents | 175 | accuracy | 0.829 (SOTAAZ blog (Kev vs Jev), C, Z) | 0.889 (oluies/jev-vs-spacy, n=360) |
| Twitter financial news topics (20) | 200 | accuracy | 0.71 (SOTAAZ blog (Kev vs Jev), B, Z) | 0.67 (elcronos/jev-vs-open-decision-models, n=4,117) |

**NLI / QA / knowledge**

| Dataset | n | Metric | Jev (suite, tier, track) | Earlier best, same hf_id + metric (suite, n) |
|---|---:|---|---|---|
| CUAD hard pairs (gold clause span vs same-contract BM25 distractor) | 200 | pairwise accuracy (1 - inversion rate, both orders averaged) | 0.68 (maybern-tripp-smith/cuad-jev-bench, B, Z) | new |
| MetaMedQA (USMLE-style, 6 options incl. none-of-the-above / I don't know) | 1,373 | top-1 accuracy | 0.748 (Jev in Medicine (arXiv 2609.34024), A, Z) | new |
| PubMedQA (expert-annotated official test split, yes/no/maybe) | 500 | top-1 accuracy | 0.784 (Jev in Medicine (arXiv 2609.34024), A, Z) | 0.9127 (Jevals (release 2026-09-18, suite 0.1.0), n=300) |
| DiagnosisArena-MCQ (4 options) | 915 | top-1 accuracy | 0.598 (Jev in Medicine (arXiv 2609.34024), A, Z) | new |
| NEJM Case Challenges (6 options) | 34 | top-1 accuracy | 0.618 (Jev in Medicine (arXiv 2609.34024), C, Z) | new |
| TraceSafe agent-trajectory safety (risk >= 3, revised labels) | 540 | positive-class F1 (on valid responses) | 0.678 (JEV as a Judge for Agent Trace Security (arXi, B, Z) | new |
| Four agent-trace benchmarks, unweighted mean | 5,219 | mean positive-class F1 | 0.778 (JEV as a Judge for Agent Trace Security (arXi, B, Z) | new |
| ANLI R1-R3 (dev + test) | 6,400 | accuracy | 0.7495 (Ordinal-Scale Bias in JEV-like Models (arXiv , A, Z) | 0.7394 (Deusser et al. 2026 (arXiv 2609.37647), n=3,200) |
| ContractNLI (17 hypotheses x 123 contracts, 3-way) | 2,091 | accuracy | 0.7738 (Same Scores, Different Decisions (arXiv 2609., A, Z) | new |
| TriviaQA as 4-option Choice | 9,960 | accuracy | 0.955 (Beyond Answer Confidence (arXiv 2610.01006), A, Z) | new |
| PopQA as 4-option Choice (real subjects) | 14,267 | accuracy | 0.708 (Beyond Answer Confidence (arXiv 2610.01006), A, Z) | new |
| SimpleQA Verified as 4-option Choice | 1,000 | accuracy | 0.753 (Beyond Answer Confidence (arXiv 2610.01006), A, Z) | new |
| MMLU-CF (contamination-free MMLU) | 10,000 | accuracy | 0.776 (Beyond Answer Confidence (arXiv 2610.01006), A, Z) | new |
| Daily Oracle yes/no news forecasting | 6,320 | accuracy | 0.651 (Beyond Answer Confidence (arXiv 2610.01006), A, Z) | new |
| TruthfulQA binary (correct vs the dataset's incorrect answer) | ? | accuracy | 0.909 (Beyond Answer Confidence (arXiv 2610.01006), B, Z) | 0.95 (jev-decision-bench (OmarMujahid), n=200) |
| CJ-Bench general subset (Chinese choice/noul/score decisions from held-out public Chinese  | 100,000 | accuracy | 0.6835 (Chinese-Jev (arXiv 2609.36965), B, Z) | new |
| MMLU (57 subjects) | 14,042 | accuracy | 0.8906 (JET: Justification Evaluation in Transformer , A, Z) | 0.935 (jev-decision-bench (OmarMujahid), n=200) |
| MMLU-Pro (clean accuracy before attack) | 100 | accuracy (clean) | 0.83 (JevOut: Natural Context Can Flip Decision Mod, C, Z) | 0.84 (Kev (jaredpalmer/kev README), n=?) |
| SuperGPQA (clean accuracy before attack) | 100 | accuracy (clean) | 0.46 (JevOut: Natural Context Can Flip Decision Mod, C, Z) | new |
| MuSR (clean accuracy before attack) | 100 | accuracy (clean) | 0.61 (JevOut: Natural Context Can Flip Decision Mod, C, Z) | 0.6609 (Decision Index 0.2.1, n=752) |
| ToMBench (clean accuracy before attack) | 100 | accuracy (clean) | 0.72 (JevOut: Natural Context Can Flip Decision Mod, C, Z) | new |
| LAR-ECHR (clean accuracy before attack) | 100 | accuracy (clean) | 0.8 (JevOut: Natural Context Can Flip Decision Mod, C, Z) | new |
| SATA (select-all-that-apply) (clean accuracy before attack) | 100 | accuracy (clean) | 0.823 (JevOut: Natural Context Can Flip Decision Mod, C, Z) | 0.2642 (Decision Index 0.2.1, n=1,650) |
| Cognitive Reflection Test items (Hagendorff et al. 2023 set) as 4-option Choice | 150 | fraction correct | 0.99 (Code Owns the Simulation, Jev Owns the Evalua, C, Z) | new |
| WANLI NLI (256-item subset) | 256 | accuracy | 0.793 (yehor-oleksiuk/bonzi-vs-jev-wanli256, C, Z) | 0.758 (Kev (robustness index), n=256) |
| CommonsenseQA | 100 | accuracy | 0.91 (Lightfield blog (Testing TypeSafe Jev on Text, C, Z) | 0.8821 (Deusser et al. 2026 (arXiv 2609.37647), n=1,221) |
| MMLU-CF | 100 | accuracy | 0.8 (Lightfield blog (Testing TypeSafe Jev on Text, C, Z) | new |
| RACE-H | 100 | accuracy | 0.95 (Lightfield blog (Testing TypeSafe Jev on Text, C, Z) | new |

**Judging / reward / safety / agents**

| Dataset | n | Metric | Jev (suite, tier, track) | Earlier best, same hf_id + metric (suite, n) |
|---|---:|---|---|---|
| deepset prompt-injections, text only (jev-edge injection template) | 662 | ROC-AUC | 0.983 (kiwi0719/jev-edge, A, Z) | new |
| deepset prompt-injections, with deployment context | 662 | ROC-AUC | 0.996 (kiwi0719/jev-edge, A, Z) | new |
| R-Judge agent-trajectory safety (risk >= 3, revised labels) | 564 | positive-class F1 (on valid responses) | 0.885 (JEV as a Judge for Agent Trace Security (arXi, B, Z) | new |
| ATBench500 agent-trajectory safety (risk >= 3, revised labels) | 500 | positive-class F1 (on valid responses) | 0.938 (JEV as a Judge for Agent Trace Security (arXi, B, Z) | new |
| MCPHunt agent-trajectory safety (risk >= 3, revised labels) | 3,615 | positive-class F1 (on valid responses) | 0.611 (JEV as a Judge for Agent Trace Security (arXi, A, Z) | new |
| RiceChem rubric criteria (full set) | 8,392 | verdict accuracy (Jev Choice) | 0.814 (JEV vs LLMs as Rubric Judges (arXiv 2609.2976, A, Z) | new |
| HealthBench rubric criteria (200 completions, physician labels) | 406 | verdict accuracy (Jev Choice) | 0.771 (JEV vs LLMs as Rubric Judges (arXiv 2609.2976, B, Z) | new |
| ELLIPSE essay traits (5 levels) | 1,548 | exact accuracy vs human label (Jev Choice) | 0.134 (JEV vs LLMs as Rubric Judges (arXiv 2609.2976, B, Z) | new |
| FED-Turn dialogue quality (3 levels) | 600 | exact accuracy vs human label (Jev Choice) | 0.562 (JEV vs LLMs as Rubric Judges (arXiv 2609.2976, B, Z) | new |
| FED-Dialogue quality (3 levels) | 250 | exact accuracy vs human label (Jev Choice) | 0.486 (JEV vs LLMs as Rubric Judges (arXiv 2609.2976, B, Z) | new |
| HelpSteer2 attributes (5 levels) | 360 | exact accuracy vs human label (Jev Choice) | 0.486 (JEV vs LLMs as Rubric Judges (arXiv 2609.2976, B, Z) | 0.4127 (Jevals (release 2026-09-18, suite 0.1.0), n=300) |
| LFQA answer quality (3-4 levels) | 360 | exact accuracy vs human label (Jev Choice) | 0.687 (JEV vs LLMs as Rubric Judges (arXiv 2609.2976, B, Z) | new |
| USR-TopicalChat (3 levels) | 360 | exact accuracy vs human label (Jev Choice) | 0.62 (JEV vs LLMs as Rubric Judges (arXiv 2609.2976, B, Z) | new |
| USR-PersonaChat (3 levels) | 300 | exact accuracy vs human label (Jev Choice) | 0.564 (JEV vs LLMs as Rubric Judges (arXiv 2609.2976, B, Z) | new |
| RewardBench preference pairs | 1,500 | accuracy | 0.925 (JEV-as-a-Judge (arXiv 2609.26550), A, Z) | new |
| JudgeBench (GPT split) | 350 | accuracy | 0.786 (JEV-as-a-Judge (arXiv 2609.26550), B, Z) | new |
| HaluEval QA | 3,000 | accuracy | 0.873 (JEV-as-a-Judge (arXiv 2609.26550), A, Z) | new |
| HaluEval summarization | 400 | accuracy | 0.698 (JEV-as-a-Judge (arXiv 2609.26550), B, Z) | new |
| HaluEval general (reference-free) | 200 | accuracy | 0.535 (JEV-as-a-Judge (arXiv 2609.26550), B, Z) | new |
| RM-Bench normal | 3,000 | accuracy | 0.854 (JEV-as-a-Judge (arXiv 2609.26550), A, Z) | new |
| RM-Bench hard (style-adversarial) | 3,000 | accuracy | 0.766 (JEV-as-a-Judge (arXiv 2609.26550), A, Z) | new |
| RewardBench 2 four-way | 100 | accuracy | 0.73 (JEV-as-a-Judge (arXiv 2609.26550), C, Z) | new |
| CyberSecEval Instruct, 50-way CWE classification | 1,916 | top-1 accuracy | 0.458 (JevVibe (arXiv 2609.34963), A, Z) | new |
| WAInjectBench-text prompt-injection detection | 1,612 | macro-F1 | 0.756 (Evaluating System One Models for Agent Securi, A, Z) | new |
| R-Judge interaction-risk classification | 236 | macro-F1 | 0.825 (Evaluating System One Models for Agent Securi, B, Z) | new |
| AgentHarm harmful-request classification | 352 | macro-F1 | 0.84 (Evaluating System One Models for Agent Securi, B, Z) | new |
| BFCL V4 (function selection) (clean accuracy before attack) | 100 | accuracy (clean) | 0.7 (JevOut: Natural Context Can Flip Decision Mod, C, Z) | 0.783 (LangWatch Jev benchmark (release 2026-09, n=1,000) |
| ALFWorld unseen games, closed loop (Jev chooses each admissible command) | 134 | success rate | 0.313 (Code Owns the Simulation, Jev Owns the Evalua, B, Z) | new |
| KoBBQ ambiguous (unanswerable) items with the "unknown" option | 300 | accuracy (choosing unknown) | 0.95 (jujumilk3/jev-calibration-audit, C, Z) | new |
| Agent execution decisions adapted from 10 public sources (AgentRewardBench, SWE-agent traj | 1,974 | strict accuracy (all 6 versions correct) | 0.7452 (KikiNLP/CanITrustYou-Jev, B, Z) | new |
| deepset prompt-injections (400 of 662) | 400 | accuracy | 0.87 (AY Automate blog (Jev vs GPT and Claude), B, Z) | 0.965 (Gaurav-Gosain/jev-sec-bench, n=662) |

**Retrieval / reranking**

| Dataset | n | Metric | Jev (suite, tier, track) | Earlier best, same hf_id + metric (suite, n) |
|---|---:|---|---|---|
| CUAD clause retrieval (100 category queries) | 100 | MRR | 0.917 (maybern-tripp-smith/cuad-jev-bench, C, Z) | new |
| SciFact claim-evidence relation (SUPPORTS/REFUTES/NEI) | 339 | macro-F1 | 0.8508 (CompleteDotTech/paper-package (run-20260918), B, Z) | new |
| BEIR SciFact rerank (BM25 top-20) | 300 | nDCG@10 | 0.7513 (EmreKaplaner/rag-jev, A, Z) | 0.772 (hev/reranker, n=300) |
| BEIR SciFact rerank (hybrid retrieval candidates, batch Noul) | 300 | nDCG@10 | 0.7929 (emretheus/jev-rag-benchmark, A, Z) | 0.772 (hev/reranker, n=300) |
| XQuAD-EN passage rerank | 1,190 | nDCG@10 | 0.9893 (emretheus/jev-rag-benchmark, A, Z) | new |
| BEIR SciFact (BM25 stage 1 + Jev verification pipeline) | 300 | nDCG@10 | 0.778 (romeromarcelo/jev-retrieval (jevr), B, Z) | 0.772 (hev/reranker, n=300) |
| BEIR NFCorpus (BM25 stage 1 + Jev verification pipeline) | 323 | nDCG@10 | 0.363 (romeromarcelo/jev-retrieval (jevr), B, Z) | 0.7337 (jev-decision-bench (OmarMujahid), n=60) |
| HAKARI-Bench NanoRTEB, reranking mode (14 tasks) | 2,390 | mean nDCG@10 | 0.79 (romeromarcelo/jev-retrieval (jevr), A, Z) | new |
| Amazon Reviews 2023 Movies and TV, next-item rerank of SASRec top-K hard candidates (K=20) | 954 | MRR (K=20) | 0.238 (Decision-Oriented Recommendation Reranking (a, C, Z) | new |
| Amazon Reviews 2023 Video Games, next-item rerank of SASRec top-K hard candidates (K=20) | 1,000 | MRR (K=20) | 0.295 (Decision-Oriented Recommendation Reranking (a, C, Z) | new |
| Amazon Reviews 2023 Books, next-item rerank of SASRec top-K hard candidates (K=20) | 626 | MRR (K=20) | 0.346 (Decision-Oriented Recommendation Reranking (a, C, Z) | new |
| MuSiQue multi-hop retrieval (Vector Graph RAG + Jev relation reranker) | 500 | Recall@5 | 0.6887 (zilliztech/vector-graph-rag (Jev reranker eva, C, Z) | new |
| HotpotQA multi-hop retrieval (Vector Graph RAG + Jev relation reranker) | 500 | Recall@5 | 0.935 (zilliztech/vector-graph-rag (Jev reranker eva, B, Z) | new |

**Other (tabular, sensors, networks, audio, video, domain)**

| Dataset | n | Metric | Jev (suite, tier, track) | Earlier best, same hf_id + metric (suite, n) |
|---|---:|---|---|---|
| UCI Bank Marketing (tabular) | 1,000 | balanced accuracy (raw decision) | 0.534 (QuicqDev/Jev-vs-ML (protocol V3.0.1), B, Z) | new |
| UCI Online Shoppers Intention (tabular) | 1,000 | balanced accuracy (raw decision) | 0.514 (QuicqDev/Jev-vs-ML (protocol V3.0.1), B, Z) | new |
| Breast Cancer Wisconsin diagnostic (tabular) | 114 | balanced accuracy (raw decision) | 0.61 (QuicqDev/Jev-vs-ML (protocol V3.0.1), C, Z) | new |
| Iris (tabular) | 30 | balanced accuracy (raw decision) | 0.97 (QuicqDev/Jev-vs-ML (protocol V3.0.1), C, Z) | new |
| DBLP-ACM entity matching (identity-disjoint held-out pairs) | 413 | macro-F1 | 0.9605 (CompleteDotTech/paper-package (run-20260918), B, Z) | new |
| DBLP-ACM entity matching (6 train demonstrations) | 413 | macro-F1 | 0.9859 (CompleteDotTech/paper-package (run-20260918), B, S) | new |
| JevForge-Mind2Web action choice (website-disjoint test) | 800 | choice top-1 accuracy | 0.543 (zwliJay/jev-forge (JevForge), C, Z) | 0.708 (LangWatch Jev benchmark (release 2026-09, n=2,329) |
| Phishing emails, balanced 50/50 sample | 100 | accuracy | 0.81 (trifleen/jev-vs-luna-phishing, C, Z) | new |
| typed-decisions (card eval set) | 2,000 | accuracy | 0.732 (Benchmarking System One models vs trained cla, B, Z) | 0.78 (JoeSlain/jev-gliclass-bench, n=100) |
| Conversations Gone Awry (derailment) | 500 | accuracy | 0.558 (Benchmarking System One models vs trained cla, B, Z) | new |
| Wikipedia power (admin vs non-admin) | 500 | accuracy | 0.608 (Benchmarking System One models vs trained cla, B, Z) | new |
| Wikipedia politeness (3 classes) | 498 | accuracy | 0.657 (Benchmarking System One models vs trained cla, B, Z) | new |
| RadEvalX report-pair error counts | 100 | Kendall tau vs expert total errors (Jev-OneQ) | 0.573 (Can Jev Judge Radiology Reports? (arXiv 2609., C, Z) | new |
| RadEvalExpert report-pair error counts | 624 | Kendall tau vs expert total errors (Jev-OneQ) | 0.398 (Can Jev Judge Radiology Reports? (arXiv 2609., B, Z) | new |
| ReXErr sentence error detection (all error types) | 19,513 | AUROC | 0.8896 (Can Jev Judge Radiology Reports? (arXiv 2609., A, Z) | new |
| SemEval-2026 Task 3 Track A DimABSA, Task 1 valence-arousal regression (10 corpora, 6 lang | ? | aggregate RMSE (lower is better) | 1.0645 (Decide, Don't Generate: DimABSA (arXiv 2609.3, A, S) | new |
| SemEval-2026 Task 3 Track A DimABSA, triplet extraction | ? | continuous F1 | 0.5209 (Decide, Don't Generate: DimABSA (arXiv 2609.3, A, S) | new |
| Brain-to-text candidate-sentence rescoring (ALS participant, published decoder n-best list | 978 | word error rate (lower is better) | 0.0746 (Jev for Speech-Neuroprosthesis Rescoring (arX, B, S) | new |
| WISDM accelerometer windows (class-balanced, text-described features) | 600 | macro-F1 (best representation) | 0.0378 (Training-free HAR with Jev (arXiv 2609.36154), C, Z) | new |
| UCI HAR (UCI341) accelerometer windows (class-balanced, text-described features) | 600 | macro-F1 (best representation) | 0.1184 (Training-free HAR with Jev (arXiv 2609.36154), C, Z) | new |
| PAMAP2 accelerometer windows (class-balanced, text-described features) | 600 | macro-F1 (best representation) | 0.0893 (Training-free HAR with Jev (arXiv 2609.36154), C, Z) | new |
| CESNET-QUICEXT-25, 10 application labels from first 10 packets | 52,000 | accuracy | 0.098 (Jev for Network Traffic Classification (arXiv, A, Z) | new |
| CESNET-QUICEXT-25 with 40 fixed labelled examples in context | 52,000 | accuracy | 0.2842 (Jev for Network Traffic Classification (arXiv, A, S) | new |
| NSL-KDD flows, 2,000-flow evaluation set | 2,000 | F1 (attack class) | 0.782 (Jev-IDS (arXiv 2610.01079), A, Z) | 0.859 (jev-ids/jev-ids, n=900) |
| AVT-VQDB-UHD-1 subjective quality, metadata only | ? | PLCC vs MOS | 0.747 (JEVQA video quality (arXiv 2609.24395), B, Z) | new |
| JevBench v1.4.2.2 public set | ? | accuracy | 0.866 (LLM2Jev (arXiv 2610.02076), B, Z) | 0.8571 (WebJev (lexmount) single-step benchmarks, n=?) |
| Sys1Cal-v1 known-probability True/False items, Noul primitive | 365 | mean OVL soft accuracy (1 - TV) | 0.918 (Sys1Cal-v1 (arXiv 2609.35342), C, Z) | new |
| XD-Violence caption anchors (text-mediated) | 200 | average precision (Jev Noul) | 0.7599 (Decision Readouts for Video Anomaly Detection, C, Z) | new |
| Breast Cancer Wisconsin diagnostic (tabular) | 114 | accuracy | 0.842 (statsguysam/jev-classification-benchmark, C, Z) | new |
| Wine (tabular, 3 classes) | 36 | accuracy | 0.333 (statsguysam/jev-classification-benchmark, C, Z) | new |
| typed-decisions test parquet | 2,000 | accuracy | 0.732 (morrenhale/decision-benchmark-jev-laya-julia , B, Z) | 0.78 (JoeSlain/jev-gliclass-bench, n=100) |
| Amazon QA answerability yes/no (Noul) | 500 | accuracy | 0.628 (Lightfield blog (Testing TypeSafe Jev on Text, B, Z) | new |
| Persuasion for Good: did they donate? (Noul) | 500 | accuracy | 0.712 (Lightfield blog (Testing TypeSafe Jev on Text, B, Z) | new |
| Persuasion for Good: persuasion strategy, 18-way (Choice) | 500 | accuracy | 0.442 (Lightfield blog (Testing TypeSafe Jev on Text, B, Z) | new |
| CraigslistBargain: reached a deal? (Noul) | 500 | accuracy | 0.918 (Lightfield blog (Testing TypeSafe Jev on Text, B, Z) | new |


---

## 3. Conflicting numbers for the same dataset, and why

### 3.1 Protocol explanations for the large discrepancies

| Dataset | Published Jev range | Main reasons |
|---|---|---|
| **BANKING77** | 0.712 – 0.832 zero-shot; 0.853 / 0.924 with train data | (a) **Label text.** Bare intent codes, such as DMB's `{label: label}`, give 0.792. Definitions (manojlds) give 0.801. Underscores turned into spaces (chepyle) give 0.806. A two-step group-then-label setup with descriptions (zhuyansen) gives 0.712, and bare names cost about −0.13 there. Rewritten descriptions add +0.015 (dhruvmehra). (b) **Label set.** BTZSC's 72 hypotheses with 200 no-positive rows dropped give 0.870 on n=100. (c) **Split mirror.** `mteb/banking77` has 3,076 rows against 3,080 in the PolyAI original. (d) **Samples.** n=77 to 500 gives about ±4–9 pts, so 0.76–0.83 is mostly noise. (e) **Train examples in the request.** 2 per label gives 0.853; 24 BM25-retrieved give 0.924. Full-test zero-shot runs cluster tightly: 0.792 / 0.797 / 0.799 / 0.800 / 0.801 / 0.806. Treat 0.80 ± 0.01 as Jev's BANKING77 accuracy. |
| **CLINC150** | acc 0.886 – 0.917; OOS recall 0.776 – 0.881 | (a) **How OOS is offered.** A plain `oos` label (Deußer) gives recall 0.776. An option described as "out of scope (fits none of the other intents)" (chepyle) gives 0.881. DMB is in between at 0.812. (b) **In-scope-only runs** with no OOS option (thisisandreeeee, 4,500 rows) give 0.920. (c) The 0.917 is partial: 4,060 of 5,500 rows, via a reseller endpoint. |
| **AG News** | 0.843 – 0.913 | Descriptions vs bare labels. Batching 20 texts per call (zhuyansen, 0.865). n=100–300 samples give 0.87–0.91. Full test: 0.885. |
| **DAIR Emotion** | acc 0.480 – 0.599 | BTZSC hypotheses on n=100 give 0.480. Bare names on the full 2,000 give 0.585 / 0.587. One-line definitions give 0.599. Jev often puts exactly 0 on the true label: 16% of items in one study. That is why its ECE (0.28–0.35) and NLL are poor even where accuracy is fine. |
| **HellaSwag** | 0.861 – 0.955 | Vercel AI Gateway `evaluate` with a different template (scienthoon, 2,000) gives 0.861. Deußer's template on the full validation set gives 0.955, and DI gives 0.945. The endpoint and template, not the model, explain the 9 pts. |
| **UNFAIR-ToS** | micro-F1 0.499 / 0.764 / 0.748 / 0.919 | **Different metrics under the same name.** Deußer scores 8 Nouls on **positives only**, giving 0.499; with thresholds tuned on dev it gives 0.748. LexGLUE's official convention (chepyle) adds a synthetic **"none" class** for the roughly 9 in 10 clauses with no label, which inflates micro-F1 to 0.764, and per-label tuning gives 0.919. Never compare across these two conventions. |
| **deepset prompt-injections** | test-116 acc 0.741 … all-662 acc 0.965 | Deußer uses a plain Noul at 0.5 on test-116: acc 0.741, recall 0.50, AUROC 0.982. So the ranking is excellent and the threshold is wrong. Adding one sentence of deployment context (Gaurav-Gosain) gives acc 0.965. Threshold 0.1 gives F1 0.938, against 0.813 at 0.5 (switchboard). ca7ai's false-negative rate of 47% is a "label-mismatch upper bound". **Threshold and framing dominate.** Report AUROC/AUPRC alongside accuracy. |
| **PubMedQA** | 0.787 vs 0.913 | Deußer: 3-way yes/no/maybe on all 1,000 `pqa_labeled`. Jevals: binary Noul on 300 items with "maybe" excluded (prior 0.62). These are different tasks. |
| **GoEmotions** | macro-F1 0.243 / 0.353; acc 0.282 | Multi-label with 28 Nouls (Deußer; 0.353 with tuned thresholds) vs single-label "primary emotion" Choice (jev-bench, 0.282). Different tasks. |
| **HelpSteer2** | ρ 0.412; acc 0.36 – 0.41; ρ 0.490 | Deußer: mean Spearman over 5 attributes on 1,038 rows. Jevals: helpfulness exact-level accuracy on 300 rows, 0.413 against a 0.417 prior. jev-bench: helpfulness accuracy on 1,000 rows, 0.363. OmarMujahid: helpfulness Spearman on 200 rows spread across levels, 0.490. |
| **ANLI** | acc 0.739; macro-F1 0.748; R3 0.66 – 0.677 | Same 3,200 items, different metric (Deußer accuracy vs DI macro-F1). R3 alone: 0.677 (Deußer subset) and 0.66 (OmarMujahid, balanced 200). |
| **LLM-AggreFact** | 0.786 / 0.831 / 0.733 | The per-source mean balanced accuracy is 0.786. Pooled over 29,320 items it is 0.831. adorosario's 495-claim MiniCheck subset gives 0.733. The public leaderboard uses the per-source mean. |
| **ToxicChat** | F1 0.757 – 0.793 | Deußer test 5,083 at threshold 0.5: 0.786; with a dev-tuned threshold of 0.42: 0.793. mbburabak 2,853 cases with a review band: 0.757. |
| **typed-decisions** | 0.727 / 0.739 / 0.7405 | Maintainer run, all 2,000 decisions, whole case per request: 0.727. LangWatch on 1,965 decisions: 0.739. WebJev rerun: 0.7405. Request shape matters: noul accuracy is 0.843 with one question per request and 0.788 when batched. |
| **SMS spam** | 0.930 – 0.980 | The metrics differ: Deußer F1 0.938 on all 5,574; accuracy 0.93–0.98 elsewhere; balanced accuracy 0.961. They are not comparable. |
| **Kev transfer-v4** | 0.855 / 0.857 | chepyle ran a 656-item dev version; the Kev README uses 764. |
| **DI vs Deußer on shared sets** | within 0.01 | WinoGrande 0.9195 vs 0.914; HellaSwag 0.945 vs 0.955; MMLU 0.917 vs 0.918; CSQA 0.875 vs 0.882. Same items, different templates. This ±1 pt band is the template noise floor for Jev on public sets. |
| **MMLU (resume pass)** | 0.8906 / 0.9173 / 0.9181 | JET (arXiv 2609.33874) measured Jev 1.13 itself on the full 14,042-item test through the native decision interface and got 0.8906, about 2.7 pts below Deußer (0.9181) and DI (0.9173). The paper does not print its request template, so the cause is unverified. Use 0.918 as the bar; treat 0.891 as a template-sensitivity warning. |
| **PubMedQA (resume pass)** | 0.784 / 0.787 / 0.913 | Jev in Medicine: 3-way on the 500-item official test split, 0.784. Deußer: 3-way on all 1,000 `pqa_labeled`, 0.787. Jevals: binary, 0.913. The two 3-way numbers agree. |
| **TREC coarse (resume pass)** | 0.335 – 0.936 | statsguysam (200 items) reports 0.335 zero-shot and 0.855 with 4 examples per class; SOTAAZ (500) gives 0.890 with label names and 0.936 with one-line descriptions; OmarMujahid (200) 0.925. The 0.335 is an outlier, probably a harness or label-mapping problem (not verified). Bar: 0.936 (descriptions, 500). |
| **BANKING77 (resume pass)** | 0.760 – 0.821 zero-shot; 0.819 – 0.924 with train data | New full-test run (morrenhale, 3,076) 0.8001, so full-test zero-shot stays at 0.80 ± 0.01. Small samples (134–231) give 0.76–0.81. Intent names only on a 770 sample: 0.821. With 8 train examples per intent in the option descriptions: 0.909. |
| **CLINC150 (resume pass)** | 0.685 – 0.926 | In-scope balanced samples: 0.910 (1,000) and 0.926 (1,500, names). With an OOS option on 800 (600 + 200 OOS): 0.913. SOTAAZ (400 items, 150 intents + OOS) is the outlier at 0.685; its OOS share is not stated. Hierarchical routing (parent first, then child) drops Jev to 0.681. |
| **deepset prompt-injections (resume pass)** | AUROC 0.982 – 0.996 | AUROC is stable across studies: 0.982 (Deußer, test 116), 0.983 (jev-edge, bare, 662), 0.990 (AY Automate, 400; switchboard, 662), 0.993 (jev-sec-bench), 0.996 (jev-edge with a deployment description). Accuracy and F1 swing with threshold and framing (§3.1 above). Compare on AUROC. |
| **R-Judge (resume pass)** | positive F1 0.885 vs macro-F1 0.825 | arXiv 2609.34862 scores all 564 trajectories with revised labels (positive-class F1). arXiv 2609.33401 scores a 236-item test partition (macro-F1). Different labels, partition and metric. |
| **MMLU-CF (resume pass)** | 0.776 vs 0.80 | 10,000 items (arXiv 2610.01006) vs a 100-item sample (Lightfield). Use 0.776. |
| **SST-2 (resume pass)** | 0.935 – 0.9644 | Full 872 validation: 0.9644 (Deußer), 0.961 (Koa-action). 200-item sample: 0.935 (statsguysam). |
| **AG News (resume pass)** | 0.8851 vs 0.8924 | Both on the full 7,600 test. Deußer 0.8851, morrenhale 0.8924. Same items, different wording: the ±1 pt template noise floor again. |
| **ANLI (resume pass)** | 0.7394 vs 0.7495 | Deußer: test R1–R3 (3,200). Ordinal-bias paper: dev + test (6,400). |
| **BEIR SciFact rerank (resume pass)** | nDCG@10 0.751 – 0.793 | The first-stage candidates decide it: BM25 top-20 gives 0.751; BM25 top-100 0.770; BM25 top-30 per pair 0.772; a BM25 + verification pipeline 0.778; hybrid retrieval candidates 0.793. Fix the candidate set before comparing. |
| **RewardBench / RM-Bench (resume pass)** | 0.925 vs 0.9258; 0.854 / 0.766 vs 0.813 | JEV-as-a-Judge uses pairwise accuracy on samples (1,500 RewardBench; 3,000 RM-Bench normal / hard). goya4140 reports the official section macros on the full sets. The metrics differ, though the RewardBench numbers agree. |
| **typed-decisions (resume pass)** | 0.727 / 0.732 / 0.739 / 0.7405 | Two new full 2,000-decision runs (arXiv 2610.00346; morrenhale) both give 0.732. |

### 3.2 Every dataset with Jev numbers from two or more suites (generated)

Same key = same upstream data. **Mixed metrics inside one row are flagged by the metric column. Do not average them.**

| Canonical dataset | # results | Jev range (same metric family) | Rows (suite: score, n, metric) |
|---|---:|---|---|
| banking77 | 25 | 0.712 - 0.832 | Deusser et al. 2026: 0.7971 (n=3076, accuracy); DMB: 0.792 (n=3080, accuracy); DMB: 0.805 (n=77, accuracy); DMB: 0.831 (n=77, accuracy); DMB: 0.763 (n=900, accuracy); Jevals: 0.7967 (n=300, accuracy); LangWatch Jev benchmark: 0.796 (n=500, accuracy); jev-bench: 0.796 (n=1000, accuracy); jev-decision-bench: 0.815 (n=200, accuracy); zhuyansen/jev-zeroshot-vs-bert: 0.712 (n=1000, accuracy); thisisandreeeee/jev-benchmarks: 0.799 (n=3080, accuracy); manojlds/classifier-bench: 0.801 (n=3080, accuracy); Alexander-Ollman/laya-ft: 0.8 (n=3080, accuracy); chepyle/jev-test: 0.806 (n=3080, accuracy); 4esv/jev-eval: 0.78 (n=300, accuracy); dhruvmehra/jevbench: 0.764 (n=500, accuracy); onlyoneaman/jev-eval: 0.76 (n=300, accuracy); mugenkyou/JEV-VS-ML: 0.789 (n=1500, accuracy); ickma2311/jev-baselines-eval: 0.832 (n=208, accuracy); Kumzha/jev-benchmark: 0.815 (n=200, accuracy); MohtashamMurshid/jev-speed-test: 0.81 (n=500, accuracy); saurabhkumar8112/jev-gpt5-routing-study: 0.832 (n=500, accuracy); cocodedk/jev-bench: 0.779 (n=308, accuracy); FirasSX914/Janus: 0.778 (n=500, accuracy); shivpratapsinghpanwar/edgefront_jev: 0.79 (n=300, accuracy) |
| ag_news | 8 | 0.843 - 0.913 | Deusser et al. 2026: 0.8851 (n=7600, accuracy); jev-decision-bench: 0.87 (n=200, accuracy); AbdelStark/jev-benchmarks: 0.91 (n=100, accuracy); zhuyansen/jev-zeroshot-vs-bert: 0.865 (n=1000, accuracy); dhruvmehra/jevbench: 0.843 (n=500, accuracy); onlyoneaman/jev-eval: 0.913 (n=300, accuracy); mugenkyou/JEV-VS-ML: 0.875 (n=1000, accuracy); dshvimer/jev-eval: 0.87 (n=100, accuracy) |
| sst2 | 6 | 0.945 - 0.9644 | Deusser et al. 2026: 0.9644 (n=872, accuracy); zhuyansen/jev-zeroshot-vs-bert: 0.96 (n=872, accuracy); thisisandreeeee/jev-benchmarks: 0.945 (n=872, accuracy); dhruvmehra/jevbench: 0.954 (n=500, accuracy); onlyoneaman/jev-eval: 0.957 (n=300, accuracy); actuallyrizzn/decision-systems-bakeoff: 0.946 (n=872, accuracy) |
| clinc150 | 6 | 0.87 - 0.917 | Deusser et al. 2026: 0.8945 (n=5500, accuracy); DMB: 0.886 (n=5500, accuracy); jev-bench: 0.893 (n=1000, accuracy); 4esv/jev-eval: 0.897 (n=300, accuracy); actuallyrizzn/decision-systems-bakeoff: 0.917 (n=4060, accuracy); ickma2311/jev-baselines-eval: 0.87 (n=200, accuracy) |
| dair_emotion | 5 | 0.48 - 0.599 | Deusser et al. 2026: 0.585 (n=2000, accuracy); jev-decision-bench: 0.505 (n=200, accuracy); AbdelStark/jev-benchmarks: 0.48 (n=100, accuracy); elcronos/jev-vs-open-decision-models: 0.587 (n=2000, accuracy); elcronos/jev-vs-open-decision-models: 0.599 (n=2000, accuracy) |
| sms_spam | 5 | 0.93 - 0.98 | Deusser et al. 2026: 0.9381 (n=5574, f1); DMB: 0.98 (n=50, accuracy); DMB: 0.93 (n=900, accuracy); jev-bench: 0.965 (n=800, accuracy); mugenkyou/JEV-VS-ML: 0.961 (n=1000, balanced accuracy) |
| mmlu | 5 | 0.9 - 0.935 | Deusser et al. 2026: 0.9181 (n=14042, accuracy); Decision Index 0.2.1: 0.9173 (n=14042, accuracy); jev-bench: 0.923 (n=1000, accuracy); jev-decision-bench: 0.935 (n=200, accuracy); Kev: 0.9 (n=?, accuracy) |
| hellaswag | 4 | 0.861 - 0.9549 | Deusser et al. 2026: 0.9549 (n=10042, accuracy); Decision Index 0.2.1: 0.9452 (n=10042, accuracy); jev-decision-bench: 0.95 (n=200, accuracy); scienthoon/jev-ood-calibration: 0.861 (n=2000, accuracy) |
| commonsense_qa | 4 | 0.87 - 0.8821 | Deusser et al. 2026: 0.8821 (n=1221, accuracy); Decision Index 0.2.1: 0.8747 (n=1221, accuracy); jev-decision-bench: 0.87 (n=200, accuracy); scienthoon/jev-ood-calibration: 0.881 (n=1221, accuracy) |
| imdb | 3 | 0.963 - 0.97 | Deusser et al. 2026: 0.9652 (n=25000, accuracy); 4esv/jev-eval: 0.97 (n=300, accuracy); mugenkyou/JEV-VS-ML: 0.963 (n=1000, balanced accuracy) |
| paws | 3 | 0.846 - 0.855 | Deusser et al. 2026: 0.8499 (n=8000, accuracy); jev-bench: 0.846 (n=1000, accuracy); zhuyansen/jev-zeroshot-vs-bert: 0.855 (n=1000, accuracy) |
| winogrande | 3 | 0.885 - 0.9195 | Deusser et al. 2026: 0.914 (n=1267, accuracy); Decision Index 0.2.1: 0.9195 (n=1267, accuracy); jev-decision-bench: 0.885 (n=200, accuracy) |
| sst5 | 3 | 0.565 - 0.5792 | Deusser et al. 2026: 0.5792 (n=2210, argmax_accuracy); jev-bench: 0.565 (n=1000, accuracy); 4esv/jev-eval: 0.57 (n=300, accuracy) |
| clinc150_oos_recall | 3 | 0.776 - 0.881 | Deusser et al. 2026: 0.776 (n=5500, oos_recall); DMB: 0.812 (n=1000, oos_recall); chepyle/jev-test: 0.881 (n=1000, oos_recall) |
| clinc150_inscope | 3 | 0.89 - 0.93 | Deusser et al. 2026: 0.9209 (n=5500, in_scope_accuracy); chepyle/jev-test: 0.89 (n=5500, in-scope accuracy); nikkoxgonzales/jev-certify: 0.93 (n=400, in-scope top-1 accuracy) |
| arc_challenge | 3 | 0.975 - 0.979 | Decision Index 0.2.1: 0.9778 (n=1172, accuracy); jev-bench: 0.979 (n=1000, accuracy); jev-decision-bench: 0.975 (n=200, accuracy) |
| email_spam | 3 | 0.9743 - 0.9864 | Decision Index 0.2.1: 0.9743 (n=3000, accuracy); bitnovus/jev-spam-eval: 0.9864 (n=5733, accuracy); Arize blog: 0.983 (n=18514, accuracy) |
| mmlu_pro | 3 | 0.827 - 0.84 | Decision Index 0.2.1: 0.827 (n=12032, accuracy); Kev: 0.84 (n=?, accuracy); WebJev: 0.834 (n=1000, accuracy) |
| typed_decisions | 3 | 0.727 - 0.7405 | typed-decisions: 0.727 (n=400, accuracy (argmax vs gold arg); LangWatch Jev benchmark: 0.739 (n=1965, accuracy); WebJev: 0.7405 (n=?, accuracy) |
| language_id | 2 | 0.99 - 0.9964 | Deusser et al. 2026: 0.9964 (n=10000, accuracy); jev-decision-bench: 0.99 (n=200, accuracy) |
| llm_aggrefact | 2 | 0.733 - 0.7858 | Deusser et al. 2026: 0.7858 (n=29320, mean_balanced_accuracy); adorosario/jev-rag-claim-verification: 0.733 (n=495, balanced accuracy) |
| boolq | 2 | 0.9131 - 0.917 | Deusser et al. 2026: 0.9131 (n=3270, accuracy); jev-bench: 0.917 (n=1000, accuracy) |
| toxic_chat | 2 | 0.7572 - 0.7862 | Deusser et al. 2026: 0.7862 (n=5083, f1); mbburabak/jev-safety-benchmark: 0.7572 (n=2853, F1 (harmful class)) |
| stsb | 2 | 0.8902 - 0.8921 | Deusser et al. 2026: 0.8902 (n=1379, spearman); thisisandreeeee/jev-benchmarks: 0.8921 (n=1379, Spearman) |
| banking77_macrof1 | 2 | 0.7883 - 0.7974 | Deusser et al. 2026: 0.7883 (n=3076, macro_f1); Decision Index 0.2.1: 0.7974 (n=3080, macro-F1) |
| dair_emotion_macrof1 | 2 | 0.497 - 0.4987 | Deusser et al. 2026: 0.4987 (n=2000, macro_f1); Alexander-Ollman/laya-ft: 0.497 (n=1000, macro-F1) |
| gsm8k_mc | 2 | 0.725 - 0.7987 | Decision Index 0.2.1: 0.7987 (n=1319, accuracy); jev-decision-bench: 0.725 (n=200, accuracy) |
| amazon_esci | 2 | 0.5521 - 0.577 | Decision Index 0.2.1: 0.5521 (n=5000, macro-F1); LangWatch Jev benchmark: 0.577 (n=1000, accuracy) |
| openbookqa | 2 | 0.94 - 0.942 | Decision Index 0.2.1: 0.94 (n=500, accuracy); scienthoon/jev-ood-calibration: 0.942 (n=500, accuracy) |
| phishnchips | 2 | 0.6255 - 0.626 | Decision Index 0.2.1: 0.6255 (n=2000, accuracy); anisselbd/jev-phishing-bench: 0.626 (n=2000, accuracy) |
| helpsteer2_helpfulness | 2 | 0.363 - 0.4127 | Jevals: 0.4127 (n=300, accuracy); jev-bench: 0.363 (n=1000, accuracy) |
| jevbench_praveen | 2 | 0.623 - 0.733 | LangWatch Jev benchmark: 0.623 (n=1200, accuracy); jev-bench: 0.733 (n=22773, macro accuracy) |
| jevbench_public | 2 | 0.857 - 0.8571 | LangWatch Jev benchmark: 0.857 (n=231, accuracy); WebJev: 0.8571 (n=?, accuracy) |
| ledgar | 2 | 0.751 - 0.753 | jev-bench: 0.751 (n=1000, accuracy); chepyle/jev-test: 0.753 (n=10000, micro-F1) |
| mnli | 2 | 0.84 - 0.883 | jev-bench: 0.883 (n=1000, accuracy); jev-decision-bench: 0.84 (n=200, accuracy) |
| banking77_fewshot | 2 | 0.8529 - 0.924 | manojlds/classifier-bench: 0.8529 (n=3080, accuracy); simonmesmith/jev-banking77-experiment: 0.924 (n=3080, accuracy) |
| kev_transfer-v4_dev | 2 | 0.855 - 0.857 | Kev suites: 0.855 (n=656, accuracy); Kev: 0.857 (n=764, accuracy) |
| deepset_prompt_injections_all | 2 | 0.938 - 0.965 | Gaurav-Gosain/jev-sec-bench: 0.965 (n=662, accuracy); aniruddh-krovvidi/switchboard: 0.938 (n=662, F1 @ threshold 0.1) |
| beir_scifact | 2 | 0.7699 - 0.772 | denser-org/rerank-bench-jev: 0.7699 (n=300, nDCG@10); hev/reranker: 0.772 (n=300, nDCG@10) |
| beir_nfcorpus | 2 | 0.358 - 0.3623 | denser-org/rerank-bench-jev: 0.3623 (n=323, nDCG@10); hev/reranker: 0.358 (n=323, nDCG@10) |

### 3.3 Datasets that gained rows in the resume pass (generated, grouped by `hf_id`)

Grouped by upstream dataset only. **Metrics are mixed inside a row; compare only like with like.** New rows are marked ★.

| hf_id | # rows | Rows (suite: score, n, metric) |
|---|---:|---|
| banking77 | 41 | Deusser et al. 2026: 0.7971 (n=3,076, accuracy); Deusser et al. 2026: 0.7883 (n=3,076, macro_f1); Decision Index 0.2.1: 0.7974 (n=3,080, macro-F1); DMB: 0.792 (n=3,080, accuracy); DMB: 0.805 (n=77, accuracy); DMB: 0.831 (n=77, accuracy); DMB: 0.763 (n=900, accuracy); DMB: 0.767 (n=900, accuracy); Jevals: 0.7967 (n=300, accuracy); Jevals: 67.78 (n=300, Decision Score (0-100; 0 = label-prior, ); LangWatch Jev benchmark: 0.891 (n=1,000, accuracy); LangWatch Jev benchmark: 0.796 (n=500, accuracy); jev-decision-bench: 0.815 (n=200, accuracy); zhuyansen/jev-zeroshot-vs-bert: 0.712 (n=1,000, accuracy); manojlds/classifier-bench: 0.801 (n=3,080, accuracy); manojlds/classifier-bench: 0.8529 (n=3,080, accuracy); simonmesmith/jev-banking77-experiment: 0.924 (n=3,080, accuracy); Alexander-Ollman/laya-ft: 0.8 (n=3,080, accuracy); chepyle/jev-test: 0.806 (n=3,080, accuracy); 4esv/jev-eval: 0.78 (n=300, accuracy); dhruvmehra/jevbench: 0.764 (n=500, accuracy); onlyoneaman/jev-eval: 0.76 (n=300, accuracy); mugenkyou/JEV-VS-ML: 0.789 (n=1,500, accuracy); ickma2311/jev-baselines-eval: 0.832 (n=208, accuracy); Kumzha/jev-benchmark: 0.815 (n=200, accuracy); MohtashamMurshid/jev-speed-test: 0.81 (n=500, accuracy); saurabhkumar8112/jev-gpt5-routing-study: 0.832 (n=500, accuracy); cocodedk/jev-bench: 0.779 (n=308, accuracy); FirasSX914/Janus: 0.778 (n=500, accuracy); shivpratapsinghpanwar/edgefront_jev: 0.79 (n=300, accuracy); manjunathshiva/jev-frontier-bench: 0.725 (n=200, accuracy); ★QuicqDev/Jev-vs-ML: 0.789 (n=1,500, balanced accuracy (raw decision)); ★QuicqDev/Jev-vs-ML: 0.819 (n=1,500, balanced accuracy); ★rupeshpoojary9/poorjev: 0.812 (n=154, accuracy); ★abhisheksharma001/jev-skill: 0.761 (n=134, accuracy (first-draft question)); ★Beyond Answer Confidence: 0.821 (n=770, accuracy); ★Beyond Answer Confidence: 0.909 (n=770, accuracy); ★morrenhale/decision-benchmark-jev-laya-j: 0.8001 (n=3,076, accuracy); ★AY Automate blog: 0.838 (n=160, accuracy); ★AY Automate blog: 0.788 (n=231, accuracy); ★SOTAAZ blog: 0.76 (n=154, accuracy) |
| clincoos | 18 | Deusser et al. 2026: 0.8945 (n=5,500, accuracy); Deusser et al. 2026: 0.776 (n=5,500, oos_recall); Deusser et al. 2026: 0.9209 (n=5,500, in_scope_accuracy); DMB: 0.886 (n=5,500, accuracy); DMB: 0.812 (n=1,000, oos_recall); LangWatch Jev benchmark: 0.934 (n=1,000, balanced accuracy); chepyle/jev-test: 0.89 (n=5,500, in-scope accuracy); chepyle/jev-test: 0.881 (n=1,000, oos_recall); 4esv/jev-eval: 0.897 (n=300, accuracy); actuallyrizzn/decision-systems-bakeoff: 0.917 (n=4,060, accuracy); ickma2311/jev-baselines-eval: 0.87 (n=200, accuracy); nikkoxgonzales/jev-certify: 0.93 (n=400, in-scope top-1 accuracy); gazelle93/decision-models-under-pressure: 0.6 (n=?, accuracy at 128 candidates); ★Benchmarking System One models vs traine: 0.913 (n=800, accuracy); ★Do System One Decisions Add Up?: 0.91 (n=1,000, accuracy (flat fine-label)); ★Beyond Answer Confidence: 0.926 (n=1,500, accuracy); ★Beyond Answer Confidence: 0.981 (n=1,500, accuracy); ★SOTAAZ blog: 0.685 (n=400, accuracy) |
| agnews | 12 | Deusser et al. 2026: 0.8851 (n=7,600, accuracy); jev-decision-bench: 0.87 (n=200, accuracy); jev-decision-bench: 0.845 (n=200, accuracy); zhuyansen/jev-zeroshot-vs-bert: 0.865 (n=1,000, accuracy); dhruvmehra/jevbench: 0.843 (n=500, accuracy); onlyoneaman/jev-eval: 0.913 (n=300, accuracy); mugenkyou/JEV-VS-ML: 0.875 (n=1,000, accuracy); dshvimer/jev-eval: 0.87 (n=100, accuracy); ★QuicqDev/Jev-vs-ML: 0.875 (n=1,000, balanced accuracy (raw decision)); ★morrenhale/decision-benchmark-jev-laya-j: 0.8924 (n=7,600, accuracy); ★SOTAAZ blog: 0.89 (n=200, accuracy); ★SOTAAZ blog: 0.9 (n=200, accuracy) |
| typeddecisions | 10 | typed-decisions: 0.727 (n=400, accuracy (argmax vs gold argmax, 2,000 d); typed-decisions: 0.775 (n=?, accuracy (noul questions)); typed-decisions: 0.72 (n=?, accuracy (choice questions)); typed-decisions: 0.696 (n=?, accuracy (score questions)); typed-decisions: 1.442 (n=400, KL from gold (lower is better)); LangWatch Jev benchmark: 0.739 (n=1,965, accuracy); WebJev: 0.7405 (n=?, accuracy); JoeSlain/jev-gliclass-bench: 0.78 (n=100, accuracy); ★Benchmarking System One models vs traine: 0.732 (n=2,000, accuracy); ★morrenhale/decision-benchmark-jev-laya-j: 0.732 (n=2,000, accuracy) |
| sst2 | 9 | Deusser et al. 2026: 0.9644 (n=872, accuracy); zhuyansen/jev-zeroshot-vs-bert: 0.96 (n=872, accuracy); thisisandreeeee/jev-benchmarks: 0.945 (n=872, accuracy); dhruvmehra/jevbench: 0.954 (n=500, accuracy); onlyoneaman/jev-eval: 0.957 (n=300, accuracy); dylantom2012 / zhlei07 open-system-one: 0.793 (n=10,000, macro accuracy); actuallyrizzn/decision-systems-bakeoff: 0.946 (n=872, accuracy); ★Koa-action: 0.961 (n=872, accuracy); ★statsguysam/jev-classification-benchmark: 0.935 (n=200, accuracy) |
| promptinjections | 9 | Deusser et al. 2026: 0.7414 (n=116, accuracy); LangWatch Jev benchmark: 0.946 (n=1,000, catch rate @ 5% false alarms); Alexander-Ollman/laya-ft: 0.787 (n=116, macro-F1); Gaurav-Gosain/jev-sec-bench: 0.965 (n=662, accuracy); aniruddh-krovvidi/switchboard: 0.938 (n=662, F1 @ threshold 0.1); ca7ai/jev-prompt-sentry: 0.473 (n=546, false-negative rate (lower is better)); ★kiwi0719/jev-edge: 0.983 (n=662, ROC-AUC); ★kiwi0719/jev-edge: 0.996 (n=662, ROC-AUC); ★AY Automate blog: 0.87 (n=400, accuracy) |
| emotion | 8 | Deusser et al. 2026: 0.585 (n=2,000, accuracy); Deusser et al. 2026: 0.4987 (n=2,000, macro_f1); jev-decision-bench: 0.505 (n=200, accuracy); elcronos/jev-vs-open-decision-models: 0.587 (n=2,000, accuracy); elcronos/jev-vs-open-decision-models: 0.599 (n=2,000, accuracy); Alexander-Ollman/laya-ft: 0.497 (n=1,000, macro-F1); ★Benchmarking System One models vs traine: 0.504 (n=498, accuracy); ★morrenhale/decision-benchmark-jev-laya-j: 0.5885 (n=2,000, accuracy) |
| smsspam | 7 | Deusser et al. 2026: 0.9381 (n=5,574, f1); DMB: 0.98 (n=50, accuracy); DMB: 0.93 (n=900, accuracy); jev-decision-bench: 0.9996 (n=200, auroc); Alexander-Ollman/laya-ft: 0.908 (n=1,000, macro-F1); mugenkyou/JEV-VS-ML: 0.961 (n=1,000, balanced accuracy); ★QuicqDev/Jev-vs-ML: 0.961 (n=1,000, balanced accuracy (raw decision)) |
| fstandhartingerjevbench | 6 | JevBench v1.5.5: 72.13 (n=1,624, JevBench score (0-100)); JevBench v1.5.5: 72 (n=1,624, Intelligence (0-100)); JevBench v1.5.5: 88.03 (n=1,624, Calibration (0-100)); LangWatch Jev benchmark: 0.857 (n=231, accuracy); WebJev: 0.8571 (n=?, accuracy); ★LLM2Jev: 0.866 (n=?, accuracy) |
| trec | 6 | jev-decision-bench: 0.925 (n=200, accuracy); ★Do System One Decisions Add Up?: 0.722 (n=500, accuracy (flat fine-label)); ★statsguysam/jev-classification-benchmark: 0.335 (n=200, accuracy); ★statsguysam/jev-classification-benchmark: 0.855 (n=200, accuracy); ★SOTAAZ blog: 0.89 (n=500, accuracy); ★SOTAAZ blog: 0.936 (n=500, accuracy) |
| scifact | 6 | denser-org/rerank-bench-jev: 0.7699 (n=300, nDCG@10); hev/reranker: 0.772 (n=300, nDCG@10); ★CompleteDotTech/paper-package: 0.8508 (n=339, macro-F1); ★EmreKaplaner/rag-jev: 0.7513 (n=300, nDCG@10); ★emretheus/jev-rag-benchmark: 0.7929 (n=300, nDCG@10); ★romeromarcelo/jev-retrieval: 0.778 (n=300, nDCG@10) |
| mmlu | 5 | Deusser et al. 2026: 0.9181 (n=14,042, accuracy); Decision Index 0.2.1: 0.9173 (n=14,042, accuracy); jev-decision-bench: 0.935 (n=200, accuracy); Kev: 0.9 (n=?, accuracy); ★JET: Justification Evaluation in Transfo: 0.8906 (n=14,042, accuracy) |
| commonsenseqa | 5 | Deusser et al. 2026: 0.8821 (n=1,221, accuracy); Decision Index 0.2.1: 0.8747 (n=1,221, accuracy); jev-decision-bench: 0.87 (n=200, accuracy); scienthoon/jev-ood-calibration: 0.881 (n=1,221, accuracy); ★Lightfield blog: 0.91 (n=100, accuracy) |
| helpsteer2 | 5 | Deusser et al. 2026: 0.4122 (n=1,038, mean_spearman); Jevals: 0.4127 (n=300, accuracy); Jevals: 9.2 (n=300, Decision Score (0-100; 0 = label-prior, ); jev-decision-bench: 0.4902 (n=200, spearman); ★JEV vs LLMs as Rubric Judges: 0.486 (n=360, exact accuracy vs human label (Jev Choic) |
| massive | 5 | Alexander-Ollman/laya-ft: 0.799 (n=1,000, macro-F1); oluies/jev-vs-spacy: 0.889 (n=360, accuracy); ★Do System One Decisions Add Up?: 0.833 (n=1,000, accuracy (flat fine-label)); ★morrenhale/decision-benchmark-jev-laya-j: 0.6767 (n=5,100, accuracy); ★SOTAAZ blog: 0.829 (n=175, accuracy) |
| imdb | 4 | Deusser et al. 2026: 0.9652 (n=25,000, accuracy); 4esv/jev-eval: 0.97 (n=300, accuracy); mugenkyou/JEV-VS-ML: 0.963 (n=1,000, balanced accuracy); ★QuicqDev/Jev-vs-ML: 0.963 (n=1,000, balanced accuracy (raw decision)) |
| anli | 4 | Deusser et al. 2026: 0.7394 (n=3,200, accuracy); Decision Index 0.2.1: 0.7479 (n=3,200, macro-F1); jev-decision-bench: 0.66 (n=200, accuracy); ★Ordinal-Scale Bias in JEV-like Models: 0.7495 (n=6,400, accuracy) |
| pubmedqa | 4 | Deusser et al. 2026: 0.787 (n=1,000, accuracy); Jevals: 0.9127 (n=300, accuracy); Jevals: 69.03 (n=300, Decision Score (0-100; 0 = label-prior, ); ★Jev in Medicine: 0.784 (n=500, top-1 accuracy) |
| mmlupro | 4 | Decision Index 0.2.1: 0.827 (n=12,032, accuracy); Kev: 0.84 (n=?, accuracy); WebJev: 0.834 (n=1,000, accuracy); ★JevOut: Natural Context Can Flip Decisio: 0.83 (n=100, accuracy (clean)) |
| halueval | 4 | LangWatch Jev benchmark: 0.803 (n=666, balanced accuracy); ★JEV-as-a-Judge: 0.873 (n=3,000, accuracy); ★JEV-as-a-Judge: 0.698 (n=400, accuracy); ★JEV-as-a-Judge: 0.535 (n=200, accuracy) |
| nfcorpus | 4 | jev-decision-bench: 0.7337 (n=60, ndcg10); denser-org/rerank-bench-jev: 0.3623 (n=323, nDCG@10); hev/reranker: 0.358 (n=323, nDCG@10); ★romeromarcelo/jev-retrieval: 0.363 (n=323, nDCG@10) |
| rmbench | 4 | goya4140/jev-reward-model-evaluation: 0.8129 (n=1,327, official 4-domain macro); goya4140/jev-reward-model-evaluation: 0.8379 (n=7,962, official 4-domain macro); ★JEV-as-a-Judge: 0.854 (n=3,000, accuracy); ★JEV-as-a-Judge: 0.766 (n=3,000, accuracy) |
| cuad | 3 | matu79go/jev-hanko: 0.519 (n=20,500, F1); ★maybern-tripp-smith/cuad-jev-bench: 0.68 (n=200, pairwise accuracy (1 - inversion rate, b); ★maybern-tripp-smith/cuad-jev-bench: 0.917 (n=100, MRR) |
| rjudge | 3 | ★JEV as a Judge for Agent Trace Security: 0.885 (n=564, positive-class F1 (on valid responses)); ★JEV as a Judge for Agent Trace Security: 0.778 (n=5,219, mean positive-class F1); ★Evaluating System One Models for Agent S: 0.825 (n=236, macro-F1) |
| amazonreviews2023 | 3 | ★Decision-Oriented Recommendation Reranki: 0.238 (n=954, MRR (K=20)); ★Decision-Oriented Recommendation Reranki: 0.295 (n=1,000, MRR (K=20)); ★Decision-Oriented Recommendation Reranki: 0.346 (n=626, MRR (K=20)) |
| contractnli | 2 | Decision Index 0.2.1: 0.7169 (n=123, macro-F1); ★Same Scores, Different Decisions: 0.7738 (n=2,091, accuracy) |
| musr | 2 | Decision Index 0.2.1: 0.6609 (n=752, accuracy); ★JevOut: Natural Context Can Flip Decisio: 0.61 (n=100, accuracy (clean)) |
| satabench | 2 | Decision Index 0.2.1: 0.2642 (n=1,650, case exact accuracy); ★JevOut: Natural Context Can Flip Decisio: 0.823 (n=100, accuracy (clean)) |
| berkeleyfunctioncallingleaderboard | 2 | LangWatch Jev benchmark: 0.783 (n=1,000, accuracy); ★JevOut: Natural Context Can Flip Decisio: 0.7 (n=100, accuracy (clean)) |
| andeytaitjevforgemind2web | 2 | LangWatch Jev benchmark: 0.708 (n=2,329, accuracy); ★zwliJay/jev-forge: 0.543 (n=800, choice top-1 accuracy) |
| truthfulqa | 2 | jev-decision-bench: 0.95 (n=200, accuracy); ★Beyond Answer Confidence: 0.909 (n=?, accuracy) |
| twitterfinancialnewstopic | 2 | elcronos/jev-vs-open-decision-models: 0.67 (n=4,117, accuracy); ★SOTAAZ blog: 0.71 (n=200, accuracy) |
| wanli | 2 | Kev: 0.758 (n=256, accuracy); ★yehor-oleksiuk/bonzi-vs-jev-wanli256: 0.793 (n=256, accuracy) |
| rewardbench | 2 | goya4140/jev-reward-model-evaluation: 0.9258 (n=2,985, official 4-section macro); ★JEV-as-a-Judge: 0.925 (n=1,500, accuracy) |
| rewardbench2 | 2 | goya4140/jev-reward-model-evaluation: 0.8115 (n=1,865, official 6-domain macro); ★JEV-as-a-Judge: 0.73 (n=100, accuracy) |
| nslkdd | 2 | jev-ids/jev-ids: 0.859 (n=900, F1 (attack)); ★Jev-IDS: 0.782 (n=2,000, F1 (attack class)) |
| dblpacm | 2 | ★CompleteDotTech/paper-package: 0.9605 (n=413, macro-F1); ★CompleteDotTech/paper-package: 0.9859 (n=413, macro-F1) |
| fed | 2 | ★JEV vs LLMs as Rubric Judges: 0.562 (n=600, exact accuracy vs human label (Jev Choic); ★JEV vs LLMs as Rubric Judges: 0.486 (n=250, exact accuracy vs human label (Jev Choic) |
| usr | 2 | ★JEV vs LLMs as Rubric Judges: 0.62 (n=360, exact accuracy vs human label (Jev Choic); ★JEV vs LLMs as Rubric Judges: 0.564 (n=300, exact accuracy vs human label (Jev Choic) |
| mmlucf | 2 | ★Beyond Answer Confidence: 0.776 (n=10,000, accuracy); ★Lightfield blog: 0.8 (n=100, accuracy) |
| semeval2026 | 2 | ★Decide, Don't Generate: DimABSA: 1.0645 (n=?, aggregate RMSE (lower is better)); ★Decide, Don't Generate: DimABSA: 0.5209 (n=?, continuous F1) |
| cesnetquicext25 | 2 | ★Jev for Network Traffic Classification: 0.098 (n=52,000, accuracy); ★Jev for Network Traffic Classification: 0.2842 (n=52,000, accuracy) |
| consumercomplaintsmedium | 2 | ★earino/zero-shot-complaint-benchmark: 0.3442 (n=6,430, accuracy); ★earino/zero-shot-complaint-benchmark: 0.4649 (n=6,430, accuracy) |
| persuasion | 2 | ★Lightfield blog: 0.712 (n=500, accuracy); ★Lightfield blog: 0.442 (n=500, accuracy) |

---

## 4. Public per-item Jev outputs (for evaluation only)

House rule: **Jev outputs never enter training data, teacher labels, calibration fits or checkpoint selection** (TypeSafe MCA §2.3(b)). Some licences forbid even evaluation use for a competing product. The rows below are for reading aggregates, or for paired re-scoring only where the licence allows it.

| Source | URL | Licence | Can meharsjev use it? |
|---|---|---|---|
| Deußer Jev responses (346k) | https://doi.org/10.5281/zenodo.23039006 (`responses.db`, 207 MB) | Jev Responses License 1.0: research/eval, **no "similar or competing product"** (§3.2), share-alike | **No.** Cite aggregates only. |
| Jevals per-decision logs | https://github.com/Jevals/jevals-data/tree/main/runs (`jev__{banking77,pubmedqa,helpsteer2}__0.1.0.jsonl`) | CC BY 4.0 (Jevals); the Jev outputs were collected under TypeSafe terms | Eval-only reading and paired scoring; attribution required |
| jev-bench (Praveenrajus) | https://huggingface.co/datasets/Praveenrajus/jev-bench/blob/main/results/jev-1.13.0/test_predictions.jsonl (15.9 MB) | "other" (upstream dataset licences) | Eval-only; check each upstream set |
| OmarMujahid jev-decision-bench | https://github.com/OmarMujahid/jev-decision-bench/tree/main/results/jev | MIT | Eval-only |
| Ibrahim & Zaki CSS | https://github.com/hazemibrahim97/decision-models-css/blob/main/analysis/items.csv (28 MB) | MIT repo; paper states CC BY 4.0 | Eval-only |
| Gaurav-Gosain jev-sec-bench | https://github.com/Gaurav-Gosain/jev-sec-bench/tree/main/results | MIT | Eval-only |
| heswy Jev-Benchmark | https://github.com/heswy/Jev-Benchmark/tree/main/results (`raw-complete.tar.gz`) | MIT | Eval-only |
| open-system-one (dylantom2012) | https://huggingface.co/datasets/dylantom2012/open-system-one-bench | MIT | Eval-only |
| scienthoon jev-ood-calibration | https://github.com/scienthoon/jev-ood-calibration | MIT | Eval-only |
| anessbelbati rerank-bench | https://github.com/anessbelbati/jev-rerank-bench | MIT | Eval-only |
| nikkoxgonzales jev-certify (CLINC, 2,412 answers) | https://github.com/nikkoxgonzales/jev-certify | MIT | Eval-only |
| Feishu-style Chinese (Laya repo) | https://github.com/NandhaKishorM/laya/blob/main/research/benchmarks/feishu_zh/results/v1/jev/raw.jsonl | MIT | Eval-only (tier X) |
| ChaosNLI study | https://zenodo.org/records/22971492 | see record | Eval-only |
| DMB raw archives | https://github.com/nibzard/decision-model-benchmark/tree/main/results | **no licence** (all rights reserved) | Read reports only |
| elcronos | https://github.com/elcronos/jev-vs-open-decision-models/blob/main/results/raw_predictions.parquet | **no licence** | Read reports only |
| Alexander-Ollman laya-ft, simonmesmith, switchboard, Janus, jev-phishing-bench | repos listed in the JSON rows | **no licence** | Read reports only |
| DecisionBench Jev artifact | `hf://buckets/Hanno-Labs/training/decision-bench/runs/jev-1-13-23900-20260921-r1` | not stated; access not verified | Read the summary JSON only |
| Not published | Decision Index (maintainers hold answers), typed-decisions, LangWatch, chepyle (gitignored), thisisandreeeee, zhuyansen, AbdelStark, TypeSafe evals | – | – |
| morrenhale decision benchmark (2026-09-27) | https://huggingface.co/datasets/morrenhale/decision-benchmark-jev-laya-julia/blob/main/predictions/jev.jsonl (19,776 rows, full probability vectors) | CC-BY-4.0 (Jev outputs collected under TypeSafe terms) | Eval-only |
| earino CFPB complaints | https://huggingface.co/datasets/earino/ecbs5200-jev-benchmark/tree/main/jev_responses | "other" | Eval-only; check terms |
| kiwi0719 jev-edge (deepset 662, suite v1) | https://github.com/kiwi0719/jev-edge/tree/main/bench/datasets | Apache-2.0 | Eval-only |
| cuad-jev-bench response cache | https://github.com/maybern-tripp-smith/cuad-jev-bench/tree/main/runs/jev/cache | MIT (code); CUAD text CC BY 4.0 | Eval-only |
| KoBBQ calibration audit | https://github.com/jujumilk3/jev-calibration-audit/tree/main/results/raw | MIT | Eval-only |
| vector-graph-rag Jev reranker | https://github.com/zilliztech/vector-graph-rag/tree/main/evaluation/jev | MIT | Eval-only |
| CompleteDotTech paper-package | https://github.com/CompleteDotTech/paper-package/tree/main/reproduction/results/jev/run-20260918 | AGPL-3.0 code; saved evidence under RIGHTS.md | Read aggregates; check RIGHTS.md before any reuse |
| statsguysam jev-classification-benchmark | https://github.com/statsguysam/jev-classification-benchmark/tree/main/results | **no licence** | Read reports only |
| trifleen phishing pilot, poorjev crossbench, Sys1Cal-v1, HydroJEV, jev-dimabsa, JevOut | repos in the JSON rows | MIT or unstated | Eval-only; read aggregates where unlicensed |
| Not published (resume pass) | all arXiv papers without a code link above, QuicqDev ("not included in this release"), LangWatch-style blogs | – | – |

---

## 5. Full list: every row, by suite (generated from the JSON)

Columns are cut down for width. Each row's protocol, CI, ECE/Brier, comparators and licence detail are in `notes` in the JSON. Track: Z = zero-shot, S = uses train data. ★ = added 2026-10-03.

#### Deusser et al. 2026 (arXiv 2609.37647) (54 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| AG News | fancyzhx/ag_news / test | 7,600 | accuracy | 0.8851 | A | Z | yes |
| IMDB | stanfordnlp/imdb / plain_text / test | 25,000 | accuracy | 0.9652 | A | Z | yes |
| Rotten Tomatoes | cornell-movie-review-data/rotten_tomatoes / test | 1,066 | accuracy | 0.9334 | A | Z | yes |
| SST-2 | stanfordnlp/sst2 / validation | 872 | accuracy | 0.9644 | A | Z | yes |
| DAIR Emotion | dair-ai/emotion / split / test | 2,000 | accuracy | 0.585 | A | Z | yes |
| Financial PhraseBank | atrost/financial_phrasebank / test | 970 | accuracy | 0.7299 | A | Z | yes |
| BANKING77 | mteb/banking77 / test | 3,076 | accuracy | 0.7971 | A | Z | yes |
| CLINC150 (plus, incl. OOS) | clinc/clinc_oos / plus / test | 5,500 | accuracy | 0.8945 | A | Z | yes |
| SIB-200 (205 langs) | Davlan/sib200 / all 205 language configs / test | 41,820 | accuracy | 0.8152 | A | Z | yes |
| Language identification (20 langs) | papluca/language-identification / test | 10,000 | accuracy | 0.9964 | A | Z | yes |
| SMS Spam | ucirvine/sms_spam / plain_text / train (only split, all rows) | 5,574 | f1 | 0.9381 | A | Z | yes |
| GoEmotions (28 labels, multi-label) | google-research-datasets/go_emotions / simplified / test | 5,427 | macro_f1 | 0.2434 | A | Z | yes |
| ANLI R1-R3 | facebook/anli / plain_text / test_r1+test_r2+test_r3 | 3,200 | accuracy | 0.7394 | A | Z | yes |
| AfriXNLI (18 langs) | masakhane/afrixnli / 18 language configs / test | 10,800 | accuracy | 0.6402 | A | Z | yes |
| PAWS | google-research-datasets/paws / labeled_final / test | 8,000 | accuracy | 0.8499 | A | Z | yes |
| LLM-AggreFact (11 sources) | lytang/LLM-AggreFact / test | 29,320 | mean_balanced_accuracy | 0.7858 | A | Z | yes |
| BoolQ | google/boolq / validation | 3,270 | accuracy | 0.9131 | A | Z | yes |
| Belebele (122 langs) | facebook/belebele / 122 language configs / test | 109,800 | accuracy | 0.8673 | A | Z | yes |
| PubMedQA (yes/no/maybe) | qiaojin/PubMedQA / pqa_labeled / train (all 1,000) | 1,000 | accuracy | 0.787 | A | Z | yes |
| MMLU (57 subjects) | tasksource/mmlu / 57 subject configs / test | 14,042 | accuracy | 0.9181 | A | Z | yes |
| C-Eval (52 subjects) | ceval/ceval-exam / 52 subject configs / test | 12,342 | accuracy | 0.8392 | A | Z | yes |
| BIG-bench MC subset (93 tasks) | tasksource/bigbench / 93 MC tasks (bigbench_mc_tasks.json) / validation | 13,228 | accuracy | 0.8139 | A | Z | yes |
| HellaSwag | Rowan/hellaswag / validation | 10,042 | accuracy | 0.9549 | A | Z | yes |
| WinoGrande | allenai/winogrande / winogrande_xl / validation | 1,267 | accuracy | 0.914 | A | Z | yes |
| ARC Easy+Challenge | allenai/ai2_arc / ARC-Easy + ARC-Challenge / test | 3,548 | accuracy | 0.9876 | A | Z | yes |
| CommonsenseQA | tau/commonsense_qa / validation | 1,221 | accuracy | 0.8821 | A | Z | yes |
| alphaNLI (ART) | allenai/art / validation | 1,532 | accuracy | 0.8388 | A | Z | yes |
| ToxiGen (annotated) | toxigen/toxigen-data / annotated / test | 940 | accuracy | 0.8777 | A | Z | yes |
| OpenAI moderation eval (8 categories) | mmathys/openai-moderation-api-evaluation / train (all 1,680) | 1,680 | mean_auprc | 0.7173 | A | Z | yes |
| ToxicChat | lmsys/toxic-chat / toxicchat0124 / test | 5,083 | f1 | 0.7862 | A | Z | yes |
| deepset prompt-injections | deepset/prompt-injections / test | 116 | accuracy | 0.7414 | A | Z | yes |
| AGB-DE (German T&C voidness) | d4br4/agb-de / test | 755 | f1 | 0.2038 | A | Z | yes |
| LexGLUE UNFAIR-ToS (8 labels) | coastalcph/lex_glue / unfair_tos / test | 1,607 | micro_f1 | 0.4993 | A | Z | yes |
| STS-B | sentence-transformers/stsb / test | 1,379 | spearman | 0.8902 | A | Z | yes |
| SST-5 | SetFit/sst5 / test | 2,210 | argmax_accuracy | 0.5792 | A | Z | yes |
| SummEval (4 dims) | mteb/summeval / test | 1,600 | mean_group_spearman | 0.5538 | A | Z | yes |
| HelpSteer2 (5 attributes) | nvidia/HelpSteer2 / validation | 1,038 | mean_spearman | 0.4122 | A | Z | yes |
| BANKING77 | mteb/banking77 / test | 3,076 | macro_f1 | 0.7883 | A | Z | yes |
| DAIR Emotion | dair-ai/emotion / split / test | 2,000 | macro_f1 | 0.4987 | A | Z | yes |
| CLINC150 (plus, incl. OOS) | clinc/clinc_oos / plus / test | 5,500 | oos_recall | 0.776 | A | Z | yes |
| CLINC150 (plus, incl. OOS) | clinc/clinc_oos / plus / test | 5,500 | in_scope_accuracy | 0.9209 | A | Z | yes |
| LLM-AggreFact (11 sources) | lytang/LLM-AggreFact / test | 29,320 | pooled_balanced_accuracy | 0.8306 | A | Z | yes |
| SST-5 | SetFit/sst5 / test | 2,210 | spearman | 0.851 | A | Z | yes |
| ToxicChat | lmsys/toxic-chat / toxicchat0124 / test | 5,083 | f1 (jailbreak head) | 0.7173 | A | Z | yes |
| LexGLUE UNFAIR-ToS (8 labels) | coastalcph/lex_glue / unfair_tos / test | 1,607 | micro_f1 (per-label thresholds tuned on 1,000 dev examples) | 0.7482 | A | S | yes |
| GoEmotions (28 labels, multi-label) | google-research-datasets/go_emotions / simplified / test | 5,427 | macro_f1 (per-label thresholds tuned on 1,000 dev examples) | 0.3529 | A | S | yes |
| ToxicChat | lmsys/toxic-chat / toxicchat0124 / test | 5,083 | f1 toxic (threshold tuned on 1,000 dev examples, thr 0.42) | 0.7933 | A | S | yes |
| MMLU memorization probe: calc.-heavy, original | tasksource/mmlu / calc-heavy subjects / test | 2,207 | accuracy | 0.9434 | X | Z | yes |
| MMLU memorization probe: other, original | tasksource/mmlu / other subjects / test | 11,835 | accuracy | 0.9134 | X | Z | yes |
| MMLU memorization probe: calc.-heavy, options rotated | tasksource/mmlu / calc-heavy subjects / test | 2,207 | accuracy | 0.9429 | X | Z | yes |
| MMLU memorization probe: calc.-heavy, question withheld | tasksource/mmlu / calc-heavy subjects / test | 2,207 | accuracy | 0.3149 | X | Z | yes |
| C-Eval memorization probe: calc.-heavy, original | ceval/ceval-exam / calc-heavy subjects / test | 3,389 | accuracy | 0.8144 | X | Z | yes |
| C-Eval memorization probe: other, original | ceval/ceval-exam / other subjects / test | 8,953 | accuracy | 0.8487 | X | Z | yes |
| C-Eval memorization probe: calc.-heavy, question withheld | ceval/ceval-exam / calc-heavy subjects / test | 3,389 | accuracy | 0.373 | X | Z | yes |

#### Decision Index 0.2.1 (55 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| BFCL | gorilla-llm/gorilla (GitHub) BFCL_v3 simple+live_simple+live_multiple / possible_answer | 1,694 | case exact accuracy | 0.9575 | A | Z |  |
| ToolRet | mangopy/ToolRet-Queries + mangopy/ToolRet-Tools / 0.2.1 answerable subset / test | 685 | nDCG@10 | 0.6528 | A | Z |  |
| API-Bank | AlibabaResearch/DAMO-ConvAI api-bank (GitHub) / lv1-lv2 samples / test | 508 | accuracy | 0.8819 | A | Z |  |
| BANKING77 | PolyAI-LDN/task-specific-datasets banking_data/test.csv (= PolyAI/banking77) / test | 3,080 | macro-F1 | 0.7974 | A | Z |  |
| CLINC150+OOS | clinc/oos-eval data_full.json (= clinc/clinc_oos) / full / test | 5,500 | macro-F1 | 0.8927 | A | Z |  |
| RouterBench | withmartian/routerbench / 0shot+5shot pkl / - | 5,001 | selected quality (quality objective) | 0.7992 | B | Z |  |
| MiniWoB++ | MiniWoB++ (interactive) / - | 9 | episode success | 1 | X | Z |  |
| Home appliance simulator | generated (decision_index/suite/build/home_appliance.py, seed 2026091807) / test (0.2.1 dedup) | 88 | case exact accuracy | 0.5227 | A | Z |  |
| SGD/SGD-X | google-research-datasets/dstc8-schema-guided-dialogue (GitHub) / test | 2,500 | macro-F1 | 0.4295 | B | Z |  |
| ContractNLI | stanfordnlp/contract-nli (GitHub) / test | 123 | macro-F1 | 0.7169 | A | Z |  |
| ANLI | facebook/anli / plain_text / test_r1+r2+r3 | 3,200 | macro-F1 | 0.7479 | A | Z |  |
| Boxoban | Boxoban (interactive) / - | 100 | levels solved | 0 | X | Z |  |
| RTFM | RTFM (interactive) / - | 100 | episode success | 0.17 | X | Z |  |
| ScienceWorld | ScienceWorld (interactive) / - | 10 | episode success | 0.067 | X | Z |  |
| Hanabi | Hanabi (interactive) / - | 100 | episode success | 0 | X | Z |  |
| Codenames | Codenames (interactive) / - | 66 | episode success | 0 | X | Z |  |
| BPoMP | Zenodo 7299879 (BPoMP) / - | 811 | accuracy | 0.906 | A | Z |  |
| Humicroedit | SemEval-2020 Task 7 (Humicroedit) / subtask-2 / test | 2,628 | accuracy | 0.6187 | A | Z |  |
| POP909-CL | AndyWeasley2004/POP909-CL-Dataset (GitHub) / - | 2,000 | accuracy | 0.181 | A | Z |  |
| cfcolor | dgp.toronto.edu cfcolor / - | 5,000 | accuracy | 0.6474 | A | Z |  |
| MMLU | cais/mmlu / all / test | 14,042 | accuracy | 0.9173 | B | Z |  |
| GPQA Diamond | idavidrein/gpqa (GitHub zip) / diamond / - | 198 | accuracy | 0.7828 | A | Z |  |
| ARC-Easy | allenai/ai2_arc / ARC-Easy / test | 2,376 | accuracy | 0.9933 | B | Z |  |
| ARC-Challenge | allenai/ai2_arc / ARC-Challenge / test | 1,172 | accuracy | 0.9778 | B | Z |  |
| WinoGrande | allenai/winogrande / winogrande_xl / validation | 1,267 | accuracy | 0.9195 | A | Z |  |
| HellaSwag | Rowan/hellaswag / validation | 10,042 | accuracy | 0.9452 | A | Z |  |
| GSM8K | openai/gsm8k / main / test (MC with algorithmic distractors, 2 tracks) | 1,319 | accuracy | 0.7987 | A | Z |  |
| ChessBench | google-deepmind/searchless_chess action_value test / test | 5,000 | accuracy | 0.1722 | A | Z |  |
| MuSR | Zayne-sprague/MuSR (GitHub) / - | 752 | accuracy | 0.6609 | A | Z |  |
| SATA-Bench | sata-bench/sata-bench (GitHub) / - | 1,650 | case exact accuracy | 0.2642 | A | Z |  |
| BRIGHT | xlangai/BRIGHT / 0.2.1 answerable subset / examples | 220 | nDCG@10 | 0.4752 | A | Z |  |
| Amazon ESCI | amazon-science/esci-data (GitHub) / 5,000 stratified pairs / - | 5,000 | macro-F1 | 0.5521 | A | Z |  |
| ACOS | NUSTM/ACOS (GitHub) / 400-review subset / - | 400 | per-review F1 | 0.2952 | A | Z |  |
| FinEntity | yixuantt/FinEntity (GitHub) / - | 979 | macro-F1 | 0.8698 | A | Z |  |
| iSarcasmEval | iabufarha/iSarcasmEval (GitHub) / task A English / test | 1,400 | Sarcasm F1 · track A, English | 0.5051 | A | Z |  |
| VAST | emilyallaway/zero-shot-stance VAST (GitHub) / test | 3,006 | macro-F1 | 0.6463 | A | Z |  |
| NLI4CT | ai-systems/Task-2-SemEval-2024 (NLI4CT, GitHub) / test | 5,500 | macro-F1 | 0.8406 | A | Z |  |
| CRUXEval | facebookresearch/cruxeval (GitHub) / - | 570 | accuracy | 0.7298 | A | Z |  |
| CLadder | causalNLP/cladder (GitHub) / balanced / - | 5,000 | accuracy | 0.7264 | A | Z |  |
| HLE | cais/hle (gated) / text-only MC / test | 501 | accuracy | 0.2036 | A | Z |  |
| ForecastBench | forecastingresearch/forecastbench-datasets / resolutions 2026-07-01..09-18 / - | 10,139 | Brier (lower is better) | 0.1736 | A | Z |  |
| Habermas Machine | google-deepmind/habermas_machine / - | 1,676 | accuracy | 0.4594 | A | Z |  |
| OpenBookQA | allenai/openbookqa / test? | 500 | accuracy | 0.94 | B | Z |  |
| CommonsenseQA | tau/commonsense_qa / validation | 1,221 | accuracy | 0.8747 | B | Z |  |
| Support-ticket calibration | scienthoon support-ticket calibration (synthetic) / - | 300 | accuracy | 0.905 | X | Z |  |
| Phishing difficulty gradient | phishing difficulty gradient (synthetic) / - | 800 | accuracy | 0.9463 | X | Z |  |
| Email spam classification | email spam classification / - | 3,000 | accuracy | 0.9743 | B | Z |  |
| PhishNChips phishing decisions | AreLit/PhishNChips core_emails.csv / all | 2,000 | accuracy | 0.6255 | A | Z |  |
| MMLU-Pro | TIGER-Lab/MMLU-Pro / test | 12,032 | accuracy | 0.827 | A | Z |  |
| BBH | suzgunmirac/BIG-Bench-Hard (23 fixed-option tasks) / - | 5,507 | accuracy | 0.9292 | A | Z |  |
| RAGTruth | ParticleMedia/RAGTruth (GitHub) / test | 2,700 | F1 on hallucinated class | 0.7653 | A | Z |  |
| HoVer | hover-nlp/hover (oracle docs) / dev | 4,000 | accuracy | 0.7285 | A | Z |  |
| When2Call | nvidia/When2Call / when2call_test_mcq / test | 3,652 | accuracy | 0.8097 | A | Z |  |
| New Yorker | jmhessel/newyorker_caption_contest / matching / test | 528 | accuracy | 0.7008 | A | Z |  |
| Decision Index (aggregate of 38 panel benchmarks) | apolinario/decision-index (suite recipe) / edition 0.2.1 / - | 119,898 | index (skill, 0-100) | 57.91 | A | Z |  |

#### DMB (nibzard/decision-model-benchmark) (11 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| BANKING77 (S6, official test) | PolyAI-LDN/task-specific-datasets (pinned commit; = PolyAI/banking77) / test | 3,080 | accuracy | 0.792 | A | Z | yes |
| CLINC150 + OOS (S7) | clinc/clinc_oos (official split, 1,000 OOS) / full/plus-equivalent / test | 5,500 | accuracy | 0.886 | A | Z | yes |
| CLINC150 + OOS (S7) OOS recall | clinc/clinc_oos / full / test | 1,000 | oos_recall | 0.812 | A | Z | yes |
| NLU++ (S8, folds 18-19, banking+hotel) | PolyAI NLU++ (GitHub, CC BY 4.0) / folds 18-19 / test (302 msgs / 13,712 binary decisions) | 13,712 | micro intent F1 | 0.483 | A | Z | yes |
| BANKING77 pilot (1 item per intent) | PolyAI/banking77 / test (77-item sample) | 77 | accuracy | 0.805 | C | Z | yes |
| SMS spam pilot | ucirvine/sms_spam (UCI) / 50-item sample | 50 | accuracy | 0.98 | C | Z |  |
| BANKING77 controls (1 item per intent) | PolyAI/banking77 / 77-item frozen sample | 77 | accuracy | 0.831 | C | Z | yes |
| S1 intent77 (BANKING77, v1.1 historical, 300 items x 3 repeats) | PolyAI/banking77 / 300-item sample x3 | 900 | accuracy | 0.763 | B | Z | yes |
| S2 SMS spam (v1.1 historical, 300 x 3) | ucirvine/sms_spam / 300-item sample x3 | 900 | accuracy | 0.93 | B | Z |  |
| S4 order stability (BANKING77-derived, 100 bases x 3 orders x 3 repeats) | PolyAI/banking77 / 100 base items | 900 | accuracy | 0.767 | C | Z |  |
| S5 forced uncertainty, underdetermined subset (synthetic) | synthetic (seed 20260918) / - | 300 | ECE (score diagnostic) | 0.246 | X | Z |  |

#### Jevals (release 2026-09-18, suite 0.1.0) (6 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| Banking77 (choice) | mteb/banking77 / default / test | 300 | accuracy | 0.7967 | B | Z | yes |
| Banking77 (choice) | mteb/banking77 / default / test | 300 | Decision Score (0-100; 0 = label-prior, proper-score based) | 67.78 | B | Z | yes |
| HelpSteer2 helpfulness (score) | nvidia/HelpSteer2 / default / validation | 300 | accuracy | 0.4127 | B | Z | yes |
| HelpSteer2 helpfulness (score) | nvidia/HelpSteer2 / default / validation | 300 | Decision Score (0-100; 0 = label-prior, proper-score based) | 9.2 | B | Z | yes |
| PubMedQA (noul) | qiaojin/PubMedQA / pqa_labeled / train | 300 | accuracy | 0.9127 | B | Z | yes |
| PubMedQA (noul) | qiaojin/PubMedQA / pqa_labeled / train | 300 | Decision Score (0-100; 0 = label-prior, proper-score based) | 69.03 | B | Z | yes |

#### typed-decisions (LocalLLaMA/typed-decisions card) (5 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| typed-decisions (4 workflows) | LocalLLaMA/typed-decisions / test | 400 | accuracy (argmax vs gold argmax, 2,000 decisions) | 0.727 | A | Z |  |
| typed-decisions noul subset | LocalLLaMA/typed-decisions / test |  | accuracy (noul questions) | 0.775 | A | Z |  |
| typed-decisions choice subset | LocalLLaMA/typed-decisions / test |  | accuracy (choice questions) | 0.72 | A | Z |  |
| typed-decisions score subset | LocalLLaMA/typed-decisions / test |  | accuracy (score questions) | 0.696 | A | Z |  |
| typed-decisions KL | LocalLLaMA/typed-decisions / test | 400 | KL from gold (lower is better) | 1.442 | A | Z |  |

#### TypeSafe workflow evals (evals.typesafe.ai) (5 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| Workflow evals average (4 workflows) | not released (evals.typesafe.ai) / - | 705 | agreement with LLM-consensus reference | 0.678 | X | Z |  |
| Security incidents | not released / - | 240 | agreement | 0.617 | X | Z |  |
| Agent trace observability | not released / - | 111 | agreement | 0.716 | X | Z |  |
| Invoice processing | not released / - | 150 | agreement | 0.618 | X | Z |  |
| Customer service | not released / - | 204 | agreement | 0.76 | X | Z |  |

#### JevBench v1.5.5 (benchmarkheaven.com) (3 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| JevBench composite score | fstandhartinger/jevbench (601 public items) / v1.5.5 / open+sealed | 1,624 | JevBench score (0-100) | 72.13 | B | Z |  |
| JevBench Intelligence axis | fstandhartinger/jevbench / v1.5.5 / open+sealed | 1,624 | Intelligence (0-100) | 72 | B | Z |  |
| JevBench Calibration axis | fstandhartinger/jevbench / v1.5.5 / open+sealed | 1,624 | Calibration (0-100) | 88.03 | B | Z |  |

#### DecisionBench (Hanno Labs) (5 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| DecisionBench overall | Hanno-Labs/decision-bench / eval (all rows) | 23,900 | primary accuracy | 0.7203 | A | Z | yes |
| DecisionBench view primitive:binary_classification | Hanno-Labs/decision-bench / eval | 6,388 | accuracy | 0.5922 | A | Z | yes |
| DecisionBench view primitive:candidate_selection | Hanno-Labs/decision-bench / eval | 15,814 | accuracy | 0.8009 | A | Z | yes |
| DecisionBench view primitive:ordinal_scoring | Hanno-Labs/decision-bench / eval | 1,698 | accuracy | 0.4511 | A | Z | yes |
| DecisionBench view suite:DecisionBench(eng, v1) | Hanno-Labs/decision-bench / eval | 22,700 | accuracy | 0.719 | A | Z | yes |

#### DecideBench v1.1 (choyiny) (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| DecideBench (8 task families, contrastive pairs) | choyiny/decidebench / test | 400 | accuracy | 0.98 | B | Z |  |

#### LangWatch Jev benchmark (release 2026-09-23.1) (15 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| Prompt injection | deepset/prompt-injections + jackhhao/jailbreak-classification + reshabhs/SPML + djapp18/JailbreaksOverTime / sample | 1,000 | catch rate @ 5% false alarms | 0.946 | B | Z |  |
| Moderation | mmathys/openai-moderation-api-evaluation + nvidia/Aegis-AI-Content-Safety-Dataset-2.0 / sample | 667 | AUROC | 0.903 | B | Z |  |
| PII | gretelai/gretel-pii-masking-en-v1 + beki/privy + nvidia/Nemotron-PII / sample | 1,000 | catch rate @ 5% false alarms | 0.908 | B | Z |  |
| RAG faithfulness | pminervini/HaluEval / sample | 666 | balanced accuracy | 0.803 | B | Z |  |
| Off-topic | clinc/clinc_oos + mteb/amazon_massive_intent + benayas/snips / sample | 1,000 | balanced accuracy | 0.934 | B | Z |  |
| Routing (20 intents) | legacy-datasets/banking77 / sample | 1,000 | accuracy | 0.891 | B | Z |  |
| Routing (77 intents) | legacy-datasets/banking77 / sample | 500 | accuracy | 0.796 | B | Z |  |
| Tool routing | gorilla-llm/Berkeley-Function-Calling-Leaderboard / sample | 1,000 | accuracy | 0.783 | B | Z |  |
| Complaint routing | BEE-spoke-data/consumer-finance-complaints / sample | 1,000 | accuracy | 0.787 | B | Z |  |
| Commit type | angular/angular + vitejs/vite commits (GitHub) / sample | 1,000 | accuracy | 0.683 | B | Z |  |
| Search relevance | tasksource/esci / sample | 1,000 | accuracy | 0.577 | B | Z |  |
| Typed decisions | LocalLLaMA/typed-decisions / sample | 1,965 | accuracy | 0.739 | B | Z |  |
| Web-agent actions | AndeyTait/JevForge-Mind2Web / sample | 2,329 | accuracy | 0.708 | B | Z |  |
| Community sets | Praveenrajus/jev-bench / sample | 1,200 | accuracy | 0.623 | B | Z |  |
| JevBench public | fstandhartinger/jevbench (public items) / sample | 231 | accuracy | 0.857 | B | Z |  |

#### jev-bench (Praveenrajus/jev-bench, Jevify) (23 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| arc_challenge (choice) | Praveenrajus/jev-bench / arc_challenge / test | 1,000 | accuracy | 0.979 | B | Z | yes |
| banking77 (choice) | Praveenrajus/jev-bench / banking77 / test | 1,000 | accuracy | 0.796 | B | Z | yes |
| boolq (noul) | Praveenrajus/jev-bench / boolq / test | 1,000 | accuracy | 0.917 | B | Z | yes |
| chaosnli (choice) | Praveenrajus/jev-bench / chaosnli / test | 1,599 | accuracy | 0.615 | B | Z | yes |
| civil_comments (noul) | Praveenrajus/jev-bench / civil_comments / test | 2,000 | accuracy | 0.729 | B | Z | yes |
| clinc150 (choice) | Praveenrajus/jev-bench / clinc150 / test | 1,000 | accuracy | 0.893 | B | Z | yes |
| fever_evidence (noul) | Praveenrajus/jev-bench / fever_evidence / test | 1,000 | accuracy | 0.972 | B | Z | yes |
| go_emotions (choice) | Praveenrajus/jev-bench / go_emotions / test | 1,000 | accuracy | 0.282 | B | Z | yes |
| helpsteer2_helpfulness (score) | Praveenrajus/jev-bench / helpsteer2_helpfulness / test | 1,000 | accuracy | 0.363 | B | Z | yes |
| helpsteer2_verbosity (score) | Praveenrajus/jev-bench / helpsteer2_verbosity / test | 1,000 | accuracy | 0.341 | B | Z | yes |
| ledgar (choice) | Praveenrajus/jev-bench / ledgar / test | 1,000 | accuracy | 0.751 | B | Z | yes |
| massive (choice) | Praveenrajus/jev-bench / massive / test | 1,000 | accuracy | 0.808 | B | Z | yes |
| measuring_hate_speech (score) | Praveenrajus/jev-bench / measuring_hate_speech / test | 1,000 | accuracy | 0.527 | B | Z | yes |
| mmlu (choice) | Praveenrajus/jev-bench / mmlu / test | 1,000 | accuracy | 0.923 | B | Z | yes |
| mnli (choice) | Praveenrajus/jev-bench / mnli / test | 1,000 | accuracy | 0.883 | B | Z | yes |
| paws (noul) | Praveenrajus/jev-bench / paws / test | 1,000 | accuracy | 0.846 | B | Z | yes |
| sms_spam (noul) | Praveenrajus/jev-bench / sms_spam / test | 800 | accuracy | 0.965 | B | Z | yes |
| sst5 (score) | Praveenrajus/jev-bench / sst5 / test | 1,000 | accuracy | 0.565 | B | Z | yes |
| strategyqa_closed (noul) | Praveenrajus/jev-bench / strategyqa_closed / test | 687 | accuracy | 0.785 | B | Z | yes |
| strategyqa_grounded (noul) | Praveenrajus/jev-bench / strategyqa_grounded / test | 687 | accuracy | 0.956 | B | Z | yes |
| stsb (score) | Praveenrajus/jev-bench / stsb / test | 1,000 | accuracy | 0.538 | B | Z | yes |
| yelp5 (score) | Praveenrajus/jev-bench / yelp5 / test | 1,000 | accuracy | 0.685 | B | Z | yes |
| jev-bench macro (22 configs) | Praveenrajus/jev-bench / all / test | 22,773 | macro accuracy | 0.733 | B | Z | yes |

#### jev-decision-bench (OmarMujahid) (39 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| News topic (AG News) | fancyzhx/ag_news / sample (per-task; see source) | 200 | accuracy | 0.87 | B | Z | yes |
| Adversarial natural language inference (ANLI R3) | facebook/anli / sample (per-task; see source) | 200 | accuracy | 0.66 | B | Z | yes |
| Arabic tweet sentiment (AJGT) | komari6/ajgt_twitter_ar / sample (per-task; see source) | 200 | auroc | 0.9539 | B | Z | yes |
| Science exam questions (ARC-Challenge) | allenai/ai2_arc / sample (per-task; see source) | 200 | accuracy | 0.975 | B | Z | yes |
| Banking intent among 77 (Banking77) | mteb/banking77 / sample (per-task; see source) | 200 | accuracy | 0.815 | B | Z | yes |
| Yes/no reading comprehension (BoolQ) | google/boolq / sample (per-task; see source) | 200 | auroc | 0.9692 | B | Z | yes |
| Commonsense question answering (CommonsenseQA) | tau/commonsense_qa / sample (per-task; see source) | 200 | accuracy | 0.87 | B | Z | yes |
| Emotion in a tweet (dair-ai/emotion) | dair-ai/emotion / sample (per-task; see source) | 200 | accuracy | 0.505 | B | Z | yes |
| Response helpfulness rating (HelpSteer2) | nvidia/HelpSteer2 / sample (per-task; see source) | 200 | spearman | 0.4902 | B | Z | yes |
| Grade-school math as multiple choice (GSM8K) | openai/gsm8k / sample (per-task; see source) | 200 | accuracy | 0.725 | B | Z | yes |
| Most plausible continuation (HellaSwag) | Rowan/hellaswag / sample (per-task; see source) | 200 | accuracy | 0.95 | B | Z | yes |
| Sentiment under prompt injection - attacked | nyu-mll/glue / sample (per-task; see source) | 150 | accuracy | 0.9467 | B | Z | yes |
| Sentiment under prompt injection - clean | nyu-mll/glue / sample (per-task; see source) | 150 | accuracy | 0.96 | B | Z | yes |
| Language identification (20 languages) | papluca/language-identification / sample (per-task; see source) | 200 | accuracy | 0.99 | B | Z | yes |
| Logical reasoning over a passage (LogiQA) | lucasmccabe/logiqa / sample (per-task; see source) | 200 | accuracy | 0.765 | B | Z | yes |
| Academic knowledge (MMLU) | cais/mmlu / sample (per-task; see source) | 200 | accuracy | 0.935 | B | Z | yes |
| Natural language inference (MultiNLI matched) | nyu-mll/glue / sample (per-task; see source) | 200 | accuracy | 0.84 | B | Z | yes |
| Passage reranking (MS MARCO v1.1) | microsoft/ms_marco / sample (per-task; see source) | 100 | mrr | 0.4987 | B | Z | yes |
| Yes/no questions and their explicit negations (BoolQ) | google/boolq / sample (per-task; see source) | 200 | accuracy | 0.805 | B | Z | yes |
| Entity typing (CoNLL-2003) | tner/conll2003 / sample (per-task; see source) | 200 | accuracy | 0.895 | B | Z | yes |
| Biomedical document reranking (BEIR NFCorpus) | BeIR/nfcorpus / sample (per-task; see source) | 60 | ndcg10 | 0.7337 | B | Z | yes |
| Topic classification under reworded instructions (AG News) | fancyzhx/ag_news / sample (per-task; see source) | 200 | accuracy | 0.845 | B | Z | yes |
| Adversarial paraphrase detection (PAWS) | google-research-datasets/paws / sample (per-task; see source) | 200 | auroc | 0.9159 | B | Z | yes |
| Star rating from review text (Yelp Review Full) | Yelp/yelp_review_full / sample (per-task; see source) | 200 | spearman | 0.9267 | B | Z | yes |
| SMS spam detection | ucirvine/sms_spam / sample (per-task; see source) | 200 | auroc | 0.9996 | B | Z | yes |
| Answer-sentence selection (SQuAD v1.1) | rajpurkar/squad / sample (per-task; see source) | 200 | accuracy | 0.945 | B | Z | yes |
| Movie review sentiment (SST-2) | nyu-mll/glue / sample (per-task; see source) | 200 | auroc | 0.9808 | B | Z | yes |
| Semantic textual similarity (STS-Benchmark) | nyu-mll/glue / sample (per-task; see source) | 200 | spearman | 0.9311 | B | Z | yes |
| Toxic comment detection (Civil Comments) | google/civil_comments / sample (per-task; see source) | 200 | auroc | 0.8716 | B | Z | yes |
| Question type (TREC coarse) | SetFit/TREC-QC / sample (per-task; see source) | 200 | accuracy | 0.925 | B | Z | yes |
| Truthful answers to misleading questions (TruthfulQA MC1) | truthfulqa/truthful_qa / sample (per-task; see source) | 200 | accuracy | 0.95 | B | Z | yes |
| Value selection from pre-extracted spans (SQuAD v1.1) | rajpurkar/squad / sample (per-task; see source) | 155 | accuracy | 0.929 | B | Z | yes |
| Pronoun-style blank filling (WinoGrande) | allenai/winogrande / sample (per-task; see source) | 200 | accuracy | 0.885 | B | Z | yes |
| Natural language inference - XNLI (Arabic) | facebook/xnli / sample (per-task; see source) | 150 | accuracy | 0.74 | B | Z | yes |
| Natural language inference - XNLI (English) | facebook/xnli / sample (per-task; see source) | 150 | accuracy | 0.86 | B | Z | yes |
| Natural language inference - XNLI (French) | facebook/xnli / sample (per-task; see source) | 150 | accuracy | 0.7733 | B | Z | yes |
| Natural language inference - XNLI (Hindi) | facebook/xnli / sample (per-task; see source) | 150 | accuracy | 0.7067 | B | Z | yes |
| Natural language inference - XNLI (Swahili) | facebook/xnli / sample (per-task; see source) | 150 | accuracy | 0.7 | B | Z | yes |
| Natural language inference - XNLI (Chinese) | facebook/xnli / sample (per-task; see source) | 150 | accuracy | 0.7933 | B | Z | yes |

#### Ibrahim & Zaki 2026 (arXiv 2609.24574), CSS annotation (18 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| Conversations Gone Awry (toxicity forecasting) | Ziems et al. 2024 CSS task 'conv_go_awry' / conversation / confirmatory split (study's own) | 500 | macro-F1 | 0.5015 | B | Z | yes |
| Discourse acts | Ziems et al. 2024 CSS task 'discourse' / conversation / discovery split (study's own) | 497 | macro-F1 | 0.5952 | B | Z | yes |
| Emotion (CARER) | Ziems et al. 2024 CSS task 'emotion' / utterance / confirmatory split (study's own) | 498 | macro-F1 | 0.4836 | B | Z | yes |
| FLUTE figurative language | Ziems et al. 2024 CSS task 'flute' / utterance / confirmatory split (study's own) | 500 | macro-F1 | 0.8633 | B | Z | yes |
| Ideological Books Corpus | Ziems et al. 2024 CSS task 'ibc' / utterance / confirmatory split (study's own) | 498 | macro-F1 | 0.6325 | B | Z | yes |
| Implicit/latent hate | Ziems et al. 2024 CSS task 'implicit_hate' / utterance / discovery split (study's own) | 498 | macro-F1 | 0.4392 | B | Z | yes |
| Indian English dialect features | Ziems et al. 2024 CSS task 'indian_english_dialect' / utterance / confirmatory split (study's own) | 266 | macro-F1 | 0.6401 | B | Z | yes |
| Media ideology (document) | Ziems et al. 2024 CSS task 'media_ideology' / document / confirmatory split (study's own) | 498 | macro-F1 | 0.654 | B | Z | yes |
| Misinfo Reaction Frames | Ziems et al. 2024 CSS task 'mrf' / utterance / confirmatory split (study's own) | 500 | macro-F1 | 0.7955 | B | Z | yes |
| Persuasion strategies (conversation) | Ziems et al. 2024 CSS task 'persuasion' / conversation / confirmatory split (study's own) | 434 | macro-F1 | 0.5605 | B | Z | yes |
| Random Acts of Pizza (persuasion) | Ziems et al. 2024 CSS task 'raop' / utterance / confirmatory split (study's own) | 399 | macro-F1 | 0.6074 | B | Z | yes |
| Reddit humor | Ziems et al. 2024 CSS task 'reddit_humor' / utterance / confirmatory split (study's own) | 500 | macro-F1 | 0.5633 | B | Z | yes |
| SemEval-2016 T6 stance | Ziems et al. 2024 CSS task 'semeval_stance' / utterance / discovery split (study's own) | 435 | macro-F1 | 0.7341 | B | Z | yes |
| TalkLife empathy | Ziems et al. 2024 CSS task 'talklife' / conversation / confirmatory split (study's own) | 498 | macro-F1 | 0.2688 | B | Z | yes |
| TempoWiC semantic change | Ziems et al. 2024 CSS task 'tempowic' / utterance / confirmatory split (study's own) | 344 | macro-F1 | 0.6826 | B | Z | yes |
| Character tropes | Ziems et al. 2024 CSS task 'tropes' / document / confirmatory split (study's own) | 114 | macro-F1 | 0.1905 | C | Z | yes |
| Wikipedia power/talk | Ziems et al. 2024 CSS task 'wiki_corpus' / conversation / confirmatory split (study's own) | 500 | macro-F1 | 0.5813 | B | Z | yes |
| Wikipedia politeness | Ziems et al. 2024 CSS task 'wiki_politeness' / conversation / confirmatory split (study's own) | 498 | macro-F1 | 0.573 | B | Z | yes |

#### AbdelStark/jev-benchmarks (BTZSC pilot v1) (3 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| AG News (BTZSC) | btzsc/btzsc / agnews / test (100 balanced) | 100 | accuracy | 0.91 | C | Z |  |
| BANKING77 (BTZSC, 72 labels, no-positive rows dropped) | btzsc/btzsc / banking77 / test (100 balanced) | 100 | accuracy | 0.87 | C | Z |  |
| DAIR Emotion (BTZSC) | btzsc/btzsc / emotiondair / test (100 balanced) | 100 | accuracy | 0.48 | C | Z |  |

#### elcronos/jev-vs-open-decision-models (5 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| DAIR Emotion | dair-ai/emotion / split / test | 2,000 | accuracy | 0.587 | A | Z | yes |
| DAIR Emotion (labels with one-line definitions) | dair-ai/emotion / split / test | 2,000 | accuracy | 0.599 | A | Z | yes |
| TweetTopic single (6 classes) | cardiffnlp/tweet_topic_single / test_2021 | 1,693 | accuracy | 0.793 | A | Z | yes |
| Twitter financial news topic (20 classes) | zeroshot/twitter-financial-news-topic / validation | 4,117 | accuracy | 0.67 | A | Z | yes |
| DailyDialog emotion (utterances, 7 classes) | OpenRL/daily_dialog (mirror of li2017dailydialog/daily_dialog) / test (flattened) | 7,740 | macro-F1 | 0.385 | A | Z | yes |

#### zhuyansen/jev-zeroshot-vs-bert (6 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| AG News | fancyzhx/ag_news / test (1,000 strat.) | 1,000 | accuracy | 0.865 | B | Z |  |
| SST-2 | stanfordnlp/sst2 / validation (all 872) | 872 | accuracy | 0.96 | A | Z |  |
| BANKING77 (two-step group-then-label) | PolyAI/banking77 / test (1,000 strat.) | 1,000 | accuracy | 0.712 | B | Z |  |
| TweetEval emotion (4 classes) | cardiffnlp/tweet_eval / emotion / test (1,000) | 1,000 | accuracy | 0.827 | B | Z |  |
| PAWS (sentence pair) | google-research-datasets/paws / labeled_final / test (1,000) | 1,000 | accuracy | 0.855 | B | Z |  |
| arXiv primary category, papers submitted >= 2026-09-17 | arXiv API (collected by author) / 8 categories / post-release | 258 | accuracy | 0.891 | C | Z |  |

#### thisisandreeeee/jev-benchmarks (5 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| BANKING77 | PolyAI banking77 (SPACE-2 release alignment) / test | 3,080 | accuracy | 0.799 | A | Z |  |
| CLINC150 (in-scope only, OOS excluded) | CLINC150 (SPACE-2 release) / test (in-scope) | 4,500 | accuracy | 0.9196 | A | Z |  |
| HWU64 | HWU64 (SPACE-2 release) / test | 1,076 | accuracy | 0.8309 | A | Z |  |
| SST-2 (Noul) | stanfordnlp/sst2 / validation | 872 | accuracy | 0.945 | A | Z |  |
| STS-B (6-level Score rubric) | sentence-transformers/stsb / test | 1,379 | Spearman | 0.8921 | A | Z |  |

#### manojlds/classifier-bench (2 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| BANKING77 (definitions only) | PolyAI/banking77 / test | 3,080 | accuracy | 0.801 | A | Z |  |
| BANKING77 (+2 training examples per label in criteria) | PolyAI/banking77 / test | 3,080 | accuracy | 0.8529 | A | S |  |

#### simonmesmith/jev-banking77-experiment (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| BANKING77 (definitions + 24 BM25-retrieved train examples) | PolyAI-LDN/task-specific-datasets @57ec275 / test | 3,080 | accuracy | 0.924 | A | S | yes |

#### Alexander-Ollman/laya-ft (7 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| BANKING77 | PolyAI/banking77 / test | 3,080 | accuracy | 0.8 | A | Z | yes |
| Prompt-injection detection | deepset/prompt-injections / test (116) | 116 | macro-F1 | 0.787 | B | Z | yes |
| SMS spam | ucirvine/sms_spam / plain_text / frozen test (1,000) | 1,000 | macro-F1 | 0.908 | B | Z | yes |
| Emotion | dair-ai/emotion / frozen test (1,000) | 1,000 | macro-F1 | 0.497 | B | Z | yes |
| Product-review counterfactuals | SetFit/amazon_counterfactual / en / frozen test (670) | 670 | macro-F1 | 0.865 | B | Z | yes |
| Assistant request routing (MASSIVE en-US, 60 intents) | AmazonScience/massive / en-US / frozen test (1,000) | 1,000 | macro-F1 | 0.799 | B | Z | yes |
| XSTest harmless prompts (false-alarm rate, lower is better) | XSTest (250 safe prompts) / all | 250 | false-alarm rate | 0.076 | B | Z | yes |

#### chepyle/jev-test (LexGLUE etc.) (12 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| LexGLUE ECtHR A | coastalcph/lex_glue / ecthr_a / test | 1,000 | micro-F1 | 0.73 | A | Z |  |
| LexGLUE ECtHR B | coastalcph/lex_glue / ecthr_b / test | 1,000 | micro-F1 | 0.754 | A | Z |  |
| LexGLUE SCOTUS | coastalcph/lex_glue / scotus / test | 1,400 | micro-F1 | 0.726 | A | Z |  |
| LexGLUE EUR-LEX | coastalcph/lex_glue / eurlex / test | 5,000 | micro-F1 | 0.391 | A | Z |  |
| LexGLUE LEDGAR | coastalcph/lex_glue / ledgar / test | 10,000 | micro-F1 | 0.753 | A | Z |  |
| LexGLUE UNFAIR-ToS | coastalcph/lex_glue / unfair_tos / test | 1,607 | micro-F1 | 0.764 | A | Z |  |
| LexGLUE CaseHOLD | coastalcph/lex_glue / case_hold / test | 3,600 | micro-F1 | 0.773 | A | Z |  |
| LexGLUE 7-task mean | coastalcph/lex_glue / all 7 / test | 23,607 | arithmetic mean micro-F1 | 0.699 | A | Z |  |
| LexGLUE 7-task mean (per-label thresholds tuned on validation) | coastalcph/lex_glue / all 7 / test | 23,607 | arithmetic mean micro-F1 | 0.742 | A | S |  |
| BANKING77 | PolyAI-LDN/task-specific-datasets @57ec275 / test | 3,080 | accuracy | 0.806 | A | Z |  |
| CLINC150 OOS+ (in-scope accuracy) | clinc/clinc_oos @155b9c7 / plus / test | 5,500 | in-scope accuracy | 0.89 | A | Z |  |
| CLINC150 OOS+ (OOS recall) | clinc/clinc_oos @155b9c7 / plus / test | 1,000 | oos_recall | 0.881 | A | Z |  |

#### Kev suites (run by chepyle/jev-test) (4 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| Kev transfer-v4 dev | jaredpalmer/kev frozen suites (public sources) / transfer-v4 / dev | 656 | accuracy | 0.855 | B | Z |  |
| Kev transfer-v4 test | jaredpalmer/kev frozen suites (public sources) / transfer-v4 / test | 656 | accuracy | 0.877 | B | Z |  |
| Kev decision-v7 dev | jaredpalmer/kev frozen suites (public sources) / decision-v7 / dev | 1,264 | accuracy | 0.845 | B | Z |  |
| Kev decision-v7 test | jaredpalmer/kev frozen suites (public sources) / decision-v7 / test | 1,200 | accuracy | 0.835 | B | Z |  |

#### Kev (jaredpalmer/kev README) (3 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| Kev transfer-v4 dev (new sources) | jaredpalmer/kev suites / transfer-v4 / dev | 764 | accuracy | 0.857 | B | Z |  |
| MMLU (Kev harness) | cais/mmlu / test (subset) |  | accuracy | 0.9 | C | Z |  |
| MMLU-Pro (Kev harness) | TIGER-Lab/MMLU-Pro / test (subset) |  | accuracy | 0.84 | C | Z |  |

#### Kev (robustness index) (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| WANLI | alisawuffles/WANLI / test (256 pairs) | 256 | accuracy | 0.758 | C | Z |  |

#### WebJev (lexmount) single-step benchmarks (8 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| JevBench public | fstandhartinger/jevbench / test |  | accuracy | 0.8571 | B | Z |  |
| MMLU-Pro (1,000) | TIGER-Lab/MMLU-Pro / test | 1,000 | accuracy | 0.834 | B | Z |  |
| typed-decisions | LocalLLaMA/typed-decisions / test |  | accuracy | 0.7405 | B | Z |  |
| Kev transfer-v4 | jaredpalmer/kev suites / test |  | accuracy | 0.8521 | B | Z |  |
| Kev decision-v7 | jaredpalmer/kev suites / test |  | accuracy | 0.8331 | B | Z |  |
| Nimble holdout | Bespoke Nimble eval / test |  | accuracy | 0.9259 | B | Z |  |
| SemIf | TheoLeeCJ/SemIf eval / test |  | accuracy | 0.9841 | B | Z |  |
| scienthoon tickets | scienthoon/jev-ood-calibration synthetic tickets / test |  | accuracy | 0.7491 | B | Z |  |

#### mbburabak/jev-safety-benchmark (8 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| HateCheck (hateful) | Paul/hatecheck / test | 3,728 | F1 (harmful class) | 0.9923 | A | Z |  |
| ToxicChat (toxic) | lmsys/toxic-chat / test | 2,853 | F1 (harmful class) | 0.7572 | A | Z |  |
| Aegis v1 unsafe prompt | nvidia/Aegis-AI-Content-Safety-Dataset-1.0 / test | 359 | F1 (harmful class) | 0.8909 | A | Z |  |
| Aegis v2 unsafe prompt | nvidia/Aegis-AI-Content-Safety-Dataset-2.0 / test | 1,928 | F1 (harmful class) | 0.8357 | A | Z |  |
| Aegis v2 unsafe response | nvidia/Aegis-AI-Content-Safety-Dataset-2.0 / test | 1,928 | F1 (harmful class) | 0.8017 | A | Z |  |
| WildGuardTest prompt harmful | allenai/wildguardmix (wildguardtest; public mirror used) / test | 1,699 | F1 (harmful class) | 0.8843 | A | Z |  |
| HarmBench harmful request (detection rate) | HarmBench (841-case convention) / test | 239 | detection rate | 0.9916 | A | Z |  |
| HarmBench harmful response | HarmBench / test | 596 | F1 (harmful class) | 0.8746 | A | Z |  |

#### Gaurav-Gosain/jev-sec-bench (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| deepset prompt-injections (all 662, with deployment context) | deepset/prompt-injections / train+test (662) | 662 | accuracy | 0.965 | A | Z | yes |

#### aniruddh-krovvidi/switchboard (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| deepset prompt-injections (662) | deepset/prompt-injections / train+test (662) | 662 | F1 @ threshold 0.1 | 0.938 | A | Z | yes |

#### ca7ai/jev-prompt-sentry (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| deepset prompt-injections (546) false-negative rate | deepset/prompt-injections / train (546) | 546 | false-negative rate (lower is better) | 0.473 | B | Z |  |

#### ASEVlad/jev-injection-bench (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| Combined injection corpus (4 HF datasets, dedup) | xTRam1/safe-guard-prompt-injection + jackhhao/jailbreak-classification + deepset/prompt-injections + leolee99/NotInject / all (11,900) | 11,900 | AUPRC | 0.98 | A | Z |  |

#### Red Hat Developer (NeMo Guardrails EvalHub) (2 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| Prompt injection (EvalHub NeMo Guardrails benchmark) | EvalHub 'prompt-injection' config / - |  | accuracy | 0.8635 | C | Z |  |
| Content safety (toxicity-profanity-safety) | EvalHub 'toxicity-profanity-safety' config / - |  | accuracy | 0.862 | C | Z |  |

#### goya4140/jev-reward-model-evaluation (8 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| RewardBench v1 | allenai/reward-bench / test | 2,985 | official 4-section macro | 0.9258 | A | Z |  |
| RewardBench 2 | allenai/reward-bench-2 / test | 1,865 | official 6-domain macro | 0.8115 | A | Z |  |
| RM-Bench structured pairwise | THU-KEG/RM-Bench / test | 1,327 | official 4-domain macro | 0.8129 | A | Z |  |
| RM-Bench pointwise | THU-KEG/RM-Bench / test | 7,962 | official 4-domain macro | 0.8379 | A | Z |  |
| RubricBench (human rubric given) | RubricBench / test | 1,147 | pairwise accuracy | 0.7602 | A | Z |  |
| PPE Human Preference V1 | lmarena-ai/PPE-Human-Preference-V1 / test | 16,038 | no-tie pairwise accuracy | 0.644 | A | Z |  |
| ProcessBench | Qwen/ProcessBench / test | 3,400 | official mean F1 | 0.6951 | A | Z |  |
| PRMBench Preview | hitsmy/PRMBench_Preview / test | 6,216 | official PRM score | 0.6638 | A | Z |  |

#### stperic/jev-medhallu-benchmark (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| MedHallu | UTAustin-AIHealth/MedHallu / test | 1,000 | accuracy | 0.929 | A | S | yes |

#### lianghsun/jev-tmmluplus-eval (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| TMMLU+ v1.1 (66 subjects) | ikala/tmmluplus / v1.1 tag / test | 19,646 | category macro accuracy | 0.7757 | A | Z |  |

#### kuaitoukuai/jev-gaokao-eval (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| GAOKAO-Bench objective single-choice | OpenLMLab/GAOKAO-Bench / objective / all testable | 1,497 | accuracy | 0.9085 | A | Z |  |

#### kokuren333/jev-jmle-benchmark (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| JMedQA (Japanese medical licensing, 2018-2026) | JMedQA / all | 3,556 | exact-set accuracy | 0.8858 | A | Z |  |

#### Saeedabdf/jev-screening-benchmark (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| Cohen 2006 systematic-review screening (15 reviews) | asreview systematic-review-datasets Cohen_2006 / all | 16,015 | recall (include) | 0.9 | A | Z |  |

#### PistachioAIHQ/jev-synergy-screening (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| Cohen 2006 ADHD abstracts | asreview Cohen_2006 ADHD / all | 851 | accuracy | 0.946 | B | Z |  |

#### drvegabermudez/jev-sat5-text-only-evaluation (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| SAT practice tests (11) | College Board SAT practice tests / all | 1,009 | accuracy | 0.912 | B | Z |  |

#### patryckalves/jev-no-enem (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ENEM 2025 | ENEM 2025 exam (INEP) / valid items | 182 | accuracy | 0.566 | C | Z |  |

#### vehas/thaiexam-jev-charts (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ThaiExam | scb10x/thai_exam / calibration subset | 567 | accuracy | 0.707 | C | Z |  |

#### shunmoridev/jev-jp-accounting-bench (2 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| JMMLU professional_accounting | nlp-waseda/JMMLU / professional_accounting / test | 150 | accuracy | 0.7467 | C | Z |  |
| jfinqa | jfinqa / test | 1,000 | accuracy | 0.731 | B | Z |  |

#### NomaDamas/kojev (5 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| KoBEST BoolQ | skt/kobest_v1 / boolq / 80 sampled | 80 | accuracy | 0.988 | C | Z |  |
| KoBEST COPA | skt/kobest_v1 / copa / 80 sampled | 80 | accuracy | 0.988 | C | Z |  |
| KoBEST WiC | skt/kobest_v1 / wic / 80 sampled | 80 | accuracy | 0.888 | C | Z |  |
| KoBEST HellaSwag | skt/kobest_v1 / hellaswag / 80 sampled | 80 | accuracy | 0.775 | C | Z |  |
| KoBEST SentiNeg | skt/kobest_v1 / sentineg / 80 sampled | 80 | accuracy | 0.938 | C | Z |  |

#### 4esv/jev-eval (4 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| CLINC150 incl. OOS (151 options) | clinc/clinc_oos / plus / test (300) | 300 | accuracy | 0.897 | B | Z |  |
| BANKING77 | mteb/banking77 / test (300) | 300 | accuracy | 0.78 | B | Z |  |
| SST-5 (Score) | SetFit/sst5 / test (300) | 300 | accuracy | 0.57 | B | Z |  |
| IMDB as noul | stanfordnlp/imdb / test (300) | 300 | accuracy | 0.97 | B | Z |  |

#### heswy/Jev-Benchmark (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| 5-dataset mean (BANKING77 300, CLINC150+OOS 200, SST-5 200, BoolQ 200, AG News 150) | mixed public test/val / test/val | 1,050 | equal-weight mean accuracy | 0.7937 | B | Z | yes |

#### dhruvmehra/jevbench (3 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| AG News | fancyzhx/ag_news / test (500) | 500 | accuracy | 0.843 | B | Z |  |
| BANKING77 | mteb/banking77 / test (500) | 500 | accuracy | 0.764 | B | Z |  |
| SST-2 | stanfordnlp/sst2 / validation (500) | 500 | accuracy | 0.954 | B | Z |  |

#### onlyoneaman/jev-eval (4 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| Enron spam | SetFit/enron_spam / test (300) | 300 | accuracy | 0.987 | B | Z |  |
| SST-2 | stanfordnlp/sst2 / 300 | 300 | accuracy | 0.957 | B | Z |  |
| AG News | fancyzhx/ag_news / test (300) | 300 | accuracy | 0.913 | B | Z |  |
| BANKING77 | PolyAI/banking77 / test (300) | 300 | accuracy | 0.76 | B | Z |  |

#### mugenkyou/JEV-VS-ML (4 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| IMDb (balanced accuracy, raw) | stanfordnlp/imdb / holdout 1,000 | 1,000 | balanced accuracy | 0.963 | B | Z |  |
| BANKING77 | PolyAI/banking77 / holdout 1,500 | 1,500 | accuracy | 0.789 | B | Z |  |
| AG News | fancyzhx/ag_news / holdout 1,000 | 1,000 | accuracy | 0.875 | B | Z |  |
| SMS spam | ucirvine/sms_spam / holdout 1,000 | 1,000 | balanced accuracy | 0.961 | B | Z |  |

#### dylantom2012 / zhlei07 open-system-one (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| 4-dataset macro (SST-2, AG News, Emotion, BANKING77; 2,500 each) | stanfordnlp/sst2, fancyzhx/ag_news, dair-ai/emotion, PolyAI/banking77 / 2,500 per dataset | 10,000 | macro accuracy | 0.793 | B | Z | yes |

#### actuallyrizzn/decision-systems-bakeoff (3 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| SST-2 | stanfordnlp/sst2 / validation | 872 | accuracy | 0.946 | A | Z |  |
| CLINC150 + OOS (partial: 4,060 of 5,500) | clinc/clinc_oos / plus / test (partial) | 4,060 | accuracy | 0.917 | C | Z |  |
| Eclipse bug-title severity | Eclipse bug reports / 2,000 | 2,000 | accuracy | 0.491 | B | Z |  |

#### ickma2311/jev-baselines-eval (2 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| BANKING77 (paired subset) | PolyAI/banking77 / 208 | 208 | accuracy | 0.832 | B | Z |  |
| CLINC150 | clinc/clinc_oos / 200 | 200 | accuracy | 0.87 | B | Z |  |

#### Kumzha/jev-benchmark (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| BANKING77 | PolyAI/banking77 / 200 | 200 | accuracy | 0.815 | B | Z |  |

#### MohtashamMurshid/jev-speed-test (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| BANKING77 | PolyAI/banking77 / 500 held-out test | 500 | accuracy | 0.81 | B | Z |  |

#### saurabhkumar8112/jev-gpt5-routing-study (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| BANKING77 | PolyAI/banking77 / 500 | 500 | accuracy | 0.832 | B | Z |  |

#### cocodedk/jev-bench (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| BANKING77 (4 per intent) | PolyAI/banking77 / 308 | 308 | accuracy | 0.779 | B | Z |  |

#### FirasSX914/Janus (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| BANKING77 (described labels) | PolyAI/banking77 / 500 | 500 | accuracy | 0.778 | B | Z | yes |

#### shivpratapsinghpanwar/edgefront_jev (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| BANKING77 | PolyAI/banking77 / 300 | 300 | accuracy | 0.79 | B | Z |  |

#### nikkoxgonzales/jev-certify (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| CLINC150 in-scope top-1 (held out) | clinc/clinc_oos / 400 | 400 | in-scope top-1 accuracy | 0.93 | B | Z | yes |

#### dshvimer/jev-eval (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| AG News | fancyzhx/ag_news / test (100, seed 42) | 100 | accuracy | 0.87 | C | Z |  |

#### oluies/jev-vs-spacy (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| MASSIVE Swedish routing | AmazonScience/massive sv-SE / 20 per scenario x 18 | 360 | accuracy | 0.889 | B | Z |  |

#### xxkuboxx/jev-eval (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| tanaos synthetic intent classifier | tanaos/synthetic-intent-classifier-dataset-v1 / test | 3,447 | accuracy | 0.8784 | B | Z |  |

#### mouadse/jev-vs-laya (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| Darija reviews (Moroccan Arabic) | Darija Reviews (HF) / frozen holdout | 171 | accuracy | 0.795 | C | Z |  |

#### bitnovus/jev-spam-eval (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| Email spam/phishing main set | public email corpora (Enron/Ling-Spam/phishing) / main | 5,733 | accuracy | 0.9864 | C | Z |  |

#### Arize blog (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| Email spam (18,514 emails) | public email-spam dataset (Kaggle) / all | 18,514 | accuracy | 0.983 | B | Z |  |

#### scienthoon/jev-ood-calibration (3 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| OpenBookQA | allenai/openbookqa / main / validation | 500 | accuracy | 0.942 | A | Z | yes |
| CommonsenseQA | tau/commonsense_qa / validation | 1,221 | accuracy | 0.881 | A | Z | yes |
| HellaSwag | Rowan/hellaswag / validation (2,000) | 2,000 | accuracy | 0.861 | B | Z | yes |

#### GautamTalksDev/jevbench (ChaosNLI) (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ChaosNLI MNLI subset (750 low + 750 high disagreement) | metaeval ChaosNLI (MNLI portion) / test | 1,500 | bias-corrected delta-ECE (hard - easy) | 0.264 | X | Z | yes |

#### anisselbd/jev-phishing-bench (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| PhishNChips phishing verdict | AreLit/PhishNChips / core_emails.csv / all | 2,000 | accuracy | 0.626 | A | Z | yes |

#### manjunathshiva/jev-frontier-bench (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| Pooled 4 tasks (BANKING77, BoolQ, Yelp-5, ChaosNLI; 50 each) | PolyAI/banking77 + google/boolq + yelp/yelp_review_full + ChaosNLI / test/val samples | 200 | accuracy | 0.725 | C | Z |  |

#### adorosario/jev-rag-claim-verification (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| LLM-AggreFact / MiniCheck claims | lytang/LLM-AggreFact / test (495) | 495 | balanced accuracy | 0.733 | B | Z |  |

#### slavadubrov/sgr-judge-bench (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| TabFact (120 tables/claims) | wenhu/tab_fact / 120 balanced | 120 | accuracy | 0.917 | C | Z |  |

#### jev-ids/jev-ids (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| NSL-KDD flows (pilot) | NSL-KDD / 300 flows x 3 seeds | 900 | F1 (attack) | 0.859 | C | Z | yes |

#### geckguy/job-posting-triage (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| Real/Fake job postings | Kaggle fake job postings / 1,000 of 3,182 test | 1,000 | PR-AUC (fraud) | 0.276 | B | Z |  |

#### goodrahstar/jev-column-race (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| Android app reviews rating | sealuzh/app_reviews / 1,000 | 1,000 | Spearman | 0.803 | B | Z |  |

#### matu79go/jev-hanko (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| CUAD clause detection (41 clauses x 500 pages) | theatticusproject/cuad / 500 pages | 20,500 | F1 | 0.519 | B | Z |  |

#### TokenTrim/jev-agent-failure-benchmark (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| Who&When Pro agent-failure attribution | Who&When Pro (arXiv 2607.09996) / all | 6,257 | 'All' score | 0.313 | B | Z |  |

#### hosamsh/jev-mind2web (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| Mind2Web train_6 shard (planning top-1) | osunlp/Mind2Web / train_6 / 706 steps | 706 | top-1 accuracy | 0.521 | B | Z |  |

#### Gaurav-Gosain/jev-headline-bench (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| Upworthy headline A/B (confirmatory pairs) | Upworthy Research Archive / confirmatory | 10,984 | accuracy | 0.645 | A | Z |  |

#### dchristopoulos/jev-bench (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| Reddit AITA verdicts | Reddit AITA (stratified) / 770 | 770 | accuracy | 0.754 | B | Z |  |

#### keltokhy/jsort (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| CommonLit readability (300 excerpts) | CommonLit readability / 300 of 4,724 | 300 | Pearson r | 0.824 | B | Z |  |

#### denser-org/rerank-bench-jev (2 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| BEIR SciFact rerank (BM25 top-100) | BeIR/scifact / test | 300 | nDCG@10 | 0.7699 | A | Z |  |
| BEIR NFCorpus rerank | BeIR/nfcorpus / test | 323 | nDCG@10 | 0.3623 | A | Z |  |

#### hev/reranker (3 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| BEIR SciFact rerank (BM25 top-30, per pair) | BeIR/scifact / test | 300 | nDCG@10 | 0.772 | A | Z |  |
| BEIR NFCorpus rerank | BeIR/nfcorpus / test | 323 | nDCG@10 | 0.358 | A | Z |  |
| BEIR FiQA rerank | BeIR/fiqa / test (300) | 300 | nDCG@10 | 0.376 | B | Z |  |

#### anessbelbati/jev-rerank-bench (2 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| BEIR+BRIGHT+CodeSearchNet rerank (8 sets, dataset mean) | BEIR/MTEB sets / test | 1,617 | mean nDCG@10 | 0.692 | A | Z | yes |
| NevIR negation pairs | orionweller/NevIR / test (1,383 pairs) | 1,383 | pairwise accuracy | 0.71 | A | Z | yes |

#### gazelle93/decision-models-under-pressure (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| Label-set pressure (CLINC/MTOP/GoEmotions/DBpedia/fin tweets), K=128 candidates | clinc_oos + mteb/mtop_intent + go_emotions + dbpedia_14 + fin tweets / frozen items |  | accuracy at 128 candidates | 0.6 | C | Z |  |

#### Adkid-Zephyr/chinese-workflow-decision-bench (Feishu-style, mirrored in Laya repo) (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| Chinese workplace decisions, single Choice | synthetic Feishu-style scenarios (Laya repo research/benchmarks/feishu_zh/data/cases.jsonl) / all | 64 | accuracy (first repeat) | 1 | X | Z | yes |

#### JoeSlain/jev-gliclass-bench (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| typed-decisions customer_service (100 rows) | LocalLLaMA/typed-decisions / customer_service / 100 (seed 42) | 100 | accuracy | 0.78 | C | Z |  |

#### QuicqDev/Jev-vs-ML (protocol V3.0.1) (9 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ AG News | fancyzhx/ag_news / sample (holdout seed 20260920) | 1,000 | balanced accuracy (raw decision) | 0.875 | B | Z |  |
| ★ BANKING77 | PolyAI/banking77 / sample (holdout seed 20260920) | 1,500 | balanced accuracy (raw decision) | 0.789 | B | Z |  |
| ★ SMS Spam | ucirvine/sms_spam / sample (holdout seed 20260920) | 1,000 | balanced accuracy (raw decision) | 0.961 | B | Z |  |
| ★ IMDb | stanfordnlp/imdb / sample (holdout seed 20260920) | 1,000 | balanced accuracy (raw decision) | 0.963 | B | Z |  |
| ★ BANKING77 (1 example per class in prompt) | PolyAI/banking77 / sample (holdout seed 20260920) | 1,500 | balanced accuracy | 0.819 | B | S |  |
| ★ UCI Bank Marketing (tabular) | UCI / scikit-learn dataset / sample (holdout seed 20260920) | 1,000 | balanced accuracy (raw decision) | 0.534 | B | Z |  |
| ★ UCI Online Shoppers Intention (tabular) | UCI / scikit-learn dataset / sample (holdout seed 20260920) | 1,000 | balanced accuracy (raw decision) | 0.514 | B | Z |  |
| ★ Breast Cancer Wisconsin diagnostic (tabular) | UCI / scikit-learn dataset / sample (holdout seed 20260920) | 114 | balanced accuracy (raw decision) | 0.61 | C | Z |  |
| ★ Iris (tabular) | UCI / scikit-learn dataset / sample (holdout seed 20260920) | 30 | balanced accuracy (raw decision) | 0.97 | C | Z |  |

#### maybern-tripp-smith/cuad-jev-bench (2 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ CUAD hard pairs (gold clause span vs same-contract BM25 distractor) | theatticusproject/cuad / pre-registered pairs (Stratum B) | 200 | pairwise accuracy (1 - inversion rate, both orders averaged) | 0.68 | B | Z | yes |
| ★ CUAD clause retrieval (100 category queries) | theatticusproject/cuad / 100 queries | 100 | MRR | 0.917 | C | Z | yes |

#### kiwi0719/jev-edge (2 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ deepset prompt-injections, text only (jev-edge injection template) | deepset/prompt-injections / train+test (662) | 662 | ROC-AUC | 0.983 | A | Z | yes |
| ★ deepset prompt-injections, with deployment context | deepset/prompt-injections / train+test (662) | 662 | ROC-AUC | 0.996 | A | Z | yes |

#### CompleteDotTech/paper-package (run-20260918) (3 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ DBLP-ACM entity matching (identity-disjoint held-out pairs) | DBLP-ACM (Magellan/DeepMatcher benchmark) / held-out evaluation (413 pairs) | 413 | macro-F1 | 0.9605 | B | Z | yes |
| ★ DBLP-ACM entity matching (6 train demonstrations) | DBLP-ACM (Magellan/DeepMatcher benchmark) / held-out evaluation (413 pairs) | 413 | macro-F1 | 0.9859 | B | S |  |
| ★ SciFact claim-evidence relation (SUPPORTS/REFUTES/NEI) | allenai/scifact / held-out evaluation (339) | 339 | macro-F1 | 0.8508 | B | Z |  |

#### EmreKaplaner/rag-jev (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ BEIR SciFact rerank (BM25 top-20) | BeIR/scifact / test | 300 | nDCG@10 | 0.7513 | A | Z |  |

#### emretheus/jev-rag-benchmark (2 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ BEIR SciFact rerank (hybrid retrieval candidates, batch Noul) | BeIR/scifact / test | 300 | nDCG@10 | 0.7929 | A | Z |  |
| ★ XQuAD-EN passage rerank | google/xquad / xquad.en (1,190) | 1,190 | nDCG@10 | 0.9893 | A | Z |  |

#### romeromarcelo/jev-retrieval (jevr) (3 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ BEIR SciFact (BM25 stage 1 + Jev verification pipeline) | BeIR/scifact / test | 300 | nDCG@10 | 0.778 | B | Z |  |
| ★ BEIR NFCorpus (BM25 stage 1 + Jev verification pipeline) | BeIR/nfcorpus / test | 323 | nDCG@10 | 0.363 | B | Z |  |
| ★ HAKARI-Bench NanoRTEB, reranking mode (14 tasks) | hakari-bench NanoRTEB / 2,390 judged queries | 2,390 | mean nDCG@10 | 0.79 | A | Z |  |

#### zwliJay/jev-forge (JevForge) (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ JevForge-Mind2Web action choice (website-disjoint test) | AndeyTait/JevForge-Mind2Web / test | 800 | choice top-1 accuracy | 0.543 | C | Z |  |

#### trifleen/jev-vs-luna-phishing (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ Phishing emails, balanced 50/50 sample | zefang-liu/phishing-email-dataset / sample (seed 42) | 100 | accuracy | 0.81 | C | Z | yes |

#### rupeshpoojary9/poorjev (crossbench) (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ BANKING77 (77-way) | PolyAI/banking77 / sample | 154 | accuracy | 0.812 | C | Z | yes |

#### abhisheksharma001/jev-skill (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ BANKING77 (77-way routing) | PolyAI/banking77 / sample | 134 | accuracy (first-draft question) | 0.761 | C | Z |  |

#### Jev in Medicine (arXiv 2609.34024) (4 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ MetaMedQA (USMLE-style, 6 options incl. none-of-the-above / I don't know) | maximegmd/MetaMedQA / test (all) | 1,373 | top-1 accuracy | 0.748 | A | Z |  |
| ★ PubMedQA (expert-annotated official test split, yes/no/maybe) | qiaojin/PubMedQA / pqa_labeled / test (500) | 500 | top-1 accuracy | 0.784 | A | Z |  |
| ★ DiagnosisArena-MCQ (4 options) | DiagnosisArena (MCQ version; HF id not stated in paper) / all (915) | 915 | top-1 accuracy | 0.598 | A | Z |  |
| ★ NEJM Case Challenges (6 options) | NEJM Case Records (not on HF) / 34 cases | 34 | top-1 accuracy | 0.618 | C | Z |  |

#### JEV as a Judge for Agent Trace Security (arXiv 2609.34862) (5 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ R-Judge agent-trajectory safety (risk >= 3, revised labels) | R-Judge (benchmark release; HF id not stated) / all | 564 | positive-class F1 (on valid responses) | 0.885 | B | Z |  |
| ★ ATBench500 agent-trajectory safety (risk >= 3, revised labels) | ATBench500 (benchmark release; HF id not stated) / all | 500 | positive-class F1 (on valid responses) | 0.938 | B | Z |  |
| ★ TraceSafe agent-trajectory safety (risk >= 3, revised labels) | TraceSafe (benchmark release; HF id not stated) / all | 540 | positive-class F1 (on valid responses) | 0.678 | B | Z |  |
| ★ MCPHunt agent-trajectory safety (risk >= 3, revised labels) | MCPHunt (benchmark release; HF id not stated) / all | 3,615 | positive-class F1 (on valid responses) | 0.611 | A | Z |  |
| ★ Four agent-trace benchmarks, unweighted mean | R-Judge + ATBench500 + TraceSafe + MCPHunt / all | 5,219 | mean positive-class F1 | 0.778 | B | Z |  |

#### JEV vs LLMs as Rubric Judges (arXiv 2609.29769) (9 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ RiceChem rubric criteria (full set) | RiceChem (Sonkar et al. 2024) / all 1,240 responses x criteria | 8,392 | verdict accuracy (Jev Choice) | 0.814 | A | Z |  |
| ★ HealthBench rubric criteria (200 completions, physician labels) | openai/healthbench (consensus subset) / stratified sample | 406 | verdict accuracy (Jev Choice) | 0.771 | B | Z |  |
| ★ ELLIPSE essay traits (5 levels) | ELLIPSE corpus (test essays) / 10-20% sample of public set | 1,548 | exact accuracy vs human label (Jev Choice) | 0.134 | B | Z |  |
| ★ FED-Turn dialogue quality (3 levels) | FED (Mehri & Eskenazi 2020) / 10-20% sample of public set | 600 | exact accuracy vs human label (Jev Choice) | 0.562 | B | Z |  |
| ★ FED-Dialogue quality (3 levels) | FED (Mehri & Eskenazi 2020) / 10-20% sample of public set | 250 | exact accuracy vs human label (Jev Choice) | 0.486 | B | Z |  |
| ★ HelpSteer2 attributes (5 levels) | nvidia/HelpSteer2 / 10-20% sample of public set | 360 | exact accuracy vs human label (Jev Choice) | 0.486 | B | Z |  |
| ★ LFQA answer quality (3-4 levels) | LFQA human eval (Xu et al. 2023) / 10-20% sample of public set | 360 | exact accuracy vs human label (Jev Choice) | 0.687 | B | Z |  |
| ★ USR-TopicalChat (3 levels) | USR (Mehri & Eskenazi 2020) / 10-20% sample of public set | 360 | exact accuracy vs human label (Jev Choice) | 0.62 | B | Z |  |
| ★ USR-PersonaChat (3 levels) | USR (Mehri & Eskenazi 2020) / 10-20% sample of public set | 300 | exact accuracy vs human label (Jev Choice) | 0.564 | B | Z |  |

#### JEV-as-a-Judge (arXiv 2609.26550) (8 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ RewardBench preference pairs | allenai/reward-bench / sample | 1,500 | accuracy | 0.925 | A | Z |  |
| ★ JudgeBench (GPT split) | ScalerLab/JudgeBench / sample | 350 | accuracy | 0.786 | B | Z |  |
| ★ HaluEval QA | pminervini/HaluEval / sample | 3,000 | accuracy | 0.873 | A | Z |  |
| ★ HaluEval summarization | pminervini/HaluEval / sample | 400 | accuracy | 0.698 | B | Z |  |
| ★ HaluEval general (reference-free) | pminervini/HaluEval / sample | 200 | accuracy | 0.535 | B | Z |  |
| ★ RM-Bench normal | THU-KEG/RM-Bench / sample | 3,000 | accuracy | 0.854 | A | Z |  |
| ★ RM-Bench hard (style-adversarial) | THU-KEG/RM-Bench / sample | 3,000 | accuracy | 0.766 | A | Z |  |
| ★ RewardBench 2 four-way | allenai/reward-bench-2 / sample | 100 | accuracy | 0.73 | C | Z |  |

#### JevVibe (arXiv 2609.34963) (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ CyberSecEval Instruct, 50-way CWE classification | meta-llama/PurpleLlama CyberSecEval instruct (filtered to 50 CWEs) / 1,916 examples | 1,916 | top-1 accuracy | 0.458 | A | Z |  |

#### Ordinal-Scale Bias in JEV-like Models (arXiv 2609.38827) (2 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ ANLI R1-R3 (dev + test) | facebook/anli / dev_r1-3 + test_r1-3 | 6,400 | accuracy | 0.7495 | A | Z |  |
| ★ 36 ordinal rating datasets, macro-average | 36 public ordinal sets (listed in paper appendix) / per-dataset samples |  | macro accuracy | 0.3767 | X | Z |  |

#### Benchmarking System One models vs trained classifiers (arXiv 2610.00346) (6 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ typed-decisions (card eval set) | LocalLLaMA/typed-decisions / eval (2,000 decisions) | 2,000 | accuracy | 0.732 | B | Z |  |
| ★ CLINC150 (600 in-scope + 200 OOS) | clinc/clinc_oos / plus / test sample (4 per intent + 200 oos) | 800 | accuracy | 0.913 | B | Z |  |
| ★ Conversations Gone Awry (derailment) | ConvoKit conversations-gone-awry / sample | 500 | accuracy | 0.558 | B | Z |  |
| ★ Wikipedia power (admin vs non-admin) | ConvoKit wiki-corpus / sample | 500 | accuracy | 0.608 | B | Z |  |
| ★ Twitter emotion (hashtag labels, 6 classes) | dair-ai/emotion / sample | 498 | accuracy | 0.504 | B | Z |  |
| ★ Wikipedia politeness (3 classes) | ConvoKit wikipedia-politeness / sample | 498 | accuracy | 0.657 | B | Z |  |

#### Do System One Decisions Add Up? (arXiv 2609.33971) (3 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ TREC fine-grained question type (50 labels) | CogComp/trec / fine_label / test (all 500) | 500 | accuracy (flat fine-label) | 0.722 | A | Z |  |
| ★ CLINC150 in-scope (label-balanced sample) | clinc/clinc_oos / plus / test sample | 1,000 | accuracy (flat fine-label) | 0.91 | B | Z |  |
| ★ MASSIVE en-US intents (label-balanced sample) | AmazonScience/massive / en-US / test sample | 1,000 | accuracy (flat fine-label) | 0.833 | B | Z |  |

#### Same Scores, Different Decisions (arXiv 2609.27678) (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ ContractNLI (17 hypotheses x 123 contracts, 3-way) | stanfordnlp/contract-nli (GitHub; HF mirror kiddothe2b/contract-nli) / official test (123 contracts) | 2,091 | accuracy | 0.7738 | A | Z |  |

#### Beyond Answer Confidence (arXiv 2610.01006) (10 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ BANKING77, intent names only | PolyAI/banking77 / test sample (10 per intent) | 770 | accuracy | 0.821 | B | Z |  |
| ★ BANKING77, intent names + 8 train examples per intent | PolyAI/banking77 / test sample (10 per intent) | 770 | accuracy | 0.909 | B | S |  |
| ★ CLINC150 in-scope, intent names only | clinc/clinc_oos / plus / test sample (10 per intent) | 1,500 | accuracy | 0.926 | B | Z |  |
| ★ CLINC150 in-scope, intent names + 8 train examples per intent | clinc/clinc_oos / plus / test sample (10 per intent) | 1,500 | accuracy | 0.981 | B | S |  |
| ★ TriviaQA as 4-option Choice | mandarjoshi/trivia_qa / all / pinned revision | 9,960 | accuracy | 0.955 | A | Z |  |
| ★ PopQA as 4-option Choice (real subjects) | akariasai/PopQA / all / pinned revision | 14,267 | accuracy | 0.708 | A | Z |  |
| ★ SimpleQA Verified as 4-option Choice | google/simpleqa-verified / all / pinned revision | 1,000 | accuracy | 0.753 | A | Z |  |
| ★ MMLU-CF (contamination-free MMLU) | microsoft/MMLU-CF / all / pinned revision | 10,000 | accuracy | 0.776 | A | Z |  |
| ★ Daily Oracle yes/no news forecasting | agentic-learning-ai-lab/daily-oracle / all / pinned revision | 6,320 | accuracy | 0.651 | A | Z |  |
| ★ TruthfulQA binary (correct vs the dataset's incorrect answer) | truthfulqa/truthful_qa / all |  | accuracy | 0.909 | B | Z |  |

#### Chinese-Jev (arXiv 2609.36965) (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ CJ-Bench general subset (Chinese choice/noul/score decisions from held-out public Chinese datasets) | CJ-Bench (authors' benchmark; release not confirmed) / benchmark (100,000 decisions) | 100,000 | accuracy | 0.6835 | B | Z |  |

#### Evaluating System One Models for Agent Security (arXiv 2609.33401) (3 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ WAInjectBench-text prompt-injection detection | WAInjectBench (text partition) / test (authors' dedup partition) | 1,612 | macro-F1 | 0.756 | A | Z |  |
| ★ R-Judge interaction-risk classification | R-Judge (Yuan et al. 2024) / test (authors' partition) | 236 | macro-F1 | 0.825 | B | Z |  |
| ★ AgentHarm harmful-request classification | ai-safety-institute/AgentHarm / official test (44 task groups) | 352 | macro-F1 | 0.84 | B | Z |  |

#### Just Ask Jev / RLCDAlignBench (arXiv 2609.29429) (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ RLCDAlignBench: 31 alignment-failure benchmarks with a Noul form (responses from small target models, mostly judge-labelled) | 38 public benchmark files (HarmBench, JailbreakBench, MASK, InjecAgent, LLM-AggreFact, RAGTruth, ...) / per-benchmark samples |  | median AUROC (generic Noul) | 0.886 | X | Z |  |

#### Beyond Calibration: probability axioms (arXiv 2609.33209) (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ Negation coherence on ChaosNLI + PubMedQA items (3 labels each) | earino/chaosnli + qiaojin/PubMedQA / 160 items (480 negation pairs) | 160 | mean /P(X)+P(not X)-1/ (lower is better) | 0.064 | X | Z |  |

#### Can Jev Judge Radiology Reports? (arXiv 2609.27607) (3 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ RadEvalX report-pair error counts | RadEvalX / all | 100 | Kendall tau vs expert total errors (Jev-OneQ) | 0.573 | C | Z |  |
| ★ RadEvalExpert report-pair error counts | RadEvalExpert / all | 624 | Kendall tau vs expert total errors (Jev-OneQ) | 0.398 | B | Z |  |
| ★ ReXErr sentence error detection (all error types) | ReXErr / 10,790 negatives + 8,723 positives | 19,513 | AUROC | 0.8896 | A | Z |  |

#### Decide, Don't Generate: DimABSA (arXiv 2609.35293) (2 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ SemEval-2026 Task 3 Track A DimABSA, Task 1 valence-arousal regression (10 corpora, 6 languages) | SemEval-2026 Task 3 DimABSA official data / official test |  | aggregate RMSE (lower is better) | 1.0645 | A | S | yes |
| ★ SemEval-2026 Task 3 Track A DimABSA, triplet extraction | SemEval-2026 Task 3 DimABSA official data / official test |  | continuous F1 | 0.5209 | A | S | yes |

#### Jev for Speech-Neuroprosthesis Rescoring (arXiv 2609.33538) (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ Brain-to-text candidate-sentence rescoring (ALS participant, published decoder n-best lists) | Card et al. 2024 brain-to-text data (Dryad; not on HF) / test (978 sentences) | 978 | word error rate (lower is better) | 0.0746 | B | S |  |

#### Decision-Oriented Recommendation Reranking (arXiv 2609.40241) (3 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ Amazon Reviews 2023 Movies and TV, next-item rerank of SASRec top-K hard candidates (K=20) | McAuley-Lab/Amazon-Reviews-2023 / 5core_last_out_w_his_Movies and TV / test users (eligible: gold in SASRec top-200) | 954 | MRR (K=20) | 0.238 | C | Z |  |
| ★ Amazon Reviews 2023 Video Games, next-item rerank of SASRec top-K hard candidates (K=20) | McAuley-Lab/Amazon-Reviews-2023 / 5core_last_out_w_his_Video Games / test users (eligible: gold in SASRec top-200) | 1,000 | MRR (K=20) | 0.295 | C | Z |  |
| ★ Amazon Reviews 2023 Books, next-item rerank of SASRec top-K hard candidates (K=20) | McAuley-Lab/Amazon-Reviews-2023 / 5core_last_out_w_his_Books / test users (eligible: gold in SASRec top-200) | 626 | MRR (K=20) | 0.346 | C | Z |  |

#### Training-free HAR with Jev (arXiv 2609.36154) (3 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ WISDM accelerometer windows (class-balanced, text-described features) | WISDM / class-balanced sample (600 windows) | 600 | macro-F1 (best representation) | 0.0378 | C | Z |  |
| ★ UCI HAR (UCI341) accelerometer windows (class-balanced, text-described features) | UCI HAR (UCI341) / class-balanced sample (600 windows) | 600 | macro-F1 (best representation) | 0.1184 | C | Z |  |
| ★ PAMAP2 accelerometer windows (class-balanced, text-described features) | PAMAP2 / class-balanced sample (600 windows) | 600 | macro-F1 (best representation) | 0.0893 | C | Z |  |

#### Jev for Network Traffic Classification (arXiv 2610.00376) (2 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ CESNET-QUICEXT-25, 10 application labels from first 10 packets | CESNET-QUICEXT-25 / 26 test weeks after training period | 52,000 | accuracy | 0.098 | A | Z |  |
| ★ CESNET-QUICEXT-25 with 40 fixed labelled examples in context | CESNET-QUICEXT-25 / 26 test weeks after training period | 52,000 | accuracy | 0.2842 | A | S |  |

#### Jev-IDS (arXiv 2610.01079) (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ NSL-KDD flows, 2,000-flow evaluation set | NSL-KDD (UNB CIC) / evaluation set (2,000 flows) | 2,000 | F1 (attack class) | 0.782 | A | Z | yes |

#### JEVQA video quality (arXiv 2609.24395) (2 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ AOM CTC encodes (AV1/H.264/HEVC/VP9, 22 sources), metadata only, VMAF as ground truth | AOM CTC encode set / all | 1,936 | PLCC vs VMAF | 0.737 | X | Z |  |
| ★ AVT-VQDB-UHD-1 subjective quality, metadata only | AVT-VQDB-UHD-1 / all |  | PLCC vs MOS | 0.747 | B | Z |  |

#### Decision Hijacking (arXiv 2609.28613) (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ InjecAgent tool-response injection (reconstructed cases) | InjecAgent (Zhan et al. 2024) / 510 clean-qualified cases | 510 | attack success rate on fresh validation calls (lower is better) | 0.035 | X | Z |  |

#### JET: Justification Evaluation in Transformer (arXiv 2609.33874) (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ MMLU (57 subjects) | cais/mmlu / all / test (full) | 14,042 | accuracy | 0.8906 | A | Z |  |

#### LLM2Jev (arXiv 2610.02076) (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ JevBench v1.4.2.2 public set | fstandhartinger/jevbench (v1.4.2.2 public split) / public |  | accuracy | 0.866 | B | Z |  |

#### this-that-model-1.0 (arXiv 2609.23886) (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ this-that released benchmark (15 synthetic question families, maze states) | this-that-model released benchmark / 2,250-question subset | 2,250 | accuracy | 0.803 | X | Z |  |

#### JevOut: Natural Context Can Flip Decision Models (arXiv 2609.30243) (7 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ MMLU-Pro (clean accuracy before attack) | TIGER-Lab/MMLU-Pro / held-out sample (100 items) | 100 | accuracy (clean) | 0.83 | C | Z | yes |
| ★ SuperGPQA (clean accuracy before attack) | m-a-p/SuperGPQA / held-out sample (100 items) | 100 | accuracy (clean) | 0.46 | C | Z | yes |
| ★ MuSR (clean accuracy before attack) | TAUR-Lab/MuSR / held-out sample (100 items) | 100 | accuracy (clean) | 0.61 | C | Z | yes |
| ★ ToMBench (clean accuracy before attack) | ToMBench / held-out sample (100 items) | 100 | accuracy (clean) | 0.72 | C | Z | yes |
| ★ LAR-ECHR (clean accuracy before attack) | LAR-ECHR / held-out sample (100 items) | 100 | accuracy (clean) | 0.8 | C | Z | yes |
| ★ SATA (select-all-that-apply) (clean accuracy before attack) | SATA-Bench / held-out sample (100 items) | 100 | accuracy (clean) | 0.823 | C | Z | yes |
| ★ BFCL V4 (function selection) (clean accuracy before attack) | gorilla-llm/Berkeley-Function-Calling-Leaderboard / held-out sample (100 items) | 100 | accuracy (clean) | 0.7 | C | Z | yes |

#### Sys1Cal-v1 (arXiv 2609.35342) (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ Sys1Cal-v1 known-probability True/False items, Noul primitive | little-g-ai/Sys1Cal-v1 (GitHub) / v1 (365 rendered examples from 92 problems) | 365 | mean OVL soft accuracy (1 - TV) | 0.918 | C | Z | yes |

#### Code Owns the Simulation, Jev Owns the Evaluation (arXiv 2610.01834) (2 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ Cognitive Reflection Test items (Hagendorff et al. 2023 set) as 4-option Choice | CRT items from Hagendorff et al. 2023 (Nature Comp. Sci.) / 150 lure questions x 4 option orders | 150 | fraction correct | 0.99 | C | Z |  |
| ★ ALFWorld unseen games, closed loop (Jev chooses each admissible command) | ALFWorld (Shridhar et al. 2021) / valid_unseen (134 games, max 50 steps) | 134 | success rate | 0.313 | B | Z |  |

#### Decision Readouts for Video Anomaly Detection (arXiv 2609.34180) (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ XD-Violence caption anchors (text-mediated) | XD-Violence (Wu et al. 2020) / development sample (20 videos, 200 anchors, 37 positives) | 200 | average precision (Jev Noul) | 0.7599 | C | Z |  |

#### HydroJEV (arXiv 2610.02048) (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ C-Town EPANET cause attribution (attack / fault / transient / sensor), in distribution | HydroJEV benchmark (github.com/mutianwei521/hydrojev) / four sealed pre-registered rounds |  | macro-F1 | 0.62 | X | Z | yes |

#### Koa-action (arXiv 2609.36115) (2 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ SST-2 | stanfordnlp/sst2 / validation (872, labelled "full test set") | 872 | accuracy | 0.961 | A | Z |  |
| ★ Amazon Reviews Polarity | fancyzhx/amazon_polarity / test sample (5,000) | 5,000 | accuracy | 0.968 | A | Z |  |

#### jujumilk3/jev-calibration-audit (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ KoBBQ ambiguous (unanswerable) items with the "unknown" option | naver-ai/kobbq / sample (300 ambiguous items) | 300 | accuracy (choosing unknown) | 0.95 | C | Z | yes |

#### zilliztech/vector-graph-rag (Jev reranker evaluation) (2 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ MuSiQue multi-hop retrieval (Vector Graph RAG + Jev relation reranker) | bdsaglam/musique (HippoRAG eval rows) / 500 rows (frozen Contriever candidates) | 500 | Recall@5 | 0.6887 | C | Z | yes |
| ★ HotpotQA multi-hop retrieval (Vector Graph RAG + Jev relation reranker) | hotpotqa/hotpot_qa (HippoRAG eval rows) / 500 rows (490 unique ids) | 500 | Recall@5 | 0.935 | B | Z | yes |

#### statsguysam/jev-classification-benchmark (5 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ SST-2 (held-out from labelled validation) | stanfordnlp/sst2 / validation sample (200) | 200 | accuracy | 0.935 | B | Z | yes |
| ★ TREC coarse question type (6 classes) | CogComp/trec / coarse_label / test sample (200) | 200 | accuracy | 0.335 | B | Z | yes |
| ★ TREC coarse question type, 4 train examples per class | CogComp/trec / coarse_label / test sample (200) | 200 | accuracy | 0.855 | B | S |  |
| ★ Breast Cancer Wisconsin diagnostic (tabular) | UCI / scikit-learn dataset / test | 114 | accuracy | 0.842 | C | Z |  |
| ★ Wine (tabular, 3 classes) | UCI / scikit-learn dataset / test | 36 | accuracy | 0.333 | C | Z |  |

#### earino/zero-shot-complaint-benchmark (CFPB 113-class) (2 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ CFPB consumer complaints, 113 issue labels (bare labels) | determined-ai/consumer_complaints_medium / evaluation rows (6,430) | 6,430 | accuracy | 0.3442 | A | Z | yes |
| ★ CFPB consumer complaints, 113 labels with form instruction + label definitions | determined-ai/consumer_complaints_medium / evaluation rows (6,430) | 6,430 | accuracy | 0.4649 | A | Z | yes |

#### morrenhale/decision-benchmark-jev-laya-julia (2026-09-27) (5 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ AG News | fancyzhx/ag_news / test (full) | 7,600 | accuracy | 0.8924 | A | Z | yes |
| ★ DAIR Emotion (6 classes) | dair-ai/emotion / split / test (full) | 2,000 | accuracy | 0.5885 | A | Z | yes |
| ★ BANKING77 (77 options per call) | PolyAI/banking77 / test (3,076 used) | 3,076 | accuracy | 0.8001 | A | Z | yes |
| ★ MASSIVE scenario (18 classes, 51 locales x 100) | AmazonScience/massive / all locales / test sample (SHA256 selection, seed 20260927) | 5,100 | accuracy | 0.6767 | A | Z | yes |
| ★ typed-decisions test parquet | LocalLLaMA/typed-decisions / test (600 choice + 800 score + 600 noul) | 2,000 | accuracy | 0.732 | B | Z | yes |

#### KikiNLP/CanITrustYou-Jev (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ Agent execution decisions adapted from 10 public sources (AgentRewardBench, SWE-agent trajectories, AgentHarm, ...), original + 5 perturbations | KikiNLP/CanITrustYou-Jev / test (1,974 of 2,000 questions after 26 failed) | 1,974 | strict accuracy (all 6 versions correct) | 0.7452 | B | Z | yes |

#### atmaneayoub/jev-ar-bench (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ Gulf-Arabic government-service routing (UAE + KSA), 60 routes | atmaneayoub/jev-ar-bench / gov_test / test | 4,974 | accuracy | 0.862 | B | Z |  |

#### yehor-oleksiuk/bonzi-vs-jev-wanli256 (1 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ WANLI NLI (256-item subset) | alisawuffles/WANLI / test subset (256) | 256 | accuracy | 0.793 | C | Z |  |

#### Lightfield blog (Testing TypeSafe Jev on Text Understanding) (7 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ CommonsenseQA | tau/commonsense_qa / sample (seed 42) | 100 | accuracy | 0.91 | C | Z |  |
| ★ MMLU-CF | microsoft/MMLU-CF / sample (seed 42) | 100 | accuracy | 0.8 | C | Z |  |
| ★ RACE-H | ehovy/race / sample (seed 42) | 100 | accuracy | 0.95 | C | Z |  |
| ★ Amazon QA answerability yes/no (Noul) | Amazon QA (McAuley, UCSD) / test sample (seed 13) | 500 | accuracy | 0.628 | B | Z |  |
| ★ Persuasion for Good: did they donate? (Noul) | Persuasion for Good (Wang et al. 2019) / test sample (seed 13) | 500 | accuracy | 0.712 | B | Z |  |
| ★ Persuasion for Good: persuasion strategy, 18-way (Choice) | Persuasion for Good (Wang et al. 2019) / test sample (seed 13) | 500 | accuracy | 0.442 | B | Z |  |
| ★ CraigslistBargain: reached a deal? (Noul) | CraigslistBargain (He et al. 2018) / test sample (seed 13) | 500 | accuracy | 0.918 | B | Z |  |

#### AY Automate blog (Jev vs GPT and Claude) (3 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ BANKING77, 8-intent subset | PolyAI/banking77 / test sample (20 per intent) | 160 | accuracy | 0.838 | C | Z |  |
| ★ BANKING77, 77 intents | PolyAI/banking77 / test sample (3 per intent) | 231 | accuracy | 0.788 | B | Z |  |
| ★ deepset prompt-injections (400 of 662) | deepset/prompt-injections / random 400 of train+test | 400 | accuracy | 0.87 | B | Z |  |

#### SOTAAZ blog (Kev vs Jev) (8 rows)

| Dataset | HF id / config / split | n | Metric | Jev | Tier | Track | Raw |
|---|---|---:|---|---:|:-:|:-:|:-:|
| ★ BANKING77, 77 intents | PolyAI/banking77 / test sample (same rows as the Kev posts) | 154 | accuracy | 0.76 | C | Z |  |
| ★ TREC coarse, label names | CogComp/trec / coarse_label / test sample (same rows as the Kev posts) | 500 | accuracy | 0.89 | B | Z |  |
| ★ TREC coarse, label names + one-line descriptions | CogComp/trec / coarse_label / test sample (same rows as the Kev posts) | 500 | accuracy | 0.936 | B | Z |  |
| ★ AG News, label names | fancyzhx/ag_news / test sample (same rows as the Kev posts) | 200 | accuracy | 0.89 | B | Z |  |
| ★ AG News, label names + descriptions | fancyzhx/ag_news / test sample (same rows as the Kev posts) | 200 | accuracy | 0.9 | B | Z |  |
| ★ CLINC150, 150 intents + out-of-scope | clinc/clinc_oos / plus / test sample (same rows as the Kev posts) | 400 | accuracy | 0.685 | B | Z |  |
| ★ MASSIVE en-US, 60 intents | AmazonScience/massive / en-US / test sample (same rows as the Kev posts) | 175 | accuracy | 0.829 | C | Z |  |
| ★ Twitter financial news topics (20) | zeroshot/twitter-financial-news-topic / test sample (same rows as the Kev posts) | 200 | accuracy | 0.71 | B | Z |  |

---

## 6. Source URLs (read 2026-10-02; resume pass 2026-10-03)

- Deußer et al.: https://arxiv.org/abs/2609.37647 · https://github.com/AppliedMachineLearning-Lab/jev-benchmarking (results/eval, docs/datasets.md, responses/LICENSE_RESPONSES.md) · https://doi.org/10.5281/zenodo.23039006
- Decision Index: https://huggingface.co/spaces/multimodalart/jev-decision-index (data/index.json, index-v0.1.json, index-v2.json) · https://github.com/apolinario/decision-index (docs/suite.md)
- DMB: https://github.com/nibzard/decision-model-benchmark (results/expanded-jev-2026-09-29, recent-pilot-2026-09-26, decisions-preview-controls-2026-09-30, v3)
- Jevals: https://jevals.com · https://github.com/Jevals/jevals-data
- typed-decisions: https://huggingface.co/datasets/LocalLLaMA/typed-decisions
- TypeSafe evals: https://evals.typesafe.ai · https://typesafe.ai/blog/introducing-system-one-models-and-jev
- JevBench: https://benchmarkheaven.com/api/jevbench/v1.5.5 · https://github.com/fstandhartinger/jevbench
- DecisionBench: https://github.com/Hanno-Labs/decision-bench-results · DecideBench: https://github.com/choyiny/decidebench
- LangWatch: https://langwatch.ai/compare/jev-benchmark
- jev-bench (Jevify): https://huggingface.co/datasets/Praveenrajus/jev-bench
- Laya: https://github.com/NandhaKishorM/laya (README "Laya (with routing) vs Jev", BENCHMARKS.md, research/benchmarks/feishu_zh)
- OpenRouter Jev Lab: https://openrouter.ai/labs/jev (no accuracy claims)
- Robustness index (catalog of about 280 Jev studies, CC0): https://github.com/Yifan-Lan/awesome-jev-robustness (data/task_benchmarks.tsv, entries.tsv)
- Awesome lists used for discovery: https://github.com/AbdelStark/awesome-typesafe-jev · https://github.com/OmniJev/awesome-jev-gallery
- Every other study URL is in the JSON `source_url` field.

- **Resume pass, arXiv papers** (HTML at `arxiv.org/html/<id>`): https://arxiv.org/abs/2609.23886 · https://arxiv.org/abs/2609.24395 · https://arxiv.org/abs/2609.26550 · https://arxiv.org/abs/2609.27607 · https://arxiv.org/abs/2609.27678 · https://arxiv.org/abs/2609.28613 · https://arxiv.org/abs/2609.29429 · https://arxiv.org/abs/2609.29769 · https://arxiv.org/abs/2609.30243 · https://arxiv.org/abs/2609.33209 · https://arxiv.org/abs/2609.33401 · https://arxiv.org/abs/2609.33538 · https://arxiv.org/abs/2609.33874 · https://arxiv.org/abs/2609.33971 · https://arxiv.org/abs/2609.34024 · https://arxiv.org/abs/2609.34180 · https://arxiv.org/abs/2609.34862 · https://arxiv.org/abs/2609.34963 · https://arxiv.org/abs/2609.35293 · https://arxiv.org/abs/2609.35342 · https://arxiv.org/abs/2609.36115 · https://arxiv.org/abs/2609.36154 · https://arxiv.org/abs/2609.36965 · https://arxiv.org/abs/2609.38827 · https://arxiv.org/abs/2609.40241 · https://arxiv.org/abs/2610.00346 · https://arxiv.org/abs/2610.00376 · https://arxiv.org/abs/2610.01006 · https://arxiv.org/abs/2610.01079 · https://arxiv.org/abs/2610.01834 · https://arxiv.org/abs/2610.02048 · https://arxiv.org/abs/2610.02076
- **Resume pass, other sources:** https://github.com/CompleteDotTech/paper-package/blob/main/tables/TABLES.md · https://github.com/EmreKaplaner/rag-jev · https://github.com/QuicqDev/Jev-vs-ML/blob/main/published_results/raw_balanced_accuracy.csv · https://github.com/abhisheksharma001/jev-skill · https://github.com/earino/zero-shot-complaint-benchmark · https://github.com/emretheus/jev-rag-benchmark · https://github.com/jujumilk3/jev-calibration-audit · https://github.com/kiwi0719/jev-edge/blob/main/bench/report.md · https://github.com/maybern-tripp-smith/cuad-jev-bench · https://github.com/romeromarcelo/jev-retrieval/blob/main/docs/BENCHMARKS.md · https://github.com/rupeshpoojary9/poorjev · https://github.com/statsguysam/jev-classification-benchmark · https://github.com/statsguysam/jev-classification-benchmark/blob/main/results/TABULAR_FINDINGS.md · https://github.com/trifleen/jev-vs-luna-phishing · https://github.com/zilliztech/vector-graph-rag/blob/main/docs/evaluation.md · https://github.com/zwliJay/jev-forge · https://huggingface.co/datasets/KikiNLP/CanITrustYou-Jev · https://huggingface.co/datasets/atmaneayoub/jev-ar-bench · https://huggingface.co/datasets/morrenhale/decision-benchmark-jev-laya-julia · https://huggingface.co/datasets/yehor-oleksiuk/bonzi-vs-jev-wanli256 · https://lightfield.app/blog/testing-typesafe-jev-on-text-understanding · https://sotaaz.com/post/kev-vs-jev-en · https://www.ayautomate.com/blog/jev-vs-llm-benchmark
- **Discovery used in the resume pass:** arXiv API (`export.arxiv.org/api/query`, queries listed in §0.1) · https://github.com/Yifan-Lan/awesome-jev-robustness `data/task_benchmarks.tsv` (pulled 2026-10-03, pushed 04:08Z) · https://github.com/Eurekaleo/awesome-jev-survey · https://github.com/youzizzz1028/Awesome-Jev · HF search `huggingface.co/api/datasets?search=jev` and `/api/spaces?search=jev` (sorted by last modified) · web search for blogs.
