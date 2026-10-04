# Benchmax: training data and supervised ceilings for every public Jev benchmark dataset

Research notes, 2026-10-02; **addendum 2026-10-03 (§11–§13)** extends coverage to every dataset in the finished master list (`jev-published.md` §2, 397 rows / ~270 keys). Everything was read-only:
- HF Hub API (`/api/datasets/<id>`) and `datasets-server.huggingface.co/size`, swept on 2026-10-02 for about 190 dataset ids;
- GitHub API reads of the benchmark kits (Decision Index `apolinario/decision-index`, Deußer `AppliedMachineLearning-Lab/jev-benchmarking`, DecisionBench) and small result JSONs;
- papers, model cards and leaderboards.

Nothing was downloaded beyond JSON, READMEs and two paper PDFs (4 pages each). No az, no torch, nothing installed. Fetched content was treated as data.

**Two tracks, as defined by the task:**
- **Z (zero-shot):** no benchmark data of any kind in training.
- **S (supervised / specialist):** the benchmark's official TRAIN split (and validation, for dev or training) is allowed. TEST is never touched. Disclosed.

The parallel "jev-published" researcher owns the master list of Jev numbers. The Jev numbers here were copied from primary files wherever possible:
- Deußer `results/eval/*.json` and `summary.md`;
- the Decision Index Space `data/index.json`, generated 2026-09-28T00:39Z;
- the study READMEs cited in `docs/research/v2/classifier-benchmarks.md`.

Tags:
- **[F]** fetched and read this session (source given);
- **[R]** recalled from the cited paper, not re-read this session (verify before quoting externally);
- **[E]** estimate;
- **[I]** inference.

---

## 0. Bottom line

### 0.1 S-track scoreboard by public evidence

**A. Winnable with an encoder of ≤150M parameters trained on the official train split.** In each case a public supervised model of ≤125M beats Jev's published number by ≥3 points, or the gap is very large.

| Dataset | Jev | Public supervised reference |
|---|---|---|
| dair-ai emotion | 0.585 | DistilBERT-66M 0.927 |
| GoEmotions macro-F1 | 0.243 (0.353 with tuned thresholds) | BERT-base 0.46 |
| Banking77, vs zero-shot Jev | 0.797 | BERT 0.937 |
| Financial PhraseBank | 0.730 | FinBERT 0.86, same split |
| fin-topic | 0.670 | FinBERT-tone 0.911 |
| tweet_topic | 0.793 | RoBERTa 0.896 (base not measured; large shown) |
| NLU++ micro-F1 | 0.483 | RoBERTa-QA 0.80–0.93 |
| deepset prompt-injections | 0.741 | DeBERTa-v3-base 0.991 |
| AGB-DE F1 | 0.204 | BERT 0.35 |
| PAWS | 0.850 | BERT ≈ 0.90–0.92 |
| AG News | 0.885 | BERT-base 0.948 |
| SMS spam F1 | 0.938 | ≥0.97. There is no train split, so this counts only as **S-cv** (k-fold out-of-fold), not S |
| HoVer | 0.729 | BERT-base 0.812 (oracle evidence) |
| ChessBench | 0.172 | **9M** searchless-chess 0.642 action accuracy |
| VAST | 0.646 | 0.71–0.80 |
| daily_dialog macro-F1 | 0.385 | RoBERTa 0.51 (6-class) |
| STS-B Spearman | 0.890 | Ettin-68M 0.911 (dev, GLUE metric) |
| CLINC150 macro-F1 | 0.893 / 0.901 | in-scope accuracy alone is 0.967 for BERT |

**B. Needs a model of about 350–400M (Ettin-400M, ModernBERT-L or DeBERTa-v3-L class) plus care. The public margin is about 0–3 points.**

| Dataset | Jev | Public supervised reference | Size needed |
|---|---|---|---|
| SST-2 | 0.964 | Ettin-400M 0.967; Ettin-150M 0.958 loses | 400M |
| SST-5 | 0.579 | RoBERTa-L 0.602 | ~400M |
| IMDB | 0.965 | RoBERTa-L 0.965, XLNet-L 0.968 | large, long context |
| αNLI | 0.839 | RoBERTa-L 0.856 dev | ~350M |
| RAGTruth F1 | 0.765 | LettuceDetect-large 0.792; -base 0.761 is a tie | 400M |
| ToxicChat F1 | 0.786 | ToxicChat-T5-large 0.822 | 770M |
| typed-decisions | 0.727 | OpenDecider-nano (Ettin-400M) 0.796 | 400M |
| Banking77, vs few-shot Jev | **0.924** | BERT 0.937 | ~110M+, tight |
| CLINC150 accuracy | 0.895 | BERT 2019: 0.899 | needs better OOS recall |
| TweetEval emotion | 0.827 | twitter-RoBERTa-base 0.832 | tie at 125M |
| iSarcasmEval F1 | 0.505 | best SemEval system 0.605 | ? |
| Humicroedit | 0.619 | 0.674 | ? |
| AfriXNLI | 0.640 | AfroXLMR-76L 0.657 (African languages only) | 560M **multilingual** |

**C. Not winnable at ≤1B on any public evidence.** The best public supervised small model is below Jev:

| Dataset | Jev | Best public supervised (or small-model) result |
|---|---|---|
| ANLI | 0.739 | 0.702 |
| BoolQ | 0.913 | 0.884 |
| rotten_tomatoes | 0.933 | 0.91 |
| language-id | 0.996 | 0.996, a tie |
| SIB-200 | 0.815 | 0.759 |
| Belebele | 0.867 | 0.60 |
| PubMedQA | 0.787 | 0.722 |
| LLM-AggreFact | 0.786 | 0.756 at 0.4B; 0.774 at 7B |
| SummEval | 0.554 | 0.474–0.514 |
| HellaSwag | 0.945–0.955 | 0.852 |
| WinoGrande | 0.914–0.920 | 0.791 |
| CommonsenseQA | 0.882 | 0.841 |
| NLI4CT | 0.841 | 0.74 for DeBERTa-L; 0.80 for 7B |
| New Yorker | 0.701 | T5-11B 0.708 |

The same holds for the whole knowledge and reasoning block (MMLU, C-Eval, BIG-bench MC, ARC, OBQA, MMLU-Pro, BBH, GPQA, HLE, GSM8K, CRUXEval, MuSR, SATA-Bench, BFCL). There, the best ≤0.6B entrant on the Decision Index scores 0.16–0.47 against Jev's 0.73–0.96.

**D. No train split usable under S's rules.** These are Z only, or need k-fold "S-cv" with disclosure:
- LLM-AggreFact: the card forbids fine-tuning on its dev or test.
- OpenAI moderation eval.
- SummEval.
- Belebele.
- AfriXNLI: has a validation split only.
- PubMedQA: Jev was scored on all 1,000 labelled items.
- SMS spam: only one split exists.
- FinEntity: the Decision Index scores all 979 documents.
- CLadder: the Decision Index samples the whole balanced set.
- The Decision Index's MMLU-Pro, BBH, GPQA, HLE, MuSR, CRUXEval, SATA-Bench, BFCL, BRIGHT, PhishNChips, BPoMP, POP909-CL and Home-appliance rows.
- DecisionBench, DecideBench, JevBench sealed items, TypeSafe evals.

**E. Unknown: there is no public supervised number in the right format, so it must be measured.**
- ToxiGen annotated, OpenAI moderation (per-category), HelpSteer2 (per-attribute ρ), UNFAIR-ToS (8-category micro-F1 without LexGLUE's "none" class).
- ContractNLI, ACOS (category-sentiment projection), Amazon ESCI (macro-F1), When2Call, ToolRet, API-Bank, PhishNChips, cfcolor, Habermas, ForecastBench, POP909-CL, BPoMP.

### 0.2 Corrections to our earlier notes, found this session

1. **The LLM-AggreFact 0.786 is a mean of per-source balanced accuracies over 11 sources.** It is not pooled; the pooled balanced accuracy is 0.831. This makes it directly comparable to the LLM-AggreFact leaderboard. Jev (78.6) is above every row the leaderboard page showed (11 of 39), including Bespoke-MiniCheck-7B at 77.4 and FactCG-DeBERTa-L at 75.6 [F: Deußer `results/eval/llm_aggrefact.json`; llm-aggrefact.github.io]. `classifier-benchmarks.md` §1.1 says "pooled", which is wrong.
2. **Deußer published tuned-threshold Jev numbers as well.** The thresholds were tuned on 1,000 dev items.
   - GoEmotions macro-F1: 0.353 tuned against 0.243 fixed.
   - UNFAIR-ToS micro-F1: 0.748 tuned against 0.499 fixed.

   "Beat every published number" means beating the **tuned** ones [F: `results/eval/thresholds.json`].
3. **There is a published "Jev given training data" number for Banking77: 92.40%.** Setup: full official test of 3,080; jev-1.13.0; category definitions plus 24 BM25-retrieved train examples per request; 2026-09-18 [F: github.com/simonmesmith/jev-banking77-experiment]. This is the natural S-track target for Banking77, and it is only 1.26 points under fine-tuned BERT (93.66).
4. **The ANLI 0.81/0.72/0.72 for ModernBERT-large-zeroshot-v2.0 is binary accuracy (entailment vs not_entailment), not 3-class.** The best public supervised 3-class encoder is DeBERTa-v3-large-mnli-fever-anli-ling-wanli at **0.702** on ANLI-all, below Jev's 0.739 [F: both model cards; the zeroshot-v2.0 card says "`entailment` vs. `not_entailment`"].
5. **CLINC150, Jev detail:**

   | Metric | Value |
   |---|---|
   | In-scope accuracy | 0.921 |
   | OOS recall | 0.776 |
   | OOS precision | 0.941 |
   | Macro-F1 | 0.901 |

   [F: Deußer `clinc150.json`]. BERT in the original paper (oos-train, OOS+) gets in-scope 0.967 and OOS recall 0.592, which works out to 0.899 overall accuracy. That barely clears Jev. The S-track win depends on OOS handling, not in-scope accuracy [F: Larson et al. 2019 Table 2].
6. **Contaminated public resources** (details in §7):
   - `KoalaAI/Text-Moderation` was trained on `mmathys/openai-moderation-api-evaluation`, the eval itself.
   - `nickmuchi/financial-classification` contains Financial PhraseBank.
   - `dair-ai/emotion` config `unsplit` (416,809 rows) is the full CARER pool and contains the test texts.
   - `tasksource/bigbench` train splits contain BBH test items.
   - MMLU-Pro contains filtered MMLU-test, TheoremQA and SciBench items.
   - When2Call test items were built from BFCL (the Decision Index stores `bfcl_source_id`).

### 0.3 What "max them all" really requires

- With a ≤150M encoder, S can credibly beat Jev on about 18 datasets (group A). With an Ettin-400M-class model and careful recipes, about 12 more (group B) become coin-flips or narrow wins.
- About 30 datasets (group C plus the knowledge block) are not reachable by any ≤1B model in the public record. On the Decision Index, the open models that match Jev on those rows are 26–35B decoders (Rune 26B-A4B, Decider 35B-A3B, Gemma-4-31B). Even those are below Jev on MMLU-Pro, BBH and GPQA.
- That is a model-class question, not a data question. More training data in the same encoder will not close a 40–60 point gap on MMLU-Pro or GPQA [I].

### 0.4 Addendum 2026-10-03 in one paragraph (details §11–§12)
- The master list added ~95 dataset keys not covered above. **New S-track wins at ≤150M on public evidence:** HWU64 (Jev 0.831 vs BERT-base 0.916), LexGLUE ECtHR-B / SCOTUS / EUR-LEX / LEDGAR / UNFAIR-ToS and the 7-task mean (Legal-BERT 79.8 vs Jev 0.699 / 0.742 tuned), jev-bench civil_comments (Jev 0.729 is below the ~0.92 majority baseline), Yelp-5, MASSIVE en, MNLI (150M), PAWS, SMS spam (jev-bench ships a train split), fake job postings, NFCorpus rerank. At RoBERTa-L size (355M): 10–11 of 17 CSS tasks (Ziems et al.'s fine-tuned baseline on the same test items).
- **New coin-flips (≈350–600M):** ChaosNLI (RoBERTa-L 0.635 vs 0.615), Aegis 1.0/2.0 (Qwen3Guard-0.6B 90.8/85.0/84.2 vs 0.891/0.836/0.802), CaseHOLD and ECtHR-A (large legal encoders ≈ Jev), NevIR, Upworthy, CommonLit, PPE human preference.
- **New losses:** RewardBench 1/2, RM-Bench, ProcessBench, PRMBench, RubricBench, HarmBench-response, WildGuardTest (tie at best), HateCheck, SciFact rerank, TruthfulQA, LogiQA, StrategyQA, FEVER-gold, TabFact, all exam sets (TMMLU+, GAOKAO, JMedQA, JMMLU, ThaiExam, ENEM, SAT).
- **Hygiene problems found (§12):** HWU64 is our dev set but is now a public Jev benchmark; the v2 mix (`mix-v2.0-no-b6`) trains on train splits of ≥13 public-Jev datasets (civil_comments, measuring-hate-speech, Aegis 2.0, LEDGAR, MNLI, FEVER-NLI, WANLI, TREC, CoNLL-2003, SQuAD v2, MTOP, DBpedia, plus tasksource αNLI/CSQA/CaseHOLD/XNLI), so v2 is **not** a Z model on those rows; the decontam reference set only covers our own jevbench, not third-party Jev eval items.

---

## 1. Calibration: what supervised encoders of each size reach

**Ettin encoders** (Weller et al. 2025, arXiv 2507.11412, Table 7). These are GLUE **dev**, fine-tuned per task with an LR sweep [F].

| Size | CoLA | SST-2 | MRPC | STS-B | QQP | MNLI | QNLI | RTE | Avg |
|---|---|---|---|---|---|---|---|---|---|
| 17M | 43.9 | 91.2 | 86.0 | 87.2 | 89.8 | 79.5 | 87.3 | 69.0 | 79.2 |
| 32M | 57.4 | 92.0 | 89.7 | 89.5 | 91.0 | 83.4 | 90.7 | 74.7 | 83.5 |
| **68M** (our v2 backbone) | 64.8 | 94.4 | 92.2 | 91.1 | 91.9 | 87.0 | 92.9 | 83.8 | 87.2 |
| 150M | 66.9 | 95.8 | 92.6 | 92.2 | 92.4 | 89.2 | 94.0 | 87.7 | 88.9 |
| 400M | 71.3 | 96.7 | 93.6 | 92.7 | 93.0 | 91.3 | 95.2 | 92.8 | 90.8 |
| 1B | 74.4 | 97.1 | 94.4 | 93.2 | 93.0 | 91.8 | 96.0 | 93.1 | 91.6 |

GLUE averages for other encoders [F: ModernBERT paper arXiv 2412.13663 Table 1]:

| Model | GLUE avg |
|---|---|
| ModernBERT-base | 88.4 |
| ModernBERT-large | 90.4 |
| DeBERTa-v3-base | 88.1 |
| DeBERTa-v3-large | 91.4 |
| RoBERTa-large | 88.9 |

DeBERTa-v3-large MNLI is 91.8/91.9 [F: via DeBERTa-v3 paper summary].

**Reading the table against Jev's numbers:**
- **SST-2.** Jev is 0.964. 68M reaches 94.4 and 150M reaches 95.8, so both lose. 400M (96.7) and 1B (97.1) win.
- **STS-B.** Jev's Spearman is 0.890. 32M (89.5) ties and 68M (91.1) wins on dev. Expect test to be about 1 point below dev [E], so 150M is the safe choice.
- **NLI-heavy rows** (ANLI, BoolQ, NLI4CT, LLM-AggreFact). Even the 1B Ettin is MNLI 91.8, the same as DeBERTa-v3-large, and DeBERTa-v3-large still loses to Jev on ANLI (0.702) and BoolQ (0.884). Size will not fix this without new data [I].

---

## 2. Deußer et al. suite (37 tasks, arXiv 2609.37647)

**Exact protocol** [F: Deußer `docs/datasets.md` and `results/eval/summary.md`, audit 2026-09-24, run completed 2026-09-27, jev-1.13.0]:
- One request per example; full eval split.
- Thresholds were tuned on 1,000 dev items for GoEmotions, UNFAIR-ToS, AGB-DE and ToxicChat.

**Column conventions:**
- "Train/val on HF" gives rows from `datasets-server /size` on 2026-10-02 [F], and the licence is the HF tag [F] unless noted.
- "S verdict" gives the minimum encoder size at which the public evidence says we beat Jev.

### 2.1 Text classification and routing

| Dataset (eval split, n) | Metric | Jev | Train / val on HF (licence) | Best public supervised reference | S verdict |
|---|---|---|---|---|---|
| `fancyzhx/ag_news` test 7,600 | acc | 0.885 | train 120,000; no val (licence `unknown`, academic) | XLNet-L 95.55 (err 4.45) [F: XLNet T4]; BERT-base FiT 94.75 (err 5.25) [F: Sun et al. 2019, arXiv 1905.05583] | **Win ≤68M** (+6) |
| `stanfordnlp/imdb` test 25,000 | acc | 0.965 | train 25,000; unsup 50,000 (`other`) | XLNet-L 96.80 [F]; RoBERTa-L 96.54 [F: search summary]; BERT-base FiT 94.60 [F] | **Tie/narrow at ≥350M**; needs long context (head+tail or 8k Ettin/ModernBERT) |
| `cornell-movie-review-data/rotten_tomatoes` test 1,066 | acc | 0.933 | train 8,530 / val 1,066 (`unknown`) | XLNet 90.8, RoBERTa 89.9 [F: search summary of a comparison study] | **Lose ≤1B** (public best about 91) |
| `stanfordnlp/sst2` validation 872 | acc | 0.964 | train 67,349 phrases (`unknown`); test labels are −1 | Ettin-400M 96.7, 1B 97.1; 150M 95.8; 68M 94.4 [F] | **Win at 400M only** |
| `dair-ai/emotion` (config `split`) test 2,000 | acc | 0.585 (ECE 0.279) | train 16,000 / val 2,000 (`other`); **`unsplit` 416,809 contains the test, so exclude it** | distilbert-base-uncased-emotion test acc 0.927 [F: model-index]; BERT-base 94.05, RoBERTa-base 93.95 (card table, likely val) [F] | **Win ≤32M** (+34) |
| `atrost/financial_phrasebank` test 970 | acc | 0.730 | train 3,100 / val 776 (no tag; source `takala` is CC BY-NC-SA 3.0). The card says it is a 64/16/20 split of `sentences_50agree` following the FinBERT paper [F] | FinBERT acc 0.86 / F1 0.84 on this same 50agree split [F: Araci 2019, arXiv 1908.10063] | **Win at ~110M** (+13) |
| `ucirvine/sms_spam` all 5,574 | F1 (spam) | 0.938 | **train only**, 5,574 (`unknown`) | BERT/RoBERTa ≥99% acc, spam F1 ≥0.97 on various splits [F: search summaries; protocols differ] | **Win ≤32M**, but only via k-fold out-of-fold predictions over all 5,574 (S-cv); or train on other spam corpora (§6.8) and evaluate Z-style |
| `papluca/language-identification` test 10,000 | acc | 0.996 | train 70,000 / val 10,000 (no tag) | xlm-roberta-base-language-detection **99.6** test; langid.py 98.5 [F: card] | **Tie at best.** Need ≥0.997 (≤30 errors); Jev's CI is 0.995–0.997 |
| `google-research-datasets/go_emotions` simplified test 5,427 | macro-F1 over 28 Nouls | **0.243 fixed / 0.353 tuned** | train 43,410 / val 5,426 (Apache-2.0); `raw` has 211,225 | BERT-base 0.46 (Demszky 2020) [F: search]; RoBERTa about 0.49–0.53 [F: search summaries] | **Win ≤68M** (+11 vs tuned) |
| `mteb/banking77` test 3,076 (Decision Index uses PolyAI 3,080) | acc / macro-F1 | 0.797 / 0.788 (DMB 0.792; **few-shot Jev 0.924**) | mteb: train 9,993 (tag MIT); PolyAI original: train 10,003 / test 3,080 (CC BY 4.0) | BERT fine-tuned **93.66**; USE+ConveRT 93.36 (Casanueva 2020) [F: search] | **Win ≤68M** vs zero-shot. Against few-shot Jev 0.924 it needs about 110M+ with a good recipe (tight) |
| `clinc/clinc_oos` plus test 5,500 | acc (macro-F1) | 0.895 (0.901); in-scope 0.921, OOS recall 0.776 | plus: train 15,250 / val 3,100 (CC BY 3.0). small, imbalanced and full share the **same test** | BERT oos-train on OOS+: in-scope 96.7, OOS recall 59.2, about 0.899 overall; oos-threshold: in-scope 96.2, recall 52.3 (Full) [F: Larson 2019 Table 2] | Macro-F1: **win ≤68M**. Accuracy: **narrow**; needs OOS recall ≥0.75 with in-scope ≥0.96. Plan: plus train + OOS augmentation + calibrated threshold on val |
| `Davlan/sib200` test 41,820 (205 × 204) | acc | 0.815 | train 701 × 205 = 143,705 / val 99 × 205 (CC BY-SA 4.0) | Fully supervised XLM-R-large (550M) **75.9** average; XLM-R-base 71.0; Glot500 64.2; English-train transfer 69.1 [F: SIB-200 paper arXiv 2309.07445] | **Lose ≤1B**; Ettin is English-only |

### 2.2 NLI, grounding and reading comprehension

| Dataset (eval split, n) | Metric | Jev | Train / val on HF (licence) | Best public supervised reference | S verdict |
|---|---|---|---|---|---|
| `facebook/anli` test r1–r3, 3,200 | acc | 0.739 (Decision Index macro-F1 0.748) | train 16,946 / 45,460 / 100,459; dev 1,000 / 1,000 / 1,200 (CC BY-NC 4.0) | DeBERTa-v3-L mnli-fever-anli-ling-wanli **0.702** (r3 0.64) [F: card]; Joelzhang DeBERTa-v3-L 0.775 / 0.636 / 0.612, about 0.670 [F: search] | **Lose ≤1B** |
| `masakhane/afrixnli` test 18 langs × 600 = 10,800 | acc | 0.640 (eng 0.905, fra 0.827; the 16 African languages average 0.612) [F: `afrixnli.json`] | **no train**; validation 450/lang = 8,100 (Apache-2.0) | AfroXLMR-76L (about 560M) in-language **65.7**, translate-test 63.6 [F: IrokoBench search summary] | **Only with an African-pretrained ~560M encoder** plus MNLI, XNLI and the validation split; English-only Ettin loses |
| `google-research-datasets/paws` labeled_final test 8,000 | acc | 0.850 | train 49,401 / val 8,000; labeled_swap 30,397; unlabeled (silver) 645,652 (`other`, PAWS terms) | BERT/DIIN supervised up to 91.9 acc on PAWS-Wiki [F: PAWS paper via search] | **Win ~68–150M** (+5) |
| `lytang/LLM-AggreFact` test 29,320 | **mean per-source BAcc** | 0.786 (pooled 0.831) | **none usable**: dev 30,420 and test 29,320, but the card says "should not be used in pretraining or fine-tuning" (CC BY-ND 4.0, gated) [F] | Bespoke-MiniCheck-7B 77.4; **FactCG-DeBERTa-L (0.4B) 75.6**; MiniCheck-Flan-T5-L 75.0 [F: leaderboard] | **Lose** on public evidence. Training sources for an attempt are in §6.2 |
| `google/boolq` validation 3,270 | acc | 0.913 | train 9,427 (CC BY-SA 3.0); test labels hidden | DeBERTa-v3-large **0.8835** [F: nfliu/deberta-v3-large_boolq]; open-jev-deberta-v3-large 0.879 [F: search] | **Lose ≤1B** |
| `facebook/belebele` test 109,800 (122 × 900) | acc | 0.867 | **no train** (CC BY-SA 4.0) | Fine-tuned on English: XLM-V-L 55.6, InfoXLM-L 56.2, XLM-R-L 54.0. Translate-train-all: 60.2 / 60.0 / 58.9 [F: Belebele paper via search] | **Lose** |
| `qiaojin/PubMedQA` pqa_labeled, all 1,000 | acc (yes/no/maybe) | 0.787 | HF: pqa_labeled 1,000 (one split), pqa_artificial 211,269, pqa_unlabeled 61,249 (MIT) | BioLinkBERT-large **72.2**, base 70.2, PubMedBERT 55.8 (reasoning-required, 500-test) [F: search; pubmedqa.github.io] | **Lose ≤400M**. Also, Jev scored all 1,000 labelled items, so S may train only on pqa_artificial or use 10-fold S-cv |

### 2.3 Knowledge, commonsense and reasoning

| Dataset (eval split, n) | Metric | Jev | Train / val on HF (licence) | Best public supervised or small reference | S verdict |
|---|---|---|---|---|---|
| `tasksource/mmlu` test 14,042 | acc | 0.918 | `cais/mmlu` auxiliary_train 99,842; val 1,531; dev 285 (MIT) | Best ≤0.6B on the Decision Index MMLU row (same test, typed format): Bosun v3.1 0.6B **0.358** [F: DI JSON] | **Lose** |
| `ceval/ceval-exam` test 12,342 | acc | 0.839 | val 1,346 / dev 260 (CC BY-NC-SA 4.0) | none ≤1B close | **Lose** |
| `tasksource/bigbench` validation, 93 MC tasks, 13,228 | acc | 0.814 | **train split exists**: 640,167 rows over 167 tasks (Apache-2.0). Task list in `jev_benchmarking/tasks/bigbench_mc_tasks.json` | no small-model number | **Unknown, lean lose.** S may legally train on the train split of the same 93 tasks; many are knowledge-bound [I]. **Trap:** these train splits contain BBH test items (§7) |
| `Rowan/hellaswag` validation 10,042 | acc | 0.955 (Decision Index 0.945) | train 39,905 (no tag; MIT upstream) | RoBERTa-L 85.2 (test, leaderboard) [F: search] | **Lose** |
| `allenai/winogrande` xl validation 1,267 | acc | 0.914 (Decision Index 0.920) | xl train 40,398 (other sizes: 160 to 10,234; debiased 9,248) | RoBERTa-L 79.1 (WinoGrande paper) [F: search] | **Lose** |
| `allenai/ai2_arc` E+C test 3,548 | acc | 0.988 | train 3,370 / val 869 (CC BY-SA 4.0) | none ≤1B close | **Lose** |
| `tau/commonsense_qa` validation 1,221 | acc | 0.882 | train 9,741; test labels hidden (MIT) | DeBERTa-v3-L **84.1** single / 85.3 ensemble [F: arXiv 2206.05033 via search] | **Lose ≤1B** without external knowledge |
| `allenai/art` (αNLI) validation 1,532 | acc | 0.839 | train 169,654 (`unknown`) | RoBERTa-L classifier **dev 85.64**; L2R2 RoBERTa-L+KLD dev 88.45 / test 86.81 [F: arXiv 2005.11223 Tables 1–2] | **Win at ~350M** (RoBERTa-L class). The listwise ranking loss is worth about +3 |

### 2.4 Safety, moderation and legal

| Dataset (eval split, n) | Metric | Jev | Train / val on HF (licence) | Best public supervised reference | S verdict |
|---|---|---|---|---|---|
| `toxigen/toxigen-data` annotated test 940; toxic = `toxicity_ai + toxicity_human > 5.5` | acc | 0.878 (Spearman 0.841 on the score head) | annotated train 8,960; machine-generated `train` 250,951 (no tag; gated upstream) | no public annotated-test accuracy; ToxiGen-RoBERTa AUC 0.93 on ToxiGen-val (machine) [F: ToxiGen paper via search] | **Unknown, plausible win at 150–400M** |
| `mmathys/openai-moderation-api-evaluation` all 1,680 | mean AUPRC over 8 | 0.717. Per category: S .926, H .857, V .606, HR .533, SH .908, S3 .579, H2 .701, V2 .627 [F: `openai_moderation.json`] | **eval only**, one "train" split = the eval (MIT) | Binary "any unsafe" AUPRC: Llama Guard 7B 0.847, OpenAI API 0.856, Perspective 0.787 [F: Llama Guard Table 2]. No per-category public numbers | **Z only**, or S-cv. Plausible with category-aligned safety data (§6.3). Jev is weakest on HR, S3, V and V2 |
| `lmsys/toxic-chat` toxicchat0124 test 5,083 | F1 (toxic) | 0.786 (P .785 / R .787; AUPRC .892) | train 5,082 (CC BY-NC 4.0); toxicchat1123 has the same sizes | **ToxicChat-T5-large (770M) F1 0.8221** (P .7983 / R .8475) [F: lmsys card] | **Win at ~400–770M**; uncertain at ≤150M |
| `deepset/prompt-injections` test 116 | acc | 0.741 | train 546 (Apache-2.0) | deberta-v3-base-injection **0.9914** on its eval set (this test split) [F: card]. One external re-test reports 0.84 on a different protocol [F: search] | **Win ≤150M** |
| `d4br4/agb-de` test 755 (37 positive) | F1 (positive class) | 0.204 | train 3,004 (CC BY-SA 4.0) | Fine-tuned BERT **F1 0.35** on the imbalanced set (0.54 when undersampled) [F: Braun & Matthes ACL 2024 via search] | **Win at ~110M** (German encoder, e.g. gbert). Ettin is English-only, so use a multilingual or German backbone |
| `coastalcph/lex_glue` unfair_tos test 1,607 | micro-F1 over 8 Nouls | **0.499 fixed / 0.748 tuned** (macro 0.577 / 0.739) | train 5,532 / val 2,275 (CC BY 4.0) | LexGLUE Legal-BERT μ-F1 96.0 / m-F1 83.0; DeBERTa 95.5 / 80.3 [F: search]. **Not comparable:** LexGLUE scores 8+1 labels including "none" | **Likely win at ~110M** (macro-F1 0.83 over the labels), but recompute on the 8 categories only |

### 2.5 Score (ordinal) tasks

| Dataset (eval split, n) | Metric | Jev | Train / val on HF (licence) | Best public supervised reference | S verdict |
|---|---|---|---|---|---|
| `sentence-transformers/stsb` test 1,379 | Spearman | 0.890 | train 5,749 / val 1,500 (no tag) | Ettin 32M 89.5 / 68M 91.1 / 150M 92.2 (GLUE dev) [F] | **Win at 68–150M** |
| `SetFit/sst5` test 2,210 | argmax acc | 0.579 (Spearman 0.851) | train 8,544 / val 1,101 (no tag) | RoBERTa-L **60.2** [F: Cheang et al. arXiv 2005.13619 via search]; RoBERTa-L+Self-Explaining 59.1 | **Win at ~400M** (+2) |
| `mteb/summeval` 1,600 (100 docs × 16) | mean per-document Spearman over 4 dims | 0.554 | **test only** (MIT) | UniEval (T5-L) summary-level 0.474; G-Eval-4 0.514 [F: search] | **Lose**; no train split |
| `nvidia/HelpSteer2` validation 1,038 | mean Spearman over 5 attributes | 0.412. Help .399, corr .369, coh .275, compl .460, verb .558 [F: `helpsteer2.json`] | train 20,324 (CC BY 4.0); also HelpSteer 35,331 / 1,789 and HelpSteer3 | no public small-model per-attribute ρ (SteerLM regression reward models are 70–340B) | **Unknown, plausible win at ~400M** (regression on 20k). Measure |

---

## 3. Decision Index 0.2.1 (38 index benchmarks plus shown rows)

**Source.** The Space `data/index.json`, generated 2026-09-28T00:39Z [F]. "Jev" is the raw native score × coverage.

**"Best ≤0.6B"** is the best entrant with `served_params` ≤ 6e8 in that JSON. It is usually Bosun v3.1 0.6B, which is the DecisionBench maintainer's model.

**Which source split the Decision Index scores** was read from the kit's builders [F: `decision_index/suite/build/*.py`]. **The suite itself is not redistributable; never train on it.**

### 3.1 Knowledge & Reasoning (weight 25.8%)

| Benchmark (n) | Metric | Jev | Decision Index scores… | Train data | Best ≤0.6B | Best public supervised | S verdict |
|---|---|---|---|---|---|---|---|
| GPQA Diamond (198) | acc | 0.786 | `dataset/gpqa_diamond.csv` | none. GPQA main/extended contain diamond (gated, CC BY 4.0) | 0.276 | – | **Lose** |
| GSM8K MC (1,319 q / 2,638 req) | acc | 0.799 | `openai/gsm8k` main test; numeric MC | train 7,473 (MIT) | Julia 1 (141M) 0.644, likely the rank shortcut (issue #32) | – | **Lose honestly.** Run the question-blind check |
| ChessBench (5,000 of 61,833 test positions) | best-move acc (ties accepted) | **0.172** | searchless_chess `test/action_value_data.bag` | **searchless_chess train: 10M games, 530M states, 15.3B action-values** (repo Apache-2.0) | 0.091 | **9M transformer 64.2% action acc**; 136M 68.5%; 270M 69.4% [F: arXiv 2402.04494 T1]. 14.7% of test boards also occur in train (paper) | **Win at 9M**. Biggest single S gap (+45). Needs FEN state plus all legal moves as options (≤255; Jev's limit too) |
| MuSR (752) | acc | 0.661 | GitHub `datasets/` (all 756) | none; generator code (MIT) | jeff 0.479 | – | **Lose** (S+ with generator only, §8) |
| SATA-Bench (1,650) | exact-set acc | 0.264 | GitHub, all | none (HF `sata-bench/sata-bench` 1,604, CC BY-NC 4.0, is the eval) | Lavoir 0.042 | – | **Lose** |
| CRUXEval (570 of 800) | acc | 0.730 | `cruxeval-org/cruxeval` test (MIT) | none | Lumma-Fev 0.416 | – | **Lose** |
| CLadder (5,000 of 10,112) | acc | 0.726 | `cladder-v1-q-balanced.json`, **whole set** | none. `tasksource/cladder` is the same data; generator code MIT | Laya 0.529 | – | **No clean train** (S+ generator only) |
| HLE text-MC (501) | acc | 0.204 (chance 0.164) | `cais/hle` test (gated, MIT) | none | GLiNER2.5-Decide 0.168 | – | **Lose** (noise-level) |
| MMLU-Pro (12,032) | acc | 0.827 | `TIGER-Lab/MMLU-Pro` test (MIT); validation 70 is CoT shots | none. **Exclude MMLU test, TheoremQA and SciBench** (MMLU-Pro sources) | Bosun 0.177 | – | **Lose** |
| BBH, 23 tasks (5,507) | acc | 0.929 | BIG-Bench-Hard repo `bbh/*.json` | none. **Exclude BIG-bench parent-task train splits** | Lavoir 0.351 | – | **Lose** |

### 3.2 Language Understanding (25.8%)

| Benchmark (n) | Metric | Jev | Decision Index scores… | Train data on HF / GitHub (licence) | Best ≤0.6B | Best public supervised | S verdict |
|---|---|---|---|---|---|---|---|
| ContractNLI (123 docs) | macro-F1 | 0.717 | `contract-nli/test.json` | 423 train / 61 dev docs [R]. HF `kiddothe2b/contract-nli` hypothesis rows: a = 6,819 / 978 / 1,991, b = 7,191 / 1,037 / 2,091 (HF says CC BY-NC-SA 4.0; GitHub says CC BY 4.0) | Kai 0.366 | Span NLI BERT-L acc 87.5, F1(C) 0.357, F1(E) 0.834; DeBERTa-v2-xl 88.5 / 0.360 / 0.855 [F: search of Koreeda & Manning 2021] | **Unknown.** Contradiction is the weak class; measure |
| ANLI (3,200) | macro-F1 | 0.748 | `facebook/anli` test_r1–r3 | see §2.2 | Laya 0.487 | 0.70 acc | **Lose** |
| WinoGrande (1,267) | acc | 0.920 | xl validation | xl train 40,398 | 0.520 | 0.791 | **Lose** |
| HellaSwag (10,042) | acc | 0.945 | validation | train 39,905 | Lavoir 0.420 | 0.852 | **Lose** |
| ACOS (400 cases) | per-review F1 (category×sentiment presence) | 0.295 | NUSTM/ACOS `*_test.tsv` (Restaurant + Laptop) | `NEUDM/acos` train 4,464 / val 497 / test 1,399 (no tag) | Bosun 0.056 | ACOS quad-extraction F1 about 0.4–0.6 [R]; the Decision Index projection is easier | **Likely win at ~150M** [E]. Measure |
| FinEntity (979) | macro-F1 | 0.870 | **all 979 docs** ("evaluation-only") | `yixuantt/FinEntity` 979 (ODC-BY) = the eval. Related: §6.9 | jeff 0.749 | FinBERT-CRF micro-F1 0.84 on the paper's own split [F: search] | **No clean train**; Z only |
| iSarcasmEval (4,600 across tasks) | sarcastic-class F1 (task A En) | 0.505 | GitHub `test/*.csv` | En train 3,468 [R]; mirror `viethq1906/isarcasm_2022_taskA_En` 3,121 / 347 / 1,423 (repo MIT) | Lavoir 0.463 | **stce 0.6052** (#1 of 43) [F: search] | **Win plausible at 150–400M** (+10 at SOTA). Extra sarcasm data helps (SemEval-18 Task 3, SARC) |
| VAST (3,006) | macro-F1 | 0.646 | `data/VAST/vast_test.csv` | VAST train 13,477 / dev 2,062 [R] (GitHub, no licence) | Laya 0.405 | TGA-Net 0.665; later BERT-based 0.71–0.80 [F: search] | **Win at ~110–400M** |
| NLI4CT 2024 (5,500) | macro-F1 | 0.841 | `test.json` + `gold_test.json` | `tasksource/nli4ct` train 1,700 / val 200 | jeff 0.550 | best SemEval F1 0.80 (Mistral-7B); DeBERTa-L 0.74; Flan-T5-XL 0.76 [F: search] | **Lose ≤1B** |
| RAGTruth (2,700) | F1 (hallucinated) | 0.765 | `dataset/response.jsonl` split=test | `wandb/RAGTruth-processed` train 15,090 / test 2,700 (repo MIT) | 0.522 | **LettuceDetect-large (ModernBERT-L 396M) 79.22**; base (150M) 76.07; Llama-2-13B FT 78.7; RAG-HAT Llama-3-8B 83.9 [F: arXiv 2502.17125] | **Win at 400M**; tie at 150M |

### 3.3 Retrieval & Classification (20%)

| Benchmark (n) | Metric | Jev | Decision Index scores… | Train data | Best ≤0.6B | Best public supervised | S verdict |
|---|---|---|---|---|---|---|---|
| BANKING77 (3,080) | macro-F1 | 0.797 | PolyAI `test.csv` | train 10,003 (CC BY 4.0) | Bosun 0.761 | 93.66 acc | **Win ≤68M** |
| CLINC150+OOS (5,500) | macro-F1 | 0.893 | oos-eval `data_full.json` test + oos_test | full: 15,000 + 100 OOS; plus: 250 OOS (CC BY 3.0) | Bosun 0.648 | in-scope 96.7 | **Win ≤68M** (macro-F1 is dominated by in-scope) |
| BRIGHT (220 queries) | nDCG@10 over 32 BM25 candidates | 0.475 | `xlangai/BRIGHT` examples (CC BY 4.0) | none. Related: `reasonir/reasonir-data` 345,491 (CC BY-NC 4.0) | Bosun 0.317 | – | **Lose likely** |
| Amazon ESCI (5,000 pairs) | macro-F1 (4 classes) | 0.552 | esci-data parquet, **split = test** only | `tasksource/esci` train 2,027,874 / test 652,490 (Apache-2.0) | Bosun 0.430 | KDD Cup 2022 Task 2 cross-encoders micro-F1 ≈0.816 alone, ensembles ≈0.83 [F: search] | **Likely win at ~150–400M** (macro-F1 not published; measure) |
| PhishNChips (2,000) | acc | 0.625 | `AreLit/PhishNChips` core_emails.csv | none (other configs: real_phishing_validation 1,000; cross_domain_legitimate_v5 333; infrastructure_phishing_expanded 54; licence "other"). Related: §6.8 | 0.600 | – | **Unknown** (Jev is weak; Gemma-4-31B 0.875) |
| HoVer (4,000) | acc (2-way, gold supporting docs given) | 0.729 | `hover_dev_release_v1.1.json` + Wikipedia doc text | **HoVer train 18,171** (`vincentkoc/hover-parquet` 18,171 / 4,000 / 4,000; repo MIT) | Lavoir 0.597 | BERT-base claim verification with **oracle evidence 81.2** [F: HoVer paper via search] | **Win at ~110M** (+8) |

### 3.4 Tools & Automation (18.3%)

| Benchmark (n) | Metric | Jev | Decision Index scores… | Train data | Best ≤0.6B | S verdict |
|---|---|---|---|---|---|---|
| BFCL (1,694) | whole-case exact | 0.958 | gorilla BFCL data, test (Apache-2.0) | none. Related: §6.6 | Bosun 0.466 | **Lose likely** |
| ToolRet (685 queries) | nDCG@10 | 0.653 | `mangopy/ToolRet-Queries` / `-Tools` (code Apache-2.0) | **`mangopy/ToolRet-Training-20w` 208,826** (no licence tag) | Bosun 0.574 | **Unknown, plausible.** The ToolRet paper reports large gains for retrievers trained on it [F: search] |
| API-Bank (508) | acc (next API out of 53) | 0.882 | DAMO-ConvAI `api-bank` samples | API-Bank training data in `liminghao1630/API-Bank` (MIT; server cannot size it) | Bosun 0.532 | **Unknown** |
| Home appliance sim (88) | case exact | 0.523 | synthetic builder in the kit | none (generator) | 0.0 | **Unknown** (Z only) |
| When2Call (3,652) | acc (4-way MCQ) | 0.810 | `nvidia/When2Call` test MCQ | **train_sft 15,000 / train_pref 9,000** (CC BY 4.0). **Dedupe vs BFCL** | Kai 0.376 | **Plausible win at ~400M.** The paper shows SFT/RPO on its train split helps [F: search] |

### 3.5 Arts & Human Taste (10%)

| Benchmark (n) | Metric | Jev | Decision Index scores… | Train data | Best ≤0.6B | Best public supervised | S verdict |
|---|---|---|---|---|---|---|---|
| BPoMP (811 cases / 5,000 req) | acc (original vs perturbed limerick) | 0.909 | Zenodo 7299879 p1–p3 | none. Self-supervised perturbation of other limerick corpora is possible | 0.587 | – | **Unknown** |
| Humicroedit (2,628 non-tie pairs) | acc | 0.619 | SemEval-2020 T7 subtask-2 `test.csv` | subtask-2 train 9,381 / dev 2,355 [R]; plus FunLines; HF `SemEvalWorkshop/humicroedit` (`unknown`) | 0.521 | **Hitachi 67.43** (#1) [F: search] | **Win at ~400M** (+5 at SOTA) |
| POP909-CL (2,000) | acc (chord pitch-class set) | 0.166 | POP909-CL repo, sampled across all songs | **no song-level split**, so POP909 itself overlaps. Related: other symbolic chord corpora (MIT repo; `ailsntua/Chordonomicon` 679,807 progressions, CC BY-NC 4.0, has no note data) | 0.049 | – | **Unknown** (a template matcher may beat 0.166 [E]) |
| cfcolor (5,000) | acc (which palette the user rates higher) | 0.644 | original test targets + train history | the dataset's own train ratings (`train_vec`) are a legitimate train split | 0.532 | – | **Unknown** |
| ForecastBench (10,139) | Brier skill | 0.306 (Brier about 0.174) | resolutions from 2026-07 onward | earlier question and resolution sets in `forecastbench-datasets` (CC BY-SA 4.0) | 0.046 | – | **Lose likely** |
| Habermas (1,676) | acc (group consensus pick) | 0.459 | EVAL cohorts IID_TEST / OOD_TEST | TRAIN cohorts in the same release (code Apache-2.0) | GLiNER2.5-Decide 0.455 | – | **Unknown** |
| New Yorker matching (528) | acc (5-way, from descriptions) | 0.701 | `matching` fold-0 test | matching train 9,792 / val 531 (CC BY 4.0) | Bosun 0.386 | **T5-11B fine-tuned 70.8** (from description); GPT-4 5-shot 84.5 [F: Hessel et al. ACL 2023 via search] | **Lose ≤1B** |

**Shown, not counted:**

| Row | Jev | Notes |
|---|---|---|
| MMLU | 0.920 | – |
| RouterBench | 0.799 | – |
| SGD | – | Build bug; test from `schema_guided_dstc8`, whose train is 16k dialogues (CC BY-SA 4.0) |
| ARC-E/C | not scored for Jev | – |

---

## 4. Other studies with public Jev numbers

| Study → dataset (n) | Metric | Jev | Train data (licence) | Best public supervised | S verdict |
|---|---|---|---|---|---|
| DMB → NLU++, folds 18–19 (302 messages / 13,712 decisions) | micro-F1 | 0.483 (macro 0.581) | NLU++ folds 0–17 on GitHub `PolyAI-LDN/task-specific-datasets` (CC BY 4.0) | RoBERTa-QA micro-F1 80.3 / 85.6 / 93.1 across the paper's three training-size regimes [F: NLU++ paper via search] | **Win ≤150M** (+30–45) |
| DMB → Banking77 (3,080), CLINC150 (5,500) | acc | 0.792 / 0.886 | as above | as above | as §2.1 |
| simonmesmith → Banking77 (3,080), **Jev plus 24 BM25 train examples** | acc | **0.924** | train 10,003 | BERT 93.66 | **Tight win at ~110M+** |
| Jevals → Banking77 (300 × 5) | acc | 0.797 | as above | – | win |
| Jevals → PubMedQA yes/no (300 × 5, "train" split = pqa_labeled) | acc | 0.913 | pqa_artificial 211,269 only (its labelled items are the eval) | – | **Unknown**; S needs disjoint data |
| Jevals → HelpSteer2 helpfulness (300 × 5) | 5-level acc | 0.413 (prior 0.417) | train 20,324 | – | **Win likely** (bar is the prior) |
| elcronos → `cardiffnlp/tweet_topic_single` test_2021 (1,693) | acc | 0.793 | train_2021 1,516; train_all 4,374; val_2021 189 (`other`) | roberta-large on train_all: acc 0.896, macro-F1 0.800 [F: cardiffnlp card] | **Win at ~125M** (large shown; base expected about 0.87–0.89 [E]) |
| elcronos → `zeroshot/twitter-financial-news-topic` validation (4,117) | acc | 0.670 | train 16,990 (MIT) | finbert-tone fine-tuned **acc 0.9106** [F: nickmuchi card via search] | **Win at ~110M** (+24) |
| elcronos → daily_dialog test (7,740 utterances, 7 classes) | macro-F1 | 0.385 (acc 0.710) | `roskoN/dailydialog` 11,118 / 1,000 / 1,000 dialogues (`li2017dailydialog` is CC BY-NC-SA 4.0, script-based) | RoBERTa macro-F1 51.17 (6 classes, neutral excluded) [F: search] | **Win likely ≤150M** (recompute on 7 classes) |
| elcronos → dair-ai emotion | acc / F1 | 0.587 / 0.500 | as above | 0.927 | win |
| zhuyansen → TweetEval emotion (n=1,000 of 1,421) | acc | 0.827 | `cardiffnlp/tweet_eval` emotion train 3,257 / val 374 | twitter-roberta-base-2021-124m acc **0.832** / macro-F1 0.794 [F: card via search] | **Tie at 125M**; needs a larger or Twitter-adapted model |
| zhuyansen → PAWS (1,000), SST-2 (872), AG News (1,000), Banking77 (1,000) | acc | 0.855 / 0.960 / 0.865 / 0.712 | as above | as above | as above |
| zhuyansen → arXiv 2026-09, 8 categories (258) | acc | 0.891 | not a fixed public dataset; pre-2026 arXiv abstracts are fair training data (dedupe by id) | – | **Unknown** |
| AbdelStark BTZSC pilot → AG / Banking77-72 / emotion (100 each) | acc | 0.910 / 0.870 / 0.480 | as above | as above | as above (n=100 means wide CIs) |
| onlyoneaman → Enron spam | acc | 0.987 | `SetFit/enron_spam` train 31,716 / test 2,000 (no tag) | about 99% typical [E] | **Narrow win** |
| scienthoon → OpenBookQA (500), CommonsenseQA (1,221), HellaSwag (2,000) | acc | 0.942 / 0.881 / 0.861 | OBQA main train 4,957 (`unknown`) | none ≤1B close | **Lose** (HellaSwag 0.861 is a weaker harness; still lose at 0.852) |
| LocalLLaMA/typed-decisions test (400 cases / 2,000 decisions) | soft-gold acc | 0.727 | `all` train **1,200 cases** (Apache-2.0); the card keeps a separate "trained on train" table | OpenDecider-nano (Ettin-400M) **0.796**; soft-decider-421m 0.774; Laya-td 0.766; ModernBERT-base 0.646; Bekko 68M 0.537 [F: card, via our leaderboards notes] | **Win at 400M** with train; 68M looks hard |
| jev-frontier-bench (200 pooled: Banking77, BoolQ, Yelp, ChaosNLI) | acc | 0.725 | per component | – | composite |
| GautamTalksDev → ChaosNLI (750 + 750) | ECE etc. | – | `earino/chaosnli` validation only: αNLI 1,532 / MNLI-m 1,599 / SNLI 1,514 (CC BY-NC 4.0). Train with SNLI/MNLI/αNLI train | – | calibration-only |
| jev-phishing-bench = PhishNChips (2,000) | acc | 0.626 | none | – | see §3.3 |
| WebJev → MMLU-Pro (1,000), typed-decisions | acc | 0.834 / 0.7405 | – | – | as above |
| Kev → MMLU / MMLU-Pro | acc | 0.90 / 0.84 | – | – | lose |
| Arize (18,514 emails) | acc | 0.983 | public Kaggle email-spam set, all rows evaluated | – | **S-cv only; see §11.10** |
| Red Hat NeMo Guardrails eval | acc | injection 0.8635, safety 0.8620 | EvalHub configs; corpora not public | – | **Skip; see §11.3** |
| Ibrahim & Zaki (18 CSS tasks on Ziems et al. 2024 test splits) | macro-F1 | median 0.581 over 15 eval tasks; −11.6 vs best LLM | original datasets' train splits | Ziems RoBERTa-L baseline | **Done: see §11.7** |

**Typed-decision boards with no public train data** (S impossible; Z only):
- DecisionBench: `Hanno-Labs/decision-bench` has a single `eval` split of 23,900 rows, licence "other". Its 43 tasks include FinQA, FOLIO and MuSiQue-derived rows, so exclude those sources' **test** portions [I]. Jev 0.720; binary view 0.592.
- DecideBench: 400 rows, CC BY 4.0, Jev 98.0%.
- JevBench: 1,624 decisions, 601 public. Training on the public items is allowed with disclosure, but the open–sealed gap is penalized.
- TypeSafe evals: 705 cases, WorkflowEvals Apache-2.0.
- jabr classifier-benchmark: 866 cases, CC0. The README allows training, so a win there proves nothing.

---

## 5. jevbench-only datasets (in our suite, no published Jev number yet)

| Dataset | Train / val on HF (licence) | Public supervised reference |
|---|---|---|
| `community-datasets/yahoo_answers_topics` (test 60,000) | train 1,400,000 (`unknown`) | BERT-ITPT-FiT about 77.6 acc [R: arXiv 1905.05583] |
| `Yelp/yelp_review_full` (test 50,000) | train 650,000 (`other`, Yelp terms) | XLNet-L 72.95 (err 27.05) [F] |
| `mteb/amazon_massive_intent` en (test 2,974) | train 11,514 / val 2,033 (Apache-2.0; MASSIVE CC BY 4.0) | XLM-R-base 88.3; mT5-base encoder 89.0 [F: MASSIVE paper via search] |
| `nyu-mll/glue` rte (val 277) | train 2,490 (`other`) | Ettin 68M 83.8 / 400M 92.8 [F] |
| `tdiggelm/climate_fever` (test 1,535) | **test only** (`unknown`). Related: FEVER (`fever/fever`, CC BY-SA 3.0 / GPL-3.0, script-based), `tals/vitaminc` 370,653 (CC BY-SA 3.0), `allenai/scifact` (CC BY-NC 2.0) | – |

---

## 6. Related public training data for datasets without a usable train split

Sizes are from `datasets-server /size` on 2026-10-02 [F] unless marked [R]. A few datasets are script-based, so the server cannot size them; their sizes come from cards or papers.

### 6.1 Knowledge, multiple-choice QA and reasoning
For MMLU, MMLU-Pro, C-Eval, ARC, OBQA, CSQA, BBH, GPQA, HLE and Belebele.

| HF id | Split sizes | Licence | Note |
|---|---|---|---|
| `cais/mmlu` auxiliary_train | 99,842 | MIT | ARC, OBQA, RACE, MCTest recast |
| `allenai/ai2_arc` | 3,370 / 869 | CC BY-SA 4.0 | – |
| `allenai/openbookqa` main | 4,957 / 500 | unknown | – |
| `allenai/sciq` | 11,679 / 1,000 | CC BY-NC 3.0 | – |
| `ehovy/race` all | 87,866 / 4,887 | other (research) | Belebele training ingredient |
| `tau/commonsense_qa` | 9,741 / 1,221 | MIT | – |
| `ybisk/piqa` | 16,113 / 1,838 [R] | unknown | script-based |
| `allenai/social_i_qa` | 33,410 / 1,954 [R] | – | script-based |
| `allenai/qasc` | 8,134 / 926 | CC BY 4.0 | – |
| `allenai/cosmos_qa` | 25,262 / 2,985 [R] | CC BY 4.0 | script-based |
| `openlifescienceai/medmcqa` | 182,822 / 4,183 | Apache-2.0 | – |
| `GBaker/MedQA-USMLE-4-options` | 10,178 | CC BY 4.0 | – |
| `allenai/math_qa` | 29,837 [R] | Apache-2.0 | – |
| `deepmind/aqua_rat` raw | 97,467 | Apache-2.0 | – |
| `lucasmccabe/logiqa` | 7,376 | – | – |
| `tasksource/reclor` | 4,638 | other (non-commercial) | – |
| `sagnikrayc/mctest` | 1,480 | other | – |
| `dataset-org/dream` | 6,116 [R] | unknown | script-based |
| `aps/super_glue` multirc | 27,243 | other | Belebele ingredient |
| `derek-thomas/ScienceQA` | 12,726 | CC BY-SA 4.0 | – |
| `Rowan/hellaswag` train | 39,905 | MIT upstream | – |
| `allenai/winogrande` xl | 40,398 | – | – |
| `allenai/art` train | 169,654 | unknown | – |
| `openai/gsm8k` main | 7,473 | MIT | – |
| `EleutherAI/hendrycks_math` | 7,500 | MIT | – |
| `meta-math/MetaMathQA` | 395,000 | MIT | – |
| `nvidia/OpenMathInstruct-2` | 13,972,791 | CC BY 4.0 | – |
| `TIGER-Lab/WebInstructSub` | 2,335,220 | Apache-2.0 | – |
| `facebook/natural_reasoning` | 1,145,824 | CC BY-NC 4.0 | – |
| `tasksource/bigbench` | 640,167 train | Apache-2.0 | **Not if BBH is evaluated** |
| `Muennighoff/flan` | 2,772,100 | – | Aggregates many benchmark train **and** some eval splits; filter |

- **Belebele's own recipe** assembled RACE, SciQ, MultiRC, MCTest, MCScript2.0 and ReClor [R: Belebele paper]. Fine-tuned XLM-R-class models still reach only 54–60.
- **C-Eval:** `ceval/ceval-exam` val 1,346 / dev 260 is usable (not test).

### 6.2 NLI and grounding
For ANLI (Z), AfriXNLI, LLM-AggreFact, BoolQ, Climate-FEVER, NLI4CT and ContractNLI.

| HF id | Rows | Licence | Note |
|---|---|---|---|
| `nyu-mll/multi_nli` | 392,702 | mixed (OANC; CC BY-SA 3.0; MIT) | – |
| `stanfordnlp/snli` | 550,152 | CC BY-SA 4.0 | – |
| `pietrolesci/nli_fever` | 208,346 | – | – |
| `alisawuffles/WANLI` | 102,885 | CC BY 4.0 | – |
| `tasksource/lingnli` | 44,982 | unknown | – |
| `tasksource/doc-nli` | 861,708 | BSD | – |
| `tals/vitaminc` | 370,653 | CC BY-SA 3.0 | – |
| `facebook/xnli` | 6,283,232 | – | machine-translated MNLI, 15 languages; AfriXNLI support |
| `sentence-transformers/all-nli` pair-class | 942,069 | – | – |
| `MoritzLaurer/mnli_anli_fevernli_wanli_lingnli_xnli_train` | MNLI 392,702 + FEVER-NLI 196,805 + **ANLI 162,865** + WANLI 102,885 + LingNLI 29,985 + XNLI 37,350 | – | **Z-unsafe for ANLI**; S-OK |
| `lytang/C2D-and-D2C-MiniCheck` | 7,076 + 7,319 | MIT | MiniCheck synthetic grounding |
| `wandb/RAGTruth-processed` train | 15,090 | – | **Its test is inside LLM-AggreFact** |

FactCG data (`derenlei/FactCG`) is not public via the API (401).

### 6.3 Safety and moderation
For the OpenAI moderation eval, ToxiGen, ToxicChat and prompt injection.

| HF id | Split sizes | Licence | Note |
|---|---|---|---|
| `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` | 30,007 / 1,445 / 1,964 | CC BY 4.0 | 13 categories, maps onto the OpenAI 8 |
| `nvidia/Aegis-AI-Content-Safety-Dataset-1.0` | 10,798 / 1,199 | CC BY 4.0 | – |
| `PKU-Alignment/BeaverTails` | 330k_train 300,567 / 30k_train 27,186 | CC BY-NC 4.0 | – |
| `allenai/wildguardmix` | – | ODC-BY | gated; 86.8k [R] |
| `google/civil_comments` | 1,804,874 / 97,320 | CC0 | – |
| `thesofakillers/jigsaw-toxic-comment-classification-challenge` | 159,571 | CC BY-SA 3.0 | – |
| `ucberkeley-dlab/measuring-hate-speech` | 135,556 | CC BY 4.0 | – |
| `OpenSafetyLab/Salad-Data` | 30,658 | Apache-2.0 | – |
| `Anthropic/hh-rlhf` | 160,800 | MIT | red-team prompts |
| `lmsys/toxic-chat` 0124 train | 5,082 | CC BY-NC 4.0 | – |
| `toxigen/toxigen-data` annotated train | 8,960 | – | plus 250,951 machine-generated |
| `deepset/prompt-injections` train | 546 | Apache-2.0 | – |

**Excluded:** `KoalaAI/Text-Moderation` and `KoalaAI/Text-Moderation-v2-small`, which were trained on or derived from the eval itself.

### 6.4 Preference and judging
For HelpSteer2 and SummEval.

| HF id | Split sizes | Licence | Note |
|---|---|---|---|
| `nvidia/HelpSteer2` | 20,324 / 1,038 | CC BY 4.0 | – |
| `nvidia/HelpSteer` | 35,331 / 1,789 | CC BY 4.0 | – |
| `nvidia/HelpSteer3` | preference 38,459 / edit 13,740 / feedback 38,782 / principle 32,881 train | CC BY 4.0 | **Dedupe prompts vs HelpSteer2 validation** |
| `openbmb/UltraFeedback` | 63,967 | MIT | – |

SummEval has no train split. Proxies such as Newsroom human eval, RealSumm and FRANK are not checked here.

### 6.5 Intent, routing and topic
For CLINC OOS augmentation and NLU++.
- `DeepPavlov/hwu64`: 8,954 / 1,076. ~~It is our dev set; do not train on it.~~ **Superseded 2026-10-03:** HWU64 now has a public Jev number (0.831 on the 1,076 test, thisisandreeeee). S may train on its train split; Z must not; and our use of its test as dev must stop or be disclosed (§12.1).
- `google-research-datasets/schema_guided_dstc8`: CC BY-SA 4.0, script-based.
- MASSIVE en: 11,514.
- `tasksource/zero-shot-label-nli`: 1,090,333, licence other. It aggregates benchmark train splits, so filter by source.
- `MoritzLaurer/synthetic_zeroshot_mixtral_v0.1`: 2,627,036, Apache-2.0.

### 6.6 Tools and retrieval
For BFCL, ToolRet, API-Bank, When2Call and BRIGHT.

| HF id | Rows | Licence | Note |
|---|---|---|---|
| `mangopy/ToolRet-Training-20w` | 208,826 | – | built from ToolACE, ToolBench and APIGen train |
| `Team-ACE/ToolACE` | 11,300 | Apache-2.0 | – |
| `glaiveai/glaive-function-calling-v2` | 112,960 | Apache-2.0 | – |
| `Salesforce/xlam-function-calling-60k` | – | CC BY 4.0 | gated |
| `nvidia/When2Call` | train_sft 15,000 / train_pref 9,000 | CC BY 4.0 | – |
| `reasonir/reasonir-data` | 345,491 | CC BY-NC 4.0 | – |
| `tasksource/esci` | 2,027,874 | Apache-2.0 | – |

### 6.7 Multilingual
For SIB-200, AfriXNLI, Belebele and AGB-DE.
- `Davlan/sib200` train: 143,705.
- `masakhane/afrixnli` validation: 8,100.
- `facebook/xnli`.

An English-only Ettin is the wrong backbone for this whole block. It needs XLM-R, mmBERT or AfroXLMR-class pretraining [I].

### 6.8 Spam and phishing
For SMS spam, Enron and PhishNChips.

| HF id | Split sizes | Licence |
|---|---|---|
| `SetFit/enron_spam` | 31,716 / 2,000 | – |
| `zefang-liu/phishing-email-dataset` | 18,650 | LGPL-3.0 |
| `puyang2025/seven-phishing-email-datasets` | 162,413 / 40,604 | other |
| `cybersectony/PhishingEmailDetectionv2.0` | 120,000 / 20,000 / 60,000 | none |

### 6.9 Finance
For FinEntity (Z), Financial PhraseBank and fin-topic.

| HF id | Split sizes | Licence | Note |
|---|---|---|---|
| `zeroshot/twitter-financial-news-sentiment` | 9,543 / 2,388 | MIT | Same tweet pool as fin-topic; S-OK, Z-unsafe |
| `TheFinAI/fiqa-sentiment-classification` | 822 / 117 / 234 | MIT | – |
| `temetnosce01/phrasebank_and_sentfin` | – | – | **contains FPB** |
| `nickmuchi/financial-classification` | 4,551 / 506 | – | **contains FPB**; excluded |

---

## 7. Contamination traps (S and Z)

1. **Test-equals-validation datasets.** Jev's eval split is the official **validation** split for SST-2, BoolQ, HellaSwag, WinoGrande, CommonsenseQA, αNLI, HelpSteer2, fin-topic and our RTE. Any aggregator that folds validation into training contaminates these: FLAN, tasksource-instruct, zero-shot-label-nli, Laurer mixes. Carve dev from train.
2. **Same text in a different dataset:**
   - SST ⊂ the Pang & Lee rotten-tomatoes sentence pool. For rotten_tomatoes in S, dedupe SST train against rotten_tomatoes test.
   - The `dair-ai/emotion` `unsplit` config contains the test.
   - `nickmuchi/financial-classification` and `temetnosce01/phrasebank_and_sentfin` contain FPB.
   - All CLINC configs share the test.
   - MASSIVE locales and SLURP; PAWS-X.
3. **The benchmark's own eval used as "train":**
   - `KoalaAI/Text-Moderation*` was trained on the OpenAI moderation eval.
   - `tasksource/cladder` is the same as the Decision Index's CLadder set.
   - `sata-bench/sata-bench` "train" is the eval.
   - FinEntity's only split is the eval.
   - PubMedQA pqa_labeled is Deußer's and Jevals' eval.
   - SMS spam's only split is the eval.
4. **Licence bans on training:**
   - LLM-AggreFact (dev and test): "should not be used in pretraining or fine-tuning".
   - GPQA: "do not reveal examples"; the Decision Index treats it as local-eval only.
   - The Decision Index suite: not redistributable.
   - Deußer's released Jev responses and all Jev outputs: "no distillation". Jev outputs never enter training on either track.
5. **Parent and child benchmarks:**
   - BIG-bench train splits contain BBH items.
   - MMLU-Pro contains MMLU-test, TheoremQA and SciBench.
   - GPQA main/extended contain diamond.
   - RAGTruth test ⊂ LLM-AggreFact test.
   - When2Call is built from BFCL.
   - ToolRet test contains API-Bank and APIGen queries; ToolRet-train is built from ToolACE, ToolBench and APIGen train. Dedupe these against the Decision Index's API-Bank, BFCL and When2Call rows.
6. **Board-specific:**
   - The GSM8K MC rank shortcut (Decision Index issue #32).
   - 14.7% of ChessBench test boards also appear in its train set (paper). Disclose this; it is inherent to chess openings.
   - JevBench penalizes the open–sealed gap if we tune on its 601 public items.
7. **typed-decisions train** is fine in S (the card keeps a separate "trained" table) but **must never reach a Z checkpoint**.
8. **Mechanical rule, kept from the v2 plan:** hash normalized test texts and drop exact matches and ≥50% 13-gram overlaps from every S training set. Commit the manifest per dataset. Laurer's models, Laya, Von and OpenDecider trained on many of these train splits. That is fine for S, but they are not Z baselines.

---

## 8. S-track protocol (proposed)

1. **Splits.**
   - Train on the official train split.
   - Select on the official validation split, unless Jev's number is on validation; then hold out 10% of train, stratified and duplicate-grouped.
   - Test is read once per release.
2. **Datasets with no train split** (SMS spam, PubMedQA-labeled, OpenAI moderation, SummEval, FinEntity, CLadder): report only as **S-cv**. Use 5- or 10-fold out-of-fold predictions grouped by source document, clearly flagged, or leave them Z-only. S-cv numbers should not be mixed into the S headline [I].
3. **"S+generator"** (CLadder, MuSR, Home appliances; BPoMP via perturbation): training on freshly generated instances from the benchmark's own released generator, with new seeds. This is not an official train split, so report it separately or not at all.
4. **Thresholds and calibration.** Jev's tuned-threshold numbers (GoEmotions, UNFAIR-ToS) were tuned on 1,000 dev items. We tune on validation too and compare against the **tuned** Jev numbers. Temperature is per dataset on validation, which is allowed in S.
5. **Comparators.** Compare against Jev's published number, with its CI from Deußer or the Decision Index, in the same metric and on the same split.
   - When several Jev numbers exist, beat the maximum: Banking77 0.924 (few-shot), UNFAIR-ToS 0.748 (tuned), GoEmotions 0.353 (tuned), HellaSwag 0.955 (Deußer), WinoGrande 0.920 (Decision Index).
   - Paired tests are possible only where raw Jev responses exist (Deußer zenodo 10.5281/zenodo.23039006, evaluation-only), so use them for paired bootstrap.
6. **Model granularity.** Report both:
   - (a) one multi-task S model (one checkpoint, all train splits);
   - (b) per-dataset specialists.

   The Decision Index forbids per-benchmark prompts and calibration for its own board, so a Decision Index submission must be (a) with one rendering.

---

## 9. Suggested order of work (S track), by gap × feasibility

1. **Cheap, huge gaps (≤68M, one run each):**

   | Dataset | Expected gain [E] |
   |---|---|
   | ChessBench | +45 |
   | NLU++ | +30 |
   | emotion | +34 |
   | fin-topic | +24 |
   | prompt-injections | +25 |
   | AGB-DE | +15 (needs a multilingual backbone) |
   | FPB | +13 |
   | Banking77 vs zero-shot | +13 |
   | GoEmotions vs tuned | +11 |
   | tweet_topic | +10 |
   | daily_dialog | +13 |
   | HoVer | +8 |
   | AG News | +6 |
   | PAWS | +5 |
   | STS-B | +2 |
   | CLINC macro-F1 | – |
   | UNFAIR-ToS vs tuned 0.748 | measure |

2. **Medium (150–400M, tuned recipes):**
   - VAST, iSarcasmEval, Humicroedit, ESCI, ACOS, ContractNLI, ToxiGen, HelpSteer2, OpenAI moderation (Z-style with safety data).
   - CLINC accuracy (OOS recall), Banking77 vs 0.924 few-shot.
3. **Large-encoder coin-flips (≈400M, Ettin-400M / ModernBERT-L / DeBERTa-v3-L):**
   - SST-2, SST-5, IMDB, αNLI, RAGTruth, ToxicChat, typed-decisions (trained table), When2Call, TweetEval emotion.
4. **Multilingual block:** AfriXNLI (AfroXLMR-76L-class, 560M), SIB-200 (lose even at 550M), Belebele (lose).
5. **Do not spend compute** on ANLI, BoolQ, rotten_tomatoes, language-id, PubMedQA, LLM-AggreFact, SummEval, NLI4CT, New Yorker, or the knowledge/reasoning block with this model class. Report them as "shown, not counted", with the public ceiling numbers above as evidence.

**Cost note.** None of this needs a Jev call. Every comparison is against a published Jev number, and Deußer's raw responses (evaluation-only) give paired statistics. So the $5 cap is untouched.

---

## 10. Sources (all read 2026-10-02 unless noted)

**Jev numbers (primary files)**
- Deußer et al. kit: https://github.com/AppliedMachineLearning-Lab/jev-benchmarking (`docs/datasets.md`, `results/eval/summary.md`, `results/eval/{llm_aggrefact,clinc150,helpsteer2,toxic_chat,agb_de,summeval,toxigen,openai_moderation,stsb,sst5,unfair_tos,go_emotions,afrixnli,pubmedqa}.json`, `results/eval/thresholds.json`). Paper: arXiv 2609.37647. Responses: doi 10.5281/zenodo.23039006.
- Decision Index: https://huggingface.co/spaces/multimodalart/jev-decision-index (`data/index.json`, generated 2026-09-28T00:39Z). Kit: https://github.com/apolinario/decision-index (`decision_index/data/benchmarks.json`, `data/index-0.2.1.json`, `suite/build/{acquire,adapters_added,adapters_scored,adapters_selection,adapters_creative,adapters_retrieval,adapters_mechanical,normalize_text,normalize_direct}.py`).
- Banking77 few-shot Jev: https://github.com/simonmesmith/jev-banking77-experiment (2026-09-18).
- DMB, Jevals, elcronos, zhuyansen, AbdelStark, onlyoneaman, scienthoon, typed-decisions, WebJev, Kev: as cited in `docs/research/v2/classifier-benchmarks.md` §6 and `docs/research/leaderboards/live-leaderboards-2026-10-02.md`.
- DecisionBench: https://huggingface.co/datasets/Hanno-Labs/decision-bench (README); https://github.com/Hanno-Labs/decision-bench (`docs/overview/tasks.md`, `task_specs/decisionbench-dev.toml`).

**Split sizes and licences.** HF Hub API `https://huggingface.co/api/datasets/<id>` and `https://datasets-server.huggingface.co/size?dataset=<id>`, swept 2026-10-02. GitHub licences via the `gh api repos/<repo>` field `license.spdx_id`.

**Supervised references**
- Ettin: arXiv 2507.11412 (Tables 6–7).
- ModernBERT: arXiv 2412.13663 (Table 1).
- DeBERTa-v3-large NLI: https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli; https://huggingface.co/Joelzhang/deberta-v3-large-snli_mnli_fever_anli_R1_R2_R3-nli.
- ModernBERT-zeroshot (binary): https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0; https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0.
- BoolQ: https://huggingface.co/nfliu/deberta-v3-large_boolq.
- Banking77: Casanueva et al. 2020, https://aclanthology.org/2020.nlp4convai-1.5.pdf.
- CLINC150: Larson et al. 2019, https://aclanthology.org/D19-1131.pdf (Table 2, read from the PDF).
- XLNet: arXiv 1906.08237 (Table 4).
- BERT-ITPT: arXiv 1905.05583.
- SST-5: arXiv 2005.13619.
- Emotion: https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion.
- GoEmotions: Demszky et al. 2020 and summaries via search.
- FinBERT: arXiv 1908.10063.
- fin-topic: https://huggingface.co/nickmuchi/finbert-tone-finetuned-finance-topic-classification.
- Language-id: https://huggingface.co/papluca/xlm-roberta-base-language-detection.
- Prompt injection: https://huggingface.co/deepset/deberta-v3-base-injection.
- ToxicChat: https://huggingface.co/lmsys/toxicchat-t5-large-v1.0.
- Llama Guard: arXiv 2312.06674 (Table 2).
- LLM-AggreFact: https://llm-aggrefact.github.io/; https://huggingface.co/datasets/lytang/LLM-AggreFact (card, licence and no-training clause).
- RAGTruth / LettuceDetect: arXiv 2502.17125; https://huggingface.co/KRLabsOrg/lettucedect-base-modernbert-en-v1.
- SIB-200: arXiv 2309.07445.
- Belebele: arXiv 2308.16884.
- IrokoBench: arXiv 2406.03368.
- PubMedQA: https://pubmedqa.github.io/.
- CommonsenseQA DeBERTa-v3: arXiv 2206.05033.
- αNLI: arXiv 2005.11223 (Tables 1–2, read from the PDF).
- HellaSwag: https://rowanzellers.com/hellaswag/.
- WinoGrande: arXiv 1907.10641.
- PAWS: https://aclanthology.org/N19-1131.pdf.
- NLU++: arXiv 2204.13021.
- HoVer: arXiv 2011.03088.
- ChessBench: arXiv 2402.04494 (Table 1).
- iSarcasmEval: https://aclanthology.org/2022.semeval-1.113/.
- VAST: https://aclanthology.org/2020.emnlp-main.717.pdf and follow-ups via search.
- Humicroedit: arXiv 2008.00304.
- New Yorker: https://aclanthology.org/2023.acl-long.41.pdf.
- NLI4CT 2024: arXiv 2404.04963.
- ContractNLI: arXiv 2110.01799.
- ESCI KDD Cup: https://amazonkddcup.github.io/papers/8408.pdf.
- FinEntity: https://aclanthology.org/2023.emnlp-main.956/.
- AGB-DE: arXiv 2406.06809.
- LexGLUE: https://aclanthology.org/2022.acl-long.297.pdf.
- TweetEval / tweet_topic: https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m-emotion; https://huggingface.co/cardiffnlp/roberta-large-tweet-topic-single-all.
- DailyDialog: arXiv 2206.07359.
- MASSIVE: arXiv 2204.08582.
- SummEval baselines: UniEval and G-Eval via search.
- ToolRet: arXiv 2503.01763.
- When2Call: https://aclanthology.org/2025.naacl-long.174/.
- Contamination evidence: https://huggingface.co/KoalaAI/Text-Moderation (datasets field); https://huggingface.co/datasets/nickmuchi/financial-classification (card); https://huggingface.co/datasets/atrost/financial_phrasebank (card).

---

## 11. Addendum (2026-10-03): datasets added by the finished master list

**Scope.** Every (dataset, metric) key in `jev-published.md` §2 that §2–§5 above did not cover. Jev numbers, n and suites are copied from `bench/public/jev_published.json` (397 rows); tiers as defined there (A full split / large run; B ≥200 items; C <200 or partial). Sizes and licences are from the HF Hub API and `datasets-server /size`, swept 2026-10-03 [F] unless marked. Same tags as above ([F] fetched, [R] recalled, [E] estimate, [I] inference). This pass also read three paper PDFs through WebFetch (Qwen3Guard, ChaosNLI, Ziems et al.; a few pages each), jev-bench's 14.5 KB `manifest.json`, and the local v2 build reports. Still no dataset, model, az or torch.

### 11.1 jev-bench (`Praveenrajus/jev-bench` v0.1.1, licence "other / mixed-see-manifest") — the one public Jev suite that **ships train splits**

Protocol [F: card + `manifest.json`, built 2026-09-20, seed 20260920]: uniform random samples of the upstream split, natural label balance; caps test ≤1,000 (civil_comments 2,000; chaosnli all 1,599), validation ≤500, train ≤8,000. MNLI test = `validation_matched`. civil_comments train/val are drawn from the HF **validation** split. go_emotions was rebuilt in v0.1.1 from the `raw` config with rater vote shares as soft labels. Jev 1.13.0 macro accuracy over 22 configs: **0.733** (22,773 test records).

**S-track rule for this suite:** train on jev-bench's own `train` config (or the upstream train split, deduped against jev-bench test), select on jev-bench `validation`; never use upstream validation rows that jev-bench sampled into its test (MNLI matched, BoolQ, PAWS, STS-B, HelpSteer2).

| Config (test n) | Upstream (HF id / config) | Jev acc | jev-bench train / val | Upstream train (licence) | Best public supervised reference | S verdict |
|---|---|---|---|---|---|---|
| banking77 (1,000) | `mteb/banking77` | 0.796 | 8,000 / 500 | 9,993 (CC BY 4.0) | BERT 93.66 | **Win ≤68M** |
| clinc150 (1,000) | `clinc/clinc_oos` plus | 0.893 | 8,000 / 500 | 15,250 (CC BY 3.0) | §2.1 | Narrow; OOS recall decides |
| massive (1,000) | `mteb/amazon_massive_intent` en | 0.808 | 8,000 / 500 | 11,514 (CC BY 4.0) | XLM-R-base 88.3, mT5-base-enc 89.0 (full en test) [F: §5] | **Win ≤110M** |
| ledgar (1,000) | `coastalcph/lex_glue` ledgar | 0.751 | 8,000 / 500 | 60,000 / val 10,000 (CC BY 4.0) | BERT μ-F1 87.6, DeBERTa 88.2 [F: LexGLUE README] | **Win ≤110M** (+12) |
| go_emotions (1,000) | go_emotions `raw`, plurality label | 0.282 | 8,000 / 500 | simplified 43,410 (Apache-2.0) | none in this format | **Likely win** [E]; measure |
| mmlu (1,000) | `cais/mmlu` all | 0.923 | 285 (= MMLU dev) / 500 | auxiliary_train 99,842 | §2.3 | **Lose** |
| arc_challenge (1,000) | `allenai/ai2_arc` ARC-Challenge | 0.979 | 1,119 / 299 | 1,119 | §2.3 | **Lose** |
| mnli (1,000; = validation_matched) | `nyu-mll/glue` mnli | 0.883 | 8,000 / 500 | 392,702 | Ettin 68M 87.0 / 150M 89.2 / 400M 91.3 (MNLI-m dev) [F: §1] | **Win at 150M** (narrow), clear at 400M |
| chaosnli (1,599; eval only) | `metaeval/chaos-mnli-ambiguity` (ChaosNLI-M) | 0.615 | none | train on SNLI+MNLI; **exclude MNLI validation_matched** (ChaosNLI-M is a subset of it) | Accuracy vs new 100-vote majority: RoBERTa-L **0.6354**, XLNet-L 0.6185, BART-L 0.5922, RoBERTa-b 0.5922, BERT-b 0.5591; estimated human 0.86 [F: Nie et al. 2020, arXiv 2010.03532, Table 5] | **Win at ~355M** (+2); base models lose |
| sst5 (1,000) | `SetFit/sst5` | 0.565 | 8,000 / 500 | 8,544 | RoBERTa-L 60.2 [§2.5] | **Win at ~400M** |
| yelp5 (1,000) | `Yelp/yelp_review_full` | 0.685 | 8,000 / 500 | 650,000 (Yelp terms) | XLNet-L 72.95 [F: §5]; BERT-base ITPT-FiT 70.58 (err 29.42) [R: arXiv 1905.05583] | **Win at ~110M** (+2), +4 at large |
| helpsteer2_helpfulness (1,000) | `nvidia/HelpSteer2` | 0.363 | 8,000 / 500 | 20,324 (CC BY 4.0) | none per-level | **Likely win** [E] |
| helpsteer2_verbosity (1,000) | `nvidia/HelpSteer2` | 0.341 | 8,000 / 500 | 20,324 | none | **Likely win** [E] |
| stsb (1,000; 6-level acc) | `sentence-transformers/stsb` | 0.538 | 5,749 / 500 | 5,749 | Spearman only (§2.5) | Measure (rounding of a regressor) |
| measuring_hate_speech (1,000; 3-level) | `ucberkeley-dlab/measuring-hate-speech` | 0.527 | 8,000 / 500 | **single split** 135,556 annotator rows (CC BY 4.0) | none in this format | **Likely win** [E]. Use jev-bench's comment-level train only (one-split source) |
| boolq (1,000) | `google/boolq` (validation) | 0.917 | 8,000 / 472 | 9,427 | DeBERTa-v3-L 0.8835 | **Lose** |
| fever_evidence (1,000; NEI dropped) | `copenlu/fever_gold_evidence` | 0.972 | 8,000 / 500 | 228,277 / 15,935 / 16,039 (CC BY-SA 3.0 + GPL-3.0 tags) | RoBERTa with gold evidence ≈ 88.7 veracity acc (3-way incl. NEI; Atanasova et al. 2022) [F: search summary; protocol differs] | **Lose / tie**; measure 2-way |
| paws (1,000) | `google-research-datasets/paws` labeled_final | 0.846 | 8,000 / 500 | 49,401 | §2.2 (≈0.90–0.92) | **Win ~68–150M** |
| civil_comments (2,000; ~8% toxic) | `google/civil_comments` | 0.729 | 8,000 / 500 (from HF validation) | 1,804,874 (CC0) | Majority-class baseline ≈ 0.92 [I: 1 − 8%]; WILDS DistilBERT ERM avg acc ≈ 92 [R] | **Win at ≤66M** (Jev is below the majority baseline) |
| sms_spam (800) | `ucirvine/sms_spam` (single split, carved) | 0.965 | **4,000 / 237** | — | ≥0.98–0.99 acc [§2.1] | **Win ≤32M** (a real train split exists here, unlike Deußer) |
| strategyqa_closed (687) | `ChilleD/StrategyQA` (train 1,603 / test 687; MIT) | 0.785 | 1,400 / 160 | 1,603 | RoBERTa*-∅ 63.6; RoBERTa* with oracle paragraphs 70.7 (Geva et al. 2021) [F: search summary] | **Lose** |
| strategyqa_grounded (687; facts in state) | same | 0.956 | 1,400 / 160 | 1,603 | no public number with gold *facts* (ORA-P uses paragraphs, 70.7) | **Lose likely**; measure |

### 11.2 LexGLUE, all 7 tasks (chepyle/jev-test, full official test) and CUAD

Upstream: `coastalcph/lex_glue` (CC BY 4.0) [F sizes]. Reference rows: LexGLUE README (micro/macro-F1; CaseHOLD accuracy) [F: github.com/coastalcph/lex-glue]; large models from MultiLegalPile (Niklaus et al., ACL 2024, Table 6, **macro-F1**) [F: arXiv 2306.02069].

| Task (test n) | Jev μ-F1 | Train / val | Base-size supervised (μ-F1 / m-F1) | Large supervised | S verdict |
|---|---|---|---|---|---|
| ECtHR A (1,000) | 0.730 | 9,000 / 1,000 | BERT 71.2/63.6; Legal-BERT 70.0/64.0; Longformer 69.9/64.7 | RoBERTa-L 73.8 μ-F1 [search summary, unverified]; Legal-en-RoBERTa-L m-F1 70.3 | **Tie at large**; lose at base |
| ECtHR B (1,000) | 0.754 | 9,000 / 1,000 | Legal-BERT **80.4**/74.7; BERT 79.7/73.4 | Legal-en-R-L m-F1 77.0 | **Win ≤110M** (+5) |
| SCOTUS (1,400) | 0.726 | 5,000 / 1,400 | CaseLaw-BERT **76.6**/65.9; Legal-BERT 76.4/66.5; BERT 68.3 | Legal-en-R-L m-F1 67.7 | **Win at 110M with legal pretraining**; generic BERT loses |
| EUR-LEX (5,000) | 0.391 | 55,000 / 5,000 | DeBERTa 72.1/57.4; BERT 71.4/57.2 | — | **Win ≤110M** (+33) |
| LEDGAR (10,000) | 0.753 | 60,000 / 10,000 | CaseLaw-BERT 88.3/83.0; DeBERTa 88.2/83.1 | — | **Win ≤110M** (+13) |
| UNFAIR-ToS (1,607; LexGLUE convention) | 0.764 | 5,532 / 2,275 | Legal-BERT 96.0/83.0 | — | **Win ≤110M** |
| CaseHOLD (3,600; 5-way MC) | 0.773 | 45,000 / 3,900 | CaseLaw-BERT 75.4; Legal-BERT 75.3; DeBERTa 72.6 | **Legal-en-R-L 77.0**; RoBERTa-L 74.4 (m-F1) | **Tie at large legal encoder**; lose at base |
| 7-task arithmetic mean | 0.699 zero-shot / **0.742 tuned** | — | Legal-BERT **79.8**/72.0; RoBERTa-L 79.4/70.8 | — | **Win ≤110M** (legal-pretrained backbone) |

- Backbone: Ettin is general-domain; a LexGLUE S run should start from a legal-pretrained encoder (Legal-BERT, CaseLaw-BERT, `lexlms/*`) or continue pretraining Ettin on a legal corpus [I].
- **CUAD** (matu79go/jev-hanko, 41 clauses × 500 pages, F1 0.519): `theatticusproject/cuad` HF has one `train` config of 84,325 QA rows (CC BY 4.0); the official split is 408 train / 102 test contracts [R]. Published supervised numbers are span-extraction AUPR (DeBERTa-xlarge 47.8 AUPR, Precision@80%R 44.0 [R: Hendrycks et al. 2021]) — a different metric from matu79go's per-page clause-presence F1. **Unknown; measure**, and confirm which contracts matu79go sampled before training.

### 11.3 Safety, moderation and injection (mbburabak, ASEVlad, LangWatch, Laya-ft, Red Hat)

Reference classifiers: **Qwen3Guard Technical Report** (arXiv 2510.14276, Tables 2–3, F1) [F: read from the PDF], WildGuard paper (arXiv 2406.18495) [F].

| Dataset (n) | Metric | Jev | Train data (licence) | Best public reference at ≤1B | Larger references | S verdict |
|---|---|---|---|---|---|---|
| Aegis 1.0 prompt (359 = AegisSafetyTest) | F1 harmful | 0.891 | `nvidia/Aegis-AI-Content-Safety-Dataset-1.0` train 10,798 / test 1,199 (CC BY 4.0) | Qwen3Guard-Gen-0.6B strict **90.8** | WildGuard-7B 89.4; PolyGuard-Qwen-7B 90.3; Qwen3Guard-8B 91.4 | **Narrow win at 0.6B** (decoder); encoder unknown |
| Aegis 2.0 prompt (1,928) | F1 harmful | 0.836 | Aegis 2.0 train 30,007 / val 1,445 / test 1,964 (CC BY 4.0) | Qwen3Guard-0.6B **85.0** | NemoGuard-8B 86.8 | **Win at ~0.6B** with in-domain train |
| Aegis 2.0 response (1,928) | F1 harmful | 0.802 | same | Qwen3Guard-0.6B **84.2** | NemoGuard-8B 87.6 | **Win at ~0.6B** |
| HarmBench response (596) | F1 harmful | 0.875 | none official (HarmBench classifier val set) | Qwen3Guard-0.6B 85.0 | Qwen3Guard-8B 87.2; HarmBench-Mistral cls 87.0; WildGuard 86.3 | **Lose** on public evidence |
| HarmBench prompt (239, all harmful) | detection rate | 0.992 | none | Qwen3Guard-0.6B prompt F1 98.7 | 4B/8B 100.0 | Tie at best |
| WildGuardTest prompt (1,699) | F1 harmful | 0.884 | `allenai/wildguardmix` train 86,759 (ODC-BY, gated) | Qwen3Guard-0.6B 87.7 | WildGuard-7B **88.9**; Qwen3Guard-8B 88.9 | **Lose / tie** |
| HateCheck (3,728) | F1 hateful | 0.992 | **test only** (`Paul/hatecheck`, CC BY 4.0) | — | — | **Z only**; Jev near ceiling |
| XSTest (250 safe prompts) | false-alarm rate ↓ | 0.076 | gated `walledai/XSTest` (CC BY 4.0), eval only | — | — | **Z only** |
| ToxicChat (mbburabak 2,853) | F1 toxic | 0.757 | §2.4 | Qwen3Guard-0.6B loose 77.7 | ToxicChat-T5-L 0.822 | as §2.4 |
| Combined injection corpus (ASEVlad, 11,900) | AUPRC | 0.980 | built from **train+test** of `xTRam1/safe-guard-prompt-injection` (8,236 / 2,060), `jackhhao/jailbreak-classification` (1,044 / 262; Apache-2.0), `deepset/prompt-injections`, `leolee99/NotInject` (3 × 113; MIT) | — | — | **No clean train** (the eval pool contains the train splits); Z only |
| deepset all-662 (Gaurav-Gosain) / 546 FNR (ca7ai) / 662 F1 (switchboard) | acc / FNR / F1 | 0.965 / 0.473 / 0.938 | the eval includes `deepset` train | — | — | **Not S-comparable**; use the 116-test row (§2.4) |
| LangWatch prompt injection (1,000) | catch @5% false alarms | 0.946 | components: deepset, jackhhao, `reshabhs/SPML_Chatbot_Prompt_Injection` (16,012; MIT), `djapp18/JailbreaksOverTime` | — | — | Sample provenance unknown; **Z only** until LangWatch's item ids are known |
| LangWatch moderation (667) | AUROC | 0.903 | OpenAI-mod eval (eval-only) + Aegis 2.0 | — | — | **Z only** (half the pool has no train split) |
| LangWatch PII (1,000) | catch @5% FA | 0.908 | `gretelai/gretel-pii-masking-en-v1` 50,000 / 5,000 / 5,000 (Apache-2.0); `nvidia/Nemotron-PII` 100,000 / 100,000 (CC BY 4.0); `beki/privy` (MIT) | — | — | Plausible win with token-level PII training [E]; measure |
| Red Hat NeMo Guardrails EvalHub (injection 0.8635, safety 0.862) | acc | — | corpora not public | — | — | skip |

### 11.4 Intent and routing additions

| Dataset (n) | Metric | Jev | Train (licence) | Best public supervised | S verdict |
|---|---|---|---|---|---|
| HWU64 (test 1,076; thisisandreeeee) | acc | **0.831** | `DeepPavlov/hwu64` train 8,954 (no HF tag; our registry says CC BY-SA 3.0) | BERT-base 91.6, RoBERTa-base 92.1, USE+ConveRT 92.62, ConvFiT 92.98, **SPACE-2 94.33** [F: search summary of arXiv 2109.10126 / SPACE-2] | **Win ≤110M** (+8). See §12.1 for the dev-set conflict |
| MASSIVE en-US (laya-ft frozen 1,000) | macro-F1 | 0.799 | `AmazonScience/massive` en-US 11,514 (CC BY 4.0; size endpoint 500 today) | accuracy refs only (88–89) | **Likely win ≤110M** |
| MASSIVE sv-SE (360 = 20 × 18 scenarios) | acc | 0.889 | sv-SE train 11,514 | per-locale numbers in the MASSIVE paper (not re-read) | Measure; multilingual backbone needed |
| tanaos synthetic intent (3,447) | acc | 0.878 | `tanaos/synthetic-intent-classifier-dataset-v1` **single split** 11,489 (MIT); xxkuboxx carved its own test | — | Win likely [E], but only with xxkuboxx's exact split |
| LangWatch routing, BANKING77 20 / 77 intents (1,000 / 500) | acc | 0.891 / 0.796 | PolyAI train | 93.66 (77-way) | **Win** |
| LangWatch off-topic (1,000) | balanced acc | 0.934 | CLINC oos + MASSIVE + `benayas/snips` train splits | — | Measure |
| LangWatch complaint routing (1,000) | acc | 0.787 | `BEE-spoke-data/consumer-finance-complaints` **single split** 4,707,579 (CC0) | product-classification BERTs ≈0.85+ [E] | **Likely win** [E]; carve by complaint id |
| LangWatch commit type (1,000) | acc | 0.683 | none packaged; conventional-commit prefixes on GitHub give free labels | — | Measure |
| LangWatch tool routing (BFCL, 1,000) | acc | 0.783 | none (§6.6) | — | as BFCL |
| Darija reviews (171; mouadse) | acc | 0.795 | small HF Darija sets | — | tier C; skip |

### 11.5 Reranking and retrieval

| Dataset (n) | Metric | Jev | Train | Public rerankers on the same protocol | S / Z verdict |
|---|---|---|---|---|---|
| BEIR SciFact, BM25 top-100 (300) | nDCG@10 | 0.770 (denser); 0.772 (hev, top-30) | `allenai/scifact` 809 train claims (CC BY-NC 2.0); MS MARCO for Z | monoBERT-340M 71.36; monoT5-220M 73.40; DeBERTa-L distilled 74.18; Cohere-v2 74.44; **monoT5-3B 76.57**; BM25 67.89 [F: RankGPT, arXiv 2304.09542 Table 1]; MiniLM-L6 CE 0.688 [R: BEIR] | **Lose ≤1B** on public numbers |
| BEIR NFCorpus (323) | nDCG@10 | 0.362 | NFCorpus train 2,590 queries [R]; MS MARCO | monoBERT 36.88; **monoT5-220M 37.38**; DeBERTa-L distilled 38.48; monoT5-3B 38.97 [F: same table]; MiniLM CE 0.350 [R] | **Win at ~220M, and on the Z track** (MS MARCO-trained cross-encoders never see NFCorpus) |
| BEIR FiQA (300 queries; hev) | nDCG@10 | 0.376 | FiQA train 5,500 queries [R]; MS MARCO | MiniLM CE 0.347 [R: BEIR] | Plausible win at ~220M+ [E]; measure |
| NevIR (1,383 test pairs) | pairwise acc | 0.710 | `orionweller/NevIR` train 948 / val 225 / test 1,383 (MIT) [F] | Zero-shot: monoT5-base 34.9, -large 45.8, -3B **50.6**; fine-tuned on NevIR train: monoT5-base ≈70–75 by epoch 20 (read off Fig. 5), ColBERTv1 ≈60–65 [F: arXiv 2305.07614] | **Tie at ~220M** with NevIR train; Z loses |
| MS MARCO v1.1 rerank (100 queries; jev-decision-bench) | MRR | 0.499 | MS MARCO train | — | tier B small; measure |
| BEIR+BRIGHT+CodeSearchNet rerank mean (8 sets, 1,617 queries) | mean nDCG@10 | 0.692 | per component | — | composite; measure |

### 11.6 Reward models, judges and process supervision (goya4140, official harnesses)

No dataset in this block has an official train split: `THU-KEG/RM-Bench` "train" 1,327 is the eval, `hitsmy/PRMBench_Preview` "train" 6,216 is the eval, `allenai/reward-bench` raw 5,123 / filtered 2,985 and `allenai/reward-bench-2` 1,865 are eval-only (ODC-BY), `lmarena-ai/PPE-Human-Preference-V1` 16,038 is test-only, `Qwen/ProcessBench` 3,400 is eval-only (Apache-2.0). Training must come from generic preference / PRM data (HelpSteer3, UltraFeedback, Skywork preference sets, PRM800K, Math-Shepherd), deduped against these evals. **PPE is drawn from Chatbot Arena votes**, so exclude `lmarena-ai/arena-human-preference-*` (and anything built from them) or the row is contaminated [I].

| Benchmark (n) | Jev | Best public ≤1B | S verdict |
|---|---|---|---|
| RewardBench v1 (2,985) | 0.926 | Skywork-Reward-V2-Qwen3-0.6B **85.2** (26M preference pairs) [F: model card] | **Lose** |
| RewardBench 2 (1,865) | 0.812 | Skywork-V2-0.6B 61.3 | **Lose** |
| RM-Bench pointwise (7,962) / pairwise (1,327) | 0.838 / 0.813 | Skywork-V2-0.6B 74.4 (official RM-Bench) | **Lose** |
| PPE human preference (16,038, no-tie pairwise acc) | 0.644 | Skywork-V2-0.6B **65.3** (PPE Preference) | **Narrow win at 0.6B** (decoder RM); encoder unknown |
| RubricBench (1,147) | 0.760 | — | **Lose likely** |
| PRMBench Preview (6,216) | 0.664 | — | **Lose** [I] |
| ProcessBench (3,400) | 0.695 | — (best open PRMs are 7B+) [R] | **Lose** |

### 11.7 Ibrahim & Zaki 2026 CSS annotation tasks (arXiv 2609.24574, runs 2026-09-20)

Items: Ziems et al. (2024, *Computational Linguistics* 50(1)) released test splits, class-stratified, ≤500 per task (7,977 total) [F: arXiv 2609.24574v1]. Ziems et al. Table 3 reports a **supervised RoBERTa-large** baseline on the same test sets (grid-searched LR/batch/epochs, mean of 3 seeds, macro-F1) [F: read from arXiv 2305.03514v3, p. 21]. Training splits come from each original dataset (Ziems' `baselines/` scripts); the repo `SALT-NLP/LLMs_for_CSS` has **no licence** and ships test files plus model answers.

| Task (n) | Jev macro-F1 | Ziems RoBERTa-L fine-tuned | Random | S verdict |
|---|---|---|---|---|
| Emotion (CARER; 498) | 0.484 | **71.6** | 16.7 | **Win** (+23) |
| Figurative / FLUTE (500) | 0.863 | **99.2** | 25.0 | **Win** (+13) |
| Humor (Reddit; 500) | 0.563 | **73.1** | 49.5 | **Win** (+17) |
| Ideology (IBC; 498) | 0.633 | **64.8** | 33.3 | Narrow win (+1.5) |
| Implicit hate (498; pilot) | 0.439 | **62.5** | 16.7 | **Win** (+19) |
| Misinfo Reaction Frames (500) | 0.796 | **81.6** | 50.0 | Narrow win (+2) |
| Persuasion, utterance (RAOP; 399) | **0.607** | 52.0 | 14.3 | Jev ahead of RoBERTa-L |
| Semantic change (TempoWiC; 344) | **0.683** | 62.3 | 50.0 | Jev ahead |
| Stance (SemEval-16; 435; pilot) | **0.734** | 36.1 | 33.3 | Jev ahead (RoBERTa baseline weak) |
| Indian-English dialect (266) | **0.640** | 3.0 | 3.3 | Jev ahead (baseline failed) |
| Discourse acts (497; pilot) | **0.595** | 49.6 | 14.3 | Jev ahead |
| Empathy (TalkLife; 498) | 0.269 | **71.6** | 33.3 | **Win**, but TalkLife is not public; S impossible without the data |
| Persuasion strategies, conversation (434) | **0.561** | 33.3 | 50.0 | Jev ahead |
| Politeness (Wikipedia; 498) | 0.573 | **75.8** | 33.3 | **Win** (+19) |
| Power (Wiki talk; 500) | 0.581 | **72.7** | 49.5 | **Win** (+15) |
| Toxicity forecasting (CGA; 500) | 0.502 | **64.6** | 50.0 | **Win** (+14) |
| Media ideology (498) | 0.654 | **85.1** | 33.3 | **Win** (+20) |
| Character tropes (114) | 0.191 | – (not run) | 36.9 | Unknown |

So a RoBERTa-L-class S model beats Jev on 11 of the 17 tasks that have a supervised baseline (10 if TalkLife, whose data is not public, is dropped); Jev leads on 6, mostly where Ziems' supervised baselines are weak or broken [I]. Two of the 11 wins are narrow (IBC +1.5, MRF +2).

### 11.8 jev-decision-bench (OmarMujahid) small-n rows (200 items unless noted; tier B)

| Row | Jev | Train | Reference | Verdict |
|---|---|---|---|---|
| TREC coarse | 0.925 | `SetFit/TREC-QC` 5,452 / test 500 | TREC-6 fine-tuned BERT ≈97–98 [R] | **Win** |
| CoNLL-2003 entity typing | 0.895 | `tner/conll2003` 14,041 / 3,250 / 3,453 (licence "other") | typing given gold spans ≈0.95+ [E] | **Win**; but v2 already trains on conll2003 (§12.2) |
| SQuAD v1.1 sentence / value selection (200 / 155) | 0.945 / 0.929 | SQuAD v1.1 train 87,599 [R] | — | Likely win [E]; confirm the sampled split |
| AJGT Arabic tweets (AUROC) | 0.954 | `komari6/ajgt_twitter_ar` single split 1,800 (licence unknown) | AraBERT ≈93.8 acc [R] | Arabic encoder needed; S-cv |
| XNLI en / fr / ar / hi / sw / zh (150 each) | 0.86 / 0.773 / 0.74 / 0.707 / 0.70 / 0.793 | `facebook/xnli` 392,702 MT train per language / val 2,490 / test 5,010 | XLM-R-L cross-lingual transfer: 89.1 / 84.1 / 79.8 / 76.9 / 73.9 / 80.2 [R: Conneau et al. 2020] | **Win at 560M multilingual** (CIs ±7 at n=150) |
| TruthfulQA MC1 | 0.95 | none (validation 817 only) | small models ≈25–45 [R] | **Lose** |
| LogiQA | 0.765 | `lucasmccabe/logiqa` 7,376 / 651 / 651 | RoBERTa ≈35 [R: LogiQA paper] | **Lose** |
| MNLI-m / ANLI-R3 / STS-B ρ / Yelp ρ / SST-2 AUROC / BoolQ AUROC / PAWS AUROC / civil AUROC | 0.84 / 0.66 / 0.931 / 0.927 / 0.981 / 0.969 / 0.916 / 0.872 | as §2 / §11.1 | as §2 | as the full-split rows; n=200 rows are not headline bars |
| HellaSwag / WinoGrande / MMLU / ARC-C / GSM8K / CSQA | 0.95 / 0.885 / 0.935 / 0.975 / 0.725 / 0.87 | §2.3 | §2.3 | **Lose** |
| BoolQ with explicit negations | 0.805 | BoolQ train (+ synthetic negations) | — | Measure |
| SST-2 under prompt injection, clean / attacked (150) | 0.96 / 0.947 | SST-2 train | — | Measure (an encoder with no instruction channel should be immune) |

### 11.9 Exams and non-English knowledge (all **Lose** for ≤1B encoders)

| Benchmark (n) | Jev | Train / dev on HF (licence) |
|---|---|---|
| TMMLU+ v1.1, 66 subjects (19,646) | 0.776 | `ikala/tmmluplus` ≤5 dev + small val per subject (MIT) |
| GAOKAO-Bench objective (1,497) | 0.909 | `OpenLMLab/GAOKAO-Bench` (HF gated / 401 for us) — eval only |
| JMedQA 2018–2026 (3,556, exact-set) | 0.886 | none public as train |
| JMMLU professional_accounting (150) | 0.747 | `nlp-waseda/JMMLU` (CC BY-NC-ND 4.0 → no derivatives) |
| ThaiExam (567) | 0.707 | `scb10x/thai_exam` (Apache-2.0; size endpoint 404) |
| ENEM 2025 (182) | 0.566 | none |
| SAT practice tests (1,009) | 0.912 | none (College Board) |
| jfinqa (1,000) | 0.731 | unknown |
| KoBEST BoolQ / COPA / HellaSwag / SentiNeg / WiC (80 each; tier C) | 0.988 / 0.988 / 0.775 / 0.938 / 0.888 | `skt/kobest_v1` train 3,665 / 3,076 / 2,029 / 3,649 / 3,318 (CC BY-SA 4.0) — S possible with a Korean encoder, but n=80 rows are noise |

### 11.10 One-off studies

| Dataset (n; study) | Metric | Jev | Train data | Public supervised reference | S verdict |
|---|---|---|---|---|---|
| Upworthy headline A/B, confirmatory pairs (10,984; Gaurav-Gosain) | pairwise acc | 0.645 | Upworthy Research Archive **exploratory + holdout** only (the eval is the confirmatory split) | `NovusEdge/vera-deberta-v3-large` (435M), trained on exploratory+confirmatory: **0.689** on all 18,485 holdout pairs, 0.671 on 10,946 near-duplicate-free pairs, 0.843 on p<0.05 pairs [F: model card] | **Plausible win at ~435M** (+3–4, different split). VERA is fine-tuned from `com-kotobalabs/open-jev-deberta-v3-large`: check that lineage for Jev distillation before using it as an init |
| CommonLit readability (300 of the 4,724-excerpt CLEAR corpus; keltokhy) | Pearson r | 0.824 | Kaggle CommonLit Readability Prize train 2,834 (competition licence) or CLEAR minus the 300 | single RoBERTa-L RMSE ≈0.46 [F: search]; target SD ≈1.03 [R] ⇒ r ≈ 0.89 [E] | **Likely win at ~355M** |
| Real/fake job postings (1,000 of a 3,182 test; geckguy) | PR-AUC (fraud) | **0.276** | EMSCAD 17,880 ads, 866 fraudulent (Kaggle) | Fraud-BERT F1 0.93, acc 0.99 [F: search summary, Springer 2025] | **Win ≤110M** (huge); needs geckguy's split |
| Android app reviews (1,000; goodrahstar) | Spearman | 0.803 | `sealuzh/app_reviews` **single split** 288,065 (licence unknown) | — | Likely win [E]; carve by review id |
| Amazon counterfactual en (670 test; laya-ft) | macro-F1 | 0.865 | `SetFit/amazon_counterfactual` en 4,018 / 335 / 670; en-ext 8,000 / 666 / 1,334 (no HF tag) | XLM-R F1 up to 0.96 on English (with SemEval-2020 T5 data) [F: search summary of O'Neill et al. 2021] | **Likely win ≤150M** |
| Reddit AITA (770; dchristopoulos) | acc | 0.754 | no fixed public split | — | Measure |
| Eclipse bug severity (2,000; bakeoff) | acc | 0.491 | public Eclipse/Mozilla severity corpora | — | Measure |
| NSL-KDD flows (900 pilot; tier C) | F1 attack | 0.859 | KDDTrain+ 125,973 [R] | tabular task; GBDT ≥0.8 [E] | Not a text task; skip |
| Cohen 2006 screening (16,015; 15 reviews) / ADHD (851) | recall / acc | 0.90 / 0.946 | none (all rows evaluated) | — | **S-cv only** |
| MedHallu (1,000; dev-tuned question + threshold) | acc | 0.929 | `UTAustin-AIHealth/MedHallu` pqa_artificial 9,000 (eval = pqa_labeled 1,000; no HF licence tag) | — | Measure |
| HaluEval RAG faithfulness (666; LangWatch) | balanced acc | 0.803 | `pminervini/HaluEval` (Apache-2.0) has **no train split** (qa 10,000, dialogue 10,000, summarization 10,000, general 4,507) | — | Z only (or S-cv on the non-sampled QA items) |
| TabFact (120 balanced; tier C) | acc | 0.917 | `wenhu/tab_fact` train ≈92k [R] (CC BY 4.0) | TAPEX-large ≈84.2 test [R] | **Lose ≤400M** |
| WANLI (256; Kev) | acc | 0.758 | `alisawuffles/WANLI` 102,885 (CC BY 4.0) | WANLI-trained RoBERTa-L ≈75 [R] | Tie; and v2 already trains on WANLI (§12.2) |
| Mind2Web (706 steps from the **train_6** shard; hosamsh) / JevForge-Mind2Web (2,329; LangWatch) | top-1 / acc | 0.521 / 0.708 | `osunlp/Mind2Web` train 1,009 tasks (CC BY 4.0) — **exclude train_6**; `AndeyTait/JevForge-Mind2Web` gold train 4,642 / dev 786 / calibration 400 / test 800 / ood 386 (CC BY 4.0) | MindAct DeBERTa candidate ranking [R] | Measure; LangWatch's split is unknown |
| Who&When Pro (6,257; TokenTrim) | "All" score | 0.313 | none | — | Z only |
| Email spam (Arize 18,514; bitnovus 5,733) | acc | 0.983 / 0.986 | Kaggle / Enron / Ling-Spam corpora; all rows evaluated | ≈0.99 [E] | **S-cv only** |
| arXiv categories ≥2026-09-17 (258; zhuyansen) | acc | 0.891 | pre-cutoff arXiv abstracts (dedupe by id) | — | Measure (v2 already trains on an `arxiv` source) |

---

## 12. Track hygiene problems found while doing the addendum

### 12.1 HWU64 is our dev set and is now a public Jev benchmark
- `jev_local/bench/registry.py:171` registers `hwu64` as split "dev" from `DeepPavlov/hwu64` **test** (cap 2,000), and `docs/research/v2/PLAN.md` moved HWU64 to dev.
- thisisandreeeee/jev-benchmarks reports Jev **0.831** on that same 1,076-item test (tier A).
- Consequence: any checkpoint selected on HWU64 test cannot claim a clean Z number on HWU64. Options: (a) move dev to a dataset with no public Jev number (ATIS has none in `jev-published.md` §1; SNIPS appears only inside LangWatch's off-topic mix), or (b) keep HWU64 as dev and report HWU64 only as "dev-selected, not counted". For S, train on its 8,954-row train split.

### 12.2 The v2 training mix is not Z-clean for several public Jev datasets
From `docs/v2/build/phase2/mixture_stats.json` (`mix-v2.0-no-b6`, built 2026-10-02T03:37Z) and `decontam_report.json` [F, local files]:
- `b2_label_semantics` includes **aegis2, civil_comments, mhs (measuring-hate-speech), ledgar, mtop, trec, dbpedia_14 / dbpedia_l123, arxiv, ultrafeedback**.
- `b3_nli` includes **mnli, fever_nli, wanli, lingnli, docnli, vitaminc**.
- `b8_extractive` includes **squad_v2, conll2003**.
- `b1_tasksource_jev` (538 groups; only the 51 capped ones are listed in the JSON) includes at least **art (αNLI), commonsense_qa, lex_glue/case_hold, multilingual/xnli, civil_comments/identity_attack_share, conll2003, piqa, social_i_qa, race, swag**.
- `b5_open_jev` includes `open_jev/workflow-controls-v1/{agent_trace_observability, customer_service, invoice_processing, security_incidents}` — the same four workflow names as TypeSafe's workflow evals (whose items are unreleased; the Open-Jev card says its labels are not model predictions).

Consequences:
1. v2 numbers on Aegis 2.0, civil_comments, measuring-hate-speech, LEDGAR, MNLI / ChaosNLI, FEVER, WANLI, TREC, CoNLL typing, SQuAD selection, αNLI, CommonsenseQA, CaseHOLD, XNLI and gazelle93's label-pressure rows (MTOP, DBpedia) are **S-track (train split seen), not Z**. Label them that way.
2. **Possible test leakage, not just train-split exposure:** measuring-hate-speech has a single HF split and jev-bench sampled its test comments from that pool; v2's `mhs` source draws from the same pool. Our decontam reference (`data/v2/bench_ref/jevbench`) only hashes **our** jevbench test files, so jev-bench's mhs test comments were not removed. The same check is needed for jev-decision-bench's TREC / SQuAD / CoNLL samples (split not recorded) and LangWatch samples.
3. Fix: add every public-Jev eval item we can enumerate (jev-bench test configs; mbburabak; Deußer; Decision Index; LangWatch where ids are published; Ziems CSS test files) to the decontam reference, rebuild the manifest, and keep a per-checkpoint Z/S label per dataset.

### 12.3 Other provenance notes
- `NovusEdge/vera-deberta-v3-large` and other `open-jev-*` checkpoints: check whether `com-kotobalabs/open-jev-deberta-v3-large` was distilled from Jev outputs before using any of them as an init or a teacher (`jev-published.md` point 3 already bans `autotrust/JEV-*`).
- Rows whose eval pool includes the upstream train split (deepset all-662, ASEVlad combined injection, Arize/bitnovus spam, Cohen screening, SMS spam in Deußer) are **Z-only or S-cv**; never put them in an S headline.
- jev-bench's own licence is "other / mixed"; it carries research-only upstreams (GLUE/MNLI, SST, Yelp). Fine for evaluation and S training under research terms; flag before any commercial release [I].

---

## 13. Sources for the addendum (read 2026-10-03)

- Master list: `docs/research/benchmax/jev-published.md` §2 and `bench/public/jev_published.json` (397 rows).
- jev-bench: https://huggingface.co/datasets/Praveenrajus/jev-bench (README, `manifest.json` v0.1.1); https://github.com/uspraveen/Jevify/blob/main/docs/DATASETS.md.
- HF sizes / licences: `https://huggingface.co/api/datasets/<id>` and `https://datasets-server.huggingface.co/size?dataset=<id>` for DeepPavlov/hwu64, AmazonScience/massive, tanaos/synthetic-intent-classifier-dataset-v1, BEE-spoke-data/consumer-finance-complaints, pminervini/HaluEval, osunlp/Mind2Web, AndeyTait/JevForge-Mind2Web, nvidia/Aegis-AI-Content-Safety-Dataset-1.0 and -2.0, allenai/wildguardmix (401), Paul/hatecheck, walledai/XSTest (401), xTRam1/safe-guard-prompt-injection, jackhhao/jailbreak-classification, leolee99/NotInject, reshabhs/SPML_Chatbot_Prompt_Injection, gretelai/gretel-pii-masking-en-v1, beki/privy, nvidia/Nemotron-PII, google/civil_comments, ucberkeley-dlab/measuring-hate-speech, coastalcph/lex_glue, theatticusproject/cuad, skt/kobest_v1, ikala/tmmluplus, SetFit/amazon_counterfactual, sealuzh/app_reviews, SetFit/TREC-QC, tner/conll2003, komari6/ajgt_twitter_ar, facebook/xnli, truthfulqa/truthful_qa, lucasmccabe/logiqa, wenhu/tab_fact, UTAustin-AIHealth/MedHallu, ChilleD/StrategyQA, copenlu/fever_gold_evidence, orionweller/NevIR, allenai/reward-bench, allenai/reward-bench-2, THU-KEG/RM-Bench, lmarena-ai/PPE-Human-Preference-V1, Qwen/ProcessBench, hitsmy/PRMBench_Preview, nlp-waseda/JMMLU, scb10x/thai_exam, OpenLMLab/GAOKAO-Bench (401).
- LexGLUE: https://github.com/coastalcph/lex-glue (README tables); MultiLegalPile, https://arxiv.org/html/2306.02069 (Table 6).
- HWU64 references: ConvFiT arXiv 2109.10126 and SPACE-2 via search summary.
- Safety: Qwen3Guard Technical Report, https://arxiv.org/pdf/2510.14276 (Tables 2, 3, 7; read from the PDF); WildGuard, https://arxiv.org/html/2406.18495.
- Reward: https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B (card).
- Reranking: RankGPT, https://arxiv.org/html/2304.09542v3 (Table 1); BEIR, arXiv 2104.08663; NevIR, https://arxiv.org/html/2305.07614.
- ChaosNLI: Nie et al. 2020, https://arxiv.org/pdf/2010.03532 (Table 5, read from the PDF).
- StrategyQA: Geva et al. 2021, arXiv 2101.02235 (via search summary).
- FEVER gold evidence: Atanasova et al. 2022, TACL, arXiv 2204.02007 (via search summary).
- CSS: Ibrahim & Zaki, https://arxiv.org/html/2609.24574v1; Ziems et al. 2024, https://arxiv.org/pdf/2305.03514v3 (Table 3, p. 21, read from the PDF); repo https://github.com/SALT-NLP/LLMs_for_CSS (no licence; `css_data/`, `baselines/`).
- Upworthy: https://huggingface.co/NovusEdge/vera-deberta-v3-large (card).
- Fake jobs: Fraud-BERT, https://link.springer.com/article/10.1007/s10791-025-09502-8 (via search summary).
- CommonLit: Kaggle CommonLit Readability Prize write-ups (via search summary).
- Amazon counterfactual: O'Neill et al. 2021, arXiv 2104.06893 (via search summary).
- Local: `jev_local/bench/registry.py`; `docs/v2/build/phase2/{mixture_stats.json,decontam_report.json}`; `docs/research/v2/{PLAN.md,training-data.md}`.
