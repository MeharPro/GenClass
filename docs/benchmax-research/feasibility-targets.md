# Benchmax feasibility and targets: which published Jev numbers meharsjev can beat, at what size, on which track

- **Date:** 2026-10-02; **rev 2, 2026-10-03.** Rev 2 reconciles this note with the two parallel notes written the same night, `jev-published.md` (397 Jev rows) and `train-data-and-supervised-ceilings.md`. It changes three things:
  - it adds **46 benchmarks** that have a full-split (tier A) or ≥687-item (tier B) Jev number but were missing here (§3 groups E-A and E-B);
  - it uses the published **"Jev given train data"** number as the S-track bar wherever one exists (Banking77 .924, UNFAIR-ToS .748, GoEmotions .353, ToxicChat .793, LexGLUE .742);
  - it re-tallies (§0, §4), adds the new contamination traps (§1.4), and re-costs the S run for the larger train mix (§7).
  
  Rev 2 also read the LexGLUE README leaderboard (`gh api`) and the Skywork-Reward-V2 model card, and folded in the sibling addendum summary (`train-data-and-supervised-ceilings.md` §0.4, 2026-10-03). Nothing else was fetched.
- **Status:** research only. Nothing was trained, downloaded at scale, provisioned or posted. Remote reads were web pages, `gh api` reads and three small JSON files (Decision Index board JSON, 0.95 MB; Dinah-0 `scores.json`, 33 KB; Deußer per-dataset result JSONs, ≤15 KB each), parsed with the Python stdlib in the session scratchpad.
- **Constraint this note is written for:** the user will spend at most **$5 on Jev, ever**. Every comparison below is against **Jev's published numbers**. No Jev call is needed for anything in this note.
- **Tracks.** **Z** = zero-shot: no data from the benchmark (any split) or its siblings in training. **S** = supervised/specialist: the benchmark's *train* split (and other public non-test data) allowed, its evaluated split never touched, disclosed.
- **Sizes.** Ettin encoders 68m / 150m / 400m / 1b (`jhu-clsp/ettin-encoder-*`, MIT), with our typed heads. Plus an optional prefill-only Qwen3 scorer (§6).
- **Tags.** [V] verified today from the cited source. [N] taken from the project's other notes (`docs/research/v2/*.md`, `docs/research/leaderboards/*.md`, and the sibling `docs/research/benchmax/*.md` and `bench/public/jev_published.json`), which cite their sources. [R] recalled from the literature and not re-fetched today: verify before quoting publicly. [E] estimate.
- **Verdicts.** **W** likely win: expected score clears the significance bar (§1.2). **T** toss-up: expected range straddles Jev's number, or can beat the point estimate but probably not the bar. **L** unlikely.

---

## 0. Bottom line

1. **"Better everywhere" is not achievable, on either track, at any size we can train.** Rev 2 counts **131** public benchmarks with a full-split or large-sample published Jev number:
   - the **85** of groups A–C (Deußer, Decision Index, and the large third-party runs);
   - **46 more** found by the parallel sweep: 33 tier-A rows (group E-A: LexGLUE, HWU64, the safety suite, reward-model suites, MedHallu, reranking and more) and 13 tier-B `jev-bench` rows on public test splits (group E-B).

   S verdicts use the stricter "Jev given train data" bar where one is published.

   | Track, size | Core 85: W | T | L | All 131: W | T | L |
   |---|---|---|---|---|---|---|
   | **S, 1b (best size)** | **27** | 26 | 32 | **39** | 46 | 46 |
   | S, 400m | 22 | 28 | 35 | 33 | 43 | 55 |
   | S, 150m | 18 | 23 | 44 | 26 | 33 | 72 |
   | S, 68m | 11 | 24 | 50 | 18 | 30 | 83 |
   | **Z, 1b (best size)** | **2** | 21 | 62 | **2** | 33 | 96 |
   | Z, 68m | 0 | 10 | 75 | 0 | 13 | 118 |

   Rev 1 had S-1b at 28 W of 85. The drop to 27 is Banking77: against Jev's few-shot .924 it becomes a toss-up.

   **46 of the 131 are L on both tracks at every size.**
   - **32 from the core set:**
     - knowledge and multi-step reasoning: MMLU, MMLU-Pro, GPQA, BBH, BIG-bench, ARC, OpenBookQA, CSQA, HellaSwag, WinoGrande, MuSR, CRUXEval, legit GSM8K, HLE, SATA-Bench, ForecastBench, New Yorker;
     - multilingual: SIB-200, Belebele, AfriXNLI, C-Eval;
     - no usable train split, or Jev already beats every public specialist: BoolQ, NLI4CT, LLM-AggreFact, SummEval, PubMedQA, BFCL, API-Bank, BRIGHT, Home appliance.
   - **14 new:**
     - reward models and PRMs: RewardBench v1/v2, RM-Bench pairwise/pointwise, RubricBench, ProcessBench, PRMBench;
     - ceilings: HateCheck .992, HarmBench-prompt .992;
     - non-English knowledge: TMMLU+, GAOKAO, JMedQA;
     - StrategyQA closed and grounded.
2. **Strong evidence for the limit, not just our estimate:**
   - On the Decision Index 0.2.1, some open entrant beats Jev on 31 of 38 benchmarks, but **no open model up to 36B beats Jev on 7**: GPQA Diamond (.786 vs best .546), MuSR (.661 vs .649), HLE (.204 vs .172), MMLU-Pro (.827 vs .696), BBH (.929 vs .811), WinoGrande (.920 vs .886), API-Bank (.882 vs .860) [V: `data/index.json`, generated 2026-09-28T00:39Z].
   - Qwen3-235B-A22B-Base scores MMLU 87.81 and BBH 88.87; Jev scores .918 and .929 [V: Qwen3 tech report, Table 3].
   - Dinah-0, a 150M ModernBERT-architecture encoder trained on public train splits (the S recipe), reached DI 27.63 against Jev's 57.91. It beat Jev on BANKING77 (.891 vs .797), CLadder (.838 vs .726), HoVer (.822 vs .729) and ChessBench (.243 vs .172). It stayed far behind on ANLI (.279), HellaSwag (.558, trained on its train split), WinoGrande (.628, trained), MMLU-Pro (.133) and BBH (.356) [V: PR #29, `scores.json`].
3. **The S track is where the wins are.** 26 benchmarks are solid wins already at 150m.
   - **Core (18):** AG News, DAIR Emotion, Financial PhraseBank, GoEmotions, PAWS, prompt-injections, AGB-DE, POP909-CL, ChessBench, CLadder, VAST, HoVer, SGD, email spam, NLU++, tweet_topic, fin-topic and daily_dialog.
   - **New in rev 2 (8):** HWU64, LexGLUE EUR-LEX, LEDGAR and UNFAIR-ToS (LexGLUE convention), civil_comments, MASSIVE en-US, GoEmotions single-label and measuring-hate-speech.
     - EUR-LEX is the largest gap on any public set: Jev .391 against BERT-base 71.4 μ-F1.
     - LEDGAR is .753 against 87.6.
   - **Banking77 and Deußer's UNFAIR-ToS moved down.** They are still clear wins against Jev's zero-shot numbers (.797, .499), but against "Jev given train data" (.924 with 24 BM25-retrieved train examples; .748 with dev-tuned thresholds) they are T at 68m–150m, and Banking77 stays T at every size.
   - **400m adds** CLINC150, When2Call, typed-decisions, UNFAIR-ToS (tuned bar), the two HelpSteer2 accuracy rows and Aegis 2.0 response.
   - **1b adds** αNLI, STS-B, HelpSteer2 ρ, ESCI, ACOS and LexGLUE ECtHR-B.
   - 1b costs 2.5–2.7× as much as 400m per token (§7).
4. **The Z track is close to hopeless as a "beat Jev" story.**
   - The only likely Z wins are GoEmotions (Jev macro-F1 .243) and Financial PhraseBank at 400m+ (Jev .730 against .80+ for clean NLI zero-shot models).
   - About 21 core and 12 new rows are toss-ups.
   - Jev is a strong zero-shot model, and no public zero-shot model of 0.5B or less is within 10 points of it on intent, topic or NLI.
   - One curiosity is LexGLUE UNFAIR-ToS under the official "none" convention. There, Jev's zero-shot .764 is **below a constant all-"none" predictor (≈ .89 [E])**, because Jev over-fires at threshold 0.5. A conservative zero-shot model could win it, but that says more about thresholds than about understanding.
5. **The prefill-only Qwen3 scorer (§6) is feasible on CPU but buys little.** A full knowledge pass costs ~$160–1,320 depending on size. The Qwen3 base report puts 4B/8B/14B below Jev on every knowledge row it reports except CoT-generated GSM8K. It may turn **C-Eval** and **BoolQ** (with a LoRA on BoolQ train) into toss-ups. CoT *generation* (not prefill) would beat Jev's GSM8K (Qwen3-14B-Base 92.49 vs .799), but that is a different class of system. Run one small pilot (~$10–20 on validation splits, then ≤$160 for the test passes); skip the full knowledge pass.
6. **Recommended spend (Azure credits, not the $5):**

   | Step | Cost [E] | Note |
   |---|---|---|
   | S-68m fine-tune from v2-68m (0.4–0.8B tokens) | ~$115–285 | |
   | S-400m (1.3B tokens, rev 2 mix) | ~$2.2–3.8k | Main contender |
   | Qwen3 pilot | ≤$160 | |
   | Clean Z retrain, 68m | ~$380–480 | Also the licence-clean leaderboard checkpoint |
   | Evaluation through all harnesses | ~$50–100 | |
   | **Core total** | **≈ $2.7–4.9k** | Rev 1: $2.2–3.9k on 85 rows |
   | Optional: targeted 1b S-spec | ~$1.2–4k | |
   | Optional: 150m Z | ~$0.8–1.2k | |

   The rev 2 S run is ~30% longer (1.3B tokens against 1.0B) because the train mix adds LexGLUE, safety, HWU64/MASSIVE, Yelp-5, civil_comments and others (§7). That is why the S-400m cost rose. Everything fits the ~$9.9k credits (balance as of 2026-10-01, before this week's cluster spend). It is still spend and needs the user's OK.
7. **Flags that change the plan:**
   - **v2 trained on `sms_spam`, which is the whole set Deußer evaluates.** Any v2-derived checkpoint's SMS-spam number is void on both tracks. v2 also trained on `imdb` (`registry.py` `KEPT_DESPITE_EARLIER_LIST`) and on ContractNLI, CLadder, ESCI and Humicroedit (leaderboard notes [N]). That rules out Z claims on these rows without the clean retrain; CLadder and ESCI may also have item overlap.
     - **Rev 2 [N: train-data note §0.4]:** the v2 mix (`mix-v2.0-no-b6`) also trains on the train splits of at least 13 more public-Jev datasets: civil_comments, measuring-hate-speech, Aegis 2.0, LEDGAR, MNLI, FEVER-NLI, WANLI, TREC, CoNLL-2003, SQuAD v2, MTOP and DBpedia, plus tasksource αNLI/CSQA/CaseHOLD/XNLI. The current decontam reference set covers only our own jevbench. Every Z verdict on those rows assumes the clean retrain.
   - **Do not use Deußer's per-item Jev responses, even for evaluation.** The Jev Responses License v1.0 §3.2 prohibits use "to develop, or to facilitate the development of, a product or service that is similar to or competes with TypeSafe's models" [V]. meharsjev is such a product. Compare against the published aggregates and CIs only. The same caution applies to DI's and DMB's per-item Jev rows [N].
   - **No paired test is possible without Jev's item outputs.** Use the unpaired bar in §1.2, which is conservative: e.g., +2.1 pts on Banking77 and +1.7 on SST-2.
   - **Same harness or no claim.** Run our model through the publisher's own code (Deußer's MIT harness, the DI kit, DMB, elcronos), not jevbench. Published Jev numbers move 5–10 pts with the harness [N].
   - **The S track is a specialist against a zero-shot generalist.** It is legitimate only if every headline says so.
   - **Where Jev has a published "given train data" number, the S claim must beat that one too** (rev 2):

     | Benchmark | Jev given train data | Jev zero-shot |
     |---|---|---|
     | Banking77 | .924 | .797 |
     | UNFAIR-ToS (Deußer) | .748 | .499 |
     | GoEmotions | .353 | .243 |
     | ToxicChat | .793 | .786 |
     | LexGLUE mean | .742 | .699 |
     | LexGLUE UNFAIR-ToS | .919 [N] | .764 |

     Otherwise a reader can fairly say "Jev with the same data would win".
   - **New cross-benchmark conflicts (rev 2):**
     - ASEVlad's injection corpus and Gaurav-Gosain's 662-row run both evaluate **all** rows of deepset/prompt-injections, train included. Training on that train split for Deußer's test-116 row voids both. Report those two from the Z checkpoint only.
     - MedHallu's test is built from PubMedQA `pqa_labeled`, which is Deußer's PubMedQA eval set [I].
     - HWU64 and MASSIVE share SLURP-lineage utterances [I]. Dedupe MASSIVE train against the HWU64 test.

---

## 1. Ground rules for beating published numbers

### 1.1 Which harness, per source

| Jev source | Jev numbers | Harness to run ours through | Licence / notes |
|---|---|---|---|
| Deußer et al. 2026 (arXiv 2609.37647) | 37 datasets, full splits, CIs | github.com/AppliedMachineLearning-Lab/jev-benchmarking (MIT; pushed 2026-09-30). Its runner talks to the TypeSafe API. Point it at our local `/v1/systemone` (base-URL override or a small patch, disclosed; **not verified that an override exists**) | Per-item Jev responses (Zenodo 10.5281/zenodo.23039006) are under the Jev Responses License: **do not download or use them** (§0.7) |
| Decision Index 0.2.1 | 38-benchmark panel + non-index rows | github.com/apolinario/decision-index, `--engine http --option base_url=…` [N] | Suite not redistributable; no truncation (over-long = Unsupported = wrong); one rendering; no per-benchmark prompts |
| DMB (nibzard), 2026-09-29 | Banking77, CLINC150, NLU++ | github.com/nibzard/decision-model-benchmark [N] | Bare label names; thresholds frozen on validation |
| elcronos | emotion, tweet_topic, fin-topic, daily_dialog | github.com/elcronos/jev-vs-open-decision-models [N] | Bare label names |
| zhuyansen | TweetEval-emotion, arXiv 2026-09, samples of AG/SST-2/B77/PAWS | github.com/zhuyansen/jev-zeroshot-vs-bert [N] | 20 texts per call; seed-0 samples |
| typed-decisions | 400 cases / 2,000 decisions | Dataset card protocol: whole case per request [N] | S result goes in the card's **fitted** table, not the zero-shot one |

### 1.2 Significance without Jev's item outputs

- **Rule [D]: claim a win only if ours − Jev > bar.**
  - Where the publisher gives a 95% CI (Deußer gives one for all 37): bar = √2 × (CI half-width).
  - Otherwise: bar = 1.96·√(2·p(1−p)/n), with p = Jev's score and n = items.
- This is the unpaired test with equal variances. It is conservative; a paired test would be tighter, but it needs Jev's item outputs, which we will not use.
- Examples:

  | Benchmark | Bar |
  |---|---|
  | Banking77 | +2.1 |
  | CLINC150 | +1.2 |
  | SST-2 (n=872) | +1.7 |
  | prompt-injections (n=116) | +11.6 |
  | language-id | +0.2 (needs ≥.998, at a .996 ceiling) |

- The DI rows use the binomial approximation even for F1/nDCG, so treat their bars as ≈.
- The column "Bar for a significant win" in §3 is the number to beat.

### 1.3 When Jev has several published numbers

- Target the **highest full-split number obtained in the harness we replicate**. Report the others for context.
- Small-n pilots (AbdelStark n=100, Janus n=500, the zhuyansen samples) are listed in group D and are not counted.
- Example: Banking77 is .797 acc (Deußer, 3,076), .797 macro-F1 (DI, 3,080), .792 (DMB, 3,080) and .870 (AbdelStark, 72 labels, n=100). The targets are .797 in the Deußer harness and .797 macro-F1 in the DI harness.

### 1.4 Track rules

**Z track:**
- Needs a **clean retrain**. The current registry excludes only the 25 jevbench test sets, the dev sets and a "reserved" list.
- Extend the exclusions to every benchmark in §3, plus siblings. Cases already known to be in v2:
  - `imdb` and `sms_spam` (`KEPT_DESPITE_EARLIER_LIST`);
  - ContractNLI, CLadder, Amazon ESCI and Humicroedit [N];
  - probably some of PubMedQA, ARC, CSQA, αNLI and SciQ inside tasksource-jev [E; scan].
- Run the exact / 13-gram / MinHash scan (`jev_local.data.v2.decontam`) against every evaluated split.

**S track:**
- Allowed:
  - official train splits;
  - other public non-test data, e.g. Lichess puzzles for ChessBench, the CLadder generator with new seeds, POP909 songs not in DI's sampled set, other spam/moderation corpora;
  - validation splits for model selection.
- Never:
  - the evaluated split, including k-fold "cross-fitting" on sets that have no train split (SMS spam, OpenAI moderation, SummEval, PubMedQA labeled, possibly FinEntity). Cross-fitting trains on test items. Those rows therefore fall back to Z-plus-related-data.
- **Cross-benchmark contamination is the main S risk.** One benchmark's train split can hold another's test items:
  - SST train vs Rotten Tomatoes test: same Pang & Lee sentences.
  - All Financial PhraseBank configs vs the atrost test.
  - PAWS-X vs PAWS.
  - SuperGLUE BoolQ vs BoolQ.
  - MMLU `auxiliary_train` vs ARC/OBQA (ARC train itself is fine).
  - Rev 2, from `train-data-and-supervised-ceilings.md` §0.2 and §7 [N]:
    - `dair-ai/emotion` config `unsplit` holds the test texts.
    - `nickmuchi/financial-classification` contains Financial PhraseBank.
    - `KoalaAI/Text-Moderation` was trained on the OpenAI moderation eval.
    - `tasksource/bigbench` train contains BBH test items.
    - MMLU-Pro contains filtered MMLU-test items.
    - When2Call test items were built from BFCL.
  - Rev 2, new rows:
    - deepset prompt-injections train vs the all-662 / combined-corpus evaluations;
    - MedHallu vs PubMedQA `pqa_labeled`;
    - MASSIVE/SLURP train vs HWU64 test;
    - LEDGAR is in Dinah-0's mix (irrelevant to us, but a reminder that DI-adjacent mixes leak).

  Dedupe the union of S training data against the union of **all** evaluated splits in §3, not just the row being trained for.
- **Specialists.** One multi-task S model per size is the headline. Per-benchmark specialist fine-tunes from it ("S-spec") are allowed, but they are disclosed and reported separately.
- **Selection.** Hyperparameters and checkpoints are chosen on validation splits only, then each harness is run once.

---

## 2. Size priors: what each Ettin size can reach

### 2.1 Supervised (S) prior: Ettin against known fine-tuned encoders [V: Ettin paper, arXiv 2507.11412, Table 7]

| Model | Layers × hidden | GLUE avg | MNLI | SST-2 | Literature analogue for transfer of published fine-tune results |
|---|---|---|---|---|---|
| Ettin-68m | 19 × 512 | 87.2 | 87.0 [N] | – | RoBERTa-base / BERT-base class |
| Ettin-150m | 22 × 768 | 88.9 | 89.2 | 95.8 | ModernBERT-base (GLUE 88.4) |
| Ettin-400m | 28 × 1024 | 90.8 | 91.3 | 96.7 | ModernBERT-large (90.4), RoBERTa-large; just under DeBERTa-v3-large (GLUE 91.37 [R]) |
| Ettin-1b | 28 × 1792 | 91.6 | 91.8 | 97.1 | DeBERTa-v2-xlarge/xxlarge class |

So published RoBERTa-large / DeBERTa-v3-large / ModernBERT-large fine-tune results are the 400m prior, and the best published encoder results (≤1.5B) are the 1b ceiling.

Key published S anchors used in §3:

| Task | Anchor | Jev |
|---|---|---|
| BoolQ | DeBERTa-v3-large .8835 [V: nfliu/deberta-v3-large_boolq] | .913 |
| ANLI, 3-class, all rounds | DeBERTa-v3-large trained on MNLI+FEVER+ANLI+Ling+WANLI .702 [V: Laurer card] | .739 |
| SST-5 | RoBERTa-large .602 [V-secondary: arXiv 2005.13619] | .579 |
| WinoGrande | RoBERTa-large 79.1 [V] | .914 |
| αNLI | RoBERTa-large .856; L2R2 .885 [V-secondary] | .839 |
| RAGTruth (response F1) | TinyLettuce-68M (Ettin) .750, LettuceDetect-base (ModernBERT 150M) .761, LettuceDetect-large (395M) .792, Llama-2-13B FT .787 [V] | .765 |
| ToxicChat | ToxicChat-T5-large (738M) F1 .822 [V] | .786 |
| Humicroedit | best SemEval-2020 T7 system .6743 [V] | .619 |
| NLI4CT | best SemEval-2024 F1 .80 (Mixtral systems) [V] | .841 |
| VAST | TGA-Net .665 [V]; DeBERTa-v3-base .755 [V-secondary] | .646 |
| SIB-200 | fully supervised per-language XLM-R-large avg .759 [V] | .815 |
| UNFAIR-ToS (LexGLUE μ-F1 / m-F1) | BERT 95.6/81.3, DeBERTa 95.5/80.3, Legal-BERT 96.0/83.0 [V] | .499 |

Notes on the anchors:
- **UNFAIR-ToS:** LexGLUE adds a 'none' label at evaluation, so its μ-F1 is not comparable to Jev's 8-label micro-F1 of .499. Jev's own ranking is excellent (mean AUROC .994 [V]); its 0.5 threshold is the problem.
- **CLINC150:** BERT oos-train in Larson et al. 2019 gets 96.3–96.9% in-scope accuracy and 40.3–59.2% OOS recall. That implies **.87–.90 overall accuracy** on the 5,500-item plus test [V-search summary; computed]. Jev scores .895.
- **The Laurer `ModernBERT-large-zeroshot-v2.0` ANLI numbers (.812/.717/.716) are binary entailment/not-entailment accuracy** [V: card, MNLI-m .942 confirms binary]. They are not comparable to Jev's 3-class .739, and should not be cited as an ANLI baseline the way the jevbench ANLI row in `classifier-benchmarks.md` does.

### 2.2 Zero-shot (Z) prior: universal classifiers [N unless marked]

- **BTZSC macro-F1:**
  - deberta-v3-large-nli (434M): AG .815, B77 .346, FPB .803, emotion .440.
  - gte-modernbert-base (149M): B77 .637.
  - Qwen3-Reranker-8B: overall .722, AG .788, B77 .691.
- **Laurer `-c` (clean):** deberta-v3-large-zeroshot-v2.0-c mean .676: AG .819, B77 .513, emotion .499.
- **GLiClass:** gliclass-edge-v3.0 (32.7M) averages .490; gliclass-large-v3.0 reaches .900 on FPB (leakage possible).
- **Laya:** emotion .587, toxic-chat .530.
- **DI best ≤0.5B:** GLiNER2.5-Decide B77 .656 and CLINC .604. Every ≤0.5B DI entrant has index ≤11.21.
- **Our own v2-68m dev** (uncalibrated): SNIPS .746, HWU64 .496, 20NG .302, tweet_topic .453, MRPC .718, SciTail .593, ToxiGen .623, prompt-injections .491 [V: `runs/v2/scores_dev.json`].

Reading: at 68m we are in the GLiClass-edge / GLiNER-small class. At 400m–1b we are, at best, at the deberta-large `-c` / Reranker-0.6B level, which is still 6–25 points under Jev on topic, intent and NLI.

### 2.3 The one in-format S precedent: Dinah-0 (150M), per-benchmark against Jev [V: DI PR #29; `scores.json` rev f1c4c6f; lukita.me write-up]

**Training:** ~4.6M examples, ~1.19B tokens. Base: moBERTo (ModernBERT architecture). Data: train splits incl. LEDGAR, News Category, HellaSwag, WinoGrande, CLINC and Lichess, plus typed decisions.

| DI benchmark | Dinah-0 | Jev | | DI benchmark | Dinah-0 | Jev |
|---|---|---|---|---|---|---|
| BANKING77 | **.891** | .797 | | ANLI | .279 | .748 |
| CLINC150 | .888 | .893 | | HellaSwag (trained) | .558 | .945 |
| CLadder | **.838** | .726 | | WinoGrande (trained) | .628 | .920 |
| HoVer | **.822** | .729 | | MMLU-Pro | .133 | .827 |
| ChessBench | **.243** | .172 | | BBH | .356 | .929 |
| GSM8K (rank shortcut; disallowed) | .840 | .799 | | RAGTruth | .517 | .765 |
| When2Call | .784 | .810 | | NLI4CT | .510 | .841 |
| VAST | .626 | .646 | | iSarcasmEval | .000 | .505 |
| ContractNLI | .686 | .717 | | FinEntity | .708 | .870 |

**Index:** 27.63 (25.72 without the GSM8K shortcut). Median latency 5.6 ms on an RTX PRO 4000.

---

## 3. Verdict table: every public benchmark with a published Jev number

Columns:
- "Bar" = Jev + the §1.2 margin.
- S/Z codes are for Ettin 68m / 150m / 400m / 1b.
- DI "Jev" values are native raw scores from `data/index.json` (generated 2026-09-28T00:39:36Z) [V].
- Deußer values and CIs come from `results/eval/summary.md` in the MIT repo [V].
- Other sources are from the project notes [N].

**Correction to `classifier-benchmarks.md`:** Deußer's LLM-AggreFact .786 is the **mean balanced accuracy over source datasets** (the leaderboard convention), not pooled. Pooled balanced accuracy is .831 [V: `llm_aggrefact.json`]. So .786 is directly comparable to the LLM-AggreFact leaderboard, where the best entry is Bespoke-MiniCheck-7B at .774 [V].

#### A. Deußer et al. 2026 (37 datasets, full splits)

| Benchmark | Source | Split, n | Metric | Jev | Bar for a significant win | S 68/150/400/1b | Z 68/150/400/1b | Key evidence |
|---|---|---|---|---|---|---|---|---|
| AG News | Deußer | test 7,600 | acc | .885 | .894 (+0.9) | W W W W | L L L T | S: sup. SOTA ~.955 [R]; Z: deberta-v3-L-zs-c .819 F1, Qwen3-Rr-8B .788 F1 (BTZSC) |
| IMDB | Deußer | test 25,000 | acc | .965 | .968 (+0.3) | L L T T | L L L L | S: RoBERTa-L ~.963, XLNet-L .962 [R]; Z: v2 mix contains imdb (registry KEPT list) |
| Rotten Tomatoes | Deußer | test 1,066 | acc | .933 | .954 (+2.1) | L L T T | L L L L | S: sentence-level fine-tunes ~.89-.92 [R]; SST train shares RT sentences (dedupe) |
| SST-2 | Deußer | validation 872 | acc | .964 | .981 (+1.7) | L L T T | L L L L | S: Ettin GLUE SST-2 150m .958 / 400m .967 / 1b .971 [V]; Z: gliclass-modern-L .933 |
| DAIR Emotion | Deußer | test 2,000 | acc | .585 | .615 (+3.0) | W W W W | T T T T | S: fine-tuned BERT ~.93 [R]; Z: Laya .587, deberta-L-c .499 F1 |
| Financial PhraseBank | Deußer | atrost test 970 | acc | .730 | .770 (+4.0) | W W W W | T T W W | S: FinBERT ~.86 (50agree) [R]; Z: deberta-v3-L-nli .803 F1, Qwen3-Rr-8B .817 (BTZSC) |
| SMS Spam | Deußer | all 5,574 (no split) | F1 | .938 | .956 (+1.8) | T T T T | T T T T | No train split; v2 mix contains sms_spam -> remove; S = transfer from other spam corpora |
| papluca language-id (20) | Deußer | test 10,000 | acc | .996 | .998 (+0.2) | L L T T | L L L L | S: XLM-R-base fine-tune 99.6% [R]; significance needs >= .998; Ettin is English-only |
| GoEmotions (28 nouls) | Deußer | test 5,427 | macro-F1 | .243 (S bar: .353 tuned) | .252 (+0.9); S ≈.365 | W W W W | T W W W | S: BERT-base .46 (Demszky 2020) [R]; Z: no published zero-shot; low bar |
| Banking77 | Deußer / DI / DMB | test 3,076 | acc (.788 F1) | .797 (S bar: **.924** given 24 BM25 train examples, simonmesmith, n=3,080 [N]) | Z .818 (+2.1); **S .937 (+1.3)** | **T T T T** (W W W W vs zero-shot .797) | L L L L | S: BERT-base FT .9366, SPACE-2 .948 [N]; Dinah-0 150M .891 F1 (DI) [V]; Z: GLiNER2.5-Decide .656, Qwen3-Rr-8B .691 |
| CLINC150 plus (151) | Deußer / DI / DMB | test 5,500 | acc | .895 | .907 (+1.2) | T T W W | L L L L | S: BERT oos-train .963 in-scope + .592 OOS recall -> .896 [V,computed]; Dinah-0 .888 F1 [V] |
| SIB-200 (205 langs) | Deußer | test 41,820 | acc | .815 | .820 (+0.5) | L L L L | L L L L | S: per-language XLM-R-large avg .759 [V]; Z: GPT-4 .487 [V]; Ettin English-only |
| ANLI r1-r3 | Deußer / DI | test 3,200 | acc (.748 F1 DI) | .739 | .760 (+2.1) | L L L T | L L L L | S: DeBERTa-v3-L +ANLI .702 (3-class) [V]; Z: Dinah-0 .279, Laya .487 F1 |
| AfriXNLI (18 langs) | Deußer | test 10,800 | acc | .640 | .653 (+1.3) | L L L L | L L L L | No train split (val+test only); English-only backbone |
| PAWS | Deußer | test 8,000 | acc | .850 | .861 (+1.1) | W W W W | L L L L | S: BERT on PAWS-Wiki ~.90 [R]; Z: deberta-c .766 |
| LLM-AggreFact (11 sets) | Deußer | test 29,320 | mean BAcc | .786 | .800 (+1.4) | L L L L | L L L L | Jev > every leaderboard row (Bespoke-MiniCheck-7B .774; best <=1B FactCG-DeBERTa-L .756) [V]; card forbids training |
| BoolQ | Deußer | validation 3,270 | acc | .913 | .926 (+1.3) | L L L L | L L L L | S: DeBERTa-v3-L .8835 [V]; DeBERTa-1.5B .904 [R]; LLM+LoRA path T |
| Belebele (122 langs) | Deußer | 109,800 | acc | .867 | .870 (+0.3) | L L L L | L L L L | No train split; English-only backbone |
| PubMedQA pqa_labeled | Deußer | all 1,000 | acc | .787 | .822 (+3.5) | L L L L | L L L L | Eval = whole labeled set; BioLinkBERT-L .722 [R]; LLM-14B T(low) |
| MMLU | Deußer / DI | test 14,042 | acc | .918 | .925 (+0.7) | L L L L | L L L L | Qwen3-14B-Base .811, Qwen3-235B-A22B-Base .878 [V] |
| C-Eval (Chinese) | Deußer | test 12,342 | acc | .839 | .849 (+1.0) | L L L L | L L L L | Ettin English-only; LLM path: Qwen2-7B-Base 83.2 [V] -> Qwen3-8B/14B T |
| BIG-bench MC (93 tasks) | Deußer | 13,227 | acc | .814 | .823 (+0.9) | L L L L | L L L L | Knowledge/reasoning |
| HellaSwag | Deußer / DI (.945) | validation 10,042 | acc | .955 | .961 (+0.6) | L L L L | L L L L | S: DeBERTa-v3-L ~.88 [secondary], Dinah-0 (trained) .558 [V]; Qwen2-72B-Base .876 [V] |
| WinoGrande | Deußer / DI (.920) | validation 1,267 | acc | .914 | .936 (+2.2) | L L L L | L L L L | S: RoBERTa-L .791 [V]; Dinah-0 (trained) .628 [V]; no DI open model beats Jev |
| ARC E+C | Deußer | test 3,548 | acc | .988 | .993 (+0.5) | L L L L | L L L L | Ceiling |
| CommonsenseQA | Deußer / DI (.875) | validation 1,221 | acc | .882 | .907 (+2.5) | L L L L | L L L L | S: DeBERTa-v3-L .841 dev [secondary] |
| alphaNLI (ART) | Deußer | validation 1,532 | acc | .839 | .867 (+2.8) | L L T W | L L L L | S: RoBERTa-L .856, L2R2 RoBERTa-L .885 [secondary] |
| ToxiGen annotated | Deußer | test 940 | acc | .878 | .907 (+2.9) | T T T T | L L L T | S: ToxiGen train exists; no published acc on this split found; Z: v2-68m dev .623 |
| OpenAI moderation (8 nouls) | Deußer | 1,680 (no split) | mean AUPRC | .717 | .776 (+5.9) | T T T T | T T T T | No train split: S = Z + other moderation corpora |
| ToxicChat 0124 | Deußer | test 5,083 | F1 toxic | .786 (S bar: .793, dev-tuned threshold) | .834 (+4.8); S ≈.841 | L T T T | L L L L | S: ToxicChat-T5-large (738M) F1 .822 [V] |
| deepset prompt-injections | Deußer | test 116 | acc | .741 | .857 (+11.6) | W W W W | L L T T | S: deberta-v3-base-injection ~.99 [R]; v2-68m dev .491 |
| AGB-DE (German) | Deußer | test 755 | F1 | .204 | .317 (+11.3) | T W W W | L L T T | Jev F1 very low; English-only tokenizer is the risk |
| UNFAIR-ToS (8 nouls, positives only) | Deußer | test 1,607 | micro-F1 | .499 (S bar: **.748** with dev-tuned thresholds) | Z .566 (+6.7); **S ≈.79–.815** | **T T W W** (W W W W vs zero-shot .499) | T T T T | LexGLUE μ-F1 95–96 includes a 'none' label (not comparable) [V]. Back-solving BERT's 95.6 at a ~10% positive rate gives ≈.81 positives-only [E, derived] |
| STS-B | Deußer | test 1,379 | Spearman | .890 | .906 (+1.6) | L T T W | L L L L | S: stsb-roberta-large cross-encoder ~.915 test [R] |
| SST-5 | Deußer | test 2,210 | argmax acc | .579 | .607 (+2.8) | L L T T | L L L L | S: RoBERTa-L .602 [V-secondary] |
| SummEval (4 scores) | Deußer | 1,600 (no split) | mean group rho | .554 | .588 (+3.4) | L L L L | L L L L | No train split; G-Eval-4 .514, UniEval .474 [R] |
| HelpSteer2 (5 scores) | Deußer | validation 1,038 | mean rho | .412 | .457 (+4.5) | T T T W | L T T T | Train 20k; needs >2k-token context |

#### B. Decision Index 0.2.1 (33 index benchmarks not in A, plus 10 non-index rows)

| Benchmark | Source | Split, n | Metric | Jev | Bar for a significant win | S 68/150/400/1b | Z 68/150/400/1b | Key evidence |
|---|---|---|---|---|---|---|---|---|
| BFCL | DI 0.2.1 | 1,694 | case-exact | .958 | .971 (+1.4) | L L L L | L L L L | Best <=0.5B .462; Dinah-0 .610 [V] |
| ToolRet | DI 0.2.1 | 685 | nDCG@10 | .653 | .703 (+5.0) | L L T T | L L L L | Bosun-0.6B .574; ToolRet-train exists [R] |
| API-Bank | DI 0.2.1 | 508 | acc | .882 | .922 (+4.0) | L L L L | L L L L | No DI open model beats Jev (best .860) |
| Home appliance sim. | DI 0.2.1 | 88 | case-exact | .523 | .670 (+14.8) | L L L L | L L L L | Small models .00-.02 |
| When2Call | DI 0.2.1 | 3,652 | acc | .810 | .828 (+1.8) | L T W W | L L L L | When2Call train exists; Dinah-0 .784 [V] |
| ContractNLI | DI 0.2.1 | 123 docs | cons. macro-F1 | .717 | .829 (+11.3) | L T T T | L L L L | v2 mix contains train; Dinah-0 .686; long NDAs > 8k tokens |
| BPoMP (limericks) | DI 0.2.1 | 811 | acc | .909 | .937 (+2.8) | L L L T | L L L L | No train; synthetic rhyme/meter breaks possible |
| Humicroedit | DI 0.2.1 | 2,628 | acc | .619 | .645 (+2.6) | L T T T | L L L L | SemEval-20 T7 best .6743 [V]; v2 mix contains train |
| POP909-CL (129 chords) | DI 0.2.1 | 2,000 | acc | .166 | .189 (+2.3) | W W W W | L L L L | Best open .661 (26B) |
| cfcolor | DI 0.2.1 | 5,000 | acc | .644 | .663 (+1.9) | T T T T | L L L L | Best open .650 |
| GPQA Diamond | DI 0.2.1 | 198 | acc | .786 | .867 (+8.1) | L L L L | L L L L | Qwen3-14B-Base .399 [V]; no open model beats Jev |
| GSM8K (MC) | DI 0.2.1 | 1,319 | acc | .799 | .829 (+3.1) | L L L L | L L L L | Rank shortcut (issue #32) forbidden; LLM CoT generation W (other system class) |
| ChessBench | DI 0.2.1 | 5,000 | acc | .172 | .187 (+1.5) | T W W W | L L L L | Dinah-0 (Lichess-trained, 150M) .243 [V] |
| MuSR | DI 0.2.1 | 752 | acc | .661 | .709 (+4.8) | L L L L | L L L L | No open model beats Jev |
| SATA-Bench | DI 0.2.1 | 1,650 | case-exact | .264 | .294 (+3.0) | L L L L | L L L L | Best <=0.5B .042 |
| CRUXEval | DI 0.2.1 | 570 | acc | .730 | .781 (+5.2) | L L L L | L L L L | Code execution |
| CLadder | DI 0.2.1 | 5,000 | acc | .726 | .744 (+1.7) | T W W W | L L L L | Dinah-0 .838 [V]; v2 mix contains CLadder; use generator, new seeds |
| HLE | DI 0.2.1 | 501 | acc | .204 | .253 (+5.0) | L L L L | L L L L | Chance .164; best open .172 |
| MMLU-Pro | DI 0.2.1 | 12,032 | acc | .827 | .837 (+1.0) | L L L L | L L L L | Qwen3-14B-Base .610 [V] |
| BBH | DI 0.2.1 | 5,507 | acc | .929 | .939 (+1.0) | L L L L | L L L L | Qwen3-14B-Base .811 (CoT) [V] |
| BRIGHT | DI 0.2.1 | 220 | nDCG@10 | .475 | .569 (+9.3) | L L L L | L L L L | No train split; best <=0.5B .201 |
| Amazon ESCI | DI 0.2.1 | 5,000 | macro-F1 | .552 | .572 (+1.9) | T T T W | L L L L | Train 1.4M+ pairs; v2 mix contains ESCI |
| ACOS | DI 0.2.1 | 400 | per-review F1 | .295 | .358 (+6.3) | T T T W | L L L L | ACOS train exists [R]; best open .356 |
| FinEntity | DI 0.2.1 | 979 | macro-F1 | .870 | .900 (+3.0) | L L L T (rev 1: L L T T) | L L L L | DI scores all 979 documents, so there is no train split [N: train-data note §0.1 D]. S = related financial-sentiment data only; GLiNER-base .701 |
| iSarcasmEval | DI 0.2.1 | 4,600 | F1 sarcastic | .505 | .526 (+2.0) | L L T T | L L T T | Lavoir (ModernBERT-L, zero-shot) .463; best open .660 |
| VAST | DI 0.2.1 | 3,006 | macro-F1 | .646 | .670 (+2.4) | T W W W | L L L L | TGA-Net .665 [V]; DeBERTa-v3-base .755 [secondary] |
| NLI4CT | DI 0.2.1 | 5,500 | macro-F1 | .841 | .854 (+1.4) | L L L L | L L L L | SemEval-24 best F1 .80 (Mixtral) [V] |
| RAGTruth | DI 0.2.1 | 2,700 | F1 halluc. | .765 | .788 (+2.3) | T T T T | L L L L | TinyLettuce-68M (Ettin) .750, LettuceDetect base .761 / large .792 [V] |
| HoVer | DI 0.2.1 | 4,000 | acc | .729 | .748 (+1.9) | T W W W | L L L L | Dinah-0 .822 [V] |
| PhishNChips | DI 0.2.1 | 2,000 | acc | .625 | .655 (+3.0) | T T T T | L T T T | Best open .875; system-one-gemma (268M) .600 |
| ForecastBench | DI 0.2.1 | 10,139 | Brier skill | .306 | .318 (+1.3) | L L L L | L L L L | Needs 2026 world knowledge |
| Habermas Machine | DI 0.2.1 | 1,676 | acc | .459 | .493 (+3.4) | T T T T | L T T T | GLiNER2.5-Decide .455 zero-shot |
| New Yorker | DI 0.2.1 | 528 | acc | .701 | .756 (+5.5) | L L L L | L L L L | Best <=0.5B .330 |
| RouterBench (non-index) | DI 0.2.1 | 5,001 | sel. quality | .799 | .815 (+1.6) | T T T T | L L L L | Best open .801; GLiNER-base .672 |
| SGD/SGD-X (non-index) | DI 0.2.1 | 2,500 | macro-F1 | .429 | .457 (+2.7) | W W W W | T T T T | SGD train exists [R] |
| ARC-Easy (non-index) | DI 0.2.1 | 2,376 | acc | .993 | .998 (+0.5) | L L L L | L L L L | Ceiling |
| ARC-Challenge (non-index) | DI 0.2.1 | 1,172 | acc | .978 | .990 (+1.2) | L L L L | L L L L | Ceiling |
| OpenBookQA (non-index) | DI 0.2.1 | 500 | acc | .940 | .969 (+2.9) | L L L L | L L L L | Knowledge |
| Support-ticket calibration (non-index) | DI 0.2.1 | 300 | acc | .905 | .952 (+4.7) | L L T T | L L T T | Synthetic; no train |
| Phishing difficulty gradient (non-index) | DI 0.2.1 | 800 | acc | .946 | .968 (+2.2) | T T T T | L L T T |  |
| Email spam (non-index) | DI 0.2.1 | 3,000 | acc | .974 | .982 (+0.8) | T W W W | T T T T | Arize: TF-IDF LR .984 on 18.5k emails |

#### C. Other public studies, large splits

| Benchmark | Source | Split, n | Metric | Jev | Bar for a significant win | S 68/150/400/1b | Z 68/150/400/1b | Key evidence |
|---|---|---|---|---|---|---|---|---|
| NLU++ (folds 18-19) | DMB 2026-09-29 | 302 msgs / 13,712 dec. | micro-F1 | .483 | .563 (+8.0) | W W W W | T T T T | Train folds 0-17 (currently our dev set) |
| tweet_topic_single | elcronos | test_2021 1,693 | acc | .793 | .820 (+2.7) | T W W W | L L T T | Our dev set; v2-68m dev .453 |
| Twitter Fin. News Topic (20) | elcronos | validation 4,117 | acc | .670 | .690 (+2.0) | W W W W | L T T T | Train 17k [R]; PrismNLI .352 |
| daily_dialog emotion (7) | elcronos | test 7,740 utt. | macro-F1 | .385 | .400 (+1.5) | W W W W | T T T T | Majority acc .817 |
| TweetEval emotion (4) | zhuyansen | test sample 1,000 | acc | .827 | .860 (+3.3) | L T T T | L L L L | deberta-c .760, bart .749 |
| arXiv 2026-09 (8 cats) | zhuyansen | 258 | acc | .891 | .945 (+5.4) | T T T T | L L L L | deberta-c .589; post-release set |
| typed-decisions | LocalLLaMA card | test 400 / 2,000 dec. | acc (soft gold) | .727 | .755 (+2.8) | T T W W | L L L T | Fitted table: OpenDecider-nano (Ettin-400m) .796, Laya-td .766 [notes] |

#### D. Small-n, composite and typed suites (not in the tallies)

| Benchmark | Source | Split, n | Metric | Jev | Bar for a significant win | S 68/150/400/1b | Z 68/150/400/1b | Key evidence |
|---|---|---|---|---|---|---|---|---|
| Jevals PubMedQA (yes/no) | Jevals | 300 x5 | acc | .913 | .958 (+4.5) | L L L L | L L L L | Items drawn from PubMedQA train split |
| Jevals HelpSteer2 helpfulness | Jevals | 300 x5 | acc (prior .417) | .413 | .492 (+7.9) | T T T T | L L T T |  |
| BTZSC pilot emotion | AbdelStark | 100 | acc | .480 | .618 (+13.8) | W W W W | T T T T | n=100 |
| BTZSC pilot Banking77 (72) | AbdelStark | 100 | acc | .870 | .963 (+9.3) | T T T T | L L L L | n=100 |
| BTZSC pilot AG News | AbdelStark | 100 | acc | .910 | .989 (+7.9) | T T T T | L L L L | n=100 |
| SST-2/AG/Emotion/B77 macro | HF blog dylantom | 4 x 2,500 | acc | .793 | .804 (+1.1) | W W W W | L L T T |  |
| CLINC in-scope top-1 | jev-certify | 400 | acc | .930 | .965 (+3.5) | T W W W | L L L L |  |
| Banking77 (described) | Janus | 500 | acc | .778 | .830 (+5.2) | W W W W | L L L L |  |
| Enron spam | onlyoneaman | sample | acc | .987 | .997 (+1.0) | T T T T | L L T T |  |
| Email spam 18,514 | Arize | 18,514 | acc | .983 | .986 (+0.3) | T W W W | T T T T | TF-IDF LR .984 |
| HellaSwag (Vercel harness) | scienthoon | 2,000 | acc | .861 | .882 (+2.1) | L L L T | L L L L | Different harness from Deußer (.955) |
| 'General decisions' (B77/BoolQ/Yelp/ChaosNLI) | OpenDecider card | 200 | acc | .730 | .817 (+8.7) | L T T T | L L L L |  |
| JevBench v1.5.5 | benchmarkheaven | 1,624 dec. | Intelligence | 72.0 | — | L L L L | L L L L | No train; sealed half |
| jabr classifier-benchmark v2 | jabr | 866 | micro acc | .967 | .984 (+1.7) | L L L L | L L L L | README: do not train on it |
| DecisionBench | Hanno Labs | 23,900 | acc | .720 | .728 (+0.8) | L L L L | L L L L | Best encoder .514 |
| DecideBench v1.1 | choyiny | 400 | acc | .980 | .999 (+1.9) | L L L L | L L L L |  |

#### E-A. Rev 2: tier-A rows from the `jev-published.md` sweep that were missing above (33 counted)

All Jev values come from `bench/public/jev_published.json` [N], whose sources were read 2026-10-02. LexGLUE comparators were read today from the LexGLUE README leaderboard [V]; it uses the official 'none' convention, the same as chepyle's Jev run. Skywork-Reward-V2 numbers were read today from its model card [V]. For F1 on a rare positive class, the binomial bar understates the noise, so treat those bars as ≈.

| Benchmark | Source (date) | Split, n | Metric | Jev | Bar for a significant win | S 68/150/400/1b | Z 68/150/400/1b | Key evidence |
|---|---|---|---|---|---|---|---|---|
| HWU64 (64 intents) | thisisandreeeee (2026-09-22) | test 1,076 | acc | .831 | .863 (+3.2) | W W W W | L L L L | S: SPACE-2 .942, BERT-base .916 [N]. **HWU64 is our v2 dev set**, so training on its train split spends that dev signal. Z: v2-68m dev .496 [V] |
| LexGLUE ECtHR A | chepyle (2026-09-21) | test 1,000 | μ-F1 | .730 | .769 (+3.9) | L L T T | L L L L | RoBERTa-L 73.8, BERT 71.2, Longformer 69.9 [V]. Fact sections run long; Ettin's 8k context ≈ LexGLUE's 64×128 hierarchical input |
| LexGLUE ECtHR B | chepyle | test 1,000 | μ-F1 | .754 | .792 (+3.8) | T T T W | L L L L | Legal-BERT 80.4, RoBERTa-L 79.8, BERT 79.7 [V] |
| LexGLUE SCOTUS (13 areas) | chepyle | test 1,400 | μ-F1 | .726 | .759 (+3.3) | L L T T | L L L L | CaseLaw-BERT 76.6, Legal-BERT 76.4, RoBERTa-L 75.5, BERT 68.3 [V]. Opinions are long: Jev's run truncated 613 of 1,400 at 48k chars |
| LexGLUE EUR-LEX (100 EuroVoc) | chepyle | test 5,000 | μ-F1 | .391 | .410 (+1.9) | W W W W | L L L T | DeBERTa / Legal-BERT 72.1, BERT 71.4, TF-IDF+SVM 63.4 [V]; train 55k |
| LexGLUE LEDGAR (100 provisions) | chepyle | test 10,000 | μ-F1 | .753 | .765 (+1.2) | W W W W | L L L T | RoBERTa-L 88.6, BERT 87.6, TF-IDF+SVM 87.0 [V]; train 60k |
| LexGLUE UNFAIR-ToS ('none' convention) | chepyle | test 1,607 | μ-F1 | .764 zero-shot; **.919** with per-label tuning [N] | S .938 (+1.9 over tuned); Z .793 | W W W W | T T T T | Legal-BERT 96.0, BERT 95.6, TF-IDF+SVM 94.7 [V]. A constant all-'none' predictor scores ≈ .89 [E, at a ~10% positive rate] |
| LexGLUE CaseHOLD (5-way) | chepyle | test 3,600 | μ-F1 (= acc) | .773 | .792 (+1.9) | L L L T | L L L L | CaseLaw-BERT 75.4, Legal-BERT 75.3, RoBERTa-L 74.4, DeBERTa 72.6 [V]. Legal knowledge: Jev beats every LexGLUE model |
| *LexGLUE 7-task mean (shown, not counted)* | chepyle | 23,607 | mean μ-F1 | .699; S bar **.742** (validation-tuned) | ≈ .755 [E] | W W W W | L L L L | RoBERTa-L 79.4, Legal-BERT 79.8, BERT 77.8 [V] |
| MedHallu | stperic (2026-09-23) | test 1,000 | acc | .929 (threshold .65 picked on 1,000 dev items, so already S-style) | .952 (+2.3) | L L T T | L L L L | GPT-5.6 Luna .951 and Jev with the MedHELM prompt .903 [N]. ~9k train items derived from pqa_artificial [R]. Test is built from PubMedQA pqa_labeled [I] |
| Aegis 1.0 prompt | mbburabak (2026-09-21) | test 359 | F1 harmful | .891 | ≈ .937 (+4.6) | L L T T | L L L L | Qwen3Guard-0.6B .908 [N]; Aegis 1.0 train ≈10.8k [R]; small n |
| Aegis 2.0 prompt | mbburabak | test 1,928 | F1 harmful | .836 | ≈ .859 (+2.3) | L T T T | L L L L | Shieldstral-3B .862, Qwen3Guard-0.6B .850 [N]; Aegis 2.0 train ≈30k [R] |
| Aegis 2.0 response | mbburabak | test 1,928 | F1 harmful | .802 | ≈ .827 (+2.5) | T T W W | L L L L | Shieldstral .872 [N]; Qwen3Guard-0.6B .842 [N: train-data note §0.4]. The v2 mix already trains on Aegis 2.0 [N], so there is no Z claim from v2 |
| WildGuardTest prompt | mbburabak | test 1,699 | F1 harmful | .884 | ≈ .906 (+2.2) | L L L T | L L L L | WildGuard-7B .889 [R], Shieldstral .881 [N]: 3–7B specialists sit under the bar. WildGuardMix train (87k) is gated under the AI2 licence |
| HateCheck | mbburabak | 3,728 | F1 hateful | .992 | .996 | L L L L | L L L L | Ceiling. Functional test suite with no train split |
| HarmBench response | mbburabak | 596 | F1 harmful | .875 | ≈ .912 (+3.8) | L L L T | L L L L | Shieldstral .870 [N]. No official train split, so S = related corpora (Aegis 2, WildGuardMix, BeaverTails) |
| HarmBench prompt | mbburabak | 239 | detection rate | .992 | ceiling | L L L L | L L L L | Shieldstral .994 [N] |
| Combined injection corpus (4 HF sets, deduped) | ASEVlad (2026-09-20) | all 11,900, **train rows included** | AUPRC | .980 | ≈ .984 | L L T T | L L T T | No clean train split, so S = Z + other corpora. ROC-AUC .990 [N]; v2-68m prompt-injection dev acc .491 [V] |
| RewardBench v1 | goya4140 (2026-09-20) | 2,985 | official 4-section macro | .926 | .939 (+1.3) | L L L L | L L L L | Skywork-Reward-V2: Qwen3-0.6B 85.2, Llama-3.2-1B 89.9, Qwen3-4B 93.4, Qwen3-8B 93.7, Llama-3.1-8B 96.4 [V]. Even ≤1B reward models trained on curated preference data stay below the bar |
| RewardBench 2 | goya4140 | 1,865 | 6-domain macro | .812 | .837 (+2.5) | L L L L | L L L L | Skywork-V2: Qwen3-8B 78.2, Llama-3.1-8B 84.1 [V] |
| RM-Bench pairwise | goya4140 | 1,327 | 4-domain macro | .813 | .843 (+3.0) | L L L L | L L L L | Skywork-V2: Qwen3-0.6B 74.4, Qwen3-8B 82.6, Llama-3.1-8B 92.8 [V] |
| RM-Bench pointwise | goya4140 | 7,962 | 4-domain macro | .838 | .849 (+1.1) | L L L L | L L L L | REWARDANYTHING-8B .864 [N] |
| RubricBench (rubric given) | goya4140 | 1,147 | pairwise acc | .760 | .795 (+3.5) | L L L L | L L L L | Oracle pipelines .806–.853 [N] |
| PPE Human Preference V1 | goya4140 | 16,038 | no-tie pairwise acc | .644 | .654 (+1.0) | L L T T | L L T T | Skywork-V2 PPE-Preference: Qwen3-0.6B 65.3, Llama-3.2-1B 66.6, Qwen3-8B 70.6 [V]. No train split, so S = Z + public preference data; dedupe against Arena prompts |
| ProcessBench | goya4140 | 3,400 | mean F1 | .695 | .717 (+2.2) | L L L L | L L L L | o1-mini .879 [N]; Qwen2.5-Math-PRM-7B ≈ .735 [R]. Needs a 7B PRM |
| PRMBench Preview | goya4140 | 6,216 | PRM score | .664 | .680 (+1.7) | L L L L | L L L L | Gemini-2.0-thinking .688 [N] |
| TMMLU+ (zh-TW, 66 subjects) | lianghsun | test 19,646 | category macro acc | .776 | .784 | L L L L | L L L L | Knowledge on an English-only backbone |
| GAOKAO-Bench objective | kuaitoukuai | 1,497 | acc | .909 | .929 | L L L L | L L L L | Chinese exam knowledge |
| JMedQA (JMLE 2018–26) | kokuren333 | 3,556 | exact-set acc | .886 | .901 | L L L L | L L L L | Japanese medical knowledge. Number via the CC0 robustness index, not re-verified at source [N] |
| Upworthy headline A/B | Gaurav-Gosain (2026-09-16) | confirmatory, 10,984 pairs | acc | .645 | .658 (+1.3) | L T T T | L L L L | Train = the archive's exploratory split [R]. Number via the robustness index, not re-verified [N] |
| BEIR SciFact rerank | denser (2026-09-22) / hev (2026-09-17) | test, 300 queries | nDCG@10 | .770 / .772 | ≈ .825 (+5.3; per-query SD .33 assumed) | L L L T | L L L T | Qwen3-Reranker-0.6B .748, Voyage .755 [N]. MS MARCO-trained rerankers are Z-eligible; SciFact train (809 claims) is the S data |
| BEIR NFCorpus rerank | denser / hev | test, 323 queries | nDCG@10 | .362 / .358 | ≈ .408 (+4.6) | L T T T | L T T T | Qwen3-Reranker-0.6B .357, Voyage .357 [N]. All strong rerankers sit at .35–.40: the point estimate is reachable, the bar is not |
| Rerank mean (BEIR + BRIGHT + CodeSearchNet, 8 sets) | anessbelbati (2026-09-25) | 1,617 queries | mean nDCG@10 | .692 | ≈ .713 | L L L T | L L L T | Cohere Rerank 4 Pro .691 [N]. The BRIGHT part is reasoning-heavy |
| NevIR (negation pairs) | anessbelbati | test, 1,383 pairs | pairwise acc | .710 | .744 (+3.4) | L L L T | L L L L | NevIR train split exists [R]. Cohere .67, Open-Jev-9B .77 [N] |

#### E-B. Rev 2: tier-B `jev-bench` rows (Praveenrajus) on public test splits that were missing above (13 counted)

- Each row is a 1,000-item sample of the official test split unless the n says otherwise.
- The dataset ships its train splits.
- Jev's raw predictions are public. Use them for evaluation only.
- Harness: one request per test record, per the HF card [N].

| Benchmark | Source | Split, n | Metric | Jev | Bar for a significant win | S 68/150/400/1b | Z 68/150/400/1b | Key evidence |
|---|---|---|---|---|---|---|---|---|
| Yelp-5 | jev-bench | test 1,000 | acc | .685 | .726 (+4.1) | L T T T | L L L L | XLNet-L .730, BERT-L+ITPT .707 [R]; train 650k. The sibling note counts it a ≤150M win on the point estimate [N]; the n=1,000 bar keeps it T |
| civil_comments (noul) | jev-bench | test 2,000 | acc | .729 | .757 (+2.8) | W W W W | T T T T | Train 1.8M. Jev is **below the ~.92 majority-class baseline** [N: train-data note §0.4]. The v2 mix trains on civil_comments, so the Z claim needs the clean retrain |
| measuring_hate_speech (score) | jev-bench | test 1,000 | acc | .527 | .571 (+4.4) | T W W W | L L L L | ~135k annotations [R]; already in the v2 mix [N] |
| MNLI | jev-bench | test 1,000 | acc | .883 | .911 (+2.8) | L T T T | L L L L | Ettin MNLI 87.0 / 89.2 / 91.3 / 91.8 [V]. The v2 mix contains MNLI, so no Z claim |
| ChaosNLI (MNLI part) | jev-bench | 1,599 | acc vs the 100-annotator majority | .615 | .649 (+3.4) | L L T T | L L L L | RoBERTa-L .635 [N]; low-agreement items |
| FEVER (gold evidence, noul) | jev-bench | test 1,000 | acc | .972 | .986 (+1.4) | L L T T | L L L L | Near ceiling |
| StrategyQA closed-book | jev-bench | 687 | acc | .785 | .828 (+4.3) | L L L L | L L L L | Knowledge |
| StrategyQA grounded | jev-bench | 687 | acc | .956 | .978 (+2.2) | L L L L | L L L L | Ceiling |
| MASSIVE en-US (60 intents) | jev-bench | test 1,000 | acc | .808 (laya-ft macro-F1 .799) | .843 (+3.5) | W W W W | L L L L | XLM-R-L en-US ≈ .88–.89 [R]. MASSIVE is also our jevbench test #13, so it is already excluded from Z; an S checkpoint trained on it no longer gives a zero-shot jevbench #13 score |
| GoEmotions single-label "primary" | jev-bench | test 1,000 | acc | .282 | .321 (+3.9) | W W W W | T T T T | Gold = first label [I] |
| HelpSteer2 helpfulness (5 levels) | jev-bench | 1,000 | exact acc | .363 | .405 (+4.2) | T T W W | L T T T | The majority-class prior (≈ .42 on the Jevals sample [N]) already beats Jev |
| HelpSteer2 verbosity (5 levels) | jev-bench | 1,000 | exact acc | .341 | .383 (+4.2) | T T W W | L T T T | |
| STS-B 6-level | jev-bench | 1,000 | exact acc | .538 | .582 (+4.4) | T T T T | L L L L | Rounding a .90+ Spearman regressor gives about .55–.60 [E] |
| *jev-bench macro, 22 configs (shown, not counted)* | jev-bench | 22,773 | macro acc | .733 | ≈ .74 | L L T T | L L L L | At any encoder size, MMLU .923, ARC .979, BoolQ .917 and StrategyQA lose about 1.1 points of summed accuracy; the W rows have to make that up |

#### F. Rev 2: listed, not counted (tier B/C, small n, or a harness we cannot reproduce)

- **LangWatch Jev benchmark** (15 tasks; samples of 500–2,329; LangWatch harness) [N]:

  | Task | Metric | Jev |
  |---|---|---|
  | routing-20 | acc | .891 |
  | routing-77 | acc | .796 |
  | tool routing | acc | .783 |
  | complaint routing | acc | .787 |
  | commit type | acc | .683 |
  | search relevance (ESCI) | acc | .577 |
  | injection | catch rate @ 5% false alarms | .946 |
  | moderation | AUROC | .903 |
  | PII | catch rate | .908 |
  | RAG faithfulness | balanced acc | .803 |
  | off-topic | balanced acc | .934 |
  | web-agent | acc | .708 |
  | community sets | acc | .623 |
  | typed decisions | acc | .739 |

  S [E]: the intent and topic rows are W–T at 150m+, the safety and RAG rows T, web-agent L.
- **Ibrahim & Zaki 2026 CSS suite** (18 tasks, about 500 items each, macro-F1) [N]:
  - Jev, high to low: FLUTE .863, MRF .796, SemEval-16 stance .734, TempoWiC .683, media ideology .654, dialect .640, IBC .633, RAOP .607, discourse .595, Wiki power .581, politeness .573, Reddit humor .563, persuasion .561, CGA .502, CARER .484, implicit hate .439, TalkLife .269 (data restricted), tropes .191 (n=114).
  - At n≈500 the bars are +6–9 points.
  - Rows where Jev is ≤ .65 and a public train split exists are T–W at 400m [E]. FLUTE, MRF and stance are T–L.
  - The sibling addendum counts 10 of 17 as winnable at ≤150M on public evidence: Ziems et al. fine-tuned RoBERTa-L on the same test items [N]. Whether those wins clear an n≈500 bar is not checked.
- **Kev suites:** decision-v7 test .835 (n=1,200); transfer-v4 test .877 (n=656). Whether a train split is usable is unknown. L–T [E].
- **Single studies, n ≥ 500:**

  | Study | Jev | Verdict |
  |---|---|---|
  | Eclipse bug severity | .491 | S W (a trained readout gets .784) |
  | Fake-job PR-AUC | .276 | S W (TF-IDF gets .780) |
  | App-review Spearman | .803 | T |
  | AITA | .754 | T (always-NTA gets .740) |
  | CUAD clause F1 | .519 | T at 400m |
  | Mind2Web | .521 | L |
  | LangWatch web-agent | .708 | L |
  | Who&When | .313 | L |
  | Cohen 2006 screening | recall .90 at precision .201 | Not a single-number bar; compare WSS@95 instead |
  | DecisionBench views | – | L (no train split) |

---

## 4. Tallies and per-size win lists

Rev 2 counts two sets:
- **Core:** 85 benchmarks (groups A–C), with the stricter S bars of rev 2 (Banking77 .924; UNFAIR-ToS .748).
- **All:** the core plus groups E-A (33) and E-B (13), **131** in total.
- Shown-not-counted composites (LexGLUE mean, jev-bench macro) and groups D and F are excluded.

| Track / size | Core W | Core T | Core L | All-131 W | All-131 T | All-131 L | All-131 W + T |
|---|---|---|---|---|---|---|---|
| S 68m | 11 | 24 | 50 | 18 | 30 | 83 | 48 |
| S 150m | 18 | 23 | 44 | 26 | 33 | 72 | 59 |
| S 400m | 22 | 28 | 35 | 33 | 43 | 55 | 76 |
| S 1b | 27 | 26 | 32 | 39 | 46 | 46 | 85 |
| Z 68m | 0 | 10 | 75 | 0 | 13 | 118 | 13 |
| Z 150m | 1 | 13 | 71 | 1 | 19 | 111 | 20 |
| Z 400m | 2 | 18 | 65 | 2 | 26 | 103 | 28 |
| Z 1b | 2 | 21 | 62 | 2 | 33 | 96 | 35 |
| S, best of 4 sizes | 27 | 26 | 32 | 39 | 46 | 46 | 85 |
| Z, best of 4 sizes | 2 | 21 | 62 | 2 | 33 | 96 | 35 |
| Either track, any size | 27 | 26 | 32 | 39 | 46 | 46 | 85 |

Rev 1 had S 68m 13/22/50, S 150m 20/21/44, S 400m 23/28/34 and S 1b 28/25/32 on the core set. The differences come from Banking77 and UNFAIR-ToS, now measured against Jev given train data, and from FinEntity: with no train split it drops to L at 400m. Rev 2 also took the sibling addendum into account (`train-data-and-supervised-ceilings.md` §0.4, 2026-10-03, still being written when this was revised). As a result, Aegis 2.0 response moves to W at 400m (Qwen3Guard-0.6B .842 against a bar of .827), and MNLI becomes T at 150m (Ettin-150m 89.2 beats the .883 point estimate, though not the bar).

**L on both tracks at every size (46 of 131):**
- **Core (32):** SIB-200 (205 langs); AfriXNLI (18 langs); LLM-AggreFact (11 sets); BoolQ; Belebele (122 langs); PubMedQA pqa_labeled; MMLU; C-Eval (Chinese); BIG-bench MC (93 tasks); HellaSwag; WinoGrande; ARC E+C; CommonsenseQA; SummEval (4 scores); BFCL; API-Bank; Home appliance sim.; GPQA Diamond; GSM8K (MC); MuSR; SATA-Bench; CRUXEval; HLE; MMLU-Pro; BBH; BRIGHT; NLI4CT; ForecastBench; New Yorker; ARC-Easy (non-index); ARC-Challenge (non-index); OpenBookQA (non-index).
- **New (14):** HateCheck; HarmBench prompt; RewardBench v1; RewardBench 2; RM-Bench pairwise; RM-Bench pointwise; RubricBench; ProcessBench; PRMBench; TMMLU+; GAOKAO; JMedQA; StrategyQA closed; StrategyQA grounded.

Some of these move only with the LLM paths:
- **Qwen3 scorer (§6):** C-Eval, BoolQ and PubMedQA can become toss-ups.
- **CoT generation:** GSM8K.
- **A ≥4–8B decoder reward model trained on public preference data (§6.1):** RewardBench v1 and RM-Bench become toss-ups. ProcessBench needs a 7B PRM.

"Either track" equals the S row: no benchmark is W/T on Z without also being W/T on S.

**S-track W rows by size** (cumulative):

| Size | Core | New (rev 2) | Total |
|---|---|---|---|
| **68m** | 11: AG News, DAIR Emotion, Financial PhraseBank, GoEmotions, PAWS, prompt-injections, POP909-CL, SGD/SGD-X, NLU++, fin-topic, daily_dialog | 7: HWU64, LexGLUE EUR-LEX, LEDGAR, LexGLUE UNFAIR-ToS, civil_comments, MASSIVE en-US, GoEmotions single-label | **18** |
| **150m** | +7: AGB-DE, ChessBench, CLadder, VAST, HoVer, email spam, tweet_topic | +1: measuring_hate_speech | **26** |
| **400m** | +4: CLINC150, When2Call, typed-decisions (fitted table), UNFAIR-ToS (Deußer, against tuned .748) | +3: HelpSteer2 helpfulness acc, HelpSteer2 verbosity acc, Aegis 2.0 response | **33** |
| **1b** | +5: αNLI, STS-B, HelpSteer2 (ρ), Amazon ESCI, ACOS | +1: LexGLUE ECtHR-B | **39** |

**S-track toss-ups that stay T even at 1b (46):**
- **Core (26):**
  - Banking77, against the few-shot .924;
  - IMDB, Rotten Tomatoes, SST-2, SST-5 and language-id: ceilings or the significance bar;
  - ANLI, ToxiGen, OpenAI moderation, ToxicChat, SMS spam;
  - ToolRet, ContractNLI, BPoMP, Humicroedit, cfcolor, FinEntity, iSarcasmEval, RAGTruth, PhishNChips, Habermas, RouterBench;
  - support-ticket calibration, phishing gradient, TweetEval-emotion, arXiv 2026-09.
- **New (20):**
  - ECtHR-A, SCOTUS, CaseHOLD;
  - MedHallu;
  - Aegis 1.0 prompt, Aegis 2.0 prompt, WildGuardTest, HarmBench response, the combined injection corpus;
  - PPE human preference;
  - Upworthy;
  - SciFact, NFCorpus, rerank-mean, NevIR;
  - Yelp-5, MNLI, ChaosNLI, FEVER, STS-B 6-level.

**Z-track W/T rows at 1b (35):**
- **W (2):** Financial PhraseBank, GoEmotions.
- **T, core (21):** AG News, DAIR Emotion, SMS spam, ToxiGen, OpenAI moderation, prompt-injections, AGB-DE, UNFAIR-ToS, HelpSteer2, iSarcasmEval, PhishNChips, Habermas, SGD, support tickets, phishing gradient, email spam, NLU++, tweet_topic, fin-topic, daily_dialog, typed-decisions.
- **T, new (12):** EUR-LEX, LEDGAR, LexGLUE UNFAIR-ToS, combined injection corpus, PPE, SciFact, NFCorpus, rerank-mean, civil_comments, GoEmotions single-label, HelpSteer2 helpfulness acc, HelpSteer2 verbosity acc.

---

## 5. Expected score ranges [E] for rows where size changes the verdict

Ranges are planning estimates from §2 anchors, not predictions to quote.

| Benchmark (metric) | Jev → bar | S 68m | S 150m | S 400m | S 1b | Anchor |
|---|---|---|---|---|---|---|
| CLINC150+OOS (acc) | .895 → .907 | .88–.91 | .89–.92 | .91–.94 | .92–.95 | BERT oos-train implies .87–.90; OOS threshold on validation adds 2–4 |
| When2Call (acc) | .810 → .828 | .76–.80 | .79–.83 | .83–.87 | .85–.89 | Dinah-0 .784 at 150M |
| typed-decisions (acc) | .727 → ~.76 | .66–.74 | .70–.77 | .76–.80 | .77–.82 | OpenDecider-nano (Ettin-400m) .796 |
| αNLI (acc) | .839 → .867 | .68–.74 | .74–.80 | .84–.88 | .86–.90 | RoBERTa-L .856, L2R2 .885 |
| STS-B (Spearman) | .890 → .906 | .86–.88 | .88–.90 | .90–.915 | .91–.925 | stsb-roberta-large ~.915 [R] |
| HelpSteer2 (mean ρ) | .412 → .457 | .38–.45 | .42–.50 | .46–.55 | .48–.58 | none published for encoders |
| Amazon ESCI (macro-F1) | .552 → ~.572 | .50–.56 | .53–.60 | .56–.63 | .58–.66 | KDD Cup 2022 task 2 [R] |
| RAGTruth (F1) | .765 → ~.788 | .73–.76 | .75–.78 | .78–.81 | .79–.82 | TinyLettuce-68M .750, LD-base .761, LD-large .792 |
| ANLI (acc) | .739 → .760 | .50–.56 | .56–.63 | .66–.72 | .70–.76 | DeBERTa-v3-L +ANLI .702 |
| BoolQ (acc) | .913 → .926 | .76–.80 | .80–.84 | .86–.89 | .88–.905 | DeBERTa-v3-L .8835 |
| SST-2 (acc) | .964 → .981 | .93–.94 | .955–.96 | .965–.97 | .97–.975 | Ettin GLUE SST-2 |
| SST-5 (acc) | .579 → .607 | .52–.55 | .55–.58 | .58–.61 | .59–.62 | RoBERTa-L .602 |
| IMDB (acc) | .965 → .968 | .94–.95 | .95–.96 | .962–.968 | .965–.972 | RoBERTa-L ~.963 [R]; 8k context avoids truncation |
| ToxicChat (F1) | .786 → .834 (S vs tuned .793: ≈.841) | .70–.78 | .74–.81 | .78–.84 | .80–.85 | T5-large .822 |
| Humicroedit (acc) | .619 → .645 | .56–.60 | .59–.63 | .62–.66 | .63–.67 | SemEval best .674 |
| VAST (macro-F1) | .646 → ~.67 | .62–.68 | .68–.74 | .72–.77 | .74–.79 | DeBERTa-v3-base .755 |
| HoVer (acc) | .729 → ~.748 | .76–.80 | .80–.85 | .83–.88 | .85–.90 | Dinah-0 .822 (kept T at 68m for long multi-hop states) |
| ChessBench (acc) | .172 → ~.187 | .17–.22 | .22–.27 | .25–.32 | .28–.35 | Dinah-0 .243 |
| AGB-DE (F1) | .204 → .317 | .25–.40 | .35–.50 | .40–.55 | .45–.60 | English-only tokenizer on German |
| Banking77 (acc), **S vs Jev given train data** (rev 2) | .924 → .937 | .925–.935 | .93–.94 | .935–.945 | .937–.947 | BERT-base .9366, SPACE-2 .948 [N] |
| UNFAIR-ToS, Deußer positives-only (micro-F1), **S vs tuned Jev** (rev 2) | .748 → ≈.79–.815 | .76–.82 | .78–.84 | .81–.87 | .82–.88 | ≈.81 derived for BERT-base [E] |
| HWU64 (acc) (rev 2) | .831 → .863 | .89–.91 | .90–.92 | .91–.93 | .92–.94 | SPACE-2 .942 [N] |
| LexGLUE EUR-LEX (μ-F1) (rev 2) | .391 → .410 | .68–.72 | .70–.73 | .71–.74 | .72–.75 | BERT 71.4, DeBERTa 72.1 [V] |
| LexGLUE LEDGAR (μ-F1) (rev 2) | .753 → .765 | .86–.88 | .87–.885 | .88–.89 | .885–.895 | BERT 87.6, RoBERTa-L 88.6 [V] |
| LexGLUE ECtHR-B (μ-F1) (rev 2) | .754 → .792 | .76–.79 | .78–.80 | .79–.81 | .80–.82 | BERT 79.7, RoBERTa-L 79.8, Legal-BERT 80.4 [V] |
| LexGLUE CaseHOLD (acc) (rev 2) | .773 → .792 | .68–.71 | .70–.73 | .73–.76 | .75–.79 | Legal-BERT 75.3, RoBERTa-L 74.4 [V] |
| **Z:** Financial PhraseBank (acc) | .730 → .770 | .65–.78 | .70–.80 | .75–.85 | .78–.88 | deberta-v3-L-nli .803 F1 |
| **Z:** DAIR Emotion (acc) | .585 → .615 | .50–.60 | .52–.62 | .55–.64 | .58–.66 | Laya .587; noisy distant labels cap ~.65 |
| **Z:** GoEmotions (macro-F1) | .243 → .252 | .25–.33 | .28–.36 | .30–.40 | .32–.42 | — |
| **Z:** AG News (acc) | .885 → .894 | .70–.80 | .75–.83 | .80–.86 | .83–.88 | deberta-L-c .819 |
| **Z:** Banking77 (acc) | .797 → .818 | .50–.62 | .58–.68 | .64–.74 | .68–.78 | GLiNER2.5-Decide .656; lev-4B .868 |
| **Z:** CLINC150 (acc) | .895 → .907 | .55–.65 | .60–.70 | .65–.78 | .72–.84 | GLiNER2.5-Decide .604 |

---

## 6. Prefill-only open-LLM scorer (Qwen3-4B/8B/14B on CPU): needed? feasible?

**Accuracy prior** [V: Qwen3 tech report, Tables 3, 5, 6, 7]. These are few-shot base-model numbers; BBH and GSM8K use CoT generation. A prefill-only option-logit readout with no CoT will be **lower** on BBH, GSM8K and GPQA [E].

| Benchmark | Qwen3-4B-Base | Qwen3-8B-Base | Qwen3-14B-Base | Qwen3-235B-A22B-Base | Jev |
|---|---|---|---|---|---|
| MMLU | 72.99 | 76.89 | 81.05 | 87.81 | .918 |
| MMLU-Pro | 50.58 | 56.73 | 61.03 | 68.18 | .827 |
| BBH | 72.59 | 78.40 | 81.07 | 88.87 | .929 |
| GPQA | 36.87 | 44.44 | 39.90 | 47.47 | .786 |
| GSM8K (CoT generation) | 87.79 | 89.84 | 92.49 | 94.39 | .799 (10-option MC) |
| CRUX-O | 55.00 | 62.00 | 68.60 | 79.00 | .730 (CRUXEval MC) |

Also:
- Qwen2-72B-Base: HellaSwag 87.6, WinoGrande 85.1, C-Eval 91.0.
- Qwen2-7B-Base: C-Eval 83.2.

[V: Qwen2 blog]. Jev: HellaSwag .955, WinoGrande .914, C-Eval .839.

**Verdict by row:**

| Row | Verdict |
|---|---|
| **C-Eval** | **T** at 8B/14B. Qwen2-7B already reaches 83.2 against .839. Check that Deußer's C-Eval *test* labels come from its HF source before relying on it |
| **BoolQ** | T-low zero-shot at 14B; **T** with a LoRA on BoolQ train (S track) |
| **PubMedQA** | T-low at 14B |
| **GSM8K** | **W** only with CoT *generation*. That is a different system class, not a one-pass typed model, and must be reported as such |
| **Everything else** in the knowledge block | **L**, including MMLU and BBH at 235B |

**Feasibility and cost [E]:**
- **Throughput.** Our measured 68m training corresponds to ~1.1–1.35 TFLOPS achieved per F80 node: 6·N_non-emb per token at 4.3–5.4k tok/s/node, sync-inclusive. Compute-only is roughly 2× that. Prefill costs ≈ 2·N per token:

  | Model | Prefill tok/s per node |
  |---|---|
  | 4B | ~170–375 |
  | 8B | ~85–195 |
  | 14B | ~45–100 |

- **Full knowledge pass** of ~40M prompt tokens (MMLU 8.7M, MMLU-Pro 10.8M, BIG-bench 5.3M, BBH 3.9M, C-Eval 3.7M, HellaSwag 2.3M, rest ~5M; Belebele would add ~60M) on 12 nodes:

  | Model | Time | Cost |
  |---|---|---|
  | 4B | 2.5–5.6 h | **$160–360** |
  | 8B | 4.8–11 h | **$320–730** |
  | 14B | 9–20 h | **$610–1,320** |

- **Targeted pilot** (C-Eval test 3.7M + BoolQ 0.6M + PubMedQA 0.35M ≈ 4.7M tokens): 8B ~$40–90, 14B ~$70–160.
- **LoRA on BoolQ train** (~5M token-passes at ~3× prefill FLOPs): 8B ~$110–250.
- **Memory.** bf16 14B weights are ~28 GB, which fits every F80 node.

**Recommendation [D]:** not needed for the bulk of the campaign. Run one 8B pilot on C-Eval and BoolQ validation (~$10–20) before deciding on the test passes. Skip the full knowledge pass: it costs $160–1,320 and the expected yield is zero wins.

### 6.1 Rev 2: a decoder reward-model path for the reward and PRM rows?

**Rows:** RewardBench v1 .926, RewardBench 2 .812, RM-Bench .813 / .838, PPE .644, RubricBench .760, ProcessBench .695, PRMBench .664 (group E-A).

**Evidence:**
- Skywork-Reward-V2 [V, model card], trained on ~26M curated pairs:

  | Model | RewardBench v1 | RewardBench 2 | PPE-Preference | RM-Bench |
  |---|---|---|---|---|
  | Qwen3-0.6B | 85.2 | 61.3 | 65.3 | 74.4 |
  | Llama-3.2-1B | 89.9 | 64.3 | 66.6 | 76.4 |
  | Qwen3-4B | 93.4 | 75.5 | 69.5 | 81.6 |
  | Qwen3-8B | 93.7 | 78.2 | 70.6 | 82.6 |
  | Llama-3.1-8B | 96.4 | 84.1 | 77.3 | 92.8 |

- Bars for comparison: RewardBench v1 .939, RewardBench 2 .837, PPE .654, RM-Bench pairwise .843.
- So the bar is cleared only at ~8B with the best data. ≤1B clears only PPE, narrowly.

**Cost of our own 8B RM on CPU [E]:**
- LoRA training runs at ≈3× prefill cost, ~30–65 tok/s per F80 node.
- On the public ~80k-pair preference set (~90M tokens) and 12 nodes, that is **~32–69 h, ≈ $2.1–4.5k**.
- A 4B model costs about half.
- Expected result: ≈ .92–.93 on RewardBench v1, the Skywork-Reward-v0.2-8B class on the same 80k data [R]. That is a toss-up against .939, and an L on RewardBench 2.

**Verdict [D]:** skip. It is $2–4.5k for at most two toss-ups. Shipping someone else's open RM checkpoint would not be "our model".

**The ≤1B route that is worth a line item:** an Ettin-1b encoder RM trained on the same public preference data, which makes PPE a toss-up (L L T T). It can share the S run as one more task block (~0.1B tokens).

---

## 7. CPU-cluster compute and cost

**Measured basis [V: `runs/v2/v2-68m.log.jsonl`]:**
- v2-68m: 3,907 steps, **1.34B tokens in 39,207 s (10.9 h)** on 8× F80as_v7 (world size 32, 4 ranks/node). That is a **34.2k tok/s** average, with 31–35k tok/s logged at the end.
- The brief quotes ~43k tok/s; it is used as the optimistic end.
- $5.46/h per F80 node. Cost = node-hours × $5.46, so it is the same on 8 or 12 nodes up to scaling losses. 12-node wall-clock below assumes 90% scaling (~11% more node-hours).

**Scaling [E]:**
- Optimistic end: ∝ total params (150m ×2.2, 400m ×5.9, 1b ×14.7), as the brief suggests.
- Pessimistic end: ∝ non-embedding params (×2.6, ×8.3, ×22.4). Non-embedding counts are ~42M / 110M / 348M / ~940M, from the Ettin dims and the 50,368-token vocabulary.
- Bigger matmuls run more efficiently on CPU, so the truth is likely between the two.

**S-mix size [E; train-split sizes R]:**

| Block | Train data | Tokens per epoch, typed format |
|---|---|---|
| Topic / sentiment | AG News 120k; IMDB 25k; RT 8.5k; SST-2 67k phrases; SST-5 8.5k; FPB train; fin-topic 17k; tweet_topic ~4.6k; pre-cutoff arXiv (cap 50k) | ~35M |
| Emotion | DAIR 16k; GoEmotions 43k (28 nouls each); daily_dialog 87k utterances; TweetEval-emotion 3.3k | ~25M |
| Intent | Banking77 10k (77 options each); CLINC plus 15.25k (151 options); NLU++ folds 0–17; SGD (cap 100k turns) | ~60M |
| NLI / pairs / QA | ANLI 163k; PAWS 49k; STS-B 5.7k; BoolQ 9.4k; αNLI 170k; HellaSwag 40k; WinoGrande 40k; CSQA+ARC+OBQA ~18k; PubMedQA artificial (cap 50k) | ~75M |
| Safety / legal | ToxiGen train (9k human + capped machine); ToxicChat 5k; prompt-injections 546; AGB-DE train; UNFAIR-ToS 5.5k; other spam/moderation corpora (cap 100k) | ~25M |
| Scores | HelpSteer2 20k (5 scores, long) | ~16M |
| DI-specific | ESCI (cap 300k pairs); RAGTruth ~15k; HoVer 18k; VAST 13.5k; iSarcasm 3.5k; ContractNLI 423 docs; NLI4CT 1.7k; Humicroedit 9.4k; When2Call; ToolRet-train (capped); API-Bank train; ACOS; POP909 (DI songs held out); Lichess (cap 200k); CLadder generator (50k); RouterBench (cap 50k) | ~200M |
| Typed | typed-decisions train | ~1M |
| **Rev 1 total** | | **~0.45B tokens/epoch** |
| Rev 2: legal (LexGLUE) | ECtHR ~9k cases (long; truncate at 8k); SCOTUS ~5k opinions (long; truncate at 8k); EUR-LEX 55k; LEDGAR 60k; CaseHOLD ~45k (5 options) | ~150M |
| Rev 2: safety | Aegis 1.0 (~10.8k); Aegis 2.0 (~30k); WildGuardMix (87k, if its licence is accepted); BeaverTails (cap 100k); civil_comments (cap 200k); measuring-hate-speech (~40k comments) | ~45M |
| Rev 2: intent / topic / sentiment | HWU64 (~9k; spends a dev set); MASSIVE en-US 11.5k; Yelp-5 (cap 150k); Upworthy exploratory pairs | ~35M |
| Rev 2: grounding / ranking / preference | MedHallu train (~9k); NevIR train; SciFact train; MS MARCO pairs (cap 200k); public preference pairs (~80k, for PPE) | ~90M |
| **Rev 2 total** | | **~0.77B tokens/epoch** |

One rev-2 S run is about 1.3B tokens:
- ~1.5 epochs of the rev-1 blocks with small sets upsampled: ~0.7B;
- ~1 epoch of the rev-2 blocks: ~0.3B;
- clean v2-mixture replay: ~0.3B.

Rev 1 was ~1.0B. The 8k-token LexGLUE documents cost more per token, because every third Ettin layer is global attention: ≈ +30–45% on those ~65M tokens, ≈ +2% on the run [E]. That is ignored below.

**Cost per run** (optimistic → pessimistic; time on 12 F80 nodes):

| Run | 68m | 150m | 400m | 1b |
|---|---|---|---|---|
| **S multi-task, rev 2, 1.3B tokens** | $367–462 · 6.2–7.8 h | $809–1,200 · 14–20 h | **$2,158–3,831 · 37–65 h** | $5,395–10,338 · 92–175 h |
| S multi-task, rev 1, 1.0B tokens | $282–355 · 4.8–6.0 h | $622–923 · 10.6–15.7 h | $1,660–2,947 · 28–50 h | $4,150–7,952 · 70–135 h |
| Z clean retrain, 1.34B tokens | $378–476 · 6.4–8.1 h | $834–1,237 · 14–21 h | $2,224–3,948 · 38–67 h | $5,560–10,656 · 94–181 h |
| S-spec / short S fine-tune, 0.3B tokens | $85–107 | $187–277 | $498–884 | $1,245–2,386 |
| Evaluation through all harnesses (~530k requests) | ~$5–15 | ~$10–30 | ~$30–80 | ~$80–220 |

**Risks for 400m/1b on CPU [V/E]:**
- **DDP synchronization already eats ~half of each step at 68m.** The log shows `sync_s_per_step` 4.8–5.4 s at 0.09–0.10 steps/s with a 135.5 MB all-reduce.
- Gradients at 400m are ~1.4 GB and at 1b ~4 GB (fp32). Raise tokens per optimizer step (≥1M by accumulation), all-reduce in bf16, and check accelerated networking on every NIC. Otherwise expect the pessimistic end or worse.
- **Memory.** 68m peaked at 24.5 GB RSS per rank. 1b fp32 AdamW state alone is ~16 GB per rank, so run 1–2 ranks per node on the 160 GiB F80als nodes [E].
- **No v2 checkpoint exists at 150m/400m/1b.** S runs start from the Ettin base with typed heads, so the 1.3B-token budget includes learning the format. 68m can start from v2-68m, which needs only ~0.3–0.6B tokens ($85–215).

**Budget fit:**
- Credits were $9,927.98 on 2026-10-01 [N], before this week's cluster use: v2-68m alone ≈ $476, plus ablations and idle time, not tallied here.

| Plan | Contents | Cost [E] | Outcome |
|---|---|---|---|
| **Core** | S-68m from v2 (0.4–0.8B tokens) + S-400m (1.3B) + clean Z-68m + Qwen3 pilot + evals | **≈ $2.7–4.9k** (rev 1: $2.2–3.9k) | On S at 400m: 22 W / 28 T of the core 85, or 33 W / 43 T of all 131 |
| **Stretch** | Core + 1b S-spec (0.3–0.5B tokens) on the 6 rows that are W only at 1b, plus the 1b toss-ups + Z-150m | **≈ $4.8–10.1k** (the top end exceeds the credits) | ≤39 W of 131 on S |
| Full 1b | S + Z at 1b | **$11.0–21.0k** | Exceeds the credits; not recommended |
| Decoder RM (§6.1) | 8B LoRA on ~80k preference pairs | ≈ $2.1–4.5k | ≤2 toss-ups; not recommended |

---

## 8. Where "better everywhere" is unrealistic, and why

1. **World knowledge and multi-step reasoning.**
   - Rows: MMLU .918, MMLU-Pro .827, GPQA .786, BBH .929, BIG-bench .814, ARC .988 / .993 / .978, OpenBookQA .94, CSQA .882, HellaSwag .955, WinoGrande .914, MuSR .661, CRUXEval .730, GSM8K .799 (legit), HLE .204, SATA-Bench .264, ForecastBench, New Yorker .701.
   - Encoders of 0.5B or less are at chance-plus on these, even when trained on the train split (Dinah-0: HellaSwag .558, WinoGrande .628).
   - Qwen3-14B-Base is below Jev on every one it reports, except GSM8K with CoT generation (a different system class). Qwen3-235B-A22B-Base is still below on MMLU and BBH.
   - No open DI entrant up to 36B beats Jev on 7 of them.
2. **Multilingual.**
   - Rows: SIB-200 .815, Belebele .867, AfriXNLI .640, C-Eval .839.
   - Ettin is English-only.
   - On SIB-200, Jev beats per-language fully supervised XLM-R-large (.759).
   - A multilingual backbone (mmBERT, AfroXLMR) would turn SIB-200 and AfriXNLI into at best toss-ups. That is outside the Ettin plan.
   - Rev 2 adds non-English knowledge exams: TMMLU+ .776, GAOKAO .909, JMedQA .886. They are both multilingual and knowledge-bound.
3. **Ceilings.** SST-2 (bar .981), IMDB (.968), language-id (.998), ARC, and (rev 2) HateCheck .992, HarmBench-prompt .992, FEVER-with-evidence .972 and StrategyQA-grounded .956. A better model can tie but cannot clear the significance bar.
4. **No usable train split, and Jev already beats every public specialist.**
   - Rows: LLM-AggreFact (.786 vs leaderboard best .774 at 7B), SummEval (.554 vs G-Eval-4 .514 [R]), PubMedQA (.787 vs BioLinkBERT-L .722 [R]), BFCL, BRIGHT, API-Bank, Home appliance, BPoMP.
   - Here S collapses to Z.
5. **Hard grounded reading.**
   - Rows: BoolQ .913, NLI4CT .841, ANLI .739.
   - The best published encoders top out at ~.88–.90 on BoolQ and ~.70–.76 on ANLI, even trained.
   - NLI4CT's best shared-task system (Mixtral) scored .80.
6. **Typed home-turf suites with no train split.** JevBench (Intelligence 72.0), jabr v2 (.967), DecisionBench (.720), DecideBench (.980). Out of reach and not trainable.
7. **Reward models and process reward models (rev 2).**
   - Rows: RewardBench v1 .926, RewardBench 2 .812, RM-Bench .813 / .838, RubricBench .760, ProcessBench .695, PRMBench .664.
   - The public record (Skywork-Reward-V2 [V]) needs ~8B decoders trained on millions of curated pairs to clear these bars.
   - ≤1B reward models trained the same way do not clear them.
   - Only PPE human preference (.644) is within reach of a ≤1B model.
8. **Safety rows where 0.6–7B specialists sit under the bar (rev 2).** WildGuardTest (.884 vs WildGuard-7B .889 [R]), HarmBench response (.875 vs Shieldstral .870) and Aegis 1.0 (.891, n=359, vs Qwen3Guard-0.6B .908 [N] against a bar of .937). A 400m–1b encoder can at best tie.
9. **Legal knowledge (rev 2).** On CaseHOLD (.773), Jev beats every LexGLUE fine-tune, legal-pretrained BERTs included (75.4). ECtHR-A (.730) and SCOTUS (.726) need long-context, legal-domain models just to tie. By contrast, EUR-LEX (.391) and LEDGAR (.753) are among the easiest wins anywhere.

What can honestly be claimed after the core plan, if the estimates hold:
- "On about 33 of 131 public benchmarks with published Jev numbers, a ≤400M supervised specialist beats Jev's best published score by a statistically clear margin. Where Jev has a number given train data, it is that number. On about 43 more it is within noise." On the core 85, the figures are 22 and 28.
- "Zero-shot, it beats Jev on 1–2 benchmarks and is competitive on about 21 of the core 85 (33 of all 131)."
- "It is far behind on knowledge, reasoning and multilingual tasks."
- Plus the calibration wins (Jev ECE .279 on emotion, .138 on FPB [V]) and exact option-order invariance.

---

## 9. Next actions (no spend; Mac-safe)

1. **Fix the contamination flags first.**
   - Drop `sms_spam` and `imdb` from the Z mixture and mark v2-derived SMS-spam results void.
   - Add a `public_jev` exclusion family to `jev_local/bench/registry.py` covering every row of §3 and its siblings (Z).
   - Keep a separate S allow-list of train splits.
2. **Pre-register the targets.** Freeze the per-row bars of §3 in `bench/public/targets.json`, which the campaign memory names but which does not exist yet: dataset id, split, n, metric, Jev value, source URL and date, bar. Tag the commit before any S or Z run.
3. **Harness adapters.**
   - Deußer runner → local `/v1/systemone`: verify whether a base-URL override exists, else patch and disclose.
   - DI kit `--engine http`.
   - DMB, elcronos, zhuyansen scripts.
   - typed-decisions card protocol.
   - Long-state chunking for ContractNLI-length rows, because DI counts Unsupported as wrong.
4. **Confirm train-split availability before counting a row as S-trainable:**
   - which agreement config the atrost FPB test is;
   - PhishNChips, cfcolor and Habermas;
   - FinEntity: answered. DI scores all 979, so there is no train split [N]; the row is now L L L T on S;
   - POP909-CL (which songs DI sampled, so they can be held out);
   - how DI formulates SGD/SGD-X;
   - ACOS, ToolRet-train, API-Bank train, RouterBench and When2Call train sizes;
   - C-Eval test labels in Deußer's source.
5. **Build the S-mix and run the cross-benchmark decontamination** (union of S data against the union of evaluated splits) on Azure, never on the Mac.
6. **Ask the user before any of this spends credits:** Core ≈ $2.7–4.9k; Stretch ≈ $4.8–10.1k (rev 2).
7. **Rev 2 additions:**
   - Add **adapters** for:
     - chepyle's LexGLUE runner (`lexglue-systemone-v1`, threshold 0.5, official 'none' column, `--max-chars 48000`);
     - mbburabak's safety harness (fan-out, 0.5 threshold, review band);
     - goya4140 only if the PPE row is pursued;
     - the denser/hev/anessbelbati rerank scripts;
     - the jev-bench HF card protocol (Praveenrajus ships its train splits);
     - thisisandreeeee (HWU64).
   - Confirm the **jev-bench sampling** of civil_comments (natural vs balanced rate) and how GoEmotions' "primary" label is defined.
   - Confirm the **licence** of the Upworthy exploratory split and of WildGuardMix (gated) before adding them to the S mix.
   - Decide whether to **spend HWU64 as an S train set**. It is our v2 dev set. A replacement dev set is needed first, e.g., SNIPS or ATIS, neither of which has a Jev number.

---

## 10. Sources

**Jev numbers**
- Deußer, Sparrenberg, Sifa 2026, arXiv 2609.37647; code and `results/eval/*.json`, `summary.md`: https://github.com/AppliedMachineLearning-Lab/jev-benchmarking (MIT; read 2026-10-02). Response licence: `responses/LICENSE_RESPONSES.md` (Jev Responses License v1.0 §3).
- Decision Index 0.2.1 board JSON: https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json (generated 2026-09-28T00:39:36Z; read 2026-10-02). Kit: https://github.com/apolinario/decision-index
- Dinah-0: https://github.com/apolinario/decision-index/pull/29 · https://huggingface.co/datasets/Lukitaduarte/dinah-0-decision-index-results (rev f1c4c6f50d7f…) · https://lukita.me/posts/dinah-0-en/
- DMB, elcronos, zhuyansen, Jevals, typed-decisions, AbdelStark and other studies: as cited in `docs/research/v2/classifier-benchmarks.md` §6 [N].

**Size priors and anchors**
- Ettin: https://arxiv.org/abs/2507.11412 (Table 1, Table 7).
- Qwen3 technical report: https://arxiv.org/abs/2505.09388 (Tables 3–8). Qwen2 blog: https://qwenlm.github.io/blog/qwen2/
- Laurer DeBERTa-v3-large-mnli-fever-anli-ling-wanli: https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli · ModernBERT-large-zeroshot-v2.0: https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0
- BoolQ DeBERTa-v3-large: https://huggingface.co/nfliu/deberta-v3-large_boolq
- LexGLUE: https://arxiv.org/abs/2110.00976 (Table 3; evaluation with the extra 'none' label).
- CLINC150 / Larson et al. 2019: https://aclanthology.org/D19-1131/
- SIB-200: https://arxiv.org/abs/2309.07445 (Table 3).
- LLM-AggreFact leaderboard: https://llm-aggrefact.github.io/
- ToxicChat-T5-large: https://huggingface.co/lmsys/toxicchat-t5-large-v1.0
- LettuceDetect / TinyLettuce: https://arxiv.org/abs/2502.17125 · https://github.com/KRLabsOrg/LettuceDetect/blob/main/docs/TINYLETTUCE.md
- Humicroedit (SemEval-2020 Task 7): https://cs.rochester.edu/u/nhossain/hossain-semeval-2020-task-7.pdf
- NLI4CT (SemEval-2024 Task 2): https://arxiv.org/abs/2404.04963
- VAST: https://aclanthology.org/2020.emnlp-main.717.pdf · TATA: https://arxiv.org/abs/2310.14450
- SST-5 RoBERTa-large: https://arxiv.org/abs/2005.13619
- αNLI L2R2: https://arxiv.org/abs/2005.11223
- WinoGrande: https://arxiv.org/abs/1907.10641
- POP909-CL: https://github.com/AndyWeasley2004/POP909-CL-Dataset

**Rev 2 sources (2026-10-03)**
- Jev rows for groups E-A, E-B and F come from `bench/public/jev_published.json` and `docs/research/benchmax/jev-published.md` (read 2026-10-02 by the parallel sweep). Primary sources:
  - LexGLUE: https://github.com/chepyle/jev-test/blob/main/RESULTS.md
  - Safety suite: https://github.com/mbburabak/jev-safety-benchmark
  - Reward-model suites: https://github.com/goya4140/jev-reward-model-evaluation
  - Injection corpus: https://github.com/ASEVlad/jev-injection-bench
  - MedHallu: https://github.com/stperic/jev-medhallu-benchmark
  - HWU64 and others: https://github.com/thisisandreeeee/jev-benchmarks
  - Reranking: https://github.com/denser-org/rerank-bench-jev · https://github.com/hev/reranker · https://github.com/anessbelbati/jev-rerank-bench
  - jev-bench configs: https://huggingface.co/datasets/Praveenrajus/jev-bench
  - Banking77 given train data: https://github.com/simonmesmith/jev-banking77-experiment
  - Upworthy, JMedQA, TMMLU+ and GAOKAO: the repos named in the JSON rows (some via the CC0 Yifan-Lan/awesome-jev-robustness index, not re-verified).
- LexGLUE leaderboard (medium, large and small model tables): https://github.com/coastalcph/lex-glue README, read via `gh api` 2026-10-03 [V].
- Skywork-Reward-V2 benchmark table: https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B, read 2026-10-03 [V].
- Supervised ceilings and contamination traps: `docs/research/benchmax/train-data-and-supervised-ceilings.md` §0.1, §0.2, §7, and the §0.4 addendum (2026-10-03: Qwen3Guard-0.6B Aegis numbers, the v2-mix overlap list, BERT-base HWU64 .916, RoBERTa-L ChaosNLI .635) [N].

**Local**
- `docs/research/benchmax/{jev-published,train-data-and-supervised-ceilings,suite-reproduction-specs}.md`, `bench/public/jev_published.json`
- `docs/research/v2/{classifier-benchmarks,open-baselines,PLAN,backbone-recipe}.md`
- `docs/research/leaderboards/{PLAN,live-leaderboards*}.md`
- `jev_local/bench/registry.py`
- `runs/v2/scores_dev.json`
- `runs/v2/v2-68m.log.jsonl`
