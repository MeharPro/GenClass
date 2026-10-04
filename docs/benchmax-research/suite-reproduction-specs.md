# Benchmax: how to reproduce every published-Jev suite exactly for meharsjev

- **Freshness re-check 2026-10-03:** every pinned source is unchanged since this file was written. GitHub HEADs are still Deußer `6bbdeb33`, Decision Index `87d4650b`, JevBench `bb05a335`, AbdelStark `0d610cc5`, Jevals `21bb47b7`, DMB `eabd88b0`, WorkflowEvals `0ac3b8ad` and Laya `fa9a2a70`. HF revisions are still typed-decisions `d0e2f0c4`, evalsafe-* `6beeb2d2`/`b1342f5a`/`fbe1ea5c`/`86355409` and BTZSC `fef2a2ac`. JevBench live is still v1.5.5 (v1.5.6 returns 404). The only DI Space commit since 2026-09-28 (2026-10-02T01:22Z) renamed another entrant and left Jev unchanged. A re-read of the Jevals `board.json` Jev rows matched §6.2.
- **Date:** 2026-10-02. **Status:** research notes and specs only. Nothing was run, trained, downloaded at scale, submitted or posted. The only remote reads were small code, JSON and markdown files fetched with `gh api` or `curl` (listed in §12).
- **Budget rule (user, 2026-10-02):** at most $5 ever on running Jev. Every comparison below uses **Jev's published numbers**. None of them needs a Jev API call. Publicly released raw Jev outputs appear only as optional, **evaluation-only** references, and §1.4 flags the licence problems with them.
- **Tags:** [F] fact read from the cited file at the cited commit; [I] inference; [E] estimate; [D] recommendation.
- **Tracks:**
  - **(Z) zero-shot:** no data from the benchmark's source datasets in training (any split, any mirror or sibling), and no checkpoint selection on them.
  - **(S) supervised/specialist:** the train splits of the benchmark's sources are allowed, test is never touched, and both are disclosed. S1 is one multi-task model over the union of allowed train splits. S2 is a per-benchmark specialist.
- **Copyright note:** instruction strings are not re-typed here. Every spec points at the exact file, symbol and commit, and the adapters import or rebuild those requests byte for byte. That is more exact than copying text, and it avoids reproducing unlicensed text (DMB and elcronos have no licence).

---

## 0. Bottom line

1. **Nine suites carry Jev numbers. Seven can be reproduced offline at $0 in Jev calls.**
   - Five are byte-exact: Deußer, Decision Index, AbdelStark, DMB and TypeSafe evals.
   - Two are exact up to an unpublished detail that published data pins down: Jevals (per-item state hashes) and typed-decisions (Prior and Uniform reference rows pin the private scorer).
   - JevBench needs a maintainer run for a comparable number.
   - Laya evals has no Jev measurement of its own.

   | Suite | What carries Jev's number | Exact reproduction route | Verification hook |
   |---|---|---|---|
   | Deußer (37 tasks) | `results/eval/*.json` @6bbdeb33 | Import their MIT task classes, push our answers through their own `evaluate.py` | n per task must equal Jev's n |
   | Decision Index 0.2.1 (38 counted) | kit fixture `tests/fixtures/board-0.2.1.json` (Jev entrant) | MIT kit `suite rebuild` + `pipeline --engine http`, then `score --edition 0.2.1` | Rebuilt uncompressed sha256 `b2b56d6f…` / `7429f3c9…` |
   | JevBench v1.5.5 | live API `benchmarkheaven.com/api/jevbench/v1.5.5` | **Not reproducible offline**: half of Intelligence comes from 720 sealed items. Only the 231 public items in git can be proxied. The real number comes from a maintainer run (free to us) | v1.2 split sha256 in `datasets/manifest.json` |
   | AbdelStark BTZSC pilot | `results/reports/btzsc-pilot-v1.md` @0d610cc5 | Apache-2.0 repo, point its TypeSafe client at our server | Manifest sha256 `ec064c52…` |
   | Jevals 2026-09-18 | `releases/2026-09-18/board.json` | Reimplement (harness not public); suite files list every item | Per-item `state_sha256` + `criteria_hash` |
   | DMB expanded | `results/expanded-jev-2026-09-29/` | Reimplement from the documented protocol (repo has **no licence**) | Item and decision counts |
   | typed-decisions | HF card @d0e2f0c4 | Rows are the request bodies; our jevbench #23 file is byte-identical | Reproduce the Prior and Uniform rows exactly (scorer not public) |
   | TypeSafe evals | HF `typesafe/evalsafe-*` `dataset.json` | TypeSafe's own Apache-2.0 `WorkflowEvals` harness with `--base-url` set to our server | Pinned dataset revisions; case content hashes |
   | Laya evals | `BENCHMARKS.md` @fa9a2a70 | No Jev measurement of its own; every Jev cell is copied from AbdelStark, typed-decisions or DMB | Map to the source suite |

2. **The same dataset was scored by Jev under 5–9 incompatible protocols.** Banking77 is the clearest case (table below). A number is comparable only if we match its protocol exactly. Our current jevbench templates match none of them for Banking77, CLINC, emotion or SST-2 (§1.3).

   | Source | Rows | Labels as sent | Where the text sits | Metric | Jev |
   |---|---|---|---|---|---|
   | Deußer | mteb/banking77 test, 3,076 | 77 keys = label names, value `null`, sorted | state `{query}` | acc / macro-F1 | 0.797 / 0.788 |
   | Decision Index | PolyAI CSV test, 3,080 | keys `option_0..76`, value = raw `snake_case` name | **instructions** (state `{}`) | macro-F1 over semantic labels | 0.7974 |
   | DMB expanded | PolyAI CSV test, 3,080 | key = value = name | state = raw string | acc | 0.792 |
   | Jevals | mteb test @18072d26, 300 items × 5 epochs | key = name, value `null`, **shuffled** per epoch | state `{text}` | acc / Decision Score | 0.7967 / 67.78 |
   | AbdelStark | BTZSC, 100 class-balanced, **72 labels** | keys `label_000..`, value = BTZSC hypothesis | state `{text}` | acc / macro-F1 | 0.870 / 0.857 |
   | DMB pilot | 77 items from **train.csv** | key = value | state = string | acc | 0.805 |
   | onlyoneaman | legacy-datasets test, 300, seed 7 | per published case file | state `{text}` | acc | 0.760 |
   | zhuyansen | 1,000 stratified, seed 0 | two-step: 11 groups, then label | 20 texts per call | acc | 0.712 |

3. **Our engine is packing-invariant and order-invariant by construction.** Each option is isolated, and state tokens never see question tokens (`jev_local/bench/ours.py` docstring). So the 5 Jevals option orders, Jev's yes/no penalty when nouls share a request with other questions (0.843 → 0.788 on typed-decisions), and the order-flip rate (Jev 0.103) cannot hurt us. Three things still matter:
   - state and instruction wording;
   - where the input text sits (state, or instructions with an empty state);
   - how labels are presented (bare names, `null`, `""`, key = value, opaque `option_i` keys with descriptions).

4. **Engine changes needed before any number is valid** (§1.2):
   - (a) wire answers must carry `choice` and `confidence`, which Deußer and DMB read, plus `score` and `legend`;
   - (b) a leaderboard mode with unrounded or sum-exact probabilities: 2-dp rounding of 77–151 options can break the Decision Index's |Σp − 1| ≤ 0.01 check and JevBench's strict 1e-3 check;
   - (c) refusal bodies that carry the Decision Index's capacity markers, with a status that can be configured to 400 or 422;
   - (d) a switch between truncating and refusing (the Decision Index forbids truncation; Deußer scores answered rows only, so refusing would cherry-pick);
   - (e) object-valued instructions and empty state handled. Deußer moderation, UNFAIR-ToS and SummEval, and Decision Index ToolRet and BRIGHT, send object instructions; about 15 Decision Index benchmarks (plus iSarcasm track C) send state `{}` or `""`.

5. **Licence and contamination flags that change the plan:**
   - **The Deußer Jev-responses licence (Part A §3.2) forbids using the data "to develop, or to facilitate the development of, a product or service that is similar to or competes with TypeSafe's models".** meharsjev arguably is such a product. [D] Use only the paper's aggregate numbers. Do not download `responses.db` (207 MB, Zenodo 10.5281/zenodo.23039006) without the user's explicit decision.
   - **DMB's NLU++ test is folds 18–19, and our registry uses NLU++ folds 18–19 as jevbench-dev** (registry comment under `DEV`). Selecting checkpoints on it makes a DMB NLU++ claim selection-contaminated. [D] Move our NLU++ dev to folds 16–17 (DMB's own validation folds) or 0–15.
   - **typed-decisions and TypeSafe's WorkflowEvals use the same four workflow names** (agent_trace_observability, customer_service, invoice_processing, security_incidents). An (S) model trained on typed-decisions `train` must not be used for a (Z) TypeSafe-evals claim.
   - DMB's pilot Banking77 items come from **train.csv**. Any (S) model trained on Banking77 train is contaminated for that 0.805 pilot number.
   - The v2 mixture is not zero-shot on several suite sources:
     - Deußer IMDB: tsj `imdb` 4,135 rows + counterfactual IMDB 2,115 rows;
     - very likely Deußer SMS spam, since `sms_spam` is in `KEPT_DESPITE_EARLIER_LIST`;
     - Decision Index ContractNLI, CLadder, ESCI and Humicroedit [F: `leaderboards/live-leaderboards.md` §3].
     - Deußer ARC, CommonsenseQA, αNLI, PubMedQA, SIB-200, Belebele, AfriXNLI, C-Eval, language-id, AGB-DE and SummEval are **unaudited**: the registry has no rule for them, and tasksource carries several of these task families [I]. Audit before any (Z) claim (§1.5).
   - **LLM-AggreFact correction:** Deußer's headline 0.786 is the *unweighted mean of per-source balanced accuracy* over 11 sources (`LLMAggreFact.extra_metrics`), not a pooled balanced accuracy as `v2/classifier-benchmarks.md` §0 says. The pooled per-request accuracy is 0.828.

6. **Budget:** every route costs $0 in Jev calls. Our model runs on the Azure CPU cluster (credits).
   - Downloads happen on Azure only: the Decision Index suite build (~7 GB) and the HF datasets.
   - Gated datasets need the **user's** HF login and terms acceptance: LLM-AggreFact, ToxiGen and HLE.

---

## 1. Cross-cutting requirements (all suites)

### 1.1 What "beat Jev" means here
- **Target:** Jev's published point estimate, under the suite's exact protocol and on the same items. If items differ, the result goes in a separate "indicative" column, never the headline.
- **Strong claim:** our 95% CI lower bound is above Jev's point. Use each suite's own CI method:
  - Deußer: bootstrap, 500 resamples, seed 0;
  - Jevals: item-cluster bootstrap, 2,000 resamples;
  - AbdelStark: target-stratified bootstrap, 2,000 resamples;
  - JevBench: paired stratified bootstrap, B = 2,000, seed 15.
- **Run-to-run noise:** Jev itself is not fully deterministic (Jevals repeat-flip 0.027 on identical Banking77 requests, 0.0233 on HelpSteer2). Treat sub-1-point margins as ties.
- **Coverage:** every suite except Deußer counts a refusal as wrong. Deußer scores only answered rows: `evaluate()` keeps responses that exist. So in a Deußer comparison we must answer 100%, or report a "missing = wrong" variant beside it.

### 1.2 Engine and wire adapter: one config for every suite
Our engine already speaks the Jev wire format (`POST /v1/systemone`, in-process `FastEngine` / `OursRunner`). Required behaviour:

| # | Requirement | Why (suite → code) |
|---|---|---|
| W1 | Choice answers carry `choice` (argmax on **unrounded** probabilities; ties go to the first key in request order), `confidence` = (K·pmax − 1)/(K − 1), and `probabilities` | Deußer `choice_metrics` reads `a["choice"]` and `a["confidence"]` (KeyError otherwise). DI `prediction()` uses `a["choice"]` and `validate()` requires it to be among the options. DMB uses `choice` + `confidence`. JevBench's TypeSafe adapter rejects a `choice` outside the labels. `OursRunner.answer` currently returns only `probabilities`, so the server's `build_answer` path must be used |
| W2 | Score answers carry `score` = Σ i·pᵢ (unrounded), `confidence`, `legend`, and `probabilities` keyed `"0".."K-1"` | Deußer Spearman uses `score`; argmax-accuracy uses int keys. JevBench v1.5 Score uses the expected position |
| W3 | **Leaderboard precision mode:** unrounded floats, or 2-dp largest-remainder rounding that sums to exactly 1.00. Today `confidence.build_answer` rounds each option to 2 dp independently (`round_digits=2`) | DI `validate`: |Σp − 1| ≤ 0.01. JevBench `SUM_TOL` 1e-3 is the pre-registered rule; `RENORM_TOL` 2e-2 is the headline. A 151-option CLINC row can drift past 0.01 with per-option rounding [I]. For paired NLL/KL comparisons with Jev (which returns 2 dp), also report a 2-dp view: `metrics.py` already floors NLL at 0.005 |
| W4 | **Refusals:** keep `{"detail":[{"type":"max_tokens_exceeded","msg":…}]}`, but the message must contain `maximum context length`. The option cap at 255 returns 400 with `options per choice` in the message. The status is configurable: 400 by default (Jev parity), 422 for JevBench | DI `http.CAPACITY_MARKERS` turns only marked 400/413/422 into `unsupported`; anything else becomes `error`, which is retried, and five errors before the first success abort the run. Our current message ("accepts at most N tokens (local limit)") matches none of the markers. Deußer's runner treats 400 + `max_tokens_exceeded` and 413/422 as permanent. JevBench issue #113 asks for 422. DMB saw Jev reject 256+ options |
| W5 | **Overflow switch** `--overflow refuse\|truncate`. `OursRunner._plan` truncates the longest state field today | DI: no truncation; refusing is mandatory. JevBench: refuse. Deußer, AbdelStark, Jevals, DMB, typed-decisions: truncation is allowed, but disclose and count it |
| W6 | Accept object-valued `instructions` and criteria values, rendered deterministically (`serialize.entry_text`) | Deußer OpenAI-moderation, UNFAIR-ToS and SummEval questions use object instructions. DI ToolRet/BRIGHT use `{"task","candidate"}`. Our jevbench templates *stringified* these for OpenRouter, so they are not Deußer requests |
| W7 | Empty state (`{}` or `""`) with all content in `instructions` | About 15 DI benchmarks: the 10 `layout.choice_row` users (ARC-E/C, MMLU, WinoGrande, HellaSwag, ANLI, BANKING77, CLINC, SimpleBench, MuSR), `normalize_text._mc_row` (VAST, CLadder), GPQA, HLE and Humicroedit, plus iSarcasm track C; BPoMP sends only a task string. `state_segments({})` returns `[]`, so verify the packer handles zero segments |
| W8 | Treat `null`, `""` and value == key the same: "read the label by its name" | Deußer uses `null`. elcronos uses `""`. DMB and DI FinEntity use value == key |
| W9 | One global calibration file: temperatures per (kind × K-bucket), fitted only on non-benchmark dev data. Never per suite | DI rule "one fixed rendering, no per-benchmark calibration", and good practice elsewhere |
| W10 | Model id `meharsjev-68m`, never `jev-*` | Deußer `import_responses.py` refuses `jev-*` responses. Trademark caution (leaderboards PLAN §0) |
| W11 | Deterministic (same bytes in, same answer out); report CPU threads and hardware | Jevals repeat-flip and order-flip become 0 for us; say so as a property, not a tuned result |

### 1.3 Our existing jevbench templates are not the published protocols
`jev_local/bench/templates.py` deliberately departs from Deußer:
- Banking77 and DAIR emotion use BTZSC hypotheses where Deußer sends `null`.
- CLINC uses programmatic descriptions where Deußer sends `null` (OOS keeps its description).
- SST-2 uses a noul where Deußer uses a 2-way choice.
- OpenAI moderation and UNFAIR-ToS are stringified.
- AG News is a 2,000-item sample and AggreFact a 5,000-item sample, where Deußer uses the full splits.

That is fine for jevbench, our internal yardstick, but **jevbench numbers must never be set against Deußer's**. [D] Add a `deusser_exact` route (§2.4) that imports Deußer's task classes directly, and keep jevbench as is.

### 1.4 Released Jev outputs: what may be used, and how

| Artifact | Licence | Use allowed by us [D] |
|---|---|---|
| Deußer Zenodo `responses.db` (207 MB) | Jev Responses Licence 1.0, Part A: research and evaluation, but §3.1 bans distillation and §3.2 bans developing or facilitating a similar or competing product | **Do not use** without the user's explicit decision. §3.2 plausibly covers meharsjev. Aggregates from the paper and `results/eval/*.json` (MIT repo) are enough |
| Jevals `runs/jev__*.jsonl` | CC BY 4.0 | Paired per-item evaluation only. Never training, selection or calibration |
| JevBench `results/v1.2/jevbench-v1.2-per-task.json` (per-item c/w outcomes, public items) | MIT repo | Paired accuracy on the 231 public items, evaluation only |
| DMB `results/expanded-jev-2026-09-29/*/…-raw.tar.gz` | **No licence** | Do not copy or redistribute. Use the published aggregates only |
| WorkflowEvals `data/run_results.parquet` (contains Jev runs) | invoice: Apache-2.0; other three: licence not stated | Evaluation only, for paired agreement |
| onlyoneaman per-item answers | MIT | Evaluation only |

Global rule, unchanged from `leaderboards/PLAN.md`: no Jev output may ever enter training, checkpoint selection or calibration (registry rule `jev_labelled`).

### 1.5 Contamination audit before any (Z) claim (Azure job, cheap)
1. Run `registry.exclusion_reason` over the tsj `release-audit.json` source list and the b2–b8 source list, using **every** suite source in this document as a pattern. New patterns needed:
   - `ai2_arc|arc_(easy|challenge)`, `commonsense_?qa`, `(allenai/)?art\b|alpha_?nli`, `pubmed_?qa`, `sib-?200`, `belebele`, `afri_?xnli|xnli`, `c-?eval`, `language-identification`, `agb`, `summeval`, `sms_spam`;
   - `contract-?nli`, `cladder`, `esci`, `humicroedit`, `nlupp`, `daily_?dialog`, `tweet_topic`, `enron_spam`, `phishnchips`, `finentity`, `acos`, `ragtruth`, `hover`, `when2call`, `newyorker_caption`, `chaos_?nli`.
2. Run exact, 13-gram and MinHash checks of the shards against the actual eval texts of every suite: Deußer full splits, the DI rebuilt suite, the JevBench public 231, Jevals' 900 items, DMB tests, typed-decisions test, and the WorkflowEvals cases.
3. Publish the overlap table per suite. A (Z) row with any overlap is either dropped or re-labelled "(Z, declared overlap: n items)".

---

## 2. Deußer, Sparrenberg & Sifa: *Evaluating and Benchmarking the System One Model Jev*

### 2.1 Source
- Paper: arXiv 2609.37647, 2026-09-29.
- Code: github.com/AppliedMachineLearning-Lab/jev-benchmarking, MIT, HEAD `6bbdeb33474849b6de2f0cccc9f5e19756abd67e` (2026-09-30). This is the same commit as our local copy in `bench/sources/deusser/` (`COMMIT` file).
- Files our copy lacks:
  - `jev_benchmarking/{cache,evaluate,thresholds}.py`
  - `tasks/probes.py`, `tasks/bigbench_mc_tasks.json`
  - `scripts/{run,evaluate,import_responses}.py`
  - `results/eval/*.json`
  
  Copies are in this session's scratchpad; refetch on the VM at the same commit.
- Jev run [F: `docs/datasets.md`]:
  - `jev-1.13.0` pinned; eval run completed 2026-09-27;
  - 346,009 requests, 217.9 M input tokens, $9.15;
  - 1,100 req/min client cap, 630 input tokens per request on average, 0.36 s mean latency;
  - one BIG-bench item rejected with HTTP 400 `max_tokens_exceeded`.

### 2.2 Protocol (all 37 tasks)
- **One request per example**, with `state` = the example. All of the task's questions go in that one request (`tasks/base.py` docstring).
- **Sampling:** the full eval split. Rows are hash-ordered by `sha256("<task>/<uid>")`, with round-robin over configs. `max_per_config` = 500 for BIG-bench only.
- **Option keys:** MC tasks use `A, B, …` (`AA, AB, …` above 26). Classification tasks use the label names as keys.
- **Revisions:** HF revisions are **not pinned** (`load_dataset` without `revision`). [D] Pin each dataset to its last commit on or before 2026-09-26, and assert `n_examples` equals Jev's (table below).
- **Metrics** (`metrics.py`):

  | Kind | Metrics |
  |---|---|
  | choice | accuracy of `a["choice"]`; ECE with 15 equal-width bins on top-p; Brier; selective accuracy at 50/80% coverage by `a["confidence"]`; macro-F1 when all examples share a label set |
  | binary | threshold 0.5 (`p >= 0.5`); acc, balanced acc, F1, AUROC, AUPRC, ECE, Brier |
  | score | Spearman / Pearson / Kendall of gold vs `a["score"]` (the expected value); `argmax_accuracy` over int keys; SummEval correlates per document group, then averages |
  | multilabel | micro/macro-F1 at 0.5, exact match, mean AUROC/AUPRC over labels that have both classes |
  
  CI: bootstrap with 500 resamples, seed 0.
- **Primary metrics and Jev values** [F: `results/eval/summary.md` and `results/eval/*.json`].
  - Code paths are in `bench/sources/deusser/jev_benchmarking/tasks/*.py`. Each task's state keys, question ids, types and criteria style come from its `build()`.
  - Abbreviations used in the table:
    - **ch** = choice, **no** = noul, **sc** = score;
    - **D** = described options (one-sentence descriptions);
    - **N** = `null` values;
    - **MC** = option texts under letter keys.

  | # | Task (class) | HF id / config / eval split | n | State keys | Questions (type, criteria) | Primary | Jev | Notes |
  |---|---|---|---|---|---|---|---|---|
  | 1 | ag_news (`AGNews`) | fancyzhx/ag_news / test | 7,600 | article | answer: ch 4 D | acc | **0.885** (CI .878–.891) | |
  | 2 | imdb | stanfordnlp/imdb plain_text / test | 25,000 | review | ch 2 D (`SENTIMENT_2`) | acc | 0.965 | v2 contaminated (tsj imdb) |
  | 3 | rotten_tomatoes | cornell-movie-review-data/rotten_tomatoes / test | 1,066 | review | ch 2 D | acc | 0.933 | |
  | 4 | sst2 | stanfordnlp/sst2 / validation | 872 | review | ch 2 D, same "movie" instruction | acc | **0.964** | A choice, not a noul |
  | 5 | emotion | dair-ai/emotion `split` / test | 2,000 | text | ch 6 N (order sadness, joy, love, anger, fear, surprise) | acc | **0.585** (mF1 .499, ECE .279) | |
  | 6 | financial_phrasebank | atrost/financial_phrasebank / test | 970 | sentence | ch 3 D (investor impact) | acc | **0.730** | |
  | 7 | banking77 | mteb/banking77 / test | 3,076 | query | ch 77 N, keys = `sorted(set(test label_text))` | acc | **0.797** (mF1 .788, sel@50 .963) | |
  | 8 | clinc150 | clinc/clinc_oos `plus` / test | 5,500 | utterance | ch 151: 150 N in ClassLabel order, then `oos` with a description | acc | **0.895** (mF1 .902; in-scope .921; OOS P .941 / R .776) | |
  | 9 | sib200 | Davlan/sib200, 205 configs / test | 41,820 | text | ch 7 N | acc | 0.815 | Broken config `nqo_Nkoo.zip` skipped |
  | 10 | language_id | papluca/language-identification / test | 10,000 | text | ch 20 D (code → language name) | acc | 0.996 | |
  | 11 | sms_spam | ucirvine/sms_spam plain_text / train (only split) | 5,574 | message | no, no criteria | F1 | 0.938 | Eval-only; likely v2-contaminated |
  | 12 | go_emotions | simplified / test | 5,427 | comment | 28 × no, no criteria (27 "express X" + a neutral variant) | macro-F1 @0.5 | **0.243**; thresholds tuned on 1,000 train rows: **0.353** (micro .239 → .387) | |
  | 13 | anli | facebook/anli plain_text / test_r1..r3 | 3,200 | premise, hypothesis | ch 3 D (`NLI_CRITERIA`) | acc | **0.739** | |
  | 14 | afrixnli | masakhane/afrixnli, 18 configs / test | 10,800 | premise, hypothesis | ch 3 D | acc | 0.640 | |
  | 15 | paws | labeled_final / test | 8,000 | sentence_1, sentence_2 | no with true/false | acc | **0.850** | |
  | 16 | llm_aggrefact | lytang/LLM-AggreFact (gated) / test | 29,320 | document, claim | no with true/false | **mean of per-source balanced acc** (11 sources) | **0.786** (pooled acc .828) | Card forbids training: Z only |
  | 17 | boolq | google/boolq / validation | 3,270 | passage, question (capitalised, "?") | no, no criteria | acc | **0.913** | |
  | 18 | belebele | facebook/belebele, 122 configs / test | 109,800 | passage, question | ch 4 MC | acc | 0.867 | No train split |
  | 19 | pubmedqa | qiaojin/PubMedQA pqa_labeled / train (the 1,000) | 1,000 | question, abstract (contexts joined by `\n`) | ch 3 D yes/no/maybe | acc | 0.787 | Never train on pqa_labeled |
  | 20 | mmlu | tasksource/mmlu, 57 configs / test | 14,042 | subject, question | ch 4 MC | acc | 0.918 | |
  | 21 | ceval | ceval/ceval-exam, 52 configs / test | 12,342 | subject, question | ch 4 MC | acc | 0.839 | |
  | 22 | bigbench | tasksource/bigbench, 93 tasks / validation, ≤500 per task | 13,228 | prompt | ch MC, gold can be a set | acc | 0.814 (13,227 answered) | |
  | 23 | hellaswag | Rowan/hellaswag / validation | 10,042 | context (lm-eval cleaning) | ch 4 MC | acc | 0.955 | |
  | 24 | winogrande | winogrande_xl / validation | 1,267 | sentence | ch 2 MC | acc | 0.914 | |
  | 25 | arc | allenai/ai2_arc Challenge + Easy / test | 3,548 | question | ch, dataset labels as keys | acc | 0.988 | |
  | 26 | commonsense_qa | tau/commonsense_qa / validation | 1,221 | question | ch 5 | acc | 0.882 | |
  | 27 | art | allenai/art `anli` / validation | 1,532 | observation_1, observation_2 | ch 2 MC | acc | 0.839 | |
  | 28 | toxigen | toxigen/toxigen-data `annotated` (gated) / test | 940 | text | toxic: no; toxicity: sc 5 | acc(toxic), gold = ai + human > 5.5 | **0.878** | |
  | 29 | openai_moderation | mmathys/openai-moderation-api-evaluation / train (only) | 1,680 | text | 8 × no with **object instructions** {category{name, definition}, question} | mean AUPRC | **0.717** (micro-F1 .608) | Z only |
  | 30 | toxic_chat | lmsys/toxic-chat toxicchat0124 / test | 5,083 | prompt | toxic, jailbreak: no | F1(toxic) | **0.786** | |
  | 31 | prompt_injections | deepset/prompt-injections / test | 116 | text | no | acc | 0.741 | |
  | 32 | agb_de | d4br4/agb-de / test | 755 | title, clause | no (German law) | F1 | 0.204 | |
  | 33 | unfair_tos | coastalcph/lex_glue unfair_tos / test | 1,607 | clause | 8 × no with **object instructions** {clause_type{name, definition}, question}; snake_case ids | micro-F1 | **0.499**; tuned thresholds **0.748** (macro .577 → .739) | |
  | 34 | stsb | sentence-transformers/stsb / test | 1,379 | sentence_1, sentence_2 | sc 6 levels; gold = score × 5 | Spearman vs `score` | **0.890** | |
  | 35 | sst5 | SetFit/sst5 / test | 2,210 | sentence | sc 5 levels | argmax acc | **0.579** (Spearman .851) | |
  | 36 | summeval | mteb/summeval / test (100 docs × 16) | 1,600 | source_document, summary | 4 × sc 5 levels, **object instructions** {criterion, question} | mean per-document Spearman | 0.554 | Z only |
  | 37 | helpsteer2 | nvidia/HelpSteer2 / validation | 1,038 | prompt, response | 5 × sc 5 levels | mean Spearman | **0.412** | |

  Calibration and selective accuracy (paper): pooled Choice ECE 0.028 over 22 datasets; selective accuracy at 50% coverage — Banking77 0.963, SIB-200 0.982, ANLI 0.869, Emotion 0.733.

### 2.3 Protocol traps (non-comparability)
- **Thresholds.** The headline Noul numbers use a fixed 0.5 threshold. Jev's per-question thresholds tuned on 1,000 train rows (GoEmotions, UNFAIR-ToS, AGB-DE, ToxicChat; `thresholds.py`, `results/eval/thresholds.json`) are an S-like Jev number: compare our (S) runs to those. Our (Z) runs are compared to the 0.5 numbers.
- **Missing answers.** `evaluate()` scores only answered rows, so a refusal raises the score. Require 100% coverage (W5 truncate mode, disclosed) or report missing = wrong.
- **Score primitive.** Spearman uses the expected value `score`, not the argmax. Argmax accuracy (SST-5) uses the int keys.
- **AggreFact headline.** It is the per-source mean of balanced accuracy, skipping sources where only one class is present. The leaderboard (llm-aggrefact.github.io) uses the same definition, so it is directly comparable to FactCG-DeBERTa-L 0.756 and MiniCheck-FT5-L 0.750.
- **Pinning.** Revisions are not pinned upstream. If a dataset changed after 2026-09-27, our n or rows can drift: assert n.

### 2.4 Adapter spec: `scripts/benchmax/deusser_run.py` (Azure; never the Mac)
1. On the VM, clone `jev-benchmarking` at `6bbdeb33…` (MIT, keep the notice) and `pip install -e .` without the `openmodel` extra. Pin `datasets` revisions as in §2.2.
2. For `task in TASKS.values()` and `e in task.examples("eval")`:
   - compute `key = cache.request_key("meharsjev-68m", e.state, e.questions)`;
   - call `resp = engine.system_one({"state": e.state, "questions": e.questions})` (W1–W3, W6, W8);
   - append `{"key", "task", "response": {"model": "meharsjev-68m", "answers", "usage"}, "latency_s"}` to a JSONL file, sharded by `int(key, 16) % N` across cluster nodes.
3. Run `python scripts/import_responses.py runs/*.jsonl`. This writes `cache/responses.meharsjev-68m.db` and refuses `jev-*` models.
4. Run `python scripts/evaluate.py all --split eval --model meharsjev-68m`. Output goes to `results/eval/open_models/meharsjev-68m/*.json` and `summary.md`, with the same code, metrics and CIs as Jev's `results/eval/summary.md`.
5. (S) threshold variant: run the same loop on `--split dev --limit 1000` for GoEmotions, UNFAIR-ToS, AGB-DE and ToxicChat. Then apply `thresholds.tune()` and `thresholds.apply()`, and compare with `results/eval/thresholds.json["Jev"]`.

Notes:
- Request bytes are identical to Jev's because the task code is identical. A key match against Zenodo would prove it, but we are not using that DB (§1.4).
- Cost: about 346k requests at roughly 630 Jev tokens each. On 12× F80 CPU nodes that is hours, not days [E].

### 2.5 (S) eligibility (train splits)
- **Train or dev splits exist:** AG News, IMDB, RT, SST-2, Emotion, FPB (validation), Banking77, CLINC, SIB-200, language-id, GoEmotions, ANLI, AfriXNLI (validation), PAWS, BoolQ, MMLU (`auxiliary_train`), C-Eval (val/dev), BIG-bench (train), HellaSwag, WinoGrande, ARC, CSQA, αNLI, ToxiGen, ToxicChat, prompt-injections, AGB-DE, UNFAIR-ToS, STS-B, SST-5, HelpSteer2.
- **Z only:** SMS spam, OpenAI moderation, SummEval and Belebele (no train split); AggreFact (card forbids training); PubMedQA (`pqa_labeled` is the eval; S may use `pqa_artificial` only).
- **Sibling traps:**
  - SST-2, SST-5 and RT share Pang & Lee sentences;
  - HelpSteer2 train and validation can share prompts [I: check before S];
  - Banking77 train has 2 rows that duplicate test rows [F: DI notes].

---

## 3. Decision Index 0.2.1

### 3.1 Source
- Kit: github.com/apolinario/decision-index, MIT, `87d4650b42b377c0291a89c1f1a879f9b31082bf` (2026-09-27, the last commit as of 2026-10-02).
- Board: HF Space `multimodalart/jev-decision-index`, `data/index-v0.2.1.json`, generated 2026-09-28T00:39Z. Jev's index is **57.91** there. The kit fixture gives **57.89** (generated 2026-09-27T02:14Z; raw index 68.08, breadth 57.07).
- Jev run [F: methodology `edition`]: `jev-1.13.0`, "Full release v1", HTTPS hosted API; $12.26 for the suite; 291.8 M input tokens.
- Jev per-request responses are **not public**. Only per-benchmark results are.

### 3.2 Protocol
- **Suite.** Rebuilt from pinned sources, byte-identical to the lab's files:
  - `selected-rows.jsonl` uncompressed sha256 `b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5`;
  - `added-rows.jsonl` `7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89`;
  - exclusions `331df32d4b719c7db43214d0e5d85859d39c3b2eb7d0b3812214cce150155e81`.
  
  Counts: 120,340 requests in 0.2.1 (119,898 scoreable after the 442 exclusions) + 30,419 added = 150,317 scoreable.
- **Sampling.** Hash order `sha256("20260919:<catalog_id>:<group_id>")`.
  - Caps: POP909 2,000; ChessBench 5,000; SGD 2,500; cfcolor 5,000; CLadder 5,000; ESCI 5,000 (locale × relevance strata).
  - Request budgets: RouterBench 10,000; BPoMP 5,000.
  - 0.2.1 subsets: ToolRet 685 and BRIGHT 220 answerable queries; Home appliances 88 rows; ACOS 400 reviews (1,565 rows).
- **The engine receives exactly `state` and `questions`.** Requests are never edited.
- **Request shapes** (load-bearing, and different from every other suite):

  | Benchmarks | State | Where the input is | Options |
  |---|---|---|---|
  | BANKING77, CLINC150, ANLI, MMLU, ARC-E/C, WinoGrande, HellaSwag, MuSR, SimpleBench (`normalize_direct.choice_row`); VAST, CLadder (`normalize_text._mc_row`); GPQA, HLE, Humicroedit | `{}` or `""` | **instructions** = fixed task line + the text | Keys `A..Z`, or **`option_0..option_N` when there are more than 26 options** (Humicroedit uses `headline_1/2`). Values = option text. BANKING77 values are raw `snake_case` names in `categories.json` order. CLINC values are labels with spaces, plus an OOS description, sorted |
  | MMLU-Pro, BBH | question or input string | generic instructions (`CHOICE`) | MMLU-Pro `A..J`. BBH options parsed from its own `Options:` block, keys `(A)`… or the option texts; fixed sets for 3 tasks |
  | ContractNLI | contract text | one question per hypothesis (17) | 3 described |
  | NLI4CT | `{primary_trial{id, section, text}, secondary_trial?}` | statement inside instructions | 2 described |
  | FinEntity | `{document, task}` | one question per entity span (key `entity_####_s_e`) | `Negative/Neutral/Positive`, value = key |
  | ACOS | `{review, task}` | ≤64 yes/no questions per row (category × sentiment) | yes/no described |
  | iSarcasmEval | text (track A) | one yes/no question per label | `{"no":"No","yes":"Yes"}` |
  | Amazon ESCI | `{search_query, product{title, description, bullet_point, brand, color}}` | fixed instruction | 4 described E/S/C/I |
  | RAGTruth | `{prompt, response}` | **noul** with true/false criteria | — |
  | HoVer | `{claim, evidence[{title, text}]}` (oracle intros) | fixed | SUPPORTED / NOT_SUPPORTED described |
  | When2Call | `{tools, question}` | fixed | `A..D` in published order |
  | New Yorker | `{scene, description, uncanny_description, entities}` | fixed | `A..E` captions |
  | PhishNChips | email `{sender, from, subject, body, link_display_text, link_url}` | **nine questions** from `anisselbd/jev-phishing-bench@1d56e8c run_jev.py`; only `verdict` is scored | — |
  | ToolRet, BRIGHT | query | one question per candidate, **object instructions** `{task, candidate}` | yes/no |
  | BFCL, API-Bank, SATA, GSM8K, CRUXEval, ChessBench, POP909, cfcolor, Habermas, RouterBench, ForecastBench | various (`adapters_*.py`) | multi-question for BFCL and SATA | — |

- **Scoring (`scoring/report.py`, `index.py`, `index02.py`, `added.py`):**
  - **Coverage first:** `raw = native × answered/requests`. Unsupported, errored and abstained requests count as wrong. Linked cases count only when every request in them succeeded.
  - **Prediction rule:** choice uses `a["choice"]`; noul uses `p ≥ 0.5`.
  - **F1 benchmarks** (4, 5, 10, 11, 12, 37, 39, 40, 41, 42) compare **semantic labels** (option descriptions). "Conservative" F1 (ContractNLI, ESCI, VAST, iSarcasm A) adds the number of missing predictions to every class's denominator.
  - **Skill:** `skill = clip((raw − chance)/(1 − chance), 0, 1)`. ForecastBench instead uses `clip((0.25 − Brier)/0.25) × coverage`.
  - **Areas:** gold benchmarks weigh 1.2 inside their area. Area weights: Knowledge 25.8%, Language 25.8%, Retrieval 20.0%, Tools 18.3%, Arts 10%.
  - **Index** = 100 × the weighted mean of the area skills. Scores within 0.25 are ties.
- **Hard rules** (README "Rules", `docs/engines.md`):
  - no truncation (refuse instead);
  - no option filtering;
  - one fixed rendering, with no per-benchmark prompt, calibration or temperature, and no benchmark detection;
  - no correctness-based retries;
  - complete, untouched runs;
  - median latency ≤ 1,000 ms on one RTX PRO 6000, measured by the maintainers.

### 3.3 Jev's targets per benchmark
[F: kit fixture `board-0.2.1.json`, entrant `jev`; chance from `data/index-0.2.1.json`]. ★ = gold benchmark (weight 1.2 in its area).

| Area | Benchmark (id) | Metric | Chance | **Jev raw** | Jev skill | (S) train split? [F/I] |
|---|---|---|---|---|---|---|
| Knowledge | GPQA Diamond (25) ★ | acc | .25 | **.786** | .714 | none; GPQA main/extended contain Diamond: forbidden |
| | GSM8K 4/10-choice (30) | acc per track | .175 | **.799** | .757 | train; beware the gold-rank shortcut (issue #32) |
| | ChessBench (31) | best-move acc | .082 | **.172** | .098 | searchless_chess train |
| | MuSR (32) | acc | .371 | **.660** | .460 | none |
| | SATA-Bench (33) | case exact | .013 | **.264** | .254 | none |
| | CRUXEval (43) | acc | .370 | **.730** | .571 | none |
| | CLadder (44) | acc | .5 | **.726** | .453 | remainder of `cladder-v1` [I, risky]; v2 already contains tsj cladder |
| | HLE (45) ★ | acc | .164 | **.201** | .044 | none (gated) |
| | MMLU-Pro (57) ★ | acc | .111 | **.827** | .806 | validation 70 only; MMLU aux_train allowed |
| | BBH (58) ★ | acc | .310 | **.929** | .897 | none (BIG-bench canary: never train) |
| Language | ContractNLI (11) | cons. macro-F1 | .309 | **.717** | .591 | train/dev (v2 contains tsj contract-nli) |
| | ANLI (12) ★ | macro-F1 | .332 | **.748** | .622 | train r1–r3 |
| | WinoGrande (28) ★ | acc | .5 | **.920** | .839 | train_xl |
| | HellaSwag (29) ★ | acc | .25 | **.945** | .927 | train |
| | ACOS (38) | per-review F1 | .031 | **.295** | .273 | train |
| | FinEntity (39) | macro-F1 | .320 | **.870** | .809 | **none**: all 979 docs are the suite |
| | iSarcasmEval (40) | sarcasm F1, A-En | .223 | **.505** | .363 | train |
| | VAST (41) | cons. macro-F1 | .333 | **.646** | .470 | train/dev |
| | NLI4CT (42) | macro-F1 | .486 | **.841** | .690 | train/dev |
| | RAGTruth (59) | F1 (hallucinated) | .518 | **.765** | .514 | train split (MS MARCO contexts NC) |
| Retrieval | BANKING77 (4) ★ | macro-F1 | .013 | **.797** | .795 | train (2 duplicates of test rows) |
| | CLINC150 (5) ★ | macro-F1 | .006 | **.893** | .892 | train/val |
| | BRIGHT (36) ★ | nDCG@10 | .116 | **.475** | .406 | none |
| | Amazon ESCI (37) | cons. macro-F1 | .203 | **.552** | .438 | train (v2 contains tsj esci) |
| | PhishNChips (56) | acc (verdict) | .5 | **.626** | .251 | **none**: all 2,000 core emails |
| | HoVer (61) | acc | .5 | **.729** | .457 | train |
| Tools | BFCL (1) ★ | case exact | .259 | **.958** | .943 | none |
| | ToolRet (2) | nDCG@10 | .134 | **.653** | .599 | [I] |
| | API-Bank (3) ★ | acc | .019 | **.882** | .880 | [I] |
| | Home appliances (9) | case exact | 0 | **.523** | .523 | dev generator [I] |
| | When2Call (62) | acc | .25 | **.810** | .746 | train_sft / train_pref |
| Arts | BPoMP (20) | acc | .5 | **.909** | .818 | [I] |
| | Humicroedit (21) | pairwise acc | .5 | **.619** | .237 | train (v2 contains tsj humicroedit) |
| | POP909 (22) | song-macro acc | .008 | **.166** | .159 | other songs [I] |
| | cfcolor (23) | user-macro acc | .5 | **.644** | .288 | other users [I] |
| | ForecastBench (48) ★ | Brier vs 0.25 | — | **.306** (Brier ≈ .174) | .306 | questions resolved before 2026-07-01 [I] |
| | Habermas (50) | group-pref acc | .311 | **.459** | .215 | train split [I] |
| | New Yorker (64) | acc | .2 | **.701** | .626 | matching train |

Shown but not counted: MMLU, ARC-E, ARC-C, SimpleBench, RouterBench, SGD.

### 3.4 Adapter spec (Azure)
1. On a VM: `pip install -e ".[rebuild]"`. Then `python -m decision_index suite rebuild --work work` followed by `suite import`.
   - Needs about 7 GB of downloads and 17 GB of working space.
   - The **user** must accept the HLE gated terms and log in to HF.
   - Run under Python 3.12: RouterBench's float summation reproduces the pinned hash only on 3.12 [F: autotrust results card].
2. Serve our engine in leaderboard mode (W1–W7, W9, W10; overflow = refuse) and run `python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8765 --option model=meharsjev-68m --out runs/meharsjev-68m`.
   - Alternative: an in-process `Engine` subclass that raises `Unsupported` past the window or past 255 options.
3. Run `score --edition 0.2.1` and compare `benchmark-summary.json` (native, on complete groups) and `index.json` (raw, skill, coverage) with §3.3.
4. Report per-benchmark wins against Jev's raw values, plus the index. A 68M (Z) index will be far below 57.89 (knowledge area) [E]. That is why the per-benchmark targets are the point.

### 3.5 Traps
- **Input in instructions.** Most of our training data puts content in `state`, so the "state `{}`, input in instructions" shape is out of distribution. Format alignment in training is allowed; precedents are Dinah-0 and JevK5.
- **Opaque keys.** Options must be read through their descriptions when keys are `option_i` or `A..`.
- **Long rows.** About 1,681 rows exceed 8,192 tokens and count as wrong unless we can read them: ContractNLI contracts, HoVer evidence, NLI4CT, BRIGHT documents, API-Bank and ToolRet catalogs [I].
- **Coverage gap with Jev.** Jev's coverage is 1.0 everywhere. The lab logged "context splits" separately for Jev [F: methodology release_notes].
- **GSM8K gold-rank shortcut** (issue #32): run the question-blind check.
- **(S) entries.** The policy is to declare overlap; test duplicates count as wrong; one row per project.

---

## 4. JevBench v1.5.5 (Benchmark Heaven)

### 4.1 Source
- Repo: github.com/fstandhartinger/jevbench, MIT, HEAD `bb05a335bc80` (2026-09-29).
- Method: `docs/METHOD-v1.5.md`, frozen 2026-09-25, sha256 `c25d3d8b…`, plus the headline amendment `METHOD-v1.5-ADDENDUM-HEADLINE-A-EQUAL-TYPES.md`.
- Live result: `https://benchmarkheaven.com/api/jevbench/v1.5.5`, revision v1.5.5, released, official, read 2026-10-02.

### 4.2 Jev's numbers
[F: API, system `jev-1.13.0`, adapter `jevbench.adapters.typesafe.TypeSafeAdapter`, API flag set]

| Item | Value |
|---|---|
| Composite (headline A) | **72.13**, CI [71.01, 72.61], rank 3 |
| Axes | Intelligence **72.00**, Calibration **88.03**, Speed **83.81**, Cost **54.73** |
| Intelligence split | I_open 71.55, I_sealed 72.45; gap −0.90; penalty 1 |
| Choice CC | open 85.73, sealed 87.59 |
| **Noul CC** | open **47.75**, sealed **48.62** (abstention rate **22.8%**) |
| Score CC | open 81.16, sealed 81.14 |
| Calibration parts | Choice ECE_hard .027, mean TVD .316; Noul ECE .056, Brier .070; Score nRPS .041 |
| Speed | p50 0.616 s |
| Cost | $0.0323 per 1k decisions |
| G_med | 5.1866 |

Sample: 904 open (601 published) + 720 sealed = 1,624 decisions.

### 4.3 Protocol
[F: `METHOD-v1.5.md`, addendum, `jevbench/adapters/{base,typesafe}.py`, `jevbench/scoring.py`]
- **Requests.** One decision per request: `{"state", "model", "questions": {"decision": {type, instructions, criteria?}}}`.
  - Criteria: a dict for choice, `{true, false}` for noul, a list of levels for score.
  - A single question per request; serial for the speed measurement.
- **Validity.** Exact label keys are required. `SUM_TOL` 1e-3 is the strict rule; inside `RENORM_TOL` 2e-2 the distribution is renormalized. Invalid answers count as wrong.
- **Choice.** Argmax, with ties broken by the **lexicographically smallest label**. CC = 100·(acc − c̄)/(1 − c̄), where c̄ is the mean of 1/K over the tier.
- **Noul.** P ≤ 0.20 is No, P ≥ 0.80 is Yes, anything else is an **abstention counted wrong**. CC = 100·(acc − 0.5)/0.5.
- **Score.** CC = 100·(1 − mean nMAE / mean nMAE_chance), where nMAE uses the **expected position** divided by (K − 1).
- **Aggregation.**
  - Tier weights: easy .10, standard .20, judge .30, hard .40.
  - Type weights: **⅓ each** (headline A amendment).
  - base = 0.5·I_open + 0.5·I_sealed.
  - Overfit penalty when the gap exceeds G_med + 8.
- **Composite.** Weighted harmonic mean with 25% per axis (headline A). Each of I, S and Cost below 50 multiplies the composite by (x/50)².
  - Speed = 100 − 20·log10(s/0.1 s), using the mean of p50 and p95. Self-hosted runs are adjusted ×2 + 0.15 s.
  - Cost = 100 − 30·log10(($/1k)/0.001).

### 4.4 What can and cannot be reproduced
- **Sealed half:** cannot be reproduced. The evaluator alone holds the 720 sealed items.
- **Public items in git:** only **231** (`datasets/public/{easy 48, original 72, hard 111}.jsonl`; v1.2 split sha256 in `datasets/manifest.json`). The other ~370 "published" v1.5 items (120 drafts + 250 pool draws) were **not found** in the repo (any branch or tag), on HF, or in the API as of 2026-10-02 [F: tree listing, branches, tags, HF search, site]. So I_open cannot be recomputed either.
- **Offline proxy [D]:** run the 231 items with the v1.5 per-item rules (§4.3) and report "JevBench-public-231 (v1.5 rules, proxy)".
  - The only per-item Jev reference is v1.3.0 accuracy outcomes (c/w) for these 231 items (`results/v1.2/jevbench-v1.2-per-task.json`, key `jev-1.13.0`). Those use the v1.3 scoring (argmax, no noul band), so the comparison is paired **accuracy** only.
  - Jev's tier totals (including held-out items): easy 72/72, standard 95/96, judge 138/146, hard 163/220.
- **The comparable number comes only from a maintainer run.** File `[bench request]: meharsjev-68m …` once the weights are public (free to us; see `leaderboards/PLAN.md` §2.3). The v1.5 scorer used by the board is not in the public repo (only `composite_v14.py`) [I]. For the proxy, implement §4.3 ourselves and unit-test it against the method text.

### 4.5 What beating 72.13 requires [E]
- Jev's weak axes are Cost (54.7) and Speed (83.8).
- With Calibration 85, Speed 95 and Cost 95 (CPU encoder, self-host adjustment applied), the composite is:

  | Intelligence | Composite |
  |---|---|
  | 55 | 78.4 |
  | 50 | 75.7 |
  | 48 | 68.7 (gate (I/50)² applies) |
  | 45 | 58.9 (gate applies) |

- So **Intelligence ≥ 50 is the bar for beating Jev's composite**, not Intelligence ≥ 72.
- Jev's Noul CC is only about 48 because 22.8% of its nouls fall in the 0.2–0.8 abstention band. A committed noul (global sharpening, or a separate yes/no temperature; precedent issue #141) is allowed, and Noul carries ⅓ of Intelligence.
- Shipping a sharpened variant is a disclosed, global config. It trades against the Calibration axis and against typed-decisions KL (§8).

### 4.6 (Z)/(S)
- Training or selection on the public items is allowed but must be disclosed (gap penalty applies). Sealed items must never be touched.
- **The open set is scored**, so (S) training on the 231 inflates I_open only. [D] Run once and disclose.

---

## 5. AbdelStark/jev-benchmarks: BTZSC pilot v1

### 5.1 Source
- github.com/AbdelStark/jev-benchmarks, Apache-2.0, `0d610cc53e79` (2026-09-17).
- Protocol `docs/PROTOCOL.md` (frozen, tag `pilot-v1-preregistered`); config `configs/pilot-v1.yaml`.
- Report `results/reports/btzsc-pilot-v1.md`. Jev resolved to `jev-1.13.0` via `jev-latest`, 2026-09-17.

### 5.2 Jev's numbers
100 items per dataset.

| Dataset | Acc | Macro-F1 | Brier | NLL | ECE (10 bins) | Coverage at ≤5% error |
|---|---|---|---|---|---|---|
| AG News | **0.910** | 0.905 | 0.146 | 0.495 | 0.064 | 0.830 |
| Banking77/BTZSC (72 labels) | **0.870** | 0.857 | 0.179 | 1.064 | 0.054 | 0.860 |
| DAIR Emotion | **0.480** | 0.479 | 0.846 | 5.588 | 0.351 | 0.000 |

On emotion Jev gave the true label probability exactly 0 on 16% of items.

### 5.3 Protocol
- **Data:** `btzsc/btzsc` @`fef2a2ac62b69c58670047dddf045c53d7c3cb5e`, test configs `agnews`, `emotiondair`, `banking77`. This is the same revision as our `templates.BTZSC_REVISION`.
- **Classes:** the class count comes from repeated texts. Labels = the first n_classes `hypothesis` strings, in BTZSC order.
- **Banking77 exclusions:** rows with no positive hypothesis are dropped (200 out-of-scope rows). That leaves 72 labels.
- **Sampling** (`data._balanced_indices`):
  - seed = 20260917 + dataset offset (agnews 0, emotiondair 1, banking77 2);
  - shuffle each class's index list with one `random.Random(seed)`, in order of each class's first appearance (dict insertion order; this order changes the RNG stream);
  - round-robin over sorted class ids, `pop()` from the end, until 100 items;
  - output sorted.
- **Request** (`adapters/jev.py`): state `{"text": text}`; one Choice question `label` whose criteria are `{"label_000": hypothesis_0, …}`; the instruction string comes from `configs/pilot-v1.yaml` (`models.jev.question`).
- **Prediction:** first max index. Vectors summing to 0.99 because of rounding are renormalized.
- **Metrics** (`metrics.py`):
  - macro-F1 over **all** n_classes (absent classes score 0);
  - multiclass Brier as a sum of squares;
  - NLL with clip at 1e-12;
  - top-label ECE with 10 bins (last bin closed);
  - coverage at error ≤ 0.05 selected on the same slice;
  - paired target-stratified bootstrap with 2,000 resamples.

### 5.4 Adapter
1. Clone at `0d610cc5` and `uv sync`.
2. Point `typesafe_sdk.TypeSafeClient` at our server with `TYPESAFE_BASE_URL=http://127.0.0.1:8765`, and set `models.jev.model_id: meharsjev-68m` in a copied config named `pilot-v1-meharsjev.yaml`.
   - Since we don't touch the `jev` backend name, document the backend label in the report.
   - Our server ignores auth on localhost.
3. Run prepare, run and report.
4. **Verify:** the `manifest.jsonl` sha256 must equal `ec064c52b149de458344cd4b4a44c158460f30b3bbb7fe8b2e7ec72d0abf3ba5`. Rows are written as `json.dumps(row, ensure_ascii=False, sort_keys=True)` per line. A match proves the identical 300 items.

### 5.5 Traps and tracks
- With n = 100, one item is 1 point. Report the paired bootstrap CI, not just the point.
- 72-label Banking77 is easier than 77-label; never mix it with Deußer's or the Decision Index's Banking77.
- (Z): BTZSC itself is in our registry (`sibling:btzsc`), and our registry excludes AG News, emotion and Banking77.
- (S): Banking77, AG News and emotion train splits are allowed for S.

---

## 6. Jevals (jevals.com), release 2026-09-18, suite 0.1.0

### 6.1 Source
- Data: github.com/Jevals/jevals-data, CC BY 4.0, `21bb47b72814` (2026-09-21): `suites/0.1.0/*.json`, `releases/2026-09-18/board.json`, `runs/<system>__<task>__0.1.0.jsonl`.
- Methodology: `https://jevals.com/methodology/` (read 2026-10-02).
- Jev harness commit `6d93f79a2310`. Jev was called as `typesafe-ai/jev` through Vercel AI Gateway with the AI SDK's `experimental_evaluate` (`ai@7.0.106`), SDK retries disabled. **The harness code is not public.**

### 6.2 Jev's numbers
[F: board.json]

| Task (primitive) | Decision Score [95% CI] | Acc | ECE (10 bins) | Repeat flip | Order flip | Gate t / coverage@t / acc@t |
|---|---|---|---|---|---|---|
| Banking77 (choice) | **67.78** [61.34, 74.49] | **0.7967** | .0981 | .0267 | .1033 | .96 / .595 / .943 |
| PubMedQA (noul) | **69.03** [59.77, 76.55] | **0.9127** | .0504 | 0 | — | .91 / .49 / .986 |
| HelpSteer2 helpfulness (score) | **9.20** [−4.37, 20.97] | **0.4127** | .1966 | .0233 | — | none |

Comparator: Gemini 3.8 Flash 74.11 / 72.98 / 4.59.

Jev's raw losses L_sys are Banking77 0.3180, PubMedQA 0.1459 and HelpSteer2 0.1489. With Decision Score = 100·(1 − L_sys/L_prior), that gives L_prior ≈ 0.987, 0.471 and 0.164 [I: derived]. Our scorer must reproduce these priors from the 300 items before our Decision Score counts. Every Jev row has n_answered 1,500, schema_valid 1 and refusal 0. p50 latency is 467 / 438 / 478 ms.

### 6.3 Protocol
- **Items.** The suite files list 300 items per task, each with `row_idx`, `state_sha256` and `target`.

  | Task | Source @ revision | Split | Items | Class counts |
  |---|---|---|---|---|
  | Banking77 | mteb/banking77 @`18072d2685ea682290f7b8924d94c62acc19c0b2` | test | 300 | proportional |
  | PubMedQA | qiaojin/PubMedQA pqa_labeled @`9001f2853fb87cab8d220904e0de81ac6973b318` | train (the 1,000 labelled) | 300 | no 114, yes 186; **maybe excluded** |
  | HelpSteer2 | nvidia/HelpSteer2 @`990b2711a36180dd19d9c94b8627844866f8982a` | validation | 300 | 0:23, 1:27, 2:35, 3:90, 4:125 |

  Seed 20260918; proportional largest-remainder allocation. Items whose state exceeds 6,000 code points were dropped before sampling.
- **State** is a JSON object of whitelisted `state_fields`:
  - Banking77: `["text"]`;
  - PubMedQA: `["question", "context.contexts"]`;
  - HelpSteer2: `["prompt", "response"]`.
  
  Instructions and criteria are in each suite file:
  - Banking77: bare names → `null`;
  - PubMedQA: noul with true/false;
  - HelpSteer2: 5 described levels.
- **Repeats.** Each item is sent 5 times. Epochs 0 and 1 use the same order (`order_seed` 0); epochs 2–4 use seeds 1–3. For choice, the option order is `shuffled(options, fnv1a("<item_id>:<order_seed>"))` with mulberry32 (code in the data README).
- **Scoring** (methodology):
  - **Decision Score** = 100·(1 − L_sys/L_prior). L is the per-item loss averaged over the 5 repeats, then over the 300 items: multiclass Brier for choice and noul, RPS = Σ_{k<K}(P_k − Y_k)²/(K − 1) for score. L_prior = the label prior of the **evaluated** items.
  - **Accuracy:** pick equals label; noul ties at 0.5 count wrong; score uses the modal level. Refused or malformed answers count wrong.
  - **ECE:** 10 bins, `min(9, floor(round(100·c)/10))`.
  - **Gate t:** the smallest confidence on the 0.01 grid with pooled error ≤ 5% and ≥ 100 decisions.
  - **CI:** item-cluster bootstrap, 2,000 resamples.

### 6.4 Adapter (reimplement; about 200 lines)
1. Load the suite JSON. Fetch the rows by `row_idx` at `hf_revision`. Build the state from `state_fields`.
2. **Find the serialization by hash.** Try candidate encodings (nested `{"context":{"contexts":[…]}}` vs the flat key `"context.contexts"`; JSON separators; `ensure_ascii`) until `sha256(serialized state)` equals the published `state_sha256` for **all 300** items. Do the same for `criteria_hash`. This pins the exact bytes even though the harness is private.
3. Send 5 epochs with the published orders. For us all 5 are identical by construction; still score 1,500 decisions so the denominators match.
4. Score with the formulas above. The Jevals prior uses the evaluated items' base rates.

### 6.5 Traps and tracks
- The `experimental_evaluate` wire shape is not documented [I]. Jev's HellaSwag fell 9 points through Vercel `evaluate` versus native (scienthoon 0.861 vs Deußer 0.955), so harness effects are real. Match the state bytes by hash and accept that the wrapper is unknown.
- Jev probabilities are rounded to 2 dp; Decision Score uses Brier, which is not log-based, so this barely matters.
- **PubMedQA here is yes/no only (maybe dropped)**, unlike Deußer's 3-way.
- (S): Banking77 train and HelpSteer2 train are allowed. PubMedQA: `pqa_labeled` is fully off-limits (it holds the eval items); only `pqa_artificial` / `pqa_unlabeled` may be used.
- (Z): Banking77 and HelpSteer2 are excluded by the registry. PubMedQA is not in the v2 mix [F: live-leaderboards §3.3], though medical QA sets are kept.
- A board listing needs a hosted endpoint: skip it (leaderboards PLAN §2.7). This suite is for our own comparison.

---

## 7. DMB: nibzard/decision-model-benchmark

### 7.1 Source
- github.com/nibzard/decision-model-benchmark, `eabd88b04706` (2026-09-30). **No licence file**: read for protocol only; do not copy code or raw archives.
- Expanded run 2026-09-29 (`results/expanded-jev-2026-09-29/README.md`): `jev-latest`; 40,226 decisions, all valid; validation $0.638 + test $1.071 = $1.71.
- Pilot 2026-09-26 (`results/recent-pilot-2026-09-26/STUDY.md`): `jev-1.13.0`.

### 7.2 Jev's numbers

| Suite (test) | Size | Main | Other |
|---|---|---|---|
| `s6_banking77_test` | 3,080 (PolyAI CSV @57ec275d) | acc **0.792** | threshold 0.96 → coverage 51.2%, error 3.68% |
| `s7_clinc150_test` | 5,500 incl. 1,000 OOS (clinc/oos-eval @828f8093) | acc **0.886** | OOS P **0.903** / R **0.812**; threshold 0.62 → 88.3% / 6.65% |
| `s8_nlupp_test` | folds 18–19, banking + hotels: 302 messages / 13,712 binary decisions | micro intent-F1 **0.483** | macro-F1 **0.581**; complete-message acc **0.043**; binary acc 0.914; no qualifying threshold |
| Pilot S1 Banking77 | 77 items, one per intent, **from train.csv** | acc 0.805 | |
| Pilot S2 SMS spam | 50 items | acc 0.98 | |
| Pilot cardinality | | accepted up to 255 options | **rejected 256, 384, 512** |

### 7.3 Protocol
[F: `docs/expanded-benchmarks.md`, `src/dmb/suites/expanded.py`, `contenders/jev.py`, `contenders/render.py`]
- **One request per decision**, with question id `decision`.
- `state`: the message text stripped (`render_jev_state`). For NLU++ it is a **JSON string** of `{message, question: <ontology description>, instruction}`.
- `instructions`: one fixed string for every suite, `render.render_jev_instructions()`. It shares semantics with the LLM prompt; do not paraphrase it.
- `criteria`: `{option: option}`, so key = value.
  - Banking77 options: `sorted(train categories)` (77).
  - CLINC options: `sorted(150 train intents)` + one explicit out-of-scope option string (`expanded.clinc_items`).
  - NLU++ options: `["no","yes"]`, one request per (message, intent of its domain + general intents).
- **Validation splits** (threshold fitting only):
  - Banking77: 770 items, 10 unique train texts per intent after removing normalized test texts, `random.Random("<DMB_SEED>:banking77-validation")`;
  - CLINC: official val + oos_val minus test-overlapping texts;
  - NLU++: folds 16–17.
- **Metrics:**
  - accuracy = correct / all requested decisions (failures count wrong);
  - CLINC OOS precision and recall (recall over all OOS queries) and in-scope success;
  - NLU++ micro and macro positive-class F1 (absent intents count 0; missing answers count as missed positives) and complete-message accuracy;
  - thresholds = the largest validation coverage with error ≤ 5% and ≥ 100 accepted, ties kept together, applied frozen to test. NLU++ uses min-confidence per message.

### 7.4 Adapter (reimplement from the documented protocol)
1. Read the pinned upstream files: PolyAI `banking_data/{train,test}.csv` @`57ec275d8078af65b7731c2a98be812d844a6d6b`, `nlupp/data/{banking,hotels}/fold{16..19}.json` + ontology, and clinc `data/data_full.json` @`828f8093932c8fe6ca7936c3d2e52903b1c523de`. The URLs and sha256 are in `src/dmb/suites/expanded_sources.json`.
2. Build items exactly as `expanded.py` describes. Our builder may call the fixed instruction function at runtime from a pinned checkout rather than storing it.
3. Run, score, fit thresholds on validation, and apply them to test.
4. **Verify** the counts: test 3,080 / 5,500 / 302 messages with 13,712 decisions; validation 770 / 3,100 minus overlap / 310 messages with 14,064 decisions.

### 7.5 Traps and tracks
- The PolyAI CSV test has 3,080 rows versus 3,076 in mteb (Deußer). Option keys are full names.
- "Accuracy" includes failures.
- NLU++ binary accuracy is inflated by negatives. **The NLU++ headline is micro-F1.**
- The 0.246 ECE that Laya quotes comes from DMB's S5 "underdetermined, planted-answer" subset. It is not a general calibration number and not a target.
- **Registry conflict:** NLU++ folds 18–19 are our dev set. Fix that before claiming a DMB NLU++ number (§0.5).
- (S): Banking77 and CLINC train are allowed. NLU++ folds 0–17 are allowed, provided no message text appears in folds 18–19.

---

## 8. typed-decisions (LocalLLaMA/typed-decisions)

### 8.1 Source
- HF dataset @`d0e2f0c42fef86cc15d1688d25a19f5ba7c85b18` (card edited 2026-10-01), Apache-2.0. Configs: the four workflows + `all`; `train` 300 and `test` 100 per workflow.
- Write-up: latentnode.pages.dev/articles/typed-decisions.html. It does not give formal metric definitions. **The scorer is not public.**

### 8.2 Jev's numbers
Measured 2026-09-18 via `jev-latest` → `jev-1.13.0`; all 2,000 decisions; $0.016.

| Item | Value |
|---|---|
| Accuracy | **0.727** |
| KL from gold | **1.442** |
| Brier | **0.148** |
| ECE | **0.144** |
| p50 latency | 710 ms |
| Per type | noul .775, choice .720, score .696 |
| Laya's quoted extras (source unverified) | soft acc .580, score MAE .391 |
| Reference rows | Prior: acc .470, KL .347, Brier .189, ECE .088. Uniform: acc .308, KL .444, Brier .238, ECE .169 |
| Ceilings | teacher self-agreement 0.735; perfect factor recovery 0.704 |

### 8.3 Protocol
- `state` and `questions` (JSON strings in the row) are exactly the request body. Send **the whole case (state + 5 questions) in one request**.
- Gold is the mean of 3 teacher samples, in the `gold` column.
- Our `bench/jevbench/test/typed_decisions__main.jsonl` (mapper `m_typed_decisions`) is byte-identical. jevbench test is read once per release (`PREREG.md`).

### 8.4 Scorer reconstruction (needed because the scorer is unpublished)
- Implement accuracy (argmax vs gold label), KL(gold ‖ model) with an epsilon, multiclass Brier, and top-label ECE.
- **Pin the definitions by reproducing the Prior and Uniform rows to 3 decimals.**
  - Uniform accuracy 0.308 pins the tie-break rule: ties to the first option or level [I].
  - Uniform KL 0.444 and Brier 0.238 pin whether averaging is over decisions or cases, and the normalization.
  - Prior = per-question `train` label frequencies.
- Only once both rows match is our number comparable with Jev's.

### 8.5 Tracks
- **(Z):** zero-shot table. The model must never have seen these workflows or question schemas. Bar: Jev 0.727. Best zero-shot: meraGPT Decider 1 at 0.768. Our KL bars: Prior 0.347 and Brier 0.189.
- **(S):** "fitted on train" table. The comparison is Jev 0.727. Fitted neighbours: OpenDecider-nano (400M) 0.796; Bekko 68M 0.537; ModernBERT-base specialist 0.646.
- **Warning:** above 0.735 the card treats a model as learning the teacher's quirks.
- **Conflict:** an (S) model trained on this `train` must not be used for TypeSafe-evals (Z) (§9.5).
- **Sharpening conflict:** noul sharpening for JevBench (§4.5) hurts KL and Brier here. Report one global config for both, or two named variants that are never mixed.

---

## 9. TypeSafe evals (evals.typesafe.ai) via the WorkflowEvals harness

### 9.1 Source
- github.com/typesafe-ai/WorkflowEvals, Apache-2.0. The repo was created 2026-09-28T23:18Z; its single commit `0ac3b8ad8454` ("init") is from 2026-09-29T16:25Z. The README says the code behind evals.typesafe.ai is published here.
- Datasets (snapshot 2026-09-28, public, ungated), each with `data/{cases,questions,run_results}.parquet` and `dataset.json`:

  | Dataset | Revision | Licence |
  |---|---|---|
  | typesafe/evalsafe-invoice-processing | `6beeb2d2acd6` | apache-2.0 |
  | typesafe/evalsafe-customer-service | `b1342f5a7045` | not stated |
  | typesafe/evalsafe-security-incidents | `fbe1ea5c69cf` | not stated |
  | typesafe/evalsafe-agent-trace-observability | `863554097391` | not stated |

- This route uses only the GitHub repo and HF datasets. The PLAN's earlier concern about scraping the evals site does not apply.

### 9.2 Jev's numbers
[F: each `dataset.json` `runs`, model `typesafe:jev-1.13.0`, `question_mode: all`]

| Workflow | Cases | Policies | Metric | **Jev** | Comparators |
|---|---|---|---|---|---|
| Security incidents | 240 | playbook | exact action-set agreement | **0.6167** (n = 240) | claude-opus-5 .6625, gpt-5.6-sol .625, claude-sonnet-5 .6083 |
| Invoice processing | 150 | startup, enterprise, high_volume_retailer | exact actions / primary action | **0.6178** / 0.8311 (n = 450) | |
| Customer service | 204 | A, B, C | exact actions | **0.7598** (n = 612) | |
| Agent trace observability | 111 | balanced, cautious | disposition (primary action) | **0.7162** (n = 222, "replayed") | |
| **Equal-workflow mean** | | | | **67.76** (site: 67.8) | |

### 9.3 Protocol
- **Workflow loop.** Each workflow is code: gates, policies and actions in `evals/<wf>/`. It calls `judge(state, node_id, questions)` at each node. With `question_mode: all`, the node's questions go to the model in one System One request (`core/session.py`).
- **Answer handling.** Answers drive the code path (`core/answers.py`). The final actions are compared with the reference: `exact_actions` = set equality including arguments; `primary_action` = the primary action including arguments (`core/scoring.py`).
- **Reference.** `consensus` = the mean of the OpenAI and Anthropic reference probabilities (`dataset.json["blend"]`). Anthropic used an Opus fallback on 88 security cases. **Labels are closed-model outputs, not ground truth.**
- **Failures.** Failed cases count as wrong. There is no train split ("nothing is partitioned").

### 9.4 Adapter
1. `uv sync --locked`.
2. Run `uv run python run.py <workflow> --model typesafe:<id> --base-url http://127.0.0.1:8765 --dataset-revision <sha above> --name meharsjev-68m`.
3. `core/providers.ModelConfig.applicable_settings` **rejects any typesafe model other than `jev-1.13.0`**. [D] Apply a documented local patch that allows `meharsjev-68m` (Apache-2.0 allows modification; record the diff). Do not run under the name `jev-1.13.0`.
4. Our server must accept what `typesafe_sdk.TypeSafeClient` sends to `/v1/systemone`.
5. `scores.json` comes out against the `consensus` reference. Then compute the equal-workflow mean.

### 9.5 Traps and tracks
- **Z only** (no train split).
- The workflow names and decision types match typed-decisions. Keep any typed-decisions-trained (S) checkpoint away from this claim, and report the domain overlap.
- Never train on the `questions`/`cases` reference labels. They are the test, and they are OpenAI/Anthropic outputs.
- Jev's agent-trace run is marked "replayed".
- Dataset `main` moves: always pin `--dataset-revision`.

---

## 10. Laya evals (NandhaKishorM/laya `BENCHMARKS.md`)

- **Source:** @`fa9a2a7070b1789912a49ae24603bbfb1a78b001` (2026-10-02), Apache-2.0. The file itself says Jev figures are "third-party published, never measured here".
- **Jev cells and where they really come from:**

  | Laya cell | Value | Real source |
  |---|---|---|
  | typed-decisions | 0.727 (+ soft acc .580, Brier .148, ECE .144, score MAE .391) | typed-decisions card (§8) |
  | AG News / DAIR Emotion / banking77 | 0.910 / 0.480 / 0.870 | AbdelStark, N = 100 (§5) |
  | "ECE 0.246" | 0.246 | DMB S5 subset, not general (§7.5) |
  | latency | 236–276 ms | DMB pilot |
  | option-order flip | 0.13 | close to Jevals' 0.103 [I] |

- **Laya's own protocol is not Jev's protocol.** Its N = 400, seed 13 (`research/scripts/bench_apps.py`).
- **Spec:** there is nothing extra to reproduce. Beating Jev "on Laya's evals" means beating the source suites in §5, §7 and §8 under their protocols.
- **Optional:** run Laya's harness to set our model beside Laya's checkpoints. That is a separate, non-Jev comparison.

---

## 11. Other public studies with Jev numbers (appendix, protocol pointers)
Lower priority. Each is a single-author study; numbers are from the repo READMEs read 2026-10-02 unless marked.

| Study (licence, commit date) | Jev numbers | Protocol essentials for exact reproduction |
|---|---|---|
| elcronos/jev-vs-open-decision-models (**no licence**, 2026-09-20) | dair-ai/emotion test 2,000: acc .587, mF1 .500, ECE .281; "defined" variant +.013. tweet_topic_single test_2021 1,693: .793 (ECE .063). fin_topic (zeroshot/twitter-financial-news-topic @acbc8af2, validation) 4,117: .670 (ECE .166). daily_dialog (OpenRL/daily_dialog @1668faf0, test, 7,740 utterances): acc .710 / **mF1 .385** (ECE .156) | Jev via OpenRouter `POST /api/alpha/decisions`, model `typesafe/jev-1.13-20260917`. **state = raw text string**. One choice question; criteria `{label: ""}` (empty string, plain variant). Per-dataset instruction in `models/common.py` and `datasets_registry.py`. Full splits. Raw predictions in `results/*/summary.json` (evaluation only) |
| zhuyansen/jev-zeroshot-vs-bert (MIT, 2026-09-19) | AG News 1,000 .865; SST-2 872 .960; Banking77 1,000 .712 (two-step); TweetEval-emotion 1,000 .827; PAWS 1,000 acc .855 / AUC .936; arXiv 2026-09 258 .891; arXiv 2020 990 .938 | OpenRouter decisions, **20 texts per call**, label descriptions from `labels/*.yaml`, stratified seed-0 samples (`data.py`) |
| onlyoneaman/jev-eval (MIT, 2026-09-18) | Enron spam (SetFit/enron_spam test) 98.7; SST-2 val 95.7; AG News test 91.3; Banking77 (legacy-datasets/banking77 test) 76.0 | **Case files published** (`cases/<dataset>.jsonl`, 300 items, seed 7, state `{text}`), so items are exact. Per-item Jev answers are published (evaluation only) |
| anisselbd/jev-phishing-bench (**no licence**, 2026-09-19) | 2,000 PhishNChips emails: verdict acc **.626** [.605, .647], ECE .154 | Same data and question set as DI PhishNChips: DI imports `QUESTIONS` from `run_jev.py@1d56e8c` and orders records with `random.Random(20260916)`. **Use the DI route (§3)** |
| scienthoon/jev-ood-calibration (MIT, 2026-09-22) | OpenBookQA 94.2% (500); CSQA 88.1% (1,221); HellaSwag 86.1% (2,000); synthetic tickets queue 89.0 / angry 91.7 / priority 44.7 [from `v2/classifier-benchmarks.md`] | Vercel `evaluate` harness; raw responses in the repo |
| GautamTalksDev/jevbench (MIT, 2026-10-02) | ChaosNLI 750 easy + 750 hard; bias-corrected ΔECE .264 [from classifier-benchmarks] | Preregistered; calibration under human disagreement |
| jaredpalmer/kev (Apache-2.0) | transfer-v4 dev 764: .857; breadth-v1 test (DI-derived, 14 sources × ~150): Jev index **54.0**; MMLU .90; MMLU-Pro .84 [from classifier-benchmarks] | `evals/breadth-v1/manifest.json`. The test partition sits in a private mirror and is rebuilt byte for byte by `scripts/build_breadth_v1.py` |
| lexmount/WebJev (Apache-2.0) | Mean over 8 single-step benchmarks **84.70** (JevBench public 85.71, typed-decisions 74.05, MMLU-Pro 1,000 83.40, …) [from classifier-benchmarks] | Not re-read |
| kayzn-io/decision-models-as-judges (Apache-2.0); imaddde867/laya-eval (no licence) | tau-bench gates; agreement on TypeSafe public cases | Not read; low priority |
| Ibrahim & Zaki, arXiv 2609.24574; jev-frontier-bench; Arize; Janus; jev-certify; themsquared | see `v2/classifier-benchmarks.md` §1.2 | Not re-verified here |

---

## 12. Sources (all read 2026-10-02)

**Deußer**
- github.com/AppliedMachineLearning-Lab/jev-benchmarking @6bbdeb33474849b6de2f0cccc9f5e19756abd67e: `jev_benchmarking/{tasks/*.py, cache.py, evaluate.py, metrics.py, thresholds.py, runner.py, config.py}`, `docs/datasets.md`, `results/eval/{summary.md, *.json, thresholds.json, probes.json}`, `responses/LICENSE_RESPONSES.md`, `scripts/{evaluate,import_responses}.py`.
- Zenodo API record 23039006 (file sizes, created 2026-09-30).

**Decision Index**
- github.com/apolinario/decision-index @87d4650b42b377c0291a89c1f1a879f9b31082bf: `README.md`, `docs/{suite,format,engines}.md`, `decision_index/{constants,editions}.py`, `engines/{base,http}.py`, `scoring/{metrics,report,index,index02,added}.py`, `suite/build/{layout,freeze,normalize_direct,normalize_text,adapters_added,adapters_mechanical}.py` + greps of the other adapters, `data/index-0.2.1.json`, `tests/fixtures/board-0.2.1.json`.
- HF Space multimodalart/jev-decision-index: file list and `data/methodology.json`.
- HF dataset autotrust/jev-decision-index-results: card only.

**JevBench**
- github.com/fstandhartinger/jevbench @bb05a335bc80: `docs/METHOD-v1.5*.md`, `jevbench/{tasks,scoring}.py`, `adapters/{base,typesafe}.py`, `datasets/manifest.json`, `datasets/public/*.jsonl`, `results/v1.2/jevbench-v1.2-per-task.json`; branches and tags.
- https://benchmarkheaven.com/api/jevbench/v1.5.5 and https://benchmarkheaven.com/jev-models.

**Other suites**
- AbdelStark: github.com/AbdelStark/jev-benchmarks @0d610cc53e79: `configs/pilot-v1.yaml`, `docs/PROTOCOL.md`, `src/jev_benchmarks/{adapters/jev,data,metrics,io,models}.py`, `results/reports/btzsc-pilot-v1.md`.
- Jevals: github.com/Jevals/jevals-data @21bb47b72814: `README.md`, `suites/0.1.0/*.json`, `releases/2026-09-18/board.json`, `runs/jev__*.jsonl` (first lines only); https://jevals.com/methodology/.
- DMB: github.com/nibzard/decision-model-benchmark @eabd88b04706: `docs/expanded-benchmarks.md`, `src/dmb/suites/{expanded,s1_intent77,s2_spam}.py`, `expanded_sources.json`, `src/dmb/contenders/{jev,render}.py`, `results/expanded-jev-2026-09-29/README.md`, `results/recent-pilot-2026-09-26/STUDY.md`.
- typed-decisions: HF LocalLLaMA/typed-decisions @d0e2f0c4 (`README.md`, `eval.yaml`); latentnode write-up.
- TypeSafe evals: github.com/typesafe-ai/WorkflowEvals @0ac3b8ad8454 (`README.md`, `run.py`, `core/{dataset,providers,clients,session,scoring}.py`, `evals/security_incidents/{eval_adapter,workflow}.py`); HF typesafe/evalsafe-* (`README.md`, `dataset.json`).
- Laya: github.com/NandhaKishorM/laya @fa9a2a7070b1 (`BENCHMARKS.md`, `docs/evals.md`).
- Appendix studies: READMEs of elcronos/jev-vs-open-decision-models (+ `models/common.py`, `datasets_registry.py`, `models/jev.py`), zhuyansen/jev-zeroshot-vs-bert, onlyoneaman/jev-eval, anisselbd/jev-phishing-bench; jaredpalmer/kev `evals/breadth-v1/manifest.json`.

**Local**
- `jev_local/bench/{registry,templates,ours,build}.py`, `jev_local/{schema,serialize,api,confidence}.py`, `bench/sources/deusser/`.
- `docs/research/v2/classifier-benchmarks.md`, `docs/research/leaderboards/{PLAN,live-leaderboards,live-leaderboards-rules,live-leaderboards-2026-10-02}.md`.
