# jevbench v1 — pre-registration

- **Registered:** 2026-10-02T00:26:44Z (UTC), **before any test read**: no system has been run on any jevbench-test or reference
  request yet, and nothing has been scored.
- **Registrant:** the `bench` agent (jev-local v2 build). The repository is not under git and the build contract
  forbids commits, so instead of a `jevbench-v1-prereg` tag this file is frozen by its sha256, which every release
  report quotes, and a copy is stored next to the manifest on VM `train` (`~/jev/bench/jevbench/PREREG.md`).
- **Design source:** `docs/research/v2/PLAN.md` §0.1 and §1. Where this file is more specific, this file is binding
  for the reads it covers.
- **Covers:** release read #1 (**M0**) and every later release read (M3, M4, M6) unless a new pre-registration is
  written and hashed before that read. Selection between releases happens on jevbench-dev only.

## 1. Systems in release read #1 (M0)

| System | What | Where it runs |
|---|---|---|
| `jev` | `typesafe/jev-1.13` via `POST https://openrouter.ai/api/alpha/decisions`; the resolved `model` field is recorded on every response | Mac, ≤ 6 concurrent, append-only cache |
| `jev-local-fast` | our v1 checkpoint (Ettin-32m), as shipped: its own `calibration.json` temperatures, no refit | VM `train`, CPU fp32, exact option chunking (§3) |

Weights: `backbone/model.safetensors` `db1d427220019ae6db00483efeec6c2428bd905e38f228a34aa0ed1425600308`,
`heads.safetensors` `ca59dfd28505383a9443cdde92e1906b5968919376a883bb6774d23244837f2f`,
`calibration.json` `b7b06a36b6f078992283714db7a0d293714be1a14d569668911caee392e4acb3`.

Jev output is evaluation-only (TypeSafe MCA §2.3(b)): it never enters training, checkpoint selection,
temperature fitting or data filtering, and raw Jev outputs are not republished. **Jev is never run on jevbench-dev.**

## 2. Datasets, sample sizes and seeds

Seed **20261001** for everything. Sampling (`jev_local/bench/build.py`):
- `all`: every row of the split(s), in HF order.
- `strat`: class-stratified, **proportional** allocation by largest remainder (ties by stratum name), then
  `random.Random("20261001:<key>:sample").sample` within each stratum.
- `random`: `random.Random("20261001:<key>:sample").sample` of row indices.
- Item ids are `<split>:<row index>` at the pinned revision; item = state = bootstrap cluster.

### 2.1 jevbench-test (25 datasets, 7 areas; counted)

| # | area | key | HF id / config | revision | split(s) | n used / total | sampling | primary metric | template id(s) | label text |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A Topic | ag_news | `fancyzhx/ag_news` | `eb185aade064` | test | 2000 / 7600 | strat | acc | choice(4), described labels | deusser |
| 2 | A Topic | yahoo_topics | `community-datasets/yahoo_answers_topics` | `6652a1e7c94f` | test | 2000 / 60000 | strat | acc | choice(10), BTZSC hypotheses | btzsc |
| 3 | A Topic | fin_topic | `zeroshot/twitter-financial-news-topic` | `acbc8af2a35c` | validation | 4117 / 4117 | all | acc | choice(20) | programmatic |
| 4 | B Sentiment | sst2 | `stanfordnlp/sst2` | `8d51e7e4887a` | validation | 872 / 872 | all | acc | noul 'is the sentiment positive?' (primary) + separate choice(2) request | deusser |
| 5 | B Sentiment | fin_phrasebank | `atrost/financial_phrasebank` | `fc7fb491db50` | test | 970 / 970 | all | acc | choice(3), investor-sentiment instruction | deusser |
| 6 | B Sentiment | sst5 | `SetFit/sst5` | `e51bdcd8cd3a` | test | 2210 / 2210 | all | acc_mode | score(5) | deusser |
| 7 | B Sentiment | yelp5 | `Yelp/yelp_review_full` | `c1f9ee939b7d` | test | 2000 / 50000 | strat | acc_mode | score(5) stars | programmatic |
| 8 | C Emotion | dair_emotion | `dair-ai/emotion` / split | `cab853a1dbdf` | test | 2000 / 2000 | all | acc | choice(6), BTZSC hypotheses | deusser+btzsc |
| 9 | C Emotion | tweeteval_emotion | `cardiffnlp/tweet_eval` / emotion | `b3a375baf0f4` | test | 1421 / 1421 | all | acc | choice(4) | programmatic |
| 10 | C Emotion | go_emotions | `google-research-datasets/go_emotions` / simplified | `add492243ff9` | test | 2000 / 5427 | random | macro_f1 | 28 nouls per state | deusser |
| 11 | D Intent | banking77 | `mteb/banking77` | `18072d2685ea` | test | 3076 / 3076 | all | acc | choice(77), BTZSC hypotheses | deusser+btzsc |
| 12 | D Intent | clinc150 | `clinc/clinc_oos` / plus | `155b9c710419` | test | 5500 / 5500 | all | acc | choice(151 incl. oos) | deusser+programmatic |
| 13 | D Intent | massive | `mteb/amazon_massive_intent` / en | `940fd47a81ea` | test | 2974 / 2974 | all | acc | choice(60), BTZSC hypotheses | btzsc |
| 14 | E Inference | boolq | `google/boolq` | `35b264d03638` | validation | 3270 / 3270 | all | acc | noul (state = passage + question) | deusser |
| 15 | E Inference | rte | `nyu-mll/glue` / rte | `bcdcba79d07b` | validation | 277 / 277 | all | acc | noul (entailment) | deusser |
| 16 | E Inference | anli | `facebook/anli` | `8e4813d81f46` | test_r1, test_r2, test_r3 | 3200 / 3200 | all | acc | choice(3) | deusser |
| 17 | E Inference | paws | `google-research-datasets/paws` / labeled_final | `161ece9501cf` | test | 2000 / 8000 | strat | acc | noul (same meaning?) | deusser |
| 18 | E Inference | stsb | `sentence-transformers/stsb` | `ab7a5ac0e35a` | test | 1379 / 1379 | all | spearman | score(6 levels, 0-5) | deusser |
| 19 | F Fact & safety | llm_aggrefact | `lytang/LLM-AggreFact` | `—` | test | 0 (skipped: gated, no HF auth) | strat_by_source | balanced_acc | noul (claim supported?) | deusser |
| 20 | F Fact & safety | climate_fever | `tdiggelm/climate_fever` | `ae61ccb9320a` | test | 1535 / 1535 | all | acc | choice(4), state = claim + 5 evidence sentences | programmatic |
| 21 | F Fact & safety | toxic_chat | `lmsys/toxic-chat` / toxicchat0124 | `29df8e4dba60` | test | 5083 / 5083 | all | f1_pos | noul toxic (+ jailbreak noul, secondary) | deusser |
| 22 | F Fact & safety | openai_moderation | `mmathys/openai-moderation-api-evaluation` | `84e5cf3bcd6a` | train | 1680 / 1680 | all | mean_auprc | 8 nouls per state | deusser |
| 23 | G Multi-question | typed_decisions | `LocalLLaMA/typed-decisions` / all | `d0e2f0c42fef` | test | 400 / 400 | all | acc | native, 5 questions per state, soft teacher gold | native |
| 24 | G Multi-question | unfair_tos | `coastalcph/lex_glue` / unfair_tos | `c23fdff1a6bf` | test | 1607 / 1607 | all | micro_f1 | 8 nouls per clause | deusser |
| 25 | G Multi-question | helpsteer2 | `nvidia/HelpSteer2` | `990b2711a361` | validation | 1038 / 1038 | all | mean_spearman | 5 scores (0-4) per state | deusser |

**LLM-AggreFact (#19) is not built**: the dataset is gated and VM `train` has no Hugging Face credentials
(`DatasetNotFoundError`, recorded in the manifest). It is excluded from the index for **every** system. Area F is
then the mean of 3 datasets; the index is computed over the 24 built datasets. Rule (b) still requires **13 wins**
(not 13 of 24 rescaled). Adding it later requires a new pre-registration before that read.

### 2.2 jevbench-dev and shown-not-counted reference rows

| role | key | HF id / config | revision | n used | notes |
|---|---|---|---|---|---|
| dev | newsgroups20 | `SetFit/20_newsgroups` | `f1b91292074e` | 2000 |  |
| dev | tweet_topic | `cardiffnlp/tweet_topic_single` | `87b7a0d1c402` | 1693 |  |
| dev | tweeteval_sentiment | `cardiffnlp/tweet_eval` / sentiment | `b3a375baf0f4` | 2000 |  |
| dev | app_reviews | `sealuzh/app_reviews` | `9eaa95f66364` | 2000 |  |
| dev | hwu64 | `DeepPavlov/hwu64` | `0dd289ccdeb1` | 1076 |  |
| dev | snips | `benayas/snips` | `16915b895754` | 1400 |  |
| dev | mrpc | `nyu-mll/glue` / mrpc | `bcdcba79d07b` | 408 |  |
| dev | scitail | `allenai/scitail` / snli_format | `0cc4353235b2` | 2000 |  |
| dev | cb | `aps/super_glue` / cb | `3de24cf8022e` | 56 |  |
| dev | toxigen | `toxigen/toxigen-data` / annotated | `ea082a8973a2` | 940 |  |
| dev | tweeteval_offensive | `cardiffnlp/tweet_eval` / offensive | `b3a375baf0f4` | 860 |  |
| dev | prompt_injections | `deepset/prompt-injections` | `4f61ecb038e9` | 116 |  |
| dev | scifact | `allenai/scifact` / claims | `—` | 0 | no mapper (not built in v1) |
| dev | tasksource_heldout | `tasksource/tasksource-jev-typed-decisions` | `8173a06c7bb6` | 2000 |  |
| ref | hellaswag | `Rowan/hellaswag` | `218ec52e09a7` | 10042 |  |
| ref | winogrande | `allenai/winogrande` / winogrande_xl | `01e74176c635` | 1267 |  |
| ref | mmlu_pro | `TIGER-Lab/MMLU-Pro` | `b189ec765aa7` | 2000 |  |
| ref | bbh | `lukaemon/bbh` | `—` | 0 | no mapper (not built in v1) |

Dev is for checkpoint, mixture and temperature selection only. SciFact (no parquet conversion; script dataset),
NLU++ and BBH are not built in v1. Reference rows (HellaSwag, WinoGrande, MMLU-Pro 2k) are shown and never counted.

## 3. Requests

- One request per item, carrying all of that item's questions; byte-identical `{state, questions}` for every system
  (the runner sends `{"model": ..., **request}`). String criteria only; a noul carries both `true` and `false`
  criteria or neither; instructions are strings (OpenRouter's observed schema).
- **Variants** (separate requests, same item id):
  - `main`: described labels in dataset label order. **The only variant used for the index and decision rule.**
  - `shuffle`: one seeded non-identity permutation (`random.Random("20261001:<key>:<item>:shuffle")`) of every
    choice question's options, for the 11 choice datasets (28,793 requests). Used for the flip rate only.
  - `bare`: choice descriptions removed (criteria values `null`), 9 described-label choice datasets (24,058).
    Robustness only (PLAN risk 7: described labels primary, bare names secondary, never the more favourable one).
  - `choice` (SST-2 only): Deußer's 2-option choice template (872). Secondary.
- Label text: Deußer et al. `jev-benchmarking` @ `6bbdeb33474849b6de2f0cccc9f5e19756abd67e` (MIT), BTZSC hypotheses
  @ `fef2a2ac62b69c58670047dddf045c53d7c3cb5e`, or a fixed programmatic frame around the label name (see
  `jev_local/bench/templates.py`). No LLM-written text. Banking77: 5 of 77 intents have no BTZSC hypothesis
  (`card_linking, exchange_via_app, get_physical_card, supported_cards_and_currencies,
  top_up_by_bank_transfer_charge`) and get the programmatic frame; MASSIVE en test has 59 intents, 58 with a
  hypothesis (`general_quirky` programmatic).
- Our model: requests longer than its 2,048-token budget are split into passes that repeat the state and carry
  some questions or a contiguous slice of one question's options; raw logits are concatenated and normalised once
  (exact, because options and questions are attention-isolated; verified to 1e-4 in `tests/test_bench_ours.py`).
  If the state plus one option cannot fit, the longest state field is cut at its end and the row is flagged
  `truncated` (reported).
- Jev: retries on 408/409/425/429/5xx with exponential backoff (6 attempts); other 4xx are permanent failures.
  Session spend cap $8 (abort, report partial coverage).

## 4. Answer math (identical for every system; `jev_local/bench/metrics.py`)

1. Probabilities are aligned to the request's canonical labels (the `main` criteria order; score levels by index;
   noul = P(yes)). Missing labels count 0, unknown labels are ignored, the vector is renormalised.
2. **Failure** = failed request, missing question, wrong answer type, or an all-zero vector. A failure is **wrong**
   for every decision metric (choice argmax = none; noul decision = not gold) and is scored as the **uniform**
   distribution (noul p = 0.5) for every probability metric. Coverage is reported.
3. **Rounding:** both systems are scored on 2-decimal probabilities (Jev's wire precision; ours rounded the way our
   API rounds). NLL floors probabilities at **0.005**.
4. **Ties:** argmax ties break by canonical label order (never by a shuffled request's order). Score mode ties go
   to the lowest level. Noul decision: P(yes) ≥ 0.5.
5. **Brier** is the multiclass sum over options for every kind (a noul is a 2-option question).

## 5. Per-dataset primary metric, chance and skill

| Primary metric | Datasets | Definition | Chance c |
|---|---|---|---|
| acc | 1-3, 5, 8, 9, 11-13, 16, 20 (choice) | argmax == gold | 1/K |
| acc (noul) | 4, 14, 15, 17 | (P(yes) ≥ 0.5) == gold | 0.5 |
| acc_mode | 6, 7 (score) | modal level == gold level | 1/K |
| spearman | 18 | Spearman(Σ i·pᵢ, gold 0-5 score) | 0 |
| macro_f1 | 10 | mean over the 28 nouls of positive-class F1 | mean over heads of 2π/(1+π) |
| f1_toxic | 21 | positive-class F1 of the `toxic` noul (the `jailbreak` noul is secondary) | 2π/(1+π) |
| mean_auprc | 22 | mean over the 8 nouls of average precision (step-wise, sklearn semantics) | mean prevalence π |
| micro_f1 | 24 | F1 pooled over the 8 nouls | 2π/(1+π), pooled π |
| mean_spearman | 25 | mean over the 5 scores of Spearman(Σ i·pᵢ, gold) | 0 |
| acc (pooled) | 23 | argmax == teacher gold label over all 2,000 decisions | mean over decisions of 1/K |
| balanced_acc | 19 | (not built) | 0.5 |

π is the test-set positive rate; 2π/(1+π) is the F1 of the all-positive predictor (the best input-blind F1).
Chance constants are computed once on the full test set. **skill = clip((m − c)/(1 − c), 0, 1).**

**Decision Score** per dataset = clip(1 − mean Brier / mean Brier_prior, −1, 1), pooled over every decision of
the dataset, where the prior predictor of each question (head) is its full-test-set mean target (class frequencies;
typed-decisions: the mean soft gold of that workflow's question).

## 6. Indices

- **Skill index** = 100 × mean over the 7 areas of the mean per-dataset skill (Decision Index convention).
- **Decision-Score index** = 100 × mean over areas of the mean per-dataset Decision Score (Jevals convention).
- Only the `main` variant of counted test datasets enters the indices.

## 7. Uncertainty

Paired bootstrap, **B = 2,000**, numpy `default_rng(20261001)`; datasets are processed in registry order and draw
from one generator. Each resample draws items with replacement **within each target stratum** (the `stratum`
column: gold label, score level, `pos/neg` for the multi-label sets, workflow for typed-decisions, `n0/n1/n2`
label-count for GoEmotions) and uses the same items for every system (paired, item-clustered: all questions of an
item move together). 95% CIs are the 2.5/97.5 percentiles. Reported for every per-dataset metric, every index,
and every difference against Jev.

## 8. Decision rule (PLAN §0.1, fixed in advance)

We claim "better than Jev on jevbench" for a system only if **all three** hold on the counted test datasets:
- **(a)** the 95% CI of index(ours) − index(Jev) has a lower bound > 0 for **both** the skill index and the
  Decision-Score index;
- **(b)** ours has a strictly higher primary metric (point estimate) than Jev on **at least 13** datasets;
- **(c)** no area's skill (× 100) trails Jev's by more than **10** points.

Otherwise the report gives per-area and per-dataset wins and losses with no headline. M0 is a baseline read: v1
(32m, trained on computer-use and generic data only) is not expected to satisfy the rule.

## 9. Secondary outputs (never used for the claim)

Per question: accuracy, macro-F1 (choice), positive-class F1, AUROC, AUPRC, JevBench band-rule accuracy (0.2/0.8)
for nouls; Brier, NLL, ECE (15 bins primary, 10 bins for AbdelStark/JevBench), AURC, coverage at ≤5% error (test
oracle threshold, because Jev is never run on dev); for scores MAE, RPS, QWK; KL to soft gold (typed-decisions).
Option-order flip rate and mean total variation (main vs shuffle); bare-names Δ; SST-2 choice-variant accuracy;
latency p50/p95 (Jev hosted including network; ours VM CPU, 2 threads per worker, not Mac numbers); coverage, cost,
resolved Jev model ids. Reference rows (§2.2) are reported as shown-not-counted.

## 10. Known deviations from PLAN §1.1 at registration

- LLM-AggreFact not built (gated, no credentials): index over 24 datasets (§2.1).
- Banking77 `mteb/banking77` test has 3,076 rows (PLAN total 57,609 holds); MASSIVE en test has 59 intents (not 60).
- SST-2: noul is primary (PLAN lists "noul, plus a choice variant"); Deußer's choice template is the secondary request.
- ToxicChat requests carry Deußer's two nouls (`toxic`, `jailbreak`); only `toxic` is scored for the index.
- OpenAI moderation and UNFAIR-ToS: Deußer's object instructions are rendered as one string (OpenRouter rejects objects).
- Dev: SciFact and NLU++ not built; the held-out tasksource-jev families are 200 rows each (2,000 total).

## 11. Frozen artefacts

- `bench/jevbench/manifest.json` sha256 `b58984151c6885136ba936b36db505490ed1e8aaedd99963305e1f512b1d9183`
  (built {"test:main": 52609, "test:shuffle": 28793, "test:bare": 24058, "test:choice": 872, "ref:main": 13309,
  "dev:main": 16549}; a from-scratch rebuild reproduced all 61 file hashes byte for byte).
- Code: `registry.py` `ddcfc4599e975e575b371f6351794461f45c738f1207317131f378229977d403`,
  `templates.py` `a9ae593fadf2cf54f996d612246462c58a82e0432a04b5aa22e809f917f2dbfd`,
  `build.py` `165592e08122810c2998a3607913af6ef5e4f5c35dcc50148495ffee131b3fb5`,
  `metrics.py` `8b1d961d55a8e959641a424492a5a817176c0b0ac5db8e16749654afb8a48238`,
  `ours.py` `b3af6b59a12ff929f78a0d44c975a355025a060756f213b97016193f374d411b`,
  `scripts/jevbench.py` `c32a0aba62e5b84681e106e6e993dceb10112966202332a306cbd3855e3290a9`.
  Later edits to `metrics.py` or `scripts/jevbench.py` that change any number require a note in the release report
  that re-scores the earlier release with the new code as well.

### 11.1 Request files

| file | rows | sha256 |
|---|---|---|
| `dev/app_reviews__main.jsonl` | 2000 | `c6492db4dcde36968d95ba82e609416d709842b81575b4b42aba78a3e272bc94` |
| `dev/cb__main.jsonl` | 56 | `1f66a902edc6675fd4fd512d6159985525ed6e930d7a05e74ddb669d6fb3607a` |
| `dev/hwu64__main.jsonl` | 1076 | `e4ce488176bdcbba7d0bc801a0e8c49bc0c3a4117bb203548df5ff9a631e8343` |
| `dev/mrpc__main.jsonl` | 408 | `f63b5772d716bd291af29b1f1fd56d728fc2dd7e549d0e55db79d828d3563dd5` |
| `dev/newsgroups20__main.jsonl` | 2000 | `ecfab807a4bdb947a89098acc9f21afe429bcb87dffd7de38e6979e367948440` |
| `dev/prompt_injections__main.jsonl` | 116 | `950fb6c7be86f273fd5a2fde3172710be84422c2dc4b50db9d85e319ae592d9c` |
| `dev/scitail__main.jsonl` | 2000 | `bfd48049761b8649d383ea67a91b2dc7bce396038c976a0a313d44789120042d` |
| `dev/snips__main.jsonl` | 1400 | `902b82223eb60840b025a659cddba0119480c427a476b7c9d5bd8b57c6841f86` |
| `dev/tasksource_heldout__main.jsonl` | 2000 | `a108c64676013ba0130926732002aee06731d1499960e262f39849953497bb4c` |
| `dev/toxigen__main.jsonl` | 940 | `b2ae02a9b9e556ee5d366b0a8d6e9ba2aa7696f0da544a6994fe8b9d736f0923` |
| `dev/tweet_topic__main.jsonl` | 1693 | `99139b3d8414962c80fcc6a4cff4127843d30ee2a33ec1014fdd0adefce7b92d` |
| `dev/tweeteval_offensive__main.jsonl` | 860 | `e4917dcc14d513423e616599cf6ae0e780729787fe9278fc75875400b022d771` |
| `dev/tweeteval_sentiment__main.jsonl` | 2000 | `d41bbd22a034e818a5603bec45db67d2bde5c4c435d29e4db745cb6fb39fd1c7` |
| `ref/hellaswag__main.jsonl` | 10042 | `1aa761f8731af57c6da80b89986f0f599f1da76d78ba9dc04b70e761396d60fb` |
| `ref/mmlu_pro__main.jsonl` | 2000 | `b0cc01302f9666d42d1a455ff35b436fc6e9955868af1e06e4be4926c31bead9` |
| `ref/winogrande__main.jsonl` | 1267 | `4e94780352d96b09fe8af576646f416f3c0fa4fb2509ea62d5e5cd0745854e62` |
| `test/ag_news__bare.jsonl` | 2000 | `573a9b54385402757d7253b6a9d1b19e307e8cc038dee2928194ad9abdd1a914` |
| `test/ag_news__main.jsonl` | 2000 | `48c566b2f18dda4b30e4d3f0cef1f497f74367249138333184c6f78b6625dcf5` |
| `test/ag_news__shuffle.jsonl` | 2000 | `64e0f25cbef5f47d8cc02a241d70580023c76bdb96db87162c3c8131e037ee63` |
| `test/anli__main.jsonl` | 3200 | `28911e75e150df4ea937c40b2a50a073871468b9146c0926e0e6d92483a500e9` |
| `test/anli__shuffle.jsonl` | 3200 | `6b0231f213c73b3f555ef38d92b855ddf14477b3da68f60ce4f783caa32b56da` |
| `test/banking77__bare.jsonl` | 3076 | `4d7dd7e1b531de516aa7aa791ff0d3b5311e7d4756ebc207ca787e4433c4a21a` |
| `test/banking77__main.jsonl` | 3076 | `9967f2601c77a0d8808083dfb7f2ef4c87ec5e97862cf06aea361dc45704264d` |
| `test/banking77__shuffle.jsonl` | 3076 | `48ad295d9324fa4f6b2695c18deb5a6b0ebe29343e3002c61c5a3e2449d67eda` |
| `test/boolq__main.jsonl` | 3270 | `65e226cdcadb0553346b3471e8b4da58346d4182e7c96caac3f747a135d98602` |
| `test/climate_fever__main.jsonl` | 1535 | `04936f828118a51396d1d9c2272ac41c03b297277eb46961419969ca4bff2b5e` |
| `test/climate_fever__shuffle.jsonl` | 1535 | `5d18c5bee5d0e2e55fc5721fe3127fc5178a96ebaa1df8fe3d641bbcdfe2513b` |
| `test/clinc150__bare.jsonl` | 5500 | `efff3c8bcfcc6cc4cb6f2907bed4515bc385cfe4e9766f5869c68c2d9b90f2c7` |
| `test/clinc150__main.jsonl` | 5500 | `b7b81f54011123a89360f167b3cfb552c7b5c0713c1a6ae380343a823fbfa441` |
| `test/clinc150__shuffle.jsonl` | 5500 | `a2acb88ad786e1200ebc819209788e50d416be6d6a4f58fbef61741be18ae260` |
| `test/dair_emotion__bare.jsonl` | 2000 | `b9eaea5fadd99ddbad28e25ef6ece1066ac4decb09a1f89850498ce71752730b` |
| `test/dair_emotion__main.jsonl` | 2000 | `b74180679ee2644d437094adc26263a5ac70d7f06cac6881188472d9ffc9b021` |
| `test/dair_emotion__shuffle.jsonl` | 2000 | `966ef12a1c54ed4a21540f9ac3f54621840cf799680a49f9b38fbe104bbd038f` |
| `test/fin_phrasebank__bare.jsonl` | 970 | `3236da6fdc3ec3ba4a1b5fcfc249ab91e20e8496e6dee8d59edb9035692075e3` |
| `test/fin_phrasebank__main.jsonl` | 970 | `b6e90cefdc775da484f1a18b9ebff10d3ac5b138d80eee34e6c0f074ff55030c` |
| `test/fin_phrasebank__shuffle.jsonl` | 970 | `1958a0c02844c4394de0d7e831163d6f120a78845e5e1f083808010c938ae2c0` |
| `test/fin_topic__bare.jsonl` | 4117 | `c585cc82cc3b8686b1fe1d9b3bb8f61fa8d8f98ddf461a7c4c6ee489921a2807` |
| `test/fin_topic__main.jsonl` | 4117 | `f44ec242397197b332614cdaee01a2c92e37e725d2807ad356eefa740e785635` |
| `test/fin_topic__shuffle.jsonl` | 4117 | `463be3e8bf27caed718dd6306f75c8df2d3322fae3f6bdaac90588efb9afea60` |
| `test/go_emotions__main.jsonl` | 2000 | `728d644534f01a0578122e89f2132478215f08031a2230c7fce76df529bd9ff0` |
| `test/helpsteer2__main.jsonl` | 1038 | `ff8fb5b2377f6ab93bc337e1433487dbe00d841d92560610be0d0af4a86a452b` |
| `test/massive__bare.jsonl` | 2974 | `c660d5a9c40a449d61baef8fcbea8ed25a7e0b7015bd1c5f73e34e8c92986344` |
| `test/massive__main.jsonl` | 2974 | `07daabb89c31575d79fe77d85c77ceb5c17e3e3e8bd5740983cb37ece2435794` |
| `test/massive__shuffle.jsonl` | 2974 | `1281fb41621a5b803c0d6759aec45421932f559a537bb5524aa1e9a90b6f98ba` |
| `test/openai_moderation__main.jsonl` | 1680 | `d7ecc9feb826668f5390fa7193dc4b29dde2afeb122d352141de2a940320e7e7` |
| `test/paws__main.jsonl` | 2000 | `450289283a07167534d3dc83b2df8fe9285858ea3fa0cc336902e2c2aa97a229` |
| `test/rte__main.jsonl` | 277 | `f6e2fdfe0fe91d96606e6aa65c73f5b20902d1283926000b674fb12df264c170` |
| `test/sst2__choice.jsonl` | 872 | `c99e9dbb2ad9fe2805431b867ea2346bb7b7935ce830702d9e246e5163339f40` |
| `test/sst2__main.jsonl` | 872 | `a8b2d2cffebeeff35e03c08ab175c2ab37c1e58fb9e4da20274ab8ec907a268e` |
| `test/sst5__main.jsonl` | 2210 | `0efbd0becee08b29f14d5261102e58d642794b0031e57d0e6534dfc1b8ea313a` |
| `test/stsb__main.jsonl` | 1379 | `a7829b31e3ab2255bcfd9d6fdee2eb914c03c78fb841b1ccfa0211f880b94272` |
| `test/toxic_chat__main.jsonl` | 5083 | `29bd3d92576ec24a55f5c9843d32ba0727345cf825fe0abfa3162b4d3680ac54` |
| `test/tweeteval_emotion__bare.jsonl` | 1421 | `40bbd46cf313fa18b53cfe3aa11a084af8cce5ebe2e26681050be29907115826` |
| `test/tweeteval_emotion__main.jsonl` | 1421 | `1d70c8382326fdc72732242f38ecc16bcf36489f0b78057b813bb158700e91c9` |
| `test/tweeteval_emotion__shuffle.jsonl` | 1421 | `574653afad0e6defbe142abdaadb47c247706187da99418c3dc9110f31fc4815` |
| `test/typed_decisions__main.jsonl` | 400 | `73c1f4e0e45a7cb221726b64857e65f469471e21f0100b5d85af83fa2b9337dd` |
| `test/unfair_tos__main.jsonl` | 1607 | `e899ae3c35b521c22437e31626f77ebb94aa0cc241e9d8ce36f975f67f051e40` |
| `test/yahoo_topics__bare.jsonl` | 2000 | `49c2b54dead534cd8ffa1adc389c7cfd5730288452b20a845a8f033c61f4dc32` |
| `test/yahoo_topics__main.jsonl` | 2000 | `47538b8aaaf4580109de244c0331dac582e44e8518c0e1bbccd8e1b13677987c` |
| `test/yahoo_topics__shuffle.jsonl` | 2000 | `4579333e8cb9f83647d0f1456c011d3f782136a34489dd245a2fa232dc80a3c5` |
| `test/yelp5__main.jsonl` | 2000 | `e4ae6cd6c1454daf76841a00cdcfe606dffa40889e464f4552bda02cd196d18a` |

