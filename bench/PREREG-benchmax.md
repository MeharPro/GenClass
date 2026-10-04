# Pre-registration: meharsjev benchmax campaign (phase 0 freeze)

- **Status:** DRAFT, LOCAL ONLY. Not published anywhere. Publishing the hash (gist, HF, GitHub) is the user's decision (PLAN §2.2.3, §10.4) **[needs user OK]**.
- **Frozen at:** 2026-10-03T10:55:00Z (UTC), by `bm-integrate`, on the Mac from the files listed in §2 (`shasum -a 256 -c bench/public/PREREG.sha256` passed at that time). No S or Z training run has started from this freeze (the v2-68m checkpoint `jev-local-fast-v2` predates it and is barred as an init, §2.3.5).
- **Companion file:** `bench/public/PREREG.sha256` (sha256sum format; `shasum -a 256 -c bench/public/PREREG.sha256` from the repo root verifies the frozen inputs).
- **Model:** meharsjev, an independent open reimplementation of a Jev-style typed-decision model; not affiliated with or endorsed by TypeSafe. Model ids are `meharsjev-<size>` (`meharsjev-68m`, `meharsjev-150m`, `meharsjev-400m`, `meharsjev-1b`), never `jev-*`. The Qwen3 scorer pilot, if run, is a different system (`meharsjev-scorer-8b`) and never counts as meharsjev.
- **Comparator:** Jev's **published** numbers only (`bench/public/jev_published.json`, 549 rows). Jev was not run by us for this campaign ($0 of the $5 lifetime cap); every per-item Jev file is used, if at all, for evaluation only and only where its licence allows (PLAN §2.6; Deußer `responses.db` is never used).

## 1. Decision rules (PLAN §1.1, frozen)

1. **Targets.** Every row of `bench/public/targets.json` is a target (549). Roles: `headline` 154 counted benchmarks (one per `benchmark_key`), `s_bar` 10, `secondary` 58, `context` 300, `composite` 4, `diagnostic` 23. Only headline rows are summed; secondary rows get their own verdict in their own harness; context, composite and diagnostic rows are reported in appendices and never counted.
2. **Win.** A row is a **Win** only if `ours − jev_score > bar` with `bar = bar_z` on the Z track and `bar_s` on the S track (`targets.json` fields). The bar is the unpaired significance margin of `feasibility-targets.md` §1.2: √2 × the publisher's CI half-width where a CI is published (the 37 Deußer rows), otherwise 1.96·√(2p(1−p)/n). Lower-is-better metrics (`higher_is_better: false`: Brier, RMSE, KL, FNR, false-alarm rate) have the bar below Jev and the inequality reversed.
3. **Other verdicts.** `Ahead (n.s.)`: above Jev's point but inside the bar. `Tie`: |Δ| < 1 point or inside the suite's own run-to-run noise. `Behind`: below Jev's point. Sub-1-point margins are never claimed as wins.
4. **Jev value conflicts.** Where publishers disagree on Jev's value (DI kit fixture vs Space `index.json`: BPoMP .909/.906, POP909 .166/.181, cfcolor .644/.6474, GPQA .786/.7828) the **stricter** value is the one to beat (`targets.json` `note`).
5. **S bar.** `max(Jev zero-shot, Jev given train data) + margin` where a "given train data" row exists (`s_bar_rows`): Banking77 .937, Deußer UNFAIR-ToS ≈.815, GoEmotions .365, ToxicChat .841, LexGLUE UNFAIR-ToS .938, LexGLUE mean .750, CESNET .290.
6. **Headline claims.** One summary line per scoreboard ("Win a · Ahead b · Tie c · Behind d of 154"). No "better than Jev" headline unless a scoreboard has more Wins than Behinds. Z and S scoreboards are never merged; S-spec, S-ens, S-cv, S+gen and the Qwen3 scorer are separate columns that never feed the S1 line.
7. **Protocol.** Our model goes through the publisher's own code or a hash-verified reimplementation (PLAN §6, `jev_local/bench/benchmax/`). Requests are the harness's own bytes; we never paraphrase them. Rows whose items we cannot reproduce exactly (hev candidate lists, CESNET sample, Koa wording, zhuyansen arXiv crawl, typed-decisions accuracy/ECE) are **indicative** and never headline.
8. **Coverage.** Every suite except Deußer counts a refusal as wrong. Deußer runs in truncate mode (100 % coverage, truncation counted and disclosed, `run.json` `overflow`), or a "missing = wrong" variant is reported beside the protocol number.
9. **Test is read once per release.** Each harness runs on its evaluated split exactly once per checkpoint; the first complete `run.json` is the reported one; reruns need `--rerun-reason` and are logged. Adapters are debugged on validation / dev / train items only (this freeze's dry passes used those only).
10. **Selection.** Z: mean skill over the clean dev set (§3) across checkpoints of one run; S: mean validation skill over the S allow-list, equally weighted per dataset. Checkpoint averaging allowed; per-row choice forbidden.
11. **No Jev output** ever enters training, selection, calibration or distillation (registry rule `jev_labelled`; Deußer `responses.db`, DMB raw archives, Jev-labelled corpora and `KoalaAI/Text-Moderation*` are never downloaded).
12. **Decontamination gate (Z).** The Z checkpoint ships only if (a) the `public_jev` source scan finds 0 matches in its manifest, (b) the item-level scan (exact, 13-gram ≥ 50 %, MinHash) finds 0 hits against the reference, (c) no dev set overlaps a Jev row. The audit report is published with the scoreboard; any residual overlap is dropped or labelled "(declared overlap: k items)".

## 2. Frozen inputs (sha256)

| File | sha256 | What it fixes |
|---|---|---|
| `bench/public/targets.json` | `d1eebdff6cb95a346e64e0ee213f999bcca6f1faa9d65f7553f0f3cfd4a23e9a` | 549 rows, roles, bars (`bar_z`/`bar_s`), verdict predictions, track eligibility, train sources, spec ids (schema `benchmax-targets/1`, generated 2026-10-03) |
| `bench/public/jev_published.json` | `9e1552f61a92b4bab30019c2bddfe1b2a11e49e34819a135d9c22a3a47d63a91` | every published Jev number with source, date, protocol, tier |
| `bench/public/exclusions.json` | `69694c71fc367a1a1e5a676036890d07f79ce9dd6c69a402df1924e097f26533` | registry `jevbench-v1.1` manifest (schema `benchmax-exclusions/1`): clean dev set, `public_jev` family (174 hf ids, 167 repo segments, 246 benchmark keys), the **S allow-list** (`s_track`: 112 train-split keys, 93 allowed ids, 13 mixed-eligibility ids, 18 deny rules) |
| `jev_local/bench/registry.py` | `c53a4bad9914eccbb457caab42b70e43f6999aced4c458697ead897fe8faee89` | `REGISTRY_VERSION = jevbench-v1.1`, `DEV_SET_VERSION = clean-2026-10-03`, generated block + hand rules (its `manifest_sha256()` equals the exclusions.json hash above) |
| `docs/research/benchmax/PLAN.md` | `121ef32d4e05b0ede4c64f47e7777eea7117c47e7bfd646b1dcd2ec4877e3628` | the binding plan |
| `docs/research/benchmax/suite-reproduction-specs.md` | `45459d5bb4ab26df8a14f8e98c9b67410541c1cfe0009a835b492e8ae752a4e7` | protocols, pins, W1–W11 |
| `docs/research/benchmax/feasibility-targets.md` | `ac445865f1f3829705590368948c7901fb029d239315d399502d9983ba412d70` | bar formula, W/T/L predictions |
| `docs/research/benchmax/train-data-and-supervised-ceilings.md` | `03e75d896411a0d5e92e77a44bcfcb3843769e4956e178d9974977a07e45ad64` | S train splits, traps |
| `scripts/benchmax_build_targets.py` | `fd7807131b23cd0ae366a6e0c36c5fb5cfdd0d985b8f9a38472cb66a2e040880` | generator of targets.json / exclusions.json (`--check` must exit 0 against the hashes above) |

Decontamination manifests (data VM `~/jev/data/bm/`, built 2026-10-03; the files stay on the VM, the hashes are frozen here):

| File | sha256 | Content |
|---|---|---|
| `reference/manifest.json` | `5613687c8391b4b8869e59a328e66f8e2a0ddd3722cab0ad5cc7b308eaa9bb1e` | decontamination reference `bm-reference-1.0`: 1,428,767 eval items / 253 datasets (16 suites + jevbench test/dev/ref/retired-dev) |
| `eval/status.json` | `d43c4421af9f8254879023397866f23a31ac8f31ea8425842288a94d3b29834b` | 15 eval suites, 1,321,413 request-shaped items, 150 n-checks (0 mismatches), 22 counted rows pending (6 gated, 16 not enumerable) |
| `eval/_revisions.json` | `550056e2875571003d0bdd11cac9d185b45ce5435b74c7feed0a56b4040e7113` | HF revisions used for every eval dataset (last commit ≤ 2026-09-26 unless the spec pins otherwise) |
| `z/public_jev_removed.json` | `201686e83b72a47862b42b6d9d12c433a013d04ec1f2dc3f56d3c9eb20e85604` | stage-1 (`public_jev`) removals from the v2-clean rows: 798,478 of 2,984,525 |
| `z/decontam_report.json` | `b7d60ace2fe7a5108aab34368f67e238209179d48a286d45d7b2f3e093906231` | item-level scan `decontam-v2.0`: 2,984,525 scanned, 2,166,610 kept, 817,906 removed (stage 1 + 19,428 item-level) |
| `z/rescan_shards.json` | `d1cd0e2784c272076fbf13205d7085b1c3e983d969d66996f0917902fd4fcda7` | rescan of the shipped Z shards: 1,839,398 rows, 0 rows with hits, 0 stage-1 hits, `exit_test_pass: true` |
| `z/balance_vs_v2.json` | `dc63f42a43408051abb57358be8edc3e297222a8588c7d66f04412d938aa07d1` | bucket shares Z vs v2 (preserved to ±0.0003) |
| `z/shards/mixture_stats.json` | `48d1557bfe18dfda2a7140830b6a447d969e26a368696a83e0b8aee783fa9d23` | mixture `mix-bmz-1.0-cleanZ`: train 1,809,770 rows / 4,620,182 decisions / 401.4M tokens; dev_mix 22,589 rows; dev_family 7,039 rows |
| `~/xfer/z_bundle.tar` | `a333fd88d5eb1bb743514776bdca2d3781a3ed46bd3b7a5ae227abbbb9bd4a0b` | 442,255,360 bytes, 93 files: the Z shards as served to the cluster on `http://10.0.0.5:8797/z_bundle.tar` |

Reference checkpoint (not a campaign model; barred as an init by PLAN §2.3.5): `jev-local-fast-v2` directory sha256 `4ddd9f5f51935ecdfcaec94b8367eab55243fb4db101089fc595277235ae6ee1` (heads `3ac9034db53c…`, calibration.json `0340d16a3475…`, meta `6bd559582493…`, max_len 8192).

## 3. Dev set (registry `DEV`, `DEV_SET_VERSION = clean-2026-10-03`)

Z selection and the global calibration file use only these. None carries a published Jev number except SNIPS, which is one component of LangWatch's off-topic mix (context row `lw_offtopic`; declared).

| key | dataset | hf id / config | split | n used | mapping | metric | note |
|---|---|---|---|---|---|---|---|
| `newsgroups20` | 20 Newsgroups | `SetFit/20_newsgroups` | test | 2000 (strat) | choice(20) | acc | |
| `tweeteval_sentiment` | TweetEval sentiment | `cardiffnlp/tweet_eval` / `sentiment` | test | 2000 (strat) | choice(3) | acc | |
| `atis` | ATIS intents | `tuetschek/atis` | test | 893 (all) | choice(intents present in test) | acc | new in v1.1; multi-intent `a#b` kept as one option; no Jev number |
| `snips` | SNIPS | `benayas/snips` | test | 2000 (random) | choice(7) | acc | declared LangWatch component (context row) |
| `mrpc` | MRPC | `nyu-mll/glue` / `mrpc` | validation | 408 (all) | noul | acc | |
| `scitail` | SciTail | `allenai/scitail` / `snli_format` | test | 2000 (strat) | noul | acc | |
| `cb` | CommitmentBank | `aps/super_glue` / `cb` | validation | 56 (all) | choice(3) | acc | |
| `tweeteval_offensive` | TweetEval offensive | `cardiffnlp/tweet_eval` / `offensive` | test | 860 (all) | noul | acc | |
| `scifact` | SciFact dev (`claims`) | `allenai/scifact` / `claims` | validation | 300 (all) | choice(3) | acc | stays; BEIR SciFact test queries (`BeIR/scifact`, T387) are `public_jev` |
| `tasksource_heldout` | held-out tasksource-jev families | `tasksource/tasksource-jev-typed-decisions` | train | 2000 (random) | native rows | soft_acc | families `ade_corpus_v2, conj_nli, equate, github-issue-similarity, help-nli, numer_sense, proofwriter, qasc, resnli` (strategy-qa retired: jev-bench T178/T179) |

Retired from dev (now `public_jev:*`, never selection data again): tweet_topic (T245), app_reviews (T380), hwu64 (T256), toxigen (T027), prompt_injections (T030 + the all-662 rows), NLU++ folds 18–19 (T112).

## 4. Harness pins (adapter commits / revisions; `jev_local/bench/benchmax/specs_a.py`, `specs_b.py`)

| spec id | harness | pinned commit / revision | counted rows |
|---|---|---|---|
| `deusser_exact@6bbdeb33` | AppliedMachineLearning-Lab/jev-benchmarking (MIT) | `6bbdeb33474849b6de2f0cccc9f5e19756abd67e` | 37 (+6) |
| `decision_index_0.2.1@87d4650b` | apolinario/decision-index kit (MIT), edition 0.2.1 | `87d4650b42b377c0291a89c1f1a879f9b31082bf`; rows `b2b56d6f…`, added `7429f3c9…`, exclusions `331df32d…` | 41 (+10) |
| `typed_decisions_card@d0e2f0c4` | LocalLLaMA/typed-decisions (HF card) | `d0e2f0c42fef86cc15d1688d25a19f5ba7c85b18`; train parquet `46a58d63…`, test `4f294f21…` | 1 (+4); only KL/Brier comparable (scorer pin) |
| `jevbench_hf_praveenrajus_v0.1.1` | Praveenrajus/jev-bench v0.1.1; scorer from uspraveen/Jevify | `c37b0f6ab1687376ec9ae8dbd9f343163a68252b`; Jevify `a1308666b5460291772161260f43ed95afa65815` | 13 (+10) |
| `chepyle_lexglue_systemone_v1` | chepyle/lexglue-systemone (Apache-2.0) | `eeb55e2c79dd9a41ec19e8f5877f6a8c65532712`; lex_glue `c23fdff1…`, clinc `155b9c71…` | T269–T280 |
| `mbburabak_safety` | mbburabak/jev-safety-bench (MIT) | `1bf1eacfc37e01cb0dd80a45d838f00c36cc1f1d` | T297–T304 |
| `dmb_expanded@eabd88b0` | nibzard/decision-model-benchmark (no licence; reimplemented) | `eabd88b04706bcc9b7769213d1ea644c09d8a7f8` | T109–T112 |
| `rerank_scripts` | denser-org / hev / anessbelbati | `41fe2570…` / `1eb47266…` / `fecba75a…` | T387–T393 (hev indicative) |
| `study:earino_zero_shot_complaint_benchmark_cfpb_113_cl` | earino/zero-shot-complaint-benchmark (MIT) | `33d5ed7467eff8831ccb47730dc14a535395544f`; data `4783de6e…` | T521–T522 |
| `study:do_system_one_decisions_add_up_arxiv_2609_33971` | arXiv 2609.33971 (TREC-50) | CogComp/trec parquet branch `65752bf5…`, seed 42 | T458 |
| `study:jev_for_network_traffic_classification_arxiv_261` | arXiv 2610.00376 (CESNET) | Lystea/CESNET-QUICEXT25-PARQUET `4180d7d9…`, seed 2026092305 | T490–T491 (indicative) |
| `study:koa_action_arxiv_2609_36115` | arXiv 2609.36115 | sst2 / amazon_polarity revisions pinned at prepare | T511–T512 (indicative) |
| `study:jev_ids_arxiv_2610_01079` | jev-ids/jev-ids (MIT) | `6aa5ac4570db1d6d9b88f769f7d16c1cd2c3784a` | T492 |
| `elcronos_plain` | elcronos/jev-vs-open-decision-models (no licence; reimplemented) | `a1901bc3d520e73936de8d4326545c0cdcf742fb` | T243–T247 |
| `zhuyansen_batch20` | zhuyansen/jev-zeroshot-vs-bert (MIT) | `edbf0713583644bd3f47299fd6b104a8b2073219` | T248–T253 (T253 indicative) |
| `thisisandreeeee` | thisisandreeeee/jev-benchmarks (MIT) | `eaed9dd0cfd6cfa085424e11f79cab8930847e16` | T254–T258 |
| `asevlad_injection` | ASEVlad/jev-injection-bench (MIT) | `c0d0f25d75f7f21908be968ae7cbe9da7d889287` | T308 (Z only) |
| `stperic_medhallu` | stperic/jev-medhallu-benchmark (MIT) | `8a7f2f88eb258fd0bc859ed4aeca3b80643e5d94` | T319 (S only) |
| `goya_rm_eval` | goya/jev-rm-eval | `d594fd4fdce39d805120a510c3804b960560acae` | T311–T318 |

Other pinned eval sources (decontamination reference, `eval/_revisions.json`): JevBench `bb05a335`, AbdelStark `0d610cc5`, Jevals `21bb47b7`, WorkflowEvals `0ac3b8ad`, evalsafe-* `6beeb2d2/b1342f5a/fbe1ea5c/86355409`, BTZSC `fef2a2ac`, PolyAI `57ec275d`, clinc `828f8093`, TMMLU+ v1.1 `94d86f1d`; every other HF dataset at its last commit on or before 2026-09-26.

## 5. Calibration policy per harness (W9; PLAN §2.1, §4.2.4, §9)

| Harness | Z track | S track |
|---|---|---|
| Decision Index (`--preset di`) | one global file, fitted on the clean dev set (temperature per kind × K-bucket, `tau_k`, `noul_platt`, **no `by_header`**) | the **same global file** (DI rule: no per-benchmark calibration; S1 checkpoint only, never S-spec) |
| jev-bench HF, JevBench, TypeSafe evals (`--preset jevbench`) | global file | global file |
| Deußer, DMB, LexGLUE (chepyle), typed-decisions, mbburabak, rerank, study adapters (`--preset deusser` / bridge default) | global file | per-dataset temperature (and per-kind for multi-question sets) + per-label thresholds for multi-label nouls, fitted on validation (or a 10 % train carve where validation is the evaluated split), chosen by **run config** (`--calib <file>` / `--thresholds`), never by inspecting requests |
| Thresholded variants (LexGLUE T277, DMB coverage, stperic 0.65, Deußer tuned rows) | n/a | thresholds fitted on validation items only (`bridge_b --fit` refuses the evaluated split) |

The server never looks at a request to pick a calibration file; `run.json` records the file path and sha256 in force. The v2 checkpoint's own `calibration.json` is the identity (τ = 1) and is only a placeholder until the Z-68m global file exists.

## 6. Engine / wire contract frozen for every run (W1–W11)

`choice` = argmax of unrounded probabilities (ties → first key in request order), `confidence` = (K·pmax−1)/(K−1); `score` = Σ i·pᵢ with `legend`; precision `exact` (leaderboards) or largest-remainder 2-dp; refusals carry `max_tokens_exceeded` + "maximum context length" / "options per choice" markers (400 default, 422 for JevBench); overflow `refuse` for DI and jev-bench, `truncate` (counted) elsewhere; object instructions, empty state and `null` = `""` = key all served; one global calibration file per run config; model id `meharsjev-<size>`; deterministic (byte-identical replays; the W11 report and every `run.json` `determinism` block record it).

## 7. What this freeze does not decide (open, for the user)

- GLUE-wide exclusion via the `nyu-mll/glue` hf id, arXiv (`ccdv/arxiv-classification`) and MS MARCO exclusions on Z (registry judgment calls; changing them means editing `targets.json` and regenerating, which changes the hashes above and requires a new freeze).
- Whether context rows (n ≤ 1,000) should drive Z exclusion (they cost ~200k rows: `label_pressure`, `race_h`, `squad_select`, `arxiv_2026`, `conll_typing`, `logiqa`, `p4g`).
- The b8_extractive binding of the Z pass (401.4M tokens vs the planned ≈1.34B): raise `max_repeat[b8]`, redistribute its 4 % share, or refill b8 — a recipe decision before phase 2.
- Gated sources (LLM-AggreFact, ToxiGen, HLE, GPQA, WildGuardMix, XSTest) and the typed-decisions accuracy/ECE scorer discussion.
- Publishing this pre-registration hash.
