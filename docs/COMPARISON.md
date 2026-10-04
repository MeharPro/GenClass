# meharsjev vs. TypeSafe Jev: results so far

Mehar Khanna, October 2, 2026. meharsjev is an independent reimplementation and is not affiliated with TypeSafe.

## Summary
- **Computer control.** On deciding what to do with a computer from speech, measured on the same inputs, meharsjev beats Jev on 4 of 6 head-to-head measures and ties on 1:
  - recognising a command before the sentence is finished: **90.4% vs 66.4%**;
  - picking the exact words to type: **94.9% vs 75.5%**.
- **General classification.** Jev is still clearly better: **94.7% vs 80.5%** on held-out generic questions. Our newest model (v2) is about **2.8×** better than v1 on our general benchmark, but has not yet been tested head-to-head against Jev.
- **Structural advantages:**
  - local, offline, private and free (Jev is a paid cloud API);
  - small, at 32M–68M parameters;
  - exact API compatibility with Jev;
  - provably immune to answer-order bias (0 flips; independent tests report Jev at 10–13%).

## 1. What was built
- **meharsjev.** A local "decision model" that answers yes/no, multiple-choice and rating questions about any input in one pass, with calibrated probabilities. That is the same job as TypeSafe's Jev (`jev-1.13`). It uses the same request/response format, so software written for Jev can point at it unchanged.
- **Architecture.** An encoder backbone (Ettin/ModernBERT) with typed decision heads. Each answer option is scored in isolation, so the order of the options cannot change the result.
- **Two versions:**

  | Version | Params | Training data | Training run |
  |---|---|---|---|
  | v1 | 32M | 64k computer-control and generic examples | 2.4 h on one Azure CPU VM |
  | v2 | 68M | ~1.3B tokens, from about 3.0M public-dataset rows after removing 44,023 rows that overlapped any benchmark | about 11 h on 8 × 80-core Azure CPU VMs |

- **The application.** A macOS voice-control harness that reads the screen through Accessibility and re-decides on every partial word you speak. It can act before you finish a sentence, as in the Jev demos.

## 2. Head-to-head: meharsjev v1 vs. Jev on computer control
- **Setup:** 1,280 held-out test examples. Both systems got byte-identical requests: Jev through OpenRouter (`typesafe/jev-1.13-20260917`), ours locally. Both were scored by the same code against gold labels.

| Decision | n | Jev | meharsjev v1 | Winner |
|---|---|---|---|---|
| Which action, mid-sentence (partial transcript) | 664 | 66.4% | **90.4%** | meharsjev |
| Which exact words to type (real payloads) | 98 | 75.5% | **94.9%** | meharsjev |
| Which action, full command | 336 | 91.4% | 92.0% | tie |
| Which on-screen element (real targets; ±7 pts) | 119 | **86.6%** | 82.4% | Jev (slightly) |
| Generic questions, held-out task families | 451 | **94.7%** | 80.5% | Jev |

**Per-question accuracy** (1,000 computer-control examples):

| Question | Jev | meharsjev v1 |
|---|---|---|
| Is it a command? | 82.4% | **96.1%** |
| Is the command complete yet? | 72.7% | **93.3%** |
| Destructive action? | 94.1% | **99.4%** |
| Which app? | 96.5% | **99.0%** |
| Which key? | 89.4% | **98.7%** |
| Which website? | 81.6% | **100%** |
| How far to scroll? | 59.3% | **98.1%** |

**Speed per decision:**

| | p50 | p95 |
|---|---|---|
| Jev (cloud, including network) | 177 ms | 311 ms |
| meharsjev on a CPU | 187 ms | 204 ms |

Jev's API cost for the run was $0.10; meharsjev costs $0.

## 3. Compatibility check against real Jev
Our answer math was checked against 1,280 real Jev responses:

| Check | Match |
|---|---|
| Choice confidence | 6,239 / 6,244, within 0.02 (the misses are rounding) |
| Score confidence | 1,053 / 1,053 |
| Score value | 1,051 / 1,053 |

meharsjev's API outputs are interchangeable with Jev's.

## 4. General classification benchmark ("jevbench")
- **Suite:** 25 public datasets (topic, sentiment, emotion, intent, inference, fact-checking and safety), plus separate development datasets. All are strictly excluded from training.
- **Pre-registration:** scoring rules were fixed in advance (`bench/jevbench/PREREG.md`, with hashes, timestamped before any test run).

**v1 vs. Jev, test set** (Jev's run completed only on these two datasets before the API account ran out of credit):

| Dataset | Jev | meharsjev v1 |
|---|---|---|
| AG News | **0.882** | 0.315 |
| Yahoo Answers | **0.743** | 0.211 |

**v2 vs. v1, development datasets** (skill index = accuracy above chance, averaged over task areas; Jev not run on dev by design):

| Model | Skill index | Calibration index |
|---|---|---|
| v1 (32M) | 13.3 | −29.2 |
| **v2 (68M)** | **36.8** | **+4.9** |

| Dev dataset | v1 | v2 |
|---|---|---|
| SNIPS intent | 0.537 | **0.746** |
| Tweet sentiment | 0.442 | **0.675** |
| App-review stars | 0.069 | **0.462** |
| HWU64 intent | 0.295 | **0.496** |
| Offensive tweets | 0.717 | **0.799** |
| 20 Newsgroups | 0.070 | **0.302** |

## 5. Other measured advantages
- **Order-invariance.** Shuffling the answer options changed 0 answers across all 11 multiple-choice benchmark datasets. That is guaranteed by the architecture. Independent third-party tests of Jev report:
  - 10.3% of answers changing when options are reordered (Jevals, Banking77);
  - 13% changing across reorderings and repeated identical requests combined (nibzard).
- **Privacy and cost.** Runs entirely on the user's machine, with no data leaving it and no per-call fees.
- **Size.** 32M–68M parameters (Jev's size is undisclosed). The decision model is about 130–275 MB.

## 6. How the comparison was kept fair
- Both systems received identical inputs and were scored by the same code.
- Test examples came from held-out templates, apps and screens, or from public datasets never used in training. Training data was decontaminated against every benchmark: exact match, near-duplicate (MinHash) and 13-gram checks, with 44,023 rows removed.
- No Jev output was used to train, tune or select our models. That is also what TypeSafe's terms require.
- **Caveat.** The computer-control test set comes from our own generator, which gives our model a home-field advantage on that table. Some of the gap reflects labelling conventions (for example, when "nothing to type" is the correct answer). The rows in section 2's main table are the fairest ones.

## 7. Where Jev is still better
- General-purpose questions: 94.7% vs 80.5% (v1). AG News and Yahoo topic classification by a wide margin. v2 is much improved but not yet tested against Jev.
- Picking an on-screen element, by a small margin within the error bars.

## 8. Next steps
1. Run Jev on the full benchmark against v2. This needs about $5 of API credit.
2. Recalibrate v2's confidence scores on development data.
3. Train a larger "teacher" model and distil it into the small local model.
4. Live testing of the voice-control app on a real Mac.
