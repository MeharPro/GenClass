# GenClass vs Jev: direct benchmarks

All numbers below come from our own runs. Jev (`typesafe/jev-1.13-20260917`, called through OpenRouter) and GenClass received **byte-identical requests**, and the same code scored both against gold labels.
- GenClass here is the 32M computer-use model that ships in the extension.
- Raw Jev outputs are not published. They were used for evaluation only.

## 1. Controlling a browser/computer from speech
- **Test set:** 1,280 held-out examples. They come from app templates, apps and screens never seen in training.
- **Rows shown:** the rows least affected by labelling conventions (see caveats).

| Decision | n | Jev | GenClass | Winner |
|---|---|---|---|---|
| Which action, while the sentence is still unfinished | 664 | 66.4% | **90.4%** | GenClass |
| Which exact words to type | 98 | 75.5% | **94.9%** | GenClass |
| Which action, full command | 336 | 91.4% | 92.0% | tie |
| Which on-screen element (±7 pts at this n) | 119 | **86.6%** | 82.4% | Jev (slightly) |
| General questions from task families held out of training | 451 | **94.7%** | 80.5% | Jev |

Per question (n = 1,000 computer-control examples):

| Question | Jev | GenClass |
|---|---|---|
| Is it a command? | 82.4% | **96.1%** |
| Is the command complete yet? | 72.7% | **93.3%** |
| Is the action destructive? | 94.1% | **99.4%** |
| Which app? | 96.5% | **99.0%** |
| Which key? | 89.4% | **98.7%** |
| Which website? (n = 49) | 81.6% | **100%** |
| How far to scroll? (n = 54) | 59.3% | **98.1%** |

## 2. Speed and cost

| | p50 | p95 | Cost per decision |
|---|---|---|---|
| Jev (cloud API, including network) | 177 ms | 311 ms | ~$0.00008 |
| GenClass (CPU, 4 threads) | 187 ms | 204 ms | $0, local |

In-browser WebGPU latency will be published with the first extension release.

## 3. Wire compatibility
GenClass speaks Jev's request/response format. Its answer math was checked against 1,280 real Jev responses:

| Check | Agreement |
|---|---|
| Choice confidence | 6,239 / 6,244, within 0.02 (the misses are rounding) |
| Score confidence | 1,053 / 1,053 |
| Score value | 1,051 / 1,053 |

## 4. Answer-order robustness
- **GenClass:** 0 answers change when options are reordered, across 11 multiple-choice datasets. Each option is scored in isolation, so this holds by construction.
- **Jev:** independent public tests report 10.3% of answers changing (Jevals, Banking77), and 13% across reorderings plus repeated identical requests (nibzard).

## 5. General classification: where Jev is ahead
- Jev is far stronger on general-purpose questions. On AG News topic classification it scores 0.882 versus 0.315 for the computer-use model. GenClass's general models are being trained now (roadmap below).
- We report every loss as well as every win.

## Caveats (read before quoting)
- **Home-field advantage.** The computer-use test set comes from our own data generator, which uses our labelling conventions; Jev answers it zero-shot. Rows like "nothing to type" reward our conventions, which is why §1 shows only the fair rows. Full tables: `runs/compare/report.md` in the development repo.
- **Small n.** The element-picking and URL rows have fewer than 120 items.
- **Not an overall claim.** This is a specialist (computer control) compared with a generalist. It is not a claim that GenClass is better than Jev overall.

## Roadmap for general benchmarks
- A pre-registered campaign covers every public benchmark with a published Jev score: 549 rows, 154 counted.
- It runs two tracks: zero-shot, and supervised on official training splits (disclosed).
- Every win and loss will be published here, measured with each benchmark publisher's own harness.
