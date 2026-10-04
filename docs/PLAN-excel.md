# Plan: make meharsjev actually excel

Written 2026-10-03. Every stage below has a measured gate. Nothing large is spent until a cheap test has shown that the expensive version will work.

## What we've learned
- **Data scale worked once.** v1 to v2 took the dev skill score from 13 to 37.
- **More passes over the same zero-shot data don't help.** The Z-68m run converged at about the same level as v2. Its late training-loss drops were memorisation.
- **Calibration fitted on 10 dev sets doesn't transfer to unseen datasets.**
- **Training is sync-bound.** About 15 s of each ~23 s step is spent waiting at the barrier for the slowest worker. That roughly doubles every cluster bill.
- **We have only ever measured on our own jevbench dev sets, never on a winnable target.** That is the biggest gap.

## The goal, made concrete

| Target | Bar |
|---|---|
| Specialist track, public Jev benchmarks | Beat Jev's published number by the pre-registered margin on ≥25 of the 85 winnable rows, using the publishers' own harnesses. |
| Leaderboards | #1 in the ≤100M / ≤150M class on DecisionBench, Decision Index, DecideBench and typed-decisions. |
| Computer use | Hold the current lead: mid-sentence intent ≥90%, typing ≥94%. |
| On the Mac | ≤60 ms per decision and ≤600 MB RAM. |

We stop chasing the knowledge and reasoning benchmarks (MMLU, BBH, GPQA, HellaSwag). They are unwinnable at this size, and they get reported as losses.

## Stage 0: measure where we stand on real targets ($15, ~3 h)
Run Z-68m and v2-68m through the publishers' harnesses (Deußer 37, DMB, elcronos, typed-decisions) on their **validation** splits, never test. The adapters already exist.

**Output:** a per-benchmark gap table, ours vs Jev's published number. This tells us where the gap is small (worth fighting for) and where it is hopeless.

## Stage 1: fix training speed (code only, ~$20 to verify)
- Deal batches by sorted length across ranks, so every rank does equal work per step. Then do one all-reduce per larger step (grad-accum 4–8).
- **Gate:** sync time below 30% of the step on an 8-node, 20-step test. Expected about 1.8× throughput, which halves the cost of every later run.

## Stage 2: specialist pilot ($60–100, ~1.5 h)
- Fine-tune Z-68m on the specialist (S) mix for about 100M tokens.
- Evaluate on held-out validation carves of datasets in that mix (Banking77, CLINC, AG News, emotion, SNIPS/ATIS) through the publisher harnesses.
- **Gate:** at least 10 of the Stage-0 rows jump to within 2 points of, or above, Jev's published number.
  - If yes, go to Stage 3.
  - If no, the problem is format or recipe, not size. Diagnose it (request-shape mismatch with the harness, label wording, truncation) before spending more.

## Stage 3: S-68m full run (~$170 after the speed fix, ~4 h)
- About 0.6B tokens.
- Per-dataset calibration fitted on each dataset's own validation split. That is allowed on the specialist track, and it is the case where calibration does transfer.
- **Gate:** win count on validation, against the Stage-2 projection.

## Stage 4: teacher, then distil (the main accuracy lever)
- **Teacher.** Train S-400m (~$1.1–1.9k after the speed fix). This is the model expected to clear 37 wins.
- **Pilot first.** Before the full run, do a 30M-token run of the 400m model. It must beat the 68m pilot by ≥5 points on the same carves, or we stop.
- **Distil.** Have S-400m label every S-track training row, plus about 2M unlabelled texts drawn from the same domains, with soft labels. Train S-68m on those labels (~$150).
- **Why.** Distillation usually keeps 92–96% of the teacher's quality. That gives a fast Mac model that is much smarter than one trained alone.
- **Zero-shot variant.** The same idea uses open licensed teachers on generic text: deberta-v3-large-zeroshot-v2.0-c, and a Qwen3-8B prefill scorer on CPU. It's the only realistic way to move the zero-shot numbers.

## Stage 5: claim and publish
- Read each test split once through the publisher harness.
- Fill the pre-registered scoreboard with wins and losses, and the specialist-track disclosure.
- Submit to the leaderboards with your OK on each submission.
- Update the shareable comparison page.

## Budget and order

| Stage | Cost | Cumulative |
|---|---|---|
| 0 Measure | $15 | $15 |
| 1 Speed fix | $20 | $35 |
| 2 S pilot | $80 | ~$115 |
| 3 S-68m | $170 | ~$285 |
| 4a 400m pilot | $60 | ~$345 |
| 4b S-400m | $1.1–1.9k | ~$1.5–2.3k |
| 4c Distil to 68m | $150 | ~$1.6–2.4k |

Every stage stops on a failed gate. The earliest real "is this working" answer comes from Stage 2, for about $115 total.

## Things I will do differently
- Report validation-set numbers on the target benchmarks after every run, not training loss.
- Say up front what each stage will and won't buy.
- One heavy job at a time on each VM, and deallocate the VMs between stages.
