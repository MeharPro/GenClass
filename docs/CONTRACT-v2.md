# jev-local v2 build contract (M0 + M1)

The design is `docs/research/v2/PLAN.md`, and that file wins on substance. This file wins on mechanics: ownership, where things run, and safety. The v1 contract (`docs/CONTRACT.md`) still defines the example format and the module layout it covers.

## Hard rules
1. **The Mac is an 8 GB M1 that has crashed under our load.** On the Mac you may:
   - edit files;
   - run *pure-Python* tests (no torch, no transformers, no datasets imports);
   - use ssh, scp and rsync;
   - make lightweight HTTP calls.

   Never run torch, model or dataset work locally. Never run more than one heavy process. Never run `az` (only the lead runs `az`, and only serially). Everything heavy runs on the Azure VMs.
2. **VMs.** Use them through `scripts/azvm.sh <host> ...`:

   | Host | Size | Private IP | Role |
   |---|---|---|---|
   | `train` | D64as_v7, 64 vCPU / 256 GB | 10.0.0.4 | Benchmarks, evals, engine tests |
   | `data` | F80as_v7, 80 physical cores / 320 GB, 512 GB disk | 10.0.0.5 | Data building |

   - Both are in VNet `vm-jev-trainVNET`; they can reach each other on their private IPs.
   - Code lives in the Mac repo. Edit locally, then `scripts/azvm.sh <host> --sync` (rsyncs `jev_local/`, `scripts/`, `tests/`, `pyproject.toml` to `~/jev`), then run remotely with `~/jev/.venv/bin/python`.
   - Don't hand-edit code on a VM.
   - Data and artifacts stay on the VMs. Copy back to the Mac only small reports (under 20 MB).
   - Several agents share each VM. Use `nice -n 5` for long jobs, write under your own directories, and never kill processes you didn't start. Long jobs run under `setsid nohup ... &` with logs, so an ssh drop can't kill them.
   - The `train` VM's 64 vCPUs are 32 physical cores; leave about 16 free for others. The `data` VM is shared by three data agents; keep any one job to 40 threads or fewer.
3. **Jev.**
   - Only the `bench` agent calls Jev: from the Mac, `POST https://openrouter.ai/api/alpha/decisions`, model `typesafe/jev-1.13`, key from `~/.jev-local/secrets/openrouter.key`.
   - Never print, log or copy the key.
   - At most 6 concurrent requests. Cache everything.
   - Jev output is for evaluation ONLY. It never goes into training, selection, calibration or filtering (TypeSafe MCA §2.3(b)).
4. **No Claude/LLM-API-generated training data.** Label descriptions and synthetic text come from self-hosted Apache-2.0 open weights running on our VMs, or from programmatic generation (PLAN §2.4, D1 = no).
5. **Decontamination is sacred.** Nothing from jevbench test or dev, or their siblings (PLAN §1.6), may enter training. The registry `jev_local/bench/registry.py` (owned by `bench`) is the single source of truth.
   - Until it exists, code against PLAN §1.1, §1.2 and §1.6.
   - Every converter applies **stage 1** (source-name exclusion) itself.
   - Stages 2–5 run once, globally, in phase 2.
6. **Licences.** Record `license_use` per source: `commercial`, `research` or `unknown`. Keep non-commercial sources, but tag them.
7. **Cost.** The VMs bill by the hour. Don't idle-loop. Finish, write your report, return.

## Data layout (on the `data` VM, `~/jev/data/v2/`)
```
raw/<bucket>/<source>.jsonl.zst      # rows from jev_local.data.v2.render.make_example (v1 example format + provenance)
raw/<bucket>/<source>.stats.json     # {"rows", "decisions", "kinds":{choice,noul,score}, "k_hist", "tokens_est", "license_use", "excluded_stage1", "seconds"}
labels/descriptions.jsonl            # {"label", "context", "descriptions":[...3]} (data-label agent)
```
- **Buckets** (PLAN §2.1): `b1_tasksource_jev`, `b2_label_semantics`, `b3_nli`, `b4_procedural`, `b5_open_jev`, `b6_synthetic` (later), `b7_laurer`, `b8_extractive`, `b9_cu`, plus `s0_families`.
- **Splits.** Each source writes a ~1% `dev_mix` split (tag `split: "dev_mix"`) for loss tracking. Programmatic and synthetic families also hold out whole families or workflows (`split: "dev_family"`).
- **Shared rendering.** Everything goes through `jev_local/data/v2/render.py`, which is owned by the lead. Ask for changes in your report; don't fork it.

## Ownership

| Agent | Owns |
|---|---|
| bench | `jev_local/bench/**`, `scripts/jevbench.py`, `bench/` (manifests), `tests/test_bench_*.py` |
| data-real | `jev_local/data/v2/{sources.py,tasksource_jev.py,nli.py,open_jev.py,laurer.py}`, `tests/test_data_v2_real*.py` |
| data-label | `jev_local/data/v2/{label_semantics.py,extractive.py,label_desc.py}`, `tests/test_data_v2_label*.py` |
| data-synth | `jev_local/data/v2/families/**`, `jev_local/data/v2/{procedural.py,cu.py,probe.py}`, `tests/test_data_v2_synth*.py` |
| engine | `jev_local/train/{ddp.py,stream.py}`, edits to `jev_local/train/{train.py,losses.py,eval.py}`, `jev_local/engine/encoder/{calibrate.py,engine.py}`, `tests/test_train_v2*.py` |
| phase-2 decontam | `jev_local/data/v2/{decontam.py,assemble.py}`, `tests/test_data_v2_decontam*.py` |

Don't edit files you don't own; list needed changes in your report instead. The exception is the phase-2 integrator, which may make minimal fixes.

## Reports
Each agent writes `docs/v2/build/report-<agent>.md` (incrementally, so a crash keeps progress) and returns the same content. Include:
- files;
- API;
- commands to reproduce;
- measured numbers (rows, tokens, timings);
- test results (run on a VM unless pure-Python);
- known gaps;
- requests for other owners.

## Status notes from the lead (live)
- **2026-10-01 21:15: the OpenRouter key is out of credits.** Every Jev call now returns HTTP 402 ("Insufficient credits").
  - 4,420 Jev answers were cached in `runs/jevbench/m0/jev.jsonl` before that; 20,818 rows there are 402 failures.
  - Stop Jev runs on the first 402. Never count 402s as wrong Jev answers.
  - Mark Jev coverage as partial in any report. No "we beat Jev" table may be published from partial coverage.
  - The runner's orphaned process spun at 100% CPU on the Mac after its parent died. Runners must exit cleanly on parent death and on systemic errors.
