#!/usr/bin/env bash
# bm-integrate: ONE end-to-end dry pass of every registered benchmax spec through scripts/benchmax.py on VM train,
# against a checkpoint, on NON-EVALUATED items only (validation / dev / train splits; DI: synthetic smoke rows,
# since the suite has no non-test split). Eval-only suites (asevlad_injection, mbburabak_safety) get `verify`
# (prepare + counts) only. Never touches a test split (PLAN §2.2.2).
#
#   scripts/azvm.sh train --sync
#   scripts/azvm.sh train 'cd ~/jev && setsid nohup nice -n 5 bash scripts/benchmax/dry_pass_validation.sh \
#        ~/jev/models/jev-local-fast-v2 runs/benchmax/jev-local-fast-v2/dry 16 > runs/benchmax/logs/dry_pass.log 2>&1 < /dev/null &'
#
# Each spec writes <OUT>/<name>/run.json (schema benchmax-run/1) and <OUT>/logs/<name>.log.
set -u
CK=${1:-$HOME/jev/models/jev-local-fast-v2}
OUT=${2:-runs/benchmax/$(basename "$CK")/dry}
THREADS=${3:-16}
cd ~/jev
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$THREADS TOKENIZERS_PARALLELISM=false
PY=~/jev/.venv/bin/python
mkdir -p "$OUT/logs"
SUMMARY="$OUT/summary.tsv"
echo -e "name\texit\tstatus\tsent\tok\trefused\terrors\ttruncated\tdeterminism\twall_s" > "$SUMMARY"

run() {  # run NAME <benchmax.py args...>
  local name=$1; shift
  echo "=== $name start $(date -u +%FT%TZ)"
  "$PY" scripts/benchmax.py "$@" --out "$OUT/$name" > "$OUT/logs/$name.log" 2>&1
  local rc=$?
  echo "=== $name exit=$rc $(date -u +%FT%TZ)"
  "$PY" - "$name" "$rc" "$OUT/$name/run.json" >> "$SUMMARY" <<'EOF'
import json, sys
name, rc, path = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.load(open(path))
    r = d.get("requests", {}); det = d.get("determinism", {})
    print("\t".join(str(x) for x in (name, rc, d.get("status"), r.get("sent"), r.get("ok"), r.get("refused"), r.get("errors"), r.get("truncated"),
                                     det.get("identical"), d.get("wall_clock_s"))))
except Exception as e:
    print("\t".join(str(x) for x in (name, rc, f"no run.json ({type(e).__name__})", "", "", "", "", "", "", "")))
EOF
}

verify() {  # verify NAME <benchmax.py args...>  (prepare + counts, no model)
  local name=$1; shift
  echo "=== $name verify start $(date -u +%FT%TZ)"
  "$PY" scripts/benchmax.py verify "$@" --out "$OUT/$name" > "$OUT/logs/$name.log" 2>&1
  local rc=$?
  echo "=== $name verify exit=$rc $(date -u +%FT%TZ)"
  echo -e "$name\t$rc\tverify-only (eval-only suite: no non-evaluated split)\t\t\t\t\t\t\t" >> "$SUMMARY"
}

COMMON=(--ckpt "$CK" --threads "$THREADS" --model-id meharsjev-68m)

# ---- group A (adapters-a)
run deusser_exact        run --spec deusser_exact@6bbdeb33 "${COMMON[@]}" --tasks self-check --split dev --limit 5 --no-ci
run decision_index_smoke run --spec decision_index_0.2.1@87d4650b "${COMMON[@]}" --di-step run --di-engine inproc --rows ~/bench_work/di-smoke-rows.jsonl.gz
SC=~/bench_work/typed_decisions_scorer.json
if [ -f "$SC" ]; then
  run typed_decisions    run --spec typed_decisions_card@d0e2f0c4 "${COMMON[@]}" --split train --limit 8 --scorer-config "$SC" --allow-unpinned
else
  run typed_decisions    run --spec typed_decisions_card@d0e2f0c4 "${COMMON[@]}" --split train --limit 8 --allow-unpinned
fi
run jevbench_hf          run --spec jevbench_hf_praveenrajus_v0.1.1 "${COMMON[@]}" --split validation --configs sst5 boolq banking77 paws --limit 5

# ---- group B (adapters-b, through bridge_b); default split = the first non-evaluated one
run nslkdd      run --spec study:jev_ids_arxiv_2610_01079 "${COMMON[@]}" --limit 10
run trec50      run --spec study:do_system_one_decisions_add_up_arxiv_2609_33971 "${COMMON[@]}" --limit 10
run elcronos    run --spec elcronos_plain "${COMMON[@]}" --limit 10
run thisisandreeeee run --spec thisisandreeeee "${COMMON[@]}" --limit 10
run zhuyansen   run --spec zhuyansen_batch20 "${COMMON[@]}" --limit 2 --tasks agnews,sst2,banking77,tweet_emotion,paws
run cfpb        run --spec study:earino_zero_shot_complaint_benchmark_cfpb_113_cl "${COMMON[@]}" --limit 10
run koa_action  run --spec study:koa_action_arxiv_2609_36115 "${COMMON[@]}" --limit 10
run dmb         run --spec dmb_expanded@eabd88b0 "${COMMON[@]}" --limit 20
run chepyle_lexglue run --spec chepyle_lexglue_systemone_v1 "${COMMON[@]}" --limit 4 --fit
run rerank      run --spec rerank_scripts "${COMMON[@]}" --limit 4 --tasks denser_scifact,hev_scifact_pair,hev_scifact_batch,aness_nevir
run goya_rm     run --spec goya_rm_eval "${COMMON[@]}" --limit 10 --tasks rewardbench1
run stperic_medhallu run --spec stperic_medhallu "${COMMON[@]}" --limit 10
verify asevlad_injection  --spec asevlad_injection --model-id meharsjev-68m
verify mbburabak_safety   --spec mbburabak_safety --model-id meharsjev-68m
run cesnet      run --spec study:jev_for_network_traffic_classification_arxiv_261 "${COMMON[@]}" --limit 10
echo "=== ALL DONE $(date -u +%FT%TZ)"
cat "$SUMMARY"
