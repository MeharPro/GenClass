#!/usr/bin/env bash
# PLAN-excel Stage 0: run one checkpoint through the publisher harness adapters on NON-EVALUATED items only
# (Deußer `dev`, group-B `validation`, typed-decisions/jev-bench self-check splits). Never passes --allow-test.
# One eval at a time (sequential); caps per task keep the whole pass ~30-40 min per 68m checkpoint.
#
#   scripts/azvm.sh train 'cd ~/jev && setsid nohup nice -n 5 bash scripts/benchmax/stage0_validation.sh \
#        ~/jev/models/meharsjev-68m-z runs/stage0/z-68m genclass-68m 32 > runs/stage0/logs/z-68m.log 2>&1 < /dev/null &'
#
# Writes <OUT>/<name>/run.json per run and <OUT>/summary.tsv. Skips a run whose run.json is already complete.
set -u
CK=${1:?ckpt dir}
OUT=${2:?out dir}
MID=${3:-genclass-68m}
THREADS=${4:-32}
LIMIT=${LIMIT:-1000}
cd ~/jev
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$THREADS TOKENIZERS_PARALLELISM=false
PY=~/jev/.venv/bin/python
mkdir -p "$OUT/logs"
SUMMARY="$OUT/summary.tsv"
[ -f "$SUMMARY" ] || echo -e "name\texit\tstatus\tsent\tok\trefused\terrors\ttruncated\twall_s" > "$SUMMARY"

run() {  # run NAME <benchmax.py run args...>
  local name=$1; shift
  if [ -f "$OUT/$name/run.json" ] && grep -q '"finished"' "$OUT/$name/run.json"; then echo "=== $name already done"; return; fi
  echo "=== $name start $(date -u +%FT%TZ)"
  timeout 3600 "$PY" scripts/benchmax.py run "$@" --ckpt "$CK" --threads "$THREADS" --model-id "$MID" --determinism 5 \
      --out "$OUT/$name" > "$OUT/logs/$name.log" 2>&1
  local rc=$?
  echo "=== $name exit=$rc $(date -u +%FT%TZ)"
  "$PY" - "$name" "$rc" "$OUT/$name/run.json" >> "$SUMMARY" <<'EOF'
import json, sys
name, rc, path = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.load(open(path)); r = d.get("requests", {})
    print("\t".join(str(x) for x in (name, rc, d.get("status"), r.get("sent"), r.get("ok"), r.get("refused"), r.get("errors"),
                                     r.get("truncated"), d.get("wall_clock_s"))))
except Exception as e:
    print("\t".join(str(x) for x in (name, rc, f"no run.json ({type(e).__name__})", "", "", "", "", "", "")))
EOF
}

# ---- Deußer (dev split = the harness's own train/validation items). Gated (llm_aggrefact, toxigen) and
#      eval-only tasks (sms_spam, belebele, pubmedqa, openai_moderation, summeval) have no usable dev split.
for t in ag_news imdb rotten_tomatoes sst2 emotion financial_phrasebank banking77 clinc150 sib200 language_id \
         go_emotions anli afrixnli paws boolq toxic_chat prompt_injections agb_de unfair_tos stsb sst5 helpsteer2; do
  run "deusser_$t" --spec deusser_exact@6bbdeb33 --tasks "$t" --split dev --limit "$LIMIT" --no-ci
done
for t in mmlu ceval bigbench hellaswag winogrande arc commonsense_qa art; do   # knowledge/reasoning: reported, not chased
  run "deusser_$t" --spec deusser_exact@6bbdeb33 --tasks "$t" --split dev --limit 300 --no-ci
done

# ---- group B through bridge_b, split=validation (adapter-defined non-evaluated items)
run elcronos        --spec elcronos_plain --split validation --limit "$LIMIT"
run dmb             --spec dmb_expanded@eabd88b0 --split validation --limit "$LIMIT"
run thisisandreeeee --spec thisisandreeeee --split validation --limit "$LIMIT"
run zhuyansen       --spec zhuyansen_batch20 --split validation --limit 200 --tasks agnews,sst2,banking77,tweet_emotion,paws
run cfpb            --spec study:earino_zero_shot_complaint_benchmark_cfpb_113_cl --split validation --limit "$LIMIT"
run trec50          --spec study:do_system_one_decisions_add_up_arxiv_2609_33971 --split validation --limit "$LIMIT"
run koa_action      --spec study:koa_action_arxiv_2609_36115 --split validation --limit "$LIMIT"
run nslkdd          --spec study:jev_ids_arxiv_2610_01079 --split validation
run cesnet          --spec study:jev_for_network_traffic_classification_arxiv_261 --split validation --limit "$LIMIT" --tasks k0
run stperic_medhallu --spec stperic_medhallu --split validation --limit 500
run chepyle_lexglue --spec chepyle_lexglue_systemone_v1 --split validation --limit 100
run goya_rm         --spec goya_rm_eval --split validation --limit 200 --tasks rewardbench1

# ---- group A self-check splits
SC=~/bench_work/typed_decisions_scorer.json
run typed_decisions --spec typed_decisions_card@d0e2f0c4 --split train --limit 200 --scorer-config "$SC" --allow-unpinned
run jevbench_hf     --spec jevbench_hf_praveenrajus_v0.1.1 --split validation --limit 500
echo "=== ALL DONE $(date -u +%FT%TZ)"
cat "$SUMMARY"
