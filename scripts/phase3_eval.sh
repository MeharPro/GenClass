#!/bin/bash
# Phase 3 runbook: evaluate a finished checkpoint through every benchmax harness (PLAN §5 phase 3 / §6).
# Usage: scripts/phase3_eval.sh RUN_NAME CKPT_NAME MODEL_ID [TRACK]
#   e.g. scripts/phase3_eval.sh z-68m meharsjev-68m-z meharsjev-68m Z
# Steps (each idempotent; rerun the script after a failure):
#   1. start the eval VM `train` (serial az; the lead runs this script from the Mac, one az at a time)
#   2. pull the checkpoint + training log from rank 0 (c01) onto `train`
#   3. dev-mix eval (uncalibrated) + fit the GLOBAL calibration file on the clean dev set (no by_header; W9)
#   4. run every spec via scripts/benchmax.py (Z track: global calibration only), detached with logs
#   5. the lead reads runs/benchmax/<ckpt>/*/run.json and writes the scoreboard (vs bench/public/targets.json)
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:?run name}"; CKPT="${2:?checkpoint dir name under ~/jev/models}"; MODEL_ID="${3:?model id, e.g. meharsjev-68m}"; TRACK="${4:-Z}"
RG=rg-jev-train

echo "== 1. eval VM up"
state=$(az vm get-instance-view -g $RG -n vm-jev-train --query "instanceView.statuses[1].displayStatus" -o tsv)
[ "$state" = "VM running" ] || az vm start -g $RG -n vm-jev-train -o none
for t in $(seq 1 30); do scripts/azvm.sh train true 2>/dev/null && break; sleep 10; done
scripts/azvm.sh train --sync

echo "== 2. checkpoint from c01 -> train (private network)"
scripts/azvm.sh c01 "cd ~/jev && tar cf /tmp/$CKPT.tar models/$CKPT runs/$RUN/log.jsonl runs/$RUN/mixture_plan.json && \
  (cd /tmp && setsid nohup timeout 7200 python3 -m http.server 8799 --bind 10.0.0.6 >/dev/null 2>&1 < /dev/null &) ; sleep 1; sha256sum /tmp/$CKPT.tar | cut -c1-16"
scripts/azvm.sh train "cd ~/jev && curl -sS --fail http://10.0.0.6:8799/$CKPT.tar -o /tmp/$CKPT.tar && tar xf /tmp/$CKPT.tar && rm /tmp/$CKPT.tar && ls models/$CKPT && tail -1 runs/$RUN/log.jsonl | cut -c1-200"

echo "== 3. dev eval + global calibration (clean dev set)"
scripts/azvm.sh train "cd ~/jev && mkdir -p runs/benchmax/calib && HF_HUB_OFFLINE=1 JEV_ENCODER=banded \
  .venv/bin/python -m jev_local.train.eval --ckpt models/$CKPT --stream data/bm/z/shards/dev_mix --split dev_mix --device cpu --batch 8 \
     --no-calib --out runs/benchmax/$CKPT-eval_dev_mix_uncal.json 2>&1 | tail -3; \
  .venv/bin/python -m jev_local.engine.encoder.calibrate --help 2>&1 | head -30"
echo ">>> CHECK the calibrate flags above, then fit the global file (by_bucket + tau_k + noul_platt, NO by_header) on the clean dev set:"
echo "    scripts/azvm.sh train 'cd ~/jev && .venv/bin/python -m jev_local.engine.encoder.calibrate --ckpt models/$CKPT ... --out runs/benchmax/calib/global-$CKPT.json'"

echo "== 4. run every spec (detached, serial, logs in runs/benchmax/$CKPT/logs/)"
scripts/azvm.sh train "cd ~/jev && mkdir -p runs/benchmax/$CKPT/logs && .venv/bin/python scripts/benchmax.py list 2>/dev/null | awk '{print \$1}' | grep -v '^\$' > runs/benchmax/$CKPT/specs.txt && cat runs/benchmax/$CKPT/specs.txt"
cat > /tmp/phase3_run_all.sh <<EOF
#!/bin/bash
# runs on VM train; one spec at a time; skips specs with a complete run.json
cd ~/jev
CALIB=runs/benchmax/calib/global-$CKPT.json
[ -f "\$CALIB" ] || { echo "missing \$CALIB"; exit 1; }
while read -r spec; do
  out=runs/benchmax/$CKPT/\$spec
  if [ -f "\$out/run.json" ] && grep -q '"complete": true' "\$out/run.json"; then echo "skip \$spec"; continue; fi
  echo "== \$spec \$(date -u +%H:%M)"
  HF_HUB_OFFLINE=1 JEV_ENCODER=banded .venv/bin/python scripts/benchmax.py run --spec "\$spec" --ckpt models/$CKPT \
     --model-id $MODEL_ID --calib "\$CALIB" --track $TRACK --threads 48 --out "\$out" > "runs/benchmax/$CKPT/logs/\$spec.log" 2>&1 || echo "FAILED \$spec"
done < runs/benchmax/$CKPT/specs.txt
echo ALL_SPECS_DONE
EOF
scripts/azvm.sh train --put /tmp/phase3_run_all.sh phase3_run_all.sh
echo ">>> after calibration exists, launch:  scripts/azvm.sh train 'setsid nohup bash ~/phase3_run_all.sh > ~/jev/runs/benchmax/$CKPT/logs/ALL.log 2>&1 < /dev/null &'"
