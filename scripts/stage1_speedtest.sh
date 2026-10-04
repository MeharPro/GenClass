#!/bin/bash
# PLAN-excel Stage 1 gate: 20-step 8-node DDP speed tests of the 68m trainer on the clean Z mixture, with
# per-rank timing logs (--timing-log). Run from the Mac after c01-c08 are started (serial az) and synced.
#   scripts/stage1_speedtest.sh RUN "<extra train args>"
# Waits for the run to finish, then copies every node's timing/rank*.jsonl + rank0 log to runs/stage1/<RUN>/.
set -euo pipefail
RUN="$1"; EXTRA="$2"
NODES="c01 c02 c03 c04 c05 c06 c07 c08"
cd "$(dirname "$0")/.."
# shellcheck disable=SC2086
scripts/launch_run.sh "$RUN" 10.0.0.6 4 20 "$NODES" -- \
  --stream data/bm/z/shards/train --mixture data/bm/z/shards/train/mix_z_run.json --passes 2.5 \
  --stream-cache runs/stream_cache_z --index-workers 16 \
  --base models/base/ettin-encoder-68m --out "runs/$RUN/model" --runs-dir runs \
  --max-len 8192 --batch-tokens 12288 --loss v2 --lr 1e-4 --head-lr 3e-4 --warmup 0.03 \
  --device cpu --amp --no-grad-ckpt --ckpt-every 100000 --log-every 5 --seed 0 --ddp-compress bf16 \
  --max-steps 20 --timing-log $EXTRA
echo "== $RUN launched $(date -u +%T); waiting"
for i in $(seq 1 240); do
  sleep 15
  if ! scripts/azvm.sh c01 "pgrep -f 'run-name $RUN[ ]' >/dev/null"; then break; fi
done
echo "== $RUN finished $(date -u +%T)"
mkdir -p "runs/stage1/$RUN"
for n in $NODES; do
  mkdir -p "runs/stage1/$RUN/$n"
  scripts/azvm.sh "$n" --get "jev/runs/$RUN/timing" "runs/stage1/$RUN/$n/" || true
done
scripts/azvm.sh c01 --get "jev/runs/$RUN/log.jsonl" "runs/stage1/$RUN/log.jsonl" || true
scripts/azvm.sh c01 "tail -5 ~/jev/runs/$RUN/rank0.out" || true
