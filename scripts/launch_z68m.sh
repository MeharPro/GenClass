#!/bin/bash
# Phase 2: launch the clean zero-shot retrain (meharsjev-68m-z) on the 12-node cluster.
# Steps: sync code -> distribute the checksummed Z bundle -> put the run mixture -> launch torchrun -> start the watcher.
# Idempotent: nodes that already have the bundle skip the download. Run from the repo root on the Mac (ssh only).
set -euo pipefail
cd "$(dirname "$0")/.."
NODES=(c01 c02 c03 c04 c05 c06 c07 c08 c09 c10 c11)
BUNDLE_SHA=a333fd88d5eb1bb743514776bdca2d3781a3ed46bd3b7a5ae227abbbb9bd4a0b
RUN=z-68m

echo "== waiting for ssh on all nodes"
for n in "${NODES[@]}" data; do
  for t in $(seq 1 30); do scripts/azvm.sh "$n" true 2>/dev/null && break; sleep 10; done
  scripts/azvm.sh "$n" true 2>/dev/null || { echo "node $n unreachable"; exit 1; }
done

echo "== syncing code (W8 serialize change etc.) to 12 nodes"
for n in "${NODES[@]}" data; do scripts/azvm.sh "$n" --sync & done; wait

echo "== distributing the Z bundle (sha256-checked) + run mixture"
for n in "${NODES[@]}"; do
  ( scripts/azvm.sh "$n" "cd ~ && if [ \$(ls ~/jev/data/bm/z/shards/train 2>/dev/null | wc -l) -ge 81 ]; then echo '$n: bundle present'; else \
      curl -sS --fail http://10.0.0.5:8797/z_bundle.tar -o /tmp/z_bundle.tar && echo '$BUNDLE_SHA  /tmp/z_bundle.tar' | sha256sum -c --quiet \
      && tar -xf /tmp/z_bundle.tar -C ~ && rm /tmp/z_bundle.tar && echo \"$n: \$(ls ~/jev/data/bm/z/shards/train | wc -l) shard files\"; fi" \
    && scripts/azvm.sh "$n" --put bench/public/mix_z_run.json jev/data/bm/z/shards/train/mix_z_run.json ) &
done; wait

echo "== launching $RUN (8 nodes x 4 ranks x 20 threads; rank 0 = c01 10.0.0.6)"
# Measured 2026-10-03 for this 68m trainer: 8 nodes/4x20 = 43k tok/s (sync ~3-5 s/step); 12 nodes/4x20 = 40k
# (sync 12 s: barrier wait grows with rank count); 12 nodes/2x40 = 21k (poor 40-thread scaling). So 8 nodes.
scripts/launch_run.sh "$RUN" 10.0.0.6 4 20 "c01 c02 c03 c04 c05 c06 c07 c08" -- \
  --stream data/bm/z/shards/train --mixture data/bm/z/shards/train/mix_z_run.json --passes 2.5 \
  --stream-cache runs/stream_cache_z --index-workers 16 \
  --base models/base/ettin-encoder-68m --out models/meharsjev-68m-z --runs-dir runs \
  --max-len 8192 --batch-tokens 12288 --grad-accum 2 --loss v2 --lr 1e-4 --head-lr 3e-4 --warmup 0.03 \
  --device cpu --amp --no-grad-ckpt --ckpt-every 150 --log-every 20 --seed 0 --resume --ddp-compress bf16
echo "== launched; watch with: scripts/watch_runs.sh 300 \"$RUN:c01\""
