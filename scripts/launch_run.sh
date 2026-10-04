#!/bin/bash
# Start a (multi-node) torchrun training job on jev cluster nodes, detached (survives ssh drops).
# Usage: scripts/launch_run.sh RUN_NAME MASTER_PRIV_IP NPROC THREADS "node1 node2 ..." -- <train.py args...>
# Each node logs to ~/jev/runs/<RUN_NAME>/rank<node_rank>.out ; stop with scripts/azvm.sh <node> 'pkill -f run-name\ RUN_NAME'
set -euo pipefail
RUN="$1"; MASTER="$2"; NPROC="$3"; THREADS="$4"; NODES=($5); shift 5; [ "${1:-}" = "--" ] && shift
NN=${#NODES[@]}; R=0
for n in "${NODES[@]}"; do
  ARGS=$(printf '%q ' "$@")
  # ';' not '&&' before the job, so '&' backgrounds only the job and ssh can return immediately
  scripts/azvm.sh "$n" "cd ~/jev; mkdir -p runs/$RUN; OMP_NUM_THREADS=$THREADS GLOO_SOCKET_IFNAME=eth0 JEV_ENCODER=banded TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 \
    setsid nohup nice -n 5 .venv/bin/python -m torch.distributed.run --nnodes $NN --nproc-per-node $NPROC --node-rank $R \
    --master-addr $MASTER --master-port 29500 -m jev_local.train.train --ddp --threads $THREADS --run-name $RUN $ARGS \
    > runs/$RUN/rank$R.out 2>&1 < /dev/null & echo started $n rank $R"
  R=$((R+1))
done
