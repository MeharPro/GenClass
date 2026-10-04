#!/bin/bash
# Run ON a fresh cluster node: venv + data/models bundle from the data VM (private network).
set -euo pipefail
bash ~/vm_bootstrap.sh >/tmp/bootstrap.log 2>&1 || { tail -5 /tmp/bootstrap.log; exit 1; }
cd ~/jev
curl -sS --fail http://10.0.0.5:8797/v2_bundle.tar | tar x
.venv/bin/python -c "import torch; print('ok', torch.__version__, torch.get_num_threads(), 'threads')"
du -sh data/v2 models/base | tr '\n' ' '; echo; nproc
