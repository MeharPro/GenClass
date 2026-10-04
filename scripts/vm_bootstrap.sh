#!/bin/bash
# One-time setup of a fresh Ubuntu 24.04 jev VM (run ON the VM). CPU-only torch; data tooling.
set -euo pipefail
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3.12-venv python3-pip rsync zstd pigz htop >/dev/null
mkdir -p ~/jev && cd ~/jev
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -q transformers tokenizers safetensors numpy rapidfuzz "pydantic>=2" huggingface_hub pytest \
    datasets pyarrow datasketch scikit-learn scipy httpx tqdm
[ -f pyproject.toml ] && .venv/bin/pip install -q -e . --no-deps
.venv/bin/python -c "import torch, datasets; print('torch', torch.__version__, 'threads', torch.get_num_threads(), 'datasets', datasets.__version__)"
