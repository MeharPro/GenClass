#!/bin/zsh
# Offline mid-sentence proof with the REAL model (dry-run, simulated screen, no microphone).
# Runs each heavy step only if the memory gate passes (RAM free >= 2.5 GB, swap-ins < 300/s),
# one heavy process at a time. Reports land in docs/build/demo-proof/.
#
#   scripts/demo_proof.sh            # text replay + audio replay (v1, CPU, base.en)
#   MODEL=v2 scripts/demo_proof.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=docs/build/demo-proof
MODEL=${MODEL:-v1}
ENGINE=${ENGINE:-fast}  # ENGINE=rule: no-weights pipeline check
mkdir -p "$OUT"

gate() {
  $PY scripts/overnight.py --check | $PY -c '
import json, sys
h = json.load(sys.stdin)["health"]
ok = h["ram_free_gb"] >= 2.5 and h["swapins_per_s"] < 300
print("memory gate: ram_free_gb=%s swapins_per_s=%s -> %s" % (h["ram_free_gb"], h["swapins_per_s"], "PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)'
}

CMDS=(
  "open textedit and type hello world"
  "open safari and go to wikipedia dot org"
  "scroll down a bit"
  "open notes"
  "quit textedit"
  "confirm"
  "hey can you pass the salt"
)

text_args=(); say_args=()
for c in "${CMDS[@]}"; do text_args+=(--text "$c"); say_args+=(--say "$c"); done

if gate; then
  $PY scripts/demo.py --engine "$ENGINE" --model "$MODEL" --device cpu --no-log --no-killswitch \
      --report "$OUT/text-$MODEL.json" "${text_args[@]}" 2>&1 | tee "$OUT/text-$MODEL.log"
else
  echo "skipped text replay (memory gate)"
fi

if gate; then
  $PY scripts/demo.py --engine "$ENGINE" --model "$MODEL" --device cpu --whisper-model base.en --no-log --no-killswitch \
      --report "$OUT/audio-$MODEL.json" "${say_args[@]}" 2>&1 | tee "$OUT/audio-$MODEL.log"
else
  echo "skipped audio replay (memory gate)"
fi
