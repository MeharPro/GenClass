#!/bin/bash
# Poll cluster runs; print a status line every interval; EXIT (so the caller is notified) when any run finishes or errors.
# Usage: scripts/watch_runs.sh INTERVAL_S "run:node ..."   (node = the node holding rank 0's log)
INT="$1"; shift; SPECS=($1)
while true; do
  line="$(date +%H:%M)"; event=""
  for s in "${SPECS[@]}"; do
    run="${s%%:*}"; node="${s##*:}"
    st=$(scripts/azvm.sh "$node" "cd ~/jev/runs/$run 2>/dev/null || exit 0;
      if grep -q -E 'Traceback|Error|error:' rank0.out 2>/dev/null && ! pgrep -f 'run-name $run' >/dev/null; then echo ERROR; tail -5 rank0.out | cut -c1-200; exit 0; fi
      if grep -q '\"interrupted\": false' rank0.out 2>/dev/null; then echo DONE; exit 0; fi
      pgrep -f 'run-name $run' >/dev/null || { echo STOPPED; tail -3 rank0.out | cut -c1-200; exit 0; }
      tail -1 log.jsonl 2>/dev/null | python3 -c 'import sys,json; r=json.loads(sys.stdin.read()); print(\"step\",r[\"step\"],\"loss\",round(r[\"loss\"],3),\"tok/s\",int(r[\"tokens_per_s\"]))'" 2>&1)
    case "$st" in ERROR*|DONE*|STOPPED*) event="$event\n[$run] $st";; esac
    line="$line | $run: $(echo "$st" | head -1)"
  done
  echo "$line"
  if [ -n "$event" ]; then echo -e "EVENT:$event"; exit 0; fi
  sleep "$INT"
done
