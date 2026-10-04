#!/bin/bash
# Manage the OpenRouter key used for Jev evaluations, and check its balance.
#   scripts/openrouter_key.sh set     # paste a new key at a hidden prompt; saves it and shows the balance
#   scripts/openrouter_key.sh check   # show the balance of the saved key
# The key lives in ~/.jev-local/secrets/openrouter.key (readable only by you). It is never printed,
# and it is sent to curl on stdin so it doesn't show up in the process list.
set -euo pipefail
KEY_FILE="$HOME/.jev-local/secrets/openrouter.key"

check() {
  [ -s "$KEY_FILE" ] || { echo "No key saved yet. Run: scripts/openrouter_key.sh set"; exit 1; }
  local key; key=$(tr -d '[:space:]' < "$KEY_FILE")
  local credits keyinfo
  credits=$(printf 'Authorization: Bearer %s\n' "$key" | curl -s --max-time 15 -H @- https://openrouter.ai/api/v1/credits || true)
  keyinfo=$(printf 'Authorization: Bearer %s\n' "$key" | curl -s --max-time 15 -H @- https://openrouter.ai/api/v1/key || true)
  CREDITS="$credits" KEYINFO="$keyinfo" python3 - <<'EOF'
import json, os
def load(name):
    try:
        return json.loads(os.environ.get(name) or "{}")
    except ValueError:
        return {}
c, k = load("CREDITS"), load("KEYINFO")
if "error" in c or "error" in k:
    err = (c.get("error") or k.get("error") or {})
    print(f"OpenRouter rejected the key: {err.get('message', err)}")
    raise SystemExit(1)
d, kd = c.get("data", {}), k.get("data", {})
total, used = d.get("total_credits"), d.get("total_usage")
if total is not None and used is not None:
    print(f"Account credits:  ${total:,.2f} purchased, ${used:,.4f} used, ${total - used:,.4f} remaining")
if kd:
    limit = kd.get("limit")
    print(f"This key:         label {kd.get('label') or '-'}, usage ${kd.get('usage', 0):,.4f}, "
          f"limit {'none' if limit is None else f'${limit:,.2f}'}, free tier: {kd.get('is_free_tier')}")
    if kd.get("limit_remaining") is not None:
        print(f"Key limit left:   ${kd['limit_remaining']:,.4f}")
if total is not None and used is not None:
    left = total - used
    print("A full jevbench Jev pass costs about $4.50." + (" You have enough." if left >= 5 else " Add credit at openrouter.ai/settings/credits."))
EOF
}

case "${1:-check}" in
  set)
    read -rsp "Paste the new OpenRouter key (input hidden), then press Enter: " key; echo
    key=$(printf '%s' "$key" | tr -d '[:space:]')
    [[ "$key" == sk-or-* ]] || { echo "That doesn't look like an OpenRouter key (it should start with sk-or-). Nothing saved."; exit 1; }
    mkdir -p "$(dirname "$KEY_FILE")"; chmod 700 "$(dirname "$KEY_FILE")"
    (umask 077; printf '%s' "$key" > "$KEY_FILE.tmp" && mv "$KEY_FILE.tmp" "$KEY_FILE")
    echo "Saved. Checking the balance..."
    check
    ;;
  check) check ;;
  *) echo "usage: scripts/openrouter_key.sh [set|check]"; exit 2 ;;
esac
