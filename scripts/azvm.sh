#!/bin/bash
# Run commands on / copy files to the jev Azure VMs by name (hosts in ~/.jev-local/azure_hosts).
#   scripts/azvm.sh HOST 'remote command'
#   scripts/azvm.sh HOST --put LOCAL REMOTE        # scp a file
#   scripts/azvm.sh HOST --get REMOTE LOCAL        # scp back (recursive)
#   scripts/azvm.sh HOST --sync                    # rsync code (jev_local, scripts, tests, pyproject) to ~/jev
# Mac safety: this is ssh/scp/rsync only — never run heavy work locally.
set -euo pipefail
H="${1:?host name}"; shift
IP=$(awk -v h="$H" '$1==h {print $2}' "$HOME/.jev-local/azure_hosts")
[ -n "$IP" ] || { echo "unknown host $H (see ~/.jev-local/azure_hosts)" >&2; exit 2; }
OPTS=(-i "$HOME/.ssh/jev_azure" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
case "${1:-}" in
  --put) exec scp -q "${OPTS[@]}" "$2" "azureuser@$IP:$3" ;;
  --get) exec scp -q -r "${OPTS[@]}" "azureuser@$IP:$2" "$3" ;;
  --sync) exec rsync -az --delete -e "ssh ${OPTS[*]}" --exclude '__pycache__' --exclude '.DS_Store' \
            "$ROOT/jev_local" "$ROOT/scripts" "$ROOT/tests" "$ROOT/pyproject.toml" "azureuser@$IP:jev/" ;;
  *) exec ssh "${OPTS[@]}" "azureuser@$IP" "$@" ;;
esac
