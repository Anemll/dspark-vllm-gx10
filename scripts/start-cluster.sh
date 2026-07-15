#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
head_env="${1:-$root/config/head.env}"
worker_env="${2:-config/worker.env}"
[[ -f "$head_env" ]] || { echo "Missing $head_env" >&2; exit 1; }

set -a
# shellcheck disable=SC1090
source "$head_env"
set +a
: "${WORKER_SSH:?WORKER_SSH is required in the head environment}"
: "${WORKER_REPO_DIR:?WORKER_REPO_DIR is required in the head environment}"

echo "Starting TP rank 1 on $WORKER_SSH..."
ssh -o BatchMode=yes "$WORKER_SSH" \
  "cd '$WORKER_REPO_DIR' && ./scripts/start-node.sh '$worker_env'"

echo "Waiting 12 seconds for rank 1 to enter the rendezvous..."
sleep 12

echo "Starting TP rank 0..."
"$root/scripts/start-node.sh" "$head_env"

for _ in $(seq 1 120); do
  if curl -fsS --max-time 3 http://127.0.0.1:8888/health >/dev/null; then
    echo "vLLM is ready: http://127.0.0.1:8888"
    curl -fsS http://127.0.0.1:8888/version
    echo
    exit 0
  fi
  sleep 5
done
echo "Timed out waiting for vLLM readiness." >&2
exit 1
