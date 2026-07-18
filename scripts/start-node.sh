#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
env_file="${1:-$root/config/head.env}"
[[ -f "$env_file" ]] || { echo "Missing environment file: $env_file" >&2; exit 1; }
if grep -q 'CHANGEME' "$env_file"; then
  echo "Replace every CHANGEME value in $env_file first." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$env_file"
set +a

for name in NODE_RANK MASTER_ADDR MASTER_PORT VLLM_HOST_IP NCCL_IB_HCA NCCL_SOCKET_IFNAME DSPARK_MODEL_HOST DSPARK_VLLM_IMAGE; do
  [[ -n "${!name:-}" ]] || { echo "$name is required in $env_file" >&2; exit 1; }
done
[[ -d "$DSPARK_MODEL_HOST" ]] || { echo "Model directory is missing: $DSPARK_MODEL_HOST" >&2; exit 1; }

# Preserve the released single-checkpoint behavior unless a candidate opts in
# to a separate native DSpark draft checkpoint. Exporting the resolved value
# ensures Compose interpolation sees it even when an older node env file does
# not yet contain DSPARK_DRAFT_MODEL_HOST.
export DSPARK_DRAFT_MODEL_HOST="${DSPARK_DRAFT_MODEL_HOST:-$DSPARK_MODEL_HOST}"
if [[ "${DSPARK_SPECULATION_MODE:-dspark}" == "dspark" ]]; then
  [[ -d "$DSPARK_DRAFT_MODEL_HOST" ]] || {
    echo "Draft model directory is missing: $DSPARK_DRAFT_MODEL_HOST" >&2
    exit 1
  }
fi

ids="$(sudo docker ps -aq --filter label=com.docker.compose.service=vllm-dspark)"
if [[ -n "$ids" ]]; then
  # shellcheck disable=SC2086
  sudo docker rm -f $ids
fi

cd "$root"
COMPOSE_DISABLE_ENV_FILE=1 sudo -E docker compose \
  -p dspark-vllm-gx10 \
  --env-file "$env_file" \
  -f docker-compose.yml \
  up -d

sudo docker ps --filter label=com.docker.compose.service=vllm-dspark \
  --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
