#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

tail_lines="${1:-}"
if [[ ! "$tail_lines" =~ ^[1-9][0-9]{0,3}$ ]] || (( tail_lines > 2000 )); then
  echo "usage: $0 <tail-lines:1..2000>" >&2
  exit 64
fi

mapfile -t container_ids < <(
  /usr/bin/docker ps -q \
    --filter label=com.docker.compose.service=vllm-dspark
)

if (( ${#container_ids[@]} != 1 )); then
  printf 'expected exactly one running vllm-dspark container, found %d\n' \
    "${#container_ids[@]}" >&2
  exit 69
fi

exec /usr/bin/docker logs --tail "$tail_lines" "${container_ids[0]}"
