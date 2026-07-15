#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

usage() { echo "Usage: $0 --role {head|worker}" >&2; exit 2; }
[[ $# -eq 2 && "$1" == "--role" ]] || usage
role="$2"
[[ "$role" == "head" || "$role" == "worker" ]] || usage

root="$(cd "$(dirname "$0")/.." && pwd)"
source_file="$root/config/$role.env.example"
env_file="$root/config/$role.env"

command -v docker >/dev/null || { echo "Docker is required." >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "NVIDIA drivers are required." >&2; exit 1; }
[[ "$(uname -m)" == "aarch64" ]] || echo "Warning: this runtime is validated on ARM64 GB10 systems." >&2

if [[ ! -f "$env_file" ]]; then
  cp "$source_file" "$env_file"
  chmod 600 "$env_file"
  echo "Created $env_file"
else
  echo "Keeping existing $env_file"
fi

image="$(sed -n 's/^DSPARK_VLLM_IMAGE=//p' "$env_file" | tail -1)"
sudo docker pull "$image"
echo "Edit $env_file and replace every CHANGEME value before starting the node."
