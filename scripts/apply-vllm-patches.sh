#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

source_dir="${1:?usage: apply-vllm-patches.sh VLLM_SOURCE_DIR}"
root="$(cd "$(dirname "$0")/.." && pwd)"

if ! git -C "$source_dir" rev-parse --git-dir >/dev/null 2>&1; then
  echo "vLLM source is not a git checkout: $source_dir" >&2
  exit 1
fi

shopt -s nullglob
patches=("$root"/patches/vllm/*.patch)
for patch in "${patches[@]}"; do
  echo "Applying vLLM patch: $(basename "$patch")"
  git -C "$source_dir" apply --check --whitespace=error-all "$patch"
  git -C "$source_dir" apply --whitespace=error-all "$patch"
done
