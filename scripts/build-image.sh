#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$root/upstream.lock"
build_root="${BUILD_ROOT:-$root/.build}"
source_dir="$build_root/vllm"
base_image="${BASE_IMAGE:-dspark-vllm-gx10:vllm-base-$VLLM_TAG}"
final_image="${FINAL_IMAGE:-ghcr.io/anemll/dspark-vllm-gx10:0.1.0}"

mkdir -p "$build_root"
if [[ ! -d "$source_dir/.git" ]]; then
  git clone "$VLLM_REPOSITORY" "$source_dir"
fi
git -C "$source_dir" fetch --tags origin
git -C "$source_dir" checkout --detach "$VLLM_COMMIT"
git -C "$source_dir" reset --hard "$VLLM_COMMIT"
git -C "$source_dir" clean -fdx
rsync -a "$root/overlay/vllm/" "$source_dir/vllm/"

# vLLM 0.25 keeps its primary Dockerfile under docker/, not at repository root.
# Permit an override for future upstream layout changes, but fail clearly instead
# of letting Docker report a misleading missing-root-Dockerfile error.
upstream_dockerfile="${VLLM_DOCKERFILE:-$source_dir/docker/Dockerfile}"
if [[ ! -f "$upstream_dockerfile" ]]; then
  echo "vLLM Dockerfile not found: $upstream_dockerfile" >&2
  echo "Set VLLM_DOCKERFILE to the upstream Dockerfile path if its layout changed." >&2
  exit 1
fi

sudo docker build \
  --file "$upstream_dockerfile" \
  --target vllm-openai \
  --build-arg torch_cuda_arch_list=12.1a \
  --tag "$base_image" \
  "$source_dir"

sudo docker build \
  --file "$root/docker/Dockerfile.runtime" \
  --build-arg VLLM_BASE="$base_image" \
  --build-arg FLASHINFER_COMMIT="$FLASHINFER_COMMIT" \
  --build-arg B12X_COMMIT="$B12X_COMMIT" \
  --build-arg CUTLASS_DSL_VERSION="$CUTLASS_DSL_VERSION" \
  --build-arg CUDA_PYTHON_VERSION="$CUDA_PYTHON_VERSION" \
  --build-arg TVM_FFI_VERSION="$TVM_FFI_VERSION" \
  --tag "$final_image" \
  "$root"

echo "Built $final_image"
