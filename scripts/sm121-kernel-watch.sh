#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
#
# Cheap, read-only release/PR watch for SM121 NVFP4 decode kernels.
# This script does not touch Docker, CUDA, checkpoints, or either GX10 node.
set -euo pipefail

PINNED_FLASHINFER_COMMIT=0472b9b3f2fba11b463f8526f390297d52a8aad7
PINNED_FLASHINFER_VERSION=0.6.15

fetch_json() {
  curl -fsS --retry 2 --connect-timeout 10 --max-time 30 "$1"
}

pypi_version() {
  local payload
  payload="$(fetch_json "https://pypi.org/pypi/$1/json" 2>/dev/null)" || return 1
  python3 -c '
import json
import sys

payload = json.load(sys.stdin)
print(payload["info"]["version"])
' <<<"${payload}"
}

pypi_recent_uploads() {
  local package="$1"
  local payload
  payload="$(
    fetch_json "https://pypi.org/pypi/${package}/json" 2>/dev/null
  )" || return 1
  python3 -c '
import json
import sys

payload = json.load(sys.stdin)
rows = []
for version, files in payload.get("releases", {}).items():
    uploads = [
        item.get("upload_time_iso_8601") or item.get("upload_time")
        for item in files
    ]
    uploads = [value for value in uploads if value]
    if uploads:
        rows.append((max(uploads), version))
for uploaded, version in sorted(rows, reverse=True)[:5]:
    print(f"  {uploaded}  {version}")
' <<<"${payload}"
}

github_item() {
  local repository="$1"
  local number="$2"
  fetch_json "https://api.github.com/repos/${repository}/issues/${number}" |
    python3 -c '
import json
import sys

item = json.load(sys.stdin)
kind = "PR" if "pull_request" in item else "issue"
number = item["number"]
state = item["state"]
updated = item["updated_at"]
title = item["title"]
print(
    f"{kind} #{number}: state={state} updated={updated} title={title}"
)
'
}

github_pull() {
  local repository="$1"
  local number="$2"
  fetch_json "https://api.github.com/repos/${repository}/pulls/${number}" |
    python3 -c '
import json
import sys

pull = json.load(sys.stdin)
number = pull["number"]
state = pull["state"]
merged = pull["merged"]
merged_at = pull["merged_at"]
head_sha = pull["head"]["sha"]
updated = pull["updated_at"]
title = pull["title"]
print(
    f"PR #{number}: state={state} merged={merged} merged_at={merged_at} "
    f"head={head_sha} updated={updated} title={title}"
)
'
}

echo "== SM121 NVFP4 kernel watch: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Pinned FlashInfer: ${PINNED_FLASHINFER_VERSION} ${PINNED_FLASHINFER_COMMIT}"
echo

echo "-- Package indexes"
for package in \
  flashinfer-python flashinfer-cubin flashinfer-jit-cache \
  tensorrt-llm nvidia-cutlass-dsl
do
  printf "%-24s " "${package}:"
  pypi_version "$package" || echo unavailable
done
echo "Recent FlashInfer uploads:"
pypi_recent_uploads flashinfer-python || echo "  unavailable"
echo "Recent TensorRT-LLM uploads:"
pypi_recent_uploads tensorrt-llm || echo "  unavailable"
echo

echo "-- FlashInfer"
github_item flashinfer-ai/flashinfer 3170
github_item flashinfer-ai/flashinfer 4003
github_pull flashinfer-ai/flashinfer 4010
github_pull flashinfer-ai/flashinfer 4038
github_pull flashinfer-ai/flashinfer 4057
echo

echo "-- TensorRT-LLM"
github_pull NVIDIA/TensorRT-LLM 11997
github_pull NVIDIA/TensorRT-LLM 12704
echo

echo "-- FlashInfer kernel-path delta after the pin"
fetch_json \
  "https://api.github.com/repos/flashinfer-ai/flashinfer/compare/${PINNED_FLASHINFER_COMMIT}...main" |
  python3 -c '
import json
import sys

payload = json.load(sys.stdin)
prefixes = (
    "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/",
    "csrc/fused_moe/cutlass_backend/",
    "include/flashinfer/trtllm/fused_moe/",
)
files = [
    (entry["status"], entry["filename"])
    for entry in payload.get("files", [])
    if entry["filename"].startswith(prefixes)
]
compare_status = payload.get("status")
ahead_by = payload.get("ahead_by")
commits = payload.get("commits", [])
head_sha = commits[-1].get("sha") if commits else None
print(
    f"compare_status={compare_status} "
    f"ahead_by={ahead_by} head={head_sha} relevant_files={len(files)}"
)
for status, filename in files:
    print(f"  {status:8s} {filename}")
'

cat <<'EOF'

Interpretation:
  * FlashInfer #4010 is a correctness/autotune repair, not a speed candidate.
  * FlashInfer #4038 is BF16 x FP4 W4A16 dense decode. It is adjacent
    evidence for split-K tactics, not a drop-in FP4 x FP4 W4A4 MoE kernel.
  * TensorRT-LLM #12704 filters invalid SM121 CUTLASS tactics. It improves
    robustness/fallback selection but does not itself add TRTLLMGen SM121
    fused-MoE cubins.
  * FlashInfer #4057 adds caller-owned CUTLASS-MoE workspace reuse. It can
    remove allocator churn only after the serving integration allocates and
    passes the reusable buffer; its presence alone is not a kernel speedup.
    Under CUDA-graph replay, require evidence that allocation/setup remains
    in the measured hot path before spending a component window.
  * CUTLASS 4.6 adds FP8 ptr-array grouped collectives and dense blockscaled
    tileN=8/16 kernels. A release must explicitly integrate equivalent NVFP4
    grouped-MoE kernels before it is a W4A4 serving candidate.

Only spend a GX10 component window when a candidate changes an SM12x
NVFP4-MoE kernel path, enables native SM121 TRTLLMGen fused MoE, or wires
caller-owned workspace reuse into the measured serving hot path.
Required gate: >=3% over the banked CUTLASS control at both M=24 and M=48,
plus the established numerical and CUDA-graph gates, before TP=2 serving.
EOF
