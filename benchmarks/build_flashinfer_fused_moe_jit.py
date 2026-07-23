#!/usr/bin/env python3
"""Build an isolated FlashInfer fused_moe_120 JIT module fail-closed."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


HEADER = Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer/data/csrc/"
    "fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh"
)
RUNNER_HEADER = Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer/data/csrc/"
    "nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-header-sha256", required=True)
    parser.add_argument("--expected-runner-header-sha256", required=True)
    parser.add_argument("--expected-workspace-suffix", default="121a")
    args = parser.parse_args()

    actual = sha256_file(HEADER)
    if actual != args.expected_header_sha256:
        raise RuntimeError(
            f"mounted FlashInfer header mismatch: {actual} != "
            f"{args.expected_header_sha256}"
        )
    actual_runner = sha256_file(RUNNER_HEADER)
    if actual_runner != args.expected_runner_header_sha256:
        raise RuntimeError(
            f"mounted FlashInfer runner header mismatch: {actual_runner} != "
            f"{args.expected_runner_header_sha256}"
        )
    if os.environ.get("FLASHINFER_JIT_DEBUG") != "0":
        raise RuntimeError("release build requires FLASHINFER_JIT_DEBUG=0")
    max_jobs = int(os.environ.get("MAX_JOBS", "0"))
    if max_jobs < 1:
        raise RuntimeError("release build requires an explicit positive MAX_JOBS")

    from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm120_module

    spec = gen_cutlass_fused_moe_sm120_module(use_fast_build=False)
    if spec.name != "fused_moe_120":
        raise RuntimeError(f"unexpected JIT module name: {spec.name}")
    if spec.is_aot:
        raise RuntimeError(f"AOT bypass failed: {spec.aot_path}")
    expected_tail = (
        f"/.cache/flashinfer/0.6.15/{args.expected_workspace_suffix}/"
        "cached_ops/fused_moe_120"
    )
    if not str(spec.build_dir).endswith(expected_tail):
        raise RuntimeError(f"unexpected isolated build directory: {spec.build_dir}")

    print(f"HEADER_SHA256={actual}", flush=True)
    print(f"RUNNER_HEADER_SHA256={actual_runner}", flush=True)
    print(f"MAX_JOBS={max_jobs}", flush=True)
    print(f"AOT_PATH={spec.aot_path}", flush=True)
    print(f"BUILD_DIR={spec.build_dir}", flush=True)
    print(f"SO_PATH={spec.jit_library_path}", flush=True)
    spec.build(verbose=True)
    if not spec.jit_library_path.is_file():
        raise RuntimeError(f"JIT module absent after build: {spec.jit_library_path}")
    module = spec.load(spec.jit_library_path)
    print(f"MODULE={module}", flush=True)
    print(f"SO_SHA256={sha256_file(spec.jit_library_path)}", flush=True)


if __name__ == "__main__":
    main()
