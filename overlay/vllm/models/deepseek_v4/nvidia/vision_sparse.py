# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated FlashInfer image-prefill module; never replaces the text binary."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import torch


def get_vision_sparse_spec():
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import current_compilation_context, gen_jit_spec

    kernels = Path(__file__).with_name("vision_kernels")
    return gen_jit_spec(
        "dspark_dsv4_vision512_v1",
        [jit_env.FLASHINFER_CSRC_DIR / "sparse_mla_sm120.cu",
         kernels / "sparse_mla_sm120_prefill.cu", kernels / "binding.cu"],
        extra_cuda_cflags=current_compilation_context.get_nvcc_flags_list(
            supported_major_versions=[12]
        ),
    )


@lru_cache(maxsize=1)
def get_vision_sparse_module():
    import tvm_ffi

    kernels = Path(__file__).with_name("vision_kernels")
    binary = kernels / "dspark_dsv4_vision512.so"
    manifest_path = kernels / "manifest.json"
    if not binary.is_file() or not manifest_path.is_file():
        raise RuntimeError("Vision prefill module is not prebuilt; run scripts/build-vision-module.py during image build")
    manifest = json.loads(manifest_path.read_text())
    if hashlib.sha256(binary.read_bytes()).hexdigest() != manifest["sha256"]:
        raise RuntimeError("Vision prefill binary does not match its build manifest")
    arch = ".".join(map(str, torch.cuda.get_device_capability())) + "a"
    if arch not in manifest["cuda_archs"]:
        raise RuntimeError(f"Vision prefill binary was not built for {arch}")
    return tvm_ffi.load_module(str(binary))


def image_sparse_prefill(q, swa_cache, indices, out, scale, sinks, lengths,
                         extra_cache=None, extra_indices=None, extra_lengths=None):
    if q.shape[0] <= 64 or indices.shape[-1] != 512 or q.shape[1] not in (32, 64):
        raise ValueError("Vision module supports prefill T>64, H32/64, topk512 only")
    # Paged indices are produced dense; rejecting unexpected views prevents
    # hiding a broken metadata contract with a per-layer contiguous copy.
    if not indices.is_contiguous() or (
        extra_indices is not None and not extra_indices.is_contiguous()
    ):
        raise ValueError("Vision sparse indices must be dense contiguous rows")
    lse = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
    get_vision_sparse_module().sparse_mla_sm120_paged_attention(
        q, swa_cache, indices, out, lse, scale, 1, lengths, sinks,
        extra_cache, extra_indices, extra_lengths,
    )
