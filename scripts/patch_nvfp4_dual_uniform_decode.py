#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Propagate decode-uniformity into vLLM's forward context.

The NVFP4 dual expert uses ``BatchDescriptor.uniform`` to distinguish
uniform decode from prefill.  The pinned V2 runner computes the authoritative
``uniform_token_count`` but drops it when constructing the public forward
descriptor, and CUDA-graph capture omits the descriptor entirely for FULL
decode graphs.  Patch both content-addressed sources so an exact target-only
decode graph carries ``uniform=True`` only when its per-request token count
matches ``decode_query_len`` and no active request is still prefilling.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


MODEL_RUNNER = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "vllm/v1/worker/gpu/model_runner.py"
)
CUDAGRAPH_UTILS = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "vllm/v1/worker/gpu/cudagraph_utils.py"
)

MODEL_RUNNER_SOURCE_SHA256 = (
    "58f45c58969cdd9cba707863e82fefda818002de45c621032b58b6eb364deedf"
)
MODEL_RUNNER_PATCHED_SHA256 = (
    "61befb32cdc06e1c58383f9481e805d3b86637c84736f79b904c04a474df34e4"
)
CUDAGRAPH_UTILS_SOURCE_SHA256 = (
    "303d762141830cd8343976d5be14b34ef1666e7d1d459e089adfd4f5b8cd3ef6"
)
CUDAGRAPH_UTILS_PATCHED_SHA256 = (
    "56031f4d39147bc4cb8ee9cf7d1914d6811c677d15b8735a7d292862cba5da4c"
)

_MODEL_RUNNER_PREFILL_ANCHOR = """\
        uniform_tok_count = get_uniform_token_count(num_reqs, num_toks, max_query_len)
"""
_MODEL_RUNNER_PREFILL_REPLACEMENT = """\
        uniform_tok_count = get_uniform_token_count(num_reqs, num_toks, max_query_len)
        if not dummy_run and uniform_tok_count is not None:
            req_ids = tuple(scheduler_output.num_scheduled_tokens)
            idx_mapping_np = np.fromiter(
                (self.req_states.req_id_to_index[req_id] for req_id in req_ids),
                dtype=np.int32,
                count=num_reqs,
            )
            prefilling = (
                self.req_states.num_computed_prefill_tokens[idx_mapping_np]
                < self.req_states.prefill_len.np[idx_mapping_np]
            )
            if bool(np.any(prefilling)):
                uniform_tok_count = None
"""

_MODEL_RUNNER_DESCRIPTOR_ANCHOR = """\
            batch_descriptor = BatchDescriptor(
                num_tokens=input_batch.num_tokens_after_padding,
                has_lora=self.lora_config is not None,
                num_active_loras=batch_desc.num_active_loras,
            )
"""
_MODEL_RUNNER_DESCRIPTOR_REPLACEMENT = """\
            batch_descriptor = BatchDescriptor(
                num_tokens=input_batch.num_tokens_after_padding,
                num_reqs=num_reqs,
                uniform=uniform_tok_count == self.decode_query_len,
                has_lora=self.lora_config is not None,
                num_active_loras=batch_desc.num_active_loras,
            )
"""

_CUDAGRAPH_UTILS_ANCHOR = """\
            def forward_fn(cg_mode: CUDAGraphMode) -> None:
                batch_descriptor = None
                if cg_mode == CUDAGraphMode.PIECEWISE:
                    assert attn_metadata is None
                    batch_descriptor = BatchDescriptor(
                        num_tokens=num_tokens,
                        has_lora=has_lora,
                        num_active_loras=desc.num_active_loras,
                    )
"""
_CUDAGRAPH_UTILS_REPLACEMENT = """\
            def forward_fn(cg_mode: CUDAGraphMode) -> None:
                batch_descriptor = None
                if cg_mode == CUDAGraphMode.FULL:
                    batch_descriptor = BatchDescriptor(
                        num_tokens=num_tokens,
                        num_reqs=num_reqs,
                        uniform=(
                            desc.uniform_token_count is not None
                            and desc.uniform_token_count == self.decode_query_len
                        ),
                        has_lora=has_lora,
                        num_active_loras=desc.num_active_loras,
                    )
                if cg_mode == CUDAGraphMode.PIECEWISE:
                    assert attn_metadata is None
                    batch_descriptor = BatchDescriptor(
                        num_tokens=num_tokens,
                        has_lora=has_lora,
                        num_active_loras=desc.num_active_loras,
                    )
"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_exact(source: str, anchor: str, replacement: str, label: str) -> str:
    count = source.count(anchor)
    if count != 1:
        raise RuntimeError(f"expected one {label} anchor, found {count}")
    return source.replace(anchor, replacement, 1)


def patch_file(
    path: Path,
    *,
    source_sha256: str,
    patched_sha256: str,
    anchor: str,
    replacement: str,
    label: str,
    additional_replacements: tuple[tuple[str, str, str], ...] = (),
) -> None:
    original = path.read_bytes()
    observed = sha256(original)
    if observed != source_sha256:
        raise RuntimeError(
            f"{label} source SHA-256 mismatch: expected {source_sha256}, "
            f"got {observed}"
        )
    patched_text = replace_exact(
        original.decode("utf-8"), anchor, replacement, label
    )
    for extra_anchor, extra_replacement, extra_label in additional_replacements:
        patched_text = replace_exact(
            patched_text, extra_anchor, extra_replacement, extra_label
        )
    patched = patched_text.encode("utf-8")
    observed_patched = sha256(patched)
    if observed_patched != patched_sha256:
        raise RuntimeError(
            f"{label} patched SHA-256 mismatch: expected {patched_sha256}, "
            f"got {observed_patched}"
        )
    path.write_bytes(patched)


def main() -> int:
    patch_file(
        MODEL_RUNNER,
        source_sha256=MODEL_RUNNER_SOURCE_SHA256,
        patched_sha256=MODEL_RUNNER_PATCHED_SHA256,
        anchor=_MODEL_RUNNER_PREFILL_ANCHOR,
        replacement=_MODEL_RUNNER_PREFILL_REPLACEMENT,
        label="model runner prefill exclusion",
        additional_replacements=((
            _MODEL_RUNNER_DESCRIPTOR_ANCHOR,
            _MODEL_RUNNER_DESCRIPTOR_REPLACEMENT,
            "model runner uniform descriptor",
        ),),
    )
    patch_file(
        CUDAGRAPH_UTILS,
        source_sha256=CUDAGRAPH_UTILS_SOURCE_SHA256,
        patched_sha256=CUDAGRAPH_UTILS_PATCHED_SHA256,
        anchor=_CUDAGRAPH_UTILS_ANCHOR,
        replacement=_CUDAGRAPH_UTILS_REPLACEMENT,
        label="CUDA graph uniform decode",
    )
    print(
        "patched NVFP4 dual uniform decode: "
        f"model_runner={MODEL_RUNNER_PATCHED_SHA256} "
        f"cudagraph_utils={CUDAGRAPH_UTILS_PATCHED_SHA256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
