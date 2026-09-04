# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in SM12x native DSpark widths (FlashInfer #4380 / vLLM #51538).

Keep the legacy policy for ordinary images and other GPU families. The native
image must bundle a verified sparse binary, not reuse a stale shared JIT cache.
"""

import os


def native_sparse_widths_enabled() -> bool:
    value = os.environ.get("VLLM_DSV4_NATIVE_SPARSE_WIDTHS", "0")
    if value not in ("0", "1"):
        raise ValueError("VLLM_DSV4_NATIVE_SPARSE_WIDTHS must be 0 or 1")
    return value == "1"


def dspark_sparse_width(window: int, draft_tokens: int, *, sm12x: bool) -> int:
    if window <= 0 or draft_tokens < 0:
        raise ValueError("invalid DSpark window or draft token count")
    alignment = 64 if sm12x and native_sparse_widths_enabled() else 128
    return (window + draft_tokens + alignment - 1) // alignment * alignment


def sparse_decode_widths() -> tuple[int, ...]:
    if native_sparse_widths_enabled():
        return (128, 192, 256, 512, 1024)
    return (128, 512, 1024)


def narrow_attention_graph_enabled() -> bool:
    value = os.environ.get("DSPARK_NARROW_ATTN_GRAPH", "0")
    if value not in ("0", "1"):
        raise ValueError("DSPARK_NARROW_ATTN_GRAPH must be 0 or 1")
    return value == "1"
