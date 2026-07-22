#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Atomics-free Triton reduction for route-major grouped FC2 output."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _route_major_reduce_kernel(
    grouped_output_ptr,
    compact_topk_ids_ptr,
    compact_row_ids_ptr,
    topk_weights_ptr,
    local_a_row_bases_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    hidden_block = tl.program_id(1)
    hidden = hidden_block * BLOCK_H + tl.arange(0, BLOCK_H)
    hidden_valid = hidden < hidden_size
    accum = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for route in range(top_k):
        pair = token * top_k + route
        local_expert = tl.load(compact_topk_ids_ptr + pair).to(tl.int32)
        local_row = tl.load(compact_row_ids_ptr + pair).to(tl.int32)
        group_base = tl.load(local_a_row_bases_ptr + local_expert).to(tl.int32)
        group_row = group_base + local_row
        router_weight = tl.load(topk_weights_ptr + pair).to(tl.float32)
        value = tl.load(
            grouped_output_ptr + group_row * hidden_size + hidden,
            mask=hidden_valid,
            other=0.0,
        ).to(tl.float32)
        accum += router_weight * value

    tl.store(
        output_ptr + token * hidden_size + hidden,
        accum,
        mask=hidden_valid,
    )


def reduce_route_major_output(
    *,
    grouped_output: torch.Tensor,
    compact_topk_ids: torch.Tensor,
    compact_row_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    local_a_row_bases: torch.Tensor,
    output: torch.Tensor,
    top_k: int,
) -> None:
    """Reduce six valid expert rows per token without touching padding rows."""
    if grouped_output.ndim != 2 or output.ndim != 2:
        raise ValueError("grouped_output and output must be matrices")
    if grouped_output.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
        raise ValueError("grouped and token output must be BF16")
    if grouped_output.shape[1] != output.shape[1]:
        raise ValueError("hidden-size drift")
    if output.shape[1] % 128:
        raise ValueError("hidden size must be a multiple of 128")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    pairs = int(output.shape[0]) * top_k
    for name, tensor, dtype in (
        ("compact_topk_ids", compact_topk_ids, torch.int32),
        ("compact_row_ids", compact_row_ids, torch.int32),
        ("topk_weights", topk_weights, torch.float32),
    ):
        if tensor.dtype != dtype or tensor.numel() != pairs:
            raise ValueError(f"{name} must be contiguous {dtype}[{pairs}]")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if local_a_row_bases.dtype != torch.int32 or not local_a_row_bases.is_contiguous():
        raise ValueError("local_a_row_bases must be contiguous int32")

    block_h = 256
    _route_major_reduce_kernel[
        (output.shape[0], triton.cdiv(output.shape[1], block_h))
    ](
        grouped_output,
        compact_topk_ids,
        compact_row_ids,
        topk_weights,
        local_a_row_bases,
        output,
        hidden_size=output.shape[1],
        top_k=top_k,
        BLOCK_H=block_h,
        num_warps=4,
    )


__all__ = ["reduce_route_major_output"]
