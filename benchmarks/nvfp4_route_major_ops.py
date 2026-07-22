#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Triton glue kernels for the bounded route-major NVFP4 prototype."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_route_input_kernel(
    x_ptr,
    compact_ids_ptr,
    compact_rows_ptr,
    grouped_ptr,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0)
    block = tl.program_id(1)
    hidden = block * BLOCK + tl.arange(0, BLOCK)
    mask = hidden < hidden_size
    token = pair // top_k
    local = tl.load(compact_ids_ptr + pair).to(tl.int32)
    row = tl.load(compact_rows_ptr + pair).to(tl.int32)
    value = tl.load(x_ptr + token * hidden_size + hidden, mask=mask)
    tl.store(
        grouped_ptr + (local * 4 + row) * hidden_size + hidden,
        value,
        mask=mask,
    )


@triton.jit
def _scatter_packed_kernel(
    source_ptr,
    bases_ptr,
    destination_ptr,
    packed_cols: tl.constexpr,
    BLOCK: tl.constexpr,
):
    local = tl.program_id(0)
    row = tl.program_id(1)
    block = tl.program_id(2)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    mask = cols < packed_cols
    destination_row = tl.load(bases_ptr + local).to(tl.int32) + row
    value = tl.load(
        source_ptr + (local * 4 + row) * packed_cols + cols,
        mask=mask,
    )
    tl.store(destination_ptr + destination_row * packed_cols + cols, value, mask=mask)


@triton.jit
def _scatter_scale_kernel(
    source_ptr,
    bases_ptr,
    destination_ptr,
    scale_cols: tl.constexpr,
    BLOCK: tl.constexpr,
):
    local = tl.program_id(0)
    row = tl.program_id(1)
    block = tl.program_id(2)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    mask = cols < scale_cols
    destination_row = tl.load(bases_ptr + local).to(tl.int32) + row
    value = tl.load(
        source_ptr + (local * 128 + row) * scale_cols + cols,
        mask=mask,
    )
    tl.store(destination_ptr + destination_row * scale_cols + cols, value, mask=mask)


@triton.jit
def _oai_swiglu_gather_kernel(
    fc1_ptr,
    bases_ptr,
    activated_ptr,
    intermediate_size: tl.constexpr,
    limit: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
    BLOCK: tl.constexpr,
):
    local = tl.program_id(0)
    row = tl.program_id(1)
    block = tl.program_id(2)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    mask = cols < intermediate_size
    grouped_row = tl.load(bases_ptr + local).to(tl.int32) + row
    base = grouped_row * (2 * intermediate_size)
    # Prepared W13 is [up/w3, gate/w1], matching the accepted kernel.
    up = tl.load(fc1_ptr + base + cols, mask=mask).to(tl.float32)
    gate = tl.load(fc1_ptr + base + intermediate_size + cols, mask=mask).to(tl.float32)
    gate = tl.minimum(gate, limit)
    up = tl.maximum(tl.minimum(up, limit), -limit)
    value = gate * tl.sigmoid(alpha * gate) * (up + beta)
    tl.store(
        activated_ptr + (local * 4 + row) * intermediate_size + cols,
        value,
        mask=mask,
    )


def gather_route_inputs(
    *,
    x: torch.Tensor,
    compact_topk_ids: torch.Tensor,
    compact_row_ids: torch.Tensor,
    grouped_input: torch.Tensor,
    top_k: int,
) -> None:
    if x.ndim != 2 or grouped_input.ndim != 3 or grouped_input.shape[1] != 4:
        raise ValueError("x must be [M,H] and grouped_input [B,4,H]")
    if x.shape[1] != grouped_input.shape[2]:
        raise ValueError("hidden-size drift")
    grouped_input.zero_()
    pairs = x.shape[0] * top_k
    if compact_topk_ids.numel() != pairs or compact_row_ids.numel() != pairs:
        raise ValueError("compact metadata must have M*top_k entries")
    block = 256
    _gather_route_input_kernel[(pairs, triton.cdiv(x.shape[1], block))](
        x,
        compact_topk_ids,
        compact_row_ids,
        grouped_input,
        hidden_size=x.shape[1],
        top_k=top_k,
        BLOCK=block,
        num_warps=4,
    )


def scatter_batched_nvfp4(
    *,
    source_packed: torch.Tensor,
    source_scale: torch.Tensor,
    local_a_row_bases: torch.Tensor,
    local_scale_row_bases: torch.Tensor,
    destination_packed: torch.Tensor,
    destination_scale: torch.Tensor,
) -> None:
    batch, rows, packed_cols = map(int, source_packed.shape)
    if rows != 4:
        raise ValueError("route-major quantization batch must use four rows")
    scale_cols = int(destination_scale.shape[1])
    if source_scale.numel() != batch * 128 * scale_cols:
        raise ValueError(
            "batched scale layout must contain one independent 128-row tile per expert"
        )
    if local_a_row_bases.numel() != batch or local_scale_row_bases.numel() != batch:
        raise ValueError("base metadata must have one entry per active expert")
    packed_block = 256
    _scatter_packed_kernel[(batch, 4, triton.cdiv(packed_cols, packed_block))](
        source_packed,
        local_a_row_bases,
        destination_packed,
        packed_cols=packed_cols,
        BLOCK=packed_block,
        num_warps=4,
    )
    scale_block = 64
    _scatter_scale_kernel[(batch, 128, triton.cdiv(scale_cols, scale_block))](
        source_scale,
        local_scale_row_bases,
        destination_scale,
        scale_cols=scale_cols,
        BLOCK=scale_block,
        num_warps=2,
    )


def oai_swiglu_gather(
    *,
    fc1_output: torch.Tensor,
    local_a_row_bases: torch.Tensor,
    activated: torch.Tensor,
    limit: float = 10.0,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> None:
    if fc1_output.ndim != 2 or activated.ndim != 3 or activated.shape[1] != 4:
        raise ValueError("fc1_output must be [P,2I] and activated [B,4,I]")
    intermediate = int(activated.shape[2])
    if fc1_output.shape[1] != 2 * intermediate:
        raise ValueError("FC1/activation width drift")
    if local_a_row_bases.numel() != activated.shape[0]:
        raise ValueError("one A-row base is required per active expert")
    block = 256
    _oai_swiglu_gather_kernel[
        (activated.shape[0], 4, triton.cdiv(intermediate, block))
    ](
        fc1_output,
        local_a_row_bases,
        activated,
        intermediate_size=intermediate,
        limit=limit,
        alpha=alpha,
        beta=beta,
        BLOCK=block,
        num_warps=4,
    )


__all__ = ["gather_route_inputs", "oai_swiglu_gather", "scatter_batched_nvfp4"]
