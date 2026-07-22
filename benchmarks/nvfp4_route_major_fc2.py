#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prototype route-major phase-2 contract for SM121 NVFP4 decode.

The serving microkernel owns phase 1 (FC1, SwiGLU and NVFP4 quantization).
This module deliberately starts at the serialization boundary: packed FP4
rows, 128x4 activation scales and compact route metadata.  It then launches
FlashInfer's grouped FC2 GEMM and an atomics-free Triton route reduction.

Imports of Torch, Triton and FlashInfer are lazy so the contract and its unit
tests remain runnable on a CPU-only development host.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


ROW_ALIGNMENT = 4
SCALE_ROW_ALIGNMENT = 128


def align_up(value: int, alignment: int) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("value must be non-negative and alignment positive")
    return ((value + alignment - 1) // alignment) * alignment


def group_scale_row_offset(group: int, m_offset: int) -> int:
    """Mirror FlashInfer SM120 grouped-NVFP4's SFA descriptor formula."""
    if group < 0 or m_offset < 0:
        raise ValueError("group and M offset must be non-negative")
    return ((m_offset + group * 127) // 128) * 128


@dataclass(frozen=True)
class RouteMajorMetadata:
    """Host proof for the compact-routes to grouped-GEMM mapping."""

    num_tokens: int
    top_k: int
    weight_experts: int
    intermediate_size: int
    hidden_size: int
    active_experts: tuple[int, ...]
    row_counts: tuple[int, ...]
    compact_topk_ids: tuple[int, ...]
    compact_row_ids: tuple[int, ...]
    local_a_row_bases: tuple[int, ...]
    local_scale_row_bases: tuple[int, ...]
    m_indptr: tuple[int, ...]
    padded_rows: int
    scale_storage_rows: int

    @property
    def routed_rows(self) -> int:
        return self.num_tokens * self.top_k

    @property
    def phase1_packed_bytes(self) -> int:
        return self.padded_rows * self.intermediate_size // 2

    @property
    def phase1_scale_bytes(self) -> int:
        return self.scale_storage_rows * (self.intermediate_size // 16)

    @property
    def grouped_output_bytes(self) -> int:
        return self.padded_rows * self.hidden_size * 2

    @property
    def handoff_bytes(self) -> int:
        return (
            self.phase1_packed_bytes
            + self.phase1_scale_bytes
            + self.grouped_output_bytes
        )


def build_route_major_metadata(
    *,
    topk_ids: Sequence[int],
    num_tokens: int,
    top_k: int,
    weight_experts: int = 256,
    intermediate_size: int = 1024,
    hidden_size: int = 4096,
) -> RouteMajorMetadata:
    """Build a deterministic first-occurrence compact-route mapping."""
    if not 1 <= num_tokens <= 4:
        raise ValueError("route-major prototype is decode-only (M=1..4)")
    if top_k <= 0 or len(topk_ids) != num_tokens * top_k:
        raise ValueError("topk_ids must contain exactly M * top_k entries")
    if intermediate_size <= 0 or intermediate_size % 128:
        raise ValueError("intermediate_size must be a positive multiple of 128")
    if hidden_size <= 0 or hidden_size % 128:
        raise ValueError("hidden_size must be a positive multiple of 128")

    active: list[int] = []
    local_by_global: dict[int, int] = {}
    compact_ids: list[int] = []
    compact_rows: list[int] = []
    counts: list[int] = []
    for raw_expert in topk_ids:
        expert = int(raw_expert)
        if not 0 <= expert < weight_experts:
            raise ValueError(f"expert {expert} is outside [0,{weight_experts})")
        local = local_by_global.get(expert)
        if local is None:
            local = len(active)
            local_by_global[expert] = local
            active.append(expert)
            counts.append(0)
        compact_ids.append(local)
        compact_rows.append(counts[local])
        counts[local] += 1

    padded_by_global = [0] * weight_experts
    for expert, rows in zip(active, counts, strict=True):
        padded_by_global[expert] = align_up(rows, ROW_ALIGNMENT)
    m_indptr = [0]
    for rows in padded_by_global:
        m_indptr.append(m_indptr[-1] + rows)

    a_bases = tuple(m_indptr[expert] for expert in active)
    scale_bases = tuple(
        group_scale_row_offset(expert, m_indptr[expert]) for expert in active
    )
    # The descriptor builder addresses every physical expert, including an
    # inactive expert 255.  Reserve its complete 128-row SFA tile.
    last = weight_experts - 1
    scale_rows = group_scale_row_offset(last, m_indptr[last]) + 128
    result = RouteMajorMetadata(
        num_tokens=num_tokens,
        top_k=top_k,
        weight_experts=weight_experts,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        active_experts=tuple(active),
        row_counts=tuple(counts),
        compact_topk_ids=tuple(compact_ids),
        compact_row_ids=tuple(compact_rows),
        local_a_row_bases=a_bases,
        local_scale_row_bases=scale_bases,
        m_indptr=tuple(m_indptr),
        padded_rows=m_indptr[-1],
        scale_storage_rows=scale_rows,
    )
    if result.handoff_bytes >= 3 * 1024 * 1024:
        raise RuntimeError(f"route-major handoff exceeded 3 MiB: {result.handoff_bytes}")
    return result


@dataclass
class Phase1Handoff:
    """Frozen interface produced by ``MoEPhase1Kernel``.

    ``packed_a`` and ``a_scale`` are caller-owned persistent workspace views;
    the dispatcher never allocates or transforms them.  The clean interface
    lets the Python probe emulate phase 1 today and the CuTeDSL microkernel
    replace it without changing phase 2.
    """

    metadata: RouteMajorMetadata
    packed_a: Any
    a_scale: Any
    m_indptr: Any
    compact_topk_ids: Any
    compact_row_ids: Any
    local_a_row_bases: Any
    topk_weights: Any

    def validate(self, torch: Any) -> None:
        meta = self.metadata
        checks = (
            (self.packed_a, torch.uint8, (meta.padded_rows, meta.intermediate_size // 2), "packed_a"),
            (self.a_scale, torch.uint8, (meta.scale_storage_rows, meta.intermediate_size // 16), "a_scale"),
            (self.m_indptr, torch.int32, (meta.weight_experts + 1,), "m_indptr"),
            (self.compact_topk_ids, torch.int32, (meta.routed_rows,), "compact_topk_ids"),
            (self.compact_row_ids, torch.int32, (meta.routed_rows,), "compact_row_ids"),
            (self.local_a_row_bases, torch.int32, (len(meta.active_experts),), "local_a_row_bases"),
            (self.topk_weights, torch.float32, (meta.routed_rows,), "topk_weights"),
        )
        device = self.packed_a.device
        if device.type != "cuda":
            raise ValueError("phase-1 handoff must reside on CUDA")
        for tensor, dtype, shape, name in checks:
            if tensor.dtype != dtype or tuple(tensor.shape) != shape:
                raise ValueError(
                    f"{name} contract drift: got {tensor.dtype} {tuple(tensor.shape)}, "
                    f"expected {dtype} {shape}"
                )
            if tensor.device != device or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous on {device}")

    def validate_values(self) -> None:
        """Synchronizing diagnostic check; never call from the timed path."""
        meta = self.metadata
        if tuple(int(v) for v in self.m_indptr.cpu().tolist()) != meta.m_indptr:
            raise ValueError("device m_indptr does not match the host route proof")
        if tuple(int(v) for v in self.compact_topk_ids.cpu().tolist()) != meta.compact_topk_ids:
            raise ValueError("device compact IDs do not match the host route proof")
        if tuple(int(v) for v in self.compact_row_ids.cpu().tolist()) != meta.compact_row_ids:
            raise ValueError("device compact rows do not match the host route proof")


@dataclass
class RouteMajorFC2Workspace:
    """Caller-owned output arena; allocate once before CUDA graph capture."""

    grouped_output: Any
    output: Any

    @classmethod
    def allocate(cls, torch: Any, metadata: RouteMajorMetadata) -> "RouteMajorFC2Workspace":
        return cls(
            grouped_output=torch.empty(
                metadata.padded_rows,
                metadata.hidden_size,
                dtype=torch.bfloat16,
                device="cuda",
            ),
            output=torch.empty(
                metadata.num_tokens,
                metadata.hidden_size,
                dtype=torch.bfloat16,
                device="cuda",
            ),
        )


class RouteMajorFC2Dispatcher:
    """Bounded Python phase-2 dispatcher for the SM121 feasibility gate."""

    def __init__(self, *, w2: Any, w2_scale: Any, g2_alpha: Any) -> None:
        self.w2 = w2
        self.w2_scale = w2_scale
        self.g2_alpha = g2_alpha

    def validate_weights(self, torch: Any, metadata: RouteMajorMetadata) -> None:
        expected = {
            "w2": (torch.uint8, (metadata.weight_experts, metadata.hidden_size, metadata.intermediate_size // 2)),
            "g2_alpha": (torch.float32, (metadata.weight_experts,)),
        }
        for name, tensor in (
            ("w2", self.w2),
            ("g2_alpha", self.g2_alpha),
        ):
            dtype, shape = expected[name]
            if tensor.dtype != dtype or tuple(tensor.shape) != shape:
                raise ValueError(
                    f"{name} contract drift: got {tensor.dtype} {tuple(tensor.shape)}, "
                    f"expected {dtype} {shape}"
                )
            if tensor.device.type != "cuda" or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous on CUDA")
        scale_shape = (
            metadata.weight_experts,
            metadata.hidden_size,
            metadata.intermediate_size // 16,
        )
        fp8_e4m3 = getattr(torch, "float8_e4m3fn", None)
        if self.w2_scale.dtype not in {torch.uint8, fp8_e4m3} or tuple(self.w2_scale.shape) != scale_shape:
            raise ValueError(
                "w2_scale must be raw uint8/E4M3 storage with shape "
                f"{scale_shape}; got {self.w2_scale.dtype} {tuple(self.w2_scale.shape)}"
            )
        if self.w2_scale.device.type != "cuda" or not self.w2_scale.is_contiguous():
            raise ValueError("w2_scale must be contiguous on CUDA")

    def launch(self, torch: Any, flashinfer: Any, handoff: Phase1Handoff, workspace: RouteMajorFC2Workspace) -> Any:
        """Launch grouped FC2 then the atomics-free route reduction."""
        handoff.validate(torch)
        meta = handoff.metadata
        self.validate_weights(torch, meta)
        if tuple(workspace.grouped_output.shape) != (meta.padded_rows, meta.hidden_size):
            raise ValueError("grouped output arena shape drift")
        if tuple(workspace.output.shape) != (meta.num_tokens, meta.hidden_size):
            raise ValueError("token output arena shape drift")
        flashinfer.gemm.group_gemm_nvfp4_nt_groupwise(
            handoff.packed_a,
            self.w2,
            handoff.a_scale,
            self.w2_scale.view(torch.uint8),
            handoff.m_indptr,
            alpha=self.g2_alpha,
            tile_m=128,
            tile_n=128,
            tile_k=128,
            swap_ab=True,
            out=workspace.grouped_output,
        )
        from benchmarks.nvfp4_route_major_reduce import reduce_route_major_output

        reduce_route_major_output(
            grouped_output=workspace.grouped_output,
            compact_topk_ids=handoff.compact_topk_ids,
            compact_row_ids=handoff.compact_row_ids,
            topk_weights=handoff.topk_weights,
            local_a_row_bases=handoff.local_a_row_bases,
            output=workspace.output,
            top_k=meta.top_k,
        )
        return workspace.output


def emulate_phase1_handoff(
    torch: Any,
    flashinfer: Any,
    *,
    activated_rows: Any,
    topk_ids: Any,
    topk_weights: Any,
    a2_gscale: Any,
    hidden_size: int,
) -> Phase1Handoff:
    """Reference serializer for the phase-1 CuTeDSL output contract.

    This intentionally uses a bounded Python loop and is benchmarked
    separately from phase 2.  It is an oracle for byte-level phase-1 parity,
    not a proposed serving implementation.
    """
    if activated_rows.ndim != 2 or topk_ids.ndim != 2:
        raise ValueError("activated_rows and topk_ids must be matrices")
    num_tokens, top_k = map(int, topk_ids.shape)
    routed_rows, intermediate_size = map(int, activated_rows.shape)
    if routed_rows != num_tokens * top_k:
        raise ValueError("activated_rows must contain one row per routed pair")
    ids_host = [int(value) for value in topk_ids.detach().cpu().reshape(-1).tolist()]
    metadata = build_route_major_metadata(
        topk_ids=ids_host,
        num_tokens=num_tokens,
        top_k=top_k,
        weight_experts=int(a2_gscale.numel()),
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
    )
    packed_a = torch.zeros(
        metadata.padded_rows,
        intermediate_size // 2,
        dtype=torch.uint8,
        device="cuda",
    )
    a_scale = torch.zeros(
        metadata.scale_storage_rows,
        intermediate_size // 16,
        dtype=torch.uint8,
        device="cuda",
    )
    ids_flat = topk_ids.reshape(-1)
    for local, expert in enumerate(metadata.active_experts):
        pair_indices = torch.nonzero(ids_flat == expert, as_tuple=False).reshape(-1)
        rows = int(pair_indices.numel())
        padded = align_up(rows, ROW_ALIGNMENT)
        segment = torch.zeros(
            padded,
            intermediate_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        segment[:rows].copy_(activated_rows.index_select(0, pair_indices))
        segment_fp4, segment_scale = flashinfer.nvfp4_quantize(
            segment,
            a2_gscale[expert : expert + 1],
            sfLayout=flashinfer.SfLayout.layout_128x4,
            sf_vec_size=16,
        )
        base = metadata.local_a_row_bases[local]
        packed_a[base : base + padded].copy_(segment_fp4.view(torch.uint8))
        scale_base = metadata.local_scale_row_bases[local]
        if tuple(segment_scale.shape) != (128, intermediate_size // 16):
            raise RuntimeError(
                f"phase-1 scale layout drift: {tuple(segment_scale.shape)}"
            )
        a_scale[scale_base : scale_base + 128].copy_(segment_scale.view(torch.uint8))

    return Phase1Handoff(
        metadata=metadata,
        packed_a=packed_a,
        a_scale=a_scale,
        m_indptr=torch.tensor(metadata.m_indptr, dtype=torch.int32, device="cuda"),
        compact_topk_ids=torch.tensor(metadata.compact_topk_ids, dtype=torch.int32, device="cuda"),
        compact_row_ids=torch.tensor(metadata.compact_row_ids, dtype=torch.int32, device="cuda"),
        local_a_row_bases=torch.tensor(metadata.local_a_row_bases, dtype=torch.int32, device="cuda"),
        topk_weights=topk_weights.reshape(-1).to(dtype=torch.float32).contiguous(),
    )


__all__ = [
    "Phase1Handoff",
    "RouteMajorFC2Dispatcher",
    "RouteMajorFC2Workspace",
    "RouteMajorMetadata",
    "align_up",
    "build_route_major_metadata",
    "emulate_phase1_handoff",
    "group_scale_row_offset",
]
