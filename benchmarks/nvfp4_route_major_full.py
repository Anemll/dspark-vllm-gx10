#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Full Python-level route-major W4A4 prototype (FC1 through reduction)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmarks.nvfp4_route_major_fc2 import RouteMajorMetadata


@dataclass
class RouteMajorFullWorkspace:
    metadata: RouteMajorMetadata
    grouped_input: Any
    fc1_packed: Any
    fc1_scale: Any
    fc1_output: Any
    activated: Any
    fc2_packed: Any
    fc2_scale: Any
    grouped_output: Any
    output: Any
    m_indptr: Any
    compact_topk_ids: Any
    compact_row_ids: Any
    local_a_row_bases: Any
    local_scale_row_bases: Any
    topk_weights: Any

    @classmethod
    def allocate(cls, torch: Any, metadata: RouteMajorMetadata, topk_weights: Any) -> "RouteMajorFullWorkspace":
        batch = len(metadata.active_experts)
        p = metadata.padded_rows
        h = metadata.hidden_size
        i = metadata.intermediate_size
        scale_rows = metadata.scale_storage_rows
        return cls(
            metadata=metadata,
            grouped_input=torch.empty(batch, 4, h, dtype=torch.bfloat16, device="cuda"),
            fc1_packed=torch.empty(p, h // 2, dtype=torch.uint8, device="cuda"),
            fc1_scale=torch.empty(scale_rows, h // 16, dtype=torch.uint8, device="cuda"),
            fc1_output=torch.empty(p, 2 * i, dtype=torch.bfloat16, device="cuda"),
            activated=torch.empty(batch, 4, i, dtype=torch.bfloat16, device="cuda"),
            fc2_packed=torch.empty(p, i // 2, dtype=torch.uint8, device="cuda"),
            fc2_scale=torch.empty(scale_rows, i // 16, dtype=torch.uint8, device="cuda"),
            grouped_output=torch.empty(p, h, dtype=torch.bfloat16, device="cuda"),
            output=torch.empty(metadata.num_tokens, h, dtype=torch.bfloat16, device="cuda"),
            m_indptr=torch.tensor(metadata.m_indptr, dtype=torch.int32, device="cuda"),
            compact_topk_ids=torch.tensor(metadata.compact_topk_ids, dtype=torch.int32, device="cuda"),
            compact_row_ids=torch.tensor(metadata.compact_row_ids, dtype=torch.int32, device="cuda"),
            local_a_row_bases=torch.tensor(metadata.local_a_row_bases, dtype=torch.int32, device="cuda"),
            local_scale_row_bases=torch.tensor(metadata.local_scale_row_bases, dtype=torch.int32, device="cuda"),
            topk_weights=topk_weights.reshape(-1).to(dtype=torch.float32).contiguous(),
        )

    @property
    def persistent_bytes(self) -> int:
        tensors = (
            self.grouped_input,
            self.fc1_packed,
            self.fc1_scale,
            self.fc1_output,
            self.activated,
            self.fc2_packed,
            self.fc2_scale,
            self.grouped_output,
            self.output,
            self.m_indptr,
            self.compact_topk_ids,
            self.compact_row_ids,
            self.local_a_row_bases,
            self.local_scale_row_bases,
            self.topk_weights,
        )
        return sum(int(t.numel()) * int(t.element_size()) for t in tensors)


class RouteMajorFullDispatcher:
    """Existing grouped GEMMs around a bounded Triton routing/activation shell."""

    def __init__(
        self,
        *,
        w13: Any,
        w13_scale: Any,
        a1_gscale: Any,
        g1_alpha: Any,
        w2: Any,
        w2_scale: Any,
        a2_gscale: Any,
        g2_alpha: Any,
        swiglu_limit: float = 10.0,
    ) -> None:
        self.w13 = w13
        self.w13_scale = w13_scale
        self.a1_gscale = a1_gscale
        self.g1_alpha = g1_alpha
        self.w2 = w2
        self.w2_scale = w2_scale
        self.a2_gscale = a2_gscale
        self.g2_alpha = g2_alpha
        self.swiglu_limit = float(swiglu_limit)

    def validate(self, torch: Any, metadata: RouteMajorMetadata) -> None:
        e, h, i = (
            metadata.weight_experts,
            metadata.hidden_size,
            metadata.intermediate_size,
        )
        expected_shapes = {
            "w13": (e, 2 * i, h // 2),
            "w13_scale": (e, 2 * i, h // 16),
            "a1_gscale": (e,),
            "g1_alpha": (e,),
            "w2": (e, h, i // 2),
            "w2_scale": (e, h, i // 16),
            "a2_gscale": (e,),
            "g2_alpha": (e,),
        }
        for name, shape in expected_shapes.items():
            tensor = getattr(self, name)
            if tuple(tensor.shape) != shape or tensor.device.type != "cuda" or not tensor.is_contiguous():
                raise ValueError(f"{name} physical contract drift: {tensor.dtype} {tuple(tensor.shape)}")
        for name in ("w13", "w2"):
            if getattr(self, name).dtype != torch.uint8:
                raise ValueError(f"{name} must be packed uint8")
        for name in ("a1_gscale", "g1_alpha", "a2_gscale", "g2_alpha"):
            if getattr(self, name).dtype != torch.float32:
                raise ValueError(f"{name} must be float32")
        # Prepared DeepSeek-V4 stores one shared activation scale expanded to
        # E.  Batched quantization is only algebraically valid under this pin.
        for name in ("a1_gscale", "a2_gscale"):
            value = getattr(self, name)
            if not bool(torch.equal(value, value[0].expand_as(value))):
                raise ValueError(f"{name} is not constant across experts")

    def launch(self, torch: Any, flashinfer: Any, x: Any, workspace: RouteMajorFullWorkspace) -> Any:
        """Run gather→Q→FC1→SwiGLU→Q→FC2→reduce on one stream."""
        from benchmarks.nvfp4_route_major_ops import (
            gather_route_inputs,
            oai_swiglu_gather,
            scatter_batched_nvfp4,
        )
        from benchmarks.nvfp4_route_major_reduce import reduce_route_major_output

        meta = workspace.metadata
        gather_route_inputs(
            x=x,
            compact_topk_ids=workspace.compact_topk_ids,
            compact_row_ids=workspace.compact_row_ids,
            grouped_input=workspace.grouped_input,
            top_k=meta.top_k,
        )
        q1, sf1 = flashinfer.nvfp4_batched_quantize(
            workspace.grouped_input,
            self.a1_gscale[:1],
        )
        scatter_batched_nvfp4(
            source_packed=q1,
            source_scale=sf1,
            local_a_row_bases=workspace.local_a_row_bases,
            local_scale_row_bases=workspace.local_scale_row_bases,
            destination_packed=workspace.fc1_packed,
            destination_scale=workspace.fc1_scale,
        )
        flashinfer.gemm.group_gemm_nvfp4_nt_groupwise(
            workspace.fc1_packed,
            self.w13,
            workspace.fc1_scale,
            self.w13_scale.view(torch.uint8),
            workspace.m_indptr,
            alpha=self.g1_alpha,
            tile_m=128,
            tile_n=128,
            tile_k=128,
            swap_ab=True,
            out=workspace.fc1_output,
        )
        oai_swiglu_gather(
            fc1_output=workspace.fc1_output,
            local_a_row_bases=workspace.local_a_row_bases,
            activated=workspace.activated,
            limit=self.swiglu_limit,
            alpha=1.0,
            beta=0.0,
        )
        q2, sf2 = flashinfer.nvfp4_batched_quantize(
            workspace.activated,
            self.a2_gscale[:1],
        )
        scatter_batched_nvfp4(
            source_packed=q2,
            source_scale=sf2,
            local_a_row_bases=workspace.local_a_row_bases,
            local_scale_row_bases=workspace.local_scale_row_bases,
            destination_packed=workspace.fc2_packed,
            destination_scale=workspace.fc2_scale,
        )
        flashinfer.gemm.group_gemm_nvfp4_nt_groupwise(
            workspace.fc2_packed,
            self.w2,
            workspace.fc2_scale,
            self.w2_scale.view(torch.uint8),
            workspace.m_indptr,
            alpha=self.g2_alpha,
            tile_m=128,
            tile_n=128,
            tile_k=128,
            swap_ab=True,
            out=workspace.grouped_output,
        )
        reduce_route_major_output(
            grouped_output=workspace.grouped_output,
            compact_topk_ids=workspace.compact_topk_ids,
            compact_row_ids=workspace.compact_row_ids,
            topk_weights=workspace.topk_weights,
            local_a_row_bases=workspace.local_a_row_bases,
            output=workspace.output,
            top_k=meta.top_k,
        )
        return workspace.output


__all__ = ["RouteMajorFullDispatcher", "RouteMajorFullWorkspace"]
