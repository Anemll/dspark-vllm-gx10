#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile/launch adapter for the isolated SM121 route-major phase-1 kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmarks.nvfp4_route_major_fc2 import Phase1Handoff, RouteMajorMetadata


@dataclass
class Phase1Runner:
    torch: Any
    compiled: Any
    workspace: Any
    x: Any
    compact_topk_ids: Any
    compact_row_ids: Any
    local_a_row_bases_kernel: Any
    local_scale_row_bases_kernel: Any
    handoff: Phase1Handoff
    w13_view: Any
    w13_scale_storage: Any
    w13_scale_ptr: Any
    a1_gscale: Any
    g1_alpha: Any
    a2_gscale: Any
    max_active_clusters: int
    current_cuda_stream: Any

    def launch(self) -> Phase1Handoff:
        ws = self.workspace
        # The prepass values are immutable for this fixed-route hardware gate.
        # Phase 1 mutates only packed input, barriers and handoff payloads.
        self.compiled(
            self.x,
            self.compact_topk_ids,
            ws.packed_a_view,
            # TVM-FFI pointer arguments cross the generated ABI as raw device
            # addresses, exactly as in FlashInfer's accepted dispatcher.
            ws.packed_input_scale.data_ptr(),
            ws.packed_a_flat,
            ws.scale_flat,
            ws.barrier_count,
            ws.barrier_epoch,
            self.w13_view,
            self.w13_scale_storage.data_ptr(),
            ws.row_counts,
            ws.active_expert_count,
            ws.weight_expert_ids,
            self.a1_gscale,
            self.g1_alpha,
            self.a2_gscale,
            self.compact_row_ids,
            self.local_a_row_bases_kernel,
            self.local_scale_row_bases_kernel,
            self.handoff.packed_a.reshape(-1),
            self.handoff.a_scale.reshape(-1),
            # max_active_clusters is a cutlass.Constexpr baked by cute.compile
            # and therefore is not part of the generated runtime ABI.
            self.current_cuda_stream(),
        )
        return self.handoff


def build_phase1_runner(
    torch: Any,
    *,
    x: Any,
    topk_ids: Any,
    topk_weights: Any,
    metadata: RouteMajorMetadata,
    w13: Any,
    w13_scale: Any,
    a1_gscale: Any,
    g1_alpha: Any,
    a2_gscale: Any,
    max_active_clusters: int | None = None,
) -> Phase1Runner:
    """Compile the isolated CuTeDSL phase 1 and bind persistent buffers."""
    import cutlass
    import cutlass.cute as cute
    from flashinfer.cute_dsl.utils import (
        convert_sf_from_mma_layout,
        convert_sf_to_mma_layout,
        current_cuda_stream,
        make_ptr,
    )
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_phase1_kernel import (
        MoEPhase1Kernel,
    )

    m, h = map(int, x.shape)
    e, two_i, packed_h = map(int, w13.shape)
    i = two_i // 2
    active = len(metadata.active_experts)
    if (m, h, e, i) != (
        metadata.num_tokens,
        metadata.hidden_size,
        metadata.weight_experts,
        metadata.intermediate_size,
    ) or packed_h != h // 2:
        raise ValueError("phase-1 input/weight shape drift")
    if tuple(topk_ids.shape) != (m, metadata.top_k):
        raise ValueError("topk_ids shape drift")
    if w13.dtype != torch.uint8 or w13_scale.device.type != "cuda":
        raise ValueError("phase-1 W13 must be prepared packed NVFP4 on CUDA")
    if tuple(w13_scale.shape) != (e, 2 * i, h // 16):
        raise ValueError(
            "phase-1 W13 scale must be expert-leading prepared storage "
            f"[{e},{2 * i},{h // 16}]"
        )
    for name, value in (
        ("a1_gscale", a1_gscale),
        ("g1_alpha", g1_alpha),
        ("a2_gscale", a2_gscale),
    ):
        if value.dtype != torch.float32 or tuple(value.shape) != (e,):
            raise ValueError(f"{name} must be float32[{e}]")

    workspace = moe_dispatch.allocate_sm120_static_workspace(
        state_E=active,
        weight_E=e,
        # Phase 1 materializes one FC1 row per routed token/expert pair.  Its
        # TMA capacity is therefore M*top-k, not merely M.
        max_rows=metadata.routed_rows,
        k=h,
        n=i,
        num_topk=metadata.top_k,
        device=x.device,
        activation_precision="fp4",
    )
    compact_ids = torch.tensor(
        metadata.compact_topk_ids, dtype=torch.int32, device="cuda"
    )
    compact_rows = torch.tensor(
        metadata.compact_row_ids, dtype=torch.int32, device="cuda"
    )
    local_a = torch.tensor(
        metadata.local_a_row_bases, dtype=torch.int32, device="cuda"
    )
    local_scale = torch.tensor(
        metadata.local_scale_row_bases, dtype=torch.int32, device="cuda"
    )
    workspace.row_counts.copy_(
        torch.tensor(metadata.row_counts, dtype=torch.int32, device="cuda")
    )
    workspace.active_expert_count.fill_(active)
    workspace.weight_expert_ids.copy_(
        torch.tensor(metadata.active_experts, dtype=torch.int32, device="cuda")
    )
    workspace.barrier_count.zero_()
    workspace.barrier_epoch.zero_()

    # The grouped FC2 descriptor rounds each expert to four rows while phase 1
    # writes only valid rows.  Initialize padding once; fixed-route replay then
    # preserves it without a timed-path memset.
    packed_output = torch.zeros(
        metadata.padded_rows,
        i // 2,
        dtype=torch.uint8,
        device="cuda",
    )
    scale_output = torch.zeros(
        metadata.scale_storage_rows,
        i // 16,
        dtype=torch.uint8,
        device="cuda",
    )
    m_indptr = torch.tensor(metadata.m_indptr, dtype=torch.int32, device="cuda")
    handoff = Phase1Handoff(
        metadata=metadata,
        packed_a=packed_output,
        a_scale=scale_output,
        m_indptr=m_indptr,
        compact_topk_ids=compact_ids,
        compact_row_ids=compact_rows,
        local_a_row_bases=local_a,
        topk_weights=topk_weights.reshape(-1).to(torch.float32).contiguous(),
    )

    sf_vec_size = 16
    mma_tiler = moe_dispatch._select_moe_mma_tiler_mn(m * metadata.top_k, i)
    phase1 = MoEPhase1Kernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler,
        output_tile_count_n=max(1, (i + mma_tiler[1] - 1) // mma_tiler[1]),
        # Prepared a1/a2 global scales are 1/checkpoint_input_scale, matching
        # the accepted prepared B12X/CUTLASS launch contract.
        input_scales_are_reciprocal=True,
        fast_math=True,
        activation="swigluoai_uninterleave",
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        swiglu_limit=10.0,
        share_input_across_experts=False,
        share_expert_scales=False,
        single_token=False,
    )
    mac = int(max_active_clusters or min(28, active * max(1, i // 128)))

    ab_dtype = cutlass.Float4E2M1FN
    sf_dtype = cutlass.Float8E4M3FN
    bf16 = cutlass.BFloat16
    f32 = cutlass.Float32
    u8 = cutlass.Uint8
    i32 = cutlass.Int32
    fake = cute.runtime.make_fake_compact_tensor
    fake_a = fake(bf16, (m, h), stride_order=(1, 0), assumed_align=16)
    fake_ids = fake(i32, (m * metadata.top_k,), assumed_align=4)
    fake_packed = fake(
        ab_dtype,
        (metadata.routed_rows, h, active),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    fake_sfa = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    rows_pad = ((metadata.routed_rows + 127) // 128) * 128
    cols_pad = (((h // 16) + 3) // 4) * 4
    fake_packed_store = fake(
        u8,
        (active * metadata.routed_rows * (h // 2),),
        assumed_align=16,
    )
    fake_scale_store = fake(u8, (active * rows_pad * cols_pad,), assumed_align=16)
    fake_barrier = fake(i32, (1,), assumed_align=4)
    fake_w13 = fake(
        ab_dtype,
        (2 * i, h, e),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    fake_sfb = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    fake_state = fake(i32, (active,), assumed_align=4)
    fake_count = fake(i32, (1,), assumed_align=4)
    fake_alpha = fake(f32, (e,), assumed_align=16)
    fake_rows = fake(i32, (m * metadata.top_k,), assumed_align=4)
    fake_packed_output = fake(
        u8, (metadata.padded_rows * (i // 2),), assumed_align=16
    )
    fake_scale_output = fake(
        u8, (metadata.scale_storage_rows * (i // 16),), assumed_align=16
    )
    compiled = cute.compile(
        phase1,
        fake_a,
        fake_ids,
        fake_packed,
        fake_sfa,
        fake_packed_store,
        fake_scale_store,
        fake_barrier,
        fake_barrier,
        fake_w13,
        fake_sfb,
        fake_state,
        fake_count,
        fake_state,
        fake_alpha,
        fake_alpha,
        fake_alpha,
        fake_rows,
        fake_state,
        fake_state,
        fake_packed_output,
        fake_scale_output,
        mac,
        current_cuda_stream(),
        options="--opt-level 2 --enable-tvm-ffi",
    )

    w13_view = w13.permute(1, 2, 0).view(torch.float4_e2m1fn_x2)
    # Prepared scale is expert-leading 2-D-swizzled storage.  Round-trip it
    # through the same MMA view -> accepted `_get_weight_views` conversion
    # used by FlashInfer, then retain the resulting contiguous TMA storage for
    # the runner lifetime so the raw pointer cannot dangle.
    w13_scale_mma = convert_sf_to_mma_layout(
        w13_scale.reshape(e * 2 * i, h // 16),
        m=2 * i,
        k=h,
        num_groups=e,
    )
    w13_scale_storage = convert_sf_from_mma_layout(
        w13_scale_mma,
        m=2 * i,
        k=h,
        num_groups=e,
    ).contiguous()
    w13_scale_ptr = make_ptr(
        sf_dtype,
        w13_scale_storage.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    handoff.validate(torch)
    handoff.validate_values()
    return Phase1Runner(
        torch=torch,
        compiled=compiled,
        workspace=workspace,
        x=x,
        compact_topk_ids=compact_ids,
        compact_row_ids=compact_rows,
        local_a_row_bases_kernel=local_a,
        local_scale_row_bases_kernel=local_scale,
        handoff=handoff,
        w13_view=w13_view,
        w13_scale_storage=w13_scale_storage,
        w13_scale_ptr=w13_scale_ptr,
        a1_gscale=a1_gscale,
        g1_alpha=g1_alpha,
        a2_gscale=a2_gscale,
        max_active_clusters=mac,
        current_cuda_stream=current_cuda_stream,
    )


__all__ = ["Phase1Runner", "build_phase1_runner"]
