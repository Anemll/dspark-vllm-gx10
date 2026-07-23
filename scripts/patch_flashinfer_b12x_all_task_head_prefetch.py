#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Add an opt-in all-task weight-head prepass to SM121 B12X micro MoE.

The accepted M=2--4 microkernel schedules compact expert work persistently.
Its ordinary two-stage pipeline does not touch a future task until that task
begins.  This experiment has each DMA warp walk its complete strided task list
before wave zero and issue read-only L2 hints for:

* the first two FC1 gate/up weight and scale tiles; and
* the first two FC2 weight and scale tiles.

For the DeepSeek V4 C4 shape the hinted working set is about 6.6 MiB, so it
fits comfortably in GB10's L2.  The prepass changes no routing, tensor layout,
MMA, activation, reduction, or output operation.  It is disabled by default
and is eligible only for NVFP4 micro launches with M in [2, 4].

This patch targets the accepted post-``patch_flashinfer_b12x_shared_input``
sources.  Both inputs are SHA-pinned and transformed in memory before either
installed file is changed.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Callable


PINNED_DISPATCH_SHA256 = (
    "253cc2f26d465adc37e48c4eee53bdb534bf6fb371a3823a5923cc8d45e2d0d3"
)
PINNED_MICRO_KERNEL_SHA256 = (
    "54fcdb14676395a05d65e27bf6ed770446715aaeb0762e49bff39bccaff693ce"
)
DEFAULT_DISPATCH_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/"
    "cute_dsl/blackwell_sm12x/moe_dispatch.py"
)
DEFAULT_MICRO_KERNEL_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/"
    "cute_dsl/blackwell_sm12x/moe_micro_kernel.py"
)


DISPATCH_ENV_ANCHOR = """\
_MICRO_SHARE_INPUT_ACROSS_EXPERTS = (
    os.environ.get("FLASHINFER_B12X_MICRO_SHARE_INPUT", "1") != "0"
)
"""
DISPATCH_ENV_REPLACEMENT = DISPATCH_ENV_ANCHOR + """\
_ALL_TASK_HEAD_PREFETCH_ENV = "FLASHINFER_B12X_ALL_TASK_HEAD_PREFETCH"
_all_task_head_prefetch_raw = os.environ.get(_ALL_TASK_HEAD_PREFETCH_ENV, "0")
if _all_task_head_prefetch_raw not in {"0", "1"}:
    raise RuntimeError(
        f"{_ALL_TASK_HEAD_PREFETCH_ENV} must be 0 or 1, "
        f"got {_all_task_head_prefetch_raw!r}"
    )
_ALL_TASK_HEAD_PREFETCH = _all_task_head_prefetch_raw == "1"
"""

DISPATCH_FACTORY_ARGS_ANCHOR = """\
    share_input_across_experts: bool = False,
    share_expert_scales: bool = False,
    single_token: bool = False,
    mac_override: int | None = None,
"""
DISPATCH_FACTORY_ARGS_REPLACEMENT = """\
    share_input_across_experts: bool = False,
    share_expert_scales: bool = False,
    single_token: bool = False,
    all_task_head_prefetch: bool = False,
    mac_override: int | None = None,
"""

DISPATCH_FACTORY_GUARD_ANCHOR = """\
    \"\"\"Compile (or retrieve cached) the SM120 micro MoE kernel.\"\"\"
    sf_vec_size = 16
"""
DISPATCH_FACTORY_GUARD_REPLACEMENT = """\
    \"\"\"Compile (or retrieve cached) the SM120 micro MoE kernel.\"\"\"
    if all_task_head_prefetch and not 2 <= m <= 4:
        raise ValueError("all-task head prefetch requires micro M in [2, 4]")
    sf_vec_size = 16
"""

DISPATCH_FACTORY_CACHE_ANCHOR = """\
        share_input_across_experts,
        share_expert_scales,
        single_token,
        activation,
"""
DISPATCH_FACTORY_CACHE_REPLACEMENT = """\
        share_input_across_experts,
        share_expert_scales,
        single_token,
        all_task_head_prefetch,
        activation,
"""

DISPATCH_FACTORY_KERNEL_ANCHOR = """\
        share_input_across_experts=share_input_across_experts,
        share_expert_scales=share_expert_scales,
        single_token=single_token,
    )
"""
DISPATCH_FACTORY_KERNEL_REPLACEMENT = """\
        share_input_across_experts=share_input_across_experts,
        share_expert_scales=share_expert_scales,
        single_token=single_token,
        all_task_head_prefetch=all_task_head_prefetch,
    )
"""

DISPATCH_POLICY_ANCHOR = """\
    use_micro = activation_precision == "fp4" and routed_rows <= micro_cutover

    sm_count = get_num_sm(torch.device("cuda"))
"""
DISPATCH_POLICY_REPLACEMENT = """\
    use_micro = activation_precision == "fp4" and routed_rows <= micro_cutover
    # Keep the hint compile-time and cache-keyed.  Static/prefill, M=1, and
    # M>4 remain byte-for-byte on the accepted path even when the env is set.
    all_task_head_prefetch = (
        _ALL_TASK_HEAD_PREFETCH and use_micro and 2 <= num_tokens <= 4
    )

    sm_count = get_num_sm(torch.device("cuda"))
"""

DISPATCH_CALL_ANCHOR = """\
            share_input_across_experts=share_input_across_experts,
            share_expert_scales=share_expert_scales,
            single_token=pairwise_routes,
            mac_override=micro_mac,
"""
DISPATCH_CALL_REPLACEMENT = """\
            share_input_across_experts=share_input_across_experts,
            share_expert_scales=share_expert_scales,
            single_token=pairwise_routes,
            all_task_head_prefetch=all_task_head_prefetch,
            mac_override=micro_mac,
"""


MICRO_INIT_ARGS_ANCHOR = """\
        share_input_across_experts: bool = False,
        share_expert_scales: bool = False,
        single_token: bool = False,
    ):
"""
MICRO_INIT_ARGS_REPLACEMENT = """\
        share_input_across_experts: bool = False,
        share_expert_scales: bool = False,
        single_token: bool = False,
        all_task_head_prefetch: bool = False,
    ):
"""

MICRO_INIT_ATTRS_ANCHOR = """\
        self.share_input_across_experts = share_input_across_experts
        self.share_expert_scales = share_expert_scales
        self.single_token = single_token
        tile_k = sf_vec_size * 8
"""
MICRO_INIT_ATTRS_REPLACEMENT = """\
        self.share_input_across_experts = share_input_across_experts
        self.share_expert_scales = share_expert_scales
        self.single_token = single_token
        self.all_task_head_prefetch = all_task_head_prefetch
        if self.all_task_head_prefetch and self.single_token:
            raise ValueError("all-task head prefetch must not specialize M=1")
        tile_k = sf_vec_size * 8
"""

MICRO_PREPASS_ANCHOR = """\
        output_tile_cnt = cute.size(gB_down, mode=[2])

        prod_state = pipeline.make_pipeline_state(
"""
MICRO_PREPASS_REPLACEMENT = """\
        output_tile_cnt = cute.size(gB_down, mode=[2])

        if cutlass.const_expr(self.all_task_head_prefetch):
            # Compact routing is complete at this point.  Collectively the DMA
            # warps cover every persistent task exactly once, including later
            # waves, before any MMA warp starts the ordinary task loop.
            if warp_idx == self.tma_load_warp_id:
                prefetch_num_persistent_clusters = Int32(gdim_z)
                prefetch_cluster_shape_mn = (
                    Int32(self.cluster_shape_mn[0]),
                    Int32(self.cluster_shape_mn[1]),
                )
                prefetch_cta_id_in_cluster = (
                    Int32(bidx % prefetch_cluster_shape_mn[0]),
                    Int32(bidy % prefetch_cluster_shape_mn[1]),
                    Int32(0),
                )
                prefetch_work_linear_idx = Int32(bidz)
                prefetch_local_expert_idx = Int32(0)
                prefetch_accum_tile_m = Int32(0)
                (
                    prefetch_tile_coord,
                    prefetch_is_valid_tile,
                    prefetch_local_expert_idx,
                    prefetch_accum_tile_m,
                ) = _compact_static_get_work_tile(
                    row_counts,
                    active_expert_count,
                    tile_m=Int32(self.tile_shape_mnk[0]),
                    num_tiles_n=Int32(self.output_tile_count_n),
                    cluster_shape_mn=prefetch_cluster_shape_mn,
                    current_work_linear_idx=prefetch_work_linear_idx,
                    current_local_expert_idx=prefetch_local_expert_idx,
                    accum_tile_m=prefetch_accum_tile_m,
                    cta_id_in_cluster=prefetch_cta_id_in_cluster,
                )

                while prefetch_is_valid_tile:
                    prefetch_intermediate_slice = prefetch_tile_coord[1]
                    prefetch_local_expert = prefetch_tile_coord[2]
                    prefetch_weight_expert = weight_expert_ids[
                        prefetch_local_expert
                    ]
                    prefetch_gate_slice = (
                        prefetch_intermediate_slice + gate_tile_cnt
                        if self.is_gated
                        else prefetch_intermediate_slice
                    )
                    prefetch_w13_gate = tBgB_w13[
                        (
                            None,
                            prefetch_gate_slice,
                            None,
                            prefetch_weight_expert,
                        )
                    ]
                    prefetch_sfb_gate = tBgSFB_w13[
                        (
                            None,
                            prefetch_gate_slice // self.sfb_tiles_per_block,
                            None,
                            prefetch_weight_expert,
                        )
                    ]
                    prefetch_w13_up = tBgB_w13[
                        (
                            None,
                            prefetch_intermediate_slice,
                            None,
                            prefetch_weight_expert,
                        )
                    ]
                    prefetch_sfb_up = tBgSFB_w13[
                        (
                            None,
                            prefetch_intermediate_slice
                            // self.sfb_tiles_per_block,
                            None,
                            prefetch_weight_expert,
                        )
                    ]

                    for prefetch_k_tile in cutlass.range(
                        cutlass.min(Int32(2), fc1_k_tile_cnt), unroll=1
                    ):
                        cute.prefetch(
                            tma_b_w13,
                            prefetch_w13_gate[(None, prefetch_k_tile)],
                        )
                        cute.prefetch(
                            tma_sfb_w13,
                            prefetch_sfb_gate[(None, prefetch_k_tile)],
                        )
                        if cutlass.const_expr(self.is_gated):
                            cute.prefetch(
                                tma_b_w13,
                                prefetch_w13_up[(None, prefetch_k_tile)],
                            )
                            cute.prefetch(
                                tma_sfb_w13,
                                prefetch_sfb_up[(None, prefetch_k_tile)],
                            )

                    for prefetch_output_tile in cutlass.range(
                        cutlass.min(Int32(2), output_tile_cnt), unroll=1
                    ):
                        cute.prefetch(
                            tma_b_down,
                            tBgB_down[
                                (
                                    None,
                                    prefetch_output_tile,
                                    prefetch_intermediate_slice,
                                    prefetch_weight_expert,
                                )
                            ],
                        )
                        cute.prefetch(
                            tma_sfb_down,
                            tBgSFB_down[
                                (
                                    None,
                                    prefetch_output_tile
                                    // self.sfb_tiles_per_block,
                                    prefetch_intermediate_slice,
                                    prefetch_weight_expert,
                                )
                            ],
                        )

                    prefetch_work_linear_idx += prefetch_num_persistent_clusters
                    (
                        prefetch_tile_coord,
                        prefetch_is_valid_tile,
                        prefetch_local_expert_idx,
                        prefetch_accum_tile_m,
                    ) = _compact_static_get_work_tile(
                        row_counts,
                        active_expert_count,
                        tile_m=Int32(self.tile_shape_mnk[0]),
                        num_tiles_n=Int32(self.output_tile_count_n),
                        cluster_shape_mn=prefetch_cluster_shape_mn,
                        current_work_linear_idx=prefetch_work_linear_idx,
                        current_local_expert_idx=prefetch_local_expert_idx,
                        accum_tile_m=prefetch_accum_tile_m,
                        cta_id_in_cluster=prefetch_cta_id_in_cluster,
                    )

            # Publish every task's hints before wave zero.  This is the only
            # added ordering point; all existing compute pipelines are intact.
            self._resident_grid_barrier(
                barrier_count,
                barrier_epoch,
                Int32(gdim_z),
                is_cta_leader,
            )

        prod_state = pipeline.make_pipeline_state(
"""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _replace_once(source: str, anchor: str, replacement: str, label: str) -> str:
    count = source.count(anchor)
    if count != 1:
        raise RuntimeError(f"expected one {label} anchor, found {count}")
    return source.replace(anchor, replacement, 1)


def patch_dispatch_source(source: str) -> str:
    replacements = (
        (DISPATCH_ENV_ANCHOR, DISPATCH_ENV_REPLACEMENT, "dispatcher env"),
        (
            DISPATCH_FACTORY_ARGS_ANCHOR,
            DISPATCH_FACTORY_ARGS_REPLACEMENT,
            "dispatcher factory args",
        ),
        (
            DISPATCH_FACTORY_GUARD_ANCHOR,
            DISPATCH_FACTORY_GUARD_REPLACEMENT,
            "dispatcher M guard",
        ),
        (
            DISPATCH_FACTORY_CACHE_ANCHOR,
            DISPATCH_FACTORY_CACHE_REPLACEMENT,
            "dispatcher cache key",
        ),
        (
            DISPATCH_FACTORY_KERNEL_ANCHOR,
            DISPATCH_FACTORY_KERNEL_REPLACEMENT,
            "dispatcher kernel construction",
        ),
        (
            DISPATCH_POLICY_ANCHOR,
            DISPATCH_POLICY_REPLACEMENT,
            "dispatcher micro policy",
        ),
        (DISPATCH_CALL_ANCHOR, DISPATCH_CALL_REPLACEMENT, "dispatcher call"),
    )
    for anchor, replacement, label in replacements:
        source = _replace_once(source, anchor, replacement, label)
    return source


def patch_micro_kernel_source(source: str) -> str:
    replacements = (
        (MICRO_INIT_ARGS_ANCHOR, MICRO_INIT_ARGS_REPLACEMENT, "micro init args"),
        (
            MICRO_INIT_ATTRS_ANCHOR,
            MICRO_INIT_ATTRS_REPLACEMENT,
            "micro init attrs",
        ),
        (MICRO_PREPASS_ANCHOR, MICRO_PREPASS_REPLACEMENT, "micro prepass"),
    )
    for anchor, replacement, label in replacements:
        source = _replace_once(source, anchor, replacement, label)
    return source


def _prepare_file(
    path: Path,
    expected_sha: str,
    transform: Callable[[str], str],
    label: str,
) -> tuple[bytes, str]:
    original = path.read_bytes()
    actual_sha = _sha256(original)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"pinned {label} SHA-256 mismatch: "
            f"expected {expected_sha}, got {actual_sha}"
        )
    patched = transform(original.decode("utf-8")).encode("utf-8")
    return patched, _sha256(patched)


def patch_paths(dispatch_target: Path, micro_kernel_target: Path) -> tuple[str, str]:
    """Validate and transform both sources before changing either file."""
    dispatch_patched, dispatch_result = _prepare_file(
        dispatch_target,
        PINNED_DISPATCH_SHA256,
        patch_dispatch_source,
        "FlashInfer B12X dispatcher",
    )
    micro_patched, micro_result = _prepare_file(
        micro_kernel_target,
        PINNED_MICRO_KERNEL_SHA256,
        patch_micro_kernel_source,
        "FlashInfer B12X microkernel",
    )
    dispatch_target.write_bytes(dispatch_patched)
    micro_kernel_target.write_bytes(micro_patched)
    return dispatch_result, micro_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch-target", type=Path, default=DEFAULT_DISPATCH_TARGET)
    parser.add_argument(
        "--micro-kernel-target", type=Path, default=DEFAULT_MICRO_KERNEL_TARGET
    )
    args = parser.parse_args()

    dispatch_result, micro_result = patch_paths(
        args.dispatch_target, args.micro_kernel_target
    )
    print(
        "added opt-in B12X all-task weight-head prefetch: "
        f"dispatch_result={dispatch_result} micro_result={micro_result}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
