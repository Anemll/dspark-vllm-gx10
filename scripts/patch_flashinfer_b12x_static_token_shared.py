#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Share one FP4 activation pack per token in FlashInfer's static MoE kernel.

DSpark verifies ``draft_tokens + 1`` target positions together.  At
concurrency four/five drafts this gives 24 target rows, and top-k=6 expands
them to 144 routed pairs.  The pinned SM121 static kernel currently quantizes
the same BF16 row independently for every routed expert.

Prepared DeepSeek V4 target layers have adapter-proven uniform FC1 input
global scales.  With the explicit environment opt-in below, this patch:

* lets only the first route of each token quantize the BF16 row to NVFP4;
* copies that packed row and its scale bytes to the token's other expert-major
  destinations after the existing route/pack barrier; and
* keeps expert grouping, FC1/FC2 work, router weights, and output scatter
  unchanged.

The path is disabled by default and fail-closes unless the dispatcher received
a scalar input scale:

``FLASHINFER_B12X_STATIC_TOKEN_SHARED_INPUT=1``
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


PINNED_SHARED_INPUT_DISPATCH_SHA256 = (
    "253cc2f26d465adc37e48c4eee53bdb534bf6fb371a3823a5923cc8d45e2d0d3"
)
PINNED_STATIC_KERNEL_SHA256 = (
    "22ebdbb99268df4d1416c105b35738e28cc44ffe46361e25c85d5f4b478e3625"
)
DEFAULT_DISPATCH_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/"
    "cute_dsl/blackwell_sm12x/moe_dispatch.py"
)
DEFAULT_STATIC_KERNEL_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/"
    "cute_dsl/blackwell_sm12x/moe_static_kernel.py"
)


DISPATCH_ENV_ANCHOR = """\
_MICRO_SHARE_INPUT_ACROSS_EXPERTS = (
    os.environ.get("FLASHINFER_B12X_MICRO_SHARE_INPUT", "1") != "0"
)
"""
DISPATCH_ENV_REPLACEMENT = """\
_MICRO_SHARE_INPUT_ACROSS_EXPERTS = (
    os.environ.get("FLASHINFER_B12X_MICRO_SHARE_INPUT", "1") != "0"
)
_STATIC_TOKEN_SHARED_INPUT_ENV = "FLASHINFER_B12X_STATIC_TOKEN_SHARED_INPUT"
_static_token_shared_input_raw = os.environ.get(
    _STATIC_TOKEN_SHARED_INPUT_ENV, "0"
)
if _static_token_shared_input_raw not in {"0", "1"}:
    raise ValueError(
        f"{_STATIC_TOKEN_SHARED_INPUT_ENV} must be 0 or 1; "
        f"got {_static_token_shared_input_raw!r}"
    )
_STATIC_TOKEN_SHARED_INPUT = _static_token_shared_input_raw == "1"
"""

STATIC_FACTORY_ARGS_ANCHOR = """\
    input_scales_are_reciprocal: bool = False,
    fast_math: bool = True,
    mac_override: int | None = None,
"""
STATIC_FACTORY_ARGS_REPLACEMENT = """\
    input_scales_are_reciprocal: bool = False,
    fast_math: bool = True,
    per_token_shared_input: bool = False,
    mac_override: int | None = None,
"""

STATIC_FACTORY_CACHE_ANCHOR = """\
        "static",
        activation_precision,
        state_E,
        weight_E,
        m,
        k,
        n,
        num_topk,
        max_rows,
        mac,
        mma_tiler_mn,
        topk_ids_dtype,
        input_scales_are_reciprocal,
        fast_math,
        activation,
"""
STATIC_FACTORY_CACHE_REPLACEMENT = """\
        "static",
        activation_precision,
        state_E,
        weight_E,
        m,
        k,
        n,
        num_topk,
        max_rows,
        mac,
        mma_tiler_mn,
        topk_ids_dtype,
        input_scales_are_reciprocal,
        fast_math,
        per_token_shared_input,
        activation,
"""

STATIC_FACTORY_KERNEL_ANCHOR = """\
        swiglu_limit=swiglu_limit,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
    )
"""
STATIC_FACTORY_KERNEL_REPLACEMENT = """\
        swiglu_limit=swiglu_limit,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        per_token_shared_input=per_token_shared_input,
    )
"""

DISPATCH_FLAGS_ANCHOR = """\
    share_expert_scales = input_gs_is_shared and down_input_scale_is_shared

    if use_micro:
"""
DISPATCH_FLAGS_REPLACEMENT = """\
    share_expert_scales = input_gs_is_shared and down_input_scale_is_shared
    per_token_shared_input = (
        activation_precision == "fp4"
        and not use_micro
        and num_tokens > 1
        and input_gs_is_shared
        and _STATIC_TOKEN_SHARED_INPUT
    )

    if use_micro:
"""

DISPATCH_STATIC_CALL_ANCHOR = """\
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
            mac_override=static_mac,
"""
DISPATCH_STATIC_CALL_REPLACEMENT = """\
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
            per_token_shared_input=per_token_shared_input,
            mac_override=static_mac,
"""

STATIC_INIT_ARGS_ANCHOR = """\
        input_scales_are_reciprocal: bool = False,
        fast_math: bool = False,
        activation: str = "silu",
"""
STATIC_INIT_ARGS_REPLACEMENT = """\
        input_scales_are_reciprocal: bool = False,
        fast_math: bool = False,
        per_token_shared_input: bool = False,
        activation: str = "silu",
"""

STATIC_INIT_ATTRS_ANCHOR = """\
        self.input_scales_are_reciprocal = input_scales_are_reciprocal
        self.fast_math = fast_math
        self.activation = activation
"""
STATIC_INIT_ATTRS_REPLACEMENT = """\
        self.input_scales_are_reciprocal = input_scales_are_reciprocal
        self.fast_math = fast_math
        self.per_token_shared_input = per_token_shared_input
        self.activation = activation
"""

STATIC_QUANT_ANCHOR = """\
            # Distribute quantization across ALL CTA threads, not just leader.
            # Each FP4 block (16 elements) is independent — perfect parallelism.
            gs_value = input_global_scale[expert_id].to(cutlass.Float32)
            if self.input_scales_are_reciprocal and gs_value != cutlass.Float32(0.0):
                if self.fast_math:
                    gs_value = rcp_approx_ftz(gs_value)
                else:
                    gs_value = cutlass.Float32(1.0) / gs_value
            sf_idx = Int32(tidx)
            while sf_idx < sf_blocks_per_row:
                block_start = sf_idx * Int32(16)
                values = cute.make_rmem_tensor((16,), cutlass.Float32)
                block_max = cutlass.Float32(0.0)
                for elem_idx in cutlass.range_constexpr(16):
                    value = cutlass.Float32(
                        a_input[token_idx, block_start + Int32(elem_idx)]
                    )
                    values[elem_idx] = value
                    block_max = fmax_f32(block_max, fabs_f32(value))
                packed64 = Uint64(0)
                scale_byte = Uint8(0)
                if self.fast_math:
                    packed64, scale_byte = quantize_block_fp4_fast(
                        values, block_max, gs_value
                    )
                else:
                    packed64, scale_byte = quantize_block_fp4(
                        values, block_max, gs_value
                    )

                output_offset = (
                    local_expert_id * max_rows * output_bytes_per_row
                    + row * output_bytes_per_row
                    + sf_idx * Int32(8)
                )
                st_global_u64(
                    get_ptr_as_int64(packed_a_storage, output_offset), packed64
                )

                m_tile_idx = row // Int32(32 * 4)
                k_tile_idx = sf_idx // Int32(4)
                outer_m_idx = row % Int32(32)
                inner_m_idx = (row % Int32(32 * 4)) // Int32(32)
                inner_k_idx = sf_idx % Int32(4)
                scale_offset = (
                    local_expert_id * expert_scale_stride
                    + m_tile_idx * num_k_tiles * Int32(32 * 4 * 4)
                    + k_tile_idx * Int32(32 * 4 * 4)
                    + outer_m_idx * Int32(4 * 4)
                    + inner_m_idx * Int32(4)
                    + inner_k_idx
                )
                scale_storage[scale_offset] = scale_byte
                sf_idx += Int32(self.threads_per_cta)
"""
STATIC_QUANT_REPLACEMENT = """\
            # With uniform input scales, the first top-k route owns the token's
            # single FP4 pack. Other routes receive byte-identical copies after
            # the route/pack barrier below.
            should_quantize = Int32(1)
            if cutlass.const_expr(self.per_token_shared_input):
                should_quantize = (
                    Int32(1)
                    if pair_idx % num_topk == Int32(0)
                    else Int32(0)
                )
            if should_quantize > Int32(0):
                scale_idx = (
                    Int32(0)
                    if cutlass.const_expr(self.per_token_shared_input)
                    else expert_id
                )
                gs_value = input_global_scale[scale_idx].to(cutlass.Float32)
                if self.input_scales_are_reciprocal and gs_value != cutlass.Float32(
                    0.0
                ):
                    if self.fast_math:
                        gs_value = rcp_approx_ftz(gs_value)
                    else:
                        gs_value = cutlass.Float32(1.0) / gs_value
                sf_idx = Int32(tidx)
                while sf_idx < sf_blocks_per_row:
                    block_start = sf_idx * Int32(16)
                    values = cute.make_rmem_tensor((16,), cutlass.Float32)
                    block_max = cutlass.Float32(0.0)
                    for elem_idx in cutlass.range_constexpr(16):
                        value = cutlass.Float32(
                            a_input[token_idx, block_start + Int32(elem_idx)]
                        )
                        values[elem_idx] = value
                        block_max = fmax_f32(block_max, fabs_f32(value))
                    packed64 = Uint64(0)
                    scale_byte = Uint8(0)
                    if self.fast_math:
                        packed64, scale_byte = quantize_block_fp4_fast(
                            values, block_max, gs_value
                        )
                    else:
                        packed64, scale_byte = quantize_block_fp4(
                            values, block_max, gs_value
                        )

                    output_offset = (
                        local_expert_id * max_rows * output_bytes_per_row
                        + row * output_bytes_per_row
                        + sf_idx * Int32(8)
                    )
                    st_global_u64(
                        get_ptr_as_int64(packed_a_storage, output_offset), packed64
                    )

                    m_tile_idx = row // Int32(32 * 4)
                    k_tile_idx = sf_idx // Int32(4)
                    outer_m_idx = row % Int32(32)
                    inner_m_idx = (row % Int32(32 * 4)) // Int32(32)
                    inner_k_idx = sf_idx % Int32(4)
                    scale_offset = (
                        local_expert_id * expert_scale_stride
                        + m_tile_idx * num_k_tiles * Int32(32 * 4 * 4)
                        + k_tile_idx * Int32(32 * 4 * 4)
                        + outer_m_idx * Int32(4 * 4)
                        + inner_m_idx * Int32(4)
                        + inner_k_idx
                    )
                    scale_storage[scale_offset] = scale_byte
                    sf_idx += Int32(self.threads_per_cta)
"""

STATIC_PACK_BARRIER_ANCHOR = """\
        self._resident_grid_barrier(
            barrier_count,
            barrier_epoch,
            Int32(gdim_z),
            is_cta_leader,
        )

        gA = cute.local_tile(mA, self.sa_tile_shape_mk, (None, None, None))
"""
STATIC_PACK_BARRIER_REPLACEMENT = """\
        self._resident_grid_barrier(
            barrier_count,
            barrier_epoch,
            Int32(gdim_z),
            is_cta_leader,
        )

        if cutlass.const_expr(self.per_token_shared_input):
            # Resolve compact expert rows from the finalized route metadata.
            # Row scans are normally one or two entries at these sparse
            # verifier shapes and avoid adding another persistent workspace.
            pair_idx = Int32(bidz)
            while pair_idx < total_pairs:
                token_idx = pair_idx // num_topk
                expert_id = topk_ids[pair_idx].to(Int32)
                local_expert_id = global_to_local_expert[expert_id].to(Int32)
                row_count = row_counts[local_expert_id]
                compact_row = Int32(0)
                scan_row = Int32(0)
                while scan_row < row_count:
                    if token_map[
                        local_expert_id * max_rows + scan_row
                    ] == token_idx:
                        compact_row = scan_row
                    scan_row += Int32(1)

                source_expert_id = topk_ids[token_idx * num_topk].to(Int32)
                source_local_expert_id = global_to_local_expert[
                    source_expert_id
                ].to(Int32)
                source_row_count = row_counts[source_local_expert_id]
                source_row = Int32(0)
                source_scan_row = Int32(0)
                while source_scan_row < source_row_count:
                    if token_map[
                        source_local_expert_id * max_rows + source_scan_row
                    ] == token_idx:
                        source_row = source_scan_row
                    source_scan_row += Int32(1)

                if pair_idx % num_topk != Int32(0):
                    word_idx = Int32(tidx)
                    words_per_row = output_bytes_per_row // Int32(8)
                    while word_idx < words_per_row:
                        source_offset = (
                            source_local_expert_id
                            * max_rows
                            * output_bytes_per_row
                            + source_row * output_bytes_per_row
                            + word_idx * Int32(8)
                        )
                        dest_offset = (
                            local_expert_id * max_rows * output_bytes_per_row
                            + compact_row * output_bytes_per_row
                            + word_idx * Int32(8)
                        )
                        st_global_u64(
                            get_ptr_as_int64(packed_a_storage, dest_offset),
                            _ld_global_u64(
                                get_ptr_as_int64(
                                    packed_a_storage, source_offset
                                )
                            ),
                        )
                        word_idx += Int32(self.threads_per_cta)

                    sf_idx = Int32(tidx)
                    while sf_idx < sf_blocks_per_row:
                        source_m_tile_idx = source_row // Int32(32 * 4)
                        source_outer_m_idx = source_row % Int32(32)
                        source_inner_m_idx = (
                            source_row % Int32(32 * 4)
                        ) // Int32(32)
                        m_tile_idx = compact_row // Int32(32 * 4)
                        k_tile_idx = sf_idx // Int32(4)
                        outer_m_idx = compact_row % Int32(32)
                        inner_m_idx = (
                            compact_row % Int32(32 * 4)
                        ) // Int32(32)
                        inner_k_idx = sf_idx % Int32(4)
                        dest_scale_offset = (
                            local_expert_id * expert_scale_stride
                            + m_tile_idx
                            * num_k_tiles
                            * Int32(32 * 4 * 4)
                            + k_tile_idx * Int32(32 * 4 * 4)
                            + outer_m_idx * Int32(4 * 4)
                            + inner_m_idx * Int32(4)
                            + inner_k_idx
                        )
                        source_scale_offset = (
                            source_local_expert_id * expert_scale_stride
                            + source_m_tile_idx
                            * num_k_tiles
                            * Int32(32 * 4 * 4)
                            + k_tile_idx * Int32(32 * 4 * 4)
                            + source_outer_m_idx * Int32(4 * 4)
                            + source_inner_m_idx * Int32(4)
                            + inner_k_idx
                        )
                        scale_storage[dest_scale_offset] = scale_storage[
                            source_scale_offset
                        ]
                        sf_idx += Int32(self.threads_per_cta)

                cute.arch.sync_threads()
                pair_idx += Int32(gdim_z)

            self._resident_grid_barrier(
                barrier_count,
                barrier_epoch,
                Int32(gdim_z),
                is_cta_leader,
            )

        gA = cute.local_tile(mA, self.sa_tile_shape_mk, (None, None, None))
"""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _replace_once(source: str, anchor: str, replacement: str, label: str) -> str:
    count = source.count(anchor)
    if count != 1:
        raise RuntimeError(f"expected one {label} anchor, found {count}")
    return source.replace(anchor, replacement, 1)


def patch_dispatch_source(source: str) -> str:
    source = _replace_once(
        source, DISPATCH_ENV_ANCHOR, DISPATCH_ENV_REPLACEMENT, "dispatcher env"
    )
    source = _replace_once(
        source,
        STATIC_FACTORY_ARGS_ANCHOR,
        STATIC_FACTORY_ARGS_REPLACEMENT,
        "static factory args",
    )
    source = _replace_once(
        source,
        STATIC_FACTORY_CACHE_ANCHOR,
        STATIC_FACTORY_CACHE_REPLACEMENT,
        "static factory cache",
    )
    source = _replace_once(
        source,
        STATIC_FACTORY_KERNEL_ANCHOR,
        STATIC_FACTORY_KERNEL_REPLACEMENT,
        "static factory kernel",
    )
    source = _replace_once(
        source, DISPATCH_FLAGS_ANCHOR, DISPATCH_FLAGS_REPLACEMENT, "dispatcher flags"
    )
    return _replace_once(
        source,
        DISPATCH_STATIC_CALL_ANCHOR,
        DISPATCH_STATIC_CALL_REPLACEMENT,
        "dispatcher static call",
    )


def patch_static_kernel_source(source: str) -> str:
    source = _replace_once(
        source, STATIC_INIT_ARGS_ANCHOR, STATIC_INIT_ARGS_REPLACEMENT, "static init args"
    )
    source = _replace_once(
        source,
        STATIC_INIT_ATTRS_ANCHOR,
        STATIC_INIT_ATTRS_REPLACEMENT,
        "static init attrs",
    )
    source = _replace_once(
        source, STATIC_QUANT_ANCHOR, STATIC_QUANT_REPLACEMENT, "static quant"
    )
    return _replace_once(
        source,
        STATIC_PACK_BARRIER_ANCHOR,
        STATIC_PACK_BARRIER_REPLACEMENT,
        "static pack barrier",
    )


def _patch_file(path: Path, expected_sha: str, transform, label: str) -> str:
    original = path.read_bytes()
    actual_sha = _sha256(original)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"pinned {label} SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
        )
    patched = transform(original.decode("utf-8")).encode("utf-8")
    path.write_bytes(patched)
    return _sha256(patched)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch-target", type=Path, default=DEFAULT_DISPATCH_TARGET)
    parser.add_argument(
        "--static-kernel-target", type=Path, default=DEFAULT_STATIC_KERNEL_TARGET
    )
    args = parser.parse_args()
    dispatch_result = _patch_file(
        args.dispatch_target,
        PINNED_SHARED_INPUT_DISPATCH_SHA256,
        patch_dispatch_source,
        "shared-input dispatcher",
    )
    static_result = _patch_file(
        args.static_kernel_target,
        PINNED_STATIC_KERNEL_SHA256,
        patch_static_kernel_source,
        "static kernel",
    )
    print(
        "patched FlashInfer B12X static token-shared input: "
        f"dispatch_result={dispatch_result} static_result={static_result}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
