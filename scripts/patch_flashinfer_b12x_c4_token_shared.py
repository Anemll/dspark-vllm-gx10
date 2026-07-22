#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Share FP4 activation quantization across tokens without expanding routes.

The accepted W4A4 decode path keeps compact expert routing for M>1 so that
colliding tokens still reuse the same expert FC1/FC2 work.  It nevertheless
quantizes the same activation once for every top-k route.  At C4/top-k=6 this
means 24 identical-token packs per layer although only four source rows exist.

This source-pinned overlay preserves compact routing and the existing MMA work:

* route CTAs write their normal compact `(expert, row)` mapping;
* the first route of each token packs that token into a reserved tail workspace
  slot; and
* after the existing resident-grid barrier, every route CTA fans out the packed
  bytes and scales into its compact destination before a second barrier.

No host synchronization, allocation, routing policy, weight selection, or
reduction arithmetic changes.  M=1 and M>4 remain byte-for-byte on the
accepted image.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


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


DISPATCH_FLAGS_ANCHOR = """\
    # Shared-scale flags let compact W4A4 micro match the ReLU2 single-token
    # specialization.
    share_input_across_experts = (
        num_tokens == 1
        and input_gs_is_shared
        and _MICRO_SHARE_INPUT_ACROSS_EXPERTS
    )
    share_expert_scales = input_gs_is_shared and down_input_scale_is_shared
"""
DISPATCH_FLAGS_REPLACEMENT = """\
    # Keep M=1's accepted direct-route path unchanged. For C2--C4 retain
    # compact expert routing, but share the per-token FP4 pack before copying
    # it into each compact `(expert, row)` destination.
    share_input_across_experts = (
        num_tokens == 1
        and input_gs_is_shared
        and _MICRO_SHARE_INPUT_ACROSS_EXPERTS
    )
    per_token_shared_input = (
        use_micro
        and 2 <= num_tokens <= 4
        and input_gs_is_shared
        and _MICRO_SHARE_INPUT_ACROSS_EXPERTS
    )
    share_expert_scales = input_gs_is_shared and down_input_scale_is_shared
"""
DISPATCH_CALL_ANCHOR = """\
            share_input_across_experts=share_input_across_experts,
            share_expert_scales=share_expert_scales,
            single_token=pairwise_routes,
"""
DISPATCH_CALL_REPLACEMENT = """\
            share_input_across_experts=share_input_across_experts,
            share_expert_scales=share_expert_scales,
            per_token_shared_input=per_token_shared_input,
            single_token=pairwise_routes,
"""
DISPATCH_FACTORY_ARGS_ANCHOR = """\
    share_input_across_experts: bool = False,
    share_expert_scales: bool = False,
    single_token: bool = False,
"""
DISPATCH_FACTORY_ARGS_REPLACEMENT = """\
    share_input_across_experts: bool = False,
    share_expert_scales: bool = False,
    per_token_shared_input: bool = False,
    single_token: bool = False,
"""
DISPATCH_FACTORY_CACHE_ANCHOR = """\
        share_input_across_experts,
        share_expert_scales,
        single_token,
"""
DISPATCH_FACTORY_CACHE_REPLACEMENT = """\
        share_input_across_experts,
        share_expert_scales,
        per_token_shared_input,
        single_token,
"""
DISPATCH_FACTORY_KERNEL_ANCHOR = """\
        share_input_across_experts=share_input_across_experts,
        share_expert_scales=share_expert_scales,
        single_token=single_token,
"""
DISPATCH_FACTORY_KERNEL_REPLACEMENT = """\
        share_input_across_experts=share_input_across_experts,
        share_expert_scales=share_expert_scales,
        per_token_shared_input=per_token_shared_input,
        single_token=single_token,
"""
MICRO_INIT_ARGS_ANCHOR = """\
        share_input_across_experts: bool = False,
        share_expert_scales: bool = False,
        single_token: bool = False,
"""
MICRO_INIT_ARGS_REPLACEMENT = """\
        share_input_across_experts: bool = False,
        share_expert_scales: bool = False,
        per_token_shared_input: bool = False,
        single_token: bool = False,
"""
MICRO_INIT_ATTRS_ANCHOR = """\
        self.share_input_across_experts = share_input_across_experts
        self.share_expert_scales = share_expert_scales
        self.single_token = single_token
"""
MICRO_INIT_ATTRS_REPLACEMENT = """\
        self.share_input_across_experts = share_input_across_experts
        self.share_expert_scales = share_expert_scales
        self.per_token_shared_input = per_token_shared_input
        self.single_token = single_token
"""
MICRO_QUANT_ANCHOR = """\
            if cutlass.const_expr(self.share_input_across_experts):
                if cutlass.const_expr(self.single_token):
                    # Match FI main: pair 0 writes the shared packed input slot,
                    # and the resident-grid barrier below makes it visible to
                    # CTAs that did not participate in Phase 1 routing work.
                    should_quantize = Int32(1) if pair_idx == Int32(0) else Int32(0)
                    quant_expert_id = topk_ids[Int32(0)].to(Int32)
                    packed_local_expert_id = Int32(0)
                    packed_row = Int32(0)
                else:
                    should_quantize = Int32(1) if pair_idx == Int32(0) else Int32(0)
                    packed_local_expert_id = Int32(0)
                    packed_row = Int32(0)
"""
MICRO_QUANT_REPLACEMENT = """\
            if cutlass.const_expr(self.share_input_across_experts):
                if cutlass.const_expr(self.single_token):
                    # Match FI main: pair 0 writes the shared packed input slot,
                    # and the resident-grid barrier below makes it visible to
                    # CTAs that did not participate in Phase 1 routing work.
                    should_quantize = Int32(1) if pair_idx == Int32(0) else Int32(0)
                    quant_expert_id = topk_ids[Int32(0)].to(Int32)
                    packed_local_expert_id = Int32(0)
                    packed_row = Int32(0)
                else:
                    should_quantize = Int32(1) if pair_idx == Int32(0) else Int32(0)
                    packed_local_expert_id = Int32(0)
                    packed_row = Int32(0)
            elif cutlass.const_expr(self.per_token_shared_input):
                # Tail slots cannot collide with compact local experts: at most
                # M*top-k routes are active while `num_experts` is state_E.
                token_route = pair_idx % num_topk
                should_quantize = Int32(1) if token_route == Int32(0) else Int32(0)
                packed_local_expert_id = num_experts - num_tokens + token_idx
                packed_row = Int32(0)
"""
MICRO_PACK_BARRIER_ANCHOR = """\
        self._resident_grid_barrier(
            barrier_count,
            barrier_epoch,
            Int32(gdim_z),
            is_cta_leader,
        )

        gA = cute.local_tile(mA, self.sa_tile_shape_mk, (None, None, None))
"""
MICRO_PACK_BARRIER_REPLACEMENT = """\
        self._resident_grid_barrier(
            barrier_count,
            barrier_epoch,
            Int32(gdim_z),
            is_cta_leader,
        )

        if cutlass.const_expr(self.per_token_shared_input):
            # Compact routing selected the destination row for every route in
            # `token_map`. Copy the one tail-slot pack for this token into that
            # destination; FC1 therefore retains the existing compact layout.
            pair_idx = Int32(bidz)
            while pair_idx < total_pairs:
                token_idx = pair_idx // num_topk
                local_expert_id = topk_ids[pair_idx].to(Int32)
                row_count = row_counts[local_expert_id]
                compact_row = Int32(0)
                scan_row = Int32(0)
                while scan_row < row_count:
                    mapped_token = token_map[local_expert_id * max_rows + scan_row]
                    if mapped_token == token_idx:
                        compact_row = scan_row
                    scan_row += Int32(1)

                source_expert = num_experts - num_tokens + token_idx
                word_idx = Int32(tidx)
                words_per_row = output_bytes_per_row // Int32(8)
                while word_idx < words_per_row:
                    source_offset = (
                        source_expert * max_rows * output_bytes_per_row
                        + word_idx * Int32(8)
                    )
                    dest_offset = (
                        local_expert_id * max_rows * output_bytes_per_row
                        + compact_row * output_bytes_per_row
                        + word_idx * Int32(8)
                    )
                    st_global_u64(
                        get_ptr_as_int64(packed_a_storage, dest_offset),
                        _ld_global_u64(get_ptr_as_int64(packed_a_storage, source_offset)),
                    )
                    word_idx += Int32(self.threads_per_cta)

                sf_idx = Int32(tidx)
                while sf_idx < sf_blocks_per_row:
                    m_tile_idx = compact_row // Int32(32 * 4)
                    k_tile_idx = sf_idx // Int32(4)
                    outer_m_idx = compact_row % Int32(32)
                    inner_m_idx = (compact_row % Int32(32 * 4)) // Int32(32)
                    inner_k_idx = sf_idx % Int32(4)
                    dest_scale_offset = (
                        local_expert_id * expert_scale_stride
                        + m_tile_idx * num_k_tiles * Int32(32 * 4 * 4)
                        + k_tile_idx * Int32(32 * 4 * 4)
                        + outer_m_idx * Int32(4 * 4)
                        + inner_m_idx * Int32(4)
                        + inner_k_idx
                    )
                    source_scale_offset = (
                        source_expert * expert_scale_stride
                        + k_tile_idx * Int32(32 * 4 * 4)
                        + inner_k_idx
                    )
                    scale_storage[dest_scale_offset] = scale_storage[source_scale_offset]
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
        source, DISPATCH_FLAGS_ANCHOR, DISPATCH_FLAGS_REPLACEMENT, "dispatcher flags"
    )
    source = _replace_once(
        source, DISPATCH_CALL_ANCHOR, DISPATCH_CALL_REPLACEMENT, "dispatcher call"
    )
    source = _replace_once(
        source,
        DISPATCH_FACTORY_ARGS_ANCHOR,
        DISPATCH_FACTORY_ARGS_REPLACEMENT,
        "dispatcher factory args",
    )
    source = _replace_once(
        source,
        DISPATCH_FACTORY_CACHE_ANCHOR,
        DISPATCH_FACTORY_CACHE_REPLACEMENT,
        "dispatcher factory cache",
    )
    return _replace_once(
        source,
        DISPATCH_FACTORY_KERNEL_ANCHOR,
        DISPATCH_FACTORY_KERNEL_REPLACEMENT,
        "dispatcher factory kernel",
    )


def patch_micro_kernel_source(source: str) -> str:
    source = _replace_once(
        source, MICRO_INIT_ARGS_ANCHOR, MICRO_INIT_ARGS_REPLACEMENT, "micro init args"
    )
    source = _replace_once(
        source, MICRO_INIT_ATTRS_ANCHOR, MICRO_INIT_ATTRS_REPLACEMENT, "micro init attrs"
    )
    source = _replace_once(
        source, MICRO_QUANT_ANCHOR, MICRO_QUANT_REPLACEMENT, "micro quant"
    )
    return _replace_once(
        source,
        MICRO_PACK_BARRIER_ANCHOR,
        MICRO_PACK_BARRIER_REPLACEMENT,
        "micro pack barrier",
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
        "--micro-kernel-target", type=Path, default=DEFAULT_MICRO_KERNEL_TARGET
    )
    args = parser.parse_args()
    dispatch_sha = _patch_file(
        args.dispatch_target,
        PINNED_DISPATCH_SHA256,
        patch_dispatch_source,
        "FlashInfer B12X dispatcher",
    )
    micro_sha = _patch_file(
        args.micro_kernel_target,
        PINNED_MICRO_KERNEL_SHA256,
        patch_micro_kernel_source,
        "FlashInfer B12X microkernel",
    )
    print(
        "patched FlashInfer B12X compact C2--C4 token-shared activations: "
        f"dispatch_result={dispatch_sha} micro_result={micro_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
