#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Add rolling L2 weight prefetch to the pinned SM121 B12X microkernel.

The accepted kernel prefetches TMA descriptors but issues no data prefetch.
During full-model decode each routed layer streams a new expert-weight set,
which costs about 46 us/layer relative to a hot one-layer replay on GB10.

This patch follows CUTLASS's Blackwell TMA tutorial: prefetch the first
``ab_stage`` tiles, then roll the prefetch window ahead of the TMA producer.
Only read-only weight and weight-scale tensors are prefetched.  Computation,
routing, numerical order, shared-memory layout, and output writes are
unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


PINNED_SOURCE_SHA256 = (
    "9ef89f9f9d806e8e2904e3bd345b69c9c8a0e1d0643d21d8975e6e3ae8c8a6ed"
)
DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/"
    "cute_dsl/blackwell_sm12x/moe_micro_kernel.py"
)

_FC1_START_ANCHOR = """\
                # ---- FC1 gate pass ----
                prod_state.reset_count()
"""
_FC1_START_REPLACEMENT = """\
                # Prime the two-stage TMA pipeline from DRAM into L2.
                prefetch_dist = self.ab_stage
                for pf_k_tile in cutlass.range(
                    cutlass.min(prefetch_dist, fc1_k_tile_cnt), unroll=1
                ):
                    cute.prefetch(
                        tma_b_w13, tBgB_w13_gate_nk[(None, pf_k_tile)]
                    )
                    cute.prefetch(
                        tma_sfb_w13, tBgSFB_w13_gate_nk[(None, pf_k_tile)]
                    )
                    if cutlass.const_expr(self.is_gated):
                        cute.prefetch(
                            tma_b_w13, tBgB_w13_up_nk[(None, pf_k_tile)]
                        )
                        cute.prefetch(
                            tma_sfb_w13,
                            tBgSFB_w13_up_nk[(None, pf_k_tile)],
                        )

                # ---- FC1 gate pass ----
                prod_state.reset_count()
"""

_FC1_ROLL_ANCHOR = """\
                    ml_pipeline.producer_commit(prod_state)
                    prod_state.advance()

                # Wait for the MMA warps to finish the FC1 gate/only pass
"""
_FC1_ROLL_REPLACEMENT = """\
                    future_k_tile = k_tile + prefetch_dist
                    if future_k_tile < fc1_k_tile_cnt:
                        cute.prefetch(
                            tma_b_w13,
                            tBgB_w13_gate_nk[(None, future_k_tile)],
                        )
                        cute.prefetch(
                            tma_sfb_w13,
                            tBgSFB_w13_gate_nk[(None, future_k_tile)],
                        )
                    ml_pipeline.producer_commit(prod_state)
                    prod_state.advance()

                # Wait for the MMA warps to finish the FC1 gate/only pass
"""

_UP_ROLL_ANCHOR = """\
                        up_pipeline.producer_commit(up_prod_state)
                        up_prod_state.advance()

                # ---- FC2 B_down loads: continuous pipeline ----
"""
_UP_ROLL_REPLACEMENT = """\
                        future_k_tile = k_tile + prefetch_dist
                        if future_k_tile < fc1_k_tile_cnt:
                            cute.prefetch(
                                tma_b_w13,
                                tBgB_w13_up_nk[(None, future_k_tile)],
                            )
                            cute.prefetch(
                                tma_sfb_w13,
                                tBgSFB_w13_up_nk[(None, future_k_tile)],
                            )
                        up_pipeline.producer_commit(up_prod_state)
                        up_prod_state.advance()

                # ---- FC2 B_down loads: continuous pipeline ----
"""

_FC2_START_ANCHOR = """\
                phase2_prod_state.reset_count()
                for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
"""
_FC2_START_REPLACEMENT = """\
                for pf_output_tile in cutlass.range(
                    cutlass.min(prefetch_dist, output_tile_cnt), unroll=1
                ):
                    cute.prefetch(
                        tma_b_down,
                        tBgB_down[
                            (
                                None,
                                pf_output_tile,
                                intermediate_slice,
                                weight_expert_idx,
                            )
                        ],
                    )
                    cute.prefetch(
                        tma_sfb_down,
                        tBgSFB_down[
                            (
                                None,
                                pf_output_tile // self.sfb_tiles_per_block,
                                intermediate_slice,
                                weight_expert_idx,
                            )
                        ],
                    )

                phase2_prod_state.reset_count()
                for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
"""

_FC2_ROLL_ANCHOR = """\
                    phase2_pipeline.producer_commit(phase2_prod_state)
                    phase2_prod_state.advance()

                # Final pass_sync: match MMA warps' barrier after FC2 sweep.
"""
_FC2_ROLL_REPLACEMENT = """\
                    future_output_tile = output_tile_idx + prefetch_dist
                    if future_output_tile < output_tile_cnt:
                        cute.prefetch(
                            tma_b_down,
                            tBgB_down[
                                (
                                    None,
                                    future_output_tile,
                                    intermediate_slice,
                                    weight_expert_idx,
                                )
                            ],
                        )
                        cute.prefetch(
                            tma_sfb_down,
                            tBgSFB_down[
                                (
                                    None,
                                    future_output_tile
                                    // self.sfb_tiles_per_block,
                                    intermediate_slice,
                                    weight_expert_idx,
                                )
                            ],
                        )
                    phase2_pipeline.producer_commit(phase2_prod_state)
                    phase2_prod_state.advance()

                # Final pass_sync: match MMA warps' barrier after FC2 sweep.
"""

_REPLACEMENTS = (
    (_FC1_START_ANCHOR, _FC1_START_REPLACEMENT, "FC1 prefetch start"),
    (_FC1_ROLL_ANCHOR, _FC1_ROLL_REPLACEMENT, "FC1 rolling prefetch"),
    (_UP_ROLL_ANCHOR, _UP_ROLL_REPLACEMENT, "up rolling prefetch"),
    (_FC2_START_ANCHOR, _FC2_START_REPLACEMENT, "FC2 prefetch start"),
    (_FC2_ROLL_ANCHOR, _FC2_ROLL_REPLACEMENT, "FC2 rolling prefetch"),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch_source(source: str) -> str:
    for anchor, replacement, label in _REPLACEMENTS:
        count = source.count(anchor)
        if count != 1:
            raise RuntimeError(f"expected one {label} anchor, found {count}")
        source = source.replace(anchor, replacement, 1)
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()

    original = args.target.read_bytes()
    original_sha = _sha256(original)
    if original_sha != PINNED_SOURCE_SHA256:
        raise RuntimeError(
            "pinned moe_micro_kernel.py SHA-256 mismatch: "
            f"expected {PINNED_SOURCE_SHA256}, got {original_sha}"
        )
    patched = patch_source(original.decode("utf-8")).encode("utf-8")
    args.target.write_bytes(patched)
    print(
        "added B12X SM121 rolling L2 weight prefetch: "
        f"source={original_sha} result={_sha256(patched)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
