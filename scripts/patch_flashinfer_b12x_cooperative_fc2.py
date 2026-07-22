#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Add an opt-in cooperative FC2-load canary to pinned SM121 B12X W4A4.

The accepted compact M=4 microkernel uses one DMA warp and TMA to stream each
8 KiB FC2 weight tile plus its 1 KiB scale tile into shared memory.  This
source-pinned experiment instead lets the four MMA warps stage FC2 together:

* 128 threads issue 128-bit ``cp.async`` copies for the packed FP4 weight;
* the same threads use the dense kernel's proven ``CopyUniversal`` scale
  partition for E4M3 block scales; and
* the existing 128-thread epilogue named barrier publishes stage zero before
  MMA and protects its reuse after scatter.

Only the exact DeepSeek V4 TP=2 M=4 shape is eligible.  The dispatcher makes
the choice compile-time and includes it in the kernel cache key.  The feature
is off unless ``FLASHINFER_B12X_COOPERATIVE_FC2_M4=1``.  M=1 and every other
shape retain the accepted TMA path.

This is a canary patch, not a default production policy.  Both inputs are
SHA-pinned to accepted revision 646be4d sources and all transformations fail
closed on missing or duplicated anchors.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Callable


ACCEPTED_DISPATCH_SHA256 = (
    "253cc2f26d465adc37e48c4eee53bdb534bf6fb371a3823a5923cc8d45e2d0d3"
)
ACCEPTED_MICRO_SHA256 = (
    "54fcdb14676395a05d65e27bf6ed770446715aaeb0762e49bff39bccaff693ce"
)

DEFAULT_DISPATCH_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/"
    "cute_dsl/blackwell_sm12x/moe_dispatch.py"
)
DEFAULT_MICRO_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/"
    "cute_dsl/blackwell_sm12x/moe_micro_kernel.py"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _replace_once(source: str, anchor: str, replacement: str, label: str) -> str:
    count = source.count(anchor)
    if count != 1:
        raise RuntimeError(f"expected one {label} anchor, found {count}")
    return source.replace(anchor, replacement, 1)


_DISPATCH_ENV_ANCHOR = """\
_MICRO_SHARE_INPUT_ACROSS_EXPERTS = (
    os.environ.get("FLASHINFER_B12X_MICRO_SHARE_INPUT", "1") != "0"
)
"""
_DISPATCH_ENV_REPLACEMENT = _DISPATCH_ENV_ANCHOR + """\
_COOPERATIVE_FC2_M4_ENV = "FLASHINFER_B12X_COOPERATIVE_FC2_M4"
_cooperative_fc2_m4_raw = os.environ.get(_COOPERATIVE_FC2_M4_ENV, "0")
if _cooperative_fc2_m4_raw not in {"0", "1"}:
    raise RuntimeError(
        f"{_COOPERATIVE_FC2_M4_ENV} must be 0 or 1, "
        f"got {_cooperative_fc2_m4_raw!r}"
    )
_COOPERATIVE_FC2_M4 = _cooperative_fc2_m4_raw == "1"
"""

_DISPATCH_PLAN_ANCHOR = """\
    # Micro always selects tile from routed rows (not just for multi-topk)
    routed_rows = m * num_topk
    mma_tiler_mn = _select_moe_mma_tiler_mn(routed_rows, n)

    cache_key = (
"""
_DISPATCH_PLAN_REPLACEMENT = """\
    # Micro always selects tile from routed rows (not just for multi-topk)
    routed_rows = m * num_topk
    mma_tiler_mn = _select_moe_mma_tiler_mn(routed_rows, n)
    # This is deliberately an exact-shape compile-time canary.  Other token
    # counts, top-k values, tiles, model widths, or expert counts keep TMA.
    cooperative_fc2 = (
        _COOPERATIVE_FC2_M4
        and m == 4
        and num_topk == 6
        and k == 4096
        and n == 1024
        and state_E == 256
        and weight_E == 256
        and mma_tiler_mn == (64, 128)
        and is_gated_activation(activation)
    )

    cache_key = (
"""

_DISPATCH_CACHE_ANCHOR = """\
        share_expert_scales,
        single_token,
        activation,
"""
_DISPATCH_CACHE_REPLACEMENT = """\
        share_expert_scales,
        single_token,
        cooperative_fc2,
        activation,
"""

_DISPATCH_CONSTRUCTOR_ANCHOR = """\
        share_expert_scales=share_expert_scales,
        single_token=single_token,
    )
"""
_DISPATCH_CONSTRUCTOR_REPLACEMENT = """\
        share_expert_scales=share_expert_scales,
        single_token=single_token,
        cooperative_fc2=cooperative_fc2,
    )
"""


def patch_dispatch_source(source: str) -> str:
    source = _replace_once(
        source,
        _DISPATCH_ENV_ANCHOR,
        _DISPATCH_ENV_REPLACEMENT,
        "dispatcher opt-in environment",
    )
    source = _replace_once(
        source,
        _DISPATCH_PLAN_ANCHOR,
        _DISPATCH_PLAN_REPLACEMENT,
        "dispatcher exact M4 plan",
    )
    source = _replace_once(
        source,
        _DISPATCH_CACHE_ANCHOR,
        _DISPATCH_CACHE_REPLACEMENT,
        "dispatcher cache key",
    )
    return _replace_once(
        source,
        _DISPATCH_CONSTRUCTOR_ANCHOR,
        _DISPATCH_CONSTRUCTOR_REPLACEMENT,
        "dispatcher microkernel construction",
    )


_MICRO_INIT_SIGNATURE_ANCHOR = """\
        share_expert_scales: bool = False,
        single_token: bool = False,
    ):
"""
_MICRO_INIT_SIGNATURE_REPLACEMENT = """\
        share_expert_scales: bool = False,
        single_token: bool = False,
        cooperative_fc2: bool = False,
    ):
"""

_MICRO_INIT_STATE_ANCHOR = """\
        self.share_expert_scales = share_expert_scales
        self.single_token = single_token
        tile_k = sf_vec_size * 8
"""
_MICRO_INIT_STATE_REPLACEMENT = """\
        self.share_expert_scales = share_expert_scales
        self.single_token = single_token
        self.cooperative_fc2 = cooperative_fc2
        tile_k = sf_vec_size * 8
"""

_MICRO_INIT_GUARD_ANCHOR = """\
        self.threads_per_cta = (self.num_mma_warps + 1) * self.num_threads_per_warp
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")
"""
_MICRO_INIT_GUARD_REPLACEMENT = """\
        self.threads_per_cta = (self.num_mma_warps + 1) * self.num_threads_per_warp
        if self.cooperative_fc2:
            if self.single_token:
                raise ValueError("cooperative FC2 is M=4-only, never M=1")
            if self.tile_shape_mnk != (64, 128, 128):
                raise ValueError(
                    "cooperative FC2 requires the exact (64, 128, 128) tile"
                )
            if self.sf_vec_size != 16:
                raise ValueError("cooperative FC2 requires NVFP4 sf_vec_size=16")
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")
"""

_MICRO_HELPER_ANCHOR = """\
    @cute.jit
    def _resident_grid_barrier(
"""
_MICRO_HELPER_REPLACEMENT = """\
    @cute.jit
    def _make_cooperative_fc2_copy(
        self,
        dtype: cutlass.Constexpr,
        tile_cols: cutlass.Constexpr[int],
    ) -> cute.TiledCopy:
        # 128 MMA threads x four 16-byte operations cover one 8 KiB FP4 tile.
        copy_bits = 128
        copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            dtype,
            num_bits_per_copy=copy_bits,
        )
        elems_per_copy = copy_bits // dtype.width
        threads_k = tile_cols // elems_per_copy
        cooperative_threads = self.num_mma_warps * self.num_threads_per_warp
        assert cooperative_threads % threads_k == 0
        thread_layout = cute.make_ordered_layout(
            (cooperative_threads // threads_k, threads_k),
            order=(1, 0),
        )
        value_layout = cute.make_layout((1, elems_per_copy))
        return cute.make_tiled_copy_tv(copy_atom, thread_layout, value_layout)

    @cute.jit
    def _make_cooperative_fc2_scale_copy(
        self,
        dtype: cutlass.Constexpr,
    ) -> cute.TiledCopy:
        # Match DenseGemmKernel._make_scale_tiled_copy, but partition the scale
        # tile over all four MMA warps instead of the one producer warp.
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dtype,
            num_bits_per_copy=dtype.width,
        )
        return cute.make_tiled_copy_tv(
            copy_atom,
            cute.make_layout(
                (self.num_mma_warps * self.num_threads_per_warp,)
            ),
            cute.make_layout((1,)),
        )

    @cute.jit
    def _resident_grid_barrier(
"""

_MICRO_LAUNCH_ARGS_ANCHOR = """\
            tma_b_down,
            gB_down,
            tma_sfb_down,
            gSFB_down,
            self.tiled_mma,
"""
_MICRO_LAUNCH_ARGS_REPLACEMENT = """\
            tma_b_down,
            gB_down,
            b_down,
            tma_sfb_down,
            gSFB_down,
            sfb_down_tensor,
            self.tiled_mma,
"""

_MICRO_KERNEL_SIGNATURE_ANCHOR = """\
        tma_b_down: cute.CopyAtom,
        mB_down: cute.Tensor,
        tma_sfb_down: cute.CopyAtom,
        mSFB_down: cute.Tensor,
        tiled_mma: cute.TiledMma,
"""
_MICRO_KERNEL_SIGNATURE_REPLACEMENT = """\
        tma_b_down: cute.CopyAtom,
        mB_down: cute.Tensor,
        directB_down: cute.Tensor,
        tma_sfb_down: cute.CopyAtom,
        mSFB_down: cute.Tensor,
        directSFB_down: cute.Tensor,
        tiled_mma: cute.TiledMma,
"""

_MICRO_PREFETCH_ANCHOR = """\
            cpasync.prefetch_descriptor(tma_b_down)
            cpasync.prefetch_descriptor(tma_sfb_down)
"""
_MICRO_PREFETCH_REPLACEMENT = """\
            if cutlass.const_expr(not self.cooperative_fc2):
                cpasync.prefetch_descriptor(tma_b_down)
                cpasync.prefetch_descriptor(tma_sfb_down)
"""

_MICRO_MMA_COPY_SETUP_ANCHOR = """\
            csSFB_full = thr_ld_SFB.partition_S(sSFB)
            csSFB_up_full = thr_ld_SFB.partition_S(sSFB_up)
            crSFB_full = thr_ld_SFB.retile(tCrSFB_full)

            num_persistent_clusters = Int32(gdim_z)
"""
_MICRO_MMA_COPY_SETUP_REPLACEMENT = """\
            csSFB_full = thr_ld_SFB.partition_S(sSFB)
            csSFB_up_full = thr_ld_SFB.partition_S(sSFB_up)
            crSFB_full = thr_ld_SFB.retile(tCrSFB_full)

            if cutlass.const_expr(self.cooperative_fc2):
                cooperative_copy_B = self._make_cooperative_fc2_copy(
                    self.b_dtype, self.tile_shape_mnk[2]
                )
                cooperative_copy_SFB = self._make_cooperative_fc2_scale_copy(
                    self.sf_dtype
                )
                direct_gB_down = cute.local_tile(
                    directB_down,
                    cute.slice_(self.tile_shape_mnk, (0, None, None)),
                    (None, None, None),
                )
                direct_gSFB_down = cute.local_tile(
                    directSFB_down,
                    self.sfb_tile_shape_nk,
                    (None, None, None),
                )
                coord_B_down = cute.local_tile(
                    cute.make_identity_tensor(cute.shape(directB_down)),
                    cute.slice_(self.tile_shape_mnk, (0, None, None)),
                    (None, None, None),
                )
                coord_SFB_down = cute.local_tile(
                    cute.make_identity_tensor(cute.shape(directSFB_down)),
                    self.sfb_tile_shape_nk,
                    (None, None, None),
                )
                cooperative_thr_B = cooperative_copy_B.get_slice(tidx)
                cooperative_thr_SFB = cooperative_copy_SFB.get_slice(tidx)
                cooperative_gB = cooperative_thr_B.partition_S(direct_gB_down)
                cooperative_sB = cooperative_thr_B.partition_D(sB)
                cooperative_cB = cooperative_thr_B.partition_S(coord_B_down)
                cooperative_gSFB = cooperative_thr_SFB.partition_S(
                    direct_gSFB_down
                )
                cooperative_sSFB = cooperative_thr_SFB.partition_D(sSFB)
                cooperative_cSFB = cooperative_thr_SFB.partition_S(
                    coord_SFB_down
                )

            num_persistent_clusters = Int32(gdim_z)
"""

_MICRO_FC2_CONSUME_ANCHOR = """\
                    phase2_peek = phase2_pipeline.consumer_try_wait(phase2_cons_state)
                    phase2_pipeline.consumer_wait(phase2_cons_state, phase2_peek)
                    csB_phase2 = csB[None, None, None, phase2_cons_state.index]
                    csSFB_phase2 = csSFB_phase2_tile[
                        None, None, None, phase2_cons_state.index
                    ]

                    # Only load B-side (B_down changes per output tile; A is hoisted)
"""
_MICRO_FC2_CONSUME_REPLACEMENT = """\
                    if cutlass.const_expr(self.cooperative_fc2):
                        cooperative_gB_tile = cooperative_gB[
                            (
                                None,
                                None,
                                None,
                                output_tile_idx,
                                intermediate_slice,
                                weight_expert_idx,
                            )
                        ]
                        cooperative_sB_stage0 = cooperative_sB[
                            (None, None, None, 0)
                        ]
                        cooperative_cB_tile = cute.slice_(
                            cooperative_cB,
                            (
                                None,
                                None,
                                None,
                                output_tile_idx,
                                intermediate_slice,
                                weight_expert_idx,
                            ),
                        )
                        cooperative_gSFB_tile = cute.filter_zeros(
                            cooperative_gSFB[
                                (
                                    None,
                                    None,
                                    None,
                                    output_tile_idx
                                    // self.sfb_tiles_per_block,
                                    intermediate_slice,
                                    weight_expert_idx,
                                )
                            ]
                        )
                        cooperative_sSFB_stage0 = cute.filter_zeros(
                            cooperative_sSFB[(None, None, None, 0)]
                        )
                        cooperative_cSFB_tile = cute.filter_zeros(
                            cute.slice_(
                                cooperative_cSFB,
                                (
                                    None,
                                    None,
                                    None,
                                    output_tile_idx
                                    // self.sfb_tiles_per_block,
                                    intermediate_slice,
                                    weight_expert_idx,
                                ),
                            )
                        )
                        self._dense_cls._cpasync_copy_2d(
                            self,
                            cooperative_copy_B,
                            cooperative_gB_tile,
                            cooperative_sB_stage0,
                            cooperative_cB_tile,
                            Int32(directB_down.shape[0]),
                            False,
                        )
                        self._dense_cls._scale_copy_2d(
                            self,
                            cooperative_copy_SFB,
                            cooperative_gSFB_tile,
                            cooperative_sSFB_stage0,
                            cooperative_cSFB_tile,
                            # Match DenseGemmKernel: the scale tensor uses an
                            # atomized physical layout, so its shape[0] is not
                            # the logical N bound.  Predicate with B's N.
                            Int32(directB_down.shape[0]),
                        )
                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        csB_phase2 = csB[None, None, None, 0]
                        csSFB_phase2 = csSFB_phase2_tile[
                            None, None, None, 0
                        ]
                    else:
                        phase2_peek = phase2_pipeline.consumer_try_wait(
                            phase2_cons_state
                        )
                        phase2_pipeline.consumer_wait(
                            phase2_cons_state, phase2_peek
                        )
                        csB_phase2 = csB[
                            None, None, None, phase2_cons_state.index
                        ]
                        csSFB_phase2 = csSFB_phase2_tile[
                            None, None, None, phase2_cons_state.index
                        ]

                    # Only load B-side (B_down changes per output tile; A is hoisted)
"""

_MICRO_FC2_RELEASE_ANCHOR = """\
                        if k_block_idx == num_k_blocks - 1:
                            phase2_pipeline.consumer_release(phase2_cons_state)
                            phase2_cons_state.advance()
"""
_MICRO_FC2_RELEASE_REPLACEMENT = """\
                        if k_block_idx == num_k_blocks - 1:
                            if cutlass.const_expr(not self.cooperative_fc2):
                                phase2_pipeline.consumer_release(
                                    phase2_cons_state
                                )
                                phase2_cons_state.advance()
"""

_MICRO_DMA_FC2_ANCHOR = """\
                phase2_prod_state.reset_count()
                for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                    phase2_pipeline.producer_acquire(phase2_prod_state)
                    cute.copy(
                        tma_b_down,
                        tBgB_down[
                            (
                                None,
                                output_tile_idx,
                                intermediate_slice,
                                weight_expert_idx,
                            )
                        ],
                        tBsB_down[(None, phase2_prod_state.index)],
                        tma_bar_ptr=phase2_pipeline.producer_get_barrier(
                            phase2_prod_state
                        ),
                    )
                    cute.copy(
                        tma_sfb_down,
                        tBgSFB_down[
                            (
                                None,
                                output_tile_idx // self.sfb_tiles_per_block,
                                intermediate_slice,
                                weight_expert_idx,
                            )
                        ],
                        tBsSFB_down[(None, phase2_prod_state.index)],
                        tma_bar_ptr=phase2_pipeline.producer_get_barrier(
                            phase2_prod_state
                        ),
                    )
                    phase2_pipeline.producer_commit(phase2_prod_state)
                    phase2_prod_state.advance()
"""
_MICRO_DMA_FC2_REPLACEMENT = """\
                if cutlass.const_expr(not self.cooperative_fc2):
                    phase2_prod_state.reset_count()
                    for output_tile_idx in range(0, output_tile_cnt, 1, unroll=4):  # type: ignore[call-overload]
                        phase2_pipeline.producer_acquire(phase2_prod_state)
                        cute.copy(
                            tma_b_down,
                            tBgB_down[
                                (
                                    None,
                                    output_tile_idx,
                                    intermediate_slice,
                                    weight_expert_idx,
                                )
                            ],
                            tBsB_down[(None, phase2_prod_state.index)],
                            tma_bar_ptr=phase2_pipeline.producer_get_barrier(
                                phase2_prod_state
                            ),
                        )
                        cute.copy(
                            tma_sfb_down,
                            tBgSFB_down[
                                (
                                    None,
                                    output_tile_idx // self.sfb_tiles_per_block,
                                    intermediate_slice,
                                    weight_expert_idx,
                                )
                            ],
                            tBsSFB_down[(None, phase2_prod_state.index)],
                            tma_bar_ptr=phase2_pipeline.producer_get_barrier(
                                phase2_prod_state
                            ),
                        )
                        phase2_pipeline.producer_commit(phase2_prod_state)
                        phase2_prod_state.advance()
"""

_MICRO_DMA_TAIL_ANCHOR = """\
            phase2_pipeline.producer_tail(phase2_prod_state)
"""
_MICRO_DMA_TAIL_REPLACEMENT = """\
            if cutlass.const_expr(not self.cooperative_fc2):
                phase2_pipeline.producer_tail(phase2_prod_state)
"""


def patch_micro_source(source: str) -> str:
    replacements = (
        (
            _MICRO_INIT_SIGNATURE_ANCHOR,
            _MICRO_INIT_SIGNATURE_REPLACEMENT,
            "microkernel constructor signature",
        ),
        (
            _MICRO_INIT_STATE_ANCHOR,
            _MICRO_INIT_STATE_REPLACEMENT,
            "microkernel constructor state",
        ),
        (
            _MICRO_INIT_GUARD_ANCHOR,
            _MICRO_INIT_GUARD_REPLACEMENT,
            "microkernel exact-shape guard",
        ),
        (
            _MICRO_HELPER_ANCHOR,
            _MICRO_HELPER_REPLACEMENT,
            "cooperative copy helpers",
        ),
        (
            _MICRO_LAUNCH_ARGS_ANCHOR,
            _MICRO_LAUNCH_ARGS_REPLACEMENT,
            "direct FC2 launch tensors",
        ),
        (
            _MICRO_KERNEL_SIGNATURE_ANCHOR,
            _MICRO_KERNEL_SIGNATURE_REPLACEMENT,
            "direct FC2 kernel tensors",
        ),
        (
            _MICRO_PREFETCH_ANCHOR,
            _MICRO_PREFETCH_REPLACEMENT,
            "FC2 TMA prefetch guard",
        ),
        (
            _MICRO_MMA_COPY_SETUP_ANCHOR,
            _MICRO_MMA_COPY_SETUP_REPLACEMENT,
            "cooperative MMA copy setup",
        ),
        (
            _MICRO_FC2_CONSUME_ANCHOR,
            _MICRO_FC2_CONSUME_REPLACEMENT,
            "cooperative FC2 consume",
        ),
        (
            _MICRO_FC2_RELEASE_ANCHOR,
            _MICRO_FC2_RELEASE_REPLACEMENT,
            "FC2 consumer release guard",
        ),
        (
            _MICRO_DMA_FC2_ANCHOR,
            _MICRO_DMA_FC2_REPLACEMENT,
            "DMA FC2 producer guard",
        ),
        (
            _MICRO_DMA_TAIL_ANCHOR,
            _MICRO_DMA_TAIL_REPLACEMENT,
            "DMA FC2 tail guard",
        ),
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


def patch_paths(dispatch_target: Path, micro_target: Path) -> tuple[str, str]:
    """Validate and transform both files before changing either one."""
    dispatch_patched, dispatch_result = _prepare_file(
        dispatch_target,
        ACCEPTED_DISPATCH_SHA256,
        patch_dispatch_source,
        "accepted dispatcher",
    )
    micro_patched, micro_result = _prepare_file(
        micro_target,
        ACCEPTED_MICRO_SHA256,
        patch_micro_source,
        "accepted microkernel",
    )
    dispatch_target.write_bytes(dispatch_patched)
    micro_target.write_bytes(micro_patched)
    return dispatch_result, micro_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dispatch-target", type=Path, default=DEFAULT_DISPATCH_TARGET
    )
    parser.add_argument("--micro-target", type=Path, default=DEFAULT_MICRO_TARGET)
    args = parser.parse_args()

    dispatch_result, micro_result = patch_paths(
        args.dispatch_target, args.micro_target
    )
    print(f"dispatch_result_sha256={dispatch_result}")
    print(f"micro_result_sha256={micro_result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
