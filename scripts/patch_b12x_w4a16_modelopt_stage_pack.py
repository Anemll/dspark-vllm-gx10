#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Benchmark a stage-once ModelOpt loader in B12X W4A16 TC decode.

The native ModelOpt weight tensor is row-major ``[E, N, K/2]`` bytes, while
the tensor-core MMA loop consumes B12X's 16x64 packed words.  The pinned
kernel's ModelOpt path copies row-major bytes into shared memory and rebuilds
each MMA word at consumption time with sixteen scalar shared loads per thread.

This benchmark-only patch adds an opt-in staging path that performs the same
permutation once while moving each K tile from global to shared memory.  A
four-lane subgroup cooperatively loads every source byte exactly once, uses
warp shuffles to assemble four packed words per lane, and writes the normal
packed shared-memory layout.  The MMA loop can then use its existing single
``ld.shared.v4.u32`` fast path.  The original ModelOpt tensors and E8M0 scales
remain untouched and no persistent packed weight copy is allocated.

Three independent FC tactic overrides (``a`` = K128/N128/256 threads,
``b`` = K64/N128/128 threads, ``c`` = K128/N64/128 threads) are also added so
the cheap A/B/C matrix can be measured before the new staging path is enabled.

Apply this after :mod:`patch_b12x_w4a16_modelopt_tc_decode`; both input and
output are content-addressed so upstream or patch-order drift fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


PINNED_SOURCE_SHA256 = (
    "c4eaa91d8a6f90b8ec6f6abf87c0f2ecb8d73dd4df6b8ae15fba18c0f1b623cd"
)
PATCHED_SOURCE_SHA256 = (
    "38a947fdb0a384c3f58600e42ed686256a90213e47f8a2a041ebe72c36e77f6e"
)
DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "b12x/moe/fused/w4a16/kernel.py"
)


_REPLACEMENTS = (
    (
        """\
    ld_global_acquire_i32,
    ld_global_v4_f32,
""",
        """\
    ld_global_acquire_i32,
    ld_global_nc_v2_u32,
    ld_global_v4_f32,
""",
        "vector global-load import",
    ),
    (
        """\
    st_shared_u32,
    st_shared_v4_f32,
""",
        """\
    st_shared_u32,
    st_shared_v4_f32,
    st_shared_v4_u32,
""",
        "vector shared-store import",
    ),
    (
        """\
_TC_DECODE_M = tuple(range(1, _TC_DECODE_MAX_M + 1))


def _tc_decode_enabled() -> bool:
""",
        """\
_TC_DECODE_M = tuple(range(1, _TC_DECODE_MAX_M + 1))
_MODELOPT_STAGE_PACK_ENV = "B12X_W4A16_MODELOPT_STAGE_PACK"
_MODELOPT_FC1_TILE_ENV = "B12X_W4A16_MODELOPT_FC1_TILE"
_MODELOPT_FC2_TILE_ENV = "B12X_W4A16_MODELOPT_FC2_TILE"
_MODELOPT_TILE_OVERRIDES = {
    "a": (128, 128, 256),
    "b": (64, 128, 128),
    "c": (128, 64, 128),
}


def _modelopt_stage_pack_enabled() -> bool:
    return os.environ.get(_MODELOPT_STAGE_PACK_ENV, "0") not in (
        "",
        "0",
        "false",
        "False",
    )


def _modelopt_tile_override(env_name: str) -> tuple[int, int, int] | None:
    value = os.environ.get(env_name, "").strip().lower()
    if not value:
        return None
    try:
        return _MODELOPT_TILE_OVERRIDES[value]
    except KeyError as exc:
        raise ValueError(
            f"{env_name} must be 'a' (K128/N128/256t), "
            "'b' (K64/N128/128t), 'c' (K128/N64/128t), or empty"
        ) from exc


def _tc_decode_enabled() -> bool:
""",
        "stage-pack and tactic environment contracts",
    ),
    (
        """\
        self.epilogue_relu2 = epilogue_activation == "relu2"
        self.weight_layout = weight_layout
        self.scale_format = scale_format
""",
        """\
        self.epilogue_relu2 = epilogue_activation == "relu2"
        self.weight_layout = weight_layout
        self.modelopt_stage_pack = bool(
            weight_layout == "modelopt" and _modelopt_stage_pack_enabled()
        )
        self.scale_format = scale_format
""",
        "kernel stage-pack specialization",
    ),
    (
        """\
            self.epilogue_relu2,
            self.weight_layout,
            self.scale_format,
""",
        """\
            self.epilogue_relu2,
            self.weight_layout,
            self.modelopt_stage_pack,
            self.scale_format,
""",
        "stage-pack cache-key specialization",
    ),
    (
        """\
        if cutlass.const_expr(self.weight_layout == "modelopt"):
            q0, q1, q2, q3 = self._load_b_registers_modelopt_shared(
""",
        """\
        if cutlass.const_expr(
            self.weight_layout == "modelopt" and not self.modelopt_stage_pack
        ):
            q0, q1, q2, q3 = self._load_b_registers_modelopt_shared(
""",
        "MMA packed shared-load selection",
    ),
    (
        """\
    @cute.jit
    def _stage_b_tile_modelopt_native(
""",
        """\
    @cute.jit
    def _stage_b_tile_modelopt_packed(
        self,
        b_u8_flat: cute.Tensor,
        smem_addr: Int32,
        expert_idx: Int32,
        output_n_tile: Int32,
        tile_idx: Int32,
        local_int4: Int32,
    ):
        # One warp emits one [K16, N64] packed tile.  A four-lane subgroup
        # shares the eight source rows needed by its four MMA words: lane d
        # loads rows d*16+tc_col and d*16+tc_col+8, then broadcasts the four
        # u32 halves to the other three lanes.  Across the subgroup every
        # source byte is loaded exactly once.
        lane = local_int4 & Int32(31)
        tile_in_stage = local_int4 // Int32(32)
        n_tiles = Int32(self.tile_n // 64)
        k_tile = tile_in_stage // n_tiles
        n_tile = tile_in_stage - k_tile * n_tiles
        tc_col = lane // Int32(4)
        source_quadrant = lane - tc_col * Int32(4)
        local_n0 = n_tile * Int32(64) + source_quadrant * Int32(16) + tc_col
        local_n1 = local_n0 + Int32(8)
        logical_n0 = output_n_tile * Int32(self.tile_n) + local_n0
        logical_n1 = output_n_tile * Int32(self.tile_n) + local_n1
        source_n0 = self._source_n_from_logical(logical_n0)
        source_n1 = self._source_n_from_logical(logical_n1)
        packed_cols = Int32(self.size_k // 2)
        expert_byte_off = Int64(expert_idx) * Int64(
            self.size_n * (self.size_k // 2)
        )
        tile_byte_off = Int64(tile_idx * Int32(self.tile_k // 2))
        local_k_byte = Int64(k_tile * Int32(8))
        row0_byte_off = (
            expert_byte_off
            + Int64(source_n0) * Int64(packed_cols)
            + tile_byte_off
            + local_k_byte
        )
        row1_byte_off = (
            expert_byte_off
            + Int64(source_n1) * Int64(packed_cols)
            + tile_byte_off
            + local_k_byte
        )
        row0_lo, row0_hi = ld_global_nc_v2_u32(
            get_ptr_as_int64(b_u8_flat, row0_byte_off)
        )
        row1_lo, row1_hi = ld_global_nc_v2_u32(
            get_ptr_as_int64(b_u8_flat, row1_byte_off)
        )

        packed = cute.make_rmem_tensor((4,), Uint32)
        subgroup_base = tc_col * Int32(4)
        byte_shift = Uint32((lane & Int32(3)) * Int32(8))
        for quadrant in cutlass.range_constexpr(4):
            source_lane = subgroup_base + Int32(quadrant)
            src0_lo = cute.arch.shuffle_sync(row0_lo, source_lane)
            src0_hi = cute.arch.shuffle_sync(row0_hi, source_lane)
            src1_lo = cute.arch.shuffle_sync(row1_lo, source_lane)
            src1_hi = cute.arch.shuffle_sync(row1_hi, source_lane)
            q0 = (src0_lo >> byte_shift) & Uint32(0xFF)
            q1 = (src0_hi >> byte_shift) & Uint32(0xFF)
            q2 = (src1_lo >> byte_shift) & Uint32(0xFF)
            q3 = (src1_hi >> byte_shift) & Uint32(0xFF)
            word = Uint32(0)
            word = self._pack_modelopt_byte_pair(word, q0, 0, 16)
            word = self._pack_modelopt_byte_pair(word, q1, 4, 20)
            word = self._pack_modelopt_byte_pair(word, q2, 8, 24)
            word = self._pack_modelopt_byte_pair(word, q3, 12, 28)
            packed[quadrant] = word
        st_shared_v4_u32(
            smem_addr,
            packed[0],
            packed[1],
            packed[2],
            packed[3],
        )

    @cute.jit
    def _stage_b_tile_modelopt_native(
""",
        "stage-once ModelOpt packed loader",
    ),
    (
        """\
            if cutlass.const_expr(self.weight_layout == "packed"):
                cp_async4_shared_global(
                    b_dst,
                    get_ptr_as_int64(b_i32_flat, b_src_int4 * Int32(4)),
                )
            else:
                self._stage_b_tile_modelopt_native(
""",
        """\
            if cutlass.const_expr(self.weight_layout == "packed"):
                cp_async4_shared_global(
                    b_dst,
                    get_ptr_as_int64(b_i32_flat, b_src_int4 * Int32(4)),
                )
            elif cutlass.const_expr(self.modelopt_stage_pack):
                self._stage_b_tile_modelopt_packed(
                    b_i32_flat,
                    b_dst,
                    expert_idx,
                    output_n_tile,
                    tile_idx,
                    Int32(i * self.cta_threads) + tid,
                )
            else:
                self._stage_b_tile_modelopt_native(
""",
        "stage-once ModelOpt dispatch",
    ),
    (
        """\
    if fc1_cta_threads != fc2_cta_threads:
        common_cta_threads = min(fc1_cta_threads, fc2_cta_threads)
""",
        """\
    if weight_layout == "modelopt":
        fc1_override = _modelopt_tile_override(_MODELOPT_FC1_TILE_ENV)
        fc2_override = _modelopt_tile_override(_MODELOPT_FC2_TILE_ENV)
        if fc1_override is not None:
            fc1_tile_k, fc1_tile_n, fc1_cta_threads = fc1_override
            if not _candidate_tile_fits(
                problem_n=fc1_cols,
                problem_k=hidden_size,
                cta_m_blocks=_covering_count(moe_block_size, 16),
                tile_n=fc1_tile_n,
                tile_k=fc1_tile_k,
                cta_threads=fc1_cta_threads,
                max_shared_mem=int(max_shared_mem) - 512,
                scale_format=scale_format,
            ):
                raise ValueError("forced ModelOpt FC1 tile does not fit")
        if fc2_override is not None:
            fc2_tile_k, fc2_tile_n, fc2_cta_threads = fc2_override
            if not _candidate_tile_fits(
                problem_n=hidden_size,
                problem_k=intermediate_size,
                cta_m_blocks=_covering_count(moe_block_size, 16),
                tile_n=fc2_tile_n,
                tile_k=fc2_tile_k,
                cta_threads=fc2_cta_threads,
                max_shared_mem=int(max_shared_mem) - 512,
                scale_format=scale_format,
            ):
                raise ValueError("forced ModelOpt FC2 tile does not fit")
    if fc1_cta_threads != fc2_cta_threads:
        common_cta_threads = min(fc1_cta_threads, fc2_cta_threads)
""",
        "benchmark-only ModelOpt tactic overrides",
    ),
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
            "pinned policy-patched B12X W4A16 kernel SHA-256 mismatch: "
            f"expected {PINNED_SOURCE_SHA256}, got {original_sha}"
        )
    patched = patch_source(original.decode("utf-8")).encode("utf-8")
    patched_sha = _sha256(patched)
    if patched_sha != PATCHED_SOURCE_SHA256:
        raise RuntimeError(
            "deterministic B12X W4A16 ModelOpt stage-pack result mismatch: "
            f"expected {PATCHED_SOURCE_SHA256}, got {patched_sha}"
        )
    args.target.write_bytes(patched)
    print(
        "patched B12X W4A16 ModelOpt stage-pack benchmark path: "
        f"source={original_sha} result={patched_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
