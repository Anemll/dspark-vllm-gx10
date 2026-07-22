#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Halve ModelOpt shared-load instructions in B12X TC decode.

Apply this benchmark-only patch after
``patch_b12x_w4a16_e8m0_k32_scale_reuse``.  The canonical ModelOpt shared
layout stores the two K/8 byte groups needed by one packed MMA word in two
adjacent aligned u32 words.  The pinned loader fetches those words with four
independent ``ld.shared.u32`` instructions.  This opt-in specialization uses
two ``ld.shared.v2.u32`` instructions instead, preserving the exact byte and
nibble permutation while halving the shared-load instruction count.

For FC1 the ModelOpt W13 half-rotation is tile aligned.  The same
specialization computes that rotation once per staged K tile rather than once
per 16-byte copy.  Global cp.async staging, scales, MMA order, epilogue, and
prefill are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


PINNED_SOURCE_SHA256 = (
    "7365bcd196a94180a523d9e8c5b3c3a2a49bc595fd5862ef2658d532661ba24d"
)
PATCHED_SOURCE_SHA256 = (
    "e23cccd7e135071f1393184132ec0ad7f277faf7851bfaba2b2a7b15e5a3a7dd"
)
DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "b12x/moe/fused/w4a16/kernel.py"
)


_REPLACEMENTS = (
    (
        """\
_E8M0_K32_SCALE_REUSE_ENV = "B12X_W4A16_E8M0_K32_SCALE_REUSE"
_MODELOPT_FC1_TILE_ENV = "B12X_W4A16_MODELOPT_FC1_TILE"
""",
        """\
_E8M0_K32_SCALE_REUSE_ENV = "B12X_W4A16_E8M0_K32_SCALE_REUSE"
_MODELOPT_VECTOR_LOAD_ENV = "B12X_W4A16_MODELOPT_VECTOR_LOAD"
_MODELOPT_FC1_TILE_ENV = "B12X_W4A16_MODELOPT_FC1_TILE"
""",
        "vector-load environment name",
    ),
    (
        """\
def _e8m0_k32_scale_reuse_enabled() -> bool:
    return os.environ.get(_E8M0_K32_SCALE_REUSE_ENV, "0") not in (
        "",
        "0",
        "false",
        "False",
    )


def _modelopt_tile_override(env_name: str) -> tuple[int, int, int] | None:
""",
        """\
def _e8m0_k32_scale_reuse_enabled() -> bool:
    return os.environ.get(_E8M0_K32_SCALE_REUSE_ENV, "0") not in (
        "",
        "0",
        "false",
        "False",
    )


def _modelopt_vector_load_enabled() -> bool:
    return os.environ.get(_MODELOPT_VECTOR_LOAD_ENV, "0") not in (
        "",
        "0",
        "false",
        "False",
    )


def _modelopt_tile_override(env_name: str) -> tuple[int, int, int] | None:
""",
        "vector-load environment parser",
    ),
    (
        """\
    e8m0_finite_fast: bool = False
    e8m0_k32_scale_reuse: bool = False


@dataclass(frozen=True)
class _W4A16GemmLaunch:
""",
        """\
    e8m0_finite_fast: bool = False
    e8m0_k32_scale_reuse: bool = False
    modelopt_vector_load: bool = False


@dataclass(frozen=True)
class _W4A16GemmLaunch:
""",
        "compile-result vector-load proof field",
    ),
    (
        """\
        self.e8m0_k32_scale_reuse = bool(
            self.e8m0_finite_fast and _e8m0_k32_scale_reuse_enabled()
        )
        if self.fused_topk_sum and not self.direct_topk_routes:
""",
        """\
        self.e8m0_k32_scale_reuse = bool(
            self.e8m0_finite_fast and _e8m0_k32_scale_reuse_enabled()
        )
        self.modelopt_vector_load = bool(
            self.e8m0_finite_fast and _modelopt_vector_load_enabled()
        )
        if self.modelopt_vector_load and self.source_n_rotation % self.tile_n != 0:
            raise ValueError(
                "ModelOpt vector-load source_n_rotation must be tile aligned"
            )
        if self.fused_topk_sum and not self.direct_topk_routes:
""",
        "decode-only vector-load specialization",
    ),
    (
        """\
            self.e8m0_finite_fast,
            self.e8m0_k32_scale_reuse,
            self.w13_layout,
""",
        """\
            self.e8m0_finite_fast,
            self.e8m0_k32_scale_reuse,
            self.modelopt_vector_load,
            self.w13_layout,
""",
        "vector-load cache key",
    ),
    (
        """\
    @cute.jit
    def _stage_b_tile_modelopt_native(
""",
        """\
    @cute.jit
    def _source_n_tile_from_logical_tile(self, logical_n_tile: Int32) -> Int32:
        source_n_tile = logical_n_tile
        if cutlass.const_expr(self.source_n_rotation != 0):
            source_n_tile += Int32(self.source_n_rotation // self.tile_n)
            if source_n_tile >= Int32(self.size_n // self.tile_n):
                source_n_tile -= Int32(self.size_n // self.tile_n)
        return source_n_tile

    @cute.jit
    def _stage_b_tile_modelopt_native(
""",
        "tile-level source-N rotation helper",
    ),
    (
        """\
        logical_n = output_n_tile * Int32(self.tile_n) + local_n
        source_n = self._source_n_from_logical(logical_n)
        packed_cols = Int32(self.size_k // 2)
""",
        """\
        if cutlass.const_expr(self.modelopt_vector_load):
            source_n = output_n_tile * Int32(self.tile_n) + local_n
        else:
            logical_n = output_n_tile * Int32(self.tile_n) + local_n
            source_n = self._source_n_from_logical(logical_n)
        packed_cols = Int32(self.size_k // 2)
""",
        "consume pre-rotated source-N tile",
    ),
    (
        """\
    @cute.jit
    def _load_modelopt_shared_packed_word_for_lane(
""",
        """\
    @cute.jit
    def _load_modelopt_shared_byte_pair(
        self,
        smem_base: Int32,
        pipe: Int32,
        n_tile: Int32,
        k_tile: Int32,
        warp_id: Int32,
        tc_col: Int32,
        tc_row: Int32,
        n_delta: cutlass.Constexpr[int],
    ):
        local_n = (
            n_tile * Int32(64)
            + warp_id * Int32(16)
            + tc_col
            + Int32(n_delta)
        )
        local_k = k_tile * Int32(16) + tc_row
        byte_offset = local_n * Int32(self.tile_k // 2) + local_k // Int32(2)
        word_byte_offset = byte_offset - (byte_offset & Int32(3))
        word0, word1 = ld_shared_v2_u32(
            smem_base
            + Int32(self.sh_b_off * 16)
            + pipe * Int32(self.b_sh_stage * 16)
            + word_byte_offset
        )
        shift = Uint32((byte_offset - word_byte_offset) * Int32(8))
        return (
            (word0 >> shift) & Uint32(0xFF),
            (word1 >> shift) & Uint32(0xFF),
        )

    @cute.jit
    def _load_modelopt_shared_packed_word_for_lane(
""",
        "vector shared-byte-pair helper",
    ),
    (
        """\
        q0 = self._load_modelopt_shared_byte(
            smem_base, pipe, n_tile, k_tile, warp_id, tc_col, tc_row, 0, 0
        )
        q1 = self._load_modelopt_shared_byte(
            smem_base, pipe, n_tile, k_tile, warp_id, tc_col, tc_row, 0, 8
        )
        q2 = self._load_modelopt_shared_byte(
            smem_base, pipe, n_tile, k_tile, warp_id, tc_col, tc_row, 8, 0
        )
        q3 = self._load_modelopt_shared_byte(
            smem_base, pipe, n_tile, k_tile, warp_id, tc_col, tc_row, 8, 8
        )
        word = Uint32(0)
""",
        """\
        if cutlass.const_expr(self.modelopt_vector_load):
            q0, q1 = self._load_modelopt_shared_byte_pair(
                smem_base, pipe, n_tile, k_tile, warp_id, tc_col, tc_row, 0
            )
            q2, q3 = self._load_modelopt_shared_byte_pair(
                smem_base, pipe, n_tile, k_tile, warp_id, tc_col, tc_row, 8
            )
        else:
            q0 = self._load_modelopt_shared_byte(
                smem_base, pipe, n_tile, k_tile, warp_id, tc_col, tc_row, 0, 0
            )
            q1 = self._load_modelopt_shared_byte(
                smem_base, pipe, n_tile, k_tile, warp_id, tc_col, tc_row, 0, 8
            )
            q2 = self._load_modelopt_shared_byte(
                smem_base, pipe, n_tile, k_tile, warp_id, tc_col, tc_row, 8, 0
            )
            q3 = self._load_modelopt_shared_byte(
                smem_base, pipe, n_tile, k_tile, warp_id, tc_col, tc_row, 8, 8
            )
        word = Uint32(0)
""",
        "vector shared-load selection",
    ),
    (
        """\
    ):
        for i in cutlass.range_constexpr(self.a_sh_wr_iters):
""",
        """\
    ):
        modelopt_output_n_tile = output_n_tile
        if cutlass.const_expr(self.modelopt_vector_load):
            modelopt_output_n_tile = self._source_n_tile_from_logical_tile(
                output_n_tile
            )
        for i in cutlass.range_constexpr(self.a_sh_wr_iters):
""",
        "hoist ModelOpt source-N tile rotation",
    ),
    (
        """\
                    expert_idx,
                    output_n_tile,
                    tile_idx,
                    Int32(i * self.cta_threads) + tid,
""",
        """\
                    expert_idx,
                    modelopt_output_n_tile,
                    tile_idx,
                    Int32(i * self.cta_threads) + tid,
""",
        "forward pre-rotated source-N tile",
    ),
    (
        """\
        e8m0_k32_scale_reuse=bool(
            kernel.fc1.e8m0_k32_scale_reuse
            and kernel.fc2.e8m0_k32_scale_reuse
        ),
    )
""",
        """\
        e8m0_k32_scale_reuse=bool(
            kernel.fc1.e8m0_k32_scale_reuse
            and kernel.fc2.e8m0_k32_scale_reuse
        ),
        modelopt_vector_load=bool(
            kernel.fc1.modelopt_vector_load and kernel.fc2.modelopt_vector_load
        ),
    )
""",
        "compile-result vector-load proof value",
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
            "pinned K32-reuse B12X W4A16 kernel SHA-256 mismatch: "
            f"expected {PINNED_SOURCE_SHA256}, got {original_sha}"
        )
    patched = patch_source(original.decode("utf-8")).encode("utf-8")
    patched_sha = _sha256(patched)
    if patched_sha != PATCHED_SOURCE_SHA256:
        raise RuntimeError(
            "deterministic B12X W4A16 ModelOpt vector-load result mismatch: "
            f"expected {PATCHED_SOURCE_SHA256}, got {patched_sha}"
        )
    args.target.write_bytes(patched)
    print(
        "patched B12X W4A16 ModelOpt vector shared-load path: "
        f"source={original_sha} result={patched_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
