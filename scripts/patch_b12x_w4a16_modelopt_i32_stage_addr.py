#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Hoist ModelOpt global-stage bases and use proven-safe i32 offsets.

Apply after :mod:`patch_b12x_w4a16_modelopt_vector_load`.  The exact DeepSeek
V4 TP-rank tensors occupy at most 2 GiB, so every 16-byte global copy starts
at a signed-i32-safe byte offset.  The generic path nevertheless rebuilds an
expert/row/K base with Int64 multiplies for every copy.  This decode-only,
opt-in specialization proves the bound at compile time, computes the expert +
N-tile + K-tile base once per staged K tile, and leaves only local i32 row/K
arithmetic in each copy.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


PINNED_SOURCE_SHA256 = (
    "e23cccd7e135071f1393184132ec0ad7f277faf7851bfaba2b2a7b15e5a3a7dd"
)
PATCHED_SOURCE_SHA256 = (
    "cf818cf05a471bf8cd3d650fc31a848a6db9817b7f56f2b47083e1db8b3b707d"
)
DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "b12x/moe/fused/w4a16/kernel.py"
)


_REPLACEMENTS = (
    (
        """\
_MODELOPT_VECTOR_LOAD_ENV = "B12X_W4A16_MODELOPT_VECTOR_LOAD"
_MODELOPT_FC1_TILE_ENV = "B12X_W4A16_MODELOPT_FC1_TILE"
""",
        """\
_MODELOPT_VECTOR_LOAD_ENV = "B12X_W4A16_MODELOPT_VECTOR_LOAD"
_MODELOPT_I32_STAGE_ADDR_ENV = "B12X_W4A16_MODELOPT_I32_STAGE_ADDR"
_MODELOPT_FC1_TILE_ENV = "B12X_W4A16_MODELOPT_FC1_TILE"
""",
        "i32-stage-address environment name",
    ),
    (
        """\
def _modelopt_vector_load_enabled() -> bool:
    return os.environ.get(_MODELOPT_VECTOR_LOAD_ENV, "0") not in (
        "",
        "0",
        "false",
        "False",
    )


def _modelopt_tile_override(env_name: str) -> tuple[int, int, int] | None:
""",
        """\
def _modelopt_vector_load_enabled() -> bool:
    return os.environ.get(_MODELOPT_VECTOR_LOAD_ENV, "0") not in (
        "",
        "0",
        "false",
        "False",
    )


def _modelopt_i32_stage_addr_enabled() -> bool:
    return os.environ.get(_MODELOPT_I32_STAGE_ADDR_ENV, "0") not in (
        "",
        "0",
        "false",
        "False",
    )


def _modelopt_tile_override(env_name: str) -> tuple[int, int, int] | None:
""",
        "i32-stage-address environment parser",
    ),
    (
        """\
    e8m0_k32_scale_reuse: bool = False
    modelopt_vector_load: bool = False


@dataclass(frozen=True)
class _W4A16GemmLaunch:
""",
        """\
    e8m0_k32_scale_reuse: bool = False
    modelopt_vector_load: bool = False
    modelopt_i32_stage_addr: bool = False


@dataclass(frozen=True)
class _W4A16GemmLaunch:
""",
        "compile-result i32-stage-address proof field",
    ),
    (
        """\
        self.modelopt_vector_load = bool(
            self.e8m0_finite_fast and _modelopt_vector_load_enabled()
        )
        if self.modelopt_vector_load and self.source_n_rotation % self.tile_n != 0:
""",
        """\
        self.modelopt_vector_load = bool(
            self.e8m0_finite_fast and _modelopt_vector_load_enabled()
        )
        self.modelopt_i32_stage_addr = bool(
            self.modelopt_vector_load and _modelopt_i32_stage_addr_enabled()
        )
        if self.modelopt_i32_stage_addr:
            last_vector_start = (
                self.num_experts * self.size_n * (self.size_k // 2) - 16
            )
            if last_vector_start < 0 or last_vector_start > 0x7FFFFFFF:
                raise ValueError(
                    "ModelOpt i32 stage address exceeds signed-i32 range"
                )
        if self.modelopt_vector_load and self.source_n_rotation % self.tile_n != 0:
""",
        "decode-only i32-stage-address specialization",
    ),
    (
        """\
            self.e8m0_k32_scale_reuse,
            self.modelopt_vector_load,
            self.w13_layout,
""",
        """\
            self.e8m0_k32_scale_reuse,
            self.modelopt_vector_load,
            self.modelopt_i32_stage_addr,
            self.w13_layout,
""",
        "i32-stage-address cache key",
    ),
    (
        """\
        b_u8_flat: cute.Tensor,
        smem_addr: Int32,
        expert_idx: Int32,
""",
        """\
        b_u8_flat: cute.Tensor,
        smem_addr: Int32,
        modelopt_tile_byte_base: Int32,
        expert_idx: Int32,
""",
        "stage helper hoisted-base argument",
    ),
    (
        """\
        packed_cols = Int32(self.size_k // 2)
        byte_offset = (
            Int64(expert_idx) * Int64(self.size_n * (self.size_k // 2))
            + Int64(source_n) * Int64(packed_cols)
            + Int64(tile_idx * Int32(self.tile_k // 2))
            + Int64(local_k_vec * Int32(16))
        )
        cp_async4_shared_global(
""",
        """\
        packed_cols = Int32(self.size_k // 2)
        if cutlass.const_expr(self.modelopt_i32_stage_addr):
            byte_offset = Int64(
                modelopt_tile_byte_base
                + local_n * packed_cols
                + local_k_vec * Int32(16)
            )
        else:
            byte_offset = (
                Int64(expert_idx) * Int64(self.size_n * (self.size_k // 2))
                + Int64(source_n) * Int64(packed_cols)
                + Int64(tile_idx * Int32(self.tile_k // 2))
                + Int64(local_k_vec * Int32(16))
            )
        cp_async4_shared_global(
""",
        "per-copy i32 local address arithmetic",
    ),
    (
        """\
        if cutlass.const_expr(self.modelopt_vector_load):
            modelopt_output_n_tile = self._source_n_tile_from_logical_tile(
                output_n_tile
            )
        for i in cutlass.range_constexpr(self.a_sh_wr_iters):
""",
        """\
        if cutlass.const_expr(self.modelopt_vector_load):
            modelopt_output_n_tile = self._source_n_tile_from_logical_tile(
                output_n_tile
            )
        modelopt_tile_byte_base = Int32(0)
        if cutlass.const_expr(self.modelopt_i32_stage_addr):
            modelopt_tile_byte_base = (
                expert_idx * Int32(self.size_n * (self.size_k // 2))
                + modelopt_output_n_tile
                * Int32(self.tile_n * (self.size_k // 2))
                + tile_idx * Int32(self.tile_k // 2)
            )
        for i in cutlass.range_constexpr(self.a_sh_wr_iters):
""",
        "hoisted i32 expert/N/K tile base",
    ),
    (
        """\
                    b_i32_flat,
                    b_dst,
                    expert_idx,
                    modelopt_output_n_tile,
""",
        """\
                    b_i32_flat,
                    b_dst,
                    modelopt_tile_byte_base,
                    expert_idx,
                    modelopt_output_n_tile,
""",
        "forward hoisted ModelOpt tile base",
    ),
    (
        """\
        modelopt_vector_load=bool(
            kernel.fc1.modelopt_vector_load and kernel.fc2.modelopt_vector_load
        ),
    )
""",
        """\
        modelopt_vector_load=bool(
            kernel.fc1.modelopt_vector_load and kernel.fc2.modelopt_vector_load
        ),
        modelopt_i32_stage_addr=bool(
            kernel.fc1.modelopt_i32_stage_addr
            and kernel.fc2.modelopt_i32_stage_addr
        ),
    )
""",
        "compile-result i32-stage-address proof value",
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
            "pinned vector-load B12X W4A16 kernel SHA-256 mismatch: "
            f"expected {PINNED_SOURCE_SHA256}, got {original_sha}"
        )
    patched = patch_source(original.decode("utf-8")).encode("utf-8")
    patched_sha = _sha256(patched)
    if patched_sha != PATCHED_SOURCE_SHA256:
        raise RuntimeError(
            "deterministic B12X W4A16 ModelOpt i32-stage-address result mismatch: "
            f"expected {PATCHED_SOURCE_SHA256}, got {patched_sha}"
        )
    args.target.write_bytes(patched)
    print(
        "patched B12X W4A16 ModelOpt i32 stage addressing: "
        f"source={original_sha} result={patched_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
