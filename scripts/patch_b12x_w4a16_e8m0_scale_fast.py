#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Add an opt-in finite-E8M0 scale fast path to B12X TC decode.

The prepared DeepSeek V4 scale-collapse contract proves every E8M0 byte is a
finite exponent (119..123 for the audited checkpoint).  The generic B12X
converter nevertheless executes saturation and NaN handling for every scale
fragment.  This benchmark-only patch adds an exact six-instruction packed
finite converter.  Weight and scale loads, global staging, MMA order,
epilogue, and prefill remain unchanged.  The C/C tactic override exists only
to reproduce the accepted real-layer baseline geometry.

Apply this after ``patch_b12x_w4a16_modelopt_tc_decode``.  Both ends are
content-addressed so patch-order or upstream drift fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


PINNED_SOURCE_SHA256 = (
    "c4eaa91d8a6f90b8ec6f6abf87c0f2ecb8d73dd4df6b8ae15fba18c0f1b623cd"
)
PATCHED_SOURCE_SHA256 = (
    "aa99c6fb7af5653302e314c4525b45dbc9f8c3621c40acf372707c221ee7181f"
)
DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "b12x/moe/fused/w4a16/kernel.py"
)


_REPLACEMENTS = (
    (
        """\
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Int32, Int64, Uint32
""",
        """\
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Int32, Int64, T, Uint32, dsl_user_op
from cutlass._mlir.dialects import llvm
""",
        "finite-E8M0 PTX imports",
    ),
    (
        """\
_TC_DECODE_M = tuple(range(1, _TC_DECODE_MAX_M + 1))


def _tc_decode_enabled() -> bool:
""",
        """\
_TC_DECODE_M = tuple(range(1, _TC_DECODE_MAX_M + 1))
_E8M0_FINITE_FAST_ENV = "B12X_W4A16_E8M0_FINITE_FAST"
_MODELOPT_FC1_TILE_ENV = "B12X_W4A16_MODELOPT_FC1_TILE"
_MODELOPT_FC2_TILE_ENV = "B12X_W4A16_MODELOPT_FC2_TILE"
_MODELOPT_TILE_OVERRIDES = {
    "b": (64, 128, 128),
    "c": (128, 64, 128),
}


def _e8m0_finite_fast_enabled() -> bool:
    return os.environ.get(_E8M0_FINITE_FAST_ENV, "0") not in (
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
            f"{env_name} must be 'b' (K64/N128), 'c' (K128/N64), or empty"
        ) from exc


@dsl_user_op
def _packed_dequant_e8m0x4_to_bfloat2x2_finite(
    packed: Uint32, *, loc=None, ip=None
):
    # Valid only after a load-time proof that all four exponent bytes are
    # <=247.  prmt widens bytes 0/2 and 1/3 into independent 16-bit lanes;
    # +7 then <<7 is exactly the finite branch of the generic converter.
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        \"\"\"
        {
            .reg .u32 q0, q1;
            prmt.b32 q0, $2, 0, 0x4240;
            prmt.b32 q1, $2, 0, 0x4341;
            add.u32 q0, q0, 0x00070007;
            add.u32 q1, q1, 0x00070007;
            shl.b32 $0, q0, 7;
            shl.b32 $1, q1, 7;
        }
        \"\"\",
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


def _tc_decode_enabled() -> bool:
""",
        "finite-E8M0 environment and intrinsic",
    ),
    (
        """\
    scale_format: str = "e4m3_k16"
    tc_decode_fused_sum: bool = False


@dataclass(frozen=True)
class _W4A16GemmLaunch:
""",
        """\
    scale_format: str = "e4m3_k16"
    tc_decode_fused_sum: bool = False
    e8m0_finite_fast: bool = False


@dataclass(frozen=True)
class _W4A16GemmLaunch:
""",
        "compile-result finite-fast proof field",
    ),
    (
        """\
        self.fused_topk_sum = bool(fused_topk_sum)
        self.fused_sum_topk = int(fused_sum_topk)
        if self.fused_topk_sum and not self.direct_topk_routes:
""",
        """\
        self.fused_topk_sum = bool(fused_topk_sum)
        self.fused_sum_topk = int(fused_sum_topk)
        self.e8m0_finite_fast = bool(
            weight_layout == "modelopt"
            and self.scale_format_e8m0_k32
            and not self.is_fp16
            and self.direct_topk_routes
            and _e8m0_finite_fast_enabled()
        )
        if self.fused_topk_sum and not self.direct_topk_routes:
""",
        "decode-only finite-fast specialization",
    ),
    (
        """\
            self.weight_layout,
            self.scale_format,
            self.w13_layout,
""",
        """\
            self.weight_layout,
            self.scale_format,
            self.e8m0_finite_fast,
            self.w13_layout,
""",
        "finite-fast cache key",
    ),
    (
        """\
        if cutlass.const_expr(self.scale_format_e8m0_k32):
            if cutlass.const_expr(self.is_fp16):
                return packed_dequant_e8m0x4_to_half2x2(packed)
            return packed_dequant_e8m0x4_to_bfloat2x2(packed)
""",
        """\
        if cutlass.const_expr(self.scale_format_e8m0_k32):
            if cutlass.const_expr(self.is_fp16):
                return packed_dequant_e8m0x4_to_half2x2(packed)
            if cutlass.const_expr(self.e8m0_finite_fast):
                return _packed_dequant_e8m0x4_to_bfloat2x2_finite(packed)
            return packed_dequant_e8m0x4_to_bfloat2x2(packed)
""",
        "finite-fast dequant dispatch",
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
        "ModelOpt FC tactic overrides",
    ),
    (
        """\
        scale_format=scale_format,
        tc_decode_fused_sum=bool(tc_decode_fused_sum),
    )
""",
        """\
        scale_format=scale_format,
        tc_decode_fused_sum=bool(tc_decode_fused_sum),
        e8m0_finite_fast=bool(
            kernel.fc1.e8m0_finite_fast and kernel.fc2.e8m0_finite_fast
        ),
    )
""",
        "compile-result finite-fast proof value",
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
            "deterministic B12X W4A16 finite-E8M0 result mismatch: "
            f"expected {PATCHED_SOURCE_SHA256}, got {patched_sha}"
        )
    args.target.write_bytes(patched)
    print(
        "patched B12X W4A16 finite-E8M0 decode scale path: "
        f"source={original_sha} result={patched_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
