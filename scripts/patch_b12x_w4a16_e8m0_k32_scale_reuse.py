#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Reuse one finite E8M0 K/32 scale across its two K/16 MMA fragments.

Apply this benchmark-only patch after
``patch_b12x_w4a16_e8m0_scale_fast``.  The specialization is opt-in and can
activate only for the same ModelOpt BF16 direct-top-k fused-sum decode path.
It leaves the first K/16 fragment's shared scale load and conversion intact;
the second fragment loads only its distinct FP4 weight words and copies the
already converted four scale registers from the first fragment.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


PINNED_SOURCE_SHA256 = (
    "aa99c6fb7af5653302e314c4525b45dbc9f8c3621c40acf372707c221ee7181f"
)
PATCHED_SOURCE_SHA256 = (
    "7365bcd196a94180a523d9e8c5b3c3a2a49bc595fd5862ef2658d532661ba24d"
)
DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "b12x/moe/fused/w4a16/kernel.py"
)


_REPLACEMENTS = (
    (
        """\
_E8M0_FINITE_FAST_ENV = "B12X_W4A16_E8M0_FINITE_FAST"
_MODELOPT_FC1_TILE_ENV = "B12X_W4A16_MODELOPT_FC1_TILE"
""",
        """\
_E8M0_FINITE_FAST_ENV = "B12X_W4A16_E8M0_FINITE_FAST"
_E8M0_K32_SCALE_REUSE_ENV = "B12X_W4A16_E8M0_K32_SCALE_REUSE"
_MODELOPT_FC1_TILE_ENV = "B12X_W4A16_MODELOPT_FC1_TILE"
""",
        "K32 scale-reuse environment name",
    ),
    (
        """\
def _e8m0_finite_fast_enabled() -> bool:
    return os.environ.get(_E8M0_FINITE_FAST_ENV, "0") not in (
        "",
        "0",
        "false",
        "False",
    )


def _modelopt_tile_override(env_name: str) -> tuple[int, int, int] | None:
""",
        """\
def _e8m0_finite_fast_enabled() -> bool:
    return os.environ.get(_E8M0_FINITE_FAST_ENV, "0") not in (
        "",
        "0",
        "false",
        "False",
    )


def _e8m0_k32_scale_reuse_enabled() -> bool:
    return os.environ.get(_E8M0_K32_SCALE_REUSE_ENV, "0") not in (
        "",
        "0",
        "false",
        "False",
    )


def _modelopt_tile_override(env_name: str) -> tuple[int, int, int] | None:
""",
        "K32 scale-reuse environment parser",
    ),
    (
        """\
    tc_decode_fused_sum: bool = False
    e8m0_finite_fast: bool = False


@dataclass(frozen=True)
class _W4A16GemmLaunch:
""",
        """\
    tc_decode_fused_sum: bool = False
    e8m0_finite_fast: bool = False
    e8m0_k32_scale_reuse: bool = False


@dataclass(frozen=True)
class _W4A16GemmLaunch:
""",
        "compile-result K32 scale-reuse proof field",
    ),
    (
        """\
        self.e8m0_finite_fast = bool(
            weight_layout == "modelopt"
            and self.scale_format_e8m0_k32
            and not self.is_fp16
            and self.direct_topk_routes
            and _e8m0_finite_fast_enabled()
        )
        if self.fused_topk_sum and not self.direct_topk_routes:
""",
        """\
        self.e8m0_finite_fast = bool(
            weight_layout == "modelopt"
            and self.scale_format_e8m0_k32
            and not self.is_fp16
            and self.direct_topk_routes
            and _e8m0_finite_fast_enabled()
        )
        self.e8m0_k32_scale_reuse = bool(
            self.e8m0_finite_fast and _e8m0_k32_scale_reuse_enabled()
        )
        if self.fused_topk_sum and not self.direct_topk_routes:
""",
        "decode-only K32 scale-reuse specialization",
    ),
    (
        """\
            self.scale_format,
            self.e8m0_finite_fast,
            self.w13_layout,
""",
        """\
            self.scale_format,
            self.e8m0_finite_fast,
            self.e8m0_k32_scale_reuse,
            self.w13_layout,
""",
        "K32 scale-reuse cache key",
    ),
    (
        """\
    @cute.jit
    def _clear_b_scale_register_bundle(self, regs: cute.Tensor):
""",
        """\
    @cute.jit
    def _load_modelopt_b_register_bundle_reuse_scale(
        self,
        dst: cute.Tensor,
        current: cute.Tensor,
        smem_base: Int32,
        b_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
    ):
        q0, q1, q2, q3 = self._load_b_registers_modelopt_shared(
            smem_base,
            b_sh_rd,
            pipe,
            kk,
        )
        dst[0, 0] = q0
        dst[0, 1] = q1
        dst[0, 2] = q2
        dst[0, 3] = q3
        for col in cutlass.range_constexpr(4):
            dst[1, col] = current[1, col]

    @cute.jit
    def _clear_b_scale_register_bundle(self, regs: cute.Tensor):
""",
        "second-K16 weight-only bundle loader",
    ),
    (
        """\
    def _load_next_fragment_bundle(
        self,
        b_scale_next: cute.Tensor,
        a_regs_next: cute.Tensor,
""",
        """\
    def _load_next_fragment_bundle(
        self,
        b_scale_cur: cute.Tensor,
        b_scale_next: cute.Tensor,
        a_regs_next: cute.Tensor,
""",
        "next-fragment current-scale argument",
    ),
    (
        """\
            if tile_idx < k_tiles:
                self._load_b_scale_register_bundle(
                    b_scale_next,
                    smem_base,
                    tid,
                    b_sh_rd,
                    s_sh_rd,
                    Int32(pipe),
                    Int32(kk + 1),
                )
                self._load_a_register_bundle(
""",
        """\
            if tile_idx < k_tiles:
                if cutlass.const_expr(
                    self.e8m0_k32_scale_reuse and (kk % 2 == 0)
                ):
                    self._load_modelopt_b_register_bundle_reuse_scale(
                        b_scale_next,
                        b_scale_cur,
                        smem_base,
                        b_sh_rd,
                        Int32(pipe),
                        Int32(kk + 1),
                    )
                else:
                    self._load_b_scale_register_bundle(
                        b_scale_next,
                        smem_base,
                        tid,
                        b_sh_rd,
                        s_sh_rd,
                        Int32(pipe),
                        Int32(kk + 1),
                    )
                self._load_a_register_bundle(
""",
        "reuse K32 scale on the second K16 fragment",
    ),
    (
        """\
                        self._load_next_fragment_bundle(
                            b_scale_next,
                            a_regs_next,
""",
        """\
                        self._load_next_fragment_bundle(
                            b_scale_cur,
                            b_scale_next,
                            a_regs_next,
""",
        "pipeline current-scale forwarding",
    ),
    (
        """\
        e8m0_finite_fast=bool(
            kernel.fc1.e8m0_finite_fast and kernel.fc2.e8m0_finite_fast
        ),
    )
""",
        """\
        e8m0_finite_fast=bool(
            kernel.fc1.e8m0_finite_fast and kernel.fc2.e8m0_finite_fast
        ),
        e8m0_k32_scale_reuse=bool(
            kernel.fc1.e8m0_k32_scale_reuse
            and kernel.fc2.e8m0_k32_scale_reuse
        ),
    )
""",
        "compile-result K32 scale-reuse proof value",
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
            "pinned finite-E8M0 B12X W4A16 kernel SHA-256 mismatch: "
            f"expected {PINNED_SOURCE_SHA256}, got {original_sha}"
        )
    patched = patch_source(original.decode("utf-8")).encode("utf-8")
    patched_sha = _sha256(patched)
    if patched_sha != PATCHED_SOURCE_SHA256:
        raise RuntimeError(
            "deterministic B12X W4A16 K32 scale-reuse result mismatch: "
            f"expected {PATCHED_SOURCE_SHA256}, got {patched_sha}"
        )
    args.target.write_bytes(patched)
    print(
        "patched B12X W4A16 ModelOpt E8M0 K32 scale reuse: "
        f"source={original_sha} result={patched_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
