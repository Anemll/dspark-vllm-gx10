#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Add an opt-in cooperative ModelOpt shared loader to B12X TC decode.

The pinned ModelOpt tensor-core path stages canonical ``[N, K/2]`` bytes with
the existing cp.async pipeline, then reconstructs each B12X MMA word with four
scalar shared loads.  Within every four-lane subgroup those loads are identical
apart from the selected byte, so the native path performs four-way duplicate
shared reads.

This benchmark-only patch keeps global staging and every pipeline barrier
unchanged.  Each lane instead loads one of the four source quadrants, broadcasts
its four aligned u32 words to its subgroup, and constructs the same four packed
MMA words in registers.  It also exposes exact FC1/FC2 B-or-C tactic overrides
so the winning C/C geometry can be proven by the real-layer runner.  The loader
specialization is possible only for the opt-in ModelOpt direct-top-k fused-sum
TC-decode kernel, so generic small-M and prefill kernels remain on the existing
path.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


PINNED_SOURCE_SHA256 = (
    "c4eaa91d8a6f90b8ec6f6abf87c0f2ecb8d73dd4df6b8ae15fba18c0f1b623cd"
)
# Filled after applying the deterministic replacements to the exact pin.
PATCHED_SOURCE_SHA256 = (
    "6a7ae10d977a08340f12c69d174809283968f8492f09d95bc7059ba55a0955c6"
)
DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "b12x/moe/fused/w4a16/kernel.py"
)


_REPLACEMENTS = (
    (
        """\
_TC_DECODE_M = tuple(range(1, _TC_DECODE_MAX_M + 1))


def _tc_decode_enabled() -> bool:
""",
        """\
_TC_DECODE_M = tuple(range(1, _TC_DECODE_MAX_M + 1))
_MODELOPT_COOPERATIVE_LOAD_ENV = "B12X_W4A16_MODELOPT_COOPERATIVE_LOAD"
_MODELOPT_FC1_TILE_ENV = "B12X_W4A16_MODELOPT_FC1_TILE"
_MODELOPT_FC2_TILE_ENV = "B12X_W4A16_MODELOPT_FC2_TILE"
_MODELOPT_TILE_OVERRIDES = {
    "b": (64, 128, 128),
    "c": (128, 64, 128),
}


def _modelopt_cooperative_load_enabled() -> bool:
    return os.environ.get(_MODELOPT_COOPERATIVE_LOAD_ENV, "0") not in (
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


def _tc_decode_enabled() -> bool:
""",
        "cooperative loader environment contract",
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
    modelopt_cooperative_load: bool = False


@dataclass(frozen=True)
class _W4A16GemmLaunch:
""",
        "compile-result cooperative proof field",
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
        self.modelopt_cooperative_load = bool(
            weight_layout == "modelopt"
            and self.direct_topk_routes
            and _modelopt_cooperative_load_enabled()
        )
        if self.fused_topk_sum and not self.direct_topk_routes:
""",
        "decode-only cooperative specialization",
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
            self.modelopt_cooperative_load,
            self.scale_format,
""",
        "cooperative loader cache-key specialization",
    ),
    (
        """\
        if cutlass.const_expr(self.weight_layout == "modelopt"):
            q0, q1, q2, q3 = self._load_b_registers_modelopt_shared(
                smem_base,
                b_sh_rd,
                pipe,
                kk,
            )
        else:
""",
        """\
        if cutlass.const_expr(self.weight_layout == "modelopt"):
            if cutlass.const_expr(self.modelopt_cooperative_load):
                q0, q1, q2, q3 = self._load_b_registers_modelopt_cooperative(
                    smem_base,
                    b_sh_rd,
                    pipe,
                    kk,
                )
            else:
                q0, q1, q2, q3 = self._load_b_registers_modelopt_shared(
                    smem_base,
                    b_sh_rd,
                    pipe,
                    kk,
                )
        else:
""",
        "cooperative ModelOpt load selection",
    ),
    (
        """\
    @cute.jit
    def _load_b_registers_modelopt_shared(
""",
        """\
    @cute.jit
    def _load_modelopt_shared_row_pair(
        self,
        smem_base: Int32,
        pipe: Int32,
        n_tile: Int32,
        k_tile: Int32,
        source_quadrant: Int32,
        tc_col: Int32,
        n_delta: cutlass.Constexpr[int],
    ):
        local_n = (
            n_tile * Int32(64)
            + source_quadrant * Int32(16)
            + tc_col
            + Int32(n_delta)
        )
        byte_offset = (
            local_n * Int32(self.tile_k // 2)
            + k_tile * Int32(8)
        )
        return ld_shared_v2_u32(
            smem_base
            + Int32(self.sh_b_off * 16)
            + pipe * Int32(self.b_sh_stage * 16)
            + byte_offset
        )

    @cute.jit
    def _load_b_registers_modelopt_cooperative(
        self,
        smem_base: Int32,
        b_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
    ):
        # One warp consumes one [K16, N64] microtile per fragment bundle.
        # Four neighboring lanes share tc_col.  Lane d owns source quadrant d,
        # loads its two N rows (four aligned u32 total), then broadcasts those
        # registers so every lane can select its distinct K byte.
        packed_word_index = (Int32(self.b_sh_stride) * kk + b_sh_rd) * Int32(4)
        words_per_k_tile = Int32((self.tile_n // 64) * 128)
        k_tile = packed_word_index // words_per_k_tile
        pos_in_k_tile = packed_word_index - k_tile * words_per_k_tile
        n_tile = pos_in_k_tile // Int32(128)
        pos = pos_in_k_tile - n_tile * Int32(128)
        th_id = pos // Int32(4)
        tc_col = th_id // Int32(4)
        source_quadrant = th_id - tc_col * Int32(4)

        row0_lo, row0_hi = self._load_modelopt_shared_row_pair(
            smem_base,
            pipe,
            n_tile,
            k_tile,
            source_quadrant,
            tc_col,
            0,
        )
        row1_lo, row1_hi = self._load_modelopt_shared_row_pair(
            smem_base,
            pipe,
            n_tile,
            k_tile,
            source_quadrant,
            tc_col,
            8,
        )

        packed = cute.make_rmem_tensor((4,), Uint32)
        subgroup_base = tc_col * Int32(4)
        byte_shift = Uint32(source_quadrant * Int32(8))
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
        return packed[0], packed[1], packed[2], packed[3]

    @cute.jit
    def _load_b_registers_modelopt_shared(
""",
        "cooperative shared loader implementation",
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
        modelopt_cooperative_load=bool(
            kernel.fc1.modelopt_cooperative_load
            and kernel.fc2.modelopt_cooperative_load
        ),
    )
""",
        "compile-result cooperative proof value",
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
            "deterministic B12X W4A16 cooperative-load result mismatch: "
            f"expected {PATCHED_SOURCE_SHA256}, got {patched_sha}"
        )
    args.target.write_bytes(patched)
    print(
        "patched B12X W4A16 ModelOpt cooperative benchmark loader: "
        f"source={original_sha} result={patched_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
