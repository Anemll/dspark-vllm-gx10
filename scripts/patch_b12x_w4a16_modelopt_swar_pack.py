#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Use a two-butterfly nibble transpose for ModelOpt MMA words.

Apply after :mod:`patch_b12x_w4a16_modelopt_vector_load`.  Four canonical
ModelOpt bytes arrive as ``L0,H0,L1,H1,L2,H2,L3,H3`` nibbles, while the MMA
word needs ``L0,L1,L2,L3,H0,H1,H2,H3``.  The pinned implementation inserts
each low/high nibble independently.  This opt-in decode-only patch performs
the same fixed permutation with two standard XOR butterfly swaps.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


PINNED_SOURCE_SHA256 = (
    "e23cccd7e135071f1393184132ec0ad7f277faf7851bfaba2b2a7b15e5a3a7dd"
)
PATCHED_SOURCE_SHA256 = (
    "96f97de061da420e29995fade53c1765b44831f6548bf3bc630004f51ca9694a"
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
_MODELOPT_SWAR_PACK_ENV = "B12X_W4A16_MODELOPT_SWAR_PACK"
_MODELOPT_FC1_TILE_ENV = "B12X_W4A16_MODELOPT_FC1_TILE"
""",
        "SWAR-pack environment name",
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


def _modelopt_swar_pack_enabled() -> bool:
    return os.environ.get(_MODELOPT_SWAR_PACK_ENV, "0") not in (
        "",
        "0",
        "false",
        "False",
    )


def _modelopt_tile_override(env_name: str) -> tuple[int, int, int] | None:
""",
        "SWAR-pack environment parser",
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
    modelopt_swar_pack: bool = False


@dataclass(frozen=True)
class _W4A16GemmLaunch:
""",
        "compile-result SWAR-pack proof field",
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
        self.modelopt_swar_pack = bool(
            self.modelopt_vector_load and _modelopt_swar_pack_enabled()
        )
        if self.modelopt_vector_load and self.source_n_rotation % self.tile_n != 0:
""",
        "decode-only SWAR-pack specialization",
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
            self.modelopt_swar_pack,
            self.w13_layout,
""",
        "SWAR-pack cache key",
    ),
    (
        """\
        word = Uint32(0)
        word = self._pack_modelopt_byte_pair(word, q0, 0, 16)
        word = self._pack_modelopt_byte_pair(word, q1, 4, 20)
        word = self._pack_modelopt_byte_pair(word, q2, 8, 24)
        word = self._pack_modelopt_byte_pair(word, q3, 12, 28)
        return word
""",
        """\
        if cutlass.const_expr(self.modelopt_swar_pack):
            word = (
                q0
                | (q1 << Uint32(8))
                | (q2 << Uint32(16))
                | (q3 << Uint32(24))
            )
            swap = ((word >> Uint32(4)) ^ word) & Uint32(0x00F000F0)
            word = word ^ swap ^ (swap << Uint32(4))
            swap = ((word >> Uint32(8)) ^ word) & Uint32(0x0000FF00)
            return word ^ swap ^ (swap << Uint32(8))
        word = Uint32(0)
        word = self._pack_modelopt_byte_pair(word, q0, 0, 16)
        word = self._pack_modelopt_byte_pair(word, q1, 4, 20)
        word = self._pack_modelopt_byte_pair(word, q2, 8, 24)
        word = self._pack_modelopt_byte_pair(word, q3, 12, 28)
        return word
""",
        "SWAR nibble-transpose selection",
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
        modelopt_swar_pack=bool(
            kernel.fc1.modelopt_swar_pack and kernel.fc2.modelopt_swar_pack
        ),
    )
""",
        "compile-result SWAR-pack proof value",
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
            "deterministic B12X W4A16 ModelOpt SWAR-pack result mismatch: "
            f"expected {PATCHED_SOURCE_SHA256}, got {patched_sha}"
        )
    args.target.write_bytes(patched)
    print(
        "patched B12X W4A16 ModelOpt SWAR nibble pack: "
        f"source={original_sha} result={patched_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
