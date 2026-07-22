# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from __future__ import annotations

import hashlib
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import probe_nvfp4_modelopt_tc_stage_pack_sm121 as probe  # noqa: E402
from scripts import patch_b12x_w4a16_modelopt_stage_pack as patcher  # noqa: E402


def _byte_at(source: bytes, tile_k: int, n: int, k_byte: int) -> int:
    return source[n * (tile_k // 2) + k_byte]


def _pack_word(q0: int, q1: int, q2: int, q3: int) -> int:
    return (
        (q0 & 0xF)
        | ((q1 & 0xF) << 4)
        | ((q2 & 0xF) << 8)
        | ((q3 & 0xF) << 12)
        | (((q0 >> 4) & 0xF) << 16)
        | (((q1 >> 4) & 0xF) << 20)
        | (((q2 >> 4) & 0xF) << 24)
        | (((q3 >> 4) & 0xF) << 28)
    )


def _reference_packed(source: bytes, tile_n: int, tile_k: int) -> list[int]:
    """Independent transcription of B12X _repack_4bit_no_perm."""
    words: list[int] = []
    for k_tile in range(tile_k // 16):
        for n_tile in range(tile_n // 64):
            for pos in range(128):
                thread = pos // 4
                quadrant = pos % 4
                tc_col = thread // 4
                tc_row = (thread % 4) * 2
                n0 = n_tile * 64 + quadrant * 16 + tc_col
                n1 = n0 + 8
                k0 = k_tile * 8 + tc_row // 2
                words.append(
                    _pack_word(
                        _byte_at(source, tile_k, n0, k0),
                        _byte_at(source, tile_k, n0, k0 + 4),
                        _byte_at(source, tile_k, n1, k0),
                        _byte_at(source, tile_k, n1, k0 + 4),
                    )
                )
    return words


def _stage_algorithm(source: bytes, tile_n: int, tile_k: int) -> list[int]:
    """CPU oracle for the patched one-warp-per-K16/N64 staging algorithm."""
    output = [0] * (tile_n * tile_k // 8)
    vectors = len(output) // 4
    for vector_index in range(vectors):
        lane = vector_index % 32
        tile_in_stage = vector_index // 32
        n_tiles = tile_n // 64
        k_tile = tile_in_stage // n_tiles
        n_tile = tile_in_stage % n_tiles
        tc_col = lane // 4
        byte_index = lane % 4
        for quadrant in range(4):
            n0 = n_tile * 64 + quadrant * 16 + tc_col
            n1 = n0 + 8
            k0 = k_tile * 8 + byte_index
            output[vector_index * 4 + quadrant] = _pack_word(
                _byte_at(source, tile_k, n0, k0),
                _byte_at(source, tile_k, n0, k0 + 4),
                _byte_at(source, tile_k, n1, k0),
                _byte_at(source, tile_k, n1, k0 + 4),
            )
    return output


class B12xModelOptStagePackPatchTests(unittest.TestCase):
    @staticmethod
    def _fixture() -> str:
        return "\n".join(
            anchor for anchor, _replacement, _label in patcher._REPLACEMENTS
        )

    def test_patches_exact_stage_and_tactic_contracts(self) -> None:
        patched = patcher.patch_source(self._fixture())
        self.assertIn("ld_global_nc_v2_u32", patched)
        self.assertIn("st_shared_v4_u32", patched)
        self.assertIn("def _stage_b_tile_modelopt_packed(", patched)
        self.assertIn("elif cutlass.const_expr(self.modelopt_stage_pack):", patched)
        self.assertIn("and not self.modelopt_stage_pack", patched)
        self.assertIn('"a": (128, 128, 256)', patched)
        self.assertIn('"b": (64, 128, 128)', patched)
        self.assertIn('"c": (128, 64, 128)', patched)
        self.assertEqual(patched.count("self.modelopt_stage_pack,"), 1)

    def test_stage_pack_is_default_off_and_specialized_in_cache_key(self) -> None:
        patched = patcher.patch_source(self._fixture())
        self.assertIn(
            'os.environ.get(_MODELOPT_STAGE_PACK_ENV, "0")',
            patched,
        )
        self.assertIn(
            'weight_layout == "modelopt" and _modelopt_stage_pack_enabled()',
            patched,
        )
        self.assertIn("self.modelopt_stage_pack,", patched)
        # The old loader remains as the exact A/B control.
        self.assertIn("def _stage_b_tile_modelopt_native(", patched)
        self.assertIn("self._load_b_registers_modelopt_shared(", patched)

    def test_stage_algorithm_matches_packed_layout_for_a_b_and_c_tiles(self) -> None:
        for tile_n, tile_k in ((128, 128), (128, 64), (64, 128)):
            with self.subTest(tile_n=tile_n, tile_k=tile_k):
                size = tile_n * tile_k // 2
                source = bytes(((index * 73 + 19) & 0xFF) for index in range(size))
                expected = _reference_packed(source, tile_n, tile_k)
                actual = _stage_algorithm(source, tile_n, tile_k)
                self.assertEqual(actual, expected)
                self.assertEqual(len(actual) * 4, size)

    def test_every_source_nibble_is_preserved_exactly(self) -> None:
        tile_n, tile_k = 64, 128
        size = tile_n * tile_k // 2
        source = bytes((index & 0xFF) for index in range(size))
        packed = _stage_algorithm(source, tile_n, tile_k)
        # Equality with the independent reference is a full byte-level
        # permutation proof; pin a digest too so an accidental shared change
        # cannot make both helpers drift unnoticed.
        self.assertEqual(packed, _reference_packed(source, tile_n, tile_k))
        payload = b"".join(word.to_bytes(4, "little") for word in packed)
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "74918743762cdeea373409a3ccf5b18c3761beec559dc0ef1b708106113662bb",
        )

    def test_patch_is_fail_closed_and_deterministic(self) -> None:
        fixture = self._fixture()
        patched = patcher.patch_source(fixture)
        with self.assertRaisesRegex(RuntimeError, "vector global-load import"):
            patcher.patch_source(patched)

        source = fixture.encode()
        expected = patched.encode()
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / "kernel.py"
            target.write_bytes(source)
            with (
                mock.patch.object(
                    patcher, "PINNED_SOURCE_SHA256", hashlib.sha256(source).hexdigest()
                ),
                mock.patch.object(
                    patcher,
                    "PATCHED_SOURCE_SHA256",
                    hashlib.sha256(expected).hexdigest(),
                ),
                mock.patch.object(sys, "argv", ["patcher", "--target", str(target)]),
            ):
                self.assertEqual(patcher.main(), 0)
            self.assertEqual(target.read_bytes(), expected)

    def test_probe_pins_patched_kernel_and_requires_a_tactic_pair(self) -> None:
        self.assertEqual(probe.EXPECTED_KERNEL_SHA256, patcher.PATCHED_SOURCE_SHA256)
        self.assertRegex(probe.EXPECTED_KERNEL_SHA256, r"^[0-9a-f]{64}$")
        parser = probe.build_parser()
        args = parser.parse_args(
            [
                "--layer-file",
                "/model/model-layer-00000.safetensors",
                "--output",
                "/tmp/out.json",
                "--fc1-tile",
                "b",
                "--fc2-tile",
                "c",
                "--stage-pack",
            ]
        )
        self.assertEqual((args.fc1_tile, args.fc2_tile), ("b", "c"))
        self.assertTrue(args.stage_pack)

        args = parser.parse_args(
            [
                "--layer-file",
                "/model/model-layer-00000.safetensors",
                "--output",
                "/tmp/out-aa.json",
                "--fc1-tile",
                "a",
                "--fc2-tile",
                "a",
            ]
        )
        self.assertEqual((args.fc1_tile, args.fc2_tile), ("a", "a"))


if __name__ == "__main__":
    unittest.main()
