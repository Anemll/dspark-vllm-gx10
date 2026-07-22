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

from scripts import patch_b12x_w4a16_modelopt_cooperative_load as patcher  # noqa: E402
from benchmarks import probe_nvfp4_modelopt_tc_cooperative_load_sm121 as probe  # noqa: E402


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
    """Independent transcription of prepare.py::_repack_4bit_no_perm."""
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


def _cooperative_packed(source: bytes, tile_n: int, tile_k: int) -> list[int]:
    """CPU oracle for the four-lane cooperative register loader."""
    output: list[int] = []
    for k_tile in range(tile_k // 16):
        for n_tile in range(tile_n // 64):
            # Each four-lane subgroup owns one tc_col.  Lane d loads the four
            # aligned words for source quadrant d; every lane then selects its
            # own byte after the conceptual subgroup broadcast.
            for group in range(8):
                owned: list[tuple[int, int, int, int]] = []
                for source_quadrant in range(4):
                    n0 = n_tile * 64 + source_quadrant * 16 + group
                    n1 = n0 + 8
                    k0 = k_tile * 8
                    owned.append(
                        (
                            int.from_bytes(
                                source[n0 * (tile_k // 2) + k0 : n0 * (tile_k // 2) + k0 + 4],
                                "little",
                            ),
                            int.from_bytes(
                                source[n0 * (tile_k // 2) + k0 + 4 : n0 * (tile_k // 2) + k0 + 8],
                                "little",
                            ),
                            int.from_bytes(
                                source[n1 * (tile_k // 2) + k0 : n1 * (tile_k // 2) + k0 + 4],
                                "little",
                            ),
                            int.from_bytes(
                                source[n1 * (tile_k // 2) + k0 + 4 : n1 * (tile_k // 2) + k0 + 8],
                                "little",
                            ),
                        )
                    )
                for lane_slot in range(4):
                    shift = lane_slot * 8
                    for words in owned:
                        q0, q1, q2, q3 = ((word >> shift) & 0xFF for word in words)
                        output.append(_pack_word(q0, q1, q2, q3))
    return output


class B12xModelOptCooperativeLoadPatchTests(unittest.TestCase):
    @staticmethod
    def _fixture() -> str:
        return "\n".join(anchor for anchor, _replacement, _label in patcher._REPLACEMENTS)

    def test_patch_adds_only_opt_in_decode_loader(self) -> None:
        patched = patcher.patch_source(self._fixture())
        self.assertIn("def _load_b_registers_modelopt_cooperative(", patched)
        self.assertIn("def _load_modelopt_shared_row_pair(", patched)
        self.assertIn("cute.arch.shuffle_sync(row0_lo, source_lane)", patched)
        self.assertIn('os.environ.get(_MODELOPT_COOPERATIVE_LOAD_ENV, "0")', patched)
        self.assertIn('weight_layout == "modelopt"', patched)
        self.assertIn("and self.direct_topk_routes", patched)
        specialization = next(
            replacement
            for _anchor, replacement, label in patcher._REPLACEMENTS
            if label == "decode-only cooperative specialization"
        )
        self.assertNotIn("and self.fused_topk_sum", specialization)
        self.assertEqual(patched.count("self.modelopt_cooperative_load,"), 1)
        self.assertIn('"c": (128, 64, 128)', patched)
        self.assertIn("modelopt_cooperative_load=bool(", patched)
        self.assertIn("kernel.fc1.modelopt_cooperative_load", patched)
        self.assertIn("kernel.fc2.modelopt_cooperative_load", patched)

    def test_patch_preserves_async_stage_and_prefetch_pipeline(self) -> None:
        fixture = self._fixture()
        patched = patcher.patch_source(fixture)
        # Cooperative reconstruction happens only at the register load.  It
        # introduces no shared store, async-stage replacement, or barrier.
        inserted = patched[len(fixture) :]
        del inserted  # The semantic assertions below are stable to fixture order.
        implementation = next(
            replacement
            for _anchor, replacement, label in patcher._REPLACEMENTS
            if label == "cooperative shared loader implementation"
        )
        self.assertNotIn("st_shared", implementation)
        self.assertNotIn("cp_async", implementation)
        self.assertNotIn("sync_threads", implementation)
        self.assertIn("def _load_b_registers_modelopt_shared(", patched)

    def test_cooperative_algorithm_matches_packed_layout_for_b_and_c_tiles(self) -> None:
        for tile_n, tile_k in ((128, 64), (64, 128)):
            with self.subTest(tile_n=tile_n, tile_k=tile_k):
                size = tile_n * tile_k // 2
                source = bytes(((index * 73 + 19) & 0xFF) for index in range(size))
                expected = _reference_packed(source, tile_n, tile_k)
                actual = _cooperative_packed(source, tile_n, tile_k)
                self.assertEqual(actual, expected)
                self.assertEqual(len(actual) * 4, size)

    def test_every_source_nibble_is_preserved_exactly(self) -> None:
        tile_n, tile_k = 128, 64
        source = bytes((index & 0xFF) for index in range(tile_n * tile_k // 2))
        packed = _cooperative_packed(source, tile_n, tile_k)
        self.assertEqual(packed, _reference_packed(source, tile_n, tile_k))
        payload = b"".join(word.to_bytes(4, "little") for word in packed)
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "a3d9684e9b004665c42f6714aab8bddbd777bce0667724753afdb1d1640bc5a7",
        )

    def test_patch_is_fail_closed_and_deterministic(self) -> None:
        fixture = self._fixture()
        patched = patcher.patch_source(fixture)
        with self.assertRaisesRegex(RuntimeError, "environment contract"):
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

    def test_probe_pins_patch_and_exposes_explicit_opt_in(self) -> None:
        self.assertEqual(probe.EXPECTED_KERNEL_SHA256, patcher.PATCHED_SOURCE_SHA256)
        parser = probe.build_parser()
        args = parser.parse_args(
            [
                "--layer-file",
                "/model/model-layer-00000.safetensors",
                "--output",
                "/tmp/cooperative.json",
                "--cooperative-load",
            ]
        )
        self.assertTrue(args.cooperative_load)
        self.assertEqual(probe.WINNING_TILE, "c")
        self.assertEqual(probe.WINNING_GEOMETRY, (128, 64))


if __name__ == "__main__":
    unittest.main()
