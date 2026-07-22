# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from __future__ import annotations

import hashlib
import pathlib
import random
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import probe_nvfp4_modelopt_tc_vector_load_sm121 as probe  # noqa: E402
from scripts import patch_b12x_w4a16_e8m0_k32_scale_reuse as reuse_patcher  # noqa: E402
from scripts import patch_b12x_w4a16_e8m0_scale_fast as finite_patcher  # noqa: E402
from scripts import patch_b12x_w4a16_modelopt_tc_decode as policy_patcher  # noqa: E402
from scripts import patch_b12x_w4a16_modelopt_vector_load as patcher  # noqa: E402


UPSTREAM_KERNEL = pathlib.Path(
    "/Users/anemll/SourceRelease/GITHUB/ML_playground/dspark-vllm-gx10/"
    ".build/b12x-upstream/b12x/moe/fused/w4a16/kernel.py"
)


class B12xModelOptVectorLoadPatchTests(unittest.TestCase):
    @staticmethod
    def _fixture() -> str:
        return "\n".join(anchor for anchor, _replacement, _label in patcher._REPLACEMENTS)

    @staticmethod
    def _scalar_pair(word0: int, word1: int, byte_position: int) -> tuple[int, int]:
        shift = byte_position * 8
        return (word0 >> shift) & 0xFF, (word1 >> shift) & 0xFF

    def test_vector_pair_is_bit_exact_to_two_scalar_loads(self) -> None:
        rng = random.Random(4104)
        for _ in range(10_000):
            word0 = rng.getrandbits(32)
            word1 = rng.getrandbits(32)
            for byte_position in range(4):
                expected = self._scalar_pair(word0, word1, byte_position)
                loaded_v2 = (word0, word1)
                actual = self._scalar_pair(*loaded_v2, byte_position)
                self.assertEqual(actual, expected)

    def test_vector_pair_address_is_always_eight_byte_aligned(self) -> None:
        # C/C uses tile_k=128.  Every source row begins on a 64-byte boundary,
        # each K16 sub-tile begins on an 8-byte boundary, and tc_row/2 is only
        # 0..3.  Rounding that byte selector down to its u32 word therefore
        # recovers the same 8-byte-aligned K16 base required by ld.shared.v2.
        tile_k = 128
        for local_n in range(64):
            for k_tile in range(tile_k // 16):
                for tc_row in (0, 2, 4, 6):
                    byte_offset = local_n * (tile_k // 2) + k_tile * 8 + tc_row // 2
                    word_byte_offset = byte_offset - byte_offset % 4
                    self.assertEqual(word_byte_offset % 8, 0)

    def test_tile_rotation_matches_per_row_rotation_for_c_tactic(self) -> None:
        size_n = 4096
        tile_n = 64
        rotation = 2048
        for logical_tile in range(size_n // tile_n):
            source_tile = (logical_tile + rotation // tile_n) % (size_n // tile_n)
            for local_n in range(tile_n):
                old = (logical_tile * tile_n + local_n + rotation) % size_n
                new = source_tile * tile_n + local_n
                self.assertEqual(new, old)

    def test_specialization_is_decode_only_and_opt_in(self) -> None:
        specialization = next(
            replacement
            for _anchor, replacement, label in patcher._REPLACEMENTS
            if label == "decode-only vector-load specialization"
        )
        self.assertIn("self.e8m0_finite_fast", specialization)
        self.assertIn("_modelopt_vector_load_enabled()", specialization)
        self.assertIn("source_n_rotation % self.tile_n", specialization)

    def test_vector_helper_halves_shared_load_instruction_count(self) -> None:
        helper = next(
            replacement
            for _anchor, replacement, label in patcher._REPLACEMENTS
            if label == "vector shared-byte-pair helper"
        )
        selection = next(
            replacement
            for _anchor, replacement, label in patcher._REPLACEMENTS
            if label == "vector shared-load selection"
        )
        self.assertEqual(helper.count("ld_shared_v2_u32("), 1)
        self.assertEqual(selection.count("_load_modelopt_shared_byte_pair("), 2)
        self.assertEqual(selection.count("_load_modelopt_shared_byte("), 4)
        self.assertNotIn("shuffle_sync", helper + selection)
        self.assertNotIn("sync_threads", helper + selection)

    def test_global_pipeline_and_mma_are_not_replaced(self) -> None:
        fixture = self._fixture()
        patched = patcher.patch_source(fixture)
        self.assertNotIn("cp_async4_shared_global(", "".join(
            replacement for _anchor, replacement, _label in patcher._REPLACEMENTS
        ))
        self.assertNotIn("_mma_", "".join(
            replacement for _anchor, replacement, _label in patcher._REPLACEMENTS
        ))
        self.assertIn("modelopt_output_n_tile", patched)

    def test_full_patch_chain_has_exact_content_hash(self) -> None:
        source = UPSTREAM_KERNEL.read_text()
        policy = policy_patcher.patch_source(source)
        finite = finite_patcher.patch_source(policy)
        reused = reuse_patcher.patch_source(finite)
        self.assertEqual(
            hashlib.sha256(reused.encode()).hexdigest(),
            patcher.PINNED_SOURCE_SHA256,
        )
        patched = patcher.patch_source(reused)
        self.assertEqual(
            hashlib.sha256(patched.encode()).hexdigest(),
            patcher.PATCHED_SOURCE_SHA256,
        )
        compile(patched, str(UPSTREAM_KERNEL), "exec", dont_inherit=True)

    def test_patch_is_fail_closed_and_deterministic(self) -> None:
        fixture = self._fixture()
        patched = patcher.patch_source(fixture)
        with self.assertRaisesRegex(RuntimeError, "environment name"):
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

    def test_probe_pins_vector_kernel_and_requires_explicit_flag(self) -> None:
        self.assertEqual(probe.EXPECTED_KERNEL_SHA256, patcher.PATCHED_SOURCE_SHA256)
        args = probe.build_parser().parse_args(
            [
                "--layer-file",
                "/model/model-layer-00000.safetensors",
                "--output",
                "/tmp/modelopt-vector.json",
                "--modelopt-vector-load",
            ]
        )
        self.assertTrue(args.modelopt_vector_load)
        self.assertEqual(probe.WINNING_TILE, "c")
        self.assertEqual(probe.WINNING_GEOMETRY, (128, 64))


if __name__ == "__main__":
    unittest.main()
