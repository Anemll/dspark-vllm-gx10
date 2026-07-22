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

from benchmarks import probe_nvfp4_modelopt_tc_swar_pack_sm121 as probe  # noqa: E402
from scripts import patch_b12x_w4a16_e8m0_k32_scale_reuse as reuse_patcher  # noqa: E402
from scripts import patch_b12x_w4a16_e8m0_scale_fast as finite_patcher  # noqa: E402
from scripts import patch_b12x_w4a16_modelopt_swar_pack as patcher  # noqa: E402
from scripts import patch_b12x_w4a16_modelopt_tc_decode as policy_patcher  # noqa: E402
from scripts import patch_b12x_w4a16_modelopt_vector_load as vector_patcher  # noqa: E402


UPSTREAM_KERNEL = pathlib.Path(
    "/Users/anemll/SourceRelease/GITHUB/ML_playground/dspark-vllm-gx10/"
    ".build/b12x-upstream/b12x/moe/fused/w4a16/kernel.py"
)


def scalar_pack(values: tuple[int, int, int, int]) -> int:
    word = 0
    for index, value in enumerate(values):
        word |= (value & 0xF) << (index * 4)
        word |= ((value >> 4) & 0xF) << (16 + index * 4)
    return word


def swar_pack(values: tuple[int, int, int, int]) -> int:
    q0, q1, q2, q3 = values
    word = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24)
    swap = ((word >> 4) ^ word) & 0x00F000F0
    word = word ^ swap ^ (swap << 4)
    swap = ((word >> 8) ^ word) & 0x0000FF00
    return (word ^ swap ^ (swap << 8)) & 0xFFFFFFFF


class B12xModelOptSwarPackPatchTests(unittest.TestCase):
    @staticmethod
    def _fixture() -> str:
        return "\n".join(anchor for anchor, _replacement, _label in patcher._REPLACEMENTS)

    def test_swar_transpose_is_bit_exact_for_edge_and_random_bytes(self) -> None:
        edge = (0, 1, 15, 16, 127, 128, 240, 255)
        for q0 in edge:
            for q1 in edge:
                for q2 in edge:
                    for q3 in edge:
                        values = (q0, q1, q2, q3)
                        self.assertEqual(swar_pack(values), scalar_pack(values))
        rng = random.Random(4104)
        for _ in range(100_000):
            values = tuple(rng.randrange(256) for _ in range(4))
            self.assertEqual(swar_pack(values), scalar_pack(values))

    def test_specialization_requires_vector_decode_path(self) -> None:
        specialization = next(
            replacement
            for _anchor, replacement, label in patcher._REPLACEMENTS
            if label == "decode-only SWAR-pack specialization"
        )
        self.assertIn("self.modelopt_vector_load", specialization)
        self.assertIn("_modelopt_swar_pack_enabled()", specialization)

    def test_kernel_selection_uses_two_butterfly_swaps(self) -> None:
        selection = next(
            replacement
            for _anchor, replacement, label in patcher._REPLACEMENTS
            if label == "SWAR nibble-transpose selection"
        )
        self.assertIn("Uint32(0x00F000F0)", selection)
        self.assertIn("Uint32(0x0000FF00)", selection)
        self.assertEqual(selection.count("_pack_modelopt_byte_pair("), 4)
        self.assertNotIn("shuffle_sync", selection)
        self.assertNotIn("sync_threads", selection)

    def test_full_patch_chain_has_exact_content_hash(self) -> None:
        source = UPSTREAM_KERNEL.read_text()
        policy = policy_patcher.patch_source(source)
        finite = finite_patcher.patch_source(policy)
        reused = reuse_patcher.patch_source(finite)
        vector = vector_patcher.patch_source(reused)
        self.assertEqual(
            hashlib.sha256(vector.encode()).hexdigest(),
            patcher.PINNED_SOURCE_SHA256,
        )
        patched = patcher.patch_source(vector)
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

    def test_probe_pins_swar_kernel_and_requires_explicit_flag(self) -> None:
        self.assertEqual(probe.EXPECTED_KERNEL_SHA256, patcher.PATCHED_SOURCE_SHA256)
        args = probe.build_parser().parse_args(
            [
                "--layer-file",
                "/model/model-layer-00000.safetensors",
                "--output",
                "/tmp/modelopt-swar.json",
                "--modelopt-swar-pack",
            ]
        )
        self.assertTrue(args.modelopt_swar_pack)
        self.assertEqual(probe.WINNING_GEOMETRY, (128, 64))


if __name__ == "__main__":
    unittest.main()
