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

from benchmarks import probe_nvfp4_modelopt_tc_e8m0_scale_reuse_sm121 as probe  # noqa: E402
from scripts import patch_b12x_w4a16_e8m0_k32_scale_reuse as patcher  # noqa: E402
from scripts import patch_b12x_w4a16_e8m0_scale_fast as finite_patcher  # noqa: E402
from scripts import patch_b12x_w4a16_modelopt_tc_decode as policy_patcher  # noqa: E402


UPSTREAM_KERNEL = pathlib.Path(
    "/Users/anemll/SourceRelease/GITHUB/ML_playground/dspark-vllm-gx10/"
    ".build/b12x-upstream/b12x/moe/fused/w4a16/kernel.py"
)


class B12xE8m0K32ScaleReusePatchTests(unittest.TestCase):
    @staticmethod
    def _fixture() -> str:
        return "\n".join(anchor for anchor, _replacement, _label in patcher._REPLACEMENTS)

    def test_scale_group_sequence_is_identical_for_all_c_tactic_warps(self) -> None:
        # C/C uses eight K16 fragments per CTA K tile.  scale_group_id is
        # (8 * warp_row + kk) // 2.  Loading on each even kk and carrying that
        # exact register bundle to the following odd kk must reproduce it.
        for warp_row in range(4):
            original = [(8 * warp_row + kk) // 2 for kk in range(8)]
            loaded: list[int] = []
            current: int | None = None
            for kk in range(8):
                if kk % 2 == 0:
                    current = (8 * warp_row + kk) // 2
                self.assertIsNotNone(current)
                loaded.append(int(current))
            self.assertEqual(loaded, original)
            self.assertEqual(len(set(loaded)), 4)

    def test_patch_is_opt_in_beneath_finite_decode_specialization(self) -> None:
        patched = patcher.patch_source(self._fixture())
        specialization = next(
            replacement
            for _anchor, replacement, label in patcher._REPLACEMENTS
            if label == "decode-only K32 scale-reuse specialization"
        )
        self.assertIn("self.e8m0_finite_fast", specialization)
        self.assertIn("_e8m0_k32_scale_reuse_enabled()", specialization)
        self.assertIn(
            "self.e8m0_k32_scale_reuse and (kk % 2 == 0)",
            patched,
        )
        self.assertEqual(patched.count("self.e8m0_k32_scale_reuse,"), 1)

    def test_second_fragment_loads_weights_and_copies_only_scale_row(self) -> None:
        helper = next(
            replacement
            for _anchor, replacement, label in patcher._REPLACEMENTS
            if label == "second-K16 weight-only bundle loader"
        )
        self.assertIn("_load_b_registers_modelopt_shared(", helper)
        self.assertIn("dst[0, 0] = q0", helper)
        self.assertIn("dst[0, 3] = q3", helper)
        self.assertIn("dst[1, col] = current[1, col]", helper)
        self.assertNotIn("_dequant_scale", helper)
        self.assertNotIn("ld_shared_v2_u32", helper)

    def test_full_patch_chain_has_exact_content_hash(self) -> None:
        source = UPSTREAM_KERNEL.read_text()
        finite = finite_patcher.patch_source(policy_patcher.patch_source(source))
        self.assertEqual(
            hashlib.sha256(finite.encode()).hexdigest(),
            patcher.PINNED_SOURCE_SHA256,
        )
        reused = patcher.patch_source(finite)
        self.assertEqual(
            hashlib.sha256(reused.encode()).hexdigest(),
            patcher.PATCHED_SOURCE_SHA256,
        )
        compile(reused, str(UPSTREAM_KERNEL), "exec", dont_inherit=True)

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

    def test_probe_pins_reused_kernel_and_requires_explicit_flag(self) -> None:
        self.assertEqual(probe.EXPECTED_KERNEL_SHA256, patcher.PATCHED_SOURCE_SHA256)
        args = probe.build_parser().parse_args(
            [
                "--layer-file",
                "/model/model-layer-00000.safetensors",
                "--output",
                "/tmp/e8m0-reuse.json",
                "--k32-scale-reuse",
            ]
        )
        self.assertTrue(args.k32_scale_reuse)
        self.assertEqual(probe.WINNING_TILE, "c")
        self.assertEqual(probe.WINNING_GEOMETRY, (128, 64))


if __name__ == "__main__":
    unittest.main()
