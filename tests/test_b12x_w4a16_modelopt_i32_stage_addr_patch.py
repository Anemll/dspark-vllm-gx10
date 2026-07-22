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

from scripts import patch_b12x_w4a16_e8m0_k32_scale_reuse as reuse_patcher  # noqa: E402
from scripts import patch_b12x_w4a16_e8m0_scale_fast as finite_patcher  # noqa: E402
from scripts import patch_b12x_w4a16_modelopt_i32_stage_addr as patcher  # noqa: E402
from scripts import patch_b12x_w4a16_modelopt_tc_decode as policy_patcher  # noqa: E402
from scripts import patch_b12x_w4a16_modelopt_vector_load as vector_patcher  # noqa: E402
from benchmarks import probe_nvfp4_modelopt_tc_i32_stage_addr_sm121 as probe  # noqa: E402


UPSTREAM_KERNEL = pathlib.Path(
    "/Users/anemll/SourceRelease/GITHUB/ML_playground/dspark-vllm-gx10/"
    ".build/b12x-upstream/b12x/moe/fused/w4a16/kernel.py"
)


class B12xModelOptI32StageAddrPatchTests(unittest.TestCase):
    @staticmethod
    def _fixture() -> str:
        return "\n".join(anchor for anchor, _replacement, _label in patcher._REPLACEMENTS)

    def test_exact_dsv4_fc_shapes_have_signed_i32_vector_starts(self) -> None:
        # (N, K) for FC1 and FC2 after TP=2.  Check every expert/tile/local
        # vector start, not only the aggregate upper-bound formula.
        for size_n, size_k in ((4096, 4096), (4096, 2048)):
            tile_n = 64
            tile_k = 128
            expert_stride = size_n * (size_k // 2)
            last = -1
            for expert in range(256):
                for n_tile in range(size_n // tile_n):
                    for k_tile in range(size_k // tile_k):
                        tile_base = (
                            expert * expert_stride
                            + n_tile * tile_n * (size_k // 2)
                            + k_tile * (tile_k // 2)
                        )
                        for local_n in (0, tile_n - 1):
                            for local_k_vec in (0, tile_k // 32 - 1):
                                offset = (
                                    tile_base
                                    + local_n * (size_k // 2)
                                    + local_k_vec * 16
                                )
                                self.assertGreaterEqual(offset, 0)
                                self.assertLessEqual(offset, 0x7FFFFFFF)
                                last = max(last, offset)
            self.assertEqual(last, 256 * expert_stride - 16)

    def test_specialization_is_opt_in_beneath_vector_decode(self) -> None:
        specialization = next(
            replacement
            for _anchor, replacement, label in patcher._REPLACEMENTS
            if label == "decode-only i32-stage-address specialization"
        )
        self.assertIn("self.modelopt_vector_load", specialization)
        self.assertIn("_modelopt_i32_stage_addr_enabled()", specialization)
        self.assertIn("last_vector_start", specialization)
        self.assertIn("0x7FFFFFFF", specialization)

    def test_hoisted_base_removes_int64_multiply_from_fast_branch(self) -> None:
        local = next(
            replacement
            for _anchor, replacement, label in patcher._REPLACEMENTS
            if label == "per-copy i32 local address arithmetic"
        )
        hoisted = next(
            replacement
            for _anchor, replacement, label in patcher._REPLACEMENTS
            if label == "hoisted i32 expert/N/K tile base"
        )
        fast_branch = local.split("else:", 1)[0]
        self.assertNotIn("Int64(expert_idx)", fast_branch)
        self.assertNotIn("Int64(source_n)", fast_branch)
        self.assertIn("modelopt_tile_byte_base", fast_branch)
        self.assertIn("expert_idx * Int32", hoisted)

    def test_full_patch_chain_has_exact_content_hash(self) -> None:
        source = UPSTREAM_KERNEL.read_text()
        source = policy_patcher.patch_source(source)
        source = finite_patcher.patch_source(source)
        source = reuse_patcher.patch_source(source)
        source = vector_patcher.patch_source(source)
        self.assertEqual(
            hashlib.sha256(source.encode()).hexdigest(),
            patcher.PINNED_SOURCE_SHA256,
        )
        patched = patcher.patch_source(source)
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

    def test_probe_pins_i32_kernel_and_requires_explicit_flag(self) -> None:
        self.assertEqual(probe.EXPECTED_KERNEL_SHA256, patcher.PATCHED_SOURCE_SHA256)
        args = probe.build_parser().parse_args(
            [
                "--layer-file",
                "/model/model-layer-00000.safetensors",
                "--output",
                "/tmp/modelopt-i32-stage.json",
                "--modelopt-vector-load",
                "--modelopt-i32-stage-addr",
            ]
        )
        self.assertTrue(args.modelopt_vector_load)
        self.assertTrue(args.modelopt_i32_stage_addr)


if __name__ == "__main__":
    unittest.main()
