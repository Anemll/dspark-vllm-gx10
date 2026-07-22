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

from benchmarks import benchmark_nvfp4_a4w4_sm121 as kernel_bench  # noqa: E402
from scripts import patch_b12x_w4a16_modelopt_tc_decode as patcher  # noqa: E402


class B12xW4A16ModelOptTcDecodePatchTests(unittest.TestCase):
    @staticmethod
    def _fixture() -> str:
        return "\n".join(
            anchor for anchor, _replacement, _label in patcher._REPLACEMENTS
        )

    def test_patch_opens_exact_three_modelopt_policy_gates(self) -> None:
        source = self._fixture()
        patched = patcher.patch_source(source)

        # One compile-time direct-top-k exclusion is removed.  The two runtime
        # policies explicitly admit the already-validated B12X layout set.
        self.assertNotIn('or weight_layout != "packed"', patched)
        self.assertEqual(patched.count("weight_layout in _WEIGHT_LAYOUTS"), 2)
        self.assertNotIn('and weight_layout == "packed"', patched)
        self.assertIn(
            "direct_topk_routes is only valid for small-M W4A16 without expert_map",
            patched,
        )
        self.assertIn("packed or ModelOpt int32 topk_ids", patched)
        self.assertIn("packed or ModelOpt bf16 decode", patched)
        self.assertIn("packed or single-copy\n# ModelOpt W4A16 weights", patched)

    def test_patch_preserves_default_off_gate_and_packed_contract(self) -> None:
        source = self._fixture()
        patched = patcher.patch_source(source)

        # The patch changes eligibility only.  It neither removes the env gate
        # nor rewrites any packed-loader implementation.
        self.assertIn("_tc_decode_enabled()", patched)
        self.assertIn("weight_layout in _WEIGHT_LAYOUTS", patched)
        self.assertNotIn("weight_layout = \"modelopt\"", patched)
        self.assertEqual(len(patcher._REPLACEMENTS), 7)
        self.assertEqual(
            len({label for _anchor, _replacement, label in patcher._REPLACEMENTS}),
            len(patcher._REPLACEMENTS),
        )

    def test_patch_is_fail_closed_on_source_or_second_application(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError, "module TC-decode layout contract"
        ):
            patcher.patch_source("not the pinned B12X kernel")

        patched = patcher.patch_source(self._fixture())
        with self.assertRaisesRegex(
            RuntimeError, "module TC-decode layout contract"
        ):
            patcher.patch_source(patched)

    def test_cli_checks_input_and_result_hashes_before_writing(self) -> None:
        source = self._fixture().encode("utf-8")
        expected = patcher.patch_source(source.decode("utf-8")).encode("utf-8")
        source_sha = hashlib.sha256(source).hexdigest()
        result_sha = hashlib.sha256(expected).hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / "kernel.py"
            target.write_bytes(source)
            with (
                mock.patch.object(patcher, "PINNED_SOURCE_SHA256", source_sha),
                mock.patch.object(patcher, "PATCHED_SOURCE_SHA256", result_sha),
                mock.patch.object(sys, "argv", ["patcher", "--target", str(target)]),
            ):
                self.assertEqual(patcher.main(), 0)
            self.assertEqual(target.read_bytes(), expected)

            target.write_bytes(source)
            with (
                mock.patch.object(patcher, "PINNED_SOURCE_SHA256", source_sha),
                mock.patch.object(patcher, "PATCHED_SOURCE_SHA256", "0" * 64),
                mock.patch.object(sys, "argv", ["patcher", "--target", str(target)]),
                self.assertRaisesRegex(RuntimeError, "deterministic .* result mismatch"),
            ):
                patcher.main()
            self.assertEqual(target.read_bytes(), source)

    def test_pins_are_content_hashes_and_benchmark_defaults_to_modelopt(self) -> None:
        self.assertRegex(patcher.PINNED_SOURCE_SHA256, r"^[0-9a-f]{64}$")
        self.assertRegex(patcher.PATCHED_SOURCE_SHA256, r"^[0-9a-f]{64}$")
        self.assertNotEqual(
            patcher.PINNED_SOURCE_SHA256, patcher.PATCHED_SOURCE_SHA256
        )

        args = kernel_bench.build_parser().parse_args(
            [
                "--model-path",
                "/model",
                "--backend",
                "both",
                "--m",
                "1,4",
                "--correctness-m",
                "1,4",
                "--dry-run",
            ]
        )
        kernel_bench.validate_args(args)
        self.assertEqual(args.w4a16_weight_layout, "modelopt")
        self.assertEqual(args.m, (1, 4))
        self.assertEqual(args.correctness_m, (1, 4))


if __name__ == "__main__":
    unittest.main()
