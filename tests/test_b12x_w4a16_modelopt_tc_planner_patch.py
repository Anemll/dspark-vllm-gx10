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

from scripts import patch_b12x_w4a16_modelopt_tc_planner as patcher  # noqa: E402
from scripts import patch_b12x_nvfp4_swiglu_limit as swiglu_patcher  # noqa: E402


UPSTREAM_TP_MOE = pathlib.Path(
    "/Users/anemll/SourceRelease/GITHUB/ML_playground/dspark-vllm-gx10/"
    ".build/b12x-upstream/b12x/integration/tp_moe.py"
)


class B12xW4A16ModelOptTcPlannerPatchTests(unittest.TestCase):
    @staticmethod
    def _pinned_source() -> str:
        return swiglu_patcher.patch_source(UPSTREAM_TP_MOE.read_text())

    @staticmethod
    def _fixture() -> str:
        return "\n".join(
            anchor for anchor, _replacement, _label in patcher._REPLACEMENTS
        )

    def test_patch_wires_both_build_and_selection_to_modelopt(self) -> None:
        patched = patcher.patch_source(self._pinned_source())
        self.assertEqual(
            patched.count('weight_layout in {"packed", "modelopt"}'),
            2,
        )
        self.assertNotIn('weight_layout == "packed"', patched)
        self.assertIn("single-copy ModelOpt small-M decode", patched)
        self.assertIn("planned_tc_decode_launches.get(token_count)", patched)
        self.assertIn("compile_w4a16_fused_moe(", patched)
        self.assertIn("tc_decode_fused_sum=True", patched)

    def test_patch_preserves_environment_and_bf16_gates(self) -> None:
        patched = patcher.patch_source(self._pinned_source())
        self.assertEqual(patched.count("_tc_decode_enabled()"), 2)
        self.assertIn('element_dtype == "bf16"', patched)
        self.assertIn("token_count <= _TC_DECODE_MAX_M", patched)
        self.assertNotIn("_TC_DECODE_MAX_M =", patched)

    def test_full_pinned_source_has_exact_result_hash(self) -> None:
        source = self._pinned_source().encode()
        self.assertEqual(
            hashlib.sha256(source).hexdigest(),
            patcher.PINNED_SOURCE_SHA256,
        )
        patched = patcher.patch_source(source.decode()).encode()
        self.assertEqual(
            hashlib.sha256(patched).hexdigest(),
            patcher.PATCHED_SOURCE_SHA256,
        )
        compile(patched, str(UPSTREAM_TP_MOE), "exec", dont_inherit=True)

    def test_explicit_layout_reaches_sizing_materialization_and_prewarm(self) -> None:
        patched = patcher.patch_source(self._pinned_source())
        self.assertIn("w4a16_weight_layout: str | None = None", patched)
        self.assertIn(
            "w4a16_weight_layout=self.caps.w4a16_weight_layout",
            patched,
        )
        self.assertIn(
            "w4a16_weight_layout=caps.w4a16_weight_layout",
            patched,
        )
        self.assertEqual(
            patched.count("w4a16_weight_layout=w4a16_weight_layout"),
            2,
        )
        self.assertIn(
            "if w4a16_weight_layout is None\n"
            "            else _normalize_w4a16_weight_layout",
            patched,
        )

    def test_default_layout_mapping_remains_packed(self) -> None:
        patched = patcher.patch_source(self._pinned_source())
        helper_start = patched.index("def _w4a16_weight_layout_for_source(")
        helper_end = patched.index("\n\n_W4A16_WEIGHT_LAYOUTS", helper_start)
        helper = patched[helper_start:helper_end]
        self.assertIn('return "packed"', helper)
        self.assertNotIn('return "modelopt"', helper)

    def test_patch_fails_closed_on_drift_or_second_application(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "scratch-cap layout field"):
            patcher.patch_source("not the pinned planner")
        patched = patcher.patch_source(self._fixture())
        with self.assertRaisesRegex(RuntimeError, "scratch-cap layout field"):
            patcher.patch_source(patched)

    def test_cli_checks_input_and_output_before_write(self) -> None:
        source = self._fixture().encode()
        expected = patcher.patch_source(source.decode()).encode()
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / "tp_moe.py"
            target.write_bytes(source)
            with (
                mock.patch.object(
                    patcher,
                    "PINNED_SOURCE_SHA256",
                    hashlib.sha256(source).hexdigest(),
                ),
                mock.patch.object(
                    patcher,
                    "PATCHED_SOURCE_SHA256",
                    hashlib.sha256(expected).hexdigest(),
                ),
                mock.patch.object(
                    sys, "argv", ["patcher", "--target", str(target)]
                ),
            ):
                self.assertEqual(patcher.main(), 0)
            self.assertEqual(target.read_bytes(), expected)


if __name__ == "__main__":
    unittest.main()
