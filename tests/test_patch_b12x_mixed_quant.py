# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "scripts" / "patch-b12x-mixed-quant.py"
SPEC = importlib.util.spec_from_file_location("patch_b12x_mixed_quant", PATCHER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PatchB12xMixedQuantTests(unittest.TestCase):
    def test_patches_exact_pinned_runtime_source(self):
        source_path = (
            ROOT
            / ".build/vllm-b12x-vision-upstream/vllm/models/deepseek_v4/quant_config.py"
        )
        if not source_path.is_file():
            self.skipTest("optional pinned B12X runtime source is absent")
        source = source_path.read_bytes()
        expected = hashlib.sha256(source).hexdigest()
        self.assertEqual(
            expected,
            "6fdd20b617640741a19643f893b9b8a31d390b0ccb04d322ece29f57ff8290c9",
        )
        patched = MODULE.patch(source, expected).decode()
        self.assertIn("def _moe_quant_algo_for_prefix", patched)
        self.assertIn("def is_mxfp4_quant", patched)
        self.assertIn(
            'if self._moe_quant_algo_for_prefix(prefix) == "NVFP4":', patched
        )
        self.assertNotIn('if self.moe_quant_algo == "NVFP4":', patched)

    def test_rejects_source_drift(self):
        with self.assertRaisesRegex(ValueError, "refusing to patch unknown"):
            MODULE.patch(b"not the runtime", "0" * 64)


if __name__ == "__main__":
    unittest.main()
