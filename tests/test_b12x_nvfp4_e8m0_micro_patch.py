from __future__ import annotations

import unittest
from pathlib import Path

from scripts import patch_b12x_nvfp4_e8m0_micro as patcher


ROOT = Path(
    "/Users/anemll/SourceRelease/GITHUB/ML_playground/dspark-vllm-gx10/"
    ".build/b12x-upstream/b12x"
)


class NvFp4E8m0MicroPatchTest(unittest.TestCase):
    def test_pinned_sources_and_patch_contract(self) -> None:
        micro = (ROOT / "moe/fused/micro.py").read_text()
        tp_moe = (ROOT / "integration/tp_moe.py").read_text()
        silu = (ROOT / "moe/fused/silu.py").read_text()
        patched_micro = patcher.patch_micro(micro)
        self.assertIn("scale_format_e8m0_k32", patched_micro)
        self.assertEqual(
            patched_micro.count(
                "cvt_e4m3_to_f32_via_f16(cvt_f32_to_e4m3(q_scale))"
            ),
            4,
        )
        patched_tp = patcher.patch_tp_moe(tp_moe)
        self.assertIn("B12X_NVFP4_MICRO_SCALE_FORMAT", patched_tp)
        self.assertIn("scale_format=micro_scale_format", patched_tp)
        patched_silu = patcher.patch_silu(silu)
        self.assertIn('scale_format: str = "e4m3_k16"', patched_silu)
        self.assertIn("scale_format=scale_format", patched_silu)

    def test_default_is_unchanged_e4m3(self) -> None:
        patched = patcher.patch_tp_moe(
            (ROOT / "integration/tp_moe.py").read_text()
        )
        self.assertIn(
            '"B12X_NVFP4_MICRO_SCALE_FORMAT", "e4m3_k16"', patched
        )


if __name__ == "__main__":
    unittest.main()
