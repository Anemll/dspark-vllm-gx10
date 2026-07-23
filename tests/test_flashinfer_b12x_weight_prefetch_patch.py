# SPDX-License-Identifier: MIT

from __future__ import annotations

import unittest

from scripts import patch_flashinfer_b12x_weight_prefetch as patcher


class B12xWeightPrefetchPatchTest(unittest.TestCase):
    def test_patch_adds_fc1_up_and_fc2_prefetch(self) -> None:
        source = "\n".join(anchor for anchor, _replacement, _label in patcher._REPLACEMENTS)
        patched = patcher.patch_source(source)

        self.assertIn("cute.prefetch(\n                        tma_b_w13", patched)
        self.assertIn("tBgB_w13_up_nk", patched)
        self.assertIn("future_output_tile", patched)
        self.assertIn("tma_b_down", patched)

    def test_patch_rejects_anchor_drift(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "FC1 prefetch start"):
            patcher.patch_source("not the pinned source")


if __name__ == "__main__":
    unittest.main()
