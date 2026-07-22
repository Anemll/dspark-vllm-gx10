# SPDX-License-Identifier: MIT

from __future__ import annotations

import unittest
from pathlib import Path

from scripts import patch_flashinfer_b12x_c4_static_cutover as cutover_patch


class FlashInferB12xC4StaticCutoverPatchTest(unittest.TestCase):
    def test_patches_only_the_multi_topk_cutover(self) -> None:
        source = "prefix\n" + cutover_patch.ANCHOR + "suffix\n"
        patched = cutover_patch.patch_source(source)
        self.assertIn(cutover_patch.REPLACEMENT, patched)
        self.assertNotIn(cutover_patch.ANCHOR, patched)

    def test_rejects_anchor_drift(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "multi-top-k micro cutover"):
            cutover_patch.patch_source("not the pinned dispatcher")

    def test_overlay_is_minimal_and_context_pins_the_patch(self) -> None:
        root = Path(__file__).resolve().parents[1]
        dockerfile = (
            root / "docker" / "Dockerfile.nvfp4-b12x-c4-static-overlay"
        ).read_text(encoding="utf-8")
        dockerignore = (
            root / "docker" / "Dockerfile.nvfp4-b12x-c4-static-overlay.dockerignore"
        ).read_text(encoding="utf-8")
        self.assertIn("patch_flashinfer_b12x_c4_static_cutover.py", dockerfile)
        self.assertNotIn("COPY overlay/", dockerfile)
        self.assertIn("!scripts/patch_flashinfer_b12x_c4_static_cutover.py", dockerignore)


if __name__ == "__main__":
    unittest.main()
