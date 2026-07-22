# SPDX-License-Identifier: MIT

from __future__ import annotations

import unittest
from pathlib import Path

from scripts import patch_flashinfer_b12x_shared_input as shared_input_patch


class FlashInferB12xSharedInputPatchTest(unittest.TestCase):
    def test_patch_is_single_token_only(self) -> None:
        wrapper = (
            "prefix\n"
            + shared_input_patch._WRAPPER_RETURN_ANCHOR
            + "middle\n"
            + shared_input_patch._WRAPPER_ARGS_ANCHOR
            + "suffix\n"
        )
        wrapper_patched = shared_input_patch.patch_wrapper_source(wrapper)
        self.assertIn("w1_alpha[:1]", wrapper_patched)
        self.assertIn("fc2_input_scale[:1]", wrapper_patched)

        dispatch = (
            shared_input_patch._DISPATCH_SHARE_INPUT_ANCHOR
            + shared_input_patch._DISPATCH_SHARE_SCALE_ANCHOR
        )
        dispatch_patched = shared_input_patch.patch_dispatch_source(dispatch)
        self.assertIn("num_tokens == 1", dispatch_patched)
        self.assertNotIn('activation == "relu2"', dispatch_patched)

    def test_patch_rejects_source_drift(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "wrapper launch"):
            shared_input_patch.patch_wrapper_source("not pinned")
        with self.assertRaisesRegex(RuntimeError, "dispatcher shared input"):
            shared_input_patch.patch_dispatch_source("not pinned")

    def test_image_bakes_patch_and_context_includes_it(self) -> None:
        root = Path(__file__).resolve().parents[1]
        dockerfile = (
            root / "docker" / "Dockerfile.nvfp4-b12x-decode-overlay"
        ).read_text(encoding="utf-8")
        dockerignore = (
            root
            / "docker"
            / "Dockerfile.nvfp4-b12x-decode-overlay.dockerignore"
        ).read_text(encoding="utf-8")
        self.assertIn("dspark-patch-flashinfer-b12x-shared-input", dockerfile)
        self.assertIn(
            "!scripts/patch_flashinfer_b12x_shared_input.py", dockerignore
        )


if __name__ == "__main__":
    unittest.main()
