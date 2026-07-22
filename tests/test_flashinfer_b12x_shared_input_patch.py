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
            + shared_input_patch._DISPATCH_PAIRWISE_DECL_ANCHOR
            + shared_input_patch._DISPATCH_ROUTING_ANCHOR
            + shared_input_patch._DISPATCH_SINGLE_TOKEN_ANCHOR
        )
        dispatch_patched = shared_input_patch.patch_dispatch_source(dispatch)
        self.assertIn("num_tokens == 1", dispatch_patched)
        self.assertNotIn('activation == "relu2"', dispatch_patched)
        self.assertIn("pairwise_routes = num_tokens == 1", dispatch_patched)
        self.assertNotIn("num_tokens <= 4", dispatch_patched)
        self.assertIn("if pairwise_routes:", dispatch_patched)
        self.assertIn("single_token=pairwise_routes", dispatch_patched)
        self.assertNotIn("torch.arange", dispatch_patched)

        micro = (
            shared_input_patch._MICRO_TOKEN_ANCHOR
            + shared_input_patch._MICRO_EXPERT_ANCHOR
            + shared_input_patch._MICRO_QUANT_EXPERT_ANCHOR
            + shared_input_patch._MICRO_MMA_WEIGHT_EXPERT_ANCHOR
            + shared_input_patch._MICRO_WEIGHT_EXPERT_ANCHOR
        )
        micro_patched = shared_input_patch.patch_micro_kernel_source(micro)
        self.assertIn("token_idx = pair_idx // num_topk", micro_patched)
        self.assertIn(
            "expert_id = topk_ids[local_expert_id].to(Int32)", micro_patched
        )
        self.assertEqual(
            micro_patched.count(
                "weight_expert_idx = topk_ids[local_expert_idx].to(Int32)"
            ),
            2,
        )

    def test_patch_rejects_source_drift(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "wrapper launch"):
            shared_input_patch.patch_wrapper_source("not pinned")
        with self.assertRaisesRegex(RuntimeError, "dispatcher shared input"):
            shared_input_patch.patch_dispatch_source("not pinned")
        with self.assertRaisesRegex(RuntimeError, "microkernel pair token"):
            shared_input_patch.patch_micro_kernel_source("not pinned")

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
        self.assertIn("blackwell_sm12x/moe_micro_kernel.py", dockerfile)
        self.assertIn(
            "!scripts/patch_flashinfer_b12x_shared_input.py", dockerignore
        )


if __name__ == "__main__":
    unittest.main()
