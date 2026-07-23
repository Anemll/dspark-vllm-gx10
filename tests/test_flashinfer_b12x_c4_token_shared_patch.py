# SPDX-License-Identifier: MIT

from __future__ import annotations

import unittest
from pathlib import Path

from scripts import patch_flashinfer_b12x_c4_token_shared as token_shared_patch


class FlashInferB12xC4TokenSharedPatchTest(unittest.TestCase):
    def test_dispatch_keeps_m1_direct_and_adds_only_c2_to_c4_compact_sharing(self) -> None:
        source = (
            token_shared_patch.DISPATCH_FLAGS_ANCHOR
            + token_shared_patch.DISPATCH_CALL_ANCHOR
            + token_shared_patch.DISPATCH_FACTORY_ARGS_ANCHOR
            + token_shared_patch.DISPATCH_FACTORY_CACHE_ANCHOR
            + token_shared_patch.DISPATCH_FACTORY_KERNEL_ANCHOR
        )
        patched = token_shared_patch.patch_dispatch_source(source)
        self.assertIn("num_tokens == 1", patched)
        self.assertIn("2 <= num_tokens <= 4", patched)
        self.assertIn("per_token_shared_input=per_token_shared_input", patched)
        self.assertIn("per_token_shared_input: bool = False", patched)
        self.assertIn("per_token_shared_input,", patched)
        self.assertNotIn("pairwise_routes = num_tokens <= 4", patched)

    def test_micro_reuses_first_route_pack_without_reserved_slots(self) -> None:
        source = (
            token_shared_patch.MICRO_INIT_ARGS_ANCHOR
            + token_shared_patch.MICRO_INIT_ATTRS_ANCHOR
            + token_shared_patch.MICRO_QUANT_ANCHOR
            + token_shared_patch.MICRO_PACK_BARRIER_ANCHOR
        )
        patched = token_shared_patch.patch_micro_kernel_source(source)
        self.assertIn("self.per_token_shared_input = per_token_shared_input", patched)
        self.assertIn(
            "source_expert = topk_ids[token_idx * num_topk].to(Int32)", patched
        )
        self.assertIn("source_row_count = row_counts[source_expert]", patched)
        self.assertIn(
            "source_expert * max_rows * output_bytes_per_row\n"
            "                        + source_row * output_bytes_per_row",
            patched,
        )
        self.assertNotIn("num_experts - num_tokens", patched)
        self.assertIn("token_map[local_expert_id * max_rows + scan_row]", patched)
        self.assertIn("st_global_u64(", patched)
        self.assertIn("_ld_global_u64(", patched)
        self.assertIn("scale_storage[dest_scale_offset] = scale_storage[source_scale_offset]", patched)
        self.assertLess(
            patched.index("source_expert = topk_ids[token_idx * num_topk].to(Int32)"),
            patched.index("gA = cute.local_tile"),
        )

    def test_patch_rejects_anchor_drift(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "dispatcher flags"):
            token_shared_patch.patch_dispatch_source("drift")
        with self.assertRaisesRegex(RuntimeError, "micro init args"):
            token_shared_patch.patch_micro_kernel_source("drift")

    def test_overlay_is_minimal(self) -> None:
        root = Path(__file__).resolve().parents[1]
        dockerfile = (
            root / "docker" / "Dockerfile.nvfp4-b12x-c4-token-shared-overlay"
        ).read_text(encoding="utf-8")
        dockerignore = (
            root
            / "docker"
            / "Dockerfile.nvfp4-b12x-c4-token-shared-overlay.dockerignore"
        ).read_text(encoding="utf-8")
        self.assertIn("patch_flashinfer_b12x_c4_token_shared.py", dockerfile)
        self.assertNotIn("COPY overlay/", dockerfile)
        self.assertIn("!scripts/patch_flashinfer_b12x_c4_token_shared.py", dockerignore)


if __name__ == "__main__":
    unittest.main()
