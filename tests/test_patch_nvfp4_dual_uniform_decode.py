# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from __future__ import annotations

import unittest

from scripts import patch_nvfp4_dual_uniform_decode as patcher


class NvFp4DualUniformDecodePatchTests(unittest.TestCase):
    def test_model_runner_propagates_exact_decode_uniformity(self) -> None:
        patched = patcher.replace_exact(
            patcher._MODEL_RUNNER_DESCRIPTOR_ANCHOR,
            patcher._MODEL_RUNNER_DESCRIPTOR_ANCHOR,
            patcher._MODEL_RUNNER_DESCRIPTOR_REPLACEMENT,
            "model runner descriptor",
        )
        self.assertIn("num_reqs=num_reqs", patched)
        self.assertIn(
            "uniform=uniform_tok_count == self.decode_query_len", patched
        )
        self.assertNotIn("uniform=uniform_tok_count is not None", patched)

    def test_uniform_one_token_prefill_is_excluded_before_dispatch(self) -> None:
        patched = patcher.replace_exact(
            patcher._MODEL_RUNNER_PREFILL_ANCHOR,
            patcher._MODEL_RUNNER_PREFILL_ANCHOR,
            patcher._MODEL_RUNNER_PREFILL_REPLACEMENT,
            "model runner prefill exclusion",
        )
        self.assertIn("if not dummy_run and uniform_tok_count is not None", patched)
        self.assertIn("num_computed_prefill_tokens", patched)
        self.assertIn("prefill_len.np", patched)
        self.assertIn("if bool(np.any(prefilling))", patched)
        self.assertIn("uniform_tok_count = None", patched)

    def test_full_cuda_graph_propagates_exact_decode_uniformity(self) -> None:
        patched = patcher.replace_exact(
            patcher._CUDAGRAPH_UTILS_ANCHOR,
            patcher._CUDAGRAPH_UTILS_ANCHOR,
            patcher._CUDAGRAPH_UTILS_REPLACEMENT,
            "CUDA graph",
        )
        full = patched.index("if cg_mode == CUDAGraphMode.FULL:")
        descriptor = patched.index("batch_descriptor = BatchDescriptor(", full)
        piecewise = patched.index("if cg_mode == CUDAGraphMode.PIECEWISE:")
        self.assertLess(full, descriptor)
        self.assertLess(descriptor, piecewise)
        self.assertIn(
            "desc.uniform_token_count == self.decode_query_len", patched
        )
        piecewise_body = patched[piecewise:]
        self.assertNotIn("uniform=", piecewise_body)

    def test_content_addressed_replacement_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "found 0"):
            patcher.replace_exact("other", "anchor", "replacement", "test")
        with self.assertRaisesRegex(RuntimeError, "found 2"):
            patcher.replace_exact(
                "anchor anchor", "anchor", "replacement", "test"
            )


if __name__ == "__main__":
    unittest.main()
