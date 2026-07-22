# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import benchmark_nvfp4_prepared_trtllm_sm121 as bench  # noqa: E402


class PreparedTrtLlmNvFp4ComparatorTests(unittest.TestCase):
    def _args(self, *extra: str):
        return bench.build_parser().parse_args(
            [
                "--layer-file",
                "layer.safetensors",
                "--output",
                "result.json",
                *extra,
            ]
        )

    def test_parser_pins_real_decode_shapes_and_clamp(self) -> None:
        args = self._args()

        self.assertEqual(args.m, (1, 4))
        self.assertEqual(args.tp_rank, 0)
        self.assertEqual(args.routing, "balanced")
        self.assertEqual(args.swiglu_limit, 10.0)
        bench.validate_args(args)

    def test_csv_parser_deduplicates_and_rejects_nonpositive(self) -> None:
        self.assertEqual(bench._csv_positive_ints("4,1,4"), (4, 1))
        for value in ("", "0", "1,-1", "x"):
            with self.subTest(value=value), self.assertRaises(
                argparse.ArgumentTypeError
            ):
                bench._csv_positive_ints(value)

    def test_validation_requires_m4_and_valid_numeric_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires M=4"):
            bench.validate_args(self._args("--m", "1"))
        with self.assertRaisesRegex(ValueError, "TP rank"):
            bench.validate_args(self._args("--tp-rank", "2"))
        with self.assertRaisesRegex(ValueError, "swiglu-limit"):
            bench.validate_args(self._args("--swiglu-limit", "nan"))
        with self.assertRaisesRegex(ValueError, "numeric-min-cosine"):
            bench.validate_args(self._args("--numeric-min-cosine", "1.01"))

    def test_support_override_accepts_only_exact_family_rejection(self) -> None:
        proof = bench.evaluate_sm121_support_override(
            capability=(12, 1),
            symbol_available=True,
            native_supported=False,
            native_reason="kernel does not support current device NVIDIA GB10",
            family_only_supported=True,
            family_only_reason=None,
        )
        self.assertTrue(proof["passed"])
        self.assertEqual(
            proof["only_changed_predicate"],
            "TrtLlmNvFp4ExpertsBase._supports_current_device",
        )

        cases = (
            {"capability": (12, 0)},
            {"symbol_available": False},
            {"native_supported": True},
            {"native_reason": "kernel does not support shape"},
            {"family_only_supported": False},
            {"family_only_reason": "still rejected"},
        )
        defaults = {
            "capability": (12, 1),
            "symbol_available": True,
            "native_supported": False,
            "native_reason": "kernel does not support current device NVIDIA GB10",
            "family_only_supported": True,
            "family_only_reason": None,
        }
        for change in cases:
            with self.subTest(change=change):
                self.assertFalse(
                    bench.evaluate_sm121_support_override(
                        **(defaults | change)
                    )["passed"]
                )

    def test_physical_call_contract_requires_routed_kernel_and_clamp(self) -> None:
        call = {
            "do_finalize": True,
            "top_k": 6,
            "num_experts": 256,
            "local_num_experts": 256,
            "gemm1_clamp_present": True,
            "gemm1_alpha_present": True,
            "gemm1_beta_present": True,
            "output_pointer_identity": True,
            "hidden_states_shape": [4, 2048],
            "hidden_states_scale_shape": [4, 256],
        }
        self.assertTrue(bench.summarize_backend_call(call)["passed"])
        for field in (
            "do_finalize",
            "gemm1_clamp_present",
            "output_pointer_identity",
        ):
            with self.subTest(field=field):
                broken = dict(call)
                broken[field] = False
                self.assertFalse(bench.summarize_backend_call(broken)["passed"])

    def test_source_contains_real_symbol_and_modelopt_algebra_proofs(self) -> None:
        source = pathlib.Path(bench.__file__).read_text()

        self.assertIn("trtllm_fp4_block_scale_routed_moe", source)
        self.assertIn("prepare_static_weights_for_trtllm_fp4_moe", source)
        self.assertIn("unswizzle_sf", source)
        self.assertIn("experts.process_weights_after_loading(owner)", source)
        self.assertIn("clamp_raw_space_match", source)
        self.assertIn("numeric_metrics_pass", source)
        self.assertIn("capture_graph", source)
        self.assertNotIn("torch.ops.trtllm.block_scale_interleave_reverse", source)


if __name__ == "__main__":
    unittest.main()
