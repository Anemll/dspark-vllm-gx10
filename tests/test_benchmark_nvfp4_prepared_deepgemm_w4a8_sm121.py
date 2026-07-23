#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import inspect
import unittest

from benchmarks import benchmark_nvfp4_prepared_deepgemm_w4a8_sm121 as probe


class _FakeTensor:
    ndim = 3

    def __init__(self, rows, grouped=False):
        self.rows = rows
        self.shape = (1, len(rows), 1)
        self.grouped = grouped

    def reshape(self, *shape):
        self.grouped = len(shape) == 4
        return self

    def flip(self, dims):
        if not self.grouped or tuple(dims) != (1,):
            raise AssertionError("unexpected fake tensor flip")
        half = len(self.rows) // 2
        return _FakeTensor(self.rows[half:] + self.rows[:half], grouped=True)

    def contiguous(self):
        return self


class DeepGemmW4A8GateTests(unittest.TestCase):
    def test_defaults_pin_real_dspark_shapes(self) -> None:
        args = probe.build_parser().parse_args(
            ["--layer-file", "layer.safetensors", "--output", "result.json"]
        )
        self.assertEqual(args.m, (1, 4, 24, 48))
        self.assertEqual(args.minimum_decision_speedup, 1.03)
        self.assertEqual(args.numeric_min_cosine, 0.98)
        self.assertEqual(args.numeric_max_nrmse, 0.25)

    def test_w13_half_swap_is_involution(self) -> None:
        original = _FakeTensor(["up0", "up1", "gate0", "gate1"])
        swapped = probe.swap_gate_up_halves(original)
        self.assertEqual(swapped.rows, ["gate0", "gate1", "up0", "up1"])
        restored = probe.swap_gate_up_halves(swapped)
        self.assertEqual(restored.rows, original.rows)

    def test_decision_requires_three_percent_at_both_shapes(self) -> None:
        base = {
            m: {
                "cutlass_graph_ms": 1.0,
                "deepgemm_graph_ms": 0.96,
                "numeric_passed": True,
                "graph_passed": True,
            }
            for m in probe.REQUIRED_M
        }
        self.assertTrue(probe.evaluate_decision_rows(base)["passed"])
        base[48]["deepgemm_graph_ms"] = 0.98
        self.assertFalse(probe.evaluate_decision_rows(base)["passed"])

    def test_m1_m4_are_diagnostic_only(self) -> None:
        rows = {
            m: {
                "cutlass_graph_ms": 1.0,
                "deepgemm_graph_ms": 0.96,
                "numeric_passed": True,
                "graph_passed": True,
            }
            for m in probe.REQUIRED_M
        }
        rows[1]["deepgemm_graph_ms"] = 10.0
        rows[4]["deepgemm_graph_ms"] = 10.0
        self.assertTrue(probe.evaluate_decision_rows(rows)["passed"])

    def test_source_encodes_lossless_and_w4a8_contracts(self) -> None:
        source = inspect.getsource(probe)
        self.assertIn('"weight_payload_transform": "projection-half swap only; no nibble change"', source)
        self.assertIn('"activation_precision": "dynamic FP8/K128 (W4A8)"', source)
        self.assertIn("collapse_nvfp4_scale_grid", source)
        self.assertIn("_pack_deepgemm_mxfp4_scales", source)
        self.assertIn("SWIGLUOAI_UNINTERLEAVE", source)
        run_source = inspect.getsource(probe.run)
        self.assertLess(
            run_source.index("deep_gemm_utils._lazy_init()"),
            run_source.index("DeepGemmQuantScaleFMT.from_oracle()"),
        )
        self.assertIn(
            "self, input: Any, output: Any, activation: Any",
            inspect.getsource(probe._make_deepgemm_runner),
        )


if __name__ == "__main__":
    unittest.main()
