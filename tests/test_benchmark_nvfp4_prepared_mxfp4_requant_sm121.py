#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import inspect
import math
import struct
import unittest
from types import SimpleNamespace

from benchmarks import benchmark_nvfp4_prepared_mxfp4_requant_sm121 as probe
from scripts import patch_b12x_w4a16_modelopt_tc_decode as tc_patch


def _float32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


class ExactMxfp4CollapseTests(unittest.TestCase):
    def test_decision_defaults_are_real_layer_m1_m4_balanced(self) -> None:
        args = probe.build_parser().parse_args(
            ["--layer-file", "layer.safetensors", "--output", "result.json"]
        )
        self.assertEqual(args.m, (1, 4))
        self.assertEqual(args.routing, "balanced")
        self.assertEqual(args.numeric_min_cosine, 0.98)
        self.assertEqual(args.numeric_max_nrmse, 0.25)
        self.assertEqual(probe.MAXIMUM_M4_LATENCY_MS, 0.682812)

    def test_real_rank_shapes(self) -> None:
        shapes = probe.expected_conversion_shapes(
            SimpleNamespace(
                num_experts=256,
                hidden_size=4096,
                intermediate_size_per_rank=1024,
            )
        )
        self.assertEqual(shapes["w13.weight"], (256, 2048, 2048))
        self.assertEqual(shapes["w2.weight"], (256, 4096, 512))
        self.assertEqual(shapes["native_w13_e8m0"], (256, 2048, 128))
        self.assertEqual(shapes["native_w2_e8m0"], (256, 4096, 32))

    def test_global_scale_canonicalization_accepts_one_ulp_only(self) -> None:
        canonical = math.ldexp(1.0, -13)
        bits = struct.unpack("<I", struct.pack("<f", canonical))[0]
        one_below = _float32_from_bits(bits - 1)
        two_below = _float32_from_bits(bits - 2)
        self.assertEqual(probe.canonical_power_of_two(canonical), (-13, canonical, 0))
        self.assertEqual(
            probe.canonical_power_of_two(one_below), (-13, canonical, 1)
        )
        with self.assertRaisesRegex(ValueError, "ulp_distance=2"):
            probe.canonical_power_of_two(two_below)

    def test_e4m3_power_of_two_decode(self) -> None:
        self.assertEqual(probe.e4m3fn_power_of_two_exponent(0x60), 5)
        self.assertEqual(probe.e4m3fn_power_of_two_exponent(0x68), 6)
        self.assertEqual(probe.e4m3fn_power_of_two_exponent(0x78), 8)
        self.assertEqual(probe.e4m3fn_power_of_two_exponent(0x01), -9)
        self.assertEqual(probe.e4m3fn_power_of_two_exponent(0x02), -8)
        self.assertEqual(probe.e4m3fn_power_of_two_exponent(0x04), -7)
        for invalid in (0x00, 0x03, 0x61, 0x80):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                probe.e4m3fn_power_of_two_exponent(invalid)

    def test_exact_pair_collapse_matches_native_e8m0_bytes(self) -> None:
        # E4M3 32/64 times 2**-13 => 2**-8 / 2**-7, encoded as
        # E8M0 exponent+bias bytes 119/120.
        self.assertEqual(
            probe.collapse_e4m3_pairs_cpu(
                bytes((0x60, 0x60, 0x68, 0x68)), math.ldexp(1.0, -13)
            ),
            bytes((119, 120)),
        )

    def test_pair_collapse_rejects_loss_or_non_power_scale(self) -> None:
        with self.assertRaisesRegex(ValueError, "adjacent K16 scales differ"):
            probe.collapse_e4m3_pairs_cpu(
                bytes((0x60, 0x68)), math.ldexp(1.0, -13)
            )
        with self.assertRaisesRegex(ValueError, "not a power of two"):
            probe.collapse_e4m3_pairs_cpu(
                bytes((0x61, 0x61)), math.ldexp(1.0, -13)
            )
        with self.assertRaisesRegex(ValueError, "out of range"):
            probe.collapse_e4m3_pairs_cpu(
                bytes((0x78, 0x78)), math.ldexp(1.0, 120)
            )

    def test_direct_trace_requires_both_native_e8m0_shapes(self) -> None:
        base = {
            "hidden_size": 4096,
            "intermediate_size": 1024,
            "num_experts": 256,
            "topk": 6,
            "activation": "silu",
            "scale_format": "e8m0_k32",
            "w13_layout": "w13",
            "grid_x": 1,
        }
        result = probe.evaluate_small_m_trace(
            [base | {"m": 1}, base | {"m": 4}]
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["passing_m"], [1, 4])
        self.assertFalse(
            probe.evaluate_small_m_trace(
                [base | {"m": 1}, base | {"m": 4, "scale_format": "e4m3_k16"}]
            )["passed"]
        )

    def test_performance_gate_is_exact_absolute_boundary(self) -> None:
        self.assertTrue(
            probe.evaluate_performance_gate({1: 0.2, 4: 0.682812})["passed"]
        )
        self.assertFalse(
            probe.evaluate_performance_gate({1: 0.2, 4: 0.682813})["passed"]
        )
        with self.assertRaises(ValueError):
            probe.evaluate_performance_gate({1: 0.2})

    def test_runtime_kernel_pin_is_exact_modelopt_tc_patch(self) -> None:
        self.assertEqual(
            probe.PINNED_SOURCE_SHA256["b12x.moe.fused.w4a16.kernel"][1],
            tc_patch.PATCHED_SOURCE_SHA256,
        )

    def test_source_has_no_weight_dequant_or_requant_path(self) -> None:
        source = inspect.getsource(probe)
        for forbidden in (
            "def decode_e2m1",
            "def dequantize_modelopt_projection",
            "def requantize_projection_mxfp4",
            "mxfp4_quantize(",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn('"weight_payload_transform": "none"', source)
        self.assertIn('"duplicate_weight_bytes": 0', source)
        self.assertIn('"B12X_W4A16_SMALL_M_DIRECT"] = "1" if direct else "0"', source)
        self.assertIn('"B12X_W4A16_TC_DECODE"] = "0" if direct else "1"', source)

    def test_direct_numeric_is_diagnostic_not_a_promotion_gate(self) -> None:
        source = inspect.getsource(probe.run)
        self.assertIn(
            'if not passed and label == f"{CANDIDATE}_vs_w4a4":',
            source,
        )


if __name__ == "__main__":
    unittest.main()
