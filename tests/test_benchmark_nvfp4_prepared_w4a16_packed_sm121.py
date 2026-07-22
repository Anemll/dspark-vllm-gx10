# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
import pathlib
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import benchmark_nvfp4_a4w4_sm121 as kernel_bench  # noqa: E402
from benchmarks import (  # noqa: E402
    benchmark_nvfp4_prepared_w4a16_packed_sm121 as bench,
)


class PreparedPackedW4A16ComparatorTests(unittest.TestCase):
    def test_default_gate_is_the_measured_serving_gap(self) -> None:
        args = bench.build_parser().parse_args(
            ["--layer-file", "layer.safetensors", "--output", "result.json"]
        )

        self.assertAlmostEqual(args.min_m4_speedup, 1.1307124069289936)
        self.assertEqual(args.max_m4_latency_ms, 0.682812)
        self.assertEqual(args.m, (1, 4))
        self.assertEqual(
            args.min_m4_speedup,
            bench.REFERENCE_W4A4_M4_MS / bench.GAP_CLOSING_M4_MAX_MS,
        )

    def test_performance_gate_requires_speedup_and_absolute_latency(self) -> None:
        boundary = bench.evaluate_performance_gate(
            {1: 1.0, 4: bench.GAP_CLOSING_M4_MIN_SPEEDUP},
            {1: 0.2, 4: bench.GAP_CLOSING_M4_MAX_MS},
            required_m4_speedup=bench.GAP_CLOSING_M4_MIN_SPEEDUP,
            maximum_m4_latency_ms=bench.GAP_CLOSING_M4_MAX_MS,
        )
        slow_relative = bench.evaluate_performance_gate(
            {4: bench.GAP_CLOSING_M4_MIN_SPEEDUP - 0.001},
            {4: bench.GAP_CLOSING_M4_MAX_MS - 0.001},
            required_m4_speedup=bench.GAP_CLOSING_M4_MIN_SPEEDUP,
            maximum_m4_latency_ms=bench.GAP_CLOSING_M4_MAX_MS,
        )
        slow_absolute = bench.evaluate_performance_gate(
            {4: bench.GAP_CLOSING_M4_MIN_SPEEDUP + 0.001},
            {4: bench.GAP_CLOSING_M4_MAX_MS + 0.001},
            required_m4_speedup=bench.GAP_CLOSING_M4_MIN_SPEEDUP,
            maximum_m4_latency_ms=bench.GAP_CLOSING_M4_MAX_MS,
        )

        self.assertTrue(boundary["passed"])
        self.assertTrue(boundary["speedup_passed"])
        self.assertTrue(boundary["latency_passed"])
        self.assertFalse(slow_relative["passed"])
        self.assertFalse(slow_relative["speedup_passed"])
        self.assertTrue(slow_relative["latency_passed"])
        self.assertFalse(slow_absolute["passed"])
        self.assertTrue(slow_absolute["speedup_passed"])
        self.assertFalse(slow_absolute["latency_passed"])

    def test_performance_gate_rejects_missing_or_nonfinite_measurements(self) -> None:
        cases = (
            ({1: 1.0}, {1: 0.2}),
            ({4: math.nan}, {4: 0.2}),
            ({4: 1.2}, {4: math.inf}),
        )
        for speedups, latencies in cases:
            with self.subTest(speedups=speedups, latencies=latencies):
                with self.assertRaises(ValueError):
                    bench.evaluate_performance_gate(
                        speedups,
                        latencies,
                        required_m4_speedup=bench.GAP_CLOSING_M4_MIN_SPEEDUP,
                        maximum_m4_latency_ms=bench.GAP_CLOSING_M4_MAX_MS,
                    )

    def test_saved_balanced_hardware_result_remains_a_negative(self) -> None:
        gate = bench.evaluate_performance_gate(
            {1: 0.9841462537298479, 4: 1.0378368052087368},
            {1: 0.19780799746513367, 4: 0.7372719943523407},
            required_m4_speedup=bench.GAP_CLOSING_M4_MIN_SPEEDUP,
            maximum_m4_latency_ms=bench.GAP_CLOSING_M4_MAX_MS,
        )

        self.assertFalse(gate["speedup_passed"])
        self.assertFalse(gate["latency_passed"])
        self.assertFalse(gate["passed"])

    def test_direct_output_proof_separates_timed_and_serving_copy_contracts(self) -> None:
        source = {
            "implementation": "flashinfer.B12xMoEWrapper",
            "serving_adapter_output_copy": True,
        }
        proof = bench.direct_output_backend_proof(source)

        self.assertNotIn("serving_adapter_output_copy", proof)
        self.assertEqual(
            proof["reference_serving_output_contract"],
            {
                "adapter_full_tensor_copy_count": 1,
                "included_in_timed_launch": False,
            },
        )
        self.assertEqual(proof["timed_output_contract"]["full_tensor_copy_count"], 0)
        self.assertTrue(
            proof["timed_output_contract"]["pointer_identity_checked_each_launch"]
        )
        self.assertEqual(source["serving_adapter_output_copy"], True)
        with self.assertRaisesRegex(ValueError, "serving output-copy"):
            bench.direct_output_backend_proof({"serving_adapter_output_copy": False})

    def test_main_harness_parser_keeps_modelopt_default_and_accepts_packed(self) -> None:
        parser = kernel_bench.build_parser()
        self.assertEqual(parser.parse_args(["--dry-run"]).w4a16_weight_layout, "modelopt")
        self.assertEqual(
            parser.parse_args(
                ["--dry-run", "--w4a16-weight-layout", "packed"]
            ).w4a16_weight_layout,
            "packed",
        )

    def test_prepare_w4a16_selects_packed_or_zero_copy_preparer(self) -> None:
        calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

        class Tensor:
            def __init__(self, pointer: int) -> None:
                self.pointer = pointer

            def data_ptr(self) -> int:
                return self.pointer

        def prepare(name: str):
            def inner(*args: object, **kwargs: object) -> object:
                calls.append((name, args, kwargs))
                source_w13 = args[0]
                source_w2 = args[3]
                packed = name == "packed"
                return SimpleNamespace(
                    w13=Tensor(1000) if packed else source_w13,
                    w2=Tensor(2000) if packed else source_w2,
                    weight_layout="packed" if packed else "modelopt_native",
                    scale_format="e8m0_k32" if packed else "modelopt_nvfp4",
                    source_format="modelopt_nvfp4",
                    w13_layout="w13",
                )

            return inner

        fake_prepare = ModuleType("b12x.moe.fused.w4a16.prepare")
        fake_prepare.prepare_w4a16_modelopt_native_weights = prepare("modelopt")
        fake_prepare.prepare_w4a16_modelopt_nvfp4_weights = prepare("packed")
        weights = SimpleNamespace(
            w13=Tensor(10),
            w13_sf_swizzled=Tensor(11),
            alpha1=Tensor(12),
            w2=Tensor(20),
            w2_sf_swizzled=Tensor(21),
            alpha2=Tensor(22),
        )
        torch = SimpleNamespace(bfloat16="bf16")

        with mock.patch.dict(
            sys.modules,
            {"b12x.moe.fused.w4a16.prepare": fake_prepare},
        ):
            native, native_proof = kernel_bench._prepare_w4a16(
                torch,
                weights,
                SimpleNamespace(
                    w4a16_weight_layout="modelopt", swiglu_limit=10.0
                ),
            )
            packed, packed_proof = kernel_bench._prepare_w4a16(
                torch,
                weights,
                SimpleNamespace(
                    w4a16_weight_layout="packed", swiglu_limit=10.0
                ),
            )

        self.assertEqual([call[0] for call in calls], ["modelopt", "packed"])
        self.assertEqual(calls[0][2]["source_format"], "modelopt_nvfp4")
        self.assertNotIn("source_format", calls[1][2])
        for _, _, kwargs in calls:
            self.assertEqual(kwargs["activation"], "silu")
            self.assertEqual(kwargs["params_dtype"], "bf16")
            self.assertEqual(kwargs["w13_layout"], "w13")
        self.assertIs(native.w13, weights.w13)
        self.assertTrue(native_proof["same_source_w13"])
        self.assertEqual(native_proof["requested_weight_layout"], "modelopt")
        self.assertIsNot(packed.w13, weights.w13)
        self.assertFalse(packed_proof["same_source_w13"])
        self.assertEqual(packed_proof["requested_weight_layout"], "packed")


if __name__ == "__main__":
    unittest.main()
