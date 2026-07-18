# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from __future__ import annotations

import inspect
import json
import math
import pathlib
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from types import SimpleNamespace


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import benchmark_nvfp4_a4w4_sm121 as bench  # noqa: E402


def _write_checkpoint_contract(path: pathlib.Path, *, omit: str | None = None) -> None:
    config = {
        "model_type": "deepseek_v4",
        "expert_dtype": "fp4",
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "swiglu_limit": 10.0,
        "quantization_config": {
            "group_size": 16,
            "moe_quant_algo": "NVFP4",
            "producer": {"name": "modelopt", "version": "test"},
        },
    }
    (path / "config.json").write_text(json.dumps(config))
    weight_map: dict[str, str] = {}
    for expert_id in (0, 255):
        prefix = f"layers.0.ffn.experts.{expert_id}"
        for projection in ("w1", "w3", "w2"):
            for suffix in (
                "weight",
                "weight_scale",
                "weight_scale_2",
                "input_scale",
            ):
                key = f"{prefix}.{projection}.{suffix}"
                if key != omit:
                    weight_map[key] = "model-00001-of-00001.safetensors"
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map})
    )


class Nvfp4A4W4Sm121HarnessTests(unittest.TestCase):
    def test_b12x_timing_mirrors_serving_adapter_output_copy(self) -> None:
        source = inspect.getsource(bench.run_benchmark)
        self.assertIn("output_local.copy_(wrapper_output)", source)

    def test_parse_positive_int_csv_preserves_order_and_deduplicates(self) -> None:
        self.assertEqual(bench.parse_positive_int_csv("128, 1,128,8"), (128, 1, 8))
        with self.assertRaisesRegex(Exception, "positive"):
            bench.parse_positive_int_csv("1,0")

    def test_percentile_and_timing_summary(self) -> None:
        self.assertAlmostEqual(bench.percentile([0.0, 10.0], 0.95), 9.5)
        stats = bench.summarize_timing_runs([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]])
        self.assertEqual(stats["samples"], 6)
        self.assertEqual(stats["repeats"], 2)
        self.assertAlmostEqual(stats["median_ms"], 2.5)
        self.assertAlmostEqual(stats["p95_ms"], 7.0)
        self.assertEqual(stats["repeat_median_ms"], [2.0, 4.0])

    def test_w4a4_crossover_summary_records_switch_without_assuming_winner(self) -> None:
        def row(m: int, b12x_ms: float, cutlass_ms: float) -> dict[str, object]:
            return {
                "m": m,
                "phase": bench.phase_for_m(m),
                "modes": {
                    "w4a4": {"eager": {"median_ms": b12x_ms}},
                    bench.FLASHINFER_CUTLASS_MODE: {
                        "eager": {"median_ms": cutlass_ms}
                    },
                },
            }

        summary = bench.summarize_w4a4_backend_crossover(
            [row(1, 1.0, 2.0), row(64, 3.0, 2.0)], "eager"
        )
        self.assertTrue(summary["crossover_observed"])
        self.assertEqual(
            [item["preferred_backend"] for item in summary["rows"]],
            ["flashinfer_b12x", bench.FLASHINFER_CUTLASS_MODE],
        )
        self.assertEqual(
            summary["switch_points"],
            [
                {
                    "m": 64,
                    "from": "flashinfer_b12x",
                    "to": bench.FLASHINFER_CUTLASS_MODE,
                }
            ],
        )

    def test_w4a4_tactic_boundaries_for_dsv4_topk(self) -> None:
        cases = [(1, "micro"), (4, "micro"), (8, "static"), (64, "static"), (128, "dynamic")]
        for m, expected in cases:
            with self.subTest(m=m):
                self.assertEqual(bench.tactic_for_shape("w4a4", m, top_k=6), expected)

    def test_backend_selections_preserve_legacy_and_add_explicit_w4a4_ab(self) -> None:
        self.assertEqual(bench.modes_for_backend("both"), ("w4a4", "w4a16"))
        self.assertEqual(
            bench.modes_for_backend("w4a4-ab"),
            ("w4a4", bench.FLASHINFER_CUTLASS_MODE),
        )
        self.assertEqual(
            bench.modes_for_backend("all"),
            ("w4a4", bench.FLASHINFER_CUTLASS_MODE, "w4a16"),
        )
        self.assertEqual(
            bench.order_modes(
                bench.modes_for_backend("all"), "cutlass-first"
            ),
            (bench.FLASHINFER_CUTLASS_MODE, "w4a4", "w4a16"),
        )
        self.assertEqual(
            bench.order_modes(bench.modes_for_backend("both"), "cutlass-first"),
            ("w4a4", "w4a16"),
        )
        self.assertEqual(
            bench.tactic_for_shape(bench.FLASHINFER_CUTLASS_MODE, 8192, 6),
            "flashinfer-cutlass",
        )
        with self.assertRaisesRegex(ValueError, "unsupported backend"):
            bench.modes_for_backend("unknown")

    def test_modelopt_cutlass_scale_contract_matches_pinned_oracle_algebra(self) -> None:
        a_gscale, g_alpha = bench.modelopt_cutlass_scale_contract(
            weight_scale_2=0.125,
            input_scale=32.0,
        )
        self.assertEqual(a_gscale, 1.0 / 32.0)
        self.assertEqual(g_alpha, 4.0)
        self.assertEqual(a_gscale * g_alpha, 0.125)

    def test_input_rms_contract_gates_per_token_extremes(self) -> None:
        passing = bench.evaluate_input_rms_contract(
            requested=1.0,
            observed_mean=1.0,
            observed_min=0.999,
            observed_max=1.001,
        )
        too_small = bench.evaluate_input_rms_contract(
            requested=1.0,
            observed_mean=0.50,
            observed_min=0.49,
            observed_max=0.51,
        )
        nonfinite = bench.evaluate_input_rms_contract(
            requested=1.0,
            observed_mean=math.inf,
            observed_min=1.0,
            observed_max=math.inf,
        )

        self.assertTrue(passing["passed"])
        self.assertFalse(too_small["passed"])
        self.assertFalse(nonfinite["passed"])
        self.assertAlmostEqual(passing["maximum_relative_error"], 0.001)
        json.dumps(nonfinite, allow_nan=False)

    def test_numeric_metrics_gate_rejects_graph_corruption(self) -> None:
        passing = {
            "finite": True,
            "cosine": 0.99,
            "normalized_rmse": 0.20,
        }
        corrupt = passing | {"cosine": 0.10, "normalized_rmse": 1.25}
        nonfinite = passing | {"cosine": math.nan}

        self.assertTrue(
            bench.numeric_metrics_pass(
                passing,
                min_cosine=0.98,
                max_normalized_rmse=0.25,
            )
        )
        for metrics in (corrupt, nonfinite, passing | {"finite": False}):
            with self.subTest(metrics=metrics):
                self.assertFalse(
                    bench.numeric_metrics_pass(
                        metrics,
                        min_cosine=0.98,
                        max_normalized_rmse=0.25,
                    )
                )

        self.assertFalse(
            bench.numeric_metrics_pass(
                passing | {"nonzero_activity": False},
                min_cosine=0.98,
                max_normalized_rmse=0.25,
            )
        )

    def test_tensor_comparison_handles_zero_and_tiny_vectors_without_underflow(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")

        zeros = torch.zeros(8, dtype=torch.float32)
        zero_metrics = bench.compare_tensors(torch, zeros, zeros.clone())
        self.assertEqual(zero_metrics["cosine"], 1.0)
        self.assertEqual(zero_metrics["normalized_rmse"], 0.0)
        self.assertFalse(zero_metrics["nonzero_activity"])
        self.assertFalse(
            bench.numeric_metrics_pass(
                zero_metrics,
                min_cosine=0.98,
                max_normalized_rmse=0.25,
            )
        )

        tiny = torch.full((8,), 1.0e-30, dtype=torch.float32)
        tiny_metrics = bench.compare_tensors(torch, tiny, tiny.clone())
        self.assertAlmostEqual(float(tiny_metrics["cosine"]), 1.0, places=6)
        self.assertEqual(tiny_metrics["normalized_rmse"], 0.0)
        self.assertTrue(tiny_metrics["nonzero_activity"])
        self.assertTrue(
            bench.numeric_metrics_pass(
                tiny_metrics,
                min_cosine=0.98,
                max_normalized_rmse=0.25,
            )
        )

        mismatch = bench.compare_tensors(torch, zeros, torch.ones_like(zeros))
        self.assertEqual(mismatch["cosine"], 0.0)
        self.assertFalse(mismatch["nonzero_activity"])

    def test_output_activity_rejects_zero_and_nan_poison(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")

        zeros = bench.tensor_activity(torch, torch.zeros(4))
        active = bench.tensor_activity(torch, torch.tensor([0.0, 1.0, -2.0]))
        poisoned = bench.tensor_activity(torch, torch.full((4,), math.nan))
        self.assertFalse(zeros["passed"])
        self.assertEqual(zeros["nonzero_count"], 0)
        self.assertTrue(active["passed"])
        self.assertFalse(poisoned["passed"])
        self.assertEqual(poisoned["nonfinite_count"], 4)

    def test_graph_vs_eager_metrics_are_enforced_as_numeric_failures(self) -> None:
        source = inspect.getsource(bench.run_benchmark)
        self.assertIn('"comparison": "graph_vs_eager"', source)
        self.assertIn('mode_result["graph_numeric_gate_passed"]', source)
        self.assertIn("output.fill_(math.nan)", source)
        self.assertIn("graph_output.fill_(math.nan)", source)
        self.assertIn('"kind": "output_activity"', source)

    def test_workspace_storage_summary_deduplicates_tensor_views(self) -> None:
        class _Storage:
            def __init__(self, pointer: int, size: int) -> None:
                self.pointer = pointer
                self.size = size

            def data_ptr(self) -> int:
                return self.pointer

            def nbytes(self) -> int:
                return self.size

        class _Tensor:
            def __init__(self, storage: _Storage) -> None:
                self.storage = storage
                self.device = "cuda:0"

            def untyped_storage(self) -> _Storage:
                return self.storage

        @dataclass
        class _Workspace:
            original: _Tensor
            view: _Tensor
            independent: _Tensor

        shared_storage = _Storage(100, 64)
        workspace = _Workspace(
            original=_Tensor(shared_storage),
            view=_Tensor(shared_storage),
            independent=_Tensor(_Storage(200, 32)),
        )

        summary = bench.summarize_unique_tensor_storage(
            SimpleNamespace(is_tensor=lambda value: isinstance(value, _Tensor)),
            (workspace,),
        )

        self.assertEqual(summary["tensor_object_count"], 3)
        self.assertEqual(summary["unique_storage_count"], 2)
        self.assertEqual(summary["unique_storage_bytes"], 96)

    def test_workspace_ceiling_requires_exact_tp2_geometry(self) -> None:
        breakdown = bench.calculate_dsv4_tp2_m8192_workspace_bytes()
        self.assertEqual(breakdown["static_workspace_bytes"], 378_803_724)
        self.assertEqual(breakdown["dynamic_workspace_bytes"], 189_231_452)
        self.assertEqual(breakdown["output_bytes"], 67_108_864)
        self.assertEqual(
            breakdown["total_bytes"],
            bench.DSV4_TP2_M8192_B12X_WRAPPER_CEILING_BYTES,
        )
        self.assertEqual(
            bench.b12x_workspace_ceiling_bytes(bench.Dsv4Shape(), 8192),
            bench.DSV4_TP2_M8192_B12X_WRAPPER_CEILING_BYTES,
        )
        self.assertIsNone(
            bench.b12x_workspace_ceiling_bytes(
                bench.Dsv4Shape(intermediate_size=1024, tp_size=1),
                8192,
            )
        )
        self.assertIsNone(
            bench.b12x_workspace_ceiling_bytes(bench.Dsv4Shape(), 4096)
        )

    def test_default_matrix_and_phase_boundary(self) -> None:
        self.assertEqual(bench.B12X_W13_LAYOUT, "w13")
        self.assertEqual(
            bench.DEFAULT_M_VALUES,
            (
                1,
                2,
                4,
                6,
                12,
                24,
                48,
                64,
                72,
                128,
                256,
                512,
                1024,
                2048,
                4096,
                8192,
            ),
        )
        self.assertLessEqual(
            set(bench.DEFAULT_CORRECTNESS_M), set(bench.DEFAULT_M_VALUES)
        )
        self.assertEqual(bench.phase_for_m(1), "decode")
        self.assertEqual(bench.phase_for_m(127), "decode")
        self.assertEqual(bench.phase_for_m(128), "prefill")
        self.assertEqual(bench.phase_for_m(8192), "prefill")

    def test_checkpoint_contract_reads_real_dsv4_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_checkpoint_contract(root)
            shape, metadata = bench.read_checkpoint_contract(
                root, layer_idx=0, tp_size=2, tp_rank=1
            )
        self.assertEqual(shape, bench.Dsv4Shape(tp_rank=1))
        self.assertEqual(shape.intermediate_size_per_rank, 1024)
        self.assertEqual(metadata["moe_quant_algo"], "NVFP4")
        self.assertEqual(metadata["indexed_shard_count"], 1)
        self.assertEqual(len(metadata["config_sha256"]), 64)

    def test_checkpoint_contract_rejects_missing_expert_tensor(self) -> None:
        missing = "layers.0.ffn.experts.255.w2.weight_scale_2"
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_checkpoint_contract(root, omit=missing)
            with self.assertRaisesRegex(KeyError, "required tensors"):
                bench.read_checkpoint_contract(
                    root, layer_idx=0, tp_size=2, tp_rank=0
                )

    def test_checkpoint_contract_requires_modelopt_activation_scales(self) -> None:
        missing = "layers.0.ffn.experts.255.w3.input_scale"
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_checkpoint_contract(root, omit=missing)
            with self.assertRaisesRegex(KeyError, "required tensors"):
                bench.read_checkpoint_contract(
                    root, layer_idx=0, tp_size=2, tp_rank=0
                )

    def test_legacy_checkpoint_contract_does_not_require_cutlass_scales(self) -> None:
        missing = "layers.0.ffn.experts.255.w3.input_scale"
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            _write_checkpoint_contract(root, omit=missing)
            _, metadata = bench.read_checkpoint_contract(
                root,
                layer_idx=0,
                tp_size=2,
                tp_rank=0,
                require_input_scales=False,
            )
        self.assertFalse(metadata["input_scales_required"])

    def test_non_cuda_dry_run_cli_is_valid_json(self) -> None:
        script = ROOT / "benchmarks" / "benchmark_nvfp4_a4w4_sm121.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--dry-run",
                "--synthetic",
                "--synthetic-experts",
                "8",
                "--m",
                "1,128",
                "--correctness-m",
                "1,128",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(completed.stdout)
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["shape"]["hidden_size"], 4096)
        self.assertEqual(report["matrix"][0]["tactics"]["w4a4"], "micro")
        self.assertEqual(report["matrix"][1]["tactics"]["w4a4"], "dynamic")
        self.assertEqual(report["matrix"][0]["phase"], "decode")
        self.assertEqual(report["matrix"][1]["phase"], "prefill")
        self.assertEqual(report["activation_contract"]["input_rms"], 1.0)
        self.assertEqual(
            report["activation_contract"]["input_rms_relative_tolerance"],
            bench.INPUT_RMS_RELATIVE_TOLERANCE,
        )

    def test_non_cuda_dry_run_expands_flashinfer_w4a4_ab(self) -> None:
        script = ROOT / "benchmarks" / "benchmark_nvfp4_a4w4_sm121.py"
        completed = subprocess.run(
            [
                sys.executable,
                "-S",
                str(script),
                "--dry-run",
                "--synthetic",
                "--synthetic-experts",
                "8",
                "--backend",
                "w4a4-ab",
                "--w4a4-order",
                "cutlass-first",
                "--m",
                "1,128",
                "--correctness-m",
                "1,128",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(completed.stdout)
        self.assertEqual(
            report["modes"], [bench.FLASHINFER_CUTLASS_MODE, "w4a4"]
        )
        self.assertEqual(report["w4a4_order"], "cutlass-first")
        for row in report["matrix"]:
            self.assertEqual(
                row["tactics"][bench.FLASHINFER_CUTLASS_MODE],
                "flashinfer-cutlass",
            )
        self.assertEqual(
            report["activation_contract"][bench.FLASHINFER_CUTLASS_MODE],
            {
                "name": "silu",
                "weight_layout": "up_gate",
                "limit": 10.0,
                "activation_scale": (
                    "checkpoint input_scale max-reduced and expanded to E"
                ),
            },
        )

    def test_input_rms_must_be_positive_and_finite(self) -> None:
        parser = bench.build_parser()
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value):
                args = parser.parse_args(
                    ["--dry-run", "--synthetic", "--input-rms", value]
                )
                with self.assertRaisesRegex(ValueError, "input-rms"):
                    bench.validate_args(args)

    def test_cutlass_comparison_rejects_unmatched_swiglu_parameters(self) -> None:
        parser = bench.build_parser()
        args = parser.parse_args(
            [
                "--dry-run",
                "--synthetic",
                "--backend",
                "w4a4-ab",
                "--swiglu-alpha",
                "1.1",
            ]
        )
        with self.assertRaisesRegex(ValueError, "CUTLASS/W4A16"):
            bench.validate_args(args)


if __name__ == "__main__":
    unittest.main()
