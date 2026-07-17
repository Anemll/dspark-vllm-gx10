# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from __future__ import annotations

import json
import math
import pathlib
import subprocess
import sys
import tempfile
import unittest


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
            for suffix in ("weight", "weight_scale", "weight_scale_2"):
                key = f"{prefix}.{projection}.{suffix}"
                if key != omit:
                    weight_map[key] = "model-00001-of-00001.safetensors"
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map})
    )


class Nvfp4A4W4Sm121HarnessTests(unittest.TestCase):
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

    def test_w4a4_tactic_boundaries_for_dsv4_topk(self) -> None:
        cases = [(1, "micro"), (4, "micro"), (8, "static"), (64, "static"), (128, "dynamic")]
        for m, expected in cases:
            with self.subTest(m=m):
                self.assertEqual(bench.tactic_for_shape("w4a4", m, top_k=6), expected)

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

    def test_input_rms_must_be_positive_and_finite(self) -> None:
        parser = bench.build_parser()
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value):
                args = parser.parse_args(
                    ["--dry-run", "--synthetic", "--input-rms", value]
                )
                with self.assertRaisesRegex(ValueError, "input-rms"):
                    bench.validate_args(args)


if __name__ == "__main__":
    unittest.main()
