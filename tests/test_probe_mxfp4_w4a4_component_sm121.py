# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from benchmarks import probe_mxfp4_w4a4_component_sm121 as probe


class NativeMxfp4W4A4ComponentTests(unittest.TestCase):
    def test_tp_rank_slices_partition_exact_native_shapes(self) -> None:
        rank0 = probe.tp_rank_slices(0)
        rank1 = probe.tp_rank_slices(1)

        self.assertEqual((rank0.w13_rows.start, rank0.w13_rows.stop), (0, 1024))
        self.assertEqual((rank1.w13_rows.start, rank1.w13_rows.stop), (1024, 2048))
        self.assertEqual((rank0.w2_packed_k.start, rank0.w2_packed_k.stop), (0, 512))
        self.assertEqual((rank1.w2_packed_k.start, rank1.w2_packed_k.stop), (512, 1024))
        self.assertEqual((rank0.w2_scale_k.start, rank0.w2_scale_k.stop), (0, 32))
        self.assertEqual((rank1.w2_scale_k.start, rank1.w2_scale_k.stop), (32, 64))
        with self.assertRaises(ValueError):
            probe.tp_rank_slices(2)

    def test_e2m1_and_e8m0_cpu_contract(self) -> None:
        positives = [probe.e2m1_code_to_float(code) for code in range(8)]
        negatives = [probe.e2m1_code_to_float(code) for code in range(8, 16)]

        self.assertEqual(positives, [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        self.assertEqual(negatives, [-value for value in positives])
        self.assertEqual(probe.e8m0_byte_to_float(127), 1.0)
        self.assertEqual(probe.e8m0_byte_to_float(128), 2.0)
        self.assertEqual(probe.e8m0_byte_to_float(126), 0.5)

    def test_e8m0_k32_scale_clamp_matches_bf16_serving_contract(self) -> None:
        self.assertEqual(probe.E8M0_K32_BF16_MAX_SCALE_BYTE, 247)
        self.assertEqual(probe.clamp_e8m0_scale_byte_for_bf16(0), 0)
        self.assertEqual(probe.clamp_e8m0_scale_byte_for_bf16(247), 247)
        self.assertEqual(probe.clamp_e8m0_scale_byte_for_bf16(248), 247)
        self.assertEqual(probe.clamp_e8m0_scale_byte_for_bf16(255), 247)
        with self.assertRaises(ValueError):
            probe.clamp_e8m0_scale_byte_for_bf16(256)

    def test_balanced_routes_are_unique_and_bounded(self) -> None:
        self.assertEqual(probe.balanced_route_experts(1), tuple(range(6)))
        self.assertEqual(probe.balanced_route_experts(4), tuple(range(24)))
        with self.assertRaises(ValueError):
            probe.balanced_route_experts(43)

    def test_index_discovery_requires_one_complete_layer_shard(self) -> None:
        root = "model.layers"
        names = probe._expected_names(root, 0)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            shard = model / "model-00002-of-00048.safetensors"
            shard.write_bytes(b"fixture")
            index = {
                "metadata": {"total_size": 1},
                "weight_map": {name: shard.name for name in names},
            }
            (model / "model.safetensors.index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )

            source = probe.discover_native_layer(
                model_dir=model, shard_file=None, layer=0
            )

        self.assertEqual(source.root, root)
        self.assertEqual(source.layer, 0)
        self.assertEqual(source.shard.name, shard.name)
        self.assertEqual(len(source.tensor_names), 256 * 3 * 2)
        self.assertRegex(source.index_sha256 or "", r"^[0-9a-f]{64}$")

    def test_index_discovery_rejects_split_or_missing_layer(self) -> None:
        names = list(probe._expected_names("layers", 0))
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            for shard_name in ("a.safetensors", "b.safetensors"):
                (model / shard_name).write_bytes(b"fixture")
            weight_map = {name: "a.safetensors" for name in names}
            weight_map[names[-1]] = "b.safetensors"
            (model / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": weight_map}), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "spans 2 shards"):
                probe.discover_native_layer(
                    model_dir=model, shard_file=None, layer=0
                )
            del weight_map[names[-1]]
            (model / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": weight_map}), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "contract drift"):
                probe.discover_native_layer(
                    model_dir=model, shard_file=None, layer=0
                )

    def test_performance_gate_is_absolute_and_fail_closed(self) -> None:
        boundary = probe.evaluate_performance_gate(
            {1: 0.2, 4: probe.DEFAULT_M4_MAX_MS},
            maximum_m4_ms=probe.DEFAULT_M4_MAX_MS,
        )
        slow = probe.evaluate_performance_gate(
            {4: probe.DEFAULT_M4_MAX_MS + 0.001},
            maximum_m4_ms=probe.DEFAULT_M4_MAX_MS,
        )

        self.assertTrue(boundary["passed"])
        self.assertFalse(slow["passed"])
        self.assertAlmostEqual(boundary["required_speedup"], 1.0 / 0.95)
        for samples in ({1: 0.1}, {4: math.nan}, {4: 0.0}):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                probe.evaluate_performance_gate(
                    samples, maximum_m4_ms=probe.DEFAULT_M4_MAX_MS
                )

    def test_parser_pins_m1_m4_and_real_source_choice(self) -> None:
        args = probe.build_parser().parse_args(
            ["--model-dir", "/models/native", "--output", "/tmp/result.json"]
        )

        self.assertEqual(args.m, (1, 4))
        self.assertEqual(args.layer, 0)
        self.assertEqual(args.tp_rank, 0)
        self.assertAlmostEqual(args.max_m4_ms, 0.7429)
        with self.assertRaises(SystemExit):
            probe.build_parser().parse_args(["--output", "/tmp/result.json"])

    def test_source_contains_two_k32_quantize_and_gemm_stages(self) -> None:
        source = Path(probe.__file__).read_text(encoding="utf-8")

        self.assertGreaterEqual(source.count("flashinfer.mxfp4_quantize"), 5)
        self.assertGreaterEqual(source.count("block_size=MXFP4_BLOCK_SIZE"), 3)
        self.assertIn('"component_only": True', source)
        self.assertIn('"serving_integration_changed": False', source)
        self.assertIn('"source_above_bf16_safe"', source)
        self.assertIn("E8M0_K32_BF16_MAX_SCALE_BYTE", source)
        self.assertIn('"quantizer_produced_b"', source)
        self.assertIn('"checkpoint_native_b_interleaved_sf"', source)
        self.assertIn('"checkpoint_native_b_raw_sf"', source)
        self.assertIn('"checkpoint_dequant_requantized_b"', source)

    def test_numeric_gate_rejects_nonfinite_zero_and_low_similarity(self) -> None:
        good = {
            "finite": True,
            "actual_nonzero": 1,
            "reference_nonzero": 1,
            "cosine": 0.99,
            "normalized_rmse": 0.1,
        }
        self.assertTrue(
            probe._numeric_gate_passed(
                good, minimum_cosine=0.98, maximum_nrmse=0.25
            )
        )
        for field, value in (
            ("finite", False),
            ("actual_nonzero", 0),
            ("reference_nonzero", 0),
            ("cosine", 0.5),
            ("normalized_rmse", math.inf),
        ):
            bad = {**good, field: value}
            with self.subTest(field=field):
                self.assertFalse(
                    probe._numeric_gate_passed(
                        bad, minimum_cosine=0.98, maximum_nrmse=0.25
                    )
                )


if __name__ == "__main__":
    unittest.main()
