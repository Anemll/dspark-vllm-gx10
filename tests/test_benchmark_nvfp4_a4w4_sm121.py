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
from unittest import mock


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
    def test_checkpoint_load_predictor_enforces_five_minute_reserve(self) -> None:
        passing = bench.checkpoint_load_prediction(4.0)
        boundary = bench.checkpoint_load_prediction(
            (bench.LOAD_PREDICTOR_DECISION_LIMIT_SECONDS - 60.0) / 43
        )
        failing = bench.checkpoint_load_prediction(5.0)

        self.assertEqual(passing["prototype_screening_total_seconds"], 232.0)
        self.assertTrue(passing["prototype_budget_passed"])
        self.assertTrue(boundary["prototype_budget_passed"])
        self.assertFalse(failing["prototype_budget_passed"])
        self.assertFalse(passing["serving_run_authorized"])
        self.assertEqual(passing["effective_non_layer_budget_seconds"], 90.0)
        self.assertEqual(passing["serving_target_seconds"], 300.0)
        self.assertAlmostEqual(
            passing["required_layer_seconds"], 210.0 / 43
        )
        with self.assertRaisesRegex(ValueError, "layer_seconds"):
            bench.checkpoint_load_prediction(math.inf)
        self.assertEqual(bench.parse_sha256("A" * 64), "a" * 64)
        with self.assertRaisesRegex(Exception, "64-character"):
            bench.parse_sha256("not-a-digest")

    def test_layer_staging_memory_contract_is_bounded_and_exact(self) -> None:
        shape = bench.Dsv4Shape()
        self.assertEqual(
            bench.expected_layer_staging_bytes(shape, prepare_cutlass=True),
            1_811_945_472,
        )
        self.assertEqual(
            bench.expected_layer_staging_bytes(shape, prepare_cutlass=False),
            1_811_942_400,
        )
        self.assertLess(
            bench.expected_layer_staging_bytes(shape, prepare_cutlass=True),
            1.7 * (1 << 30),
        )

    def test_checkpoint_load_memory_contract_rejects_swap_or_cuda_peak(self) -> None:
        before = {
            "available": True,
            "process_rss_kib": 100,
            "process_hwm_kib": 100,
            "mem_available_kib": 10_000_000,
            "swap_free_kib": 1_000,
        }
        healthy = before | {
            "process_rss_kib": 200,
            "process_hwm_kib": 300,
            "mem_available_kib": 9_000_000,
        }
        passing = bench.checkpoint_load_memory_contract(
            before,
            {"host_and_cuda_resident": healthy},
            healthy,
            cuda_peak_allocated_bytes=3 << 30,
            staged_host_bytes=1_811_945_472,
        )
        swapped = bench.checkpoint_load_memory_contract(
            before,
            {"host_and_cuda_resident": healthy | {"swap_free_kib": 999}},
            healthy,
            cuda_peak_allocated_bytes=3 << 30,
            staged_host_bytes=1_811_945_472,
        )
        oversized = bench.checkpoint_load_memory_contract(
            before,
            {"host_and_cuda_resident": healthy},
            healthy,
            cuda_peak_allocated_bytes=4 << 30,
            staged_host_bytes=1_811_945_472,
        )
        self.assertTrue(passing["passed"])
        self.assertFalse(swapped["passed"])
        self.assertFalse(oversized["passed"])
        self.assertEqual(passing["minimum_mem_available_kib"], 9_000_000)

    def test_checkpoint_reference_gate_compares_semantic_invariants(self) -> None:
        candidate = {
            "checkpoint_load_strategy": "per-expert",
            "requested_backend_selection": bench.FLASHINFER_CUTLASS_MODE,
            "config_sha256": "config",
            "index_sha256": "index",
            "layer_idx": 0,
            "tp_offset": 0,
            "sample_fingerprints": {
                "w13": "abc",
                "w2": "def",
                "w13_scale_modelopt_swizzled": "ghi",
                "w2_scale_modelopt_swizzled": "jkl",
                "cutlass_a1_gscale": "a1",
                "cutlass_a2_gscale": "a2",
                "cutlass_g1_alphas": "g1",
                "cutlass_g2_alphas": "g2",
            },
            "weight_preparation_contract": {
                "flashinfer_b12x": False,
                bench.FLASHINFER_CUTLASS_MODE: True,
                "required_sample_fingerprints": sorted(
                    bench.COMMON_SAMPLE_FINGERPRINTS
                    | bench.CUTLASS_SAMPLE_FINGERPRINTS
                ),
            },
            "checkpoint_input_scale_stats": {"w13_global_max": 1.0},
            "checkpoint_input_scale_tensor_count": 768,
            "w1_w3_scale2_max_mismatch": 0.0,
            "w13_layout": "w13 (up/w3, gate/w1; B12X up_gate)",
            "checkpoint_cache_evidence": {
                "method": "POSIX_FADV_DONTNEED",
                "layer_idx": 0,
            },
        }
        self.assertEqual(
            bench.checkpoint_reference_failures(
                candidate,
                {
                    "checkpoint": candidate.copy(),
                    "settings": {
                        "checkpoint_load_strategy": "per-expert",
                        "backend_selection": bench.FLASHINFER_CUTLASS_MODE,
                    },
                },
            ),
            [],
        )
        corrupt = candidate.copy()
        corrupt["sample_fingerprints"] = candidate["sample_fingerprints"] | {
            "w13": "wrong"
        }
        failures = bench.checkpoint_reference_failures(
            corrupt,
            {
                "checkpoint": candidate.copy(),
                "settings": {
                    "checkpoint_load_strategy": "per-expert",
                    "backend_selection": bench.FLASHINFER_CUTLASS_MODE,
                },
            },
        )
        self.assertEqual(failures[0]["kind"], "reference_mismatch")
        self.assertEqual(failures[0]["field"], "sample_fingerprints.w13")

        legacy_reference = candidate.copy()
        legacy_reference["sample_fingerprints"] = candidate[
            "sample_fingerprints"
        ] | {
            "w13_scale_b12x_baked_swizzled": "legacy-w13",
            "w2_scale_b12x_baked_swizzled": "legacy-w2",
        }
        self.assertEqual(
            bench.checkpoint_reference_failures(
                candidate,
                {
                    "checkpoint": legacy_reference,
                    "settings": {
                        "checkpoint_load_strategy": "per-expert",
                        "backend_selection": bench.FLASHINFER_CUTLASS_MODE,
                    },
                },
            ),
            [],
        )
        pre_global_reference = legacy_reference.copy()
        pre_global_reference["sample_fingerprints"] = {
            name: digest
            for name, digest in legacy_reference["sample_fingerprints"].items()
            if name
            not in {
                "cutlass_a1_gscale",
                "cutlass_a2_gscale",
                "cutlass_g1_alphas",
                "cutlass_g2_alphas",
            }
        }
        failures = bench.checkpoint_reference_failures(
            candidate,
            {
                "checkpoint": pre_global_reference,
                "settings": {
                    "checkpoint_load_strategy": "per-expert",
                    "backend_selection": bench.FLASHINFER_CUTLASS_MODE,
                },
            },
        )
        self.assertTrue(
            any(
                failure["kind"] == "reference_contract"
                and failure.get("field")
                == "sample_fingerprints.cutlass_g1_alphas"
                for failure in failures
            )
        )
        unexpected_candidate = candidate.copy()
        unexpected_candidate["sample_fingerprints"] = legacy_reference[
            "sample_fingerprints"
        ]
        failures = bench.checkpoint_reference_failures(
            unexpected_candidate,
            {
                "checkpoint": legacy_reference,
                "settings": {
                    "checkpoint_load_strategy": "per-expert",
                    "backend_selection": bench.FLASHINFER_CUTLASS_MODE,
                },
            },
        )
        self.assertEqual(failures[0]["kind"], "reference_contract")
        self.assertEqual(failures[0]["field"], "sample_fingerprints.keys")

        corrupt_global = candidate.copy()
        corrupt_global["sample_fingerprints"] = candidate[
            "sample_fingerprints"
        ] | {"cutlass_g1_alphas": "wrong"}
        failures = bench.checkpoint_reference_failures(
            corrupt_global,
            {
                "checkpoint": candidate.copy(),
                "settings": {
                    "checkpoint_load_strategy": "per-expert",
                    "backend_selection": bench.FLASHINFER_CUTLASS_MODE,
                },
            },
        )
        self.assertEqual(failures[0]["kind"], "reference_mismatch")
        self.assertEqual(
            failures[0]["field"], "sample_fingerprints.cutlass_g1_alphas"
        )

        dual_candidate = candidate.copy()
        dual_candidate["sample_fingerprints"] = candidate[
            "sample_fingerprints"
        ] | {
            "w13_scale_b12x_baked_swizzled": "dual-w13",
            "w2_scale_b12x_baked_swizzled": "dual-w2",
        }
        dual_candidate["weight_preparation_contract"] = {
            "flashinfer_b12x": True,
            bench.FLASHINFER_CUTLASS_MODE: True,
            "required_sample_fingerprints": sorted(
                bench.COMMON_SAMPLE_FINGERPRINTS
                | bench.B12X_SAMPLE_FINGERPRINTS
                | bench.CUTLASS_SAMPLE_FINGERPRINTS
            ),
        }
        failures = bench.checkpoint_reference_failures(
            dual_candidate,
            {
                "checkpoint": dual_candidate.copy(),
                "settings": {
                    "checkpoint_load_strategy": "per-expert",
                    "backend_selection": bench.FLASHINFER_CUTLASS_MODE,
                },
            },
        )
        self.assertEqual(failures[0]["kind"], "reference_contract")
        self.assertEqual(failures[0]["field"], "weight_preparation_contract")
        self.assertEqual(
            failures[0]["expected"],
            {
                "flashinfer_b12x": False,
                bench.FLASHINFER_CUTLASS_MODE: True,
            },
        )

        failures = bench.checkpoint_reference_failures(
            candidate,
            {
                "checkpoint": candidate.copy(),
                "settings": {
                    "checkpoint_load_strategy": "per-expert",
                    "backend_selection": "w4a4-ab",
                },
            },
        )
        self.assertEqual(failures[0]["kind"], "reference_contract")
        self.assertEqual(failures[0]["field"], "settings.backend_selection")

        staged_reference = candidate.copy()
        staged_reference["checkpoint_load_strategy"] = "layer-staged"
        failures = bench.checkpoint_reference_failures(
            candidate,
            {
                "checkpoint": staged_reference,
                "settings": {
                    "checkpoint_load_strategy": "layer-staged",
                    "backend_selection": bench.FLASHINFER_CUTLASS_MODE,
                },
            },
        )
        self.assertEqual(failures[0]["kind"], "reference_contract")
        self.assertEqual(failures[0]["field"], "checkpoint_load_strategy")

    def test_checkpoint_page_advice_targets_only_selected_layer_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "a.safetensors").write_bytes(b"a" * 11)
            (root / "b.safetensors").write_bytes(b"b" * 13)
            (root / "c.safetensors").write_bytes(b"c" * 17)
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "layers.0.ffn.a": "a.safetensors",
                            "layers.0.ffn.b": "b.safetensors",
                            "layers.1.ffn.c": "c.safetensors",
                        }
                    }
                )
            )
            advised: list[tuple[int, int, int, int]] = []
            with (
                mock.patch.object(
                    bench.os,
                    "posix_fadvise",
                    side_effect=lambda *args: advised.append(args),
                    create=True,
                ),
                mock.patch.object(
                    bench.os,
                    "POSIX_FADV_DONTNEED",
                    4,
                    create=True,
                ),
            ):
                evidence = bench.evict_checkpoint_layer_pages(root, layer_idx=0)

        self.assertEqual(evidence["shards"], ["a.safetensors", "b.safetensors"])
        self.assertEqual(evidence["shard_bytes"], 24)
        self.assertEqual(len(advised), 2)
        self.assertTrue(all(call[1:] == (0, 0, 4) for call in advised))

    def test_load_strategy_contract_preserves_fused_up_gate_order(self) -> None:
        source = inspect.getsource(bench.load_checkpoint_weights)
        self.assertEqual(
            bench.CHECKPOINT_LOAD_STRATEGIES,
            ("per-expert", "layer-staged"),
        )
        self.assertIn('(\"w3\", w13_host[expert_id, :intermediate])', source)
        self.assertIn('(\"w1\", w13_host[expert_id, intermediate:])', source)
        self.assertIn("w13 = torch.cat((w3, w1), dim=1)", source)
        self.assertIn('copy_profile["bulk_device_transfer"]', source)

    def test_backend_specific_scale_preparation_omits_b12x_for_cutlass(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")

        shape = bench.Dsv4Shape(
            hidden_size=128,
            intermediate_size=256,
            num_experts=2,
            top_k=1,
            tp_size=2,
            tp_rank=0,
        )

        def inputs() -> dict[str, object]:
            experts = shape.num_experts
            intermediate = shape.intermediate_size_per_rank
            hidden = shape.hidden_size
            return {
                "w13": torch.zeros(
                    experts, 2 * intermediate, hidden // 2, dtype=torch.uint8
                ),
                "w13_scale": torch.ones(
                    experts, 2 * intermediate, hidden // 16
                ),
                "w13_scale_2": torch.ones(experts),
                "w13_input_scale": torch.ones(experts),
                "w2": torch.zeros(
                    experts, hidden, intermediate // 2, dtype=torch.uint8
                ),
                "w2_scale": torch.ones(
                    experts, hidden, intermediate // 16
                ),
                "w2_scale_2": torch.ones(experts),
                "w2_input_scale": torch.ones(experts),
            }

        fake_nvfp4_utils = SimpleNamespace(
            swizzle_blockscale=mock.Mock(side_effect=lambda tensor: tensor.clone())
        )
        real_import = __import__

        def import_with_fake_nvfp4_utils(
            name: str,
            globals: object = None,
            locals: object = None,
            fromlist: object = (),
            level: int = 0,
        ) -> object:
            if name == (
                "vllm.model_executor.layers.quantization.utils.nvfp4_utils"
            ):
                return fake_nvfp4_utils
            return real_import(name, globals, locals, fromlist, level)

        with (
            mock.patch("builtins.__import__", side_effect=import_with_fake_nvfp4_utils),
            mock.patch.object(torch.cuda, "synchronize"),
            mock.patch.object(bench, "_bake_expert_scales") as bake,
            mock.patch.object(bench, "_scale_to_mma") as scale_to_mma,
        ):
            cutlass = bench._finish_scale_preparation(
                torch,
                **inputs(),
                shape=shape,
                metadata={"source": "checkpoint"},
                prepare_cutlass=True,
                prepare_b12x=False,
            )

        bake.assert_not_called()
        scale_to_mma.assert_not_called()
        self.assertEqual(fake_nvfp4_utils.swizzle_blockscale.call_count, 2)
        self.assertIsNone(cutlass.w13_sf_swizzled)
        self.assertIsNone(cutlass.w2_sf_swizzled)
        self.assertIsNone(cutlass.w13_sf_mma)
        self.assertIsNone(cutlass.w2_sf_mma)
        self.assertIsNotNone(cutlass.w13_sf_modelopt)
        self.assertIsNotNone(cutlass.w2_sf_modelopt)
        self.assertEqual(
            set(cutlass.metadata["sample_fingerprints"]),
            bench.COMMON_SAMPLE_FINGERPRINTS
            | bench.CUTLASS_SAMPLE_FINGERPRINTS,
        )
        self.assertEqual(
            cutlass.metadata["weight_preparation_contract"],
            {
                "flashinfer_b12x": False,
                bench.FLASHINFER_CUTLASS_MODE: True,
                "required_sample_fingerprints": sorted(
                    bench.COMMON_SAMPLE_FINGERPRINTS
                    | bench.CUTLASS_SAMPLE_FINGERPRINTS
                ),
            },
        )

        fake_nvfp4_utils.swizzle_blockscale.reset_mock()
        real_ones = torch.ones

        def ones_on_cpu(*args: object, **kwargs: object) -> object:
            forwarded = dict(kwargs)
            if forwarded.get("device") == "cuda":
                forwarded["device"] = "cpu"
            return real_ones(*args, **forwarded)

        with (
            mock.patch("builtins.__import__", side_effect=import_with_fake_nvfp4_utils),
            mock.patch.object(torch, "ones", side_effect=ones_on_cpu),
            mock.patch.object(torch.cuda, "synchronize"),
            mock.patch.object(
                bench,
                "_bake_expert_scales",
                side_effect=lambda torch_module, scale, global_scale: scale,
            ) as bake,
            mock.patch.object(
                bench,
                "_scale_to_mma",
                side_effect=lambda torch_module, scale, **kwargs: scale,
            ) as scale_to_mma,
        ):
            dual = bench._finish_scale_preparation(
                torch,
                **inputs(),
                shape=shape,
                metadata={"source": "checkpoint"},
                prepare_cutlass=True,
                prepare_b12x=True,
            )

        self.assertEqual(bake.call_count, 2)
        self.assertEqual(scale_to_mma.call_count, 2)
        self.assertEqual(fake_nvfp4_utils.swizzle_blockscale.call_count, 4)
        self.assertIsNotNone(dual.w13_sf_swizzled)
        self.assertIsNotNone(dual.w2_sf_swizzled)
        self.assertIsNotNone(dual.w13_sf_mma)
        self.assertIsNotNone(dual.w2_sf_mma)
        self.assertEqual(
            set(dual.metadata["sample_fingerprints"]),
            bench.COMMON_SAMPLE_FINGERPRINTS
            | bench.B12X_SAMPLE_FINGERPRINTS
            | bench.CUTLASS_SAMPLE_FINGERPRINTS,
        )

    def test_tiny_cuda_checkpoint_staged_and_per_expert_are_bit_exact(self) -> None:
        try:
            import torch
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest("PyTorch/safetensors are not installed")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        def values(shape: tuple[int, ...], offset: int) -> object:
            count = math.prod(shape)
            return (
                torch.arange(count, dtype=torch.int64).add(offset).remainder(251)
                .to(torch.uint8)
                .reshape(shape)
            )

        rank_outputs: dict[int, object] = {}
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            tensors: dict[str, object] = {}
            for expert_id in range(2):
                prefix = f"layers.0.ffn.experts.{expert_id}"
                for projection_id, projection in enumerate(("w1", "w3", "w2")):
                    base = 10_000 * expert_id + 1_000 * projection_id
                    weight_shape = (256, 64) if projection != "w2" else (128, 128)
                    scale_shape = (256, 8) if projection != "w2" else (128, 16)
                    tensors[f"{prefix}.{projection}.weight"] = values(
                        weight_shape, base
                    )
                    tensors[f"{prefix}.{projection}.weight_scale"] = (
                        values(scale_shape, base + 17)
                        .remainder(31)
                        .add(1)
                        .to(torch.float32)
                        .mul(2.0**-7)
                        .to(torch.float8_e4m3fn)
                    )
                    tensors[f"{prefix}.{projection}.weight_scale_2"] = torch.tensor(
                        0.25 + expert_id * 0.01 + projection_id * 0.001,
                        dtype=torch.float32,
                    )
                    tensors[f"{prefix}.{projection}.input_scale"] = torch.tensor(
                        2.0 + expert_id * 0.1 + projection_id * 0.01,
                        dtype=torch.float32,
                    )
            shard = "model-00001-of-00001.safetensors"
            save_file(tensors, root / shard)
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {key: shard for key in tensors}})
            )

            for tp_rank in (0, 1):
                shape = bench.Dsv4Shape(
                    hidden_size=128,
                    intermediate_size=256,
                    num_experts=2,
                    top_k=1,
                    tp_size=2,
                    tp_rank=tp_rank,
                )
                captures: dict[str, dict[str, object]] = {}

                def capture_finish(torch_module: object, **kwargs: object) -> object:
                    strategy = str(kwargs["metadata"]["checkpoint_load_strategy"])
                    captures[strategy] = {
                        key: kwargs[key].detach().cpu().clone()
                        for key in (
                            "w13",
                            "w13_scale",
                            "w13_scale_2",
                            "w13_input_scale",
                            "w2",
                            "w2_scale",
                            "w2_scale_2",
                            "w2_input_scale",
                        )
                    }
                    return SimpleNamespace(metadata=kwargs["metadata"])

                with mock.patch.object(
                    bench,
                    "_finish_scale_preparation",
                    side_effect=capture_finish,
                ):
                    for strategy in bench.CHECKPOINT_LOAD_STRATEGIES:
                        bench.load_checkpoint_weights(
                            torch,
                            root,
                            shape,
                            layer_idx=0,
                            checkpoint_metadata={"layer_idx": 0},
                            prepare_cutlass=True,
                            load_strategy=strategy,
                        )
                        torch.cuda.empty_cache()

                baseline = captures["per-expert"]
                staged = captures["layer-staged"]
                self.assertEqual(set(baseline), set(staged))
                for key in baseline:
                    left = baseline[key]
                    right = staged[key]
                    if left.element_size() == 1:
                        left = left.view(torch.uint8)
                        right = right.view(torch.uint8)
                    self.assertTrue(torch.equal(left, right), f"mismatch in {key}")
                rank_outputs[tp_rank] = staged["w13"]

        self.assertFalse(torch.equal(rank_outputs[0], rank_outputs[1]))

    def test_load_predictor_cli_requires_load_only_real_checkpoint(self) -> None:
        parser = bench.build_parser()
        args = parser.parse_args(
            ["--dry-run", "--synthetic", "--require-load-predictor"]
        )
        with self.assertRaisesRegex(ValueError, "requires --load-only"):
            bench.validate_args(args)

        args = parser.parse_args(["--dry-run", "--synthetic", "--load-only"])
        with self.assertRaisesRegex(ValueError, "real --model-path"):
            bench.validate_args(args)

        args = parser.parse_args(
            [
                "--dry-run",
                "--model-path",
                "/does/not-matter-for-validation",
                "--load-only",
                "--require-load-predictor",
            ]
        )
        with self.assertRaisesRegex(ValueError, "backend flashinfer_cutlass"):
            bench.validate_args(args)

        args = parser.parse_args(
            [
                "--dry-run",
                "--model-path",
                "/does/not-matter-for-validation",
                "--load-only",
                "--require-load-predictor",
                "--backend",
                bench.FLASHINFER_CUTLASS_MODE,
                "--checkpoint-load-strategy",
                "layer-staged",
                "--reference-load-json",
                "/immutable/reference.json",
                "--reference-load-sha256",
                "0" * 64,
                "--evict-checkpoint-pages",
            ]
        )
        bench.validate_args(args)

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

    def test_synthetic_projection_seeds_are_stable_per_expert_and_projection(self) -> None:
        self.assertEqual(bench.synthetic_projection_seed(4104, 0, 0), 4104)
        self.assertEqual(bench.synthetic_projection_seed(4104, 0, 1), 4105)
        self.assertEqual(bench.synthetic_projection_seed(4104, 7, 0), 4118)
        self.assertEqual(bench.synthetic_projection_seed(4104, 7, 1), 4119)
        with self.assertRaisesRegex(ValueError, "expert id"):
            bench.synthetic_projection_seed(4104, -1, 0)
        with self.assertRaisesRegex(ValueError, "projection lane"):
            bench.synthetic_projection_seed(4104, 0, 2)

    def test_synthetic_fixture_metadata_distinguishes_default_and_legacy(self) -> None:
        default = bench.synthetic_fixture_metadata(
            seed=4104,
            legacy_degenerate=False,
        )
        legacy = bench.synthetic_fixture_metadata(
            seed=4104,
            legacy_degenerate=True,
        )

        self.assertEqual(default["synthetic_fixture"], bench.SYNTHETIC_RANDOM_FIXTURE)
        self.assertEqual(default["weight_seed"], 4104)
        self.assertEqual(default["source_distribution"], "torch.randn / 15")
        self.assertEqual(
            default["quantizer"], "vllm._custom_ops.scaled_fp4_quant"
        )
        self.assertEqual(default["scale_layout_before_preparation"], "linear")
        self.assertNotIn("packed_fill", default)

        self.assertEqual(legacy["synthetic_fixture"], bench.SYNTHETIC_LEGACY_FIXTURE)
        self.assertEqual(legacy["packed_fill"], "0x11")
        self.assertEqual(legacy["logical_scale"], 2.0**-7)
        self.assertNotIn("weight_seed", legacy)

    def test_default_synthetic_fixture_uses_upstream_quantizer_contract(self) -> None:
        source = inspect.getsource(bench.make_synthetic_weights)
        self.assertIn("from vllm import _custom_ops as ops", source)
        self.assertIn("ops.scaled_fp4_quant(", source)
        self.assertIn("is_sf_swizzled_layout=False", source)
        self.assertIn("/ 15.0", source)
        self.assertIn("for expert_id in range(experts)", source)
        self.assertIn("legacy_degenerate", source)

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

    def test_effective_failures_suppresses_only_numeric_comparisons(self) -> None:
        numeric = {"kind": "numeric", "comparison": "graph_vs_eager"}
        activity = {"kind": "output_activity", "stage": "eager"}
        graph = {"kind": "cuda_graph", "error": "capture failed"}
        failures = [numeric, activity, graph]

        self.assertEqual(
            bench.effective_failures(
                failures,
                no_correctness_gate=False,
            ),
            failures,
        )
        self.assertEqual(
            bench.effective_failures(
                failures,
                no_correctness_gate=True,
            ),
            [activity, graph],
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

    def test_fail_fast_is_recorded_and_stops_after_completed_row_cleanup(self) -> None:
        parser = bench.build_parser()
        args = parser.parse_args(
            [
                "--dry-run",
                "--synthetic",
                "--m",
                "1,2",
                "--correctness-m",
                "1,2",
                "--fail-fast",
                "--require-graphs",
            ]
        )
        bench.validate_args(args)
        plan = bench.build_dry_run_plan(args, ROOT)
        self.assertTrue(plan["timing"]["fail_fast"])
        self.assertTrue(plan["timing"]["require_graphs"])
        self.assertFalse(plan["timing"]["no_correctness_gate"])

        source = inspect.getsource(bench.run_benchmark)
        cleanup = source.index(
            "del eager_outputs, launches, keepalive, x, topk_ids, topk_weights"
        )
        row_stop = source.index('report["fail_fast_stop"]', cleanup)
        self.assertLess(cleanup, row_stop)
        self.assertIn('"after_m": m', source[row_stop:])
        self.assertIn(
            '"remaining_m": list(matrix_m_values[m_index + 1 :])',
            source[row_stop:],
        )
        self.assertIn('"after_m": None', source[:cleanup])
        self.assertIn('"remaining_m": list(args.m)', source[:cleanup])

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
        self.assertIsNone(
            bench.b12x_workspace_ceiling_bytes(bench.Dsv4Shape(), 1)
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
        self.assertEqual(
            report["checkpoint"]["synthetic_fixture"],
            bench.SYNTHETIC_RANDOM_FIXTURE,
        )
        self.assertEqual(report["checkpoint"]["weight_seed"], 4104)
        self.assertEqual(
            report["checkpoint"]["quantizer"],
            "vllm._custom_ops.scaled_fp4_quant",
        )
        self.assertEqual(
            report["checkpoint_loading"]["strategy"], "per-expert"
        )
        self.assertFalse(report["checkpoint_loading"]["load_only"])
        self.assertEqual(
            report["checkpoint_loading"]["predictor_contract"]["routed_layers"],
            43,
        )
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
                    "unit synthetic input_scale (upstream kernel-test contract)"
                ),
            },
        )

    def test_legacy_synthetic_dry_run_preserves_degenerate_fixture_contract(self) -> None:
        parser = bench.build_parser()
        args = parser.parse_args(
            ["--dry-run", "--synthetic", "--legacy-degenerate-synthetic"]
        )
        bench.validate_args(args)
        report = bench.build_dry_run_plan(args, ROOT)
        self.assertEqual(
            report["checkpoint"]["synthetic_fixture"],
            bench.SYNTHETIC_LEGACY_FIXTURE,
        )
        self.assertEqual(report["checkpoint"]["packed_fill"], "0x11")
        self.assertEqual(report["checkpoint"]["logical_scale"], 2.0**-7)

    def test_legacy_synthetic_flag_requires_synthetic_source(self) -> None:
        parser = bench.build_parser()
        args = parser.parse_args(["--dry-run", "--legacy-degenerate-synthetic"])
        with self.assertRaisesRegex(ValueError, "requires --synthetic"):
            bench.validate_args(args)

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

    def test_require_graphs_rejects_disabled_cuda_graphs(self) -> None:
        parser = bench.build_parser()
        args = parser.parse_args(
            [
                "--dry-run",
                "--synthetic",
                "--require-graphs",
                "--no-cuda-graph",
            ]
        )
        with self.assertRaisesRegex(ValueError, "requires --cuda-graph"):
            bench.validate_args(args)


if __name__ == "__main__":
    unittest.main()
