# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from __future__ import annotations

import importlib
from pathlib import Path
import unittest

from benchmarks import probe_nvfp4_staged_routed_loader as probe


class Nvfp4StagedRoutedLoaderProbeTest(unittest.TestCase):
    def test_probe_is_copied_into_immutable_candidate_image(self) -> None:
        root = Path(__file__).parents[1]
        dockerfile = (
            root / "docker" / "Dockerfile.nvfp4-aot-overlay"
        ).read_text()
        copy_contract = (
            "COPY benchmarks/probe_nvfp4_staged_routed_loader.py "
            "/usr/local/bin/dspark-probe-nvfp4-staged-routed-loader"
        )
        self.assertEqual(dockerfile.count(copy_contract), 1)
        dockerignore = (
            root / "docker" / "Dockerfile.nvfp4-aot-overlay.dockerignore"
        ).read_text().splitlines()
        self.assertIn("!benchmarks/", dockerignore)
        self.assertIn(
            "!benchmarks/probe_nvfp4_staged_routed_loader.py",
            dockerignore,
        )

    def test_exact_moe_config_exposes_both_tp_size_paths(self) -> None:
        for rank in range(probe.TP_SIZE):
            config = probe._make_moe_config(rank)
            self.assertEqual(config.tp_rank, rank)
            self.assertEqual(config.tp_size, probe.TP_SIZE)
            self.assertEqual(
                config.moe_parallel_config.tp_size,
                probe.TP_SIZE,
            )

    def test_contract_is_cpu_first_tiny_and_covers_all_raw_storages(self) -> None:
        args = probe.build_parser().parse_args([])
        self.assertEqual(args.device, "cpu")
        self.assertEqual(len(probe.PARAMETER_ORDER), 8)
        self.assertEqual(
            set(probe.PARAMETER_ORDER),
            probe.PACKED_PARAMETERS
            | probe.RAW_BLOCK_SCALE_PARAMETERS
            | probe.FP32_PARAMETERS,
        )
        self.assertEqual(
            set(probe.EXPECTED_STAGER_SOURCE_SHA256),
            {
                "begin_source",
                "destination",
                "complete_source",
                "_commit_active_layer",
                "finish",
            },
        )
        self.assertIn(
            "_load_single_value",
            probe.EXPECTED_ROUTED_SOURCE_SHA256,
        )
        self.assertEqual(
            probe.EXPECTED_PARAM_ATTRIBUTE_CHAINS["_load_single_value"],
            ["param.data"],
        )
        self.assertEqual(
            len(probe.EXPECTED_STAGER_FACTORY_SOURCE_SHA256),
            64,
        )

    def test_factory_preflight_descriptors_are_shape_only_and_exact_size(self) -> None:
        shapes = probe._official_factory_parameter_shapes()
        self.assertEqual(set(shapes), set(probe.PARAMETER_ORDER))
        observed_bytes = 0
        for name, shape in shapes.items():
            element_bytes = 4 if name in probe.FP32_PARAMETERS else 1
            descriptor = probe._ShapeOnlyCudaParameter(
                shape,
                dtype=name,
                element_bytes=element_bytes,
            )
            self.assertEqual(descriptor.device.type, "cuda")
            self.assertEqual(descriptor.element_size(), element_bytes)
            observed_bytes += descriptor.numel() * element_bytes
        self.assertEqual(observed_bytes, probe.OFFICIAL_STAGE_BYTES)

    def test_exact_main_checkpoint_source_contract_is_recorded(self) -> None:
        self.assertEqual(
            probe.MAIN_TARGET_SOURCE_CONTRACT,
            {
                "weight": {"dtype": "torch.uint8", "rank": 2},
                "weight_scale": {
                    "dtype": "torch.float8_e4m3fn",
                    "rank": 2,
                },
                "weight_scale_2": {
                    "dtype": "torch.float32",
                    "rank": 0,
                    "shape": [],
                },
                "input_scale": {
                    "dtype": "torch.float32",
                    "rank": 0,
                    "shape": [],
                },
            },
        )
        layout = probe.CHECKPOINT_LAYOUT_CONTRACT
        self.assertEqual(layout["main_target"]["checkpoint_shards"], 46)
        self.assertEqual(layout["main_target"]["layers"], 43)
        self.assertEqual(layout["main_target"]["tensors_per_layer"], 3_072)
        self.assertTrue(layout["main_target"]["layers_contiguous"])
        self.assertTrue(layout["main_target"]["one_layer_per_shard"])
        self.assertEqual(layout["mtp_excluded"]["weight_dtype"], "torch.int8")
        self.assertEqual(
            layout["mtp_excluded"]["weight_scale_dtype"],
            "torch.float8_e8m0fnu",
        )

    def test_proxy_contract_copies_every_optional_loader_attribute(self) -> None:
        self.assertEqual(
            probe.EXPECTED_PROXY_COPIED_ATTRS,
            probe.EXPECTED_OPTIONAL_PARAM_ATTRS,
        )

    def test_ast_contract_finds_param_attrs_and_device_branch(self) -> None:
        source = """
def weight_loader(param, loaded_weight):
    is_transposed = getattr(param, "is_transposed", False)
    if is_transposed:
        return loaded_weight.to(param.data.device)
    return param.data.shape
"""
        contract = probe._param_source_contract(source)
        self.assertEqual(
            contract["attribute_chains"],
            ["param.data", "param.data.device", "param.data.shape"],
        )
        self.assertEqual(contract["optional_attrs"], ["is_transposed"])
        self.assertEqual(
            contract["device_branches"],
            ["loaded_weight.to(param.data.device)"],
        )

    def test_proxy_attribute_extraction_is_exhaustive(self) -> None:
        source = """
def destination(proxy, actual_parameter):
    for attr in ("is_transposed", "quant_method"):
        if hasattr(actual_parameter, attr):
            setattr(proxy, attr, getattr(actual_parameter, attr))
"""
        self.assertEqual(
            probe._proxy_copied_attrs(source),
            ["is_transposed", "quant_method"],
        )

    def test_real_cpu_probe_when_candidate_runtime_is_available(self) -> None:
        try:
            importlib.import_module("torch")
            importlib.import_module(
                "vllm.model_executor.layers.fused_moe.routed_experts"
            )
            importlib.import_module(
                "vllm.models.deepseek_v4.nvidia.staged_weight_loading"
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"assembled candidate runtime is unavailable: {exc}")
        report = probe.run_probe("cpu")
        self.assertTrue(report["ok"], report)
        self.assertTrue(report["default_cpu_proof"])
        self.assertFalse(report["model_loaded"])
        self.assertFalse(report["checkpoint_opened"])
        self.assertTrue(report["factory_preflight"]["passed"])
        self.assertTrue(
            report["factory_preflight"]["shape_only_cuda_descriptors"]
        )
        self.assertFalse(report["factory_preflight"]["tensor_storage_allocated"])
        self.assertEqual(report["factory_preflight"]["layers"], 43)
        self.assertEqual(report["factory_preflight"]["experts"], 256)
        self.assertEqual(report["factory_preflight"]["mapping_keys"], 768)
        self.assertEqual(
            report["factory_preflight"]["source_keys_per_layer"],
            3_072,
        )
        self.assertEqual(
            report["factory_preflight"]["virtual_bytes_per_layer"],
            probe.OFFICIAL_STAGE_BYTES,
        )
        self.assertEqual([row["tp_rank"] for row in report["ranks"]], [0, 1])
        for rank in report["ranks"]:
            self.assertEqual(set(rank["storages"]), set(probe.PARAMETER_ORDER))
            self.assertTrue(rank["passed"])
            for suffix in ("weight_scale_2", "input_scale"):
                self.assertEqual(
                    rank["checkpoint_source_contract"][suffix]["shapes"],
                    [[]],
                )
                self.assertEqual(
                    rank["checkpoint_source_contract"][suffix]["ranks"],
                    [0],
                )
            self.assertEqual(
                rank["checkpoint_source_contract"]["weight_scale"]["dtypes"],
                ["torch.float8_e4m3fn"],
            )
            self.assertEqual(
                rank["stager_dispatch_contract"]["weight_scale"]["dtypes"],
                ["torch.uint8"],
            )
            for name, storage in rank["storages"].items():
                self.assertTrue(storage["staged_matches_explicit_oracle"], name)
                self.assertTrue(storage["direct_comparison_required"], name)
                self.assertTrue(storage["direct_matches_explicit_oracle"], name)


if __name__ == "__main__":
    unittest.main()
