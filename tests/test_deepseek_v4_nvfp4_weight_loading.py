# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = (
    ROOT / "overlay/vllm/models/deepseek_v4/nvidia/weight_loading.py"
)


def _load_helper():
    spec = importlib.util.spec_from_file_location(
        "_deepseek_v4_nvfp4_weight_loading_under_test", HELPER_PATH
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not import {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _expert_mapping(
    *, num_experts: int = 256, lora_base_layer_prefix: str = ""
):
    return [
        (
            "experts.routed_experts."
            f"{lora_base_layer_prefix}"
            + ("w13_" if projection in ("w1", "w3") else "w2_"),
            f"experts.{expert_id}.{projection}.{lora_base_layer_prefix}",
            expert_id,
            projection,
        )
        for expert_id in range(num_experts)
        for projection in ("w1", "w2", "w3")
    ]


class DeepseekV4NvFp4WeightLoadingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.helper = _load_helper()

    def test_all_official_nvfp4_target_names_use_index(self) -> None:
        mapping = _expert_mapping()
        index = self.helper.build_expert_mapping_index(mapping)
        suffixes = ("weight", "weight_scale", "weight_scale_2", "input_scale")
        seen = 0
        for layer in range(43):
            for expert_id in range(256):
                for projection in ("w1", "w2", "w3"):
                    for suffix in suffixes:
                        name = (
                            f"layers.{layer}.ffn.experts.{expert_id}."
                            f"{projection}.{suffix}"
                        )
                        match, candidates = self.helper.select_expert_mappings(
                            name, mapping, index
                        )
                        self.assertIsNotNone(match)
                        self.assertEqual(len(candidates), 1)
                        selected = candidates[0]
                        self.assertEqual(
                            self.helper.map_expert_parameter_name(
                                name, selected[0], selected[1], match
                            ),
                            name.replace(selected[1], selected[0]),
                        )
                        seen += 1
        self.assertEqual(seen, 132_096)

    def test_projection_mapping(self) -> None:
        mapping = _expert_mapping(num_experts=1)
        index = self.helper.build_expert_mapping_index(mapping)
        expected = {
            "w1": ("experts.routed_experts.w13_", "w1"),
            "w2": ("experts.routed_experts.w2_", "w2"),
            "w3": ("experts.routed_experts.w13_", "w3"),
        }
        for projection, (param_name, shard_id) in expected.items():
            name = f"layers.0.ffn.experts.0.{projection}.weight"
            match, candidates = self.helper.select_expert_mappings(
                name, mapping, index
            )
            self.assertIsNotNone(match)
            self.assertEqual(candidates[0][0], param_name)
            self.assertEqual(candidates[0][3], shard_id)

    def test_lora_base_layer_exact_key(self) -> None:
        mapping = _expert_mapping(num_experts=16, lora_base_layer_prefix="base_layer.")
        index = self.helper.build_expert_mapping_index(mapping)
        name = "layers.7.ffn.experts.12.w3.base_layer.weight"
        match, candidates = self.helper.select_expert_mappings(name, mapping, index)
        self.assertIsNotNone(match)
        self.assertEqual(len(candidates), 1)
        selected = candidates[0]
        self.assertEqual(selected[1], "experts.12.w3.base_layer.")
        self.assertEqual(
            self.helper.map_expert_parameter_name(
                name, selected[0], selected[1], match
            ),
            "layers.7.ffn.experts.routed_experts.base_layer.w13_weight",
        )

    def test_mega_moe_parameter_prefix(self) -> None:
        mapping = [
            ("experts.w13_", "experts.12.w1.", 12, "w1"),
            ("experts.w2_", "experts.12.w2.", 12, "w2"),
            ("experts.w13_", "experts.12.w3.", 12, "w3"),
        ]
        index = self.helper.build_expert_mapping_index(mapping)
        name = "layers.7.ffn.experts.12.w1.weight_scale"
        match, candidates = self.helper.select_expert_mappings(name, mapping, index)
        selected = candidates[0]
        self.assertEqual(
            self.helper.map_expert_parameter_name(
                name, selected[0], selected[1], match
            ),
            "layers.7.ffn.experts.w13_weight_scale",
        )

    def test_duplicate_logical_key_preserves_order(self) -> None:
        mapping = [
            ("experts.routed_experts.w13_", "experts.0.w1.", 5, "w1"),
            ("experts.routed_experts.w13_", "experts.0.w1.", 9, "w1"),
        ]
        index = self.helper.build_expert_mapping_index(mapping)
        match, candidates = self.helper.select_expert_mappings(
            "layers.0.ffn.experts.0.w1.weight", mapping, index
        )
        self.assertIsNotNone(match)
        self.assertEqual([candidate[2] for candidate in candidates], [5, 9])

    def test_eplb_duplicate_attempt_order_matches_legacy_scan(self) -> None:
        # Captures the shape of RoutedExperts.build_expert_params_mapping when
        # redundant physical experts map back to the same logical expert.
        mapping = [
            ("experts.routed_experts.w13_", "experts.0.w1.", 0, "w1"),
            ("experts.routed_experts.w2_", "experts.0.w2.", 0, "w2"),
            ("experts.routed_experts.w13_", "experts.0.w3.", 0, "w3"),
            ("experts.routed_experts.w13_", "experts.1.w1.", 1, "w1"),
            ("experts.routed_experts.w2_", "experts.1.w2.", 1, "w2"),
            ("experts.routed_experts.w13_", "experts.1.w3.", 1, "w3"),
            ("experts.routed_experts.w13_", "experts.0.w1.", 2, "w1"),
            ("experts.routed_experts.w2_", "experts.0.w2.", 2, "w2"),
            ("experts.routed_experts.w13_", "experts.0.w3.", 2, "w3"),
        ]
        name = "layers.0.ffn.experts.0.w1.weight_scale_2"
        index = self.helper.build_expert_mapping_index(mapping)
        match, fast_candidates = self.helper.select_expert_mappings(
            name, mapping, index
        )
        legacy_candidates = [
            candidate for candidate in mapping if candidate[1] in name
        ]
        self.assertEqual(list(fast_candidates), legacy_candidates)
        self.assertEqual([candidate[2] for candidate in fast_candidates], [0, 2])

        for outcomes in ((False, True), (False, False)):
            legacy_attempts = []
            fast_attempts = []
            for candidate, success in zip(legacy_candidates, outcomes, strict=True):
                legacy_attempts.append(candidate[2])
                if success:
                    break
            for candidate, success in zip(fast_candidates, outcomes, strict=True):
                fast_attempts.append(candidate[2])
                mapped = self.helper.map_expert_parameter_name(
                    name, candidate[0], candidate[1], match
                )
                self.assertEqual(mapped, name.replace(candidate[1], candidate[0]))
                if success:
                    break
            self.assertEqual(fast_attempts, legacy_attempts)

    def test_unknown_grammar_uses_legacy_fallback(self) -> None:
        mapping = _expert_mapping(num_experts=1)
        index = self.helper.build_expert_mapping_index(mapping)
        name = "layers.0.mlp.experts.0.w1.weight"
        match, candidates = self.helper.select_expert_mappings(name, mapping, index)
        self.assertIsNone(match)
        self.assertIs(candidates, mapping)
        selected = candidates[0]
        self.assertEqual(
            self.helper.map_expert_parameter_name(
                name, selected[0], selected[1], match
            ),
            name.replace(selected[1], selected[0]),
        )

    def test_known_grammar_unknown_key_uses_legacy_fallback(self) -> None:
        mapping = _expert_mapping(num_experts=1)
        index = self.helper.build_expert_mapping_index(mapping)
        match, candidates = self.helper.select_expert_mappings(
            "layers.0.ffn.experts.999.w1.weight", mapping, index
        )
        self.assertIsNone(match)
        self.assertIs(candidates, mapping)

    def test_unknown_suffix_uses_exact_legacy_replacement(self) -> None:
        mapping = _expert_mapping(num_experts=1)
        index = self.helper.build_expert_mapping_index(mapping)
        name = "layers.0.ffn.experts.0.w1.foo.experts.0.w1."
        match, candidates = self.helper.select_expert_mappings(name, mapping, index)
        self.assertIsNone(match)
        selected = candidates[0]
        self.assertEqual(
            self.helper.map_expert_parameter_name(
                name, selected[0], selected[1], match
            ),
            name.replace(selected[1], selected[0]),
        )

    def test_mixed_plain_and_lora_keys_force_legacy_fallback(self) -> None:
        mapping = _expert_mapping(num_experts=1) + _expert_mapping(
            num_experts=1, lora_base_layer_prefix="base_layer."
        )
        index = self.helper.build_expert_mapping_index(mapping)
        self.assertFalse(index.safe)
        for name in (
            "layers.0.ffn.experts.0.w1.weight",
            "layers.0.ffn.experts.0.w1.base_layer.weight",
        ):
            match, candidates = self.helper.select_expert_mappings(
                name, mapping, index
            )
            self.assertIsNone(match)
            self.assertIs(candidates, mapping)

    def test_unknown_mapping_key_forces_legacy_fallback(self) -> None:
        mapping = _expert_mapping(num_experts=1) + [
            ("experts.future_", "experts.0.w1.future.", 0, "w1")
        ]
        index = self.helper.build_expert_mapping_index(mapping)
        self.assertFalse(index.safe)
        match, candidates = self.helper.select_expert_mappings(
            "layers.0.ffn.experts.0.w1.weight", mapping, index
        )
        self.assertIsNone(match)
        self.assertIs(candidates, mapping)

    def test_fused_mapping_keeps_numeric_fast_path_and_fused_fallback(self) -> None:
        mapping = [
            ("experts.routed_experts.w13_weight", "experts.w13", 0, "w1"),
            ("experts.routed_experts.w13_weight", "experts.w13", 1, "w3"),
        ] + _expert_mapping(num_experts=1)
        index = self.helper.build_expert_mapping_index(mapping)
        self.assertTrue(index.safe)
        match, candidates = self.helper.select_expert_mappings(
            "layers.0.ffn.experts.0.w1.weight", mapping, index
        )
        self.assertIsNotNone(match)
        self.assertEqual(len(candidates), 1)
        fused_name = "layers.0.ffn.experts.w13.weight"
        fused_match, fused_candidates = self.helper.select_expert_mappings(
            fused_name, mapping, index
        )
        self.assertIsNone(fused_match)
        self.assertIs(fused_candidates, mapping)
        self.assertEqual(
            self.helper.map_expert_parameter_name(
                fused_name, mapping[0][0], mapping[0][1], fused_match
            ),
            fused_name.replace(mapping[0][1], mapping[0][0]),
        )

    def test_mismatched_fast_candidate_uses_legacy_behavior(self) -> None:
        name = "layers.0.ffn.experts.0.w1.weight"
        match = self.helper.parse_expert_name(name)
        self.assertIsNotNone(match)
        self.assertEqual(
            self.helper.map_expert_parameter_name(
                name,
                "experts.routed_experts.w13_",
                "experts.0.w3.",
                match,
            ),
            name,
        )

    def test_mtp_name_is_not_claimed(self) -> None:
        self.assertIsNone(
            self.helper.parse_expert_name(
                "mtp.0.ffn.experts.0.w1.weight_scale_2"
            )
        )

    def test_optional_model_prefix_is_preserved(self) -> None:
        match = self.helper.parse_expert_name(
            "model.layers.7.ffn.experts.12.w3.weight_scale_2"
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.layer, 7)
        self.assertEqual(match.logical_expert, 12)
        self.assertEqual(match.projection, "w3")
        self.assertEqual(
            match.map_parameter_name("experts.routed_experts.w13_"),
            "model.layers.7.ffn.experts.routed_experts.w13_weight_scale_2",
        )


if __name__ == "__main__":
    unittest.main()
