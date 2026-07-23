# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from __future__ import annotations

import ast
import importlib.util
import hashlib
import pathlib
import sys
import unittest
from types import SimpleNamespace


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPERT = (
    ROOT
    / "overlay/vllm/model_executor/layers/fused_moe/experts/"
    "nvfp4_dual_decode_moe.py"
)
B12X_EXPERT = EXPERT.with_name("b12x_mxfp4_moe.py")
POLICY = EXPERT.with_name("nvfp4_dual_decode_policy.py")
ORACLE = ROOT / "overlay/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py"
ENVS = ROOT / "overlay/vllm/envs.py"
PREPARED = ROOT / "overlay/vllm/models/deepseek_v4/nvidia/prepared_weight_loading.py"
COMPOSE = ROOT / "docker-compose.yml"


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Nvfp4DualDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = _load(POLICY, "_nvfp4_dual_decode_policy_under_test")
        cls.prepared = _load(PREPARED, "_nvfp4_dual_prepared_under_test")

    def test_default_cutover_keeps_m1_and_prefill_on_cutlass(self) -> None:
        bounds = self.policy.validate_dual_decode_bounds(2, 8)
        selected = {
            m: self.policy.use_w4a16_decode(m, bounds, uniform_decode=True)
            for m in (1, 2, 4, 8, 9, 128)
        }
        self.assertEqual(
            selected,
            {1: False, 2: True, 4: True, 8: True, 9: False, 128: False},
        )

    def test_tiny_prefill_never_uses_w4a16_decode_branch(self) -> None:
        bounds = self.policy.validate_dual_decode_bounds(2, 8)
        for m in range(2, 9):
            with self.subTest(m=m):
                self.assertFalse(
                    self.policy.use_w4a16_decode(
                        m, bounds, uniform_decode=False
                    )
                )

    def test_cutover_bounds_fail_closed(self) -> None:
        for minimum, maximum in ((1, 8), (0, 8), (9, 8), (-1, -1)):
            with self.subTest(minimum=minimum, maximum=maximum):
                with self.assertRaises(ValueError):
                    self.policy.validate_dual_decode_bounds(minimum, maximum)

    def test_oracle_gate_is_default_off_and_class_is_explicit(self) -> None:
        env_source = ENVS.read_text()
        oracle_source = ORACLE.read_text()
        compose_source = COMPOSE.read_text()
        self.assertIn('os.getenv("VLLM_NVFP4_W4A16_DUAL_DECODE", "0")', env_source)
        self.assertIn('os.getenv("VLLM_NVFP4_NATIVE_B12X", "0")', env_source)
        self.assertIn("if envs.VLLM_NVFP4_NATIVE_B12X:", oracle_source)
        self.assertIn("if envs.VLLM_NVFP4_W4A16_DUAL_DECODE:", oracle_source)
        self.assertIn("NvFp4NativeB12xExperts", oracle_source)
        self.assertIn("NvFp4CutlassW4A16DualExperts", oracle_source)
        for name, default in (
            ("VLLM_NVFP4_W4A16_DUAL_DECODE", "0"),
            ("VLLM_NVFP4_NATIVE_B12X", "0"),
            ("VLLM_NVFP4_W4A16_DECODE_MIN_M", "2"),
            ("VLLM_NVFP4_W4A16_DECODE_MAX_M", "8"),
            ("B12X_W4A16_TC_DECODE", "0"),
            ("B12X_W4A16_SMALL_M_DIRECT", "0"),
            ("B12X_W4A16_E8M0_FINITE_FAST", "0"),
            ("B12X_W4A16_E8M0_K32_SCALE_REUSE", "0"),
            ("B12X_W4A16_MODELOPT_VECTOR_LOAD", "0"),
            ("B12X_W4A16_MODELOPT_FC1_TILE", ""),
            ("B12X_W4A16_MODELOPT_FC2_TILE", ""),
        ):
            self.assertIn(f'${{{name}:-{default}}}', compose_source)

    def test_expert_keeps_single_weight_storage_and_two_branches(self) -> None:
        source = EXPERT.read_text()
        self.assertIn("class NvFp4CutlassW4A16DualExperts(FlashInferExperts)", source)
        self.assertIn("return super().apply(", source)
        self.assertIn("_run_b12x_moe_fp4(", source)
        self.assertIn("duplicate_weight_bytes=0", source)
        self.assertIn("source.untyped_storage().data_ptr()", source)
        self.assertIn('source_format="fp4_e8m0_k32"', source)
        self.assertIn('w13_layout="w13"', source)
        self.assertIn('if weight_layout == "packed":', source)
        self.assertIn("from b12x.integration.tp_moe import (", source)
        self.assertIn('w4a16_weight_layout="modelopt"', source)
        self.assertIn("NVFP4_DUAL_DECODE event=selected", source)
        self.assertIn("uniform_decode=true", source)

    def test_native_b12x_reuses_storage_and_has_no_cutlass_branch(self) -> None:
        source = EXPERT.read_text()
        self.assertIn(
            "class NvFp4NativeB12xExperts(NvFp4CutlassW4A16DualExperts)",
            source,
        )
        self.assertIn("prepare_w4a16_fp4_e8m0_k32_weights(", source)
        self.assertIn("reuse_input_storage=True", source)
        self.assertIn('prepared.weight_layout != "packed"', source)
        self.assertIn("NVFP4_NATIVE_B12X event=prepared", source)
        native_source = source.split("class NvFp4NativeB12xExperts", 1)[1]
        self.assertNotIn("return super().apply(", native_source)

    def test_native_packed_planner_uses_pinned_caps_abi(self) -> None:
        tree = ast.parse(EXPERT.read_text())
        caps_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "TPMoEScratchCaps"
        ]
        self.assertEqual(len(caps_calls), 1)
        keywords = {item.arg for item in caps_calls[0].keywords}
        self.assertNotIn("w4a16_weight_layout", keywords)
        self.assertTrue(
            {
                "quant_mode",
                "source_format",
                "w13_layout",
                "swiglu_limit",
                "frozen",
            }.issubset(keywords)
        )

    def test_scratch_helper_preserves_default_and_forwards_explicit_layout(self) -> None:
        source = B12X_EXPERT.read_text()
        self.assertIn("w4a16_weight_layout: str | None = None", source)
        self.assertIn(
            "w4a16_weight_layout=w4a16_weight_layout",
            source,
        )
        self.assertEqual(source.count("w4a16_weight_layout"), 3)

    def test_runtime_expert_source_pin_matches_exact_file(self) -> None:
        observed = hashlib.sha256(EXPERT.read_bytes()).hexdigest()
        self.assertEqual(
            self.prepared.PINNED_DUAL_DECODE_EXPERTS_SHA256,
            observed,
        )

    def test_runtime_policy_source_pin_matches_exact_file(self) -> None:
        observed = hashlib.sha256(POLICY.read_bytes()).hexdigest()
        self.assertEqual(
            self.prepared.PINNED_DUAL_DECODE_POLICY_SHA256,
            observed,
        )

    def test_prepared_finalizer_initializes_dual_sidecar_once(self) -> None:
        helper = self.prepared
        backend = object()
        calls: list[object] = []
        experts_cls = type("NvFp4CutlassW4A16DualExperts", (), {})
        routed = SimpleNamespace(
            w13_weight_scale_2=object(),
            w2_weight_scale_2=object(),
            w13_input_scale=object(),
            w2_input_scale=object(),
            w13_weight_scale=object(),
            w2_weight_scale=object(),
            swiglu_limit=10.0,
            _expert_routing_tables=lambda: ("r0", "r1", "r2"),
        )
        quant_method = SimpleNamespace(
            nvfp4_backend=backend,
            moe_quant_config=None,
            moe_kernel=None,
            moe="moe",
            experts_cls=experts_cls,
        )
        fused_experts = SimpleNamespace(
            initialize_prepared_w4a16_decode=lambda layer: calls.append(layer),
            _w4a16_additional_scale_bytes=1234,
        )
        kernel = SimpleNamespace(fused_experts=fused_experts)
        state = helper.PreparedPostloadState(0, loaded=True)
        helper._finalize_prepared_cutlass(
            quant_method,
            routed,
            state,
            quant_config_factory=lambda **kwargs: SimpleNamespace(**kwargs),
            kernel_factory=lambda **kwargs: kernel,
            expected_backend=backend,
        )
        self.assertTrue(state.finalized)
        self.assertEqual(calls, [routed])
        with self.assertRaisesRegex(RuntimeError, "loaded/unfinalized"):
            helper._finalize_prepared_cutlass(
                quant_method,
                routed,
                state,
                quant_config_factory=lambda **kwargs: object(),
                kernel_factory=lambda **kwargs: kernel,
                expected_backend=backend,
            )

    def test_prepared_hook_accepts_only_exact_dual_identity(self) -> None:
        helper = self.prepared

        def original(self, layer):
            del self, layer

        original.__module__ = "vllm.model_executor.layers.quantization.modelopt"
        original.__qualname__ = (
            "ModelOptNvFp4FusedMoE.process_weights_after_loading"
        )
        method_cls = type(
            "ModelOptNvFp4FusedMoE", (), {"process_weights_after_loading": original}
        )
        experts_cls = type("NvFp4CutlassW4A16DualExperts", (), {})
        experts_cls.__module__ = (
            "vllm.model_executor.layers.fused_moe.experts.nvfp4_dual_decode_moe"
        )
        method = method_cls()
        method.nvfp4_backend = SimpleNamespace(value=helper.PREPARED_BACKEND)
        method.experts_cls = experts_cls
        routed = SimpleNamespace(quant_method=method)
        helper._install_prepared_postload_hook(
            routed, helper.PreparedPostloadState(0, loaded=True)
        )
        self.assertTrue(callable(method.process_weights_after_loading))


if __name__ == "__main__":
    unittest.main()
