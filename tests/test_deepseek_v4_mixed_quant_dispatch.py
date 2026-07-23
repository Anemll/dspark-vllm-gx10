# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
QUANT_CONFIG_PATH = ROOT / "overlay/vllm/models/deepseek_v4/quant_config.py"


class _RoutedExperts:
    def __init__(self, moe_backend: str = "auto") -> None:
        self.moe_config = SimpleNamespace(moe_backend=moe_backend)


class _UnquantizedFusedMoEMethod:
    def __init__(self, moe_config) -> None:
        self.moe_config = moe_config


class _Mxfp4MoEMethod:
    def __init__(self, moe_config) -> None:
        self.moe_config = moe_config


class _ModelOptNvFp4Config:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _ModelOptNvFp4FusedMoE:
    def __init__(self, *, quant_config, moe_config) -> None:
        self.quant_config = quant_config
        self.moe_config = moe_config


class _Fp8Fallback:
    pass


class _Fp8Config:
    def __init__(self, *args, **kwargs) -> None:
        del args
        self.ignored_layers = kwargs.get("ignored_layers") or []
        self.packed_modules_mapping = {}

    def get_quant_method(self, layer, prefix):
        del layer, prefix
        return _Fp8Fallback()


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _module(name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _is_layer_skipped(*, prefix, ignored_layers, fused_mapping) -> bool:
    del fused_mapping
    return prefix in ignored_layers


class DeepseekV4MixedQuantDispatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.hf_config = SimpleNamespace(
            expert_dtype="fp4",
            num_hidden_layers=43,
            quantization_config={"moe_quant_algo": "NVFP4"},
        )
        self.current_config = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=self.hf_config),
            parallel_config=SimpleNamespace(
                enable_dbo=False,
                ubatch_size=0,
                use_ubatching=False,
            ),
        )

        modules = {
            name: _package(name)
            for name in (
                "vllm",
                "vllm.model_executor",
                "vllm.model_executor.layers",
                "vllm.model_executor.layers.quantization",
                "vllm.model_executor.layers.quantization.utils",
                "vllm.models",
                "vllm.models.deepseek_v4",
                "vllm.models.deepseek_v4.nvidia",
            )
        }
        modules.update(
            {
                "vllm.config": _module(
                    "vllm.config",
                    get_current_vllm_config=lambda: self.current_config,
                ),
                "vllm.logger": _module(
                    "vllm.logger",
                    init_logger=lambda name: SimpleNamespace(
                        info_once=lambda *args, **kwargs: None
                    ),
                ),
                "vllm.model_executor.layers.fused_moe": _module(
                    "vllm.model_executor.layers.fused_moe",
                    RoutedExperts=_RoutedExperts,
                    UnquantizedFusedMoEMethod=_UnquantizedFusedMoEMethod,
                ),
                "vllm.model_executor.layers.quantization.fp8": _module(
                    "vllm.model_executor.layers.quantization.fp8",
                    Fp8Config=_Fp8Config,
                ),
                "vllm.model_executor.layers.quantization.mxfp4": _module(
                    "vllm.model_executor.layers.quantization.mxfp4",
                    Mxfp4MoEMethod=_Mxfp4MoEMethod,
                ),
                "vllm.model_executor.layers.quantization.modelopt": _module(
                    "vllm.model_executor.layers.quantization.modelopt",
                    ModelOptNvFp4Config=_ModelOptNvFp4Config,
                    ModelOptNvFp4FusedMoE=_ModelOptNvFp4FusedMoE,
                ),
                "vllm.model_executor.layers.quantization.utils.quant_utils": (
                    _module(
                        "vllm.model_executor.layers.quantization.utils.quant_utils",
                        is_layer_skipped=_is_layer_skipped,
                    )
                ),
                "vllm.models.deepseek_v4.nvidia.prepared_weight_loading": (
                    _module(
                        "vllm.models.deepseek_v4.nvidia.prepared_weight_loading",
                        prepared_load_requested=lambda: False,
                    )
                ),
            }
        )
        modules[
            "vllm.model_executor.layers.quantization"
        ].QuantizationMethods = str

        module_patcher = patch.dict(sys.modules, modules)
        module_patcher.start()
        self.addCleanup(module_patcher.stop)

        spec = importlib.util.spec_from_file_location(
            "_deepseek_v4_quant_config_under_test", QUANT_CONFIG_PATH
        )
        if spec is None or spec.loader is None:
            self.fail(f"Could not load quantization config from {QUANT_CONFIG_PATH}")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def _new_config(self):
        return self.module.DeepseekV4FP8Config()

    def test_target_layers_use_modelopt_nvfp4(self) -> None:
        for layer_index in (0, 1, 41, 42):
            with self.subTest(layer_index=layer_index):
                config = self._new_config()
                prefix = f"model.layers.{layer_index}.ffn.experts"
                method = config.get_quant_method(_RoutedExperts(), prefix)

                self.assertIsInstance(method, _ModelOptNvFp4FusedMoE)
                self.assertFalse(config.is_mxfp4_quant(prefix, _RoutedExperts()))

    def test_target_layers_share_one_model_scoped_b12x_wrapper_token(self) -> None:
        config = self._new_config()
        first = config.get_quant_method(
            _RoutedExperts(), "model.layers.0.ffn.experts"
        )
        last = config.get_quant_method(
            _RoutedExperts(), "model.layers.42.ffn.experts"
        )
        another_model = self._new_config().get_quant_method(
            _RoutedExperts(), "model.layers.0.ffn.experts"
        )

        self.assertIs(
            first.moe_config._b12x_wrapper_scope,
            last.moe_config._b12x_wrapper_scope,
        )
        self.assertIsNot(
            first.moe_config._b12x_wrapper_scope,
            another_model.moe_config._b12x_wrapper_scope,
        )
        self.assertFalse(first.moe_config._b12x_wrapper_concurrent_execution)

    def test_ubatching_is_propagated_as_unsafe_for_shared_wrapper(self) -> None:
        self.current_config.parallel_config.use_ubatching = True

        method = self._new_config().get_quant_method(
            _RoutedExperts(), "model.layers.0.ffn.experts"
        )

        self.assertTrue(method.moe_config._b12x_wrapper_concurrent_execution)

    def test_missing_ubatching_state_fails_closed(self) -> None:
        del self.current_config.parallel_config.use_ubatching

        with self.assertRaises(AttributeError):
            self._new_config().get_quant_method(
                _RoutedExperts(), "model.layers.0.ffn.experts"
            )

    def test_production_dspark_draft_prefixes_use_native_mxfp4(self) -> None:
        for prefix in (
            "model.layers.43.ffn.experts",
            "model.layers.44.ffn.experts",
            "model.layers.45.ffn.experts",
        ):
            with self.subTest(prefix=prefix):
                config = self._new_config()
                method = config.get_quant_method(_RoutedExperts(), prefix)

                self.assertIsInstance(method, _Mxfp4MoEMethod)
                self.assertTrue(config.is_mxfp4_quant(prefix, _RoutedExperts()))

    def test_prepared_target_scopes_cutlass_without_mutating_auto_draft(
        self,
    ) -> None:
        config = self._new_config()
        target = _RoutedExperts("auto")
        draft = _RoutedExperts("auto")

        with patch.object(
            self.module, "_prepared_nvfp4_load_requested", return_value=True
        ):
            target_method = config.get_quant_method(
                target, "model.layers.0.ffn.experts"
            )
            draft_method = config.get_quant_method(
                draft, "model.layers.43.ffn.experts"
            )

        self.assertIsInstance(target_method, _ModelOptNvFp4FusedMoE)
        self.assertEqual(target.moe_config.moe_backend, "flashinfer_cutlass")
        self.assertIsInstance(draft_method, _Mxfp4MoEMethod)
        self.assertEqual(draft.moe_config.moe_backend, "auto")

    def test_prepared_target_scopes_b12x_without_mutating_auto_draft(
        self,
    ) -> None:
        config = self._new_config()
        target = _RoutedExperts("auto")
        draft = _RoutedExperts("auto")

        with (
            patch.object(
                self.module, "_prepared_nvfp4_load_requested", return_value=True
            ),
            patch.dict(
                os.environ,
                {
                    "VLLM_DSV4_NVFP4_PREPARED_MOE_BACKEND": (
                        "flashinfer_b12x"
                    )
                },
            ),
        ):
            target_method = config.get_quant_method(
                target, "model.layers.0.ffn.experts"
            )
            draft_method = config.get_quant_method(
                draft, "model.layers.43.ffn.experts"
            )

        self.assertIsInstance(target_method, _ModelOptNvFp4FusedMoE)
        self.assertEqual(target.moe_config.moe_backend, "flashinfer_b12x")
        self.assertIsInstance(draft_method, _Mxfp4MoEMethod)
        self.assertEqual(draft.moe_config.moe_backend, "auto")

    def test_prepared_target_rejects_invalid_scoped_backend(self) -> None:
        target = _RoutedExperts("auto")
        with (
            patch.object(
                self.module, "_prepared_nvfp4_load_requested", return_value=True
            ),
            patch.dict(
                os.environ,
                {"VLLM_DSV4_NVFP4_PREPARED_MOE_BACKEND": "marlin"},
            ),
            self.assertRaisesRegex(
                ValueError,
                "VLLM_DSV4_NVFP4_PREPARED_MOE_BACKEND",
            ),
        ):
            self._new_config().get_quant_method(
                target, "model.layers.0.ffn.experts"
            )

    def test_prepared_target_keeps_target_only_explicit_cutlass_compatible(
        self,
    ) -> None:
        target = _RoutedExperts("flashinfer_cutlass")
        with patch.object(
            self.module, "_prepared_nvfp4_load_requested", return_value=True
        ):
            method = self._new_config().get_quant_method(
                target, "model.layers.42.ffn.experts"
            )

        self.assertIsInstance(method, _ModelOptNvFp4FusedMoE)
        self.assertEqual(target.moe_config.moe_backend, "flashinfer_cutlass")

    def test_prepared_target_keeps_explicit_b12x_for_conversion(self) -> None:
        target = _RoutedExperts("flashinfer_b12x")
        with patch.object(
            self.module, "_prepared_nvfp4_load_requested", return_value=True
        ):
            method = self._new_config().get_quant_method(
                target, "model.layers.0.ffn.experts"
            )

        self.assertIsInstance(method, _ModelOptNvFp4FusedMoE)
        self.assertEqual(target.moe_config.moe_backend, "flashinfer_b12x")

    def test_prepared_target_rejects_other_runner_wide_backends(self) -> None:
        target = _RoutedExperts("marlin")
        with (
            patch.object(
                self.module, "_prepared_nvfp4_load_requested", return_value=True
            ),
            self.assertRaisesRegex(ValueError, "requires runner moe_backend"),
        ):
            self._new_config().get_quant_method(
                target, "model.layers.0.ffn.experts"
            )

    def test_standard_mtp_layer_uses_native_mxfp4(self) -> None:
        config = self._new_config()

        method = config.get_quant_method(
            _RoutedExperts(), "model.mtp.layers.43.ffn.experts"
        )

        self.assertIsInstance(method, _Mxfp4MoEMethod)

    def test_official_ignore_contract_keeps_mixed_expert_dispatch(self) -> None:
        official_ignore = [
            "*.attn.*",
            "*.ffn.shared_experts.*",
            "head",
            "mtp.*",
        ]
        self.hf_config.quantization_config = {
            "quant_method": "fp8",
            "quant_algo": "MIXED_PRECISION",
            "moe_quant_algo": "NVFP4",
            "ignore": official_ignore,
        }
        config = self._new_config()

        target = config.get_quant_method(
            _RoutedExperts(), "model.layers.42.ffn.experts"
        )
        draft = config.get_quant_method(
            _RoutedExperts(), "model.layers.43.ffn.experts"
        )

        self.assertIsInstance(target, _ModelOptNvFp4FusedMoE)
        self.assertIsInstance(draft, _Mxfp4MoEMethod)
        self.assertNotIsInstance(draft, _UnquantizedFusedMoEMethod)

    def test_explicitly_ignored_expert_stays_unquantized(self) -> None:
        prefix = "model.layers.0.ffn.experts"
        config = self.module.DeepseekV4FP8Config(ignored_layers=[prefix])

        method = config.get_quant_method(_RoutedExperts(), prefix)

        self.assertIsInstance(method, _UnquantizedFusedMoEMethod)

    def test_native_mxfp4_checkpoint_keeps_all_fp4_experts_on_mxfp4(
        self,
    ) -> None:
        self.hf_config.quantization_config = {"moe_quant_algo": ""}
        config = self._new_config()

        target = config.get_quant_method(
            _RoutedExperts(), "model.layers.0.ffn.experts"
        )
        draft = config.get_quant_method(
            _RoutedExperts(), "model.layers.43.ffn.experts"
        )

        self.assertIsInstance(target, _Mxfp4MoEMethod)
        self.assertIsInstance(draft, _Mxfp4MoEMethod)

    def test_fp8_expert_checkpoint_falls_through_to_fp8_config(self) -> None:
        self.hf_config.expert_dtype = "fp8"
        config = self._new_config()

        method = config.get_quant_method(
            _RoutedExperts(), "model.layers.0.ffn.experts"
        )

        self.assertIsInstance(method, _Fp8Fallback)

    def test_unstructured_prefix_preserves_legacy_global_nvfp4_choice(
        self,
    ) -> None:
        config = self._new_config()

        method = config.get_quant_method(_RoutedExperts(), "model.experts")

        self.assertIsInstance(method, _ModelOptNvFp4FusedMoE)


if __name__ == "__main__":
    unittest.main()
