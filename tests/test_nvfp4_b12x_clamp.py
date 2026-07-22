# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from __future__ import annotations

import ast
import importlib.util
import math
import sys
import unittest
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
EXPERT_PATH = ROOT / (
    "overlay/vllm/model_executor/layers/fused_moe/experts/"
    "flashinfer_b12x_moe.py"
)
ORACLE_PATH = ROOT / (
    "overlay/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py"
)


class _MoEActivation(Enum):
    SILU = "silu"
    RELU2_NO_MUL = "relu2_no_mul"


class _Platform:
    def __init__(self) -> None:
        self.family = 121

    @staticmethod
    def is_cuda() -> bool:
        return True

    def is_device_capability_family(self, family: int) -> bool:
        return family == 120 and self.family in (120, 121)


class _ExpertsBase:
    def __init__(self, moe_config, quant_config) -> None:
        self.moe_config = moe_config
        self.quant_config = quant_config
        self.w1_scale = None
        self.w2_scale = None
        self.g1_alphas = None
        self.g2_alphas = None
        self.a2_gscale = None


class _Tensor:
    pass


class _RuntimeTensor:
    def __init__(
        self,
        *,
        shape=(4, 7168),
        dtype=None,
        device="cuda:0",
        data_ptr=1000,
        contiguous=True,
    ) -> None:
        self.shape = shape
        self.dtype = _TORCH.bfloat16 if dtype is None else dtype
        self.device = device
        self._data_ptr = data_ptr
        self._contiguous = contiguous
        self.to_calls = []

    def is_contiguous(self) -> bool:
        return self._contiguous

    def data_ptr(self) -> int:
        return self._data_ptr

    def to(self, dtype):
        self.to_calls.append(dtype)
        return _RuntimeTensor(
            shape=self.shape,
            dtype=dtype,
            device=self.device,
            data_ptr=self._data_ptr + 1,
        )


_TORCH = ModuleType("torch")
_TORCH.Tensor = _Tensor
_TORCH.dtype = type("dtype", (), {})
_TORCH.nn = SimpleNamespace(Module=type("Module", (), {}))
_TORCH.bfloat16 = object()
_TORCH.float32 = object()
_TORCH.int32 = object()
_TORCH.ones = lambda *args, **kwargs: None
_TORCH.cuda = SimpleNamespace(current_device=lambda: 0)


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    return module


def _load_expert_module():
    modules = {
        name: _package(name)
        for name in (
            "vllm",
            "vllm.model_executor",
            "vllm.model_executor.layers",
            "vllm.model_executor.layers.fused_moe",
            "vllm.model_executor.layers.fused_moe.experts",
            "vllm.model_executor.layers.quantization",
            "vllm.model_executor.layers.quantization.utils",
            "vllm.platforms",
            "vllm.utils",
        )
    }
    modules["torch"] = _TORCH

    mk = ModuleType("vllm.model_executor.layers.fused_moe.modular_kernel")
    mk.FusedMoEExpertsModular = _ExpertsBase
    mk.FusedMoEActivationFormat = SimpleNamespace(Standard=object())
    mk.TopKWeightAndReduce = type("TopKWeightAndReduce", (), {})
    mk.ExpertTokensMetadata = type("ExpertTokensMetadata", (), {})
    modules[mk.__name__] = mk

    activation = ModuleType("vllm.model_executor.layers.fused_moe.activation")
    activation.MoEActivation = _MoEActivation
    modules[activation.__name__] = activation

    config = ModuleType("vllm.model_executor.layers.fused_moe.config")
    config.FusedMoEConfig = type("FusedMoEConfig", (), {})
    config.FusedMoEParallelConfig = type("FusedMoEParallelConfig", (), {})
    config.FusedMoEQuantConfig = type("FusedMoEQuantConfig", (), {})
    modules[config.__name__] = config

    reduce_module = ModuleType(
        "vllm.model_executor.layers.fused_moe.topk_weight_and_reduce"
    )
    reduce_module.TopKWeightAndReduceNoOP = type(
        "TopKWeightAndReduceNoOP", (), {}
    )
    modules[reduce_module.__name__] = reduce_module

    quant = ModuleType(
        "vllm.model_executor.layers.quantization.utils.quant_utils"
    )
    quant.QuantKey = type("QuantKey", (), {})
    quant.kNvfp4Dynamic = object()
    quant.kNvfp4Static = object()
    modules[quant.__name__] = quant

    platform = _Platform()
    modules["vllm.platforms"].current_platform = platform

    flashinfer_utils = ModuleType("vllm.utils.flashinfer")
    flashinfer_utils.flashinfer_convert_sf_to_mma_layout = lambda *a, **k: None
    flashinfer_utils.has_flashinfer_b12x_moe = lambda: True
    modules[flashinfer_utils.__name__] = flashinfer_utils

    spec = importlib.util.spec_from_file_location(
        "_test_flashinfer_b12x_moe", EXPERT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module, platform, quant


def _moe_config(*, limit: float | None) -> SimpleNamespace:
    return SimpleNamespace(
        in_dtype=_TORCH.bfloat16,
        num_local_experts=256,
        moe_parallel_config=SimpleNamespace(ep_rank=0),
        num_experts=256,
        experts_per_token=8,
        hidden_dim=7168,
        intermediate_size_per_partition=2048,
        max_num_tokens=4096,
        max_capture_size=64,
        activation=_MoEActivation.SILU,
        swiglu_limit=limit,
        swiglu_alpha=None,
        swiglu_beta=None,
    )


def _quant_config(*, limit: float | None) -> SimpleNamespace:
    return SimpleNamespace(
        quant_dtype="nvfp4",
        gemm1_clamp_limit=limit,
        gemm1_alpha=None,
        gemm1_beta=None,
    )


class B12xClampAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module, cls.platform, cls.quant_keys = _load_expert_module()

    def setUp(self) -> None:
        self.module._B12X_WRAPPER_CACHE.clear()

    def tearDown(self) -> None:
        self.module._B12X_WRAPPER_CACHE.clear()

    def test_deepseek_clamp_maps_to_b12x_oai_parameters(self) -> None:
        experts = self.module.FlashInferB12xExperts(
            _moe_config(limit=None),
            _quant_config(limit=10.0),
        )

        self.assertEqual(experts._activation_str, "swigluoai_uninterleave")
        self.assertEqual(experts._swiglu_alpha, 1.0)
        self.assertEqual(experts._swiglu_beta, 0.0)
        self.assertEqual(experts._swiglu_limit, 10.0)

        recorded = {}

        class _Wrapper:
            def __init__(self, **kwargs) -> None:
                recorded.update(kwargs)

        flashinfer = _package("flashinfer")
        fused_moe = ModuleType("flashinfer.fused_moe")
        fused_moe.B12xMoEWrapper = _Wrapper
        flashinfer.fused_moe = fused_moe
        with patch.dict(
            sys.modules,
            {"flashinfer": flashinfer, "flashinfer.fused_moe": fused_moe},
        ):
            experts._ensure_wrapper()

        self.assertEqual(recorded["activation"], "swigluoai_uninterleave")
        self.assertEqual(recorded["swiglu_alpha"], 1.0)
        self.assertEqual(recorded["swiglu_beta"], 0.0)
        self.assertEqual(recorded["swiglu_limit"], 10.0)
        self.assertEqual(recorded["quant_mode"], "nvfp4")
        self.assertEqual(recorded["source_format"], "modelopt")
        self.assertEqual(recorded["device"], "cuda:0")

    def test_43_target_layers_share_exactly_one_wrapper(self) -> None:
        scope = object()
        configs = []
        for _ in range(43):
            config = _moe_config(limit=None)
            config._b12x_wrapper_scope = scope
            config._b12x_wrapper_concurrent_execution = False
            configs.append(config)
        experts = [
            self.module.FlashInferB12xExperts(
                config,
                _quant_config(limit=10.0),
            )
            for config in configs
        ]
        constructed = []

        class _Wrapper:
            def __init__(self, **kwargs) -> None:
                constructed.append(kwargs)

        flashinfer = _package("flashinfer")
        fused_moe = ModuleType("flashinfer.fused_moe")
        fused_moe.B12xMoEWrapper = _Wrapper
        flashinfer.fused_moe = fused_moe
        with patch.dict(
            sys.modules,
            {"flashinfer": flashinfer, "flashinfer.fused_moe": fused_moe},
        ):
            wrappers = [expert._ensure_wrapper() for expert in experts]

        self.assertEqual(len(constructed), 1)
        self.assertTrue(all(wrapper is wrappers[0] for wrapper in wrappers))

    def test_micro_decode_and_prefill_use_distinct_shared_capacities(self) -> None:
        scope = object()
        experts = []
        for _ in range(43):
            config = _moe_config(limit=None)
            config._b12x_wrapper_scope = scope
            config._b12x_wrapper_concurrent_execution = False
            experts.append(
                self.module.FlashInferB12xExperts(
                    config,
                    _quant_config(limit=10.0),
                )
            )
        constructed = []

        class _Wrapper:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs
                constructed.append(self)

        flashinfer = _package("flashinfer")
        fused_moe = ModuleType("flashinfer.fused_moe")
        fused_moe.B12xMoEWrapper = _Wrapper
        flashinfer.fused_moe = fused_moe
        with patch.dict(
            sys.modules,
            {"flashinfer": flashinfer, "flashinfer.fused_moe": fused_moe},
        ):
            micro = [expert._ensure_wrapper(4) for expert in experts]
            decode = [expert._ensure_wrapper(5) for expert in experts]
            prefill = [expert._ensure_wrapper(65) for expert in experts]

        self.assertEqual(len(constructed), 3)
        self.assertEqual(
            sorted(wrapper.kwargs["max_num_tokens"] for wrapper in constructed),
            [4, 64, 4096],
        )
        self.assertTrue(all(wrapper is micro[0] for wrapper in micro))
        self.assertTrue(all(wrapper is decode[0] for wrapper in decode))
        self.assertTrue(all(wrapper is prefill[0] for wrapper in prefill))
        self.assertIsNot(micro[0], decode[0])
        self.assertIsNot(decode[0], prefill[0])

    def test_micro_capacity_never_exceeds_capture_frontier(self) -> None:
        config = _moe_config(limit=None)
        config.max_capture_size = 2
        experts = self.module.FlashInferB12xExperts(
            config,
            _quant_config(limit=10.0),
        )
        constructed = []

        class _Wrapper:
            def __init__(self, **kwargs) -> None:
                constructed.append(kwargs)

        flashinfer = _package("flashinfer")
        fused_moe = ModuleType("flashinfer.fused_moe")
        fused_moe.B12xMoEWrapper = _Wrapper
        flashinfer.fused_moe = fused_moe
        with patch.dict(
            sys.modules,
            {"flashinfer": flashinfer, "flashinfer.fused_moe": fused_moe},
        ):
            experts._ensure_wrapper(2)

        self.assertEqual(constructed[0]["max_num_tokens"], 2)

    def test_wrapper_capacity_rejects_out_of_range_tokens(self) -> None:
        experts = self.module.FlashInferB12xExperts(
            _moe_config(limit=None),
            _quant_config(limit=10.0),
        )
        with self.assertRaisesRegex(ValueError, "token count"):
            experts._ensure_wrapper(0)
        with self.assertRaisesRegex(ValueError, "exceeds configured capacity"):
            experts._ensure_wrapper(4097)

    def test_independent_model_scopes_do_not_share_wrapper(self) -> None:
        experts = []
        for _ in range(2):
            config = _moe_config(limit=None)
            config._b12x_wrapper_scope = object()
            experts.append(
                self.module.FlashInferB12xExperts(
                    config,
                    _quant_config(limit=10.0),
                )
            )
        constructed = []

        class _Wrapper:
            def __init__(self, **kwargs) -> None:
                constructed.append(kwargs)

        flashinfer = _package("flashinfer")
        fused_moe = ModuleType("flashinfer.fused_moe")
        fused_moe.B12xMoEWrapper = _Wrapper
        flashinfer.fused_moe = fused_moe
        with patch.dict(
            sys.modules,
            {"flashinfer": flashinfer, "flashinfer.fused_moe": fused_moe},
        ):
            wrappers = [expert._ensure_wrapper() for expert in experts]

        self.assertEqual(len(constructed), 2)
        self.assertIsNot(wrappers[0], wrappers[1])

    def test_bound_wrapper_fast_path_skips_device_and_cache_work(self) -> None:
        experts = self.module.FlashInferB12xExperts(
            _moe_config(limit=None),
            _quant_config(limit=10.0),
        )
        constructed = []
        device_queries = []

        class _Wrapper:
            def __init__(self, **kwargs) -> None:
                constructed.append(kwargs)

        flashinfer = _package("flashinfer")
        fused_moe = ModuleType("flashinfer.fused_moe")
        fused_moe.B12xMoEWrapper = _Wrapper
        flashinfer.fused_moe = fused_moe
        original_current_device = _TORCH.cuda.current_device
        _TORCH.cuda.current_device = lambda: device_queries.append(True) or 0
        self.addCleanup(
            setattr,
            _TORCH.cuda,
            "current_device",
            original_current_device,
        )
        with patch.dict(
            sys.modules,
            {"flashinfer": flashinfer, "flashinfer.fused_moe": fused_moe},
        ):
            first = experts._ensure_wrapper()
            second = experts._ensure_wrapper()

        self.assertIs(first, second)
        self.assertEqual(len(constructed), 1)
        self.assertEqual(len(device_queries), 1)

    def test_shared_wrapper_rejects_concurrent_ubatching(self) -> None:
        config = _moe_config(limit=None)
        config._b12x_wrapper_scope = object()
        config._b12x_wrapper_concurrent_execution = True

        with self.assertRaisesRegex(ValueError, "non-reentrant"):
            self.module.FlashInferB12xExperts(
                config,
                _quant_config(limit=10.0),
            )

    def _runtime_experts(self):
        experts = self.module.FlashInferB12xExperts(
            _moe_config(limit=None),
            _quant_config(limit=10.0),
        )
        experts.w1_scale = object()
        experts.w2_scale = object()
        experts.g1_alphas = object()
        experts.g2_alphas = object()
        experts._fc2_input_scale = object()
        experts.w1_sf_mma = object()
        experts.w2_sf_mma = object()
        return experts

    def test_direct_output_alias_is_explicit_and_avoids_both_copies(self) -> None:
        experts = self._runtime_experts()
        output = _RuntimeTensor(data_ptr=4104)
        hidden = _RuntimeTensor(data_ptr=4200)
        topk_ids = _RuntimeTensor(
            shape=(4, 8), dtype=_TORCH.int32, data_ptr=4300
        )
        observed = {}

        class _Wrapper:
            use_cuda_graph = True
            _moe_output = _RuntimeTensor(data_ptr=9999)

            def run(self, **kwargs):
                observed.update(kwargs)
                return _RuntimeTensor(
                    shape=self._moe_output.shape,
                    dtype=self._moe_output.dtype,
                    device=self._moe_output.device,
                    data_ptr=self._moe_output.data_ptr(),
                )

        wrapper = _Wrapper()
        experts._ensure_wrapper = lambda *_: wrapper
        self.assertTrue(experts.supports_output_alias)

        experts.apply(
            output,
            hidden,
            object(),
            object(),
            _RuntimeTensor(shape=(4, 8)),
            topk_ids,
            _MoEActivation.SILU,
            256,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

        self.assertIs(wrapper._moe_output, output)
        self.assertIs(observed["token_selected_experts"], topk_ids)
        self.assertEqual(topk_ids.to_calls, [])

    def test_direct_output_alias_rejects_pointer_drift(self) -> None:
        experts = self._runtime_experts()
        output = _RuntimeTensor(data_ptr=4104)
        hidden = _RuntimeTensor(data_ptr=4200)

        class _Wrapper:
            use_cuda_graph = True
            _moe_output = _RuntimeTensor(data_ptr=9999)

            def run(self, **kwargs):
                return _RuntimeTensor(data_ptr=12345)

        experts._ensure_wrapper = lambda *_: _Wrapper()
        with self.assertRaisesRegex(RuntimeError, "aliased output buffer"):
            experts.apply(
                output,
                hidden,
                object(),
                object(),
                _RuntimeTensor(shape=(4, 8)),
                _RuntimeTensor(shape=(4, 8), dtype=_TORCH.int32),
                _MoEActivation.SILU,
                256,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

    def test_direct_output_alias_rejects_noncontiguous_output(self) -> None:
        experts = self._runtime_experts()
        experts._ensure_wrapper = lambda *_: SimpleNamespace(
            use_cuda_graph=True, _moe_output=object()
        )
        with self.assertRaisesRegex(RuntimeError, "contract mismatch"):
            experts.apply(
                _RuntimeTensor(contiguous=False),
                _RuntimeTensor(),
                object(),
                object(),
                _RuntimeTensor(shape=(4, 8)),
                _RuntimeTensor(shape=(4, 8), dtype=_TORCH.int32),
                _MoEActivation.SILU,
                256,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

    def test_unclamped_silu_retains_plain_mapping(self) -> None:
        experts = self.module.FlashInferB12xExperts(
            _moe_config(limit=None),
            _quant_config(limit=None),
        )

        self.assertEqual(experts._activation_str, "silu")
        self.assertEqual(experts._swiglu_alpha, 1.702)
        self.assertEqual(experts._swiglu_beta, 1.0)
        self.assertIsNone(experts._swiglu_limit)

    def test_configured_clamp_parameters_override_silu_defaults(self) -> None:
        config = _moe_config(limit=9.0)
        config.swiglu_alpha = 1.25
        config.swiglu_beta = -0.5
        experts = self.module.FlashInferB12xExperts(
            config,
            _quant_config(limit=None),
        )

        self.assertEqual(
            (
                experts._activation_str,
                experts._swiglu_alpha,
                experts._swiglu_beta,
                experts._swiglu_limit,
            ),
            ("swigluoai_uninterleave", 1.25, -0.5, 9.0),
        )

    def test_clamped_formula_matches_reference_beyond_both_limits(self) -> None:
        activation, alpha, beta, limit = self.module._resolve_b12x_activation(
            "silu", None, None, 10.0
        )
        self.assertEqual(activation, "swigluoai_uninterleave")
        assert limit is not None

        gate = [-20.0, -10.0, -1.0, 0.0, 9.0, 10.0, 20.0]
        up = [-20.0, -10.0, -2.0, 0.0, 8.0, 10.0, 20.0]
        for gate_value, up_value in zip(gate, up, strict=True):
            reference_gate = min(gate_value, limit)
            reference_up = max(min(up_value, limit), -limit)
            reference = (
                reference_gate
                * (1.0 / (1.0 + math.exp(-reference_gate)))
                * reference_up
            )

            b12x_gate = min(gate_value, limit)
            b12x_up = max(min(up_value, limit), -limit)
            b12x = (
                b12x_gate
                * (1.0 / (1.0 + math.exp(-(alpha * b12x_gate))))
                * (b12x_up + beta)
            )
            self.assertEqual(b12x, reference)

    def test_sm120_family_and_tp_are_supported_but_ep_is_not(self) -> None:
        self.platform.family = 120
        self.assertTrue(self.module.FlashInferB12xExperts._supports_current_device())
        self.platform.family = 121
        self.assertTrue(self.module.FlashInferB12xExperts._supports_current_device())
        self.platform.family = 100
        self.assertFalse(self.module.FlashInferB12xExperts._supports_current_device())
        self.platform.family = 121

        self.assertTrue(
            self.module.FlashInferB12xExperts._supports_parallel_config(
                SimpleNamespace(use_ep=False)
            )
        )
        self.assertFalse(
            self.module.FlashInferB12xExperts._supports_parallel_config(
                SimpleNamespace(use_ep=True)
            )
        )
        self.assertTrue(
            self.module.FlashInferB12xExperts._supports_quant_scheme(
                self.quant_keys.kNvfp4Static,
                self.quant_keys.kNvfp4Dynamic,
            )
        )


class _NvFp4MoeBackend(Enum):
    FLASHINFER_TRTLLM = "FLASHINFER_TRTLLM"
    FLASHINFER_CUTLASS = "FLASHINFER_CUTLASS"
    FLASHINFER_CUTEDSL = "FLASHINFER_CUTEDSL"
    FLASHINFER_CUTEDSL_BATCHED = "FLASHINFER_CUTEDSL_BATCHED"
    FLASHINFER_B12X = "FLASHINFER_B12X"
    VLLM_CUTLASS = "VLLM_CUTLASS"
    MARLIN = "MARLIN"
    HUMMING = "HUMMING"
    EMULATION = "EMULATION"


def _load_oracle_select(*, observed: list[_NvFp4MoeBackend]):
    tree = ast.parse(ORACLE_PATH.read_text())
    select_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "select_nvfp4_moe_backend"
    )
    module = ast.fix_missing_locations(ast.Module(body=[select_node], type_ignores=[]))

    class _Kernel:
        @staticmethod
        def is_supported_config(cls, config, weight_key, activation_key, fmt):
            del cls, config, weight_key, activation_key, fmt
            return True, None

    def backend_to_kernel_cls(backend):
        observed.append(backend)
        return [_Kernel]

    namespace = {
        "FusedMoEConfig": object,
        "QuantKey": object,
        "NvFp4MoeBackend": _NvFp4MoeBackend,
        "mk": SimpleNamespace(
            FusedMoEExperts=object,
            FusedMoEActivationFormat=SimpleNamespace(
                Standard="standard", BatchedExperts="batched"
            ),
        ),
        "map_nvfp4_backend": lambda backend: {
            "flashinfer_b12x": _NvFp4MoeBackend.FLASHINFER_B12X
        }[backend],
        "backend_to_kernel_cls": backend_to_kernel_cls,
        "logger": SimpleNamespace(
            info_once=lambda *a, **k: None,
            debug_once=lambda *a, **k: None,
        ),
        "envs": SimpleNamespace(VLLM_TEST_FORCE_FP8_MARLIN=False),
    }
    exec(compile(module, ORACLE_PATH, "exec"), namespace)
    return namespace["select_nvfp4_moe_backend"]


class NvFp4OracleClampTest(unittest.TestCase):
    def test_clamped_b12x_is_explicitly_accepted(self) -> None:
        observed = []
        select = _load_oracle_select(observed=observed)
        config = SimpleNamespace(
            moe_backend="flashinfer_b12x",
            swiglu_limit=10.0,
            moe_parallel_config=SimpleNamespace(
                use_batched_activation_format=False
            ),
        )

        backend, _ = select(config, object(), object())

        self.assertEqual(backend, _NvFp4MoeBackend.FLASHINFER_B12X)
        self.assertEqual(observed, [_NvFp4MoeBackend.FLASHINFER_B12X])

    def test_b12x_remains_excluded_from_auto_selection(self) -> None:
        observed = []
        select = _load_oracle_select(observed=observed)
        config = SimpleNamespace(
            moe_backend="auto",
            swiglu_limit=10.0,
            moe_parallel_config=SimpleNamespace(
                use_batched_activation_format=False
            ),
        )

        select(config, object(), object())

        self.assertNotIn(_NvFp4MoeBackend.FLASHINFER_B12X, observed)


if __name__ == "__main__":
    unittest.main()
