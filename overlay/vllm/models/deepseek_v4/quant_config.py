# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantization config for DeepSeek V4."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)

_DEEPSEEK_V4_EXPERT_DTYPES = ("fp4", "fp8")

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
    )


def _extract_deepseek_v4_layer_index(prefix: str) -> int | None:
    """Extract the decoder index following the last ``layers`` component."""
    parts = prefix.split(".")
    for index in range(len(parts) - 2, -1, -1):
        if parts[index] != "layers":
            continue
        try:
            return int(parts[index + 1])
        except ValueError:
            return None
    return None


class DeepseekV4FP8Config(Fp8Config):
    """FP8 config for DeepSeek V4 with expert-dtype-aware MoE dispatch.

    DeepSeek V4 checkpoints always use FP8 block quantization for
    linear/attention layers. The MoE expert weights vary by checkpoint:
    - ``expert_dtype="fp4"`` (e.g. DeepSeek-V4-Flash): MXFP4 experts
      with ue8m0 (e8m0fnu) FP8 linear scales.
    - ``expert_dtype="fp8"`` (e.g. DeepSeek-V4-Flash-Base): FP8 block
      experts with float32 FP8 linear scales.

    The dispatch and the linear scale dtype are both keyed off
    ``expert_dtype`` from the model's hf_config; missing values default
    to ``"fp4"`` so existing FP4 checkpoints stay unchanged.

    NOTE: ``expert_dtype`` is resolved lazily because this config is
    constructed during VllmConfig setup, before ``set_current_vllm_config``
    is active. Reading hf_config eagerly in ``__init__`` would always see
    the default ``"fp4"`` and silently misroute Flash-Base checkpoints.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resolved_expert_dtype: str | None = None
        self._resolved_moe_quant_algo: str | None = None
        self._resolved_target_num_hidden_layers: int | None = None
        self._nvfp4_config: ModelOptNvFp4Config | None = None
        # All target-layer B12X expert objects belonging to this model share
        # one graph workspace. The opaque token prevents accidental sharing
        # with another model replica in the same process.
        self._nvfp4_b12x_wrapper_scope = object()
        # ``is_scale_e8m0`` is a property that resolves on first read,
        # by which time the current vllm_config has been set.

    @property
    def expert_dtype(self) -> str:
        if self._resolved_expert_dtype is None:
            try:
                hf_config = get_current_vllm_config().model_config.hf_config
            except Exception:
                # vllm_config not yet set; defer the decision until a
                # later call lands inside set_current_vllm_config.
                return "fp4"
            expert_dtype = getattr(hf_config, "expert_dtype", "fp4")
            if expert_dtype not in _DEEPSEEK_V4_EXPERT_DTYPES:
                raise ValueError(
                    f"Unsupported DeepSeek V4 expert_dtype={expert_dtype!r}; "
                    f"expected one of {_DEEPSEEK_V4_EXPERT_DTYPES}."
                )
            self._resolved_expert_dtype = expert_dtype
            from vllm.logger import init_logger

            init_logger(__name__).info_once(
                "DeepSeek V4 expert_dtype resolved to %r", expert_dtype
            )
        return self._resolved_expert_dtype

    @property
    def is_scale_e8m0(self) -> bool:
        # FP4 checkpoints store FP8 linear scales as e8m0fnu; FP8 expert
        # checkpoints (Flash-Base) store them as float32.
        return self.expert_dtype == "fp4"

    def _resolve_moe_overrides(self) -> None:
        if self._resolved_moe_quant_algo is not None:
            return
        try:
            hf_config = get_current_vllm_config().model_config.hf_config
        except Exception:
            return
        quant_cfg = getattr(hf_config, "quantization_config", None) or {}
        algo = (quant_cfg.get("moe_quant_algo") or "").upper() or None
        self._resolved_moe_quant_algo = algo or ""
        target_num_hidden_layers = getattr(hf_config, "num_hidden_layers", None)
        if target_num_hidden_layers is not None:
            self._resolved_target_num_hidden_layers = int(target_num_hidden_layers)

    @property
    def moe_quant_algo(self) -> str:
        self._resolve_moe_overrides()
        return self._resolved_moe_quant_algo or ""

    @property
    def target_num_hidden_layers(self) -> int | None:
        self._resolve_moe_overrides()
        return self._resolved_target_num_hidden_layers

    def _uses_nvfp4_experts(self, prefix: str) -> bool:
        """Whether this routed-expert module contains NVIDIA NVFP4 weights.

        NVIDIA's DeepSeek-V4 NVFP4 checkpoint quantizes target decoder layers
        only. MTP/DSpark draft weights remain native MXFP4, while DSpark gives
        those blocks synthetic decoder prefixes starting at
        ``num_hidden_layers``. Keep the legacy global decision if a caller does
        not provide a recognizable decoder-layer prefix.
        """
        if self.moe_quant_algo != "NVFP4":
            return False
        layer_index = _extract_deepseek_v4_layer_index(prefix)
        target_num_hidden_layers = self.target_num_hidden_layers
        if layer_index is None or target_num_hidden_layers is None:
            return True
        return layer_index < target_num_hidden_layers

    def _get_nvfp4_config(self) -> ModelOptNvFp4Config:
        if self._nvfp4_config is None:
            from vllm.model_executor.layers.quantization.modelopt import (
                ModelOptNvFp4Config,
            )

            self._nvfp4_config = ModelOptNvFp4Config(
                is_checkpoint_nvfp4_serialized=True,
                kv_cache_quant_algo=None,
                exclude_modules=[],
                group_size=16,
            )
        return self._nvfp4_config

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "deepseek_v4_fp8"

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        if not (
            isinstance(hf_quant_cfg, dict)
            and hf_quant_cfg.get("quant_method") in ("fp8", "deepseek_v4_fp8")
        ):
            return None
        model_type = getattr(hf_config, "model_type", None)
        if model_type == "deepseek_v4" or user_quant == "deepseek_v4_fp8":
            return "deepseek_v4_fp8"
        return None

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, RoutedExperts):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            if self.expert_dtype == "fp4":
                if self._uses_nvfp4_experts(prefix):
                    from vllm.model_executor.layers.quantization.modelopt import (
                        ModelOptNvFp4FusedMoE,
                    )

                    # The pinned FlashInfer wrapper owns roughly 0.6 GiB of
                    # graph workspace at the DSv4 8K shape. A wrapper per one
                    # of 43 layers would hide about 25 GiB from vLLM's memory
                    # planner. Mark this model's otherwise-identical target
                    # layers with a shared, model-scoped workspace token.
                    layer.moe_config._b12x_wrapper_scope = (
                        self._nvfp4_b12x_wrapper_scope
                    )
                    uses_concurrent_ubatches = bool(
                        get_current_vllm_config().parallel_config.use_ubatching
                    )
                    layer.moe_config._b12x_wrapper_concurrent_execution = (
                        uses_concurrent_ubatches
                    )
                    return ModelOptNvFp4FusedMoE(
                        quant_config=self._get_nvfp4_config(),
                        moe_config=layer.moe_config,
                    )
                return Mxfp4MoEMethod(layer.moe_config)
            # expert_dtype == "fp8": fall through to Fp8Config which
            # returns Fp8MoEMethod with block-wise float32 scales.
        return super().get_quant_method(layer, prefix)

    def is_mxfp4_quant(self, prefix, layer):
        if not isinstance(layer, RoutedExperts) or self.expert_dtype != "fp4":
            return False
        return not self._uses_nvfp4_experts(prefix)
