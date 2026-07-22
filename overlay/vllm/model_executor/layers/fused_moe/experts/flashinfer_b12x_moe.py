# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from threading import Lock
from typing import Any
from weakref import WeakValueDictionary

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_convert_sf_to_mma_layout,
    has_flashinfer_b12x_moe,
)


# A graph-enabled B12xMoEWrapper owns large static and dynamic routing arenas
# plus a max-token output buffer. DeepSeek V4 has 43 identical target MoE
# layers, so constructing one wrapper per layer would reserve roughly 25 GiB
# per TP rank outside vLLM's workspace planner. The quantization adapter marks
# those layers with a model-scoped opaque token; this weak cache then shares
# exactly one non-reentrant wrapper for that model/device/shape. Models without
# an explicit scope retain upstream's per-layer behavior.
_B12X_WRAPPER_CACHE: WeakValueDictionary[tuple[Any, ...], Any] = (
    WeakValueDictionary()
)
_B12X_WRAPPER_CACHE_LOCK = Lock()


def _resolve_b12x_activation(
    activation: str,
    swiglu_alpha: float | None,
    swiglu_beta: float | None,
    swiglu_limit: float | None,
) -> tuple[str, float, float, float | None]:
    """Translate vLLM's clamped SILU contract to FlashInfer B12X.

    B12X's plain ``silu`` mode does not apply a clamp.  DeepSeek V4 represents
    its clamped SwiGLU as SILU plus ``swiglu_limit``, so select B12X's packed
    gate/up implementation and the mathematically equivalent SILU parameters
    (alpha=1, beta=0) when a limit is present.  Unclamped activations retain
    the wrapper's existing defaults.
    """
    if activation == "silu" and swiglu_limit is not None:
        alpha = 1.0 if swiglu_alpha is None else float(swiglu_alpha)
        beta = 0.0 if swiglu_beta is None else float(swiglu_beta)
        return (
            "swigluoai_uninterleave",
            alpha,
            beta,
            float(swiglu_limit),
        )
    return activation, 1.702, 1.0, None


class FlashInferB12xExperts(mk.FusedMoEExpertsModular):
    """FlashInfer CuteDSL fused MoE expert for SM12x (SM120/SM121,
    RTX Pro 6000 / DGX Spark).

    Uses ``b12x_fused_moe`` from FlashInfer PR #3080 which fuses token
    dispatch, two GEMMs, SwiGLU activation, and topk-weight reduction into a
    single kernel call.  Input quantization (BF16→FP4) is performed inside the
    kernel so BF16 hidden states are passed directly.

    Weight scale factors are converted to the MMA layout produced by
    ``convert_sf_to_mma_layout`` once during ``process_weights_after_loading``
    and cached as ``w1_sf_mma`` / ``w2_sf_mma``.

    Only NVFP4 (kNvfp4Static/kNvfp4Dynamic) quantization is supported.
    """

    _ACTIVATION_MAP: dict[MoEActivation, str] = {
        MoEActivation.SILU: "silu",
        MoEActivation.RELU2_NO_MUL: "relu2",
    }

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        assert quant_config.quant_dtype == "nvfp4", (
            "FlashInferB12xExperts only supports nvfp4 quantization."
        )
        self.out_dtype = moe_config.in_dtype
        self.num_local_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank
        # FC2 input scale tensor bound in process_weights_after_loading: the
        # calibrated (now-zeroed) a2_gscale for static-quant checkpoints, or
        # a synthesized uniform-1.0 tensor for W4A16 checkpoints that lack
        # one. Holding it on the instance keeps apply() alloc-free.
        self._fc2_input_scale: torch.Tensor | None = None

        # Shape params for B12xMoEWrapper construction.
        self.global_num_experts = moe_config.num_experts
        self.topk = moe_config.experts_per_token
        self.hidden_dim = moe_config.hidden_dim
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.max_num_tokens = moe_config.max_num_tokens
        self.max_capture_size = moe_config.max_capture_size
        if not 0 < self.max_capture_size <= self.max_num_tokens:
            raise ValueError(
                "FlashInfer B12X decode capacity must be positive and no larger "
                "than max_num_tokens: "
                f"capture={self.max_capture_size}, max={self.max_num_tokens}"
            )
        self.local_expert_offset = self.ep_rank * self.num_local_experts
        wrapper_scope = getattr(moe_config, "_b12x_wrapper_scope", None)
        self._b12x_wrapper_scope = (
            wrapper_scope if wrapper_scope is not None else object()
        )
        if wrapper_scope is not None and getattr(
            moe_config, "_b12x_wrapper_concurrent_execution", False
        ):
            raise ValueError(
                "FlashInfer B12X shared graph workspaces are non-reentrant; "
                "disable DBO/concurrent ubatching for DeepSeek V4 NVFP4."
            )

        activation = moe_config.activation
        if activation not in self._ACTIVATION_MAP:
            raise ValueError(
                f"FlashInferB12xExperts does not support "
                f"activation {activation!r}. "
                f"Supported: {list(self._ACTIVATION_MAP.keys())}"
            )
        activation_str = self._ACTIVATION_MAP[activation]
        swiglu_limit = quant_config.gemm1_clamp_limit
        if swiglu_limit is None:
            swiglu_limit = moe_config.swiglu_limit
        swiglu_alpha = quant_config.gemm1_alpha
        if swiglu_alpha is None:
            swiglu_alpha = moe_config.swiglu_alpha
        swiglu_beta = quant_config.gemm1_beta
        if swiglu_beta is None:
            swiglu_beta = moe_config.swiglu_beta
        (
            self._activation_str,
            self._swiglu_alpha,
            self._swiglu_beta,
            self._swiglu_limit,
        ) = _resolve_b12x_activation(
            activation_str,
            swiglu_alpha,
            swiglu_beta,
            swiglu_limit,
        )

        # Decode and prefill use separate model-scoped arenas.  FlashInfer's
        # B12X dispatch cost grows measurably with max_num_tokens even when the
        # actual M is tiny, so decode should not pay for the 8192-token prefill
        # capacity.  The large wrapper remains unchanged for M values beyond
        # the CUDA-graph capture frontier.
        self._decode_wrapper: Any | None = None
        self._prefill_wrapper: Any | None = None
        self.w1_sf_mma: torch.Tensor | None = None
        self.w2_sf_mma: torch.Tensor | None = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Normalise block scales to absorb the per-expert weight global scale
        # (w_gs).  vLLM's NVFP4 convention stores:
        #   block_scale = max_abs * w_gs / fp4_max,  g1_alphas = 1/w_gs
        # The SM12x kernel treats w1_alpha (= g1_alphas) as a per-expert weight
        # dequant multiplier separate from input_gs (activation scale).  We bake
        # w_gs into the block scales so that w1_alpha = 1.0 and the kernel sees
        # the simpler form:
        #   block_scale = max_abs / fp4_max,  w1_alpha = 1.0
        # The FP4-packed values and dequantised results are identical in both
        # representations.  We set scale_2 = 1.0 to signal that the bake-in is
        # already done.
        layer.w13_weight_scale.data = (
            layer.w13_weight_scale.float() * layer.w13_weight_scale_2.view(-1, 1, 1)
        ).to(layer.w13_weight_scale.dtype)
        layer.w13_weight_scale_2.data.fill_(1.0)

        layer.w2_weight_scale.data = (
            layer.w2_weight_scale.float() * layer.w2_weight_scale_2.view(-1, 1, 1)
        ).to(layer.w2_weight_scale.dtype)
        layer.w2_weight_scale_2.data.fill_(1.0)

        # The SM12x kernel uses dynamic per-block quantization for FC2 input
        # activations (the SwiGLU output before the down projection).  The
        # calibrated a2_gscale from the modelopt checkpoint (~tens to hundreds)
        # is intended for static-quantisation backends (TRTLLM/CUTLASS) and
        # causes every intermediate activation to saturate at max FP4 when
        # multiplied by values that large.  Force to 1.0 so the kernel uses
        # its own per-block dynamic scale.
        if self.a2_gscale is not None:
            self.a2_gscale.fill_(1.0)
            self._fc2_input_scale = self.a2_gscale
        else:
            # W4A16 NVFP4 checkpoints have no calibrated a2_gscale; b12x
            # performs dynamic per-block FC2-input quantization, so a uniform
            # 1.0 scale per expert is equivalent to the bake-in above for
            # static-quant checkpoints. Allocate once here so apply() stays
            # alloc-free.
            self._fc2_input_scale = torch.ones(
                self.num_local_experts,
                device=layer.w13_weight.device,
                dtype=torch.float32,
            )

        # Precompute MMA-layout views of the weight scale factors once here
        # rather than recomputing on every forward pass.
        assert self.w1_scale is not None
        num_experts_w1, m1, k1_sf = self.w1_scale.shape
        k1 = k1_sf * 16
        self.w1_sf_mma = flashinfer_convert_sf_to_mma_layout(
            self.w1_scale.reshape(num_experts_w1 * m1, k1_sf),
            m=m1,
            k=k1,
            num_groups=num_experts_w1,
        )

        assert self.w2_scale is not None
        num_experts_w2, m2, k2_sf = self.w2_scale.shape
        k2 = k2_sf * 16
        self.w2_sf_mma = flashinfer_convert_sf_to_mma_layout(
            self.w2_scale.reshape(num_experts_w2 * m2, k2_sf),
            m=m2,
            k=k2,
            num_groups=num_experts_w2,
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(120)
            and has_flashinfer_b12x_moe()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # b12x performs in-kernel BF16->FP4 activation quant, so W4A16
        # NVFP4 checkpoints (activation_key=None, e.g. mixed-precision
        # compressed-tensors layouts) are runtime-compatible.
        return (weight_key, activation_key) in (
            (kNvfp4Static, kNvfp4Dynamic),
            (kNvfp4Static, None),
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SILU, MoEActivation.RELU2_NO_MUL)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # B12xMoEWrapper does not yet support expert parallelism: its local
        # expert count must equal the global expert count.
        return not moe_parallel_config.use_ep

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # b12x_fused_moe applies topk weights internally.
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # b12x_fused_moe manages its own internal workspace.
        workspace1 = (1,)
        workspace2 = (0,)
        output_shape = (M, K)
        return (workspace1, workspace2, output_shape)

    @property
    def expects_unquantized_inputs(self) -> bool:
        # B12xMoEWrapper expects BF16 hidden states and performs its own FP4
        # quantization internally.  Returning True prevents the modular kernel
        # from pre-quantizing activations.
        return True

    @property
    def supports_output_alias(self) -> bool:
        """Allow the modular kernel to pass its final output buffer directly.

        ``B12xMoEWrapper`` already emits the routed, weighted, reduced
        ``(M, K)`` result.  Writing into the modular kernel's caller-owned
        output therefore removes both the adapter copy and the downstream
        ``TopKWeightAndReduceNoOP`` copy.  The exact pointer contract is
        checked again in :meth:`apply` before the output is accepted.
        """
        return True

    def _ensure_wrapper(self, num_tokens: int | None = None) -> Any:
        """Acquire the decode- or prefill-sized graph workspace.

        Each wrapper is deliberately shared across sequential target layers:
        weights, scales, and the caller-owned output are call arguments. The
        cache is model-scoped rather than shape-global so independent replicas
        cannot race on mutable barriers/output. DBO is rejected in
        ``__init__``.
        """

        if num_tokens is None:
            num_tokens = self.max_num_tokens
        if not 0 < num_tokens <= self.max_num_tokens:
            raise ValueError(
                f"B12X token count {num_tokens} exceeds configured capacity "
                f"{self.max_num_tokens}"
            )
        decode = num_tokens <= self.max_capture_size
        wrapper_attr = "_decode_wrapper" if decode else "_prefill_wrapper"
        capacity = self.max_capture_size if decode else self.max_num_tokens

        # Shape and device are immutable after this expert is constructed.
        # Keep the steady eager/capture path free of cache-key construction and
        # CUDA device queries after the selected arena has been acquired.
        bound_wrapper = getattr(self, wrapper_attr)
        if bound_wrapper is not None:
            return bound_wrapper

        from flashinfer.fused_moe import B12xMoEWrapper

        device_index = torch.cuda.current_device()
        cache_key = (
            self._b12x_wrapper_scope,
            device_index,
            self.global_num_experts,
            self.num_local_experts,
            self.topk,
            self.hidden_dim,
            self.intermediate_size_per_partition,
            self.max_num_tokens,
            self.out_dtype,
            self._activation_str,
            self._swiglu_alpha,
            self._swiglu_beta,
            self._swiglu_limit,
            capacity,
            "nvfp4",
            "modelopt",
        )
        with _B12X_WRAPPER_CACHE_LOCK:
            wrapper = _B12X_WRAPPER_CACHE.get(cache_key)
            if wrapper is None:
                wrapper = B12xMoEWrapper(
                    num_experts=self.global_num_experts,
                    top_k=self.topk,
                    hidden_size=self.hidden_dim,
                    intermediate_size=self.intermediate_size_per_partition,
                    use_cuda_graph=True,
                    max_num_tokens=capacity,
                    num_local_experts=self.num_local_experts,
                    output_dtype=self.out_dtype,
                    device=f"cuda:{device_index}",
                    activation=self._activation_str,
                    swiglu_alpha=self._swiglu_alpha,
                    swiglu_beta=self._swiglu_beta,
                    swiglu_limit=self._swiglu_limit,
                    quant_mode="nvfp4",
                    source_format="modelopt",
                )
                _B12X_WRAPPER_CACHE[cache_key] = wrapper

        setattr(self, wrapper_attr, wrapper)
        return wrapper

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        assert self.w1_scale is not None and self.w2_scale is not None, (
            "w1_scale and w2_scale must not be None for FlashInferB12xExperts"
        )
        assert self.g1_alphas is not None and self.g2_alphas is not None, (
            "g1_alphas and g2_alphas must not be None for FlashInferB12xExperts"
        )
        assert self._fc2_input_scale is not None, (
            "_fc2_input_scale must be set by process_weights_after_loading"
        )
        assert self.w1_sf_mma is not None and self.w2_sf_mma is not None, (
            "process_weights_after_loading must run before FlashInferB12xExperts.apply"
        )

        wrapper = self._ensure_wrapper(int(hidden_states.shape[0]))

        if not getattr(wrapper, "use_cuda_graph", False):
            raise RuntimeError(
                "FlashInfer B12X output alias requires a graph-enabled wrapper"
            )
        if not hasattr(wrapper, "_moe_output"):
            raise RuntimeError(
                "FlashInfer B12X wrapper no longer exposes its output arena"
            )
        if (
            output.shape != hidden_states.shape
            or output.dtype != self.out_dtype
            or output.device != hidden_states.device
            or not output.is_contiguous()
        ):
            raise RuntimeError(
                "FlashInfer B12X output alias contract mismatch: "
                f"output={tuple(output.shape)}/{output.dtype}/{output.device}/"
                f"contiguous={output.is_contiguous()}, "
                f"hidden={tuple(hidden_states.shape)}/{hidden_states.dtype}/"
                f"{hidden_states.device}"
            )

        # The pinned FlashInfer wrapper slices ``_moe_output`` to the current
        # token count and passes that tensor as the kernel's scatter_output.
        # Redirect it to vLLM's final buffer immediately before every launch;
        # the wrapper is shared across layers but execution is explicitly
        # non-reentrant (DBO is rejected in __init__).
        wrapper._moe_output = output
        selected_experts = (
            topk_ids
            if topk_ids.dtype == torch.int32
            else topk_ids.to(torch.int32)
        )
        wrapper_output = wrapper.run(
            x=hidden_states,
            w1_weight=w1,
            w1_weight_sf=self.w1_sf_mma,
            w1_alpha=self.g1_alphas,
            fc2_input_scale=self._fc2_input_scale,
            w2_weight=w2,
            w2_weight_sf=self.w2_sf_mma,
            w2_alpha=self.g2_alphas,
            token_selected_experts=selected_experts,
            token_final_scales=topk_weights,
        )
        if (
            wrapper_output.shape != output.shape
            or wrapper_output.dtype != output.dtype
            or wrapper_output.device != output.device
            or wrapper_output.data_ptr() != output.data_ptr()
        ):
            raise RuntimeError(
                "FlashInfer B12X did not write into the aliased output buffer"
            )
