# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepared NVFP4 CUTLASS/W4A16 decode-only dual expert.

The prepared DeepSeek-V4 checkpoint already stores the FP4 payload in the
single-copy ModelOpt layout consumed by both FlashInfer CUTLASS and B12X.  This
expert preserves CUTLASS W4A4 for M=1 and prefill, while retaining an exact
E8M0/K32 scale view for the B12X W4A16 tensor-core path at M=2..8.

The class is selected only by the explicit vLLM environment gate and must be
initialized by the prepared-checkpoint post-load hook.  Raw ModelOpt loading is
deliberately rejected: constructing the E8M0 view depends on the audited
prepared scale algebra and physical W13 layout.
"""

from __future__ import annotations

from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.b12x_mxfp4_moe import (
    _b12x_activation_name,
    _b12x_scratch_nbytes,
    _ceil_div,
    _dtype_element_size,
    _normalize_b12x_moe_topk_ids,
    _normalize_b12x_moe_topk_weights,
    _plan_b12x_moe_fp4_scratch,
    _run_b12x_moe_fp4,
    _workspace2_as_b12x_scratch,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.layers.fused_moe.experts.nvfp4_dual_decode_policy import (
    use_w4a16_decode,
    validate_dual_decode_bounds,
)


logger = init_logger(__name__)


def dual_decode_bounds() -> tuple[int, int]:
    """Return and validate the exact routed-token cutover interval."""

    minimum = int(envs.VLLM_NVFP4_W4A16_DECODE_MIN_M)
    maximum = int(envs.VLLM_NVFP4_W4A16_DECODE_MAX_M)
    return validate_dual_decode_bounds(minimum, maximum)


def _is_uniform_decode_forward() -> bool:
    """Use vLLM's batch descriptor instead of treating tiny prefill as decode."""

    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    if not is_forward_context_available():
        return False
    descriptor = get_forward_context().batch_descriptor
    return bool(descriptor is not None and descriptor.uniform)


def _canonical_global_exponents(
    raw_global: torch.Tensor, *, name: str
) -> torch.Tensor:
    value = raw_global.to(torch.float32).contiguous()
    if value.ndim != 1 or not bool(torch.isfinite(value).all().item()) or not bool(
        (value > 0).all().item()
    ):
        raise RuntimeError(f"{name} global scale must be finite positive [E]")
    exponent = torch.round(torch.log2(value)).to(torch.int32)
    canonical = torch.ldexp(torch.ones_like(value), exponent)
    maximum_ulp_distance = int(
        (value.view(torch.int32) - canonical.view(torch.int32)).abs().max().item()
    )
    if maximum_ulp_distance > 1:
        raise RuntimeError(
            f"{name} global scale is not the prepared power-of-two contract: "
            f"maximum ULP distance={maximum_ulp_distance}"
        )
    return exponent


def _collapse_nvfp4_scale_grid(
    swizzled_scale: torch.Tensor,
    raw_global: torch.Tensor,
    *,
    rows: int,
    cols: int,
    name: str,
) -> torch.Tensor:
    """Collapse exact E4M3/K16 pairs into E8M0/K32 exponent bytes."""

    from b12x.moe.fused.w4a16.host import unswizzle_expert_scales

    if cols % 32:
        raise ValueError(f"{name} requires K divisible by 32, got {cols}")
    linear = unswizzle_expert_scales(
        swizzled_scale, rows=rows, cols=cols
    ).contiguous()
    linear_bytes = linear.view(torch.uint8)
    experts = int(raw_global.numel())
    expected = (experts, rows, cols // 16)
    if tuple(linear_bytes.shape) != expected:
        raise RuntimeError(
            f"{name} unswizzled scale shape drifted: expected={expected}, "
            f"observed={tuple(linear_bytes.shape)}"
        )
    pairs = linear_bytes.reshape(experts, rows, cols // 32, 2)
    unequal = int(torch.count_nonzero(pairs[..., 0] != pairs[..., 1]).item())
    if unequal:
        raise RuntimeError(f"{name} has {unequal} non-identical K16 scale pairs")
    byte = pairs[..., 0]
    if bool(torch.any((byte & 0x80) != 0).item()):
        raise RuntimeError(f"{name} contains negative E4M3 scale bytes")

    exponent_field = ((byte >> 3) & 0x0F).to(torch.int32)
    mantissa = byte & 0x07
    normal = exponent_field > 0
    valid_subnormal = (mantissa == 1) | (mantissa == 2) | (mantissa == 4)
    valid = (normal & (mantissa == 0)) | ((~normal) & valid_subnormal)
    invalid = int(torch.count_nonzero(~valid).item())
    if invalid:
        raise RuntimeError(
            f"{name} has {invalid} E4M3 scales that are not powers of two"
        )
    subnormal_exponent = torch.where(
        mantissa == 1,
        torch.full_like(exponent_field, -9),
        torch.where(
            mantissa == 2,
            torch.full_like(exponent_field, -8),
            torch.full_like(exponent_field, -7),
        ),
    )
    block_exponent = torch.where(normal, exponent_field - 7, subnormal_exponent)
    global_exponent = _canonical_global_exponents(raw_global, name=name)
    e8m0 = block_exponent + global_exponent[:, None, None] + 127
    minimum_byte = int(e8m0.min().item())
    maximum_byte = int(e8m0.max().item())
    if minimum_byte < 0 or maximum_byte > 247:
        raise RuntimeError(
            f"{name} E8M0 range [{minimum_byte}, {maximum_byte}] exceeds "
            "the BF16 serving contract [0, 247]"
        )
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    result = e8m0.to(torch.uint8).contiguous()
    return result if e8m0_dtype is None else result.view(e8m0_dtype)


class NvFp4CutlassW4A16DualExperts(FlashInferExperts):
    """CUTLASS W4A4 with a prepared, decode-only B12X W4A16 sidecar."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._prepared_w4a16: Any | None = None
        self._w4a16_unit_scale: torch.Tensor | None = None
        self._w4a16_additional_scale_bytes = 0
        self._w4a16_bounds = dual_decode_bounds()
        self._w4a16_selection_logged = False

    @property
    def expects_unquantized_inputs(self) -> bool:
        # W4A16 consumes BF16 directly. FlashInfer CUTLASS also supports BF16
        # with input_sf=None and performs its W4A4 activation quantization in
        # expandInputRowsKernel, so one prepare contract serves both branches.
        return True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise RuntimeError(
            "NVFP4 dual decode is prepared-checkpoint-only; the prepared "
            "post-load hook must initialize its E8M0/K32 sidecar"
        )

    def initialize_prepared_w4a16_decode(self, layer: torch.nn.Module) -> None:
        """Build the exact E8M0/K32 scale view without copying FP4 weights."""

        if self._prepared_w4a16 is not None:
            raise RuntimeError("prepared W4A16 decode sidecar initialized twice")
        if self.quant_dtype != "nvfp4":
            raise RuntimeError(
                f"dual decode requires NVFP4 quantization, got {self.quant_dtype!r}"
            )
        if self.w1_scale is None or self.w2_scale is None:
            raise RuntimeError("dual decode requires both prepared CUTLASS scales")
        if self.g1_alphas is None or self.g2_alphas is None:
            raise RuntimeError("dual decode requires both prepared CUTLASS alphas")
        if self.a1_gscale is None or self.a2_gscale is None:
            raise RuntimeError("dual decode requires both reciprocal input scales")

        w13 = layer.w13_weight
        w2 = layer.w2_weight
        if w13.device.type != "cuda" or w2.device != w13.device:
            raise RuntimeError("dual decode preparation requires colocated CUDA weights")
        if w13.dtype != torch.uint8 or w2.dtype != torch.uint8:
            raise RuntimeError("dual decode requires packed uint8 FP4 weights")
        experts = int(w13.shape[0])
        hidden = int(w2.shape[1])
        intermediate = int(w2.shape[2]) * 2
        if tuple(w13.shape[1:]) != (2 * intermediate, hidden // 2):
            raise RuntimeError(
                "dual decode prepared W13 shape drifted: "
                f"{tuple(w13.shape)}"
            )

        raw_g1 = (self.g1_alphas * self.a1_gscale).to(torch.float32)
        raw_g2 = (self.g2_alphas * self.a2_gscale).to(torch.float32)
        w13_e8m0 = _collapse_nvfp4_scale_grid(
            self.w1_scale,
            raw_g1,
            rows=2 * intermediate,
            cols=hidden,
            name="w13",
        )
        w2_e8m0 = _collapse_nvfp4_scale_grid(
            self.w2_scale,
            raw_g2,
            rows=hidden,
            cols=intermediate,
            name="w2",
        )
        from b12x.moe.fused.w4a16.prepare import (
            prepare_w4a16_e8m0_native_weights,
        )

        unit_scale = torch.ones(experts, dtype=torch.float32, device=w13.device)
        prepared = prepare_w4a16_e8m0_native_weights(
            w13,
            w13_e8m0,
            unit_scale,
            w2,
            w2_e8m0,
            unit_scale.clone(),
            activation="silu",
            params_dtype=self.out_dtype,
            # Prepared physical W13 is [w3/up, w1/gate] (B12X up_gate).
            w13_layout="w13",
        )
        if prepared.weight_layout != "modelopt":
            raise RuntimeError(
                f"dual decode copied/repacked weights: {prepared.weight_layout!r}"
            )
        if prepared.source_format != "fp4_e8m0_k32":
            raise RuntimeError(
                f"dual decode scale format drifted: {prepared.source_format!r}"
            )
        for name, source, candidate in (
            ("w13", w13, prepared.w13),
            ("w2", w2, prepared.w2),
        ):
            if (
                int(source.data_ptr()) != int(candidate.data_ptr())
                or int(source.untyped_storage().data_ptr())
                != int(candidate.untyped_storage().data_ptr())
            ):
                raise RuntimeError(f"dual decode duplicated {name} FP4 storage")

        self._prepared_w4a16 = prepared
        self._w4a16_unit_scale = unit_scale
        # Count unique retained scale storages.  The micro aliases intentionally
        # reference the same K32 grids, while the two global-scale tensors have
        # distinct storage.  Do not hide these bytes behind a tensor-numel sum.
        retained_scales = (
            prepared.w13_scale,
            prepared.w2_scale,
            prepared.w13_global_scale,
            prepared.w2_global_scale,
        )
        unique_scale_storages: dict[int, int] = {}
        for value in retained_scales:
            storage = value.untyped_storage()
            unique_scale_storages.setdefault(
                int(storage.data_ptr()), int(storage.nbytes())
            )
        self._w4a16_additional_scale_bytes = sum(
            unique_scale_storages.values()
        )
        logger.info(
            "NVFP4_DUAL_DECODE event=prepared bounds=%s scale_bytes=%d "
            "duplicate_weight_bytes=0",
            self._w4a16_bounds,
            self._w4a16_additional_scale_bytes,
        )

    def _w4a16_plan(
        self,
        *,
        tokens: int,
        k: int,
        topk: int,
        device: torch.device,
        dtype: torch.dtype,
        activation: MoEActivation,
        apply_router_weight_on_input: bool = False,
    ) -> Any:
        prepared = self._prepared_w4a16
        if prepared is None:
            raise RuntimeError("prepared W4A16 decode sidecar is unavailable")
        weight_layout = getattr(prepared, "weight_layout", None)
        if weight_layout not in ("modelopt", "packed"):
            raise RuntimeError(
                f"prepared W4A16 weight layout drifted: {weight_layout!r}"
            )
        if weight_layout == "packed":
            # The immutable serving base carries the production B12X planner
            # whose TPMoEScratchCaps predates the optional modelopt-layout
            # field.  Native packed storage is that planner's default, so use
            # its exact ABI instead of passing a newer optional keyword.
            from b12x.integration.tp_moe import (
                TPMoEScratchCaps,
                plan_tp_moe_scratch,
            )

            return plan_tp_moe_scratch(
                TPMoEScratchCaps(
                    max_tokens=max(int(tokens), 1),
                    weight_E=int(prepared.num_experts),
                    k=int(k),
                    n=int(prepared.intermediate_size),
                    num_topk=int(topk),
                    device=device,
                    dtype=dtype,
                    core_token_counts=(max(int(tokens), 1),),
                    route_num_experts=0,
                    quant_mode="w4a16",
                    activation=_b12x_activation_name(activation),
                    apply_router_weight_on_input=apply_router_weight_on_input,
                    swiglu_limit=getattr(
                        self.quant_config, "gemm1_clamp_limit", None
                    ),
                    source_format="fp4_e8m0_k32",
                    w13_layout="w13",
                    frozen=True,
                )
            )
        return _plan_b12x_moe_fp4_scratch(
            tokens=tokens,
            weight_E=int(prepared.num_experts),
            k=k,
            n=int(prepared.intermediate_size),
            topk=topk,
            device=device,
            dtype=dtype,
            activation=_b12x_activation_name(activation),
            quant_mode="w4a16",
            source_format="fp4_e8m0_k32",
            w13_layout="w13",
            # The pinned planner's default is native packed storage.  Only the
            # legacy dual-view arm needs the explicit modelopt override.
            w4a16_weight_layout="modelopt",
            apply_router_weight_on_input=apply_router_weight_on_input,
            swiglu_limit=getattr(self.quant_config, "gemm1_clamp_limit", None),
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: Any | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # Because inputs remain BF16, K is already the physical hidden size;
        # unlike the ordinary pre-quantized NVFP4 path it must not be doubled.
        output_shape = (M, K)
        if not use_w4a16_decode(
            M,
            self._w4a16_bounds,
            uniform_decode=_is_uniform_decode_forward(),
        ):
            return (M, K), (0,), output_shape
        prepared = self._prepared_w4a16
        if prepared is None:
            raise RuntimeError("W4A16 workspace requested before prepared post-load")
        plan = self._w4a16_plan(
            tokens=max(int(M), 1),
            k=int(K),
            topk=int(topk),
            device=prepared.w13.device,
            dtype=self.out_dtype,
            activation=activation,
        )
        scratch_elements = max(
            1,
            _ceil_div(
                _b12x_scratch_nbytes(plan),
                _dtype_element_size(self.out_dtype),
            ),
        )
        return (0,), (scratch_elements,), output_shape

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
        expert_tokens_meta: Any | None,
        apply_router_weight_on_input: bool | None,
    ) -> None:
        tokens = int(hidden_states.shape[0])
        if not use_w4a16_decode(
            tokens,
            self._w4a16_bounds,
            uniform_decode=_is_uniform_decode_forward(),
        ):
            if a1q_scale is not None:
                raise RuntimeError(
                    "dual CUTLASS branch requires BF16 input with input_sf=None"
                )
            return super().apply(
                output=output,
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                a1q_scale=None,
                a2_scale=a2_scale,
                workspace13=workspace13,
                workspace2=workspace2,
                expert_tokens_meta=expert_tokens_meta,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        self._apply_prepared_w4a16(
            output=output,
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            expert_map=expert_map,
            workspace2=workspace2,
            apply_router_weight_on_input=apply_router_weight_on_input,
            log_selection=True,
        )

    def _apply_prepared_w4a16(
        self,
        *,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        expert_map: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        apply_router_weight_on_input: bool | None,
        log_selection: bool,
    ) -> None:
        tokens = int(hidden_states.shape[0])
        prepared = self._prepared_w4a16
        unit_scale = self._w4a16_unit_scale
        if prepared is None or unit_scale is None:
            raise RuntimeError("W4A16 decode selected before prepared post-load")
        if hidden_states.dtype != self.out_dtype or hidden_states.ndim != 2:
            raise RuntimeError(
                "W4A16 decode requires standard 2D BF16 activations, got "
                f"{tuple(hidden_states.shape)}/{hidden_states.dtype}"
            )
        if expert_map is not None:
            raise RuntimeError("W4A16 dual decode does not support expert_map")
        selected_ids = _normalize_b12x_moe_topk_ids(topk_ids)
        selected_weights = _normalize_b12x_moe_topk_weights(topk_weights)
        apply_weight_on_input = bool(apply_router_weight_on_input or False)
        plan = self._w4a16_plan(
            tokens=tokens,
            k=int(hidden_states.shape[1]),
            topk=int(selected_ids.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            activation=activation,
            apply_router_weight_on_input=apply_weight_on_input,
        )
        scratch = _workspace2_as_b12x_scratch(workspace2, plan)
        if log_selection and not self._w4a16_selection_logged:
            logger.info(
                "NVFP4_DUAL_DECODE event=selected tokens=%d bounds=%s "
                "uniform_decode=true",
                tokens,
                self._w4a16_bounds,
            )
            self._w4a16_selection_logged = True
        _run_b12x_moe_fp4(
            a=hidden_states,
            a1_gscale=unit_scale,
            w1_fp4=prepared.w13,
            w1_blockscale=prepared.w13_scale,
            w1_alphas=unit_scale,
            a2_gscale=unit_scale,
            w2_fp4=prepared.w2,
            w2_blockscale=prepared.w2_scale,
            w2_alphas=unit_scale,
            output=output,
            topk_weights=selected_weights,
            topk_ids=selected_ids,
            apply_router_weight_on_input=apply_weight_on_input,
            input_scales_are_reciprocal=True,
            input_scales_static=True,
            activation=_b12x_activation_name(activation),
            quant_mode="w4a16",
            unit_scale_contract=True,
            source_format="fp4_e8m0_k32",
            w13_layout="w13",
            prepared_w4a16=prepared,
            swiglu_limit=getattr(self.quant_config, "gemm1_clamp_limit", None),
            plan=plan,
            scratch=scratch,
        )


class NvFp4NativeB12xExperts(NvFp4CutlassW4A16DualExperts):
    """Prepared NVFP4 converted once to native-packed B12X W4A16.

    The prepared checkpoint's exact E4M3/K16 expansion is collapsed back to
    E8M0/K32, then the existing FP4 parameter storage is repacked in place.
    Every routed-expert forward uses the resulting native B12X W4A16 path;
    there is no CUTLASS branch and no duplicate FP4 payload.
    """

    def initialize_prepared_w4a16_decode(self, layer: torch.nn.Module) -> None:
        if self._prepared_w4a16 is not None:
            raise RuntimeError("prepared native B12X weights initialized twice")
        if self.quant_dtype != "nvfp4":
            raise RuntimeError(
                f"native B12X requires NVFP4 quantization, got {self.quant_dtype!r}"
            )
        if self.w1_scale is None or self.w2_scale is None:
            raise RuntimeError("native B12X requires both prepared CUTLASS scales")
        if self.g1_alphas is None or self.g2_alphas is None:
            raise RuntimeError("native B12X requires both prepared CUTLASS alphas")
        if self.a1_gscale is None or self.a2_gscale is None:
            raise RuntimeError("native B12X requires both reciprocal input scales")

        w13 = layer.w13_weight
        w2 = layer.w2_weight
        if w13.device.type != "cuda" or w2.device != w13.device:
            raise RuntimeError("native B12X preparation requires colocated CUDA weights")
        if w13.dtype != torch.uint8 or w2.dtype != torch.uint8:
            raise RuntimeError("native B12X requires packed uint8 FP4 weights")
        experts = int(w13.shape[0])
        hidden = int(w2.shape[1])
        intermediate = int(w2.shape[2]) * 2
        if tuple(w13.shape[1:]) != (2 * intermediate, hidden // 2):
            raise RuntimeError(
                "native B12X prepared W13 shape drifted: "
                f"{tuple(w13.shape)}"
            )

        raw_g1 = (self.g1_alphas * self.a1_gscale).to(torch.float32)
        raw_g2 = (self.g2_alphas * self.a2_gscale).to(torch.float32)
        w13_e8m0 = _collapse_nvfp4_scale_grid(
            self.w1_scale,
            raw_g1,
            rows=2 * intermediate,
            cols=hidden,
            name="w13",
        )
        w2_e8m0 = _collapse_nvfp4_scale_grid(
            self.w2_scale,
            raw_g2,
            rows=hidden,
            cols=intermediate,
            name="w2",
        )
        from b12x.moe.fused.w4a16.prepare import (
            prepare_w4a16_fp4_e8m0_k32_weights,
        )

        unit_scale = torch.ones(experts, dtype=torch.float32, device=w13.device)
        prepared = prepare_w4a16_fp4_e8m0_k32_weights(
            w13,
            w13_e8m0,
            unit_scale,
            w2,
            w2_e8m0,
            unit_scale.clone(),
            activation="silu",
            params_dtype=self.out_dtype,
            # Prepared physical W13 is [w3/up, w1/gate].  The native packer
            # folds the half rotation into its one-time in-place transform.
            w13_layout="w13",
            reuse_input_storage=True,
        )
        if prepared.weight_layout != "packed":
            raise RuntimeError(
                "native B12X preparation did not produce packed weights: "
                f"{prepared.weight_layout!r}"
            )
        if prepared.source_format != "fp4_e8m0_k32":
            raise RuntimeError(
                f"native B12X source format drifted: {prepared.source_format!r}"
            )
        for name, source, candidate in (
            ("w13", w13, prepared.w13),
            ("w2", w2, prepared.w2),
        ):
            if (
                int(source.data_ptr()) != int(candidate.data_ptr())
                or int(source.untyped_storage().data_ptr())
                != int(candidate.untyped_storage().data_ptr())
            ):
                raise RuntimeError(f"native B12X duplicated {name} FP4 storage")

        self._prepared_w4a16 = prepared
        self._w4a16_unit_scale = unit_scale
        retained_scales = (
            prepared.w13_scale,
            prepared.w2_scale,
            prepared.w13_global_scale,
            prepared.w2_global_scale,
        )
        unique_scale_storages: dict[int, int] = {}
        for value in retained_scales:
            storage = value.untyped_storage()
            unique_scale_storages.setdefault(
                int(storage.data_ptr()), int(storage.nbytes())
            )
        self._w4a16_additional_scale_bytes = sum(unique_scale_storages.values())
        logger.info(
            "NVFP4_NATIVE_B12X event=prepared scale_bytes=%d "
            "duplicate_weight_bytes=0 weight_layout=packed",
            self._w4a16_additional_scale_bytes,
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: Any | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        prepared = self._prepared_w4a16
        if prepared is None:
            raise RuntimeError(
                "native B12X workspace requested before prepared post-load"
            )
        plan = self._w4a16_plan(
            tokens=max(int(M), 1),
            k=int(K),
            topk=int(topk),
            device=prepared.w13.device,
            dtype=self.out_dtype,
            activation=activation,
        )
        scratch_elements = max(
            1,
            _ceil_div(
                _b12x_scratch_nbytes(plan),
                _dtype_element_size(self.out_dtype),
            ),
        )
        return (0,), (scratch_elements,), (M, K)

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
        expert_tokens_meta: Any | None,
        apply_router_weight_on_input: bool | None,
    ) -> None:
        if a1q_scale is not None:
            raise RuntimeError("native B12X requires unquantized BF16 inputs")
        self._apply_prepared_w4a16(
            output=output,
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            expert_map=expert_map,
            workspace2=workspace2,
            apply_router_weight_on_input=apply_router_weight_on_input,
            log_selection=False,
        )
