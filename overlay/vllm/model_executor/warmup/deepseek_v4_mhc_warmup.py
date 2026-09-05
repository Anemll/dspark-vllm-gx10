# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up DeepSeek V4 mHC TileLang kernels before serving requests.

Ported from lucifer1004/vllm-jasl with the two env-var knobs removed
(`VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP`, `VLLM_DEEPSEEK_V4_MHC_WARMUP_TOKEN_SIZES`).
Gating is intrinsic: non-DSv4 models and layers without hc_* attributes
return early, so the warmup is a no-op except where it's needed.
"""

import time
from collections.abc import Iterable

import torch

from vllm.logger import init_logger
from vllm.tracing import instrument
from vllm.utils.math_utils import cdiv

logger = init_logger(__name__)

_AUTO_WARMUP_MAX_TOKENS = 16_384
_DEFAULT_TOKEN_SIZE_CANDIDATES = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16_384,
)


def _compute_mhc_pre_num_split(
    *,
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    num_sms: int,
) -> int:
    block_k = 64
    block_m = 64
    k = hc_mult * hidden_size
    grid_size = cdiv(num_tokens, block_m)
    split_k = num_sms // grid_size
    num_block_k = cdiv(k, block_k)
    split_k = min(split_k, num_block_k // 4)
    return max(split_k, 1)


def _normalize_token_sizes(
    token_sizes: Iterable[int],
    *,
    max_tokens: int,
) -> list[int]:
    return sorted({size for size in token_sizes if 1 <= size <= max_tokens})


def _select_mhc_warmup_token_sizes(
    *,
    max_tokens: int,
    cudagraph_capture_sizes: list[int],
    hidden_size: int | None = None,
    hc_mult: int | None = None,
    num_sms: int | None = None,
) -> list[int]:
    if max_tokens <= 0:
        return []

    max_auto_tokens = min(max_tokens, _AUTO_WARMUP_MAX_TOKENS)
    candidates = list(_DEFAULT_TOKEN_SIZE_CANDIDATES)
    candidates.extend(cudagraph_capture_sizes)
    candidates.append(max_auto_tokens)
    sizes = _normalize_token_sizes(candidates, max_tokens=max_auto_tokens)
    if hidden_size is not None and hc_mult is not None and num_sms is not None:
        # TileLang's token dimension is dynamic, but the reduction split count
        # is a compile-time argument. Powers of two miss splits such as 9 on
        # GB10 (257..320 tokens). Add one representative per missing split,
        # using the same 64-token grid and SM-count rule as the real pre op.
        def split_for(size: int) -> int:
            return _compute_mhc_pre_num_split(
                num_tokens=size,
                hidden_size=hidden_size,
                hc_mult=hc_mult,
                num_sms=num_sms,
            )

        covered = {split_for(size) for size in sizes}
        for size in range(1, max_auto_tokens + 1, 64):
            split = split_for(size)
            if split not in covered:
                sizes.append(size)
                covered.add(split)
    return sorted(sizes)


def _uses_functional_mhc(layer: torch.nn.Module) -> bool:
    # NVIDIA target and DSpark draft share this decoder layout. The legacy
    # hc_pre/hc_post callables were removed when post/pre and norm were fused.
    return all(
        hasattr(layer, attr)
        for attr in (
            "attn_norm",
            "ffn_norm",
            "rms_norm_eps",
            "hc_eps",
            "hc_post_alpha",
            "hc_sinkhorn_iters",
        )
    ) and not (hasattr(layer, "hc_pre") and hasattr(layer, "hc_post"))


def _find_first_mhc_layer(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV4DecoderLayer":
            continue
        if all(
            hasattr(module, attr)
            for attr in (
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_attn_base",
                "hc_ffn_fn",
                "hc_ffn_scale",
                "hc_ffn_base",
            )
        ) and (
            (hasattr(module, "hc_pre") and hasattr(module, "hc_post"))
            or _uses_functional_mhc(module)
        ):
            return module
    return None


def _find_deepseek_v4_model(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        if module.__class__.__name__ not in (
            "DeepseekV4Model",
            "DSparkDeepseekV4Model",
        ):
            continue
        if all(
            hasattr(module, attr)
            for attr in ("hc_head_fn", "hc_head_scale", "hc_head_base")
        ):
            return module
    return None


def _warmup_layer_mhc(
    layer: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    if _uses_functional_mhc(layer):
        _warmup_functional_layer_mhc(layer, token_sizes)
        return

    max_tokens = max(token_sizes)
    hidden_size = int(layer.hidden_size)
    hc_mult = int(layer.hc_mult)
    device = layer.hc_attn_fn.device
    residual = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    for size in token_sizes:
        residual_slice = residual[:size]
        for fn, scale, base in (
            (layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base),
            (layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base),
        ):
            layer_input, post_mix, comb_mix = layer.hc_pre(
                residual_slice,
                fn,
                scale,
                base,
            )
            layer.hc_post(layer_input, residual_slice, post_mix, comb_mix)


def _warmup_functional_layer_mhc(
    layer: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    from vllm.model_executor.kernels.mhc.tilelang import (
        mhc_fused_post_pre_tilelang,
        mhc_post_tilelang,
        mhc_pre_tilelang,
    )

    residual = torch.zeros(
        max(token_sizes),
        int(layer.hc_mult),
        int(layer.hidden_size),
        dtype=torch.bfloat16,
        device=layer.hc_attn_fn.device,
    )
    common = (
        layer.rms_norm_eps,
        layer.hc_eps,
        layer.hc_eps,
        layer.hc_post_alpha,
        layer.hc_sinkhorn_iters,
    )
    for size in token_sizes:
        current_residual = residual[:size]
        # First decoder layer; preserve the fused RMSNorm arguments used by
        # DeepseekV4DecoderLayer.forward, not the old non-norm warmup path.
        post_mix, comb_mix, layer_input = mhc_pre_tilelang(
            current_residual,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            *common,
            norm_weight=layer.attn_norm.weight.data,
            norm_eps=layer.attn_norm.variance_epsilon,
        )
        # FFN pre and the next layer's attention pre share the fused post/pre
        # wrapper. Exercise both norm parameter contracts without executing
        # attention, MoE, collectives, or changing any model/cache state.
        for fn, scale, base, norm in (
            (layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, layer.ffn_norm),
            (
                layer.hc_attn_fn,
                layer.hc_attn_scale,
                layer.hc_attn_base,
                layer.attn_norm,
            ),
        ):
            current_residual, post_mix, comb_mix, layer_input = (
                mhc_fused_post_pre_tilelang(
                    layer_input,
                    current_residual,
                    post_mix,
                    comb_mix,
                    fn,
                    scale,
                    base,
                    *common,
                    n_splits=1,
                    tile_n=1,
                    norm_weight=norm.weight.data,
                    norm_eps=norm.variance_epsilon,
                )
            )
        mhc_post_tilelang(layer_input, current_residual, post_mix, comb_mix)


def _warmup_hc_head(
    model: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    # Upstream a8887c208 ("[DSV4] aiter mhc support (ROCm)") refactored
    # ``hc_head`` from a free function into the ``HCHeadOp`` CustomOp
    # instance attached to the model as ``hc_head_op``. We call through
    # that instance so the warmup exercises the same dispatched
    # implementation as the inference path.
    hc_head_op = getattr(model, "hc_head_op", None)
    if hc_head_op is None:
        # Current NVIDIA target and DSpark models call this function directly.
        # Keep the legacy CustomOp dispatch above when it is available.
        layer = _find_first_mhc_layer(model)
        if layer is None or not _uses_functional_mhc(layer):
            return
        from vllm.model_executor.kernels.mhc.tilelang import (
            hc_head_fused_kernel_tilelang,
        )

        hc_head_op = hc_head_fused_kernel_tilelang

    max_tokens = max(token_sizes)
    hidden_size = int(model.config.hidden_size)
    hc_mult = int(model.hc_mult)
    device = model.hc_head_fn.device
    hidden_states = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    for size in token_sizes:
        hc_head_op(
            hidden_states[:size],
            model.hc_head_fn,
            model.hc_head_scale,
            model.hc_head_base,
            model.rms_norm_eps,
            model.hc_eps,
        )


@instrument(span_name="DeepSeek V4 mHC warmup")
def deepseek_v4_mhc_warmup(
    model: torch.nn.Module,
    *,
    max_tokens: int,
    cudagraph_capture_sizes: list[int] | None = None,
) -> None:
    # Cheap model-type gate before walking ``model.modules()``. The class
    # walk below is O(num_layers) and shows up in startup time on very
    # large checkpoints; bail out for any model that is not DeepSeek V4.
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None) if config is not None else None
    if model_type is not None and model_type != "deepseek_v4":
        return

    layer = _find_first_mhc_layer(model)
    if layer is None:
        return

    device = layer.hc_attn_fn.device
    if device.type != "cuda":
        return

    deepseek_model = _find_deepseek_v4_model(model)
    split_kwargs: dict[str, int] = {}
    if _uses_functional_mhc(layer):
        from vllm.utils.deep_gemm import is_deep_gemm_supported

        if is_deep_gemm_supported():
            split_kwargs = {
                "hidden_size": int(layer.hidden_size),
                "hc_mult": int(layer.hc_mult),
                "num_sms": int(
                    torch.cuda.get_device_properties(device).multi_processor_count
                ),
            }
    token_sizes = _select_mhc_warmup_token_sizes(
        max_tokens=max_tokens,
        cudagraph_capture_sizes=cudagraph_capture_sizes or [],
        **split_kwargs,
    )
    if not token_sizes:
        return

    started = time.perf_counter()
    logger.info(
        "Warming up DeepSeek V4 mHC TileLang kernels for token sizes: %s",
        token_sizes,
    )
    with torch.inference_mode():
        _warmup_layer_mhc(layer, token_sizes)
        if deepseek_model is not None:
            _warmup_hc_head(deepseek_model, token_sizes)
        torch.accelerator.synchronize()
    logger.info(
        "DeepSeek V4 mHC TileLang warmup finished in %.2f seconds.",
        time.perf_counter() - started,
    )
