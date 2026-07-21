# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up DeepSeek V4 mHC TileLang kernels before serving requests.

Ported from lucifer1004/vllm-jasl with the two env-var knobs removed
(`VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP`, `VLLM_DEEPSEEK_V4_MHC_WARMUP_TOKEN_SIZES`).
Gating is intrinsic: non-DSv4 models and layers without hc_* attributes
return early, so the warmup is a no-op except where it's needed.

The DeepGEMM mHC pre-norm kernel is specialized by ``num_splits`` rather than
by the exact token count. Cover every reachable split specialization up front
so DeepGEMM never has to bind an unseen CUBIN to a hot CUDA context.
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


def _select_mhc_split_representatives(
    *,
    max_tokens: int,
    hidden_size: int,
    hc_mult: int,
    num_sms: int,
) -> dict[int, int]:
    """Return the first token count selecting every reachable split value.

    The scan is intentionally exhaustive over the bounded warmup range.  It
    runs only once during startup and prevents heuristic drift from silently
    leaving a serving-reachable specialization cold.
    """
    representatives: dict[int, int] = {}
    for num_tokens in range(1, max_tokens + 1):
        num_splits = _compute_mhc_pre_num_split(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            hc_mult=hc_mult,
            num_sms=num_sms,
        )
        representatives.setdefault(num_splits, num_tokens)
    return representatives


def _select_mhc_warmup_token_sizes(
    *,
    max_tokens: int,
    cudagraph_capture_sizes: list[int],
    hidden_size: int,
    hc_mult: int,
    num_sms: int,
) -> list[int]:
    if max_tokens <= 0:
        return []

    max_auto_tokens = min(max_tokens, _AUTO_WARMUP_MAX_TOKENS)
    split_representatives = _select_mhc_split_representatives(
        max_tokens=max_auto_tokens,
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        num_sms=num_sms,
    )
    candidates = list(_DEFAULT_TOKEN_SIZE_CANDIDATES)
    candidates.extend(cudagraph_capture_sizes)
    candidates.extend(split_representatives.values())
    candidates.append(max_auto_tokens)
    return _normalize_token_sizes(candidates, max_tokens=max_auto_tokens)


def _find_first_mhc_layer(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV4DecoderLayer":
            continue
        if all(
            hasattr(module, attr)
            for attr in (
                "hc_pre",
                "hc_post",
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_attn_base",
                "hc_ffn_fn",
                "hc_ffn_scale",
                "hc_ffn_base",
            )
        ):
            return module
    return None


def _find_deepseek_v4_model(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV4Model":
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
        return

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
    hidden_size = int(layer.hidden_size)
    hc_mult = int(layer.hc_mult)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    token_sizes = _select_mhc_warmup_token_sizes(
        max_tokens=max_tokens,
        cudagraph_capture_sizes=cudagraph_capture_sizes or [],
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        num_sms=num_sms,
    )
    if not token_sizes:
        return

    split_coverage = sorted(
        {
            _compute_mhc_pre_num_split(
                num_tokens=size,
                hidden_size=hidden_size,
                hc_mult=hc_mult,
                num_sms=num_sms,
            )
            for size in token_sizes
        }
    )
    started = time.perf_counter()
    logger.info(
        "Warming up DeepSeek V4 mHC TileLang kernels for token sizes: %s; "
        "DeepGEMM split specializations: %s",
        token_sizes,
        split_coverage,
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
