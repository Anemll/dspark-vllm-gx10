#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Real-layer MXFP4 W4A4 component gate for SM121.

This probe deliberately stops short of serving integration.  It loads only
the selected experts from one native-MXFP4 DeepSeek-V4 layer, applies the
exact TP=2 slicing contract, and executes the smallest complete W4A4 MoE
datapath using existing FlashInfer primitives:

* BF16 routed rows -> MXFP4 (E2M1 + UE8M0/K32)
* grouped MXFP4 FC1
* DeepSeek-V4 OAI SwiGLU (alpha=1, beta=0, limit=10)
* MXFP4 intermediate quantization
* grouped MXFP4 FC2 and router-weight reduction

M=1 and M=4 use unique balanced experts so every group contains one real
row.  Eager/graph output parity and a dequantized-native-weight BF16 oracle
are mandatory.  Checkpoint I/O, scale swizzling, dequantization, graph
capture, and compilation are outside the timed region.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


SCHEMA_VERSION = 1
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 2048
INTERMEDIATE_PER_RANK = INTERMEDIATE_SIZE // 2
NUM_EXPERTS = 256
TOP_K = 6
TP_SIZE = 2
MXFP4_BLOCK_SIZE = 32
E8M0_K32_BF16_MAX_SCALE_BYTE = 247
SWIGLU_LIMIT = 10.0
REFERENCE_CUTLASS_W4A4_M4_MS = 0.7820
DEFAULT_M4_MAX_MS = REFERENCE_CUTLASS_W4A4_M4_MS * 0.95

EXPERT_RE = re.compile(
    r"^(?P<root>(?:model\.)?layers)\.(?P<layer>[0-9]+)\.ffn\.experts\."
    r"(?P<expert>[0-9]+)\.(?P<projection>w[123])\."
    r"(?P<suffix>weight|scale)$"
)


@dataclass(frozen=True)
class NativeLayerSource:
    shard: Path
    root: str
    layer: int
    index_sha256: str | None
    tensor_names: tuple[str, ...]


@dataclass(frozen=True)
class RankSlices:
    w13_rows: slice
    w2_packed_k: slice
    w2_scale_k: slice


@dataclass
class NativeRankWeights:
    w13: Any
    w13_scale_raw: Any
    w13_scale_swizzled: Any
    w2: Any
    w2_scale_raw: Any
    w2_scale_swizzled: Any
    expert_ids: tuple[int, ...]
    scale_sanitization: dict[str, dict[str, int]]


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_positive_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("M values must be integers") from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("M values must be positive")
    return result


def tp_rank_slices(tp_rank: int) -> RankSlices:
    """Return the exact native-MXFP4 TP=2 expert slices."""

    if tp_rank not in (0, 1):
        raise ValueError("TP rank must be 0 or 1")
    row_begin = tp_rank * INTERMEDIATE_PER_RANK
    row_end = row_begin + INTERMEDIATE_PER_RANK
    packed_begin = tp_rank * (INTERMEDIATE_PER_RANK // 2)
    packed_end = packed_begin + INTERMEDIATE_PER_RANK // 2
    scale_begin = tp_rank * (INTERMEDIATE_PER_RANK // MXFP4_BLOCK_SIZE)
    scale_end = scale_begin + INTERMEDIATE_PER_RANK // MXFP4_BLOCK_SIZE
    return RankSlices(
        w13_rows=slice(row_begin, row_end),
        w2_packed_k=slice(packed_begin, packed_end),
        w2_scale_k=slice(scale_begin, scale_end),
    )


def e2m1_code_to_float(code: int) -> float:
    """Decode one four-bit E2M1 value; useful for the CPU unit contract."""

    if not 0 <= code < 16:
        raise ValueError("E2M1 code must be in [0, 15]")
    sign = -1.0 if code & 8 else 1.0
    payload = code & 7
    exponent_raw, mantissa = divmod(payload, 2)
    if exponent_raw == 0:
        magnitude = mantissa / 2.0
    else:
        magnitude = (2 + mantissa) / 2.0 * (2.0 ** (exponent_raw - 1))
    return sign * magnitude


def e8m0_byte_to_float(byte: int) -> float:
    if not 0 <= byte <= 255:
        raise ValueError("E8M0 byte must be in [0, 255]")
    return 2.0 ** (byte - 127)


def clamp_e8m0_scale_byte_for_bf16(byte: int) -> int:
    """Clamp one native E8M0/K32 scale byte to the serving BF16 contract."""

    if not 0 <= byte <= 255:
        raise ValueError("E8M0 byte must be in [0, 255]")
    return min(byte, E8M0_K32_BF16_MAX_SCALE_BYTE)


def balanced_route_experts(m: int, top_k: int = TOP_K) -> tuple[int, ...]:
    """Use one unique physical expert per route for a deterministic gate."""

    if m <= 0 or top_k <= 0 or m * top_k > NUM_EXPERTS:
        raise ValueError("balanced route geometry is outside the expert set")
    return tuple(range(m * top_k))


def _expected_names(root: str, layer: int) -> tuple[str, ...]:
    return tuple(
        f"{root}.{layer}.ffn.experts.{expert}.{projection}.{suffix}"
        for expert in range(NUM_EXPERTS)
        for projection in ("w1", "w2", "w3")
        for suffix in ("weight", "scale")
    )


def discover_native_layer(
    *,
    model_dir: Path | None,
    shard_file: Path | None,
    layer: int,
) -> NativeLayerSource:
    """Resolve a complete native-MXFP4 layer without opening tensor payloads."""

    if (model_dir is None) == (shard_file is None):
        raise ValueError("exactly one of model_dir or shard_file is required")
    if layer < 0:
        raise ValueError("layer must be non-negative")

    index_sha: str | None = None
    weight_map: Mapping[str, str] | None = None
    if model_dir is not None:
        model_dir = model_dir.expanduser().resolve(strict=True)
        index_path = model_dir / "model.safetensors.index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(index.get("weight_map"), dict):
            raise RuntimeError("checkpoint index has no weight_map object")
        weight_map = index["weight_map"]
        index_sha = _sha256_path(index_path)
        names = tuple(
            name
            for name in weight_map
            if (match := EXPERT_RE.fullmatch(name))
            and int(match.group("layer")) == layer
        )
    else:
        shard_file = shard_file.expanduser().resolve(strict=True)  # type: ignore[union-attr]
        from safetensors import safe_open

        with safe_open(str(shard_file), framework="pt", device="cpu") as handle:
            names = tuple(
                name
                for name in handle.keys()
                if (match := EXPERT_RE.fullmatch(name))
                and int(match.group("layer")) == layer
            )

    roots = {
        match.group("root")
        for name in names
        if (match := EXPERT_RE.fullmatch(name)) is not None
    }
    if len(roots) != 1:
        raise RuntimeError(f"layer {layer} has ambiguous roots: {sorted(roots)}")
    root = next(iter(roots))
    expected = set(_expected_names(root, layer))
    observed = set(names)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise RuntimeError(
            f"native layer contract drift: missing={len(missing)}, extra={len(extra)}"
        )

    if weight_map is not None:
        shard_names = {weight_map[name] for name in expected}
        if len(shard_names) != 1:
            raise RuntimeError(
                f"native layer {layer} spans {len(shard_names)} shards; expected one"
            )
        assert model_dir is not None
        shard = (model_dir / next(iter(shard_names))).resolve(strict=True)
    else:
        assert shard_file is not None
        shard = shard_file
    return NativeLayerSource(
        shard=shard,
        root=root,
        layer=layer,
        index_sha256=index_sha,
        tensor_names=tuple(sorted(expected)),
    )


def evaluate_performance_gate(
    graph_median_ms: Mapping[int, float],
    *,
    maximum_m4_ms: float,
) -> dict[str, Any]:
    if 4 not in graph_median_ms:
        raise ValueError("performance screen requires M=4")
    if not math.isfinite(maximum_m4_ms) or maximum_m4_ms <= 0:
        raise ValueError("M=4 ceiling must be positive and finite")
    for value in graph_median_ms.values():
        if not math.isfinite(value) or value <= 0:
            raise ValueError("latencies must be positive and finite")
    observed = float(graph_median_ms[4])
    return {
        "comparison": "mxfp4_w4a4_component_vs_accepted_cutlass_w4a4",
        "accepted_cutlass_w4a4_m4_ms": REFERENCE_CUTLASS_W4A4_M4_MS,
        "maximum_m4_ms": maximum_m4_ms,
        "observed_m4_ms": observed,
        "required_speedup": REFERENCE_CUTLASS_W4A4_M4_MS / maximum_m4_ms,
        "observed_speedup": REFERENCE_CUTLASS_W4A4_M4_MS / observed,
        "passed": observed <= maximum_m4_ms,
    }


def _as_uint8(torch: Any, tensor: Any, *, name: str) -> Any:
    if tensor.element_size() != 1:
        raise RuntimeError(f"{name} must use one-byte storage, got {tensor.dtype}")
    return tensor.view(torch.uint8).contiguous()


def _clamp_e8m0_scales_for_bf16(
    torch: Any,
    scale: Any,
    *,
    name: str,
) -> tuple[Any, dict[str, int]]:
    """Return BF16-safe E8M0 bytes and retain source-range evidence.

    Native MXFP4 checkpoints may contain scale bytes up to 255.  Decoding
    those exponents into BF16 GEMM operands can overflow before the OAI
    activation clamp is reached.  The existing packed-W4A16 reference path
    uses the same 247 ceiling; applying it here keeps the W4A4 kernel and its
    BF16 oracle on one explicit numeric contract.
    """

    source = _as_uint8(torch, scale, name=name)
    source_min = int(source.amin().item())
    source_max = int(source.amax().item())
    above = int(
        torch.count_nonzero(source > E8M0_K32_BF16_MAX_SCALE_BYTE).item()
    )
    at_255 = int(torch.count_nonzero(source == 255).item())
    sanitized = source.clamp(max=E8M0_K32_BF16_MAX_SCALE_BYTE).contiguous()
    sanitized_max = int(sanitized.amax().item())
    if sanitized_max > E8M0_K32_BF16_MAX_SCALE_BYTE:
        raise RuntimeError(f"{name} E8M0 clamp did not take effect")
    return sanitized, {
        "elements": int(source.numel()),
        "source_min_byte": source_min,
        "source_max_byte": source_max,
        "source_above_bf16_safe": above,
        "source_at_255": at_255,
        "applied_max_byte": E8M0_K32_BF16_MAX_SCALE_BYTE,
        "sanitized_max_byte": sanitized_max,
        "clamped_elements": above,
    }


def _load_native_rank_weights(
    torch: Any,
    flashinfer: Any,
    source: NativeLayerSource,
    expert_ids: tuple[int, ...],
    tp_rank: int,
) -> NativeRankWeights:
    from safetensors import safe_open

    slices = tp_rank_slices(tp_rank)
    families: dict[str, list[Any]] = {
        "w1.weight": [],
        "w1.scale": [],
        "w2.weight": [],
        "w2.scale": [],
        "w3.weight": [],
        "w3.scale": [],
    }
    with safe_open(str(source.shard), framework="pt", device="cpu") as handle:
        for expert in expert_ids:
            prefix = f"{source.root}.{source.layer}.ffn.experts.{expert}"
            for projection in ("w1", "w2", "w3"):
                for suffix in ("weight", "scale"):
                    key = f"{prefix}.{projection}.{suffix}"
                    tensor = handle.get_tensor(key)
                    if projection in ("w1", "w3"):
                        tensor = tensor[slices.w13_rows]
                    elif suffix == "weight":
                        tensor = tensor[:, slices.w2_packed_k]
                    else:
                        tensor = tensor[:, slices.w2_scale_k]
                    families[f"{projection}.{suffix}"].append(tensor.contiguous())

    stacked = {
        name: _as_uint8(torch, torch.stack(values).to("cuda"), name=name)
        for name, values in families.items()
    }
    expected_shapes = {
        "w1.weight": (len(expert_ids), INTERMEDIATE_PER_RANK, HIDDEN_SIZE // 2),
        "w3.weight": (len(expert_ids), INTERMEDIATE_PER_RANK, HIDDEN_SIZE // 2),
        "w2.weight": (len(expert_ids), HIDDEN_SIZE, INTERMEDIATE_PER_RANK // 2),
        "w1.scale": (
            len(expert_ids),
            INTERMEDIATE_PER_RANK,
            HIDDEN_SIZE // MXFP4_BLOCK_SIZE,
        ),
        "w3.scale": (
            len(expert_ids),
            INTERMEDIATE_PER_RANK,
            HIDDEN_SIZE // MXFP4_BLOCK_SIZE,
        ),
        "w2.scale": (
            len(expert_ids),
            HIDDEN_SIZE,
            INTERMEDIATE_PER_RANK // MXFP4_BLOCK_SIZE,
        ),
    }
    for name, expected in expected_shapes.items():
        if tuple(stacked[name].shape) != expected:
            raise RuntimeError(
                f"native {name} shape drift: {tuple(stacked[name].shape)} != {expected}"
            )

    scale_sanitization: dict[str, dict[str, int]] = {}
    for name in ("w1.scale", "w2.scale", "w3.scale"):
        stacked[name], scale_sanitization[name] = _clamp_e8m0_scales_for_bf16(
            torch,
            stacked[name],
            name=name,
        )

    # Source order is w1/gate then w3/up.  Keep that explicit for the OAI
    # activation below.  Native scales are linear UE8M0; grouped_mm_fp4 needs
    # the same 128x4 scale interleave produced by mxfp4_quantize.
    w13 = torch.cat((stacked["w1.weight"], stacked["w3.weight"]), dim=1)
    w13_scale_raw = torch.cat(
        (stacked["w1.scale"], stacked["w3.scale"]), dim=1
    )
    w2 = stacked["w2.weight"]
    w2_scale_raw = stacked["w2.scale"]

    def swizzle(raw: Any) -> Any:
        e, n, ksf = map(int, raw.shape)
        flat = flashinfer.nvfp4_block_scale_interleave(
            raw.reshape(e * n, ksf).contiguous()
        )
        if int(flat.numel()) != e * n * ksf:
            raise RuntimeError("unexpected MXFP4 weight-scale padding")
        return flat.reshape(e, n, ksf).contiguous()

    return NativeRankWeights(
        w13=w13,
        w13_scale_raw=w13_scale_raw,
        w13_scale_swizzled=swizzle(w13_scale_raw),
        w2=w2,
        w2_scale_raw=w2_scale_raw,
        w2_scale_swizzled=swizzle(w2_scale_raw),
        expert_ids=expert_ids,
        scale_sanitization=scale_sanitization,
    )


def _dequantize_e2m1_e8m0(torch: Any, packed: Any, scale: Any) -> Any:
    """Dequantize native linear MXFP4 tensors without format conversion."""

    packed = packed.view(torch.uint8)
    scale = scale.view(torch.uint8)
    lo = packed & 0x0F
    hi = packed >> 4
    codes = torch.stack((lo, hi), dim=-1).flatten(start_dim=-2)
    sign = torch.where(
        codes >= 8,
        torch.tensor(-1.0, device=codes.device),
        torch.tensor(1.0, device=codes.device),
    )
    payload = codes & 7
    exponent_raw = payload >> 1
    mantissa = payload & 1
    magnitude = torch.where(
        exponent_raw == 0,
        mantissa.float() * 0.5,
        (2.0 + mantissa.float()) * 0.5 * torch.exp2(exponent_raw.float() - 1.0),
    )
    values = sign * magnitude
    block_scale = torch.exp2(scale.float() - 127.0).repeat_interleave(
        MXFP4_BLOCK_SIZE, dim=-1
    )
    if values.shape != block_scale.shape:
        raise RuntimeError(
            f"MXFP4 dequant shape mismatch: {values.shape} != {block_scale.shape}"
        )
    return (values * block_scale).to(torch.bfloat16)


def _oai_swiglu(torch: Any, fc1: Any) -> Any:
    gate = fc1[:, :INTERMEDIATE_PER_RANK].float().clamp(max=SWIGLU_LIMIT)
    up = fc1[:, INTERMEDIATE_PER_RANK:].float().clamp(
        min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT
    )
    return (torch.sigmoid(gate) * gate * up).to(torch.bfloat16)


def _metrics(torch: Any, actual: Any, reference: Any) -> dict[str, Any]:
    actual_f = actual.float().reshape(-1)
    reference_f = reference.float().reshape(-1)
    finite = bool(torch.isfinite(actual_f).all() and torch.isfinite(reference_f).all())
    difference = actual_f - reference_f
    rmse = float(torch.sqrt(torch.mean(difference.square())).item())
    reference_rms = float(torch.sqrt(torch.mean(reference_f.square())).item())
    denom = float(torch.linalg.vector_norm(actual_f) * torch.linalg.vector_norm(reference_f))
    cosine = (
        float(torch.dot(actual_f, reference_f).item()) / denom if denom > 0 else 1.0
    )
    return {
        "finite": finite,
        "cosine": cosine,
        "rmse": rmse,
        "normalized_rmse": rmse / max(reference_rms, 1.0e-12),
        "max_abs": float(difference.abs().max().item()),
        "actual_nonzero": int(torch.count_nonzero(actual_f).item()),
        "reference_nonzero": int(torch.count_nonzero(reference_f).item()),
    }


def _numeric_gate_passed(
    metrics: Mapping[str, Any],
    *,
    minimum_cosine: float,
    maximum_nrmse: float,
) -> bool:
    return bool(
        metrics["finite"]
        and metrics["actual_nonzero"] > 0
        and metrics["reference_nonzero"] > 0
        and metrics["cosine"] >= minimum_cosine
        and metrics["normalized_rmse"] <= maximum_nrmse
    )


def _run_one_expert_layout_controls(
    torch: Any,
    flashinfer: Any,
    grouped_mm_fp4: Callable[..., Any],
    weights: NativeRankWeights,
    *,
    seed: int,
    minimum_cosine: float,
    maximum_nrmse: float,
) -> dict[str, Any]:
    """Isolate cuDNN execution from native checkpoint packing/layout.

    The synthetic control quantizes both A and a real-shape BF16 B with the
    same FlashInfer quantizer consumed by grouped_mm_fp4.  The native controls
    then hold A and the BF16 oracle fixed while selecting raw versus 128x4
    interleaved checkpoint scales.  Finally, the native B is dequantized and
    requantized through FlashInfer; that distinguishes native byte-layout
    incompatibility from a cuDNN grouped-MXFP4 failure on SM121.
    """

    torch.manual_seed(seed)
    n = 2 * INTERMEDIATE_PER_RANK
    k = HIDDEN_SIZE
    max_m = 4
    a_bf16 = (
        torch.randn(max_m, k, dtype=torch.bfloat16, device="cuda") * 0.125
    ).contiguous()
    synthetic_b_bf16 = (
        torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.125
    ).contiguous()
    synthetic_b_fp4, synthetic_b_sf = flashinfer.mxfp4_quantize(
        synthetic_b_bf16
    )
    synthetic_b_fp4 = synthetic_b_fp4.reshape(1, n, k // 2).contiguous()
    if int(synthetic_b_sf.numel()) != n * (k // MXFP4_BLOCK_SIZE):
        raise RuntimeError(
            "real-shape quantizer B scale unexpectedly requires padding"
        )
    synthetic_b_sf = synthetic_b_sf.reshape(
        1, n, k // MXFP4_BLOCK_SIZE
    ).contiguous()

    native_b_fp4 = weights.w13[:1]
    native_b_raw_sf = weights.w13_scale_raw[:1]
    native_b_swizzled_sf = weights.w13_scale_swizzled[:1]
    native_b_bf16 = _dequantize_e2m1_e8m0(
        torch,
        native_b_fp4,
        native_b_raw_sf,
    )[0].contiguous()
    requantized_b_fp4, requantized_b_sf = flashinfer.mxfp4_quantize(
        native_b_bf16
    )
    requantized_b_fp4 = requantized_b_fp4.reshape(1, n, k // 2).contiguous()
    if int(requantized_b_sf.numel()) != n * (k // MXFP4_BLOCK_SIZE):
        raise RuntimeError(
            "native-dequantized B requantizer scale unexpectedly requires padding"
        )
    requantized_b_sf = requantized_b_sf.reshape(
        1, n, k // MXFP4_BLOCK_SIZE
    ).contiguous()

    variants = (
        (
            "quantizer_produced_b",
            synthetic_b_fp4,
            synthetic_b_sf,
            synthetic_b_bf16,
            True,
        ),
        (
            "checkpoint_native_b_interleaved_sf",
            native_b_fp4,
            native_b_swizzled_sf,
            native_b_bf16,
            True,
        ),
        (
            "checkpoint_native_b_raw_sf",
            native_b_fp4,
            native_b_raw_sf,
            native_b_bf16,
            False,
        ),
        (
            "checkpoint_dequant_requantized_b",
            requantized_b_fp4,
            requantized_b_sf,
            native_b_bf16,
            True,
        ),
    )
    results: list[dict[str, Any]] = []
    for m in (1, 4):
        a = a_bf16[:m]
        a_fp4, a_sf = flashinfer.mxfp4_quantize(a)
        m_indptr = torch.tensor([0, m], dtype=torch.int32, device="cuda")
        for name, b_fp4, b_sf, b_reference, required in variants:
            result: dict[str, Any] = {
                "m": m,
                "variant": name,
                "required": required,
                "a_scale_shape": list(map(int, a_sf.shape)),
                "b_scale_shape": list(map(int, b_sf.shape)),
            }
            try:
                output = grouped_mm_fp4(
                    a_fp4,
                    b_fp4,
                    a_sf,
                    b_sf,
                    m_indptr,
                    block_size=MXFP4_BLOCK_SIZE,
                    out_dtype=torch.bfloat16,
                )
                torch.cuda.synchronize()
                reference = (
                    a.float() @ b_reference.float().transpose(0, 1)
                ).to(torch.bfloat16)
                numeric = _metrics(torch, output, reference)
                result["numeric_vs_bf16"] = numeric
                result["passed"] = _numeric_gate_passed(
                    numeric,
                    minimum_cosine=minimum_cosine,
                    maximum_nrmse=maximum_nrmse,
                )
            except BaseException as error:
                result["passed"] = False
                result["error"] = f"{type(error).__name__}: {error}"
            results.append(result)

    required_results = [result for result in results if result["required"]]
    return {
        "purpose": "cuDNN-vs-native-packed-layout isolation",
        "real_shape": {"n": n, "k": k, "m": [1, 4], "experts": 1},
        "raw_scale_variant_is_diagnostic_only": True,
        "required_passed": all(result["passed"] for result in required_results),
        "results": results,
    }


def _measure(
    torch: Any,
    launch: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
    repeats: int,
) -> dict[str, Any]:
    all_samples: list[float] = []
    repeat_medians: list[float] = []
    for _ in range(repeats):
        for _ in range(warmup):
            launch()
        torch.cuda.synchronize()
        samples: list[float] = []
        for _ in range(iters):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            launch()
            end.record()
            end.synchronize()
            samples.append(float(begin.elapsed_time(end)))
        all_samples.extend(samples)
        repeat_medians.append(statistics.median(samples))
    return {
        "median_ms": statistics.median(repeat_medians),
        "mean_ms": statistics.mean(all_samples),
        "min_ms": min(all_samples),
        "max_ms": max(all_samples),
        "repeat_medians_ms": repeat_medians,
        "samples": len(all_samples),
    }


def _capture_graph(torch: Any, launch: Callable[[], Any]) -> tuple[Callable[[], Any], Any, Any]:
    # Resolve FlashInfer/cuDNN plans before capture.
    eager_output = launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = launch()

    def replay() -> Any:
        graph.replay()
        return graph_output

    replay()
    torch.cuda.synchronize()
    return replay, graph_output, graph


def _make_case(
    torch: Any,
    flashinfer: Any,
    grouped_mm_fp4: Callable[..., Any],
    weights: NativeRankWeights,
    *,
    m: int,
    seed: int,
) -> tuple[Callable[[], Any], Any, dict[str, Any]]:
    routed_rows = m * TOP_K
    if len(weights.expert_ids) < routed_rows:
        raise RuntimeError("loaded expert set is smaller than routed rows")
    torch.manual_seed(seed)
    x = torch.randn(m, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    x = x / torch.sqrt(torch.mean(x.float().square(), dim=-1, keepdim=True)).to(
        torch.bfloat16
    )
    routed_x = x.repeat_interleave(TOP_K, dim=0).contiguous()
    topk_weights = torch.full(
        (m, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device="cuda"
    )
    indptr = torch.arange(routed_rows + 1, dtype=torch.int32, device="cuda")
    w13 = weights.w13[:routed_rows]
    w13_sf = weights.w13_scale_swizzled[:routed_rows]
    w2 = weights.w2[:routed_rows]
    w2_sf = weights.w2_scale_swizzled[:routed_rows]

    def launch() -> Any:
        a1, a1_sf = flashinfer.mxfp4_quantize(routed_x)
        fc1 = grouped_mm_fp4(
            a1,
            w13,
            a1_sf,
            w13_sf,
            indptr,
            block_size=MXFP4_BLOCK_SIZE,
            out_dtype=torch.bfloat16,
        )
        activated = _oai_swiglu(torch, fc1)
        a2, a2_sf = flashinfer.mxfp4_quantize(activated)
        routed_output = grouped_mm_fp4(
            a2,
            w2,
            a2_sf,
            w2_sf,
            indptr,
            block_size=MXFP4_BLOCK_SIZE,
            out_dtype=torch.bfloat16,
        )
        return (
            routed_output.reshape(m, TOP_K, HIDDEN_SIZE).float()
            * topk_weights[:, :, None]
        ).sum(dim=1).to(torch.bfloat16)

    # Same native MXFP4 weights, but dequantized once and multiplied by BF16
    # activations.  This isolates activation-quantization error from any
    # checkpoint-format or expert-selection change.
    w13_ref = _dequantize_e2m1_e8m0(
        torch, weights.w13[:routed_rows], weights.w13_scale_raw[:routed_rows]
    )
    w2_ref = _dequantize_e2m1_e8m0(
        torch, weights.w2[:routed_rows], weights.w2_scale_raw[:routed_rows]
    )
    fc1_ref = torch.bmm(w13_ref.float(), routed_x.float().unsqueeze(-1)).squeeze(-1)
    activated_ref = _oai_swiglu(torch, fc1_ref)
    routed_ref = torch.bmm(
        w2_ref.float(), activated_ref.float().unsqueeze(-1)
    ).squeeze(-1)
    reference = (
        routed_ref.reshape(m, TOP_K, HIDDEN_SIZE)
        * topk_weights[:, :, None]
    ).sum(dim=1).to(torch.bfloat16)
    del w13_ref, w2_ref, fc1_ref, activated_ref, routed_ref
    return launch, reference, {
        "m": m,
        "routed_rows": routed_rows,
        "experts": list(weights.expert_ids[:routed_rows]),
        "top_k": TOP_K,
        "w13_order": "w1/gate then w3/up",
        "scale_contract": "UE8M0/K32; no global alpha",
        "weight_scale_sanitization": weights.scale_sanitization,
    }


def run(args: argparse.Namespace) -> int:
    import flashinfer
    import torch
    from flashinfer.grouped_mm import grouped_mm_fp4

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("MXFP4 component gate requires exactly one CUDA device")
    capability = tuple(torch.cuda.get_device_capability())
    if capability != (12, 1):
        raise RuntimeError(f"MXFP4 component gate requires SM121, got {capability}")
    if args.m != (1, 4):
        raise ValueError("the bounded component gate requires exactly --m 1,4")
    if args.warmup < 0 or args.iters <= 0 or args.repeats <= 0:
        raise ValueError("invalid timing iteration counts")

    source = discover_native_layer(
        model_dir=args.model_dir,
        shard_file=args.shard_file,
        layer=args.layer,
    )
    expert_ids = balanced_route_experts(max(args.m))
    load_started = time.perf_counter()
    weights = _load_native_rank_weights(
        torch, flashinfer, source, expert_ids, args.tp_rank
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    failures: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    graph_medians: dict[int, float] = {}
    keepalive: list[Any] = [weights]
    layout_controls = _run_one_expert_layout_controls(
        torch,
        flashinfer,
        grouped_mm_fp4,
        weights,
        seed=args.seed + 1000,
        minimum_cosine=args.numeric_min_cosine,
        maximum_nrmse=args.numeric_max_nrmse,
    )
    if not layout_controls["required_passed"]:
        failures.append(
            {
                "kind": "one_expert_layout_control",
                "failed_required_variants": [
                    {
                        "m": result["m"],
                        "variant": result["variant"],
                        "error": result.get("error"),
                        "numeric_vs_bf16": result.get("numeric_vs_bf16"),
                    }
                    for result in layout_controls["results"]
                    if result["required"] and not result["passed"]
                ],
            }
        )
    for m in args.m:
        launch, reference, proof = _make_case(
            torch,
            flashinfer,
            grouped_mm_fp4,
            weights,
            m=m,
            seed=args.seed + m,
        )
        eager = launch()
        torch.cuda.synchronize()
        eager = eager.clone()
        numeric = _metrics(torch, eager, reference)
        numeric_passed = _numeric_gate_passed(
            numeric,
            minimum_cosine=args.numeric_min_cosine,
            maximum_nrmse=args.numeric_max_nrmse,
        )
        if not numeric_passed:
            failures.append({"kind": "numeric", "m": m, **numeric})

        replay, graph_output, graph = _capture_graph(torch, launch)
        replay()
        torch.cuda.synchronize()
        graph_numeric = _metrics(torch, graph_output, eager)
        graph_passed = bool(
            graph_numeric["finite"]
            and graph_numeric["actual_nonzero"] > 0
            and graph_numeric["cosine"] >= 0.99999
            and graph_numeric["normalized_rmse"] <= 1.0e-5
        )
        if not graph_passed:
            failures.append({"kind": "graph_parity", "m": m, **graph_numeric})
        timing = _measure(
            torch,
            replay,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        graph_medians[m] = float(timing["median_ms"])
        results.append(
            {
                **proof,
                "numeric_vs_dequantized_w4a16": numeric,
                "numeric_passed": numeric_passed,
                "cuda_graph_vs_eager": graph_numeric,
                "cuda_graph_passed": graph_passed,
                "cuda_graph_timing": timing,
            }
        )
        keepalive.extend((reference, eager, graph_output, graph))

    performance = evaluate_performance_gate(
        graph_medians,
        maximum_m4_ms=args.max_m4_ms,
    )
    if not performance["passed"]:
        failures.append({"kind": "performance", **performance})

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "probe": "native_mxfp4_w4a4_component_sm121",
        "ok": not failures,
        "gpu": {
            "name": torch.cuda.get_device_name(),
            "capability": list(capability),
            "torch": torch.__version__,
            "flashinfer": getattr(flashinfer, "__version__", None),
        },
        "checkpoint": {
            "shard": str(source.shard),
            "shard_size": source.shard.stat().st_size,
            "index_sha256": source.index_sha256,
            "root": source.root,
            "layer": source.layer,
            "tp_rank": args.tp_rank,
            "tensor_contract_count": len(source.tensor_names),
            "selected_experts": list(expert_ids),
            "load_seconds": load_seconds,
        },
        "backend_proof": {
            "quantizer": "flashinfer.mxfp4_quantize",
            "gemm": "flashinfer.grouped_mm_fp4",
            "block_size": MXFP4_BLOCK_SIZE,
            "scale_dtype": "UE8M0/uint8",
            "weight_scale_max_byte": E8M0_K32_BF16_MAX_SCALE_BYTE,
            "weight_scale_sanitization": weights.scale_sanitization,
            "global_scale": None,
            "activation": "OAI SwiGLU alpha=1 beta=0 limit=10",
            "component_only": True,
            "serving_integration_changed": False,
        },
        "settings": {
            "m": list(args.m),
            "top_k": TOP_K,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "seed": args.seed,
            "numeric_min_cosine": args.numeric_min_cosine,
            "numeric_max_nrmse": args.numeric_max_nrmse,
        },
        "performance_gate": performance,
        "one_expert_layout_controls": layout_controls,
        "results": results,
        "memory": {
            "allocated_gib": torch.cuda.memory_allocated() / (1 << 30),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1 << 30),
            "reserved_gib": torch.cuda.memory_reserved() / (1 << 30),
        },
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": report["ok"], "performance": performance}, sort_keys=True))
    print(f"Wrote {args.output}")
    return 0 if not failures else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-dir", type=Path)
    source.add_argument("--shard-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tp-rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--m", type=_csv_positive_ints, default=(1, 4))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--numeric-min-cosine", type=float, default=0.98)
    parser.add_argument("--numeric-max-nrmse", type=float, default=0.25)
    parser.add_argument(
        "--max-m4-ms",
        type=float,
        default=DEFAULT_M4_MAX_MS,
        help=(
            "Maximum complete component CUDA-graph M=4 latency; default is "
            f"5%% faster than the accepted {REFERENCE_CUTLASS_W4A4_M4_MS:.4f} "
            "ms CUTLASS W4A4 layer."
        ),
    )
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
