#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Gate the exact NVFP4 -> native-MXFP4 scale collapse on one real layer.

The NVIDIA checkpoint was produced by expanding each native E8M0/K32 scale
into two identical E4M3/K16 values plus a power-of-two tensor global scale.
The E2M1 nibble payload was copied unchanged.  This bounded diagnostic reverses
only that scale algebra: it proves every adjacent K16 pair is byte-identical,
canonicalizes the prepared global scale by at most one FP32 ULP, and reconstructs
one E8M0 byte by exact exponent addition.  It never dequantizes or requantizes a
weight and B12X must retain the original W13/W2 storage pointers.

The same native object is measured twice against current FlashInfer W4A4 at
M=1,4 balanced: retained direct-micro is diagnostic only; the decision arm is
the patched ModelOpt tensor-core path (``SMALL_M_DIRECT=0``, ``TC_DECODE=1``).
Promotion additionally requires active finite outputs, graph/eager parity,
candidate/W4A4 cosine >= 0.98 and NRMSE <= 0.25, and decision M=4 graph latency
<= 0.682812 ms.  It never constructs or serves a full model.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import benchmark_nvfp4_a4w4_sm121 as kernel_bench
from benchmarks import benchmark_nvfp4_prepared_b12x_sm121 as prepared_bench
from benchmarks import benchmark_nvfp4_prepared_w4a16_packed_sm121 as w4a16_bench


SCHEMA_VERSION = 2
CANDIDATE = "native_mxfp4_e8m0_k32_modelopt_tc"
DIRECT_DIAGNOSTIC = "native_mxfp4_e8m0_k32_direct"
REQUIRED_M = (1, 4)
REQUIRED_ROUTING = "balanced"
MAXIMUM_M4_LATENCY_MS = 0.682812
PINNED_B12X_COMMIT = "7dc6fb8fcc6446ea093537d1657df81985fa5f43"

# Pin complete files rather than only call names.  These functions encode the
# scale layout, global-scale convention, and small-M dispatch contract.  A
# dependency update must explicitly refresh this benchmark instead of silently
# reinterpreting a stored checkpoint.
PINNED_SOURCE_SHA256 = {
    "b12x.moe.fused.w4a16.host": (
        "b12x/moe/fused/w4a16/host.py",
        "c2ca075546e646e449eed1d381a663fec5b63216029302763b65cc7cceb0d98e",
    ),
    "b12x.moe.fused.w4a16.prepare": (
        "b12x/moe/fused/w4a16/prepare.py",
        "55a5e5ff35d09af704aadfd9b933dbb64a927d2aa46c8144e08433522f009033",
    ),
    "b12x.moe.fused.w4a16.kernel": (
        "b12x/moe/fused/w4a16/kernel.py",
        # Exact result of scripts/patch_b12x_w4a16_modelopt_tc_decode.py.
        "c4eaa91d8a6f90b8ec6f6abf87c0f2ecb8d73dd4df6b8ae15fba18c0f1b623cd",
    ),
}

PINNED_API_PARAMETERS = {
    "unswizzle_expert_scales": ("swizzled", "rows", "cols"),
    "prepare_w4a16_e8m0_native_weights": (
        "w13_fp4",
        "w13_e8m0_scale",
        "w13_global_scale",
        "w2_fp4",
        "w2_e8m0_scale",
        "w2_global_scale",
        "activation",
        "params_dtype",
        "w13_layout",
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_contracts(symbols: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on source path/hash or API-signature drift."""

    required_symbols = set(PINNED_API_PARAMETERS)
    if set(symbols) != required_symbols:
        raise RuntimeError(
            "runtime API symbol set drifted: "
            f"expected={sorted(required_symbols)}, observed={sorted(symbols)}"
        )

    files: dict[str, dict[str, Any]] = {}
    api: dict[str, Any] = {}
    for name, symbol in symbols.items():
        parameters = tuple(inspect.signature(symbol).parameters)
        expected = PINNED_API_PARAMETERS[name]
        if parameters != expected:
            raise RuntimeError(
                f"{name} signature drifted: expected={expected}, "
                f"observed={parameters}"
            )
        source_name = str(getattr(symbol, "__module__", ""))
        if source_name not in PINNED_SOURCE_SHA256:
            raise RuntimeError(
                f"{name} module is not pinned: observed={source_name!r}"
            )
        suffix, expected_sha = PINNED_SOURCE_SHA256[source_name]
        source_file = inspect.getsourcefile(symbol)
        if source_file is None:
            raise RuntimeError(f"{name} has no inspectable source file")
        source_path = Path(source_file).resolve()
        if not source_path.as_posix().endswith(suffix):
            raise RuntimeError(
                f"{name} source path drifted: expected suffix={suffix}, "
                f"observed={source_path}"
            )
        observed_sha = _sha256_file(source_path)
        if observed_sha != expected_sha:
            raise RuntimeError(
                f"{source_name} source drifted: expected={expected_sha}, "
                f"observed={observed_sha}"
            )
        files[source_name] = {
            "path": str(source_path),
            "sha256": observed_sha,
            "expected_sha256": expected_sha,
        }
        api[name] = {
            "module": source_name,
            "parameters": list(parameters),
        }

    # The runner is intentionally checked by complete source-file hash above;
    # also pin the parameters this harness actually supplies.
    from b12x.moe.fused.w4a16.kernel import run_w4a16_moe

    run_parameters = tuple(inspect.signature(run_w4a16_moe).parameters)
    required_run_parameters = {
        "a_input",
        "prepared",
        "topk_weights",
        "topk_ids",
        "activation",
        "intermediate_cache13",
        "intermediate_cache2",
        "output",
        "fast_math",
        "swiglu_limit",
    }
    missing = sorted(required_run_parameters - set(run_parameters))
    if missing:
        raise RuntimeError(f"run_w4a16_moe API drifted; missing={missing}")
    kernel_module = run_w4a16_moe.__module__
    suffix, expected_sha = PINNED_SOURCE_SHA256[kernel_module]
    kernel_path = Path(inspect.getsourcefile(run_w4a16_moe) or "").resolve()
    if not kernel_path.as_posix().endswith(suffix):
        raise RuntimeError(f"run_w4a16_moe source path drifted: {kernel_path}")
    kernel_sha = _sha256_file(kernel_path)
    if kernel_sha != expected_sha:
        raise RuntimeError(
            f"{kernel_module} source drifted: expected={expected_sha}, "
            f"observed={kernel_sha}"
        )
    files[kernel_module] = {
        "path": str(kernel_path),
        "sha256": kernel_sha,
        "expected_sha256": expected_sha,
    }
    api["run_w4a16_moe"] = {
        "module": kernel_module,
        "required_parameters": sorted(required_run_parameters),
        "parameters": list(run_parameters),
    }
    return {
        "b12x_commit": PINNED_B12X_COMMIT,
        "files": files,
        "api": api,
        "passed": True,
    }


def expected_conversion_shapes(shape: Any) -> dict[str, tuple[int, ...]]:
    experts = int(shape.num_experts)
    hidden = int(shape.hidden_size)
    intermediate = int(shape.intermediate_size_per_rank)
    if hidden <= 0 or intermediate <= 0 or experts <= 0:
        raise ValueError("conversion dimensions must be positive")
    if hidden % 32 or intermediate % 32:
        raise ValueError("native MXFP4 conversion requires K dimensions divisible by 32")
    return {
        "w13.weight": (experts, 2 * intermediate, hidden // 2),
        "w2.weight": (experts, hidden, intermediate // 2),
        "w13.weight_scale": (experts, 2 * intermediate, hidden // 16),
        "w2.weight_scale": (experts, hidden, intermediate // 16),
        "a1_gscale": (experts,),
        "a2_gscale": (experts,),
        "g1_alphas": (experts,),
        "g2_alphas": (experts,),
        "native_w13_e8m0": (experts, 2 * intermediate, hidden // 32),
        "native_w2_e8m0": (experts, hidden, intermediate // 32),
    }


def validate_prepared_contract(torch: Any, tensors: Mapping[str, Any], shape: Any) -> dict[str, Any]:
    expected = expected_conversion_shapes(shape)
    if set(tensors) != set(prepared_bench.PREPARED_FAMILY_ORDER):
        raise RuntimeError(
            "prepared family set drifted: "
            f"expected={list(prepared_bench.PREPARED_FAMILY_ORDER)}, "
            f"observed={sorted(tensors)}"
        )
    expected_dtypes = {
        "w13.weight": torch.uint8,
        "w2.weight": torch.uint8,
        "w13.weight_scale": torch.float8_e4m3fn,
        "w2.weight_scale": torch.float8_e4m3fn,
        "a1_gscale": torch.float32,
        "a2_gscale": torch.float32,
        "g1_alphas": torch.float32,
        "g2_alphas": torch.float32,
    }
    rows: dict[str, Any] = {}
    for name in prepared_bench.PREPARED_FAMILY_ORDER:
        tensor = tensors[name]
        if tensor.dtype != expected_dtypes[name]:
            raise RuntimeError(
                f"prepared dtype drifted for {name}: expected={expected_dtypes[name]}, "
                f"observed={tensor.dtype}"
            )
        if tuple(tensor.shape) != expected[name]:
            raise RuntimeError(
                f"prepared shape drifted for {name}: expected={expected[name]}, "
                f"observed={tuple(tensor.shape)}"
            )
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise RuntimeError(f"prepared tensor must be contiguous CUDA: {name}")
        rows[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    return {
        "families": list(prepared_bench.PREPARED_FAMILY_ORDER),
        "tensors": rows,
        "w13_physical_layout": "prepared [w3/up, w1/gate] (B12X w13/up_gate)",
        "raw_global_scale_recovery": "weight_scale_2 = g_alpha * a_gscale",
        "exact_scale_collapse": (
            "two identical E4M3/K16 bytes + canonical power-of-two "
            "weight_scale_2 -> one E8M0/K32 exponent byte"
        ),
        "passed": True,
    }


def recover_raw_global_scale(g_alpha: Any, a_gscale: Any) -> Any:
    """Recover ModelOpt ``weight_scale_2`` from prepared CUTLASS algebra."""

    return g_alpha * a_gscale


def _positive_float32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def canonical_power_of_two(value: float, *, maximum_ulp_distance: int = 1) -> tuple[int, float, int]:
    """Return the nearest exact power of two, rejecting non-ModelOpt drift."""

    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("global scale must be positive and finite")
    exponent = int(round(math.log2(value)))
    canonical = math.ldexp(1.0, exponent)
    ulp_distance = abs(
        _positive_float32_bits(value) - _positive_float32_bits(canonical)
    )
    if ulp_distance > maximum_ulp_distance:
        raise ValueError(
            "prepared global scale is not the expected power-of-two value: "
            f"value={value}, nearest=2**{exponent}, ulp_distance={ulp_distance}"
        )
    return exponent, canonical, ulp_distance


def e4m3fn_power_of_two_exponent(byte: int) -> int:
    """Decode one positive E4M3FN power-of-two byte without floating math."""

    if byte < 0 or byte > 255 or byte & 0x80:
        raise ValueError(f"E4M3 scale byte must be positive: {byte}")
    exponent_field = (byte >> 3) & 0x0F
    mantissa = byte & 0x07
    if exponent_field:
        if mantissa:
            raise ValueError(f"E4M3 scale is not a power of two: 0x{byte:02x}")
        return exponent_field - 7
    subnormal_exponents = {1: -9, 2: -8, 4: -7}
    try:
        return subnormal_exponents[mantissa]
    except KeyError as error:
        raise ValueError(f"E4M3 scale is not a positive power of two: 0x{byte:02x}") from error


def collapse_e4m3_pairs_cpu(
    scale_bytes: bytes,
    global_scale: float,
    *,
    maximum_e8m0_byte: int = 247,
) -> bytes:
    """Pure-CPU algebra oracle used by focused unit tests."""

    if len(scale_bytes) % 2:
        raise ValueError("K16 scale-byte count must be even")
    global_exponent, _, _ = canonical_power_of_two(global_scale)
    result = bytearray()
    for offset in range(0, len(scale_bytes), 2):
        first, second = scale_bytes[offset : offset + 2]
        if first != second:
            raise ValueError(
                f"adjacent K16 scales differ at pair {offset // 2}: "
                f"0x{first:02x} != 0x{second:02x}"
            )
        e8m0_byte = e4m3fn_power_of_two_exponent(first) + global_exponent + 127
        if not 0 <= e8m0_byte <= maximum_e8m0_byte:
            raise ValueError(f"collapsed E8M0 byte is out of range: {e8m0_byte}")
        result.append(e8m0_byte)
    return bytes(result)


def _canonical_global_exponents(
    torch: Any, raw_global: Any, *, name: str
) -> tuple[Any, dict[str, Any]]:
    value = raw_global.to(torch.float32).contiguous()
    if value.ndim != 1 or not bool(torch.isfinite(value).all().item()) or not bool(
        (value > 0).all().item()
    ):
        raise RuntimeError(f"{name} global scale must be finite positive [E]")
    exponent = torch.round(torch.log2(value)).to(torch.int32)
    canonical = torch.ldexp(torch.ones_like(value), exponent)
    ulp = (value.view(torch.int32) - canonical.view(torch.int32)).abs()
    max_ulp = int(ulp.max().item())
    if max_ulp > 1:
        raise RuntimeError(
            f"{name} global-scale power-of-two recovery exceeded one ULP: {max_ulp}"
        )
    return exponent, {
        "count": int(value.numel()),
        "minimum_exponent": int(exponent.min().item()),
        "maximum_exponent": int(exponent.max().item()),
        "maximum_ulp_distance": max_ulp,
        "canonicalization": "nearest FP32 power of two; <=1 ULP",
        "passed": True,
    }


def collapse_nvfp4_scale_grid(
    torch: Any,
    swizzled_scale: Any,
    raw_global: Any,
    *,
    rows: int,
    cols: int,
    name: str,
    unswizzle_expert_scales: Callable[..., Any],
) -> tuple[Any, dict[str, Any]]:
    """Collapse E4M3/K16 pairs to exact E8M0/K32 exponent bytes."""

    if cols % 32:
        raise ValueError(f"{name} requires K divisible by 32, got {cols}")
    linear = unswizzle_expert_scales(
        swizzled_scale, rows=rows, cols=cols
    ).contiguous()
    linear_bytes = linear.view(torch.uint8)
    expected = (int(raw_global.numel()), rows, cols // 16)
    if tuple(linear_bytes.shape) != expected:
        raise RuntimeError(
            f"{name} unswizzled scale shape drifted: expected={expected}, "
            f"observed={tuple(linear_bytes.shape)}"
        )
    pairs = linear_bytes.reshape(expected[0], rows, cols // 32, 2)
    unequal_count = int(torch.count_nonzero(pairs[..., 0] != pairs[..., 1]).item())
    if unequal_count:
        raise RuntimeError(f"{name} has {unequal_count} non-identical K16 scale pairs")
    byte = pairs[..., 0]
    if bool(torch.any((byte & 0x80) != 0).item()):
        raise RuntimeError(f"{name} contains negative E4M3 scales")
    exponent_field = ((byte >> 3) & 0x0F).to(torch.int32)
    mantissa = byte & 0x07
    normal = exponent_field > 0
    valid_subnormal = (mantissa == 1) | (mantissa == 2) | (mantissa == 4)
    valid = (normal & (mantissa == 0)) | ((~normal) & valid_subnormal)
    invalid_count = int(torch.count_nonzero(~valid).item())
    if invalid_count:
        raise RuntimeError(
            f"{name} has {invalid_count} E4M3 scales that are not powers of two"
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
    global_exponent, global_proof = _canonical_global_exponents(
        torch, raw_global, name=name
    )
    combined_exponent = block_exponent + global_exponent[:, None, None]
    e8m0 = combined_exponent + 127
    minimum_byte = int(e8m0.min().item())
    maximum_byte = int(e8m0.max().item())
    if minimum_byte < 0 or maximum_byte > 247:
        raise RuntimeError(
            f"{name} collapsed E8M0 range [{minimum_byte}, {maximum_byte}] "
            "exceeds the pinned BF16 serving contract [0, 247]"
        )
    native = e8m0.to(torch.uint8).contiguous()
    # This is an algebra proof, not an approximate tensor comparison: every
    # source byte denotes 2**block_exponent, the canonical global is
    # 2**global_exponent, and E8M0 stores their exact exponent sum.
    if not torch.equal(native.to(torch.int32) - 127, combined_exponent):
        raise RuntimeError(f"{name} exact exponent reconstruction failed")
    pair_count = int(native.numel())
    return native, {
        "k16_scale_count": pair_count * 2,
        "k32_scale_count": pair_count,
        "identical_pair_count": pair_count,
        "nonidentical_pair_count": unequal_count,
        "non_power_of_two_count": invalid_count,
        "e8m0_minimum_byte": minimum_byte,
        "e8m0_maximum_byte": maximum_byte,
        "global_scale": global_proof,
        "exact_exponent_reconstruction": True,
        "passed": True,
    }


def convert_prepared_to_native_mxfp4(
    torch: Any,
    tensors: Mapping[str, Any],
    shape: Any,
) -> tuple[Any, dict[str, Any]]:
    """Reverse NVIDIA's exact scale expansion without touching FP4 payloads."""

    from b12x.moe.fused.w4a16.host import unswizzle_expert_scales
    from b12x.moe.fused.w4a16.prepare import prepare_w4a16_e8m0_native_weights

    contract = validate_prepared_contract(torch, tensors, shape)
    source_contract = verify_source_contracts(
        {
            "unswizzle_expert_scales": unswizzle_expert_scales,
            "prepare_w4a16_e8m0_native_weights": (
                prepare_w4a16_e8m0_native_weights
            ),
        }
    )
    device = tensors["w13.weight"].device
    raw_g1 = recover_raw_global_scale(
        tensors["g1_alphas"], tensors["a1_gscale"]
    ).to(torch.float32)
    raw_g2 = recover_raw_global_scale(
        tensors["g2_alphas"], tensors["a2_gscale"]
    ).to(torch.float32)
    for name, value in (("raw_g1", raw_g1), ("raw_g2", raw_g2)):
        if tuple(value.shape) != (shape.num_experts,):
            raise RuntimeError(f"{name} shape drifted: {tuple(value.shape)}")
        if not bool(torch.isfinite(value).all().item()) or not bool(
            (value > 0).all().item()
        ):
            raise RuntimeError(f"{name} must be positive and finite")

    started = time.perf_counter()
    native_w13_scale, w13_collapse = collapse_nvfp4_scale_grid(
        torch,
        tensors["w13.weight_scale"],
        raw_g1,
        rows=2 * shape.intermediate_size_per_rank,
        cols=shape.hidden_size,
        name="w13",
        unswizzle_expert_scales=unswizzle_expert_scales,
    )
    native_w2_scale, w2_collapse = collapse_nvfp4_scale_grid(
        torch,
        tensors["w2.weight_scale"],
        raw_g2,
        rows=shape.hidden_size,
        cols=shape.intermediate_size_per_rank,
        name="w2",
        unswizzle_expert_scales=unswizzle_expert_scales,
    )
    torch.cuda.synchronize()
    conversion_seconds = time.perf_counter() - started
    ones = torch.ones(shape.num_experts, dtype=torch.float32, device=device)
    prepared = prepare_w4a16_e8m0_native_weights(
        tensors["w13.weight"],
        native_w13_scale,
        ones,
        tensors["w2.weight"],
        native_w2_scale,
        ones.clone(),
        activation="silu",
        params_dtype=torch.bfloat16,
        # Prepared physical W13 is [w3/up, w1/gate], called w13/up_gate by
        # the pinned B12X small-M implementation.
        w13_layout="w13",
    )
    if prepared.weight_layout != "modelopt":
        raise RuntimeError(f"native MXFP4 weight layout drifted: {prepared.weight_layout}")
    if prepared.source_format != "fp4_e8m0_k32":
        raise RuntimeError(f"native MXFP4 source format drifted: {prepared.source_format}")
    pointer_proof = {
        "w13_source_data_ptr": int(tensors["w13.weight"].data_ptr()),
        "w13_candidate_data_ptr": int(prepared.w13.data_ptr()),
        "w2_source_data_ptr": int(tensors["w2.weight"].data_ptr()),
        "w2_candidate_data_ptr": int(prepared.w2.data_ptr()),
        "w13_same_data_ptr": int(prepared.w13.data_ptr())
        == int(tensors["w13.weight"].data_ptr()),
        "w2_same_data_ptr": int(prepared.w2.data_ptr())
        == int(tensors["w2.weight"].data_ptr()),
        "w13_same_storage": int(prepared.w13.untyped_storage().data_ptr())
        == int(tensors["w13.weight"].untyped_storage().data_ptr()),
        "w2_same_storage": int(prepared.w2.untyped_storage().data_ptr())
        == int(tensors["w2.weight"].untyped_storage().data_ptr()),
        "duplicate_weight_bytes": 0,
    }
    if not pointer_proof["w13_same_data_ptr"] or not pointer_proof[
        "w13_same_storage"
    ]:
        raise RuntimeError("B12X native E8M0 preparation copied W13 weights")
    if not pointer_proof["w2_same_data_ptr"] or not pointer_proof[
        "w2_same_storage"
    ]:
        raise RuntimeError("B12X native E8M0 preparation copied W2 weights")
    torch.cuda.synchronize()
    proof = {
        "prepared_checkpoint_contract": contract,
        "source_api_contract": source_contract,
        "conversion": {
            "source": "prepared ModelOpt NVFP4 E4M3/K16",
            "destination": "native MXFP4 E8M0/K32",
            "method": "exact pairwise scale-exponent collapse",
            "weight_payload_transform": "none",
            "weight_payload_pointer_proof": pointer_proof,
            "w13_scale_collapse": w13_collapse,
            "w2_scale_collapse": w2_collapse,
            "native_preparer": (
                "b12x.moe.fused.w4a16.prepare."
                "prepare_w4a16_e8m0_native_weights"
            ),
            "conversion_seconds": conversion_seconds,
            "dequantized_weight_bytes": 0,
            "requantized_weight_bytes": 0,
            "timed_region_includes_conversion": False,
        },
        "runtime": {
            "weight_layout": prepared.weight_layout,
            "source_format": prepared.source_format,
            "scale_format": "e8m0_k32",
            "activation_precision": "bf16",
            "w13_layout": prepared.w13_layout,
            "same_original_w13_storage": True,
            "same_original_w2_storage": True,
        },
    }
    return prepared, proof


def install_small_m_trace(kernel_module: Any) -> tuple[list[dict[str, Any]], Any]:
    """Trace the exact B12X native-small-M compiler selected at runtime."""

    original = kernel_module._compile_w4a16_small_m_direct
    events: list[dict[str, Any]] = []

    def traced(**kwargs: Any) -> Any:
        result = original(**kwargs)
        events.append(
            {
                "m": int(kwargs["m"]),
                "hidden_size": int(kwargs["hidden_size"]),
                "intermediate_size": int(kwargs["intermediate_size"]),
                "num_experts": int(kwargs["num_experts"]),
                "topk": int(kwargs["topk"]),
                "activation": str(kwargs["activation"]),
                "scale_format": str(kwargs["scale_format"]),
                "w13_layout": str(kwargs["w13_layout"]),
                "grid_x": int(result.grid_x),
            }
        )
        return result

    kernel_module._compile_w4a16_small_m_direct = traced
    return events, original


def evaluate_small_m_trace(events: list[dict[str, Any]]) -> dict[str, Any]:
    unique = [dict(items) for items in sorted({tuple(sorted(row.items())) for row in events})]
    passing_m = sorted(
        {
            int(row["m"])
            for row in unique
            if row["scale_format"] == "e8m0_k32"
            and row["w13_layout"] == "w13"
            and row["activation"] == "silu"
            and row["grid_x"] > 0
        }
    )
    return {
        "required_m": list(REQUIRED_M),
        "required_scale_format": "e8m0_k32",
        "required_w13_layout": "w13",
        "observed_unique": unique,
        "passing_m": passing_m,
        "passed": passing_m == list(REQUIRED_M),
    }


def evaluate_performance_gate(candidate_ms: Mapping[int, float]) -> dict[str, Any]:
    if set(candidate_ms) != set(REQUIRED_M):
        raise ValueError(
            f"performance gate requires M={REQUIRED_M}, observed={sorted(candidate_ms)}"
        )
    if any(not math.isfinite(value) or value <= 0 for value in candidate_ms.values()):
        raise ValueError("candidate latency values must be positive and finite")
    m4 = float(candidate_ms[4])
    return {
        "metric": "CUDA-graph median latency (ms)",
        "candidate_ms_by_m": {str(m): float(candidate_ms[m]) for m in REQUIRED_M},
        "maximum_m4_latency_ms": MAXIMUM_M4_LATENCY_MS,
        "m4_latency_ms": m4,
        "passed": m4 <= MAXIMUM_M4_LATENCY_MS,
    }


def run(args: argparse.Namespace) -> int:
    import torch
    from b12x.moe.fused.w4a16 import kernel as w4a16_kernel
    from vllm.models.deepseek_v4.nvidia.prepared_weight_loading import (
        validate_prepared_layer_file,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("native MXFP4 gate requires exactly one CUDA device")
    capability = tuple(torch.cuda.get_device_capability())
    if capability != (12, 1):
        raise RuntimeError(f"native MXFP4 gate requires SM121; got {capability}")
    if tuple(args.m) != REQUIRED_M:
        raise ValueError(f"native MXFP4 decision gate requires --m 1,4; got {args.m}")
    if args.routing != REQUIRED_ROUTING:
        raise ValueError(
            f"native MXFP4 decision gate requires balanced routing; got {args.routing}"
        )
    if args.tp_rank not in (0, 1):
        raise ValueError("TP rank must be 0 or 1")

    physical = validate_prepared_layer_file(args.layer_file, layer=0)
    shape = kernel_bench.Dsv4Shape(tp_rank=args.tp_rank)
    shape.validate()
    tensors = prepared_bench._load_rank(torch, args.layer_file, args.tp_rank)
    w4a4_weights = prepared_bench._prepare_weights(torch, tensors, shape)
    candidate_weights, conversion_proof = convert_prepared_to_native_mxfp4(
        torch, tensors, shape
    )
    shared_comparator_weights = {
        "w13_same_data_ptr": int(candidate_weights.w13.data_ptr())
        == int(w4a4_weights.w13.data_ptr()),
        "w2_same_data_ptr": int(candidate_weights.w2.data_ptr())
        == int(w4a4_weights.w2.data_ptr()),
    }
    if not all(shared_comparator_weights.values()):
        raise RuntimeError(
            "native E8M0 candidate and FlashInfer W4A4 do not share FP4 payloads: "
            f"{shared_comparator_weights}"
        )
    conversion_proof["conversion"][
        "shared_fp4_payload_with_w4a4"
    ] = shared_comparator_weights

    runner_args = SimpleNamespace(
        m=args.m,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        swiglu_limit=10.0,
        fast_math=True,
        w4a16_weight_layout="modelopt",
    )
    w4a4_wrapper, w4a4_proof = kernel_bench._make_w4a4_runner(
        torch, w4a4_weights, shape, runner_args
    )
    w4a4_proof = w4a16_bench.direct_output_backend_proof(w4a4_proof)
    w4a4_arena = w4a4_wrapper._moe_output
    if w4a4_arena is None:
        raise RuntimeError("graph-enabled W4A4 wrapper has no output arena")

    direct_events, original_direct_compile = install_small_m_trace(w4a16_kernel)
    tc_events, original_tc_compile = w4a16_bench.install_compile_trace(
        w4a16_kernel
    )
    original_environment = {
        "B12X_W4A16_SMALL_M_DIRECT": os.environ.get(
            "B12X_W4A16_SMALL_M_DIRECT"
        ),
        "B12X_W4A16_TC_DECODE": os.environ.get("B12X_W4A16_TC_DECODE"),
    }

    def select_mode(*, direct: bool) -> None:
        os.environ["B12X_W4A16_SMALL_M_DIRECT"] = "1" if direct else "0"
        os.environ["B12X_W4A16_TC_DECODE"] = "0" if direct else "1"

    failures: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    candidate_ms: dict[int, float] = {}
    keepalive: list[Any] = [w4a4_wrapper, candidate_weights]
    try:
        for m in args.m:
            x, topk_ids, topk_weights = kernel_bench.make_routes(
                torch,
                shape,
                m,
                routing=args.routing,
                seed=args.seed + m,
                input_rms=1.0,
            )
            w4a4_launch, w4a4_output = prepared_bench._b12x_launch(
                torch,
                w4a4_wrapper,
                w4a4_arena,
                w4a4_weights,
                x,
                topk_ids,
                topk_weights,
                direct_output=True,
            )
            direct_launch, direct_buffers = kernel_bench._make_w4a16_launch(
                torch,
                candidate_weights,
                x,
                topk_ids,
                topk_weights,
                runner_args,
            )
            tc_launch, tc_buffers = kernel_bench._make_w4a16_launch(
                torch,
                candidate_weights,
                x,
                topk_ids,
                topk_weights,
                runner_args,
            )

            eager: dict[str, Any] = {}
            activity: dict[str, Any] = {}
            output = w4a4_launch()
            torch.cuda.synchronize()
            eager["w4a4"] = output.clone()
            activity["w4a4"] = kernel_bench.tensor_activity(torch, output)
            if not activity["w4a4"]["passed"]:
                failures.append(
                    {"kind": "output_activity", "m": m, "backend": "w4a4"}
                )
            for name, launch, direct in (
                (DIRECT_DIAGNOSTIC, direct_launch, True),
                (CANDIDATE, tc_launch, False),
            ):
                select_mode(direct=direct)
                output = launch()
                torch.cuda.synchronize()
                eager[name] = output.clone()
                activity[name] = kernel_bench.tensor_activity(torch, output)
                if not activity[name]["passed"]:
                    failures.append({"kind": "output_activity", "m": m, "backend": name})

            numeric: dict[str, Any] = {}
            numeric_passed: dict[str, bool] = {}
            for label, actual, reference in (
                (f"{DIRECT_DIAGNOSTIC}_vs_w4a4", DIRECT_DIAGNOSTIC, "w4a4"),
                (f"{CANDIDATE}_vs_w4a4", CANDIDATE, "w4a4"),
                (f"{CANDIDATE}_vs_direct", CANDIDATE, DIRECT_DIAGNOSTIC),
            ):
                comparison = kernel_bench.compare_tensors(
                    torch, eager[actual], eager[reference]
                )
                passed = kernel_bench.numeric_metrics_pass(
                    comparison,
                    min_cosine=args.numeric_min_cosine,
                    max_normalized_rmse=args.numeric_max_nrmse,
                )
                numeric[label] = comparison
                numeric_passed[label] = passed
                # The retained-layout direct microkernel is diagnostic only.
                # Its scale ABI is intentionally recorded, but it must not
                # veto the tensor-core decision arm.  Promotion is gated only
                # by the candidate-vs-current-W4A4 comparison.
                if not passed and label == f"{CANDIDATE}_vs_w4a4":
                    failures.append(
                        {"kind": "numeric", "m": m, "comparison": label, **comparison}
                    )

            graph_launches: dict[str, Any] = {}
            graph_status: dict[str, Any] = {}
            for name, launch, direct in (
                ("w4a4", w4a4_launch, None),
                (DIRECT_DIAGNOSTIC, direct_launch, True),
                (CANDIDATE, tc_launch, False),
            ):
                if direct is not None:
                    select_mode(direct=direct)
                replay, graph_output, graph = kernel_bench.capture_graph(torch, launch)
                replay()
                torch.cuda.synchronize()
                graph_numeric = kernel_bench.compare_tensors(
                    torch, graph_output, eager[name]
                )
                graph_passed = kernel_bench.numeric_metrics_pass(
                    graph_numeric,
                    min_cosine=args.numeric_min_cosine,
                    max_normalized_rmse=args.numeric_max_nrmse,
                )
                graph_launches[name] = replay
                graph_status[name] = {
                    "captured": True,
                    "vs_eager": graph_numeric,
                    "passed": graph_passed,
                }
                keepalive.extend((graph_output, graph))
                if not graph_passed:
                    failures.append(
                        {"kind": "graph_numeric", "m": m, "backend": name}
                    )

            direct_timing = prepared_bench._time_orders(
                torch,
                graph_launches,
                warmup=args.warmup,
                iters=args.iters,
                repeats=args.repeats,
                pair=(DIRECT_DIAGNOSTIC, "w4a4"),
            )
            tc_timing = prepared_bench._time_orders(
                torch,
                graph_launches,
                warmup=args.warmup,
                iters=args.iters,
                repeats=args.repeats,
                pair=(CANDIDATE, "w4a4"),
            )
            candidate_ms[m] = float(
                tc_timing["combined"][CANDIDATE]["median_ms"]
            )
            results.append(
                {
                    "m": m,
                    "routing": args.routing,
                    "routed_rows": m * shape.top_k,
                    "activity": activity,
                    "numeric": numeric,
                    "numeric_passed": numeric_passed,
                    "cuda_graph_status": graph_status,
                    "cuda_graph": {
                        "direct_diagnostic_vs_w4a4": direct_timing,
                        "modelopt_tc_decision_vs_w4a4": tc_timing,
                    },
                    "decision_candidate": CANDIDATE,
                }
            )
            keepalive.extend((direct_buffers, tc_buffers))
    finally:
        w4a16_kernel._compile_w4a16_small_m_direct = original_direct_compile
        w4a16_kernel.compile_w4a16_fused_moe = original_tc_compile
        for name, value in original_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    direct_path_gate = evaluate_small_m_trace(direct_events)
    if not direct_path_gate["passed"]:
        failures.append({"kind": "native_direct_path", **direct_path_gate})
    tc_path_gate = w4a16_bench.evaluate_modelopt_tc_contract(
        tc_events, tuple(args.m)
    )
    if not tc_path_gate["passed"]:
        failures.append({"kind": "native_modelopt_tc_path", **tc_path_gate})
    performance_gate = evaluate_performance_gate(candidate_ms)
    if not performance_gate["passed"]:
        failures.append({"kind": "performance", **performance_gate})
    peak_gib = torch.cuda.max_memory_allocated() / (1 << 30)
    memory_gate = {
        "peak_allocated_gib": peak_gib,
        "maximum_peak_allocated_gib": args.max_peak_allocated_gib,
        "passed": peak_gib <= args.max_peak_allocated_gib,
    }
    if not memory_gate["passed"]:
        failures.append({"kind": "memory", **memory_gate})

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "probe": "prepared_nvfp4_to_native_mxfp4_w4a16_vs_w4a4_sm121",
        "gpu": {
            "name": torch.cuda.get_device_name(),
            "capability": list(capability),
            "torch": torch.__version__,
        },
        "checkpoint": {
            "layer_file": str(args.layer_file.resolve()),
            "physical_validation": physical,
            "tp_rank": args.tp_rank,
        },
        "settings": {
            "m": list(args.m),
            "routing": args.routing,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "seed": args.seed,
            "numeric_min_cosine": args.numeric_min_cosine,
            "numeric_max_nrmse": args.numeric_max_nrmse,
            "direct_diagnostic_environment": {
                "B12X_W4A16_SMALL_M_DIRECT": "1",
                "B12X_W4A16_TC_DECODE": "0",
            },
            "decision_environment": {
                "B12X_W4A16_SMALL_M_DIRECT": "0",
                "B12X_W4A16_TC_DECODE": "1",
            },
        },
        "backend_proof": {
            "w4a4": w4a4_proof,
            CANDIDATE: conversion_proof,
        },
        "native_direct_path_gate": direct_path_gate,
        "native_modelopt_tc_path_gate": tc_path_gate,
        "performance_gate": performance_gate,
        "memory_gate": memory_gate,
        "results": results,
        "failures": failures,
        "ok": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(performance_gate, sort_keys=True))
    print(f"Wrote {args.output}")
    return 0 if not failures else 2


def _csv_positive_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("M values must be positive")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--m", type=_csv_positive_ints, default=REQUIRED_M)
    parser.add_argument(
        "--routing", choices=("balanced", "random", "hot"), default=REQUIRED_ROUTING
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--numeric-min-cosine", type=float, default=0.98)
    parser.add_argument("--numeric-max-nrmse", type=float, default=0.25)
    parser.add_argument("--max-peak-allocated-gib", type=float, default=12.0)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
