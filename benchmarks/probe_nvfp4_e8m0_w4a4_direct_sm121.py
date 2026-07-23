#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Real-layer gate for W4A4 arithmetic with packed E8M0/K32 weight scales.

The accepted B12X NVFP4 direct kernel consumes E4M3/K16 weight scales.  The
same kernel already contains an E8M0/K32 load branch used by W4A16.  This probe
keeps activation and weight arithmetic W4A4, collapses each exact pair of
power-of-two K16 scales into one K32 byte, and compares that candidate with
both the accepted E4M3 direct kernel and FlashInfer CUTLASS on the same real
prepared layer, BF16 input, and routes.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from benchmarks import benchmark_nvfp4_a4w4_sm121 as kernel_bench
from benchmarks import benchmark_nvfp4_prepared_b12x_sm121 as prepared_bench
from benchmarks import probe_nvfp4_direct_micro_sm121 as direct_probe


M_VALUE = 4
SCHEMA_VERSION = 1


def candidate_scale_algebra() -> dict[str, str]:
    return {
        "e8m0_block_scale": "E4M3_K16 * raw_weight_scale_2, collapsed in exact K32 pairs",
        "raw_weight_scale_2": "g_alpha * a_gscale",
        "candidate_alpha": "g_alpha / raw_weight_scale_2 == 1 / a_gscale",
        "candidate_a_gscale": "unchanged reciprocal activation input scale",
        "arithmetic": "W4A4; w4a16_mode=False",
    }


def _collapse_scale_grid(
    torch: Any,
    swizzled_scale: Any,
    raw_global: Any,
    *,
    rows: int,
    cols: int,
    name: str,
) -> Any:
    """Collapse the audited power-of-two E4M3/K16 pairs to E8M0/K32."""

    from b12x.moe.fused.w4a16.host import unswizzle_expert_scales

    linear = unswizzle_expert_scales(
        swizzled_scale, rows=rows, cols=cols
    ).contiguous()
    pairs = linear.view(torch.uint8).reshape(
        int(raw_global.numel()), rows, cols // 32, 2
    )
    unequal = int(torch.count_nonzero(pairs[..., 0] != pairs[..., 1]).item())
    if unequal:
        raise RuntimeError(f"{name} has {unequal} unequal K16 pairs")
    byte = pairs[..., 0]
    exponent_field = ((byte >> 3) & 0x0F).to(torch.int32)
    mantissa = byte & 0x07
    normal = exponent_field > 0
    valid_subnormal = (mantissa == 1) | (mantissa == 2) | (mantissa == 4)
    valid = (normal & (mantissa == 0)) | ((~normal) & valid_subnormal)
    if int(torch.count_nonzero(~valid).item()):
        raise RuntimeError(f"{name} contains non-power-of-two E4M3 scales")
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
    global_exponent = torch.round(torch.log2(raw_global)).to(torch.int32)
    canonical = torch.ldexp(torch.ones_like(raw_global), global_exponent)
    if int(
        (raw_global.view(torch.int32) - canonical.view(torch.int32))
        .abs()
        .max()
        .item()
    ) > 1:
        raise RuntimeError(f"{name} global scale is not canonical power-of-two")
    e8m0 = block_exponent + global_exponent[:, None, None] + 127
    if int(e8m0.min().item()) < 0 or int(e8m0.max().item()) > 247:
        raise RuntimeError(f"{name} collapsed E8M0 range is outside [0,247]")
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    result = e8m0.to(torch.uint8).contiguous()
    return result if e8m0_dtype is None else result.view(e8m0_dtype)


def _make_e8m0_tensors(
    torch: Any,
    tensors: dict[str, Any],
    shape: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from b12x.moe.fused.w4a16.prepare import _pack_e8m0_k32_scales

    raw_g1 = (tensors["g1_alphas"] * tensors["a1_gscale"]).to(torch.float32)
    raw_g2 = (tensors["g2_alphas"] * tensors["a2_gscale"]).to(torch.float32)
    w13_e8m0 = _collapse_scale_grid(
        torch,
        tensors["w13.weight_scale"],
        raw_g1,
        rows=2 * shape.intermediate_size_per_rank,
        cols=shape.hidden_size,
        name="w13",
    )
    w2_e8m0 = _collapse_scale_grid(
        torch,
        tensors["w2.weight_scale"],
        raw_g2,
        rows=shape.hidden_size,
        cols=shape.intermediate_size_per_rank,
        name="w2",
    )
    # The direct micro reads physical ModelOpt W13 rows and therefore needs
    # the physical (unrotated) scale rows.  The med/large W4A16 tensor-core
    # preparer rotates W13 for its logical-N GEMM and is not the micro layout.
    w13_packed = _pack_e8m0_k32_scales(
        w13_e8m0,
        size_k=shape.hidden_size,
        size_n=2 * shape.intermediate_size_per_rank,
        row_rotation=None,
    )
    w2_packed = _pack_e8m0_k32_scales(
        w2_e8m0,
        size_k=shape.intermediate_size_per_rank,
        size_n=shape.hidden_size,
        row_rotation=None,
    )

    candidate = dict(tensors)
    candidate["w13.weight_scale"] = w13_packed
    candidate["w2.weight_scale"] = w2_packed
    candidate["g1_alphas"] = (1.0 / tensors["a1_gscale"]).contiguous()
    candidate["g2_alphas"] = (1.0 / tensors["a2_gscale"]).contiguous()
    scale_bytes = int(w13_packed.untyped_storage().nbytes()) + int(
        w2_packed.untyped_storage().nbytes()
    )
    original_scale_bytes = int(
        tensors["w13.weight_scale"].untyped_storage().nbytes()
    ) + int(tensors["w2.weight_scale"].untyped_storage().nbytes())
    proof = {
        "algebra": candidate_scale_algebra(),
        "original_e4m3_k16_bytes": original_scale_bytes,
        "candidate_e8m0_k32_bytes": scale_bytes,
        "scale_byte_ratio": scale_bytes / original_scale_bytes,
        "w13_shape": list(w13_packed.shape),
        "w2_shape": list(w2_packed.shape),
        "w13_row_rotation": None,
        "duplicate_weight_bytes": 0,
    }
    if scale_bytes * 2 != original_scale_bytes:
        raise RuntimeError(f"E8M0 sidecar is not exactly half size: {proof}")
    return candidate, proof


def _compile(
    torch: Any,
    tensors: dict[str, Any],
    shape: Any,
    x: Any,
    topk_ids: Any,
    topk_weights: Any,
    *,
    scale_format: str,
    max_active_ctas: int,
) -> Any:
    original = direct_probe.direct_kernel_kwargs

    def kwargs() -> dict[str, Any]:
        result = original()
        result["scale_format"] = scale_format
        return result

    direct_probe.direct_kernel_kwargs = kwargs
    try:
        runner = direct_probe._compile_direct_runner(
            torch,
            tensors,
            shape,
            x,
            topk_ids,
            topk_weights,
            expected_scale_format=scale_format,
            max_active_ctas=max_active_ctas,
        )
    finally:
        direct_probe.direct_kernel_kwargs = original
    if runner.kernel.scale_format != scale_format or runner.kernel.w4a16_mode:
        raise RuntimeError("compiled candidate kernel semantic contract drifted")
    return runner


def _pair_timing(
    torch: Any,
    launches: dict[str, Any],
    *,
    warmup: int,
    iters: int,
    repeats: int,
) -> dict[str, Any]:
    rows: dict[str, list[float]] = {name: [] for name in launches}
    orders = (tuple(launches), tuple(reversed(tuple(launches))))
    raw: dict[str, Any] = {}
    for order in orders:
        label = "-".join(order)
        raw[label] = {}
        for name in order:
            result = kernel_bench.measure_cuda_events(
                torch,
                launches[name],
                warmup=warmup,
                iters=iters,
                repeats=repeats,
                flush_l2=None,
            )
            raw[label][name] = result
            rows[name].append(float(result["median_ms"]))
    medians = {name: statistics.median(values) for name, values in rows.items()}
    return {
        "orders": raw,
        "median_ms": medians,
        "speedup_e8m0_over_e4m3": medians["e4m3_k16"] / medians["e8m0_k32"],
        "speedup_e8m0_over_cutlass": medians["flashinfer_cutlass"]
        / medians["e8m0_k32"],
    }


def run(args: argparse.Namespace) -> int:
    import torch
    from vllm.models.deepseek_v4.nvidia.prepared_weight_loading import (
        validate_prepared_layer_file,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("E8M0 W4A4 gate requires exactly one CUDA device")
    capability = tuple(torch.cuda.get_device_capability())
    if capability != (12, 1):
        raise RuntimeError(f"E8M0 W4A4 gate requires SM121, got {capability}")
    physical = validate_prepared_layer_file(args.layer_file, layer=0)
    if physical.get("fingerprints_match") is not True:
        raise RuntimeError("prepared physical-layer fingerprints did not match")
    shape = kernel_bench.Dsv4Shape(tp_rank=args.tp_rank)
    shape.validate()

    started = time.perf_counter()
    tensors = prepared_bench._load_rank(torch, args.layer_file, args.tp_rank)
    direct_probe._validate_prepared_tensors(torch, tensors, shape)
    accepted_weights = prepared_bench._prepare_weights(torch, tensors, shape)
    e8m0_tensors, sidecar_proof = _make_e8m0_tensors(torch, tensors, shape)
    load_seconds = time.perf_counter() - started

    x, topk_ids, topk_weights = kernel_bench.make_routes(
        torch,
        shape,
        M_VALUE,
        routing="balanced",
        seed=args.seed,
        input_rms=1.0,
    )
    e4m3 = _compile(
        torch,
        tensors,
        shape,
        x,
        topk_ids,
        topk_weights,
        scale_format="e4m3_k16",
        max_active_ctas=args.mac,
    )
    e8m0 = _compile(
        torch,
        e8m0_tensors,
        shape,
        x,
        topk_ids,
        topk_weights,
        scale_format="e8m0_k32",
        max_active_ctas=args.mac,
    )
    cutlass_runner, cutlass_proof = kernel_bench._make_flashinfer_cutlass_runner(
        torch,
        accepted_weights,
        shape,
        SimpleNamespace(m=(M_VALUE,), swiglu_limit=10.0),
    )
    cutlass_launch, cutlass_output = kernel_bench._make_flashinfer_cutlass_launch(
        torch,
        cutlass_runner,
        accepted_weights,
        shape,
        x,
        topk_ids,
        topk_weights,
    )

    launches = {
        "e8m0_k32": e8m0.launch,
        "e4m3_k16": e4m3.launch,
        "flashinfer_cutlass": cutlass_launch,
    }
    for launch in launches.values():
        launch()
    torch.cuda.synchronize()
    outputs = {
        "e8m0_k32": e8m0.output.clone(),
        "e4m3_k16": e4m3.output.clone(),
        "flashinfer_cutlass": cutlass_output.clone(),
    }
    numeric = {
        name: kernel_bench.compare_tensors(torch, output, outputs["flashinfer_cutlass"])
        for name, output in outputs.items()
        if name != "flashinfer_cutlass"
    }
    numeric_passed = all(
        kernel_bench.numeric_metrics_pass(
            row,
            min_cosine=args.numeric_min_cosine,
            max_normalized_rmse=args.numeric_max_nrmse,
        )
        for row in numeric.values()
    )
    activity = {
        name: kernel_bench.tensor_activity(torch, output)
        for name, output in outputs.items()
    }
    activity_passed = all(row["passed"] for row in activity.values())
    timing = _pair_timing(
        torch,
        launches,
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
    )
    performance_passed = (
        timing["speedup_e8m0_over_e4m3"] >= args.minimum_speedup
    )
    failures = []
    if not numeric_passed:
        failures.append({"kind": "numeric", "metrics": numeric})
    if not activity_passed:
        failures.append({"kind": "output_activity", "activity": activity})
    if not performance_passed:
        failures.append(
            {
                "kind": "performance",
                "minimum_speedup": args.minimum_speedup,
                "observed": timing["speedup_e8m0_over_e4m3"],
            }
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "probe": "prepared_nvfp4_e8m0_w4a4_direct_m4_sm121",
        "gpu": {
            "name": torch.cuda.get_device_name(),
            "capability": list(capability),
            "torch": torch.__version__,
        },
        "checkpoint": {
            "layer_file": str(args.layer_file.resolve()),
            "tp_rank": args.tp_rank,
            "physical_validation": physical,
            "load_and_prepare_seconds": load_seconds,
        },
        "settings": {
            "m": M_VALUE,
            "routing": "balanced",
            "seed": args.seed,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "minimum_speedup": args.minimum_speedup,
            "mac": args.mac,
        },
        "backend_proof": {
            "e8m0_w4a4": {
                "kernel_cache_key": repr(e8m0.kernel.__cache_key__),
                "grid_x": e8m0.grid_x,
                "scale_sidecar": sidecar_proof,
                "serving_integration_claimed": False,
            },
            "e4m3_w4a4": {
                "kernel_cache_key": repr(e4m3.kernel.__cache_key__),
                "grid_x": e4m3.grid_x,
            },
            "flashinfer_cutlass": cutlass_proof,
        },
        "correctness": {
            "activity": activity,
            "vs_flashinfer_cutlass": numeric,
            "passed": numeric_passed and activity_passed,
        },
        "timing": timing,
        "performance_gate": {
            "minimum_speedup_e8m0_over_e4m3": args.minimum_speedup,
            "observed_speedup": timing["speedup_e8m0_over_e4m3"],
            "passed": performance_passed,
        },
        "failures": failures,
        "ok": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "correctness": report["correctness"]["passed"],
                "median_ms": timing["median_ms"],
                "speedup_e8m0_over_e4m3": timing[
                    "speedup_e8m0_over_e4m3"
                ],
                "ok": report["ok"],
            },
            sort_keys=True,
        )
    )
    print(f"Wrote {args.output}")
    return 0 if report["ok"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--numeric-min-cosine", type=float, default=0.98)
    parser.add_argument("--numeric-max-nrmse", type=float, default=0.25)
    parser.add_argument("--minimum-speedup", type=float, default=1.02)
    parser.add_argument("--mac", type=int, default=40)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
