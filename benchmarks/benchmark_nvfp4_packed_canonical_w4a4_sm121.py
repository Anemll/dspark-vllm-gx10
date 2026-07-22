#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Real-layer M=4 packed-canonical W4A4 inverse-loader milestone.

This bounded probe answers whether a W4A16-packed canonical weight object can
feed the existing A4/CUTLASS expert kernel without a second full model copy.
For balanced M=4 routing it packs only the 24 routed experts, reverses their
W4A16 nibble layout into one caller-owned scratch arena, and then launches the
unchanged FlashInfer CUTLASS W4A4 backend with the original raw E4M3 K/16
scales.  The CUDA-graph timing includes both inverse scatters and W4A4 compute.

This is deliberately not serving integration and not yet the final fused TMA
loader.  It is the first compile/correctness/latency gate for the layout port.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import benchmark_nvfp4_a4w4_sm121 as kernel_bench
from benchmarks import benchmark_nvfp4_prepared_b12x_sm121 as prepared_bench
from benchmarks.nvfp4_packed_inverse_ops import unpack_w4a16_packed


SCHEMA_VERSION = 1


def _slice_tensor(value: Any, experts: int) -> Any:
    if value is None:
        return None
    return value[:experts]


def compact_prepared_weights(
    torch: Any,
    weights: kernel_bench.PreparedWeights,
    shape: kernel_bench.Dsv4Shape,
    *,
    experts: int,
) -> kernel_bench.PreparedWeights:
    """Take a contiguous expert prefix and rebuild only its logical SF views."""

    if not 1 <= experts <= shape.num_experts:
        raise ValueError("experts must be in [1, shape.num_experts]")
    w13_sf_swizzled = _slice_tensor(weights.w13_sf_swizzled, experts)
    w2_sf_swizzled = _slice_tensor(weights.w2_sf_swizzled, experts)
    if w13_sf_swizzled is None or w2_sf_swizzled is None:
        raise RuntimeError("packed-canonical probe requires B12X scale storage")
    # convert_sf_to_mma_layout returns a view over the sliced raw scale storage;
    # no second scale payload is materialized.
    w13_sf_mma = kernel_bench._scale_to_mma(
        torch,
        w13_sf_swizzled,
        rows=2 * shape.intermediate_size_per_rank,
        cols=shape.hidden_size,
    )
    w2_sf_mma = kernel_bench._scale_to_mma(
        torch,
        w2_sf_swizzled,
        rows=shape.hidden_size,
        cols=shape.intermediate_size_per_rank,
    )
    w13 = _slice_tensor(weights.w13, experts)
    w2 = _slice_tensor(weights.w2, experts)
    return kernel_bench.PreparedWeights(
        w13=w13,
        w13_sf_modelopt=_slice_tensor(weights.w13_sf_modelopt, experts),
        w13_sf_swizzled=w13_sf_swizzled,
        w13_sf_mma=w13_sf_mma,
        w2=w2,
        w2_sf_modelopt=_slice_tensor(weights.w2_sf_modelopt, experts),
        w2_sf_swizzled=w2_sf_swizzled,
        w2_sf_mma=w2_sf_mma,
        alpha1=_slice_tensor(weights.alpha1, experts),
        alpha2=_slice_tensor(weights.alpha2, experts),
        fc2_input_scale=_slice_tensor(weights.fc2_input_scale, experts),
        cutlass_a1_gscale=_slice_tensor(weights.cutlass_a1_gscale, experts),
        cutlass_a2_gscale=_slice_tensor(weights.cutlass_a2_gscale, experts),
        cutlass_g1_alphas=_slice_tensor(weights.cutlass_g1_alphas, experts),
        cutlass_g2_alphas=_slice_tensor(weights.cutlass_g2_alphas, experts),
        metadata={
            **weights.metadata,
            "packed_canonical_active_expert_prefix": experts,
            "full_layer_experts": shape.num_experts,
            "source_weight_data_ptrs": {
                "w13": int(w13.data_ptr()),
                "w2": int(w2.data_ptr()),
            },
            "checkpoint_input_scale_tensor_count": 3 * experts,
        },
    )


def candidate_weights_from_scratch(
    reference: kernel_bench.PreparedWeights,
    *,
    w13: Any,
    w2: Any,
) -> kernel_bench.PreparedWeights:
    result = dataclasses.replace(reference, w13=w13, w2=w2)
    result.metadata = {
        **reference.metadata,
        "weight_source": "w4a16-packed inverse-scatter scratch",
        "source_weight_data_ptrs": {
            "w13": int(w13.data_ptr()),
            "w2": int(w2.data_ptr()),
        },
    }
    return result


def run(args: argparse.Namespace) -> int:
    import torch
    from vllm.models.deepseek_v4.nvidia.prepared_weight_loading import (
        validate_prepared_layer_file,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("packed-canonical probe requires exactly one CUDA device")
    capability = tuple(torch.cuda.get_device_capability())
    if capability != (12, 1):
        raise RuntimeError(f"packed-canonical probe requires SM121; got {capability}")
    if args.m != 4:
        raise ValueError("first packed-canonical milestone is pinned to M=4")
    if args.routing != "balanced":
        raise ValueError("first milestone requires deterministic balanced routing")
    if args.tp_rank not in (0, 1):
        raise ValueError("TP rank must be 0 or 1")

    physical = validate_prepared_layer_file(args.layer_file, layer=0)
    full_shape = kernel_bench.Dsv4Shape(tp_rank=args.tp_rank)
    full_shape.validate()
    required_experts = args.m * full_shape.top_k
    if args.active_experts != required_experts:
        raise ValueError(
            "balanced M=4 requires exactly M*top_k active experts: "
            f"expected={required_experts}, got={args.active_experts}"
        )
    shape = dataclasses.replace(full_shape, num_experts=args.active_experts)
    shape.validate()

    tensors = prepared_bench._load_rank(torch, args.layer_file, args.tp_rank)
    full_weights = prepared_bench._prepare_weights(torch, tensors, full_shape)
    reference_weights = compact_prepared_weights(
        torch, full_weights, full_shape, experts=args.active_experts
    )
    runner_args = SimpleNamespace(
        m=(args.m,),
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        swiglu_limit=10.0,
        fast_math=True,
        w4a16_weight_layout="packed",
    )

    packed, packed_proof = kernel_bench._prepare_w4a16(
        torch, reference_weights, runner_args
    )
    if getattr(packed, "weight_layout", None) != "packed":
        raise RuntimeError("W4A16 preparation did not produce packed weights")
    restored_w13 = torch.empty_like(reference_weights.w13)
    restored_w2 = torch.empty_like(reference_weights.w2)
    candidate_weights = candidate_weights_from_scratch(
        reference_weights, w13=restored_w13, w2=restored_w2
    )

    def inverse_launch() -> Any:
        unpack_w4a16_packed(
            packed=packed.w13,
            output=restored_w13,
            row_rotation=shape.intermediate_size_per_rank,
        )
        unpack_w4a16_packed(packed=packed.w2, output=restored_w2)
        return restored_w13

    # Compile the Triton adapter and prove bit-exact recovery before any output
    # comparison.  The source/reference weights are diagnostic-only and are not
    # part of the candidate memory contract.
    inverse_launch()
    torch.cuda.synchronize()
    w13_bit_exact = bool(torch.equal(restored_w13, reference_weights.w13))
    w2_bit_exact = bool(torch.equal(restored_w2, reference_weights.w2))
    if not w13_bit_exact or not w2_bit_exact:
        raise RuntimeError(
            "packed inverse was not bit exact: "
            f"w13={w13_bit_exact}, w2={w2_bit_exact}"
        )

    reference_runner, reference_proof = kernel_bench._make_flashinfer_cutlass_runner(
        torch, reference_weights, shape, runner_args
    )
    candidate_runner, candidate_proof = kernel_bench._make_flashinfer_cutlass_runner(
        torch, candidate_weights, shape, runner_args
    )
    x, topk_ids, topk_weights = kernel_bench.make_routes(
        torch,
        shape,
        args.m,
        routing=args.routing,
        seed=args.seed,
        input_rms=1.0,
    )
    expected_ids = torch.arange(
        args.active_experts, device="cuda", dtype=torch.int32
    ).reshape(args.m, shape.top_k)
    if not bool(torch.equal(topk_ids, expected_ids)):
        raise RuntimeError("balanced routing no longer maps to the compact prefix")

    reference_launch, reference_output = kernel_bench._make_flashinfer_cutlass_launch(
        torch,
        reference_runner,
        reference_weights,
        shape,
        x,
        topk_ids,
        topk_weights,
    )
    candidate_compute, candidate_output = kernel_bench._make_flashinfer_cutlass_launch(
        torch,
        candidate_runner,
        candidate_weights,
        shape,
        x,
        topk_ids,
        topk_weights,
    )

    def combined_launch() -> Any:
        inverse_launch()
        return candidate_compute()

    reference_eager = reference_launch().clone()
    candidate_eager = combined_launch().clone()
    torch.cuda.synchronize()
    numeric = kernel_bench.compare_tensors(torch, candidate_eager, reference_eager)
    numeric_passed = kernel_bench.numeric_metrics_pass(
        numeric,
        min_cosine=args.numeric_min_cosine,
        max_normalized_rmse=args.numeric_max_nrmse,
    )
    activity = {
        "reference": kernel_bench.tensor_activity(torch, reference_eager),
        "candidate": kernel_bench.tensor_activity(torch, candidate_eager),
    }

    reference_graph, reference_graph_output, reference_graph_obj = (
        kernel_bench.capture_graph(torch, reference_launch)
    )
    candidate_graph, candidate_graph_output, candidate_graph_obj = (
        kernel_bench.capture_graph(torch, combined_launch)
    )
    inverse_graph, _, inverse_graph_obj = kernel_bench.capture_graph(
        torch, inverse_launch
    )
    candidate_graph()
    reference_graph()
    torch.cuda.synchronize()
    graph_numeric = kernel_bench.compare_tensors(
        torch, candidate_graph_output, reference_graph_output
    )
    graph_passed = kernel_bench.numeric_metrics_pass(
        graph_numeric,
        min_cosine=args.numeric_min_cosine,
        max_normalized_rmse=args.numeric_max_nrmse,
    )

    timing = prepared_bench._time_orders(
        torch,
        {"packed_inverse_w4a4": candidate_graph, "reference_w4a4": reference_graph},
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
        pair=("packed_inverse_w4a4", "reference_w4a4"),
    )
    inverse_timing = kernel_bench.measure_cuda_events(
        torch,
        inverse_graph,
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
        flush_l2=None,
    )
    combined_ms = float(
        timing["combined"]["packed_inverse_w4a4"]["median_ms"]
    )
    reference_ms = float(timing["combined"]["reference_w4a4"]["median_ms"])
    inverse_ms = float(inverse_timing["median_ms"])

    packed_weight_bytes = int(packed.w13.numel() * packed.w13.element_size()) + int(
        packed.w2.numel() * packed.w2.element_size()
    )
    scratch_bytes = int(restored_w13.numel()) + int(restored_w2.numel())
    full_layer_weight_bytes = full_shape.num_experts * (
        2
        * full_shape.intermediate_size_per_rank
        * (full_shape.hidden_size // 2)
        + full_shape.hidden_size * (full_shape.intermediate_size_per_rank // 2)
    )
    failures: list[dict[str, Any]] = []
    if not numeric_passed:
        failures.append({"kind": "numeric", "phase": "eager", **numeric})
    if not graph_passed:
        failures.append({"kind": "numeric", "phase": "cuda_graph", **graph_numeric})
    for name, proof in activity.items():
        if not proof["passed"]:
            failures.append({"kind": "output_activity", "backend": name})
    for name, value in (
        ("combined_ms", combined_ms),
        ("reference_ms", reference_ms),
        ("inverse_ms", inverse_ms),
    ):
        if not math.isfinite(value) or value <= 0:
            failures.append({"kind": "timing", "field": name, "value": value})

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "probe": "packed_canonical_active_expert_inverse_w4a4_sm121",
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
            "m": args.m,
            "top_k": shape.top_k,
            "routing": args.routing,
            "active_experts": args.active_experts,
            "full_layer_experts": full_shape.num_experts,
            "seed": args.seed,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
        },
        "layout_proof": {
            "packed_weight_layout": getattr(packed, "weight_layout", None),
            "packed_w13_layout": getattr(packed, "w13_layout", None),
            "w13_row_rotation": shape.intermediate_size_per_rank,
            "w13_bit_exact": w13_bit_exact,
            "w2_bit_exact": w2_bit_exact,
            "raw_scale_storage_reused": (
                int(candidate_weights.w13_sf_swizzled.data_ptr())
                == int(reference_weights.w13_sf_swizzled.data_ptr())
                and int(candidate_weights.w2_sf_swizzled.data_ptr())
                == int(reference_weights.w2_sf_swizzled.data_ptr())
            ),
            "candidate_scratch_bytes": scratch_bytes,
            "packed_active_weight_bytes": packed_weight_bytes,
            "full_layer_weight_bytes": full_layer_weight_bytes,
            "scratch_fraction_of_full_layer": scratch_bytes / full_layer_weight_bytes,
            "full_model_weight_copy_constructed": False,
            "scope": "contiguous balanced-routing expert prefix; benchmark only",
        },
        "backend_proof": {
            "packed_w4a16": packed_proof,
            "reference_w4a4": reference_proof,
            "candidate_w4a4": candidate_proof,
            "activation_precision": "nvfp4/A4",
            "raw_scale_format": "E4M3 K/16",
        },
        "correctness": {
            "eager": numeric,
            "eager_passed": numeric_passed,
            "cuda_graph": graph_numeric,
            "cuda_graph_passed": graph_passed,
            "activity": activity,
        },
        "timing": {
            "ordered_pair": timing,
            "inverse_only": inverse_timing,
            "combined_median_ms": combined_ms,
            "reference_w4a4_median_ms": reference_ms,
            "inverse_only_median_ms": inverse_ms,
            "combined_over_reference": combined_ms / reference_ms,
            "reference_over_combined_speedup": reference_ms / combined_ms,
            "note": "combined includes two inverse-scatter kernels plus unchanged W4A4",
        },
        "memory": {
            "allocated_gib": torch.cuda.memory_allocated() / (1 << 30),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1 << 30),
            "reserved_gib": torch.cuda.memory_reserved() / (1 << 30),
        },
        "failures": failures,
        "ok": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "combined_ms": combined_ms,
                "reference_ms": reference_ms,
                "inverse_ms": inverse_ms,
                "numeric_passed": numeric_passed,
                "graph_passed": graph_passed,
                "scratch_fraction": scratch_bytes / full_layer_weight_bytes,
            },
            sort_keys=True,
        )
    )
    print(f"Wrote {args.output}")
    # Keep graph-owned tensors alive through report publication.
    _ = (candidate_output, reference_output, reference_graph_obj, candidate_graph_obj, inverse_graph_obj)
    return 0 if not failures else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--active-experts", type=int, default=24)
    parser.add_argument("--routing", choices=("balanced",), default="balanced")
    parser.add_argument("--seed", type=int, default=4108)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--numeric-min-cosine", type=float, default=0.999999)
    parser.add_argument("--numeric-max-nrmse", type=float, default=1.0e-6)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
