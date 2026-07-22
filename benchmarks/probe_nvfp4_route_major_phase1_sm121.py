#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Real-layer gate for CuTe phase 1 + grouped FC2 route-major decode."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import benchmark_nvfp4_a4w4_sm121 as kernel_bench
from benchmarks import benchmark_nvfp4_prepared_b12x_sm121 as prepared_bench
from benchmarks.nvfp4_route_major_fc2 import (
    RouteMajorFC2Dispatcher,
    RouteMajorFC2Workspace,
    build_route_major_metadata,
)
from benchmarks.nvfp4_route_major_phase1 import build_phase1_runner


def _measure(torch, launch, *, warmup: int, iters: int, repeats: int) -> dict:
    samples = []
    repeat_medians = []
    for _ in range(repeats):
        for _ in range(warmup):
            launch()
        torch.cuda.synchronize()
        current = []
        for _ in range(iters):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            launch()
            end.record()
            end.synchronize()
            current.append(float(begin.elapsed_time(end)))
        samples.extend(current)
        repeat_medians.append(statistics.median(current))
    return {
        "median_ms": statistics.median(repeat_medians),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "repeat_medians_ms": repeat_medians,
        "samples": len(samples),
    }


def _measure_pair(
    torch,
    launches: dict[str, object],
    *,
    warmup: int,
    iters: int,
    repeats: int,
) -> dict:
    """Measure a matched A/B in alternating execution order."""
    names = tuple(launches)
    if len(names) != 2:
        raise ValueError("paired timing requires exactly two launchers")
    samples = {name: [] for name in names}
    repeat_medians = {name: [] for name in names}
    execution_orders = []
    for repeat in range(repeats):
        order = names if repeat % 2 == 0 else tuple(reversed(names))
        execution_orders.append(list(order))
        for name in order:
            launch = launches[name]
            for _ in range(warmup):
                launch()
            torch.cuda.synchronize()
            current = []
            for _ in range(iters):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                launch()
                end.record()
                end.synchronize()
                current.append(float(begin.elapsed_time(end)))
            samples[name].extend(current)
            repeat_medians[name].append(statistics.median(current))
    return {
        "execution_orders": execution_orders,
        "results": {
            name: {
                "median_ms": statistics.median(repeat_medians[name]),
                "mean_ms": statistics.mean(samples[name]),
                "min_ms": min(samples[name]),
                "max_ms": max(samples[name]),
                "repeat_medians_ms": repeat_medians[name],
                "samples": len(samples[name]),
            }
            for name in names
        },
    }


def run(args: argparse.Namespace) -> int:
    import flashinfer
    import torch
    from vllm.models.deepseek_v4.nvidia.prepared_weight_loading import (
        validate_prepared_layer_file,
    )

    if not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability()) != (12, 1):
        raise RuntimeError("phase-1 route-major gate requires one SM121 GPU")
    if args.m not in (2, 4):
        raise ValueError("phase-1 route-major gate is limited to M=2 or M=4")
    physical = validate_prepared_layer_file(args.layer_file, layer=0)
    shape = kernel_bench.Dsv4Shape(tp_rank=args.tp_rank)
    shape.validate()
    tensors = prepared_bench._load_rank(torch, args.layer_file, args.tp_rank)
    accepted_weights = prepared_bench._prepare_weights(torch, tensors, shape)
    runner_args = SimpleNamespace(
        m=(args.m,),
        b12x_max_num_tokens=args.m,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        swiglu_limit=10.0,
    )
    accepted_wrapper, accepted_proof = kernel_bench._make_w4a4_runner(
        torch, accepted_weights, shape, runner_args
    )
    x, topk_ids, topk_weights = kernel_bench.make_routes(
        torch,
        shape,
        args.m,
        routing=args.routing,
        seed=args.seed,
        input_rms=1.0,
    )
    metadata = build_route_major_metadata(
        topk_ids=tuple(int(v) for v in topk_ids.cpu().reshape(-1).tolist()),
        num_tokens=args.m,
        top_k=shape.top_k,
        weight_experts=shape.num_experts,
        intermediate_size=shape.intermediate_size_per_rank,
        hidden_size=shape.hidden_size,
    )
    phase1 = build_phase1_runner(
        torch,
        x=x,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        metadata=metadata,
        w13=tensors["w13.weight"],
        w13_scale=tensors["w13.weight_scale"],
        a1_gscale=tensors["a1_gscale"],
        g1_alpha=tensors["g1_alphas"],
        a2_gscale=tensors["a2_gscale"],
        max_active_clusters=args.max_active_clusters,
    )
    handoff = phase1.handoff
    phase2_workspace = RouteMajorFC2Workspace.allocate(torch, metadata)
    phase2 = RouteMajorFC2Dispatcher(
        w2=tensors["w2.weight"],
        w2_scale=tensors["w2.weight_scale"],
        g2_alpha=tensors["g2_alphas"],
    )
    accepted_launch, accepted_output = prepared_bench._b12x_launch(
        torch,
        accepted_wrapper,
        accepted_wrapper._moe_output,
        accepted_weights,
        x,
        topk_ids,
        topk_weights,
        direct_output=True,
    )

    def phase2_launch():
        return phase2.launch(torch, flashinfer, handoff, phase2_workspace)

    def route_major_launch():
        phase1.launch()
        return phase2_launch()

    # Compile every component before correctness/timing.
    accepted_launch()
    route_major_launch()
    torch.cuda.synchronize()
    accepted_reference = accepted_output.clone()
    phase2_workspace.output.fill_(float("nan"))
    route_output = route_major_launch()
    torch.cuda.synchronize()
    activity = kernel_bench.tensor_activity(torch, route_output)
    numeric = kernel_bench.compare_tensors(torch, route_output, accepted_reference)
    numeric_passed = kernel_bench.numeric_metrics_pass(
        numeric,
        min_cosine=args.numeric_min_cosine,
        max_normalized_rmse=args.numeric_max_nrmse,
    )
    phase1_timing = _measure(
        torch, phase1.launch, warmup=args.warmup, iters=args.iters, repeats=args.repeats
    )
    phase2_timing = _measure(
        torch, phase2_launch, warmup=args.warmup, iters=args.iters, repeats=args.repeats
    )
    paired = _measure_pair(
        torch,
        {"route_major": route_major_launch, "accepted_fused": accepted_launch},
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
    )
    route_timing = paired["results"]["route_major"]
    accepted_timing = paired["results"]["accepted_fused"]
    speedup = accepted_timing["median_ms"] / route_timing["median_ms"]
    if args.m == 4:
        performance = {
            "kind": "absolute_serving_projection",
            "maximum_median_ms": args.m4_max_ms,
            "minimum_speedup": args.m4_min_speedup,
            "passed": bool(
                route_timing["median_ms"] <= args.m4_max_ms
                and speedup >= args.m4_min_speedup
            ),
        }
    else:
        performance = {
            "kind": "matched_m2_regression",
            "maximum_relative_regression": args.m2_max_regression,
            "passed": bool(
                route_timing["median_ms"]
                <= accepted_timing["median_ms"] * (1.0 + args.m2_max_regression)
            ),
        }
    passed = bool(activity["passed"] and numeric_passed and performance["passed"])
    report = {
        "probe": "nvfp4_route_major_phase1_sm121",
        "passed": passed,
        "checkpoint": {
            "layer_file": str(args.layer_file.resolve()),
            "tp_rank": args.tp_rank,
            "physical_validation": physical,
        },
        "settings": {
            "m": args.m,
            "routing": args.routing,
            "seed": args.seed,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "max_active_clusters": phase1.max_active_clusters,
        },
        "route_metadata": {
            "active_experts": list(metadata.active_experts),
            "row_counts": list(metadata.row_counts),
            "padded_rows": metadata.padded_rows,
            "scale_storage_rows": metadata.scale_storage_rows,
            "handoff_bytes": metadata.handoff_bytes,
        },
        "backend_proof": {
            "accepted": accepted_proof,
            "route_major": {
                "phase1": "MoEPhase1Kernel FC1+OAI-SwiGLU+NVFP4",
                "phase1_weights": (
                    "raw prepared W13 + accepted FlashInfer MMA-to-TMA "
                    "W13-scale view + reciprocal a1 + g1"
                ),
                "handoff_quant_scale": "reciprocal raw prepared a2_gscale",
                "phase2": "group_gemm_nvfp4_nt_groupwise raw W2 + g2",
                "reduction": "Triton atomics-free router-weight sum",
                "prefill_changed": False,
            },
        },
        "activity": activity,
        "numeric_vs_accepted": numeric,
        "numeric_passed": numeric_passed,
        "timing": {
            "phase1": phase1_timing,
            "phase2": phase2_timing,
            "route_major_end_to_end": route_timing,
            "accepted_fused": accepted_timing,
            "paired_execution_orders": paired["execution_orders"],
            "speedup_route_major_over_accepted": speedup,
            "component_sum_ms": phase1_timing["median_ms"] + phase2_timing["median_ms"],
        },
        "performance_gate": performance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": passed, "numeric": numeric, "timing": report["timing"], "performance_gate": performance}, sort_keys=True))
    print(f"Wrote {args.output}")
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--routing", choices=("balanced", "random", "hot"), default="balanced")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--max-active-clusters", type=int)
    parser.add_argument("--numeric-min-cosine", type=float, default=0.98)
    parser.add_argument("--numeric-max-nrmse", type=float, default=0.25)
    parser.add_argument("--m4-max-ms", type=float, default=0.682812)
    parser.add_argument("--m4-min-speedup", type=float, default=1.130712)
    parser.add_argument("--m2-max-regression", type=float, default=0.03)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
