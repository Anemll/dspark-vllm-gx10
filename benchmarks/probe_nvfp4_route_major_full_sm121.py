#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Real-layer SM121 gate for the full grouped-GEMM route-major prototype."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import benchmark_nvfp4_a4w4_sm121 as kernel_bench
from benchmarks import benchmark_nvfp4_prepared_b12x_sm121 as prepared_bench
from benchmarks.nvfp4_route_major_fc2 import build_route_major_metadata
from benchmarks.nvfp4_route_major_full import (
    RouteMajorFullDispatcher,
    RouteMajorFullWorkspace,
)


def _measure(torch, launch, *, warmup: int, iters: int, repeats: int) -> dict:
    repeat_medians = []
    samples = []
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
        repeat_medians.append(statistics.median(current))
        samples.extend(current)
    return {
        "median_ms": statistics.median(repeat_medians),
        "repeat_medians_ms": repeat_medians,
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples": len(samples),
    }


def run(args: argparse.Namespace) -> int:
    import flashinfer
    import torch
    from vllm.models.deepseek_v4.nvidia.prepared_weight_loading import (
        validate_prepared_layer_file,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("route-major full probe requires one CUDA device")
    capability = tuple(torch.cuda.get_device_capability())
    if capability != (12, 1):
        raise RuntimeError(f"route-major full probe requires SM121, got {capability}")
    if args.tp_rank not in (0, 1):
        raise ValueError("tp-rank must be 0 or 1")
    if args.m not in (2, 4):
        raise ValueError("prototype gate is limited to M=2 or M=4")

    physical = validate_prepared_layer_file(args.layer_file, layer=0)
    shape = kernel_bench.Dsv4Shape(tp_rank=args.tp_rank)
    shape.validate()
    load_started = time.perf_counter()
    tensors = prepared_bench._load_rank(torch, args.layer_file, args.tp_rank)
    accepted_weights = prepared_bench._prepare_weights(torch, tensors, shape)
    runner_args = SimpleNamespace(
        m=(args.m,),
        b12x_max_num_tokens=args.m,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        swiglu_limit=10.0,
    )
    wrapper, accepted_proof = kernel_bench._make_w4a4_runner(
        torch, accepted_weights, shape, runner_args
    )
    if wrapper._moe_output is None:
        raise RuntimeError("accepted wrapper did not allocate a fixed output arena")
    load_seconds = time.perf_counter() - load_started

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
    workspace = RouteMajorFullWorkspace.allocate(torch, metadata, topk_weights)
    route_major = RouteMajorFullDispatcher(
        w13=tensors["w13.weight"],
        w13_scale=tensors["w13.weight_scale"],
        a1_gscale=tensors["a1_gscale"],
        g1_alpha=tensors["g1_alphas"],
        w2=tensors["w2.weight"],
        w2_scale=tensors["w2.weight_scale"],
        a2_gscale=tensors["a2_gscale"],
        g2_alpha=tensors["g2_alphas"],
        swiglu_limit=10.0,
    )
    route_major.validate(torch, metadata)

    accepted_launch, accepted_output = prepared_bench._b12x_launch(
        torch,
        wrapper,
        wrapper._moe_output,
        accepted_weights,
        x,
        topk_ids,
        topk_weights,
        direct_output=True,
    )

    def route_major_launch():
        return route_major.launch(torch, flashinfer, x, workspace)

    # Compile both paths before correctness or timing. Poison the destination
    # to prove the prototype writes the entire token output.
    accepted_launch()
    route_major_launch()
    torch.cuda.synchronize()
    accepted_reference = accepted_output.clone()
    workspace.output.fill_(float("nan"))
    route_output = route_major_launch()
    torch.cuda.synchronize()
    activity = kernel_bench.tensor_activity(torch, route_output)
    numeric = kernel_bench.compare_tensors(torch, route_output, accepted_reference)
    numeric_passed = kernel_bench.numeric_metrics_pass(
        numeric,
        min_cosine=args.numeric_min_cosine,
        max_normalized_rmse=args.numeric_max_nrmse,
    )

    # Order-balanced eager timing. Quantizer allocations are intentionally in
    # the measured route-major path; this is an honest feasibility bound.
    timing_rounds = []
    for order in (("accepted", "route_major"), ("route_major", "accepted")):
        launches = {"accepted": accepted_launch, "route_major": route_major_launch}
        row = {"order": list(order)}
        for name in order:
            row[name] = _measure(
                torch,
                launches[name],
                warmup=args.warmup,
                iters=args.iters,
                repeats=args.repeats,
            )
        timing_rounds.append(row)
    accepted_ms = statistics.median(row["accepted"]["median_ms"] for row in timing_rounds)
    route_ms = statistics.median(row["route_major"]["median_ms"] for row in timing_rounds)
    timing = {
        "rounds": timing_rounds,
        "accepted_median_ms": accepted_ms,
        "route_major_median_ms": route_ms,
        "route_major_over_accepted": route_ms / accepted_ms,
        "speedup_route_major_over_accepted": accepted_ms / route_ms,
    }
    if args.m == 4:
        performance_gate = {
            "kind": "absolute_serving_projection",
            "maximum_route_major_median_ms": args.m4_max_ms,
            "minimum_speedup_over_accepted": args.m4_min_speedup,
            "passed": bool(
                route_ms <= args.m4_max_ms
                and timing["speedup_route_major_over_accepted"]
                >= args.m4_min_speedup
            ),
        }
    else:
        performance_gate = {
            "kind": "matched_m2_regression",
            "maximum_relative_regression": args.m2_max_regression,
            "passed": bool(route_ms <= accepted_ms * (1.0 + args.m2_max_regression)),
        }
    passed = bool(activity["passed"] and numeric_passed and performance_gate["passed"])
    report = {
        "probe": "nvfp4_route_major_full_sm121",
        "passed": passed,
        "gpu": {
            "name": torch.cuda.get_device_name(),
            "capability": list(capability),
            "torch": torch.__version__,
        },
        "checkpoint": {
            "layer_file": str(args.layer_file.resolve()),
            "tp_rank": args.tp_rank,
            "physical_validation": physical,
            "load_seconds": load_seconds,
        },
        "settings": {
            "m": args.m,
            "routing": args.routing,
            "top_k": shape.top_k,
            "seed": args.seed,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
        },
        "route_metadata": {
            "active_experts": list(metadata.active_experts),
            "row_counts": list(metadata.row_counts),
            "routed_rows": metadata.routed_rows,
            "padded_rows": metadata.padded_rows,
            "scale_storage_rows": metadata.scale_storage_rows,
            "handoff_bytes": metadata.handoff_bytes,
            "persistent_workspace_bytes": workspace.persistent_bytes,
        },
        "backend_proof": {
            "accepted": accepted_proof,
            "route_major": {
                "fc1": "flashinfer.group_gemm_nvfp4_nt_groupwise",
                "activation": "triton_oai_swiglu_gather_up_gate",
                "fc2": "flashinfer.group_gemm_nvfp4_nt_groupwise",
                "reduction": "triton_atomically_free_token_reduce",
                "quantizer": "flashinfer.nvfp4_batched_quantize",
                "weight_storage": "exact_prepared_checkpoint_no_transform",
                "prefill_changed": False,
            },
        },
        "activity": activity,
        "numeric_vs_accepted": numeric,
        "numeric_passed": numeric_passed,
        "timing": timing,
        "performance_gate": performance_gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": passed, **timing}, sort_keys=True))
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
    parser.add_argument("--numeric-min-cosine", type=float, default=0.98)
    parser.add_argument("--numeric-max-nrmse", type=float, default=0.25)
    parser.add_argument("--m4-max-ms", type=float, default=0.682812)
    parser.add_argument("--m4-min-speedup", type=float, default=1.130712)
    parser.add_argument("--m2-max-regression", type=float, default=0.03)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
