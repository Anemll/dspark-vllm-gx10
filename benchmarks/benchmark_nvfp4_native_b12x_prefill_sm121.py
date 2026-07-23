#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Compare native-packed B12X W4A16 with CUTLASS W4A4 at prefill shapes.

This is the cheapest single-payload integration gate after proving that B12X's
modelopt/shared layout is too slow.  It asks whether one native B12X payload can
serve both decode and prefill without a phase-local weight repack.  Both arms
use the same real prepared layer, input activations, and balanced routes.
"""

from __future__ import annotations

import argparse
import json
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
from benchmarks import benchmark_nvfp4_shared_layout_dual_sm121 as shared_bench
from benchmarks import benchmark_prepared_vs_abliterated_b12x_w4a16_sm121 as exact_bench


REQUIRED_M = (512, 1024, 2048)


def _csv_positive_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(","))
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("M values must be positive")
    return result


def run(args: argparse.Namespace) -> int:
    if tuple(args.m) != REQUIRED_M:
        raise RuntimeError(f"prefill gate requires --m {','.join(map(str, REQUIRED_M))}")

    import torch

    if tuple(torch.cuda.get_device_capability()) != (12, 1):
        raise RuntimeError("native B12X prefill gate requires SM121")
    shape = kernel_bench.Dsv4Shape(tp_rank=args.tp_rank)
    shape.validate()
    tensors = prepared_bench._load_rank(torch, args.layer_file, args.tp_rank)
    _, native, conversion = shared_bench._make_scale_views(torch, tensors, shape)

    # CUTLASS keeps the original prepared payload.  Construct it before the
    # private native oracle is repacked in place.
    cutlass_weights = prepared_bench._prepare_weights(torch, tensors, shape)
    cutlass_runner, cutlass_proof = kernel_bench._make_flashinfer_cutlass_runner(
        torch,
        cutlass_weights,
        shape,
        SimpleNamespace(m=args.m, swiglu_limit=args.swiglu_limit),
    )
    native_runner, native_proof = exact_bench._make_exact_b12x_runner(
        torch,
        native,
        max_tokens=max(args.m),
        top_k=shape.top_k,
        swiglu_limit=args.swiglu_limit,
        w13_layout="w31",
    )

    failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    keepalive: list[Any] = [cutlass_runner, native_runner]
    for m in args.m:
        x, topk_ids, topk_weights = kernel_bench.make_routes(
            torch,
            shape,
            m,
            routing="balanced",
            seed=args.seed + m,
            input_rms=1.0,
        )
        native_launch, _, native_scratch = exact_bench._make_launch(
            torch, native_runner, x, topk_ids, topk_weights
        )
        cutlass_launch, _ = kernel_bench._make_flashinfer_cutlass_launch(
            torch,
            cutlass_runner,
            cutlass_weights,
            shape,
            x,
            topk_ids,
            topk_weights,
        )
        keepalive.append(native_scratch)

        eager: dict[str, Any] = {}
        activity: dict[str, Any] = {}
        graphs: dict[str, Any] = {}
        graph_numeric: dict[str, Any] = {}
        for arm, launch in (("native", native_launch), ("cutlass", cutlass_launch)):
            eager[arm] = launch().clone()
            torch.cuda.synchronize()
            activity[arm] = kernel_bench.tensor_activity(torch, eager[arm])
            replay, graph_output, graph = kernel_bench.capture_graph(torch, launch)
            replay()
            torch.cuda.synchronize()
            graph_numeric[arm] = kernel_bench.compare_tensors(
                torch, graph_output, eager[arm]
            )
            graphs[arm] = replay
            keepalive.extend((graph_output, graph))

        cross_numeric = kernel_bench.compare_tensors(
            torch, eager["native"], eager["cutlass"]
        )
        numeric_passed = kernel_bench.numeric_metrics_pass(
            cross_numeric,
            min_cosine=args.numeric_min_cosine,
            max_normalized_rmse=args.numeric_max_nrmse,
        )
        graph_passed = all(
            value["finite"] and float(value["normalized_rmse"]) == 0.0
            for value in graph_numeric.values()
        )
        activity_passed = all(value["passed"] for value in activity.values())
        timing = prepared_bench._time_orders(
            torch,
            graphs,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
            pair=("native", "cutlass"),
        )
        combined = timing["combined"]
        native_ms = float(combined["native"]["median_ms"])
        cutlass_ms = float(combined["cutlass"]["median_ms"])
        native_delta = native_ms / cutlass_ms - 1.0
        decision_passed = native_delta <= args.maximum_native_slowdown
        row = {
            "m": m,
            "activity": activity,
            "activity_passed": activity_passed,
            "graph_vs_eager": graph_numeric,
            "graph_passed": graph_passed,
            "native_vs_cutlass": cross_numeric,
            "numeric_passed": numeric_passed,
            "timing": timing,
            "native_delta_vs_cutlass": native_delta,
            "decision_passed": decision_passed,
        }
        rows.append(row)
        if not (activity_passed and graph_passed and numeric_passed):
            failures.append({"kind": "correctness", "m": m})
        if not decision_passed:
            failures.append({"kind": "decision", "m": m})
        print(
            f"M={m:>4} native={native_ms:.6f} ms "
            f"cutlass={cutlass_ms:.6f} ms delta={native_delta:+.2%}"
        )

    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.layer_file.resolve()),
        "settings": {
            "m": list(args.m),
            "maximum_native_slowdown": args.maximum_native_slowdown,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "conversion": conversion,
        "backend_proof": {
            "native": native_proof,
            "cutlass": cutlass_proof,
        },
        "rows": rows,
        "failures": failures,
        "passed": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")
    print("PASSED" if result["passed"] else f"FAILED: {len(failures)}")
    return 0 if result["passed"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--m", type=_csv_positive_ints, default=REQUIRED_M)
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--swiglu-limit", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--numeric-min-cosine", type=float, default=0.98)
    parser.add_argument("--numeric-max-nrmse", type=float, default=0.25)
    parser.add_argument("--maximum-native-slowdown", type=float, default=0.03)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
