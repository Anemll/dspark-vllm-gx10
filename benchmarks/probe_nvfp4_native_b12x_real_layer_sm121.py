#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Exercise the native-B12X prepared expert on one real NVFP4 layer."""

from __future__ import annotations

import argparse
import json
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
from benchmarks import probe_nvfp4_dual_decode_real_layer_sm121 as dual_probe


REQUIRED_M = (1, 24, 512)


def _csv_positive_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(","))
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("M values must be positive")
    return result


def run(args: argparse.Namespace) -> int:
    if tuple(args.m) != REQUIRED_M:
        raise RuntimeError("native B12X probe requires --m 1,24,512")
    if os.getenv("VLLM_NVFP4_NATIVE_B12X") != "1":
        raise RuntimeError("native B12X probe requires VLLM_NVFP4_NATIVE_B12X=1")

    import torch
    from vllm.model_executor.layers.fused_moe.experts.nvfp4_dual_decode_moe import (
        NvFp4NativeB12xExperts,
    )

    if tuple(torch.cuda.get_device_capability()) != (12, 1):
        raise RuntimeError("native B12X probe requires SM121")
    shape = kernel_bench.Dsv4Shape(tp_rank=args.tp_rank)
    shape.validate()

    # Keep the CUTLASS oracle physically independent because native B12X
    # deliberately repacks its source parameter storage in place.
    control_tensors = prepared_bench._load_rank(
        torch, args.layer_file, args.tp_rank
    )
    native_tensors = prepared_bench._load_rank(
        torch, args.layer_file, args.tp_rank
    )
    control_weights = prepared_bench._prepare_weights(
        torch, control_tensors, shape
    )
    native_weights = prepared_bench._prepare_weights(
        torch, native_tensors, shape
    )
    runner_args = SimpleNamespace(m=args.m, swiglu_limit=args.swiglu_limit)
    control_runner, control_proof = kernel_bench._make_flashinfer_cutlass_runner(
        torch, control_weights, shape, runner_args
    )
    base_runner, _ = kernel_bench._make_flashinfer_cutlass_runner(
        torch, native_weights, shape, runner_args
    )
    native = NvFp4NativeB12xExperts(
        moe_config=base_runner.experts.moe_config,
        quant_config=base_runner.experts.quant_config,
    )
    layer = SimpleNamespace(
        w13_weight=native_weights.w13,
        w2_weight=native_weights.w2,
    )
    native.initialize_prepared_w4a16_decode(layer)
    prepared = native._prepared_w4a16
    if prepared is None:
        raise RuntimeError("native B12X initialization produced no prepared weights")

    alias = {
        "w13_data_ptr": int(prepared.w13.data_ptr())
        == int(native_weights.w13.data_ptr()),
        "w13_storage_ptr": int(prepared.w13.untyped_storage().data_ptr())
        == int(native_weights.w13.untyped_storage().data_ptr()),
        "w2_data_ptr": int(prepared.w2.data_ptr())
        == int(native_weights.w2.data_ptr()),
        "w2_storage_ptr": int(prepared.w2.untyped_storage().data_ptr())
        == int(native_weights.w2.untyped_storage().data_ptr()),
    }
    if not all(alias.values()) or prepared.weight_layout != "packed":
        raise RuntimeError(
            f"native B12X one-copy contract failed: {alias}, "
            f"layout={prepared.weight_layout!r}"
        )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    keepalive: list[Any] = [control_runner, base_runner, native, prepared]
    for m in args.m:
        x, topk_ids, topk_weights = kernel_bench.make_routes(
            torch,
            shape,
            m,
            routing="balanced",
            seed=args.seed + m,
            input_rms=1.0,
        )
        native_launch, _ = dual_probe._make_launch(
            torch,
            native,
            native_weights,
            shape,
            base_runner.activation,
            x,
            topk_ids,
            topk_weights,
            uniform_decode=False,
            dispatch_flag=[False],
        )
        cutlass_launch, _ = kernel_bench._make_flashinfer_cutlass_launch(
            torch,
            control_runner,
            control_weights,
            shape,
            x,
            topk_ids,
            topk_weights,
        )

        eager: dict[str, Any] = {}
        activity: dict[str, Any] = {}
        graph_numeric: dict[str, Any] = {}
        graph_launches: dict[str, Any] = {}
        for arm, launch in (
            ("native_b12x", native_launch),
            ("cutlass", cutlass_launch),
        ):
            eager[arm] = launch().clone()
            torch.cuda.synchronize()
            activity[arm] = kernel_bench.tensor_activity(torch, eager[arm])
            replay, graph_output, graph = kernel_bench.capture_graph(torch, launch)
            replay()
            torch.cuda.synchronize()
            graph_numeric[arm] = kernel_bench.compare_tensors(
                torch, graph_output, eager[arm]
            )
            graph_launches[arm] = replay
            keepalive.extend((graph_output, graph))

        cross = kernel_bench.compare_tensors(
            torch, eager["native_b12x"], eager["cutlass"]
        )
        numeric_passed = kernel_bench.numeric_metrics_pass(
            cross,
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
            graph_launches,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
            pair=("native_b12x", "cutlass"),
        )
        row = {
            "m": m,
            "activity": activity,
            "graph_vs_eager": graph_numeric,
            "native_vs_cutlass": cross,
            "numeric_passed": numeric_passed,
            "graph_passed": graph_passed,
            "activity_passed": activity_passed,
            "timing": timing,
        }
        rows.append(row)
        if not (numeric_passed and graph_passed and activity_passed):
            failures.append({"kind": "correctness", "m": m})
        combined = timing["combined"]
        print(
            f"M={m:>3} native={combined['native_b12x']['median_ms']:.6f} ms "
            f"cutlass={combined['cutlass']['median_ms']:.6f} ms"
        )

    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.layer_file.resolve()),
        "settings": {
            "m": list(args.m),
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "backend_proof": {
            "class": type(native).__name__,
            "module": type(native).__module__,
            "weight_layout": prepared.weight_layout,
            "source_format": prepared.source_format,
            "scale_bytes": native._w4a16_additional_scale_bytes,
            "weight_alias": alias,
            "cutlass_oracle": control_proof,
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
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--numeric-min-cosine", type=float, default=0.98)
    parser.add_argument("--numeric-max-nrmse", type=float, default=0.25)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
