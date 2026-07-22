#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Matched B12X-vs-CUTLASS gate from one prepared NVFP4 layer.

The production checkpoint stores the exact CUTLASS-prepared payload.  This
probe loads one TP rank, preserves the packed weights, converts a distinct
copy of the scale storage to B12X's baked-scale contract, and times both
FlashInfer backends against identical activations and routes.  It is a bounded
hardware gate for the serving optimization; it never constructs a full model.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from benchmarks import benchmark_nvfp4_a4w4_sm121 as kernel_bench
except ImportError:
    # The immutable diagnostic image installs both benchmark entry points as
    # sibling files in /usr/local/bin rather than as an importable package.
    # Load the pinned sibling explicitly so the baked-image gate exercises the
    # exact same implementation as the source-tree test path.
    _kernel_bench_path = Path(__file__).with_name(
        "dspark-benchmark-nvfp4-a4w4-sm121"
    )
    _kernel_bench_loader = importlib.machinery.SourceFileLoader(
        "dspark_benchmark_nvfp4_a4w4_sm121",
        str(_kernel_bench_path),
    )
    _kernel_bench_spec = importlib.util.spec_from_loader(
        _kernel_bench_loader.name,
        _kernel_bench_loader,
    )
    if _kernel_bench_spec is None or _kernel_bench_spec.loader is None:
        raise ImportError(f"cannot load benchmark dependency: {_kernel_bench_path}")
    kernel_bench = importlib.util.module_from_spec(_kernel_bench_spec)
    sys.modules[_kernel_bench_spec.name] = kernel_bench
    _kernel_bench_spec.loader.exec_module(kernel_bench)


SCHEMA_VERSION = 1
PREPARED_NAMESPACE = "__dspark_tp2_nvfp4_cutlass_v1__"
PREPARED_FAMILY_ORDER = (
    "w13.weight",
    "w2.weight",
    "w13.weight_scale",
    "w2.weight_scale",
    "a1_gscale",
    "a2_gscale",
    "g1_alphas",
    "g2_alphas",
)


def _csv_positive_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("M values must be positive")
    return result


def _load_rank(
    torch: Any,
    layer_file: Path,
    tp_rank: int,
) -> dict[str, Any]:
    from safetensors import safe_open

    prefix = f"{PREPARED_NAMESPACE}.layers.0.experts."
    tensors: dict[str, Any] = {}
    with safe_open(str(layer_file), framework="pt", device="cpu") as handle:
        for family in PREPARED_FAMILY_ORDER:
            source = handle.get_tensor(f"{prefix}{family}")[tp_rank]
            if not source.is_contiguous():
                raise RuntimeError(f"prepared rank slice is not contiguous: {family}")
            tensors[family] = source.to("cuda")
    torch.cuda.synchronize()
    return tensors


def _prepare_weights(
    torch: Any,
    tensors: dict[str, Any],
    shape: kernel_bench.Dsv4Shape,
) -> kernel_bench.PreparedWeights:
    cutlass_w13_scale = tensors["w13.weight_scale"]
    cutlass_w2_scale = tensors["w2.weight_scale"]
    cutlass_a1 = tensors["a1_gscale"]
    cutlass_a2 = tensors["a2_gscale"]
    cutlass_g1 = tensors["g1_alphas"]
    cutlass_g2 = tensors["g2_alphas"]

    # Recover ModelOpt's raw per-expert scale_2 from the prepared CUTLASS
    # algebra, then bake it into distinct block-scale storage for B12X.
    raw_g1 = (cutlass_g1 * cutlass_a1).to(torch.float32).contiguous()
    raw_g2 = (cutlass_g2 * cutlass_a2).to(torch.float32).contiguous()
    b12x_w13_scale = cutlass_w13_scale.clone()
    b12x_w2_scale = cutlass_w2_scale.clone()
    kernel_bench._bake_expert_scales(torch, b12x_w13_scale, raw_g1)
    kernel_bench._bake_expert_scales(torch, b12x_w2_scale, raw_g2)
    b12x_w13_mma = kernel_bench._scale_to_mma(
        torch,
        b12x_w13_scale,
        rows=2 * shape.intermediate_size_per_rank,
        cols=shape.hidden_size,
    )
    b12x_w2_mma = kernel_bench._scale_to_mma(
        torch,
        b12x_w2_scale,
        rows=shape.hidden_size,
        cols=shape.intermediate_size_per_rank,
    )
    ones = torch.ones(shape.num_experts, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()

    return kernel_bench.PreparedWeights(
        w13=tensors["w13.weight"],
        w13_sf_modelopt=cutlass_w13_scale,
        w13_sf_swizzled=b12x_w13_scale,
        w13_sf_mma=b12x_w13_mma,
        w2=tensors["w2.weight"],
        w2_sf_modelopt=cutlass_w2_scale,
        w2_sf_swizzled=b12x_w2_scale,
        w2_sf_mma=b12x_w2_mma,
        alpha1=ones,
        alpha2=ones.clone(),
        fc2_input_scale=ones.clone(),
        cutlass_a1_gscale=cutlass_a1,
        cutlass_a2_gscale=cutlass_a2,
        cutlass_g1_alphas=cutlass_g1,
        cutlass_g2_alphas=cutlass_g2,
        metadata={
            "source": "prepared-physical-layer0",
            "source_weight_data_ptrs": {
                "w13": int(tensors["w13.weight"].data_ptr()),
                "w2": int(tensors["w2.weight"].data_ptr()),
            },
            "weight_preparation_contract": {
                "flashinfer_b12x": True,
                "flashinfer_cutlass": True,
            },
            "checkpoint_input_scale_tensor_count": 3 * shape.num_experts,
            "modelopt_activation_scale_contract": {
                "loaded_from_prepared_checkpoint": True,
                "raw_weight_scale_2_recovery": "g_alpha * a_gscale",
            },
        },
    )


def _b12x_launch(
    torch: Any,
    wrapper: Any,
    wrapper_arena: Any,
    weights: kernel_bench.PreparedWeights,
    x: Any,
    topk_ids: Any,
    topk_weights: Any,
    *,
    direct_output: bool,
) -> tuple[Any, Any]:
    output = torch.empty_like(x)
    adapter_output = None if direct_output else torch.empty_like(x)

    def launch() -> Any:
        wrapper._moe_output = output if direct_output else wrapper_arena
        wrapper_output = wrapper.run(
            x=x,
            w1_weight=weights.w13,
            w1_weight_sf=weights.w13_sf_mma,
            w1_alpha=weights.alpha1,
            fc2_input_scale=weights.fc2_input_scale,
            w2_weight=weights.w2,
            w2_weight_sf=weights.w2_sf_mma,
            w2_alpha=weights.alpha2,
            token_selected_experts=topk_ids,
            token_final_scales=topk_weights,
        )
        if direct_output:
            if wrapper_output.data_ptr() != output.data_ptr():
                raise RuntimeError("B12X direct-output pointer contract failed")
        else:
            # Model the exact legacy full-serving chain: wrapper arena ->
            # expert adapter output -> modular kernel final output.
            assert adapter_output is not None
            adapter_output.copy_(wrapper_output)
            output.copy_(adapter_output)
        return output

    return launch, output


def _time_orders(
    torch: Any,
    launches: dict[str, Any],
    *,
    warmup: int,
    iters: int,
    repeats: int,
    pair: tuple[str, str] = ("b12x", "cutlass"),
) -> dict[str, Any]:
    rounds: dict[str, Any] = {}
    orders = (pair, tuple(reversed(pair)))
    for execution_order in orders:
        label = f"{execution_order[0]}_first"
        rounds[label] = {}
        for backend in execution_order:
            rounds[label][backend] = kernel_bench.measure_cuda_events(
                torch,
                launches[backend],
                warmup=warmup,
                iters=iters,
                repeats=repeats,
                flush_l2=None,
            )
    combined: dict[str, Any] = {}
    for backend in pair:
        medians = [
            float(rounds[label][backend]["median_ms"]) for label in rounds
        ]
        combined[backend] = {
            "order_medians_ms": medians,
            "median_ms": statistics.median(medians),
        }
    combined[f"speedup_{pair[0]}_over_{pair[1]}"] = (
        combined[pair[1]]["median_ms"] / combined[pair[0]]["median_ms"]
    )
    return {"rounds": rounds, "combined": combined}


def run(args: argparse.Namespace) -> int:
    import torch
    from vllm.models.deepseek_v4.nvidia.prepared_weight_loading import (
        PREPARED_FAMILY_ORDER as RUNTIME_PREPARED_FAMILY_ORDER,
        PREPARED_NAMESPACE as RUNTIME_PREPARED_NAMESPACE,
        validate_prepared_layer_file,
    )

    if (
        tuple(RUNTIME_PREPARED_FAMILY_ORDER) != PREPARED_FAMILY_ORDER
        or RUNTIME_PREPARED_NAMESPACE != PREPARED_NAMESPACE
    ):
        raise RuntimeError("prepared loader namespace/family contract drifted")

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("prepared B12X gate requires exactly one CUDA device")
    capability = tuple(torch.cuda.get_device_capability())
    if capability != (12, 1):
        raise RuntimeError(f"prepared B12X gate requires SM121; got {capability}")
    if args.tp_rank not in (0, 1):
        raise RuntimeError("TP rank must be 0 or 1")

    physical = validate_prepared_layer_file(args.layer_file, layer=0)
    shape = kernel_bench.Dsv4Shape(tp_rank=args.tp_rank)
    shape.validate()
    load_started = time.perf_counter()
    tensors = _load_rank(torch, args.layer_file, args.tp_rank)
    weights = _prepare_weights(torch, tensors, shape)
    load_seconds = time.perf_counter() - load_started
    runner_args = SimpleNamespace(
        m=args.m,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        swiglu_limit=10.0,
    )
    b12x_wrapper, b12x_proof = kernel_bench._make_w4a4_runner(
        torch, weights, shape, runner_args
    )
    b12x_wrapper_arena = b12x_wrapper._moe_output
    if b12x_wrapper_arena is None:
        raise RuntimeError("graph-enabled B12X wrapper has no output arena")
    cutlass_runner, cutlass_proof = kernel_bench._make_flashinfer_cutlass_runner(
        torch, weights, shape, runner_args
    )

    failures: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    speedups_by_m: dict[int, float] = {}
    keepalive: list[Any] = [b12x_wrapper, cutlass_runner]
    for m in args.m:
        x, topk_ids, topk_weights = kernel_bench.make_routes(
            torch,
            shape,
            m,
            routing="balanced",
            seed=args.seed + m,
            input_rms=1.0,
        )
        b12x_launch, b12x_output = _b12x_launch(
            torch,
            b12x_wrapper,
            b12x_wrapper_arena,
            weights,
            x,
            topk_ids,
            topk_weights,
            direct_output=True,
        )
        legacy_b12x_launch, legacy_b12x_output = _b12x_launch(
            torch,
            b12x_wrapper,
            b12x_wrapper_arena,
            weights,
            x,
            topk_ids,
            topk_weights,
            direct_output=False,
        )
        cutlass_launch, cutlass_output = kernel_bench._make_flashinfer_cutlass_launch(
            torch,
            cutlass_runner,
            weights,
            shape,
            x,
            topk_ids,
            topk_weights,
        )
        launches = {"b12x": b12x_launch, "cutlass": cutlass_launch}
        eager = {}
        activity = {}
        for backend, launch in launches.items():
            output = launch()
            torch.cuda.synchronize()
            eager[backend] = output.clone()
            activity[backend] = kernel_bench.tensor_activity(torch, output)
            if not activity[backend]["passed"]:
                failures.append(
                    {"kind": "output_activity", "m": m, "backend": backend}
                )
        numeric = kernel_bench.compare_tensors(
            torch, eager["b12x"], eager["cutlass"]
        )
        legacy_output = legacy_b12x_launch()
        torch.cuda.synchronize()
        legacy_numeric = kernel_bench.compare_tensors(
            torch, legacy_output, eager["b12x"]
        )
        eager["legacy_two_copy"] = legacy_output.clone()
        # These are two independent B12X launches.  The routed FP4 reduction
        # is not bit deterministic, so require the same numerical envelope as
        # every other cross-launch comparison instead of exact equality.  The
        # output-alias contract itself is proven separately by pointer identity.
        legacy_numeric_passed = kernel_bench.numeric_metrics_pass(
            legacy_numeric,
            min_cosine=args.numeric_min_cosine,
            max_normalized_rmse=args.numeric_max_nrmse,
        )
        if not legacy_numeric_passed:
            failures.append({"kind": "legacy_copy_parity", "m": m})
        numeric_passed = kernel_bench.numeric_metrics_pass(
            numeric,
            min_cosine=args.numeric_min_cosine,
            max_normalized_rmse=args.numeric_max_nrmse,
        )
        if not numeric_passed:
            failures.append({"kind": "numeric", "m": m, **numeric})

        eager_timing = _time_orders(
            torch,
            launches,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        eager_copy_timing = _time_orders(
            torch,
            {
                "direct_output": b12x_launch,
                "legacy_two_copy": legacy_b12x_launch,
            },
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
            pair=("direct_output", "legacy_two_copy"),
        )
        graph_launches = {}
        graph_status = {}
        for backend, launch in {
            **launches,
            "legacy_two_copy": legacy_b12x_launch,
        }.items():
            replay, graph_output, graph = kernel_bench.capture_graph(torch, launch)
            graph_launches[backend] = replay
            keepalive.extend((graph_output, graph))
            replay()
            torch.cuda.synchronize()
            graph_numeric = kernel_bench.compare_tensors(
                torch, graph_output, eager[backend]
            )
            graph_status[backend] = {
                "captured": True,
                "vs_eager": graph_numeric,
                "passed": kernel_bench.numeric_metrics_pass(
                    graph_numeric,
                    min_cosine=args.numeric_min_cosine,
                    max_normalized_rmse=args.numeric_max_nrmse,
                ),
            }
            if not graph_status[backend]["passed"]:
                failures.append(
                    {"kind": "graph_numeric", "m": m, "backend": backend}
                )
        graph_timing = _time_orders(
            torch,
            graph_launches,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        graph_copy_timing = _time_orders(
            torch,
            {
                "direct_output": graph_launches["b12x"],
                "legacy_two_copy": graph_launches["legacy_two_copy"],
            },
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
            pair=("direct_output", "legacy_two_copy"),
        )
        cutlass_speedup = float(
            graph_timing["combined"]["speedup_b12x_over_cutlass"]
        )
        speedups_by_m[m] = cutlass_speedup
        results.append(
            {
                "m": m,
                "routed_rows": m * shape.top_k,
                "numeric": numeric,
                "numeric_passed": numeric_passed,
                "activity": activity,
                "eager": eager_timing,
                "cuda_graph": graph_timing,
                "cuda_graph_status": graph_status,
                "copy_elimination": {
                    "legacy_vs_direct_numeric": legacy_numeric,
                    "legacy_vs_direct_numeric_passed": legacy_numeric_passed,
                    "eager": eager_copy_timing,
                    "cuda_graph": graph_copy_timing,
                    "graph_saved_us": 1000.0
                    * (
                        graph_copy_timing["combined"]["legacy_two_copy"][
                            "median_ms"
                        ]
                        - graph_copy_timing["combined"]["direct_output"][
                            "median_ms"
                        ]
                    ),
                },
            }
        )

    if not speedups_by_m or any(m >= 128 for m in speedups_by_m):
        raise RuntimeError("decode gate requires only M values below 128")
    decode_geomean = math.exp(
        sum(math.log(value) for value in speedups_by_m.values())
        / len(speedups_by_m)
    )
    decode_passed = bool(
        decode_geomean >= args.min_geomean_speedup
        and min(speedups_by_m.values()) >= args.min_per_shape_speedup
    )
    performance_gate = {
        "scope": "decode-only; MTP disabled in the subsequent serving A/B",
        "minimum_geomean_speedup": args.min_geomean_speedup,
        "minimum_per_shape_speedup": args.min_per_shape_speedup,
        "b12x_over_cutlass_geomean": decode_geomean,
        "b12x_over_cutlass_by_m": {
            str(m): value for m, value in speedups_by_m.items()
        },
        "passed": decode_passed,
    }
    if not decode_passed:
        failures.append({"kind": "performance", **performance_gate})
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "probe": "prepared_nvfp4_b12x_vs_cutlass_sm121",
        "gpu": {
            "name": torch.cuda.get_device_name(),
            "capability": list(capability),
            "torch": torch.__version__,
        },
        "checkpoint": {
            "layer_file": str(args.layer_file.resolve()),
            "physical_validation": physical,
            "tp_rank": args.tp_rank,
            "load_and_prepare_seconds": load_seconds,
        },
        "settings": {
            "m": list(args.m),
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "backend_proof": {
            "b12x": {
                **b12x_proof,
                "direct_output_alias": True,
                "legacy_full_serving_copy_count": 2,
            },
            "cutlass": cutlass_proof,
        },
        "performance_gate": performance_gate,
        "results": results,
        "failures": failures,
        "ok": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(performance_gate, sort_keys=True))
    print(f"Wrote {args.output}")
    return 0 if not failures else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--m", type=_csv_positive_ints, default=(1, 4))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--numeric-min-cosine", type=float, default=0.98)
    parser.add_argument("--numeric-max-nrmse", type=float, default=0.25)
    parser.add_argument("--min-geomean-speedup", type=float, default=1.01)
    parser.add_argument("--min-per-shape-speedup", type=float, default=0.97)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
