#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Gate one-copy NVFP4 weights for CUTLASS prefill plus B12X decode.

The prepared checkpoint's FP4 payload remains in its existing ``w13``
``[up, gate]`` layout.  E8M0/K32 scales are generated losslessly at load time
and B12X is asked to consume that same storage through its source-rotation
contract.  A temporary native ``w31`` copy is retained only as the benchmark
oracle.  Promotion requires:

* shared-layout B12X output numerically matches native-layout B12X;
* shared-layout B12X is within 1% of native B12X at M=24 and M=48;
* shared-layout B12X is at least 3% faster than W4A4 CUTLASS at both shapes;
* the prepared B12X object aliases the original FP4 weight storage.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import benchmark_nvfp4_a4w4_sm121 as kernel_bench
from benchmarks import benchmark_nvfp4_prepared_b12x_sm121 as prepared_bench
from benchmarks import benchmark_nvfp4_prepared_deepgemm_w4a8_sm121 as deep_bench
from benchmarks import benchmark_nvfp4_prepared_mxfp4_requant_sm121 as collapse_bench
from benchmarks import benchmark_prepared_vs_abliterated_b12x_w4a16_sm121 as exact_bench


REQUIRED_M = (1, 4, 24, 48)
DECISION_M = (24, 48)


def _csv_positive_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(","))
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("M values must be positive")
    return result


def _make_scale_views(
    torch: Any,
    tensors: Mapping[str, Any],
    shape: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from b12x.moe.fused.w4a16.host import unswizzle_expert_scales

    raw_g1 = collapse_bench.recover_raw_global_scale(
        tensors["g1_alphas"], tensors["a1_gscale"]
    ).to(torch.float32)
    raw_g2 = collapse_bench.recover_raw_global_scale(
        tensors["g2_alphas"], tensors["a2_gscale"]
    ).to(torch.float32)
    w13_scale, w13_proof = collapse_bench.collapse_nvfp4_scale_grid(
        torch,
        tensors["w13.weight_scale"],
        raw_g1,
        rows=2 * shape.intermediate_size_per_rank,
        cols=shape.hidden_size,
        name="w13",
        unswizzle_expert_scales=unswizzle_expert_scales,
    )
    w2_scale, w2_proof = collapse_bench.collapse_nvfp4_scale_grid(
        torch,
        tensors["w2.weight_scale"],
        raw_g2,
        rows=shape.hidden_size,
        cols=shape.intermediate_size_per_rank,
        name="w2",
        unswizzle_expert_scales=unswizzle_expert_scales,
    )
    shared = {
        "w13": tensors["w13.weight"],
        "w13_scale": w13_scale,
        "w2": tensors["w2.weight"],
        "w2_scale": w2_scale,
    }
    native = {
        "w13": deep_bench.swap_gate_up_halves(shared["w13"]),
        "w13_scale": deep_bench.swap_gate_up_halves(w13_scale),
        "w2": shared["w2"],
        "w2_scale": w2_scale,
    }
    proof = {
        "w13": w13_proof,
        "w2": w2_proof,
        "shared_layout": "w13 (up_gate)",
        "native_oracle_layout": "w31 (gate_up)",
        "persistent_duplicate_weight_bytes": 0,
        "generated_scale_bytes": int(w13_scale.nbytes + w2_scale.nbytes),
    }
    return shared, native, proof


def _paired_timing(
    torch: Any,
    launches: Mapping[str, Any],
    *,
    warmup: int,
    iters: int,
    repeats: int,
) -> dict[str, Any]:
    orders = (
        ("shared", "native", "cutlass"),
        ("cutlass", "native", "shared"),
        ("native", "shared", "cutlass"),
    )
    rounds: dict[str, Any] = {}
    for index, order in enumerate(orders):
        label = f"order_{index}"
        rounds[label] = {"execution_order": list(order)}
        for arm in order:
            rounds[label][arm] = kernel_bench.measure_cuda_events(
                torch,
                launches[arm],
                warmup=warmup,
                iters=iters,
                repeats=repeats,
                flush_l2=None,
            )
    combined: dict[str, Any] = {}
    for arm in ("shared", "native", "cutlass"):
        values = [
            float(rounds[label][arm]["median_ms"]) for label in rounds
        ]
        combined[arm] = {
            "order_medians_ms": values,
            "median_ms": statistics.median(values),
        }
    combined["shared_delta_vs_native"] = (
        combined["shared"]["median_ms"] / combined["native"]["median_ms"] - 1.0
    )
    combined["shared_speedup_over_cutlass"] = (
        combined["cutlass"]["median_ms"] / combined["shared"]["median_ms"]
    )
    return {"rounds": rounds, "combined": combined}


def run(args: argparse.Namespace) -> int:
    if tuple(args.m) != REQUIRED_M:
        raise RuntimeError("shared-layout gate requires --m 1,4,24,48")
    import torch

    if tuple(torch.cuda.get_device_capability()) != (12, 1):
        raise RuntimeError("shared-layout gate requires SM121")
    shape = kernel_bench.Dsv4Shape(tp_rank=args.tp_rank)
    shape.validate()
    tensors = prepared_bench._load_rank(torch, args.layer_file, args.tp_rank)
    collapse_bench.validate_prepared_contract(torch, tensors, shape)
    shared, native, conversion = _make_scale_views(torch, tensors, shape)

    shared_runner, shared_proof = exact_bench._make_exact_b12x_runner(
        torch,
        shared,
        max_tokens=max(args.m),
        top_k=shape.top_k,
        swiglu_limit=args.swiglu_limit,
        w13_layout="w13",
    )
    native_runner, native_proof = exact_bench._make_exact_b12x_runner(
        torch,
        native,
        max_tokens=max(args.m),
        top_k=shape.top_k,
        swiglu_limit=args.swiglu_limit,
        w13_layout="w31",
    )
    prepared_shared = shared_runner["prepared_w4a16"]
    alias_proof = {
        "w13_same_data_ptr": int(prepared_shared.w13.data_ptr())
        == int(shared["w13"].data_ptr()),
        "w2_same_data_ptr": int(prepared_shared.w2.data_ptr())
        == int(shared["w2"].data_ptr()),
        "w13_same_storage_ptr": int(prepared_shared.w13.untyped_storage().data_ptr())
        == int(shared["w13"].untyped_storage().data_ptr()),
        "w2_same_storage_ptr": int(prepared_shared.w2.untyped_storage().data_ptr())
        == int(shared["w2"].untyped_storage().data_ptr()),
    }
    if not all(alias_proof.values()):
        raise RuntimeError(f"shared-layout B12X duplicated FP4 storage: {alias_proof}")

    cutlass_weights = deep_bench._make_cutlass_weights(torch, tensors)
    cutlass_runner, cutlass_proof = kernel_bench._make_flashinfer_cutlass_runner(
        torch,
        cutlass_weights,
        shape,
        SimpleNamespace(m=args.m, swiglu_limit=args.swiglu_limit),
    )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    keepalive: list[Any] = [shared_runner, native_runner, cutlass_runner]
    for m in args.m:
        x, topk_ids, topk_weights = kernel_bench.make_routes(
            torch,
            shape,
            m,
            routing="balanced",
            seed=args.seed + m,
            input_rms=1.0,
        )
        shared_launch, _, shared_scratch = exact_bench._make_launch(
            torch, shared_runner, x, topk_ids, topk_weights
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
        keepalive.extend((shared_scratch, native_scratch))
        eager = {}
        activity = {}
        graph_launches = {}
        graph_compare = {}
        for arm, launch in (
            ("shared", shared_launch),
            ("native", native_launch),
            ("cutlass", cutlass_launch),
        ):
            eager[arm] = launch().clone()
            torch.cuda.synchronize()
            activity[arm] = kernel_bench.tensor_activity(torch, eager[arm])
            graph_launch, _, graph = kernel_bench.capture_graph(torch, launch)
            keepalive.append(graph)
            graph_value = graph_launch().clone()
            torch.cuda.synchronize()
            graph_compare[arm] = kernel_bench.compare_tensors(
                torch, graph_value, eager[arm]
            )
            graph_launches[arm] = graph_launch

        shared_vs_native = kernel_bench.compare_tensors(
            torch, eager["shared"], eager["native"]
        )
        shared_vs_cutlass = kernel_bench.compare_tensors(
            torch, eager["shared"], eager["cutlass"]
        )
        numeric_passed = bool(
            shared_vs_native["finite"]
            and shared_vs_native["cosine"] >= args.numeric_min_cosine
            and shared_vs_native["normalized_rmse"] <= args.numeric_max_nrmse
            and shared_vs_cutlass["finite"]
            and shared_vs_cutlass["cosine"] >= args.numeric_min_cosine
            and shared_vs_cutlass["normalized_rmse"] <= args.numeric_max_nrmse
        )
        activity_passed = all(bool(value["passed"]) for value in activity.values())
        graph_passed = all(
            bool(value["finite"]) and float(value["normalized_rmse"]) == 0.0
            for value in graph_compare.values()
        )
        timing = _paired_timing(
            torch,
            graph_launches,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        decision_passed = True
        if m in DECISION_M:
            decision_passed = bool(
                abs(timing["combined"]["shared_delta_vs_native"]) <= 0.01
                and timing["combined"]["shared_speedup_over_cutlass"] >= 1.03
            )
        row = {
            "m": m,
            "activity": activity,
            "activity_passed": activity_passed,
            "graph_vs_eager": graph_compare,
            "graph_passed": graph_passed,
            "shared_vs_native": shared_vs_native,
            "shared_vs_cutlass": shared_vs_cutlass,
            "numeric_passed": numeric_passed,
            "timing": timing,
            "decision_passed": decision_passed,
        }
        rows.append(row)
        if not (numeric_passed and activity_passed and graph_passed):
            failures.append({"kind": "correctness", "m": m})
        if not decision_passed:
            failures.append({"kind": "decision", "m": m})
        print(
            f"M={m:>2} shared={timing['combined']['shared']['median_ms']:.6f} "
            f"native={timing['combined']['native']['median_ms']:.6f} "
            f"cutlass={timing['combined']['cutlass']['median_ms']:.6f} ms "
            f"shared/native={timing['combined']['shared_delta_vs_native']:+.2%} "
            f"speedup/CUTLASS={timing['combined']['shared_speedup_over_cutlass']:.4f}x"
        )

    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.layer_file.resolve()),
        "settings": {
            "m": list(args.m),
            "decision_m": list(DECISION_M),
            "maximum_shared_native_delta": 0.01,
            "minimum_cutlass_speedup": 1.03,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "conversion": conversion,
        "weight_alias_proof": alias_proof,
        "backend_proof": {
            "shared": shared_proof,
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
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
