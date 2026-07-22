#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Real-layer SM121 gate for isolated persistent route-major W4A4 decode."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import benchmark_nvfp4_a4w4_sm121 as kernel_bench
from benchmarks import benchmark_nvfp4_prepared_b12x_sm121 as prepared_bench
from benchmarks.nvfp4_route_major_persistent import (
    install_isolated_persistent_kernel,
    readiness_bank,
    simulate_expert_publication,
)


def _measure(
    torch: Any,
    launch: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
    repeats: int,
) -> dict[str, Any]:
    samples: list[float] = []
    repeat_medians: list[float] = []
    for _ in range(repeats):
        for _ in range(warmup):
            launch()
        torch.cuda.synchronize()
        current: list[float] = []
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


def _paired_measure(
    torch: Any,
    launches: dict[str, Callable[[], Any]],
    *,
    warmup: int,
    iters: int,
    repeats: int,
) -> dict[str, Any]:
    names = tuple(launches)
    if len(names) != 2:
        raise ValueError("paired measure requires exactly two launchers")
    rounds = []
    for order in (names, tuple(reversed(names))):
        row: dict[str, Any] = {"order": list(order)}
        for name in order:
            row[name] = _measure(
                torch,
                launches[name],
                warmup=warmup,
                iters=iters,
                repeats=repeats,
            )
        rounds.append(row)
    combined = {
        name: {
            "median_ms": statistics.median(
                float(row[name]["median_ms"]) for row in rounds
            ),
            "order_medians_ms": [float(row[name]["median_ms"]) for row in rounds],
        }
        for name in names
    }
    return {"rounds": rounds, "combined": combined}


def _queue_snapshot(workspace: Any, marker: int) -> dict[str, Any]:
    tail = int(workspace.task_tail.item())
    head = int(workspace.task_head.item())
    ready = tuple(int(value) for value in workspace.task_ready[:tail].cpu().tolist())
    valid_rows = tuple(
        int(value) for value in workspace.task_valid_rows[:tail].cpu().tolist()
    )
    tile_counts = tuple(int(value) for value in workspace.tile_write_count.cpu().tolist())
    return {
        "task_head": head,
        "task_tail": tail,
        "ready_release_count": sum(value == 1 for value in ready),
        "all_ready": bool(ready and all(value == 1 for value in ready)),
        "valid_rows_min": min(valid_rows) if valid_rows else None,
        "valid_rows_max": max(valid_rows) if valid_rows else None,
        "overlap_marker": marker,
        "overlap_observed": any(value >= marker for value in tile_counts),
        "marked_tile_count": sum(value >= marker for value in tile_counts),
        "all_tasks_consumed": bool(tail > 0 and head >= tail),
    }


def _make_wrapper(torch: Any, weights: Any, shape: Any, m: int) -> Any:
    wrapper_args = SimpleNamespace(
        m=(m,),
        b12x_max_num_tokens=m,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        swiglu_limit=10.0,
    )
    wrapper, _ = kernel_bench._make_w4a4_runner(
        torch, weights, shape, wrapper_args
    )
    return wrapper


def run(args: argparse.Namespace) -> int:
    import torch
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
    from vllm.models.deepseek_v4.nvidia.prepared_weight_loading import (
        validate_prepared_layer_file,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("persistent route-major probe requires one CUDA GPU")
    capability = tuple(torch.cuda.get_device_capability())
    if capability != (12, 1):
        raise RuntimeError(f"persistent route-major probe requires SM121, got {capability}")
    if args.m != 4:
        raise ValueError("persistent route-major hardware gate is pinned to M=4")
    if args.tp_rank not in (0, 1):
        raise ValueError("tp-rank must be 0 or 1")

    physical = validate_prepared_layer_file(args.layer_file, layer=0)
    shape = kernel_bench.Dsv4Shape(tp_rank=args.tp_rank)
    shape.validate()
    tensors = prepared_bench._load_rank(torch, args.layer_file, args.tp_rank)
    weights = prepared_bench._prepare_weights(torch, tensors, shape)
    x, topk_ids, topk_weights = kernel_bench.make_routes(
        torch,
        shape,
        args.m,
        routing=args.routing,
        seed=args.seed,
        input_rms=1.0,
    )
    flat_experts = tuple(int(value) for value in topk_ids.cpu().reshape(-1).tolist())
    publication = simulate_expert_publication(flat_experts)
    active_experts = tuple(dict.fromkeys(flat_experts))
    tile_n = moe_dispatch._level_tile_n("fp4")
    slice_chunk = moe_dispatch._DYNAMIC_SLICE_CHUNK
    gate_tiles = (
        shape.intermediate_size_per_rank + tile_n - 1
    ) // tile_n
    task_groups_per_expert = (
        gate_tiles + slice_chunk - 1
    ) // slice_chunk
    expected_tasks = len(active_experts) * task_groups_per_expert

    accepted_wrapper = _make_wrapper(torch, weights, shape, args.m)
    bank_wrappers = tuple(
        _make_wrapper(torch, weights, shape, args.m) for _ in range(2)
    )
    for wrapper in bank_wrappers:
        wrapper._dynamic_workspace = moe_dispatch.allocate_sm120_dynamic_workspace(
            state_E=shape.num_experts,
            weight_E=shape.num_experts,
            routed_rows=args.m * shape.top_k,
            k=shape.hidden_size,
            n=shape.intermediate_size_per_rank,
            num_topk=shape.top_k,
            device=torch.device("cuda"),
            activation_precision="fp4",
        )
    workspaces = tuple(wrapper._dynamic_workspace for wrapper in bank_wrappers)
    if workspaces[0].task_ready.data_ptr() == workspaces[1].task_ready.data_ptr():
        raise RuntimeError("persistent readiness banks alias")

    accepted_launch, accepted_output = prepared_bench._b12x_launch(
        torch,
        accepted_wrapper,
        accepted_wrapper._moe_output,
        weights,
        x,
        topk_ids,
        topk_weights,
        direct_output=True,
    )
    bank_launches = []
    bank_outputs = []
    for wrapper in bank_wrappers:
        launch, output = prepared_bench._b12x_launch(
            torch,
            wrapper,
            wrapper._moe_output,
            weights,
            x,
            topk_ids,
            topk_weights,
            direct_output=True,
        )
        bank_launches.append(launch)
        bank_outputs.append(output)

    keepalive: list[Any] = []
    failures: list[dict[str, Any]] = []
    dynamic_source = args.dynamic_source.resolve() if args.dynamic_source else None
    if moe_dispatch._DYNAMIC_KERNEL_CACHE:
        raise RuntimeError("dynamic cache was populated before isolated gate")

    with install_isolated_persistent_kernel(
        moe_dispatch, source_path=dynamic_source
    ) as source_proof:
        accepted_launch()
        bank_launches[readiness_bank(0)]()
        torch.cuda.synchronize()
        accepted_reference = accepted_output.clone()
        activity = kernel_bench.tensor_activity(torch, bank_outputs[0])
        numeric = kernel_bench.compare_tensors(
            torch, bank_outputs[0], accepted_reference
        )
        numeric_passed = kernel_bench.numeric_metrics_pass(
            numeric,
            min_cosine=args.numeric_min_cosine,
            max_normalized_rmse=args.numeric_max_nrmse,
        )
        queue_bank0 = _queue_snapshot(
            workspaces[0], source_proof.overlap_observation_marker
        )

        # The second physical workspace is not decorative: execute it against
        # the same immutable inputs and require matched output/queue evidence.
        bank_launches[readiness_bank(1)]()
        torch.cuda.synchronize()
        bank_numeric = kernel_bench.compare_tensors(
            torch, bank_outputs[1], bank_outputs[0]
        )
        bank_numeric_passed = kernel_bench.numeric_metrics_pass(
            bank_numeric,
            min_cosine=args.numeric_min_cosine,
            max_normalized_rmse=args.numeric_max_nrmse,
        )
        queue_bank1 = _queue_snapshot(
            workspaces[1], source_proof.overlap_observation_marker
        )

        for bank, queue in enumerate((queue_bank0, queue_bank1)):
            if queue["task_tail"] != expected_tasks:
                failures.append(
                    {
                        "kind": "task_count",
                        "bank": bank,
                        "expected": expected_tasks,
                        "observed": queue["task_tail"],
                    }
                )
            if not queue["all_ready"] or not queue["all_tasks_consumed"]:
                failures.append({"kind": "queue_drain", "bank": bank, **queue})
            if not queue["overlap_observed"]:
                failures.append({"kind": "no_runtime_overlap", "bank": bank})
        if not activity["passed"]:
            failures.append({"kind": "output_activity"})
        if not numeric_passed:
            failures.append({"kind": "numeric_vs_accepted", **numeric})
        if not bank_numeric_passed:
            failures.append({"kind": "double_buffer_numeric", **bank_numeric})

        accepted_graph_launch, accepted_graph_output, accepted_graph = (
            kernel_bench.capture_graph(torch, accepted_launch)
        )
        bank_graph_launches = []
        for launch in bank_launches:
            replay, output, graph = kernel_bench.capture_graph(torch, launch)
            bank_graph_launches.append(replay)
            keepalive.extend((output, graph))
        keepalive.extend((accepted_graph_output, accepted_graph))

        accepted_graph_launch()
        bank_graph_launches[0]()
        torch.cuda.synchronize()
        graph_numeric = kernel_bench.compare_tensors(
            torch, bank_outputs[0], accepted_graph_output
        )
        graph_numeric_passed = kernel_bench.numeric_metrics_pass(
            graph_numeric,
            min_cosine=args.numeric_min_cosine,
            max_normalized_rmse=args.numeric_max_nrmse,
        )
        if not graph_numeric_passed:
            failures.append({"kind": "graph_numeric", **graph_numeric})

        timing = _paired_measure(
            torch,
            {
                "persistent_route_major": bank_graph_launches[0],
                "accepted_fused": accepted_graph_launch,
            },
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        persistent_ms = float(
            timing["combined"]["persistent_route_major"]["median_ms"]
        )
        accepted_ms = float(timing["combined"]["accepted_fused"]["median_ms"])
        performance_gate = {
            "maximum_persistent_median_ms": args.m4_max_ms,
            "observed_persistent_median_ms": persistent_ms,
            "accepted_median_ms": accepted_ms,
            "speedup_over_accepted": accepted_ms / persistent_ms,
            "passed": bool(persistent_ms <= args.m4_max_ms),
        }
        if not performance_gate["passed"]:
            failures.append({"kind": "performance", **performance_gate})

        report = {
            "probe": "nvfp4_route_major_persistent_sm121",
            "passed": not failures,
            "gpu": {
                "name": torch.cuda.get_device_name(),
                "capability": list(capability),
                "torch": torch.__version__,
            },
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
                "swiglu_alpha": 1.0,
                "swiglu_beta": 0.0,
                "swiglu_limit": 10.0,
            },
            "source_proof": asdict(source_proof),
            "producer_consumer_proof": {
                "active_experts": list(active_experts),
                "expert_publication_route_indices": [
                    {"route": route, "expert": expert}
                    for route, expert in publication
                ],
                "gate_tiles_per_expert": gate_tiles,
                "task_slice_chunk": slice_chunk,
                "task_groups_per_expert": task_groups_per_expert,
                "expected_tasks": expected_tasks,
                "readiness": [queue_bank0, queue_bank1],
                "readiness_banks": [
                    int(workspaces[0].task_ready.data_ptr()),
                    int(workspaces[1].task_ready.data_ptr()),
                ],
                "no_global_route_compute_boundary": True,
                "fc2_tma_mma_pipeline_stages": 2,
                "prefill_changed": False,
            },
            "activity": activity,
            "numeric_vs_accepted": numeric,
            "numeric_passed": numeric_passed,
            "double_buffer_numeric": bank_numeric,
            "double_buffer_numeric_passed": bank_numeric_passed,
            "graph_numeric_vs_accepted": graph_numeric,
            "graph_numeric_passed": graph_numeric_passed,
            "timing": timing,
            "performance_gate": performance_gate,
            "failures": failures,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "persistent_median_ms": report["performance_gate"][
                    "observed_persistent_median_ms"
                ],
                "accepted_median_ms": report["performance_gate"][
                    "accepted_median_ms"
                ],
                "overlap_observed": all(
                    row["overlap_observed"]
                    for row in report["producer_consumer_proof"]["readiness"]
                ),
                "numeric": report["numeric_vs_accepted"],
            },
            sort_keys=True,
        )
    )
    print(f"Wrote {args.output}")
    return 0 if report["passed"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dynamic-source", type=Path)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument(
        "--routing", choices=("balanced", "random", "hot"), default="balanced"
    )
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--numeric-min-cosine", type=float, default=0.98)
    parser.add_argument("--numeric-max-nrmse", type=float, default=0.25)
    parser.add_argument("--m4-max-ms", type=float, default=0.682812)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
