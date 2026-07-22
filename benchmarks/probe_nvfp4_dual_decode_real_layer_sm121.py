#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Prove the prepared NVFP4 CUTLASS/W4A16 dual expert on one real layer.

This is an integration gate, not a synthetic kernel microbenchmark.  It loads
one immutable prepared DeepSeek-V4 layer, constructs the exact vLLM
``NvFp4CutlassW4A16DualExperts`` object, initializes its E8M0/K32 scale
sidecar, and exercises the serving ``apply`` method.  The gate keeps the FP4
payload single-copy, keeps M=1 on CUTLASS W4A4, and requires M=2/4/8 to compile
the ModelOpt ``tc_decode_fused_sum`` W4A16 branch.

The direct runner temporarily supplies a uniform-decode forward descriptor
because it does not construct a complete vLLM scheduler.  The normal policy
still decides the cutover: M=1 remains CUTLASS and only M=2..8 can enter
W4A16.  Tiny prefill is covered by the CPU policy tests and is not represented
as decode by this probe.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import benchmark_nvfp4_a4w4_sm121 as kernel_bench
from benchmarks import benchmark_nvfp4_cutlass_fused_input_quant_sm121 as cutlass_bench
from benchmarks import benchmark_nvfp4_prepared_b12x_sm121 as prepared_bench
from benchmarks import benchmark_nvfp4_prepared_w4a16_packed_sm121 as w4a16_bench


SCHEMA_VERSION = 1
EXPECTED_M = (1, 2, 4, 8)
W4A16_M = (2, 4, 8)
EXPECTED_ENVIRONMENT = {
    "VLLM_NVFP4_W4A16_DUAL_DECODE": "1",
    "VLLM_NVFP4_W4A16_DECODE_MIN_M": "2",
    "VLLM_NVFP4_W4A16_DECODE_MAX_M": "8",
    "B12X_W4A16_TC_DECODE": "1",
    "B12X_W4A16_SMALL_M_DIRECT": "0",
}


def _csv_positive_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("M values must be positive")
    return result


def require_exact_m(values: Sequence[int]) -> tuple[int, int, int, int]:
    result = tuple(int(value) for value in values)
    if result != EXPECTED_M:
        raise ValueError(
            "dual real-layer gate is pinned to ordered M=1,2,4,8; "
            f"got {result}"
        )
    return EXPECTED_M


def require_environment(environ: dict[str, str]) -> dict[str, str]:
    observed = {name: environ.get(name, "") for name in EXPECTED_ENVIRONMENT}
    if observed != EXPECTED_ENVIRONMENT:
        raise RuntimeError(
            "dual real-layer environment drifted: "
            f"expected={EXPECTED_ENVIRONMENT}, observed={observed}"
        )
    return observed


def expected_sidecar_bytes(shape: Any) -> dict[str, int]:
    """Return exact unique E8M0/K32 and FP32-global bytes for one TP rank."""

    experts = int(shape.num_experts)
    hidden = int(shape.hidden_size)
    intermediate = int(shape.intermediate_size_per_rank)
    if hidden % 32 or intermediate % 32:
        raise ValueError("dual sidecar geometry requires K dimensions divisible by 32")
    rows = {
        "w13_e8m0_k32": experts * (2 * intermediate) * (hidden // 32),
        "w2_e8m0_k32": experts * hidden * (intermediate // 32),
        "w13_global_fp32": experts * 4,
        "w2_global_fp32": experts * 4,
    }
    return rows | {"total": sum(rows.values())}


def tensor_storage_identity(source: Any, candidate: Any) -> dict[str, bool]:
    return {
        "same_data_ptr": int(source.data_ptr()) == int(candidate.data_ptr()),
        "same_storage_ptr": int(source.untyped_storage().data_ptr())
        == int(candidate.untyped_storage().data_ptr()),
        "same_storage_bytes": int(source.untyped_storage().nbytes())
        == int(candidate.untyped_storage().nbytes()),
    }


def evaluate_branch_contract(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    expected = {1: "flashinfer_cutlass", 2: "w4a16", 4: "w4a16", 8: "w4a16"}
    observed = {int(row["m"]): str(row["candidate_branch"]) for row in rows}
    per_m = {
        str(m): {
            "expected": branch,
            "observed": observed.get(m),
            "passed": observed.get(m) == branch,
        }
        for m, branch in expected.items()
    }
    return {
        "expected": {str(m): branch for m, branch in expected.items()},
        "observed": {str(m): branch for m, branch in sorted(observed.items())},
        "per_m": per_m,
        "passed": set(observed) == set(expected)
        and all(item["passed"] for item in per_m.values()),
    }


def _make_dual_experts(torch: Any, weights: Any, shape: Any, args: Any) -> tuple[Any, Any]:
    from vllm.model_executor.layers.fused_moe.experts.nvfp4_dual_decode_moe import (
        NvFp4CutlassW4A16DualExperts,
    )

    base_runner, base_proof = kernel_bench._make_flashinfer_cutlass_runner(
        torch, weights, shape, args
    )
    base = base_runner.experts
    dual = NvFp4CutlassW4A16DualExperts(
        moe_config=base.moe_config,
        quant_config=base.quant_config,
    )
    layer = SimpleNamespace(w13_weight=weights.w13, w2_weight=weights.w2)
    return dual, (layer, base_runner.activation, base_proof)


def _workspace(torch: Any, dual: Any, shape: Any, m: int, activation: Any) -> tuple[Any, Any]:
    workspace13_shape, workspace2_shape, output_shape = dual.workspace_shapes(
        M=m,
        N=shape.intermediate_size_per_rank,
        K=shape.hidden_size,
        topk=shape.top_k,
        global_num_experts=shape.num_experts,
        local_num_experts=shape.num_experts,
        expert_tokens_meta=None,
        activation=activation,
    )
    if tuple(output_shape) != (m, shape.hidden_size):
        raise RuntimeError(f"dual output shape drifted: {output_shape}")

    def allocate(shape_value: tuple[int, ...]) -> Any | None:
        count = math.prod(shape_value)
        if count <= 0:
            return None
        return torch.empty(shape_value, dtype=torch.bfloat16, device="cuda")

    return allocate(workspace13_shape), allocate(workspace2_shape)


def _make_launch(
    torch: Any,
    dual: Any,
    weights: Any,
    shape: Any,
    activation: Any,
    x: Any,
    topk_ids: Any,
    topk_weights: Any,
    *,
    uniform_decode: bool,
    dispatch_flag: list[bool],
) -> tuple[Callable[[], Any], Any]:
    dispatch_flag[0] = uniform_decode
    workspace13, workspace2 = _workspace(
        torch, dual, shape, int(x.shape[0]), activation
    )
    output = torch.empty_like(x)

    def launch() -> Any:
        dispatch_flag[0] = uniform_decode
        dual.apply(
            output=output,
            hidden_states=x,
            w1=weights.w13,
            w2=weights.w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=shape.num_experts,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )
        return output

    return launch, output


def _one_call_branch(
    torch: Any,
    launch: Callable[[], Any],
    call_counts: dict[str, int],
) -> tuple[Any, str, dict[str, int]]:
    before = dict(call_counts)
    output = launch()
    torch.cuda.synchronize()
    delta = {name: call_counts[name] - before[name] for name in call_counts}
    active = [name for name, count in delta.items() if count == 1]
    if len(active) != 1 or any(count not in (0, 1) for count in delta.values()):
        raise RuntimeError(f"dual branch trace is not exclusive: {delta}")
    return output, active[0], delta


def _install_branch_trace(dual_module: Any, cutlass_module: Any) -> tuple[Any, ...]:
    originals = (
        dual_module._run_b12x_moe_fp4,
        cutlass_module.flashinfer_cutlass_fused_moe,
    )
    counts = {"w4a16": 0, "flashinfer_cutlass": 0}

    def w4a16(*args: Any, **kwargs: Any) -> Any:
        counts["w4a16"] += 1
        return originals[0](*args, **kwargs)

    def cutlass(*args: Any, **kwargs: Any) -> Any:
        counts["flashinfer_cutlass"] += 1
        return originals[1](*args, **kwargs)

    dual_module._run_b12x_moe_fp4 = w4a16
    cutlass_module.flashinfer_cutlass_fused_moe = cutlass
    return counts, originals


def run(args: argparse.Namespace) -> int:
    import torch
    from b12x.moe.fused.w4a16 import kernel as w4a16_kernel
    from vllm.model_executor.layers.fused_moe.experts import (
        flashinfer_cutlass_moe as cutlass_module,
    )
    from vllm.model_executor.layers.fused_moe.experts import (
        nvfp4_dual_decode_moe as dual_module,
    )
    from vllm.models.deepseek_v4.nvidia.prepared_weight_loading import (
        validate_prepared_layer_file,
    )

    requested_m = require_exact_m(args.m)
    environment = require_environment(dict(os.environ))
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("dual real-layer gate requires exactly one CUDA device")
    capability = tuple(torch.cuda.get_device_capability())
    if capability != (12, 1):
        raise RuntimeError(f"dual real-layer gate requires SM121; got {capability}")
    if args.tp_rank not in (0, 1):
        raise RuntimeError("TP rank must be 0 or 1")

    physical = validate_prepared_layer_file(args.layer_file, layer=0)
    shape = kernel_bench.Dsv4Shape(tp_rank=args.tp_rank)
    shape.validate()
    tensors = prepared_bench._load_rank(torch, args.layer_file, args.tp_rank)
    weights = cutlass_bench._prepare_cutlass_weights(torch, tensors, shape)
    runner_args = SimpleNamespace(
        m=requested_m,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        swiglu_limit=10.0,
    )
    dual, (layer, activation, cutlass_proof) = _make_dual_experts(
        torch, weights, shape, runner_args
    )

    torch.cuda.synchronize()
    allocated_before_sidecar = int(torch.cuda.memory_allocated())
    dual.initialize_prepared_w4a16_decode(layer)
    torch.cuda.synchronize()
    allocated_after_sidecar = int(torch.cuda.memory_allocated())
    prepared = dual._prepared_w4a16
    if prepared is None:
        raise RuntimeError("dual post-load hook did not create W4A16 sidecar")

    weight_identity = {
        "w13": tensor_storage_identity(weights.w13, prepared.w13),
        "w2": tensor_storage_identity(weights.w2, prepared.w2),
    }
    expected_bytes = expected_sidecar_bytes(shape)
    retained = {
        "w13_e8m0_k32": prepared.w13_scale,
        "w2_e8m0_k32": prepared.w2_scale,
        "w13_global_fp32": prepared.w13_global_scale,
        "w2_global_fp32": prepared.w2_global_scale,
    }
    unique_storages: dict[int, int] = {}
    observed_sidecar: dict[str, dict[str, Any]] = {}
    for name, tensor in retained.items():
        storage = tensor.untyped_storage()
        unique_storages.setdefault(int(storage.data_ptr()), int(storage.nbytes()))
        observed_sidecar[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "storage_bytes": int(storage.nbytes()),
            "expected_storage_bytes": expected_bytes[name],
            "passed": int(storage.nbytes()) == expected_bytes[name],
        }
    observed_sidecar_bytes = sum(unique_storages.values())
    sidecar_contract = {
        "expected": expected_bytes,
        "observed": observed_sidecar,
        "observed_unique_storage_count": len(unique_storages),
        "observed_unique_storage_bytes": observed_sidecar_bytes,
        "class_reported_bytes": int(dual._w4a16_additional_scale_bytes),
        "cuda_allocated_delta_bytes": allocated_after_sidecar
        - allocated_before_sidecar,
        "duplicate_weight_bytes": 0,
        "passed": observed_sidecar_bytes == expected_bytes["total"]
        and int(dual._w4a16_additional_scale_bytes) == expected_bytes["total"]
        and len(unique_storages) == 4
        and all(row["passed"] for row in observed_sidecar.values())
        and all(all(row.values()) for row in weight_identity.values()),
    }

    dispatch_flag = [False]
    original_uniform = dual_module._is_uniform_decode_forward
    dual_module._is_uniform_decode_forward = lambda: dispatch_flag[0]
    call_counts, branch_originals = _install_branch_trace(dual_module, cutlass_module)
    compile_events, original_compile = w4a16_bench.install_compile_trace(w4a16_kernel)
    failures: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    keepalive: list[Any] = [dual, tensors, weights, prepared]
    torch.cuda.reset_peak_memory_stats()
    try:
        for m in requested_m:
            x, topk_ids, topk_weights = kernel_bench.make_routes(
                torch,
                shape,
                m,
                routing=args.routing,
                seed=args.seed + m,
                input_rms=1.0,
            )
            if x.dtype != torch.bfloat16 or x.ndim != 2:
                raise RuntimeError(f"probe input is not 2D BF16: {x.shape}/{x.dtype}")
            reference_launch, _ = _make_launch(
                torch,
                dual,
                weights,
                shape,
                activation,
                x,
                topk_ids,
                topk_weights,
                uniform_decode=False,
                dispatch_flag=dispatch_flag,
            )
            candidate_launch, _ = _make_launch(
                torch,
                dual,
                weights,
                shape,
                activation,
                x,
                topk_ids,
                topk_weights,
                uniform_decode=True,
                dispatch_flag=dispatch_flag,
            )
            reference_output, reference_branch, reference_delta = _one_call_branch(
                torch, reference_launch, call_counts
            )
            candidate_output, candidate_branch, candidate_delta = _one_call_branch(
                torch, candidate_launch, call_counts
            )
            reference_eager = reference_output.clone()
            candidate_eager = candidate_output.clone()
            activity = {
                "w4a4_reference": kernel_bench.tensor_activity(
                    torch, reference_eager
                ),
                "dual_candidate": kernel_bench.tensor_activity(
                    torch, candidate_eager
                ),
            }
            numeric = kernel_bench.compare_tensors(
                torch, candidate_eager, reference_eager
            )
            numeric_passed = kernel_bench.numeric_metrics_pass(
                numeric,
                min_cosine=args.numeric_min_cosine,
                max_normalized_rmse=args.numeric_max_nrmse,
            )
            graph_launches: dict[str, Any] = {}
            graph_status: dict[str, Any] = {}
            for name, launch, eager in (
                ("w4a4_reference", reference_launch, reference_eager),
                ("dual_candidate", candidate_launch, candidate_eager),
            ):
                replay, graph_output, graph = kernel_bench.capture_graph(torch, launch)
                replay()
                torch.cuda.synchronize()
                graph_numeric = kernel_bench.compare_tensors(
                    torch, graph_output, eager
                )
                graph_passed = kernel_bench.numeric_metrics_pass(
                    graph_numeric,
                    min_cosine=args.numeric_min_cosine,
                    max_normalized_rmse=args.numeric_max_nrmse,
                )
                graph_activity = kernel_bench.tensor_activity(torch, graph_output)
                graph_status[name] = {
                    "captured": True,
                    "vs_eager": graph_numeric,
                    "activity": graph_activity,
                    "passed": graph_passed and graph_activity["passed"],
                }
                graph_launches[name] = replay
                keepalive.extend((graph_output, graph))
            timing = prepared_bench._time_orders(
                torch,
                graph_launches,
                warmup=args.warmup,
                iters=args.iters,
                repeats=args.repeats,
                pair=("dual_candidate", "w4a4_reference"),
            )
            row = {
                "m": m,
                "input": {
                    "dtype": str(x.dtype),
                    "shape": list(x.shape),
                    "bf16_2d": x.dtype == torch.bfloat16 and x.ndim == 2,
                },
                "reference_branch": reference_branch,
                "reference_branch_delta": reference_delta,
                "candidate_branch": candidate_branch,
                "candidate_branch_delta": candidate_delta,
                "activity": activity,
                "numeric_vs_w4a4": numeric,
                "numeric_passed": numeric_passed,
                "cuda_graph_status": graph_status,
                "cuda_graph_timing": timing,
            }
            results.append(row)
            if reference_branch != "flashinfer_cutlass":
                failures.append({"kind": "reference_branch", "m": m})
            if not numeric_passed:
                failures.append({"kind": "numeric", "m": m, **numeric})
            if not all(item["passed"] for item in activity.values()):
                failures.append({"kind": "output_activity", "m": m})
            if not all(item["passed"] for item in graph_status.values()):
                failures.append({"kind": "cuda_graph", "m": m})
    finally:
        w4a16_kernel.compile_w4a16_fused_moe = original_compile
        dual_module._run_b12x_moe_fp4 = branch_originals[0]
        cutlass_module.flashinfer_cutlass_fused_moe = branch_originals[1]
        dual_module._is_uniform_decode_forward = original_uniform

    branch_contract = evaluate_branch_contract(results)
    compile_contract = w4a16_bench.evaluate_modelopt_tc_contract(
        compile_events, W4A16_M
    )
    if not sidecar_contract["passed"]:
        failures.append({"kind": "single_copy_sidecar", **sidecar_contract})
    if not branch_contract["passed"]:
        failures.append({"kind": "branch_dispatch", **branch_contract})
    if not compile_contract["passed"]:
        failures.append({"kind": "modelopt_tc_compile", **compile_contract})

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "probe": "prepared_nvfp4_dual_decode_real_layer_sm121",
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
            "m": list(requested_m),
            "routing": args.routing,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "seed": args.seed,
            "environment": environment,
            "scheduler_substitute": (
                "probe-local uniform-decode descriptor; policy remains active"
            ),
        },
        "backend_proof": {
            "implementation": (
                f"{dual.__class__.__module__}.{dual.__class__.__qualname__}"
            ),
            "cutlass": cutlass_proof,
            "weight_identity": weight_identity,
            "sidecar": sidecar_contract,
            "branch_dispatch": branch_contract,
            "modelopt_tc_compile": compile_contract,
        },
        "results": results,
        "memory": {
            "allocated_before_sidecar_bytes": allocated_before_sidecar,
            "allocated_after_sidecar_bytes": allocated_after_sidecar,
            "sidecar_allocated_delta_bytes": allocated_after_sidecar
            - allocated_before_sidecar,
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
        },
        "failures": failures,
        "ok": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "branches": branch_contract["observed"],
                "compile_contract": compile_contract["passed"],
                "sidecar_bytes": observed_sidecar_bytes,
                "single_copy": all(
                    all(row.values()) for row in weight_identity.values()
                ),
                "passed": not failures,
            },
            sort_keys=True,
        )
    )
    print(f"Wrote {args.output}")
    return 0 if not failures else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--m", type=_csv_positive_ints, default=EXPECTED_M)
    parser.add_argument(
        "--routing", choices=("balanced", "random", "hot"), default="balanced"
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--numeric-min-cosine", type=float, default=0.98)
    parser.add_argument("--numeric-max-nrmse", type=float, default=0.25)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
