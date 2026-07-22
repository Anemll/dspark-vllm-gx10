#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Real-layer gate for FlashInfer CUTLASS fused NVFP4 input quantization.

The current vLLM NVFP4 path quantizes BF16 activations with
``moe_kernel_quantize_input`` before calling ``FlashInferExperts.apply``.
FlashInfer's CUTLASS ``expandInputRowsKernel`` can instead accept the original
BF16 rows with ``input_sf=None`` and quantize while it expands/routes them.

This bounded one-layer probe compares those two paths on identical prepared
weights, activations, and routes at M=1 and M=4.  It records eager and CUDA-
graph latency in milliseconds, plus graph component timings for the external
quantizer and the pre-quantized CUTLASS call.  Direct interception of the
``flashinfer_cutlass_fused_moe`` Python boundary proves the input dtype/shape,
scale presence, weight identity, and lack of a Python fallback before timing.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import benchmark_nvfp4_a4w4_sm121 as kernel_bench
from benchmarks import benchmark_nvfp4_prepared_b12x_sm121 as prepared_bench


SCHEMA_VERSION = 1
EXPECTED_M = (1, 4)
CURRENT_PATH = "external_prequant_nvfp4"
FUSED_PATH = "bf16_expand_input_rows_quant"


def _csv_positive_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("M values must be positive")
    return result


def _require_exact_m(values: Sequence[int]) -> tuple[int, int]:
    result = tuple(values)
    if result != EXPECTED_M:
        raise ValueError(
            "fused-input-quant gate is pinned to the ordered M set 1,4; "
            f"got {result}"
        )
    return (1, 4)


def _prepare_cutlass_weights(
    torch: Any,
    tensors: dict[str, Any],
    shape: kernel_bench.Dsv4Shape,
) -> kernel_bench.PreparedWeights:
    """Build only the single-copy CUTLASS view of a prepared layer."""

    ones = torch.ones(shape.num_experts, dtype=torch.float32, device="cuda")
    return kernel_bench.PreparedWeights(
        w13=tensors["w13.weight"],
        w13_sf_modelopt=tensors["w13.weight_scale"],
        w13_sf_swizzled=None,
        w13_sf_mma=None,
        w2=tensors["w2.weight"],
        w2_sf_modelopt=tensors["w2.weight_scale"],
        w2_sf_swizzled=None,
        w2_sf_mma=None,
        alpha1=ones,
        alpha2=ones.clone(),
        fc2_input_scale=ones.clone(),
        cutlass_a1_gscale=tensors["a1_gscale"],
        cutlass_a2_gscale=tensors["a2_gscale"],
        cutlass_g1_alphas=tensors["g1_alphas"],
        cutlass_g2_alphas=tensors["g2_alphas"],
        metadata={
            "source": "prepared-physical-layer0",
            "source_weight_data_ptrs": {
                "w13": int(tensors["w13.weight"].data_ptr()),
                "w2": int(tensors["w2.weight"].data_ptr()),
            },
            "weight_preparation_contract": {
                "flashinfer_b12x": False,
                "flashinfer_cutlass": True,
            },
            "checkpoint_input_scale_tensor_count": 3 * shape.num_experts,
            "modelopt_activation_scale_contract": {
                "loaded_from_prepared_checkpoint": True,
                "prepared_payload_is_authoritative": True,
            },
        },
    )


def _apply_launch(
    torch: Any,
    runner: kernel_bench.FlashInferCutlassRunner,
    weights: kernel_bench.PreparedWeights,
    shape: kernel_bench.Dsv4Shape,
    *,
    hidden_states: Any,
    input_sf: Any | None,
    topk_ids: Any,
    topk_weights: Any,
) -> tuple[Callable[[], Any], Any]:
    output = torch.empty(
        (topk_ids.shape[0], shape.hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
    )

    def launch() -> Any:
        runner.experts.apply(
            output=output,
            hidden_states=hidden_states,
            w1=weights.w13,
            w2=weights.w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=runner.activation,
            global_num_experts=shape.num_experts,
            expert_map=None,
            a1q_scale=input_sf,
            a2_scale=None,
            workspace13=None,
            workspace2=None,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )
        return output

    return launch, output


def _make_launches(
    torch: Any,
    runner: kernel_bench.FlashInferCutlassRunner,
    weights: kernel_bench.PreparedWeights,
    shape: kernel_bench.Dsv4Shape,
    x: Any,
    topk_ids: Any,
    topk_weights: Any,
) -> dict[str, Any]:
    from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

    quant_config = runner.experts.quant_config

    def quantize() -> tuple[Any, Any]:
        return moe_kernel_quantize_input(
            x,
            quant_config.a1_gscale,
            quant_dtype=quant_config.quant_dtype,
            per_act_token_quant=quant_config.per_act_token_quant,
            block_shape=quant_config.block_shape,
            is_scale_swizzled=quant_config.is_scale_swizzled,
            mx_alignment=quant_config.mx_alignment,
        )

    prequant_x, prequant_sf = quantize()
    torch.cuda.synchronize()
    prequant_kernel, prequant_output = _apply_launch(
        torch,
        runner,
        weights,
        shape,
        hidden_states=prequant_x,
        input_sf=prequant_sf,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )
    fused_internal, fused_output = _apply_launch(
        torch,
        runner,
        weights,
        shape,
        hidden_states=x,
        input_sf=None,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )
    current_output = torch.empty_like(x)

    def current_total() -> Any:
        current_x, current_sf = quantize()
        runner.experts.apply(
            output=current_output,
            hidden_states=current_x,
            w1=weights.w13,
            w2=weights.w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=runner.activation,
            global_num_experts=shape.num_experts,
            expert_map=None,
            a1q_scale=current_sf,
            a2_scale=None,
            workspace13=None,
            workspace2=None,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )
        return current_output

    def external_quant_only() -> Any:
        return quantize()[0]

    return {
        "current_total": current_total,
        "current_output": current_output,
        "prequant_kernel": prequant_kernel,
        "prequant_output": prequant_output,
        "fused_internal": fused_internal,
        "fused_output": fused_output,
        "external_quant_only": external_quant_only,
        "prequant_x": prequant_x,
        "prequant_sf": prequant_sf,
    }


def _trace_one_flashinfer_call(
    torch: Any,
    launch: Callable[[], Any],
    *,
    label: str,
) -> tuple[Any, dict[str, Any], dict[str, str | None]]:
    """Intercept one exact Python-to-FlashInfer boundary invocation."""

    import vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe as module

    original = module.flashinfer_cutlass_fused_moe
    records: list[dict[str, Any]] = []

    def traced(*args: Any, **kwargs: Any) -> Any:
        if args:
            raise RuntimeError("FlashInfer path unexpectedly used positional arguments")
        value = kwargs["input"]
        scale = kwargs.get("input_sf")
        w1 = kwargs["fc1_expert_weights"]
        w2 = kwargs["fc2_expert_weights"]
        output = kwargs["output"]
        records.append(
            {
                "label": label,
                "input_dtype": str(value.dtype),
                "input_shape": list(value.shape),
                "input_sf_present": scale is not None,
                "input_sf_dtype": None if scale is None else str(scale.dtype),
                "input_sf_shape": None if scale is None else list(scale.shape),
                "w1_data_ptr": int(w1.untyped_storage().data_ptr()),
                "w2_data_ptr": int(w2.untyped_storage().data_ptr()),
                "output_data_ptr": int(output.data_ptr()),
                "quant_scale_count": len(kwargs.get("quant_scales") or []),
                "tp_size": int(kwargs["tp_size"]),
                "tp_rank": int(kwargs["tp_rank"]),
                "ep_size": int(kwargs["ep_size"]),
                "ep_rank": int(kwargs["ep_rank"]),
                "use_deepseek_fp8_block_scale": bool(
                    kwargs["use_deepseek_fp8_block_scale"]
                ),
                "use_mxfp8_act_scaling": bool(kwargs["use_mxfp8_act_scaling"]),
                "use_w4_group_scaling": bool(kwargs["use_w4_group_scaling"]),
            }
        )
        return original(**kwargs)

    module.flashinfer_cutlass_fused_moe = traced
    try:
        output = launch()
        torch.cuda.synchronize()
        output_copy = output.clone()
        torch.cuda.synchronize()
    finally:
        module.flashinfer_cutlass_fused_moe = original
    if len(records) != 1:
        raise RuntimeError(
            f"{label} did not invoke exactly one FlashInfer CUTLASS backend: "
            f"calls={len(records)}"
        )
    symbol = {
        "module": getattr(original, "__module__", None),
        "qualname": getattr(original, "__qualname__", None),
        "source": inspect.getsourcefile(original),
    }
    return output_copy, records[0], symbol


def validate_path_contract(
    traces: Sequence[dict[str, Any]],
    *,
    m: int,
    hidden_size: int,
    tp_rank: int,
    expected_w1_data_ptr: int,
    expected_w2_data_ptr: int,
) -> dict[str, Any]:
    """Pure-Python fail-closed contract for the two intercepted calls."""

    by_label = {record.get("label"): record for record in traces}
    if len(traces) != 2 or set(by_label) != {CURRENT_PATH, FUSED_PATH}:
        raise RuntimeError(
            "path trace must contain exactly one current and one fused call"
        )
    current = by_label[CURRENT_PATH]
    fused = by_label[FUSED_PATH]
    expected_common = {
        "w1_data_ptr": expected_w1_data_ptr,
        "w2_data_ptr": expected_w2_data_ptr,
        "quant_scale_count": 6,
        "tp_size": 2,
        "tp_rank": tp_rank,
        "ep_size": 1,
        "ep_rank": 0,
        "use_deepseek_fp8_block_scale": False,
        "use_mxfp8_act_scaling": False,
        "use_w4_group_scaling": False,
    }
    for label, record in by_label.items():
        for field, expected in expected_common.items():
            if record.get(field) != expected:
                raise RuntimeError(
                    f"{label} FlashInfer contract drifted at {field}: "
                    f"expected={expected!r}, actual={record.get(field)!r}"
                )
    if current.get("input_dtype") != "torch.uint8":
        raise RuntimeError("current path did not pass packed NVFP4 activations")
    if current.get("input_shape") != [m, hidden_size // 2]:
        raise RuntimeError("current packed activation shape drifted")
    if current.get("input_sf_present") is not True:
        raise RuntimeError("current path omitted its external activation scales")
    if fused.get("input_dtype") != "torch.bfloat16":
        raise RuntimeError("fused path did not pass BF16 activations")
    if fused.get("input_shape") != [m, hidden_size]:
        raise RuntimeError("fused BF16 activation shape drifted")
    if fused.get("input_sf_present") is not False:
        raise RuntimeError("fused path unexpectedly passed pre-quantized scales")
    return {
        "passed": True,
        "direct_flashinfer_call_count": 2,
        "python_fallback_observed": False,
        "current_external_nvfp4_quantization": True,
        "candidate_bf16_input": True,
        "candidate_input_sf_none": True,
        "candidate_internal_expand_quantization": True,
        "same_checkpoint_weight_storage": True,
    }


def _numeric_gate(
    torch: Any,
    actual: Any,
    reference: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], bool]:
    result = kernel_bench.compare_tensors(torch, actual, reference)
    passed = kernel_bench.numeric_metrics_pass(
        result,
        min_cosine=args.numeric_min_cosine,
        max_normalized_rmse=args.numeric_max_nrmse,
    )
    return result, passed


def _measure_component_graph(
    torch: Any,
    launch: Callable[[], Any],
    args: argparse.Namespace,
) -> tuple[Callable[[], Any], Any, Any, dict[str, Any]]:
    replay, output, graph = kernel_bench.capture_graph(torch, launch)
    timing = kernel_bench.measure_cuda_events(
        torch,
        replay,
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
        flush_l2=None,
    )
    return replay, output, graph, timing


def run(args: argparse.Namespace) -> int:
    import torch
    from vllm.models.deepseek_v4.nvidia.prepared_weight_loading import (
        PREPARED_FAMILY_ORDER as RUNTIME_PREPARED_FAMILY_ORDER,
        PREPARED_NAMESPACE as RUNTIME_PREPARED_NAMESPACE,
        validate_prepared_layer_file,
    )

    m_values = _require_exact_m(args.m)
    if tuple(prepared_bench.PREPARED_FAMILY_ORDER) != tuple(
        RUNTIME_PREPARED_FAMILY_ORDER
    ) or prepared_bench.PREPARED_NAMESPACE != RUNTIME_PREPARED_NAMESPACE:
        raise RuntimeError("prepared loader namespace/family contract drifted")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("fused-input-quant gate requires exactly one CUDA device")
    capability = tuple(torch.cuda.get_device_capability())
    if capability != (12, 1):
        raise RuntimeError(f"fused-input-quant gate requires SM121; got {capability}")
    if args.tp_rank not in (0, 1):
        raise ValueError("TP rank must be 0 or 1")

    physical = validate_prepared_layer_file(args.layer_file, layer=0)
    shape = kernel_bench.Dsv4Shape(tp_rank=args.tp_rank)
    shape.validate()
    load_started = time.perf_counter()
    tensors = prepared_bench._load_rank(torch, args.layer_file, args.tp_rank)
    weights = _prepare_cutlass_weights(torch, tensors, shape)
    load_seconds = time.perf_counter() - load_started
    runner_args = SimpleNamespace(
        m=m_values,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        swiglu_limit=10.0,
    )
    runner, backend_proof = kernel_bench._make_flashinfer_cutlass_runner(
        torch, weights, shape, runner_args
    )
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
        FlashInferExperts,
        is_valid_flashinfer_cutlass_fused_moe,
    )

    if type(runner.experts) is not FlashInferExperts:
        raise RuntimeError("benchmark did not instantiate the exact FlashInferExperts class")
    # The production modular wiring currently pre-quantizes NVFP4 because this
    # property is false.  This benchmark deliberately calls apply directly to
    # test the already-supported internal kernel branch before integration.
    serving_expects_unquantized = bool(runner.experts.expects_unquantized_inputs)
    if serving_expects_unquantized:
        raise RuntimeError(
            "serving wiring already defers NVFP4 quantization; benchmark premise drifted"
        )

    failures: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    keepalive: list[Any] = [runner, tensors, weights]
    speedups: list[float] = []
    backend_symbols: list[dict[str, str | None]] = []
    for m in m_values:
        x, topk_ids, topk_weights = kernel_bench.make_routes(
            torch,
            shape,
            m,
            routing="balanced",
            seed=args.seed + m,
            input_rms=1.0,
        )
        if not is_valid_flashinfer_cutlass_fused_moe(x, weights.w13, weights.w2):
            raise RuntimeError("FlashInfer rejected the exact BF16/uint8 candidate dtypes")
        launches = _make_launches(
            torch, runner, weights, shape, x, topk_ids, topk_weights
        )
        current_eager, current_trace, current_symbol = _trace_one_flashinfer_call(
            torch, launches["current_total"], label=CURRENT_PATH
        )
        fused_eager, fused_trace, fused_symbol = _trace_one_flashinfer_call(
            torch, launches["fused_internal"], label=FUSED_PATH
        )
        if current_symbol != fused_symbol:
            raise RuntimeError("the two paths resolved different FlashInfer symbols")
        backend_symbols.append(current_symbol)
        path_contract = validate_path_contract(
            (current_trace, fused_trace),
            m=m,
            hidden_size=shape.hidden_size,
            tp_rank=args.tp_rank,
            expected_w1_data_ptr=int(weights.w13.untyped_storage().data_ptr()),
            expected_w2_data_ptr=int(weights.w2.untyped_storage().data_ptr()),
        )
        eager_numeric, eager_numeric_passed = _numeric_gate(
            torch, fused_eager, current_eager, args
        )
        prequant_kernel_eager = launches["prequant_kernel"]().clone()
        torch.cuda.synchronize()
        component_numeric, component_numeric_passed = _numeric_gate(
            torch, prequant_kernel_eager, current_eager, args
        )
        activity = {
            CURRENT_PATH: kernel_bench.tensor_activity(torch, current_eager),
            FUSED_PATH: kernel_bench.tensor_activity(torch, fused_eager),
            "prequant_cutlass_component": kernel_bench.tensor_activity(
                torch, prequant_kernel_eager
            ),
        }
        if not eager_numeric_passed:
            failures.append({"kind": "numeric", "stage": "eager", "m": m})
        if not component_numeric_passed:
            failures.append(
                {"kind": "numeric", "stage": "prequant_component", "m": m}
            )
        for path, proof in activity.items():
            if not proof["passed"]:
                failures.append(
                    {"kind": "output_activity", "stage": "eager", "m": m, "path": path}
                )

        eager_timing = prepared_bench._time_orders(
            torch,
            {
                CURRENT_PATH: launches["current_total"],
                FUSED_PATH: launches["fused_internal"],
            },
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
            pair=(FUSED_PATH, CURRENT_PATH),
        )
        current_graph, current_graph_output, current_graph_obj, current_graph_timing = (
            _measure_component_graph(torch, launches["current_total"], args)
        )
        fused_graph, fused_graph_output, fused_graph_obj, fused_graph_timing = (
            _measure_component_graph(torch, launches["fused_internal"], args)
        )
        quant_graph, quant_graph_output, quant_graph_obj, quant_graph_timing = (
            _measure_component_graph(torch, launches["external_quant_only"], args)
        )
        kernel_graph, kernel_graph_output, kernel_graph_obj, kernel_graph_timing = (
            _measure_component_graph(torch, launches["prequant_kernel"], args)
        )
        graph_pair_timing = prepared_bench._time_orders(
            torch,
            {CURRENT_PATH: current_graph, FUSED_PATH: fused_graph},
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
            pair=(FUSED_PATH, CURRENT_PATH),
        )
        current_graph()
        fused_graph()
        torch.cuda.synchronize()
        graph_numeric, graph_numeric_passed = _numeric_gate(
            torch, fused_graph_output, current_graph_output, args
        )
        current_graph_vs_eager, current_graph_vs_eager_passed = _numeric_gate(
            torch, current_graph_output, current_eager, args
        )
        fused_graph_vs_eager, fused_graph_vs_eager_passed = _numeric_gate(
            torch, fused_graph_output, fused_eager, args
        )
        if not graph_numeric_passed:
            failures.append({"kind": "numeric", "stage": "cuda_graph", "m": m})
        if not current_graph_vs_eager_passed:
            failures.append(
                {"kind": "graph_vs_eager", "path": CURRENT_PATH, "m": m}
            )
        if not fused_graph_vs_eager_passed:
            failures.append({"kind": "graph_vs_eager", "path": FUSED_PATH, "m": m})
        graph_activity = {
            CURRENT_PATH: kernel_bench.tensor_activity(torch, current_graph_output),
            FUSED_PATH: kernel_bench.tensor_activity(torch, fused_graph_output),
        }
        for path, proof in graph_activity.items():
            if not proof["passed"]:
                failures.append(
                    {
                        "kind": "output_activity",
                        "stage": "cuda_graph",
                        "m": m,
                        "path": path,
                    }
                )

        # Use the alternating-order pair for the decision metric.  The
        # individually measured timings remain below as component evidence.
        current_ms = float(
            graph_pair_timing["combined"][CURRENT_PATH]["median_ms"]
        )
        fused_ms = float(
            graph_pair_timing["combined"][FUSED_PATH]["median_ms"]
        )
        quant_ms = float(quant_graph_timing["median_ms"])
        kernel_ms = float(kernel_graph_timing["median_ms"])
        for field, value in {
            "current_graph_ms": current_ms,
            "fused_graph_ms": fused_ms,
            "external_quant_graph_ms": quant_ms,
            "prequant_kernel_graph_ms": kernel_ms,
        }.items():
            if not math.isfinite(value) or value <= 0:
                failures.append(
                    {"kind": "timing", "m": m, "field": field, "value": value}
                )
        speedup = current_ms / fused_ms
        speedups.append(speedup)
        results.append(
            {
                "m": m,
                "routed_rows": m * shape.top_k,
                "path_contract": path_contract,
                "path_trace": {
                    CURRENT_PATH: current_trace,
                    FUSED_PATH: fused_trace,
                },
                "correctness": {
                    "eager_fused_vs_current": eager_numeric,
                    "eager_passed": eager_numeric_passed,
                    "prequant_component_vs_current": component_numeric,
                    "prequant_component_passed": component_numeric_passed,
                    "cuda_graph_fused_vs_current": graph_numeric,
                    "cuda_graph_passed": graph_numeric_passed,
                    "current_graph_vs_eager": current_graph_vs_eager,
                    "current_graph_vs_eager_passed": current_graph_vs_eager_passed,
                    "fused_graph_vs_eager": fused_graph_vs_eager,
                    "fused_graph_vs_eager_passed": fused_graph_vs_eager_passed,
                    "eager_activity": activity,
                    "cuda_graph_activity": graph_activity,
                },
                "timing_ms": {
                    "eager_ordered_pair": eager_timing,
                    "cuda_graph_ordered_pair": graph_pair_timing,
                    "current_total_graph": current_graph_timing,
                    "fused_internal_graph": fused_graph_timing,
                    "external_quant_only_graph": quant_graph_timing,
                    "prequant_cutlass_only_graph": kernel_graph_timing,
                    "current_total_median_ms": current_ms,
                    "fused_internal_median_ms": fused_ms,
                    "current_total_individual_median_ms": float(
                        current_graph_timing["median_ms"]
                    ),
                    "fused_internal_individual_median_ms": float(
                        fused_graph_timing["median_ms"]
                    ),
                    "external_quant_only_median_ms": quant_ms,
                    "prequant_cutlass_only_median_ms": kernel_ms,
                    "current_over_fused_speedup": speedup,
                    "external_quant_fraction_of_current": quant_ms / current_ms,
                    "component_sum_median_ms": quant_ms + kernel_ms,
                    "component_sum_minus_current_ms": quant_ms + kernel_ms - current_ms,
                },
            }
        )
        keepalive.extend(
            (
                x,
                topk_ids,
                topk_weights,
                launches,
                current_graph_output,
                current_graph_obj,
                fused_graph_output,
                fused_graph_obj,
                quant_graph_output,
                quant_graph_obj,
                kernel_graph_output,
                kernel_graph_obj,
            )
        )

    if not all(symbol == backend_symbols[0] for symbol in backend_symbols):
        raise RuntimeError("M=1 and M=4 resolved different FlashInfer symbols")
    geomean_speedup = math.exp(
        sum(math.log(value) for value in speedups) / len(speedups)
    )
    performance_passed = bool(
        geomean_speedup >= args.minimum_geomean_speedup
        and min(speedups) >= args.minimum_per_shape_speedup
    )
    performance_gate = {
        "scope": "decode M=1/M=4 real prepared layer",
        "minimum_geomean_speedup": args.minimum_geomean_speedup,
        "minimum_per_shape_speedup": args.minimum_per_shape_speedup,
        "current_over_fused_geomean_speedup": geomean_speedup,
        "current_over_fused_speedup_by_m": {
            str(m): value for m, value in zip(m_values, speedups, strict=True)
        },
        "passed": performance_passed,
    }
    if not performance_passed:
        failures.append({"kind": "performance", **performance_gate})
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "probe": "nvfp4_cutlass_fused_input_quant_sm121",
        "gpu": {
            "name": torch.cuda.get_device_name(),
            "capability": list(capability),
            "torch": torch.__version__,
        },
        "checkpoint": {
            "layer_file": str(args.layer_file.resolve()),
            "physical_validation": physical,
            "tp_rank": args.tp_rank,
            "load_seconds": load_seconds,
        },
        "settings": {
            "m": list(m_values),
            "routing": "balanced",
            "seed": args.seed,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "numeric_min_cosine": args.numeric_min_cosine,
            "numeric_max_nrmse": args.numeric_max_nrmse,
        },
        "backend_proof": {
            **backend_proof,
            "exact_experts_class": (
                f"{runner.experts.__class__.__module__}."
                f"{runner.experts.__class__.__qualname__}"
            ),
            "flashinfer_boundary_symbol": backend_symbols[0],
            "all_shapes_resolved_same_symbol": all(
                symbol == backend_symbols[0] for symbol in backend_symbols
            ),
            "serving_expects_unquantized_inputs_before_integration": (
                serving_expects_unquantized
            ),
            "candidate_calls_apply_directly": True,
            "fallback_allowed": False,
        },
        "performance_gate": performance_gate,
        "results": results,
        "memory": {
            "allocated_gib": torch.cuda.memory_allocated() / (1 << 30),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1 << 30),
            "reserved_gib": torch.cuda.memory_reserved() / (1 << 30),
        },
        "failures": failures,
        "ok": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(performance_gate, sort_keys=True))
    for row in results:
        timing = row["timing_ms"]
        print(
            f"M={row['m']} current={timing['current_total_median_ms']:.6f} ms "
            f"fused={timing['fused_internal_median_ms']:.6f} ms "
            f"external_quant={timing['external_quant_only_median_ms']:.6f} ms "
            f"speedup={timing['current_over_fused_speedup']:.6f}x"
        )
    print(f"Wrote {args.output}")
    _ = keepalive
    return 0 if not failures else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--m", type=_csv_positive_ints, default=EXPECTED_M)
    parser.add_argument("--seed", type=int, default=4108)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--numeric-min-cosine", type=float, default=0.999)
    parser.add_argument("--numeric-max-nrmse", type=float, default=0.02)
    parser.add_argument("--minimum-geomean-speedup", type=float, default=1.0)
    parser.add_argument("--minimum-per-shape-speedup", type=float, default=0.98)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
