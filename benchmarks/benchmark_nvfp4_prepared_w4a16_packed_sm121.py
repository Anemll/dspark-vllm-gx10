#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare prepared W4A4 with W4A16 on one real TP2 layer.

This diagnostic answers one narrow question: is the decode advantage of the
native MXFP4 serving arm caused by its once-prepared packed tensor-core layout,
rather than by BF16 activation math alone?  It loads one immutable prepared
NVFP4 layer and times both paths on identical activations and routes.  The
default comparator makes one separate packed W4A16 copy.  The opt-in
``modelopt`` comparator keeps the existing W4A4 tensors as the only physical
weight copy and requires the B12X tensor-core decode schedule to prove that it
did not silently fall back to the older direct microkernel.  It never
constructs or serves a full model.
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
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import benchmark_nvfp4_a4w4_sm121 as kernel_bench
from benchmarks import benchmark_nvfp4_prepared_b12x_sm121 as prepared_bench


SCHEMA_VERSION = 1
REFERENCE_W4A4_M4_MS = 0.772064
GAP_CLOSING_M4_MAX_MS = 0.682812
GAP_CLOSING_M4_MIN_SPEEDUP = REFERENCE_W4A4_M4_MS / GAP_CLOSING_M4_MAX_MS


def candidate_label(weight_layout: str) -> str:
    if weight_layout == 'packed':
        return 'packed_w4a16'
    if weight_layout == 'modelopt':
        return 'modelopt_tc_w4a16'
    raise ValueError(f'unsupported W4A16 weight layout {weight_layout!r}')


def require_modelopt_tc_environment(environ: dict[str, str]) -> None:
    """Fail closed unless the two B12X selectors force the intended path."""

    if environ.get('B12X_W4A16_TC_DECODE') != '1':
        raise RuntimeError('modelopt comparator requires B12X_W4A16_TC_DECODE=1')
    if environ.get('B12X_W4A16_SMALL_M_DIRECT') != '0':
        raise RuntimeError(
            'modelopt comparator requires B12X_W4A16_SMALL_M_DIRECT=0'
        )


def install_compile_trace(kernel_module: Any) -> tuple[list[dict[str, Any]], Any]:
    """Record the actual fused compile result selected by ``run_w4a16_moe``."""

    original = kernel_module.compile_w4a16_fused_moe
    events: list[dict[str, Any]] = []

    def traced_compile(**kwargs: Any) -> Any:
        result = original(**kwargs)
        events.append(
            {
                'size_m': int(result.size_m),
                'weight_layout': str(result.weight_layout),
                'direct_topk_routes': bool(result.direct_topk_routes),
                'tc_decode_fused_sum': bool(result.tc_decode_fused_sum),
                'zero_fc2_output': bool(result.zero_fc2_output),
                'element_dtype': str(result.element_dtype),
            }
        )
        return result

    kernel_module.compile_w4a16_fused_moe = traced_compile
    return events, original


def evaluate_modelopt_tc_contract(
    events: list[dict[str, Any]], requested_m: tuple[int, ...]
) -> dict[str, Any]:
    """Prove every measured M compiled the single-copy tensor-core path."""

    unique = [dict(items) for items in sorted({tuple(sorted(e.items())) for e in events})]
    required_m = sorted(set(requested_m))
    passing_m = sorted(
        {
            int(event['size_m'])
            for event in unique
            if event['weight_layout'] == 'modelopt'
            and event['direct_topk_routes'] is True
            and event['tc_decode_fused_sum'] is True
            and event['zero_fc2_output'] is False
            and event['element_dtype'] == 'bf16'
        }
    )
    return {
        'required': {
            'weight_layout': 'modelopt',
            'direct_topk_routes': True,
            'tc_decode_fused_sum': True,
            'zero_fc2_output': False,
            'element_dtype': 'bf16',
            'size_m': required_m,
        },
        'observed_unique_compile_results': unique,
        'passing_m': passing_m,
        # Frozen serving arenas deliberately precompile every supported small-M
        # TC shape.  Extra proven shapes are not a contract failure; every
        # measured shape must be present in that superset.
        'passed': set(required_m).issubset(passing_m),
    }


def _csv_positive_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(','))
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError('M values must be positive')
    return result


def direct_output_backend_proof(proof: dict[str, Any]) -> dict[str, Any]:
    """Describe the comparator's output path without rewriting serving truth."""

    result = dict(proof)
    serving_adapter_output_copy = result.pop(
        'serving_adapter_output_copy', None
    )
    if serving_adapter_output_copy is not True:
        raise ValueError('unexpected W4A4 serving output-copy contract')
    result['reference_serving_output_contract'] = {
        'adapter_full_tensor_copy_count': 1,
        'included_in_timed_launch': False,
    }
    result['timed_output_contract'] = {
        'name': 'direct_output_alias',
        'full_tensor_copy_count': 0,
        'pointer_identity_checked_each_launch': True,
    }
    return result


def evaluate_performance_gate(
    speedups: dict[int, float],
    candidate_median_ms: dict[int, float],
    *,
    required_m4_speedup: float,
    maximum_m4_latency_ms: float,
    candidate: str = 'packed_w4a16',
) -> dict[str, Any]:
    if 4 not in speedups:
        raise ValueError(f'{candidate} gate requires M=4')
    if 4 not in candidate_median_ms:
        raise ValueError(f'{candidate} latency gate requires M=4')
    if not speedups or any(not math.isfinite(value) or value <= 0 for value in speedups.values()):
        raise ValueError('speedups must be positive and finite')
    if not candidate_median_ms or any(
        not math.isfinite(value) or value <= 0
        for value in candidate_median_ms.values()
    ):
        raise ValueError(f'{candidate} latencies must be positive and finite')
    if not math.isfinite(required_m4_speedup) or required_m4_speedup <= 0:
        raise ValueError('required M=4 speedup must be positive and finite')
    if not math.isfinite(maximum_m4_latency_ms) or maximum_m4_latency_ms <= 0:
        raise ValueError('maximum M=4 latency must be positive and finite')
    speedup_passed = speedups[4] >= required_m4_speedup
    latency_passed = candidate_median_ms[4] <= maximum_m4_latency_ms
    return {
        'comparison': f'{candidate}_over_flashinfer_b12x_w4a4',
        'candidate': candidate,
        'reference_w4a4_m4_ms': REFERENCE_W4A4_M4_MS,
        'required_m4_speedup': required_m4_speedup,
        'maximum_m4_latency_ms': maximum_m4_latency_ms,
        'speedup_by_m': {str(m): value for m, value in sorted(speedups.items())},
        f'{candidate}_median_ms_by_m': {
            str(m): value for m, value in sorted(candidate_median_ms.items())
        },
        'm4_speedup': speedups[4],
        'm4_latency_ms': candidate_median_ms[4],
        'speedup_passed': speedup_passed,
        'latency_passed': latency_passed,
        'passed': speedup_passed and latency_passed,
    }


def run(args: argparse.Namespace) -> int:
    import torch
    from b12x.moe.fused.w4a16 import kernel as w4a16_kernel
    from vllm.models.deepseek_v4.nvidia.prepared_weight_loading import (
        validate_prepared_layer_file,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('packed W4A16 gate requires exactly one CUDA device')
    capability = tuple(torch.cuda.get_device_capability())
    if capability != (12, 1):
        raise RuntimeError(f'packed W4A16 gate requires SM121; got {capability}')
    if args.tp_rank not in (0, 1):
        raise ValueError('TP rank must be 0 or 1')
    candidate = candidate_label(args.w4a16_weight_layout)
    compile_events: list[dict[str, Any]] = []
    original_compile = None
    if args.w4a16_weight_layout == 'modelopt':
        require_modelopt_tc_environment(dict(os.environ))
        compile_events, original_compile = install_compile_trace(w4a16_kernel)

    physical = validate_prepared_layer_file(args.layer_file, layer=0)
    shape = kernel_bench.Dsv4Shape(tp_rank=args.tp_rank)
    shape.validate()
    tensors = prepared_bench._load_rank(torch, args.layer_file, args.tp_rank)
    weights = prepared_bench._prepare_weights(torch, tensors, shape)

    runner_args = SimpleNamespace(
        m=args.m,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        swiglu_limit=10.0,
        fast_math=True,
        w4a16_weight_layout=args.w4a16_weight_layout,
    )
    w4a4_wrapper, w4a4_proof = kernel_bench._make_w4a4_runner(
        torch, weights, shape, runner_args
    )
    w4a4_proof = direct_output_backend_proof(w4a4_proof)
    w4a4_arena = w4a4_wrapper._moe_output
    if w4a4_arena is None:
        raise RuntimeError('graph-enabled W4A4 wrapper has no output arena')
    candidate_w4a16, candidate_proof = kernel_bench._prepare_w4a16(
        torch, weights, runner_args
    )
    if getattr(candidate_w4a16, 'weight_layout', None) != args.w4a16_weight_layout:
        raise RuntimeError(
            'W4A16 comparator did not use the requested weight layout: '
            f"expected {args.w4a16_weight_layout}, got "
            f"{getattr(candidate_w4a16, 'weight_layout', None)}"
        )
    torch.cuda.synchronize()

    failures: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    speedups: dict[int, float] = {}
    candidate_median_ms: dict[int, float] = {}
    keepalive: list[Any] = [w4a4_wrapper, candidate_w4a16]
    for m in args.m:
        x, topk_ids, topk_weights = kernel_bench.make_routes(
            torch,
            shape,
            m,
            routing=args.routing,
            seed=args.seed + m,
            input_rms=1.0,
        )
        w4a4_launch, w4a4_output = prepared_bench._b12x_launch(
            torch,
            w4a4_wrapper,
            w4a4_arena,
            weights,
            x,
            topk_ids,
            topk_weights,
            direct_output=True,
        )
        w4a16_launch, w4a16_buffers = kernel_bench._make_w4a16_launch(
            torch,
            candidate_w4a16,
            x,
            topk_ids,
            topk_weights,
            runner_args,
        )

        eager_outputs: dict[str, Any] = {}
        activity: dict[str, Any] = {}
        for name, launch in (
            ('w4a4', w4a4_launch),
            (candidate, w4a16_launch),
        ):
            output = launch()
            torch.cuda.synchronize()
            eager_outputs[name] = output.clone()
            activity[name] = kernel_bench.tensor_activity(torch, output)
            if not activity[name]['passed']:
                failures.append({'kind': 'output_activity', 'm': m, 'backend': name})

        numeric = kernel_bench.compare_tensors(
            torch, eager_outputs[candidate], eager_outputs['w4a4']
        )
        numeric_passed = kernel_bench.numeric_metrics_pass(
            numeric,
            min_cosine=args.numeric_min_cosine,
            max_normalized_rmse=args.numeric_max_nrmse,
        )
        if not numeric_passed:
            failures.append({'kind': 'numeric', 'm': m, **numeric})

        graph_launches: dict[str, Any] = {}
        graph_status: dict[str, Any] = {}
        for name, launch in (
            ('w4a4', w4a4_launch),
            (candidate, w4a16_launch),
        ):
            replay, graph_output, graph = kernel_bench.capture_graph(torch, launch)
            replay()
            torch.cuda.synchronize()
            graph_numeric = kernel_bench.compare_tensors(
                torch, graph_output, eager_outputs[name]
            )
            graph_passed = kernel_bench.numeric_metrics_pass(
                graph_numeric,
                min_cosine=args.numeric_min_cosine,
                max_normalized_rmse=args.numeric_max_nrmse,
            )
            graph_launches[name] = replay
            graph_status[name] = {
                'captured': True,
                'vs_eager': graph_numeric,
                'passed': graph_passed,
            }
            keepalive.extend((graph_output, graph))
            if not graph_passed:
                failures.append({'kind': 'graph_numeric', 'm': m, 'backend': name})

        timing = prepared_bench._time_orders(
            torch,
            graph_launches,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
            pair=(candidate, 'w4a4'),
        )
        speedup = float(timing['combined'][f'speedup_{candidate}_over_w4a4'])
        speedups[m] = speedup
        candidate_median_ms[m] = float(
            timing['combined'][candidate]['median_ms']
        )
        unique_experts, counts = torch.unique(topk_ids, return_counts=True)
        results.append(
            {
                'm': m,
                'routing': args.routing,
                'routed_rows': m * shape.top_k,
                'unique_experts': int(unique_experts.numel()),
                'maximum_expert_multiplicity': int(counts.max().item()),
                'activity': activity,
                'numeric': numeric,
                'numeric_passed': numeric_passed,
                'cuda_graph_status': graph_status,
                'cuda_graph': timing,
                f'speedup_{candidate}_over_w4a4': speedup,
            }
        )

    performance_gate = evaluate_performance_gate(
        speedups,
        candidate_median_ms,
        required_m4_speedup=args.min_m4_speedup,
        maximum_m4_latency_ms=args.max_m4_latency_ms,
        candidate=candidate,
    )
    if not performance_gate['passed']:
        failures.append({'kind': 'performance', **performance_gate})

    modelopt_tc_contract = None
    if args.w4a16_weight_layout == 'modelopt':
        modelopt_tc_contract = evaluate_modelopt_tc_contract(
            compile_events, args.m
        )
        if not modelopt_tc_contract['passed']:
            failures.append(
                {'kind': 'modelopt_tc_compile_contract', **modelopt_tc_contract}
            )
    if original_compile is not None:
        w4a16_kernel.compile_w4a16_fused_moe = original_compile

    report = {
        'schema_version': SCHEMA_VERSION,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'probe': f'prepared_nvfp4_{candidate}_vs_w4a4_sm121',
        'gpu': {
            'name': torch.cuda.get_device_name(),
            'capability': list(capability),
            'torch': torch.__version__,
        },
        'checkpoint': {
            'layer_file': str(args.layer_file.resolve()),
            'physical_validation': physical,
            'tp_rank': args.tp_rank,
        },
        'settings': {
            'm': list(args.m),
            'routing': args.routing,
            'warmup': args.warmup,
            'iters': args.iters,
            'repeats': args.repeats,
            'seed': args.seed,
            'b12x_w4a16_tc_decode': os.getenv('B12X_W4A16_TC_DECODE', '0'),
            'b12x_w4a16_small_m_direct': os.getenv(
                'B12X_W4A16_SMALL_M_DIRECT', '1'
            ),
            'w4a16_weight_layout': args.w4a16_weight_layout,
        },
        'backend_proof': {
            'w4a4': w4a4_proof,
            candidate: candidate_proof,
        },
        'modelopt_tc_compile_contract': modelopt_tc_contract,
        'performance_gate': performance_gate,
        'results': results,
        'memory': {
            'allocated_gib': torch.cuda.memory_allocated() / (1 << 30),
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / (1 << 30),
            'reserved_gib': torch.cuda.memory_reserved() / (1 << 30),
        },
        'failures': failures,
        'ok': not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(performance_gate, sort_keys=True))
    print(f'Wrote {args.output}')
    return 0 if not failures else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layer-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tp-rank', type=int, default=0)
    parser.add_argument(
        '--w4a16-weight-layout',
        choices=('packed', 'modelopt'),
        default='packed',
        help=(
            'Use the existing packed-copy diagnostic or the single-copy '
            'ModelOpt tensor-core decode experiment. ModelOpt requires '
            'B12X_W4A16_TC_DECODE=1 and B12X_W4A16_SMALL_M_DIRECT=0.'
        ),
    )
    parser.add_argument('--m', type=_csv_positive_ints, default=(1, 4))
    parser.add_argument('--routing', choices=('balanced', 'random', 'hot'), default='balanced')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=4104)
    parser.add_argument('--numeric-min-cosine', type=float, default=0.98)
    parser.add_argument('--numeric-max-nrmse', type=float, default=0.25)
    parser.add_argument(
        '--min-m4-speedup',
        type=float,
        default=GAP_CLOSING_M4_MIN_SPEEDUP,
        help=(
            'Minimum packed-W4A16/W4A4 M=4 speedup required to close the '
            'measured serving gap (default: %(default).9f).'
        ),
    )
    parser.add_argument(
        '--max-m4-latency-ms',
        type=float,
        default=GAP_CLOSING_M4_MAX_MS,
        help=(
            'Absolute packed-W4A16 M=4 CUDA-graph latency ceiling required '
            'to close the measured serving gap (default: %(default).6f ms).'
        ),
    )
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
