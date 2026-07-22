#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare prepared W4A4 with packed W4A16 on one real TP2 layer.

This diagnostic answers one narrow question: is the decode advantage of the
native MXFP4 serving arm caused by its once-prepared packed tensor-core layout,
rather than by BF16 activation math alone?  It loads one immutable prepared
NVFP4 layer, keeps the existing FlashInfer-B12X W4A4 tensors untouched, makes
one separate packed W4A16 copy, and times both paths on identical activations
and routes.  It never constructs or serves a full model.
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
    packed_w4a16_median_ms: dict[int, float],
    *,
    required_m4_speedup: float,
    maximum_m4_latency_ms: float,
) -> dict[str, Any]:
    if 4 not in speedups:
        raise ValueError('packed W4A16 gate requires M=4')
    if 4 not in packed_w4a16_median_ms:
        raise ValueError('packed W4A16 latency gate requires M=4')
    if not speedups or any(not math.isfinite(value) or value <= 0 for value in speedups.values()):
        raise ValueError('speedups must be positive and finite')
    if not packed_w4a16_median_ms or any(
        not math.isfinite(value) or value <= 0
        for value in packed_w4a16_median_ms.values()
    ):
        raise ValueError('packed W4A16 latencies must be positive and finite')
    if not math.isfinite(required_m4_speedup) or required_m4_speedup <= 0:
        raise ValueError('required M=4 speedup must be positive and finite')
    if not math.isfinite(maximum_m4_latency_ms) or maximum_m4_latency_ms <= 0:
        raise ValueError('maximum M=4 latency must be positive and finite')
    speedup_passed = speedups[4] >= required_m4_speedup
    latency_passed = packed_w4a16_median_ms[4] <= maximum_m4_latency_ms
    return {
        'comparison': 'packed_w4a16_over_flashinfer_b12x_w4a4',
        'reference_w4a4_m4_ms': REFERENCE_W4A4_M4_MS,
        'required_m4_speedup': required_m4_speedup,
        'maximum_m4_latency_ms': maximum_m4_latency_ms,
        'speedup_by_m': {str(m): value for m, value in sorted(speedups.items())},
        'packed_w4a16_median_ms_by_m': {
            str(m): value for m, value in sorted(packed_w4a16_median_ms.items())
        },
        'm4_speedup': speedups[4],
        'm4_latency_ms': packed_w4a16_median_ms[4],
        'speedup_passed': speedup_passed,
        'latency_passed': latency_passed,
        'passed': speedup_passed and latency_passed,
    }


def run(args: argparse.Namespace) -> int:
    import torch
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
        w4a16_weight_layout='packed',
    )
    w4a4_wrapper, w4a4_proof = kernel_bench._make_w4a4_runner(
        torch, weights, shape, runner_args
    )
    w4a4_proof = direct_output_backend_proof(w4a4_proof)
    w4a4_arena = w4a4_wrapper._moe_output
    if w4a4_arena is None:
        raise RuntimeError('graph-enabled W4A4 wrapper has no output arena')
    packed_w4a16, packed_proof = kernel_bench._prepare_w4a16(
        torch, weights, runner_args
    )
    if getattr(packed_w4a16, 'weight_layout', None) != 'packed':
        raise RuntimeError('W4A16 comparator did not use packed weights')
    torch.cuda.synchronize()

    failures: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    speedups: dict[int, float] = {}
    packed_w4a16_median_ms: dict[int, float] = {}
    keepalive: list[Any] = [w4a4_wrapper, packed_w4a16]
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
            packed_w4a16,
            x,
            topk_ids,
            topk_weights,
            runner_args,
        )

        eager_outputs: dict[str, Any] = {}
        activity: dict[str, Any] = {}
        for name, launch in (
            ('w4a4', w4a4_launch),
            ('packed_w4a16', w4a16_launch),
        ):
            output = launch()
            torch.cuda.synchronize()
            eager_outputs[name] = output.clone()
            activity[name] = kernel_bench.tensor_activity(torch, output)
            if not activity[name]['passed']:
                failures.append({'kind': 'output_activity', 'm': m, 'backend': name})

        numeric = kernel_bench.compare_tensors(
            torch, eager_outputs['packed_w4a16'], eager_outputs['w4a4']
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
            ('packed_w4a16', w4a16_launch),
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
            pair=('packed_w4a16', 'w4a4'),
        )
        speedup = float(
            timing['combined']['speedup_packed_w4a16_over_w4a4']
        )
        speedups[m] = speedup
        packed_w4a16_median_ms[m] = float(
            timing['combined']['packed_w4a16']['median_ms']
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
                'speedup_packed_w4a16_over_w4a4': speedup,
            }
        )

    performance_gate = evaluate_performance_gate(
        speedups,
        packed_w4a16_median_ms,
        required_m4_speedup=args.min_m4_speedup,
        maximum_m4_latency_ms=args.max_m4_latency_ms,
    )
    if not performance_gate['passed']:
        failures.append({'kind': 'performance', **performance_gate})

    report = {
        'schema_version': SCHEMA_VERSION,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'probe': 'prepared_nvfp4_packed_w4a16_vs_w4a4_sm121',
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
        },
        'backend_proof': {
            'w4a4': w4a4_proof,
            'packed_w4a16': packed_proof,
        },
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
