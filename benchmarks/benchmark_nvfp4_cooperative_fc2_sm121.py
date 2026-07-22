#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Bounded real-layer gate for the opt-in SM121 cooperative-FC2 canary.

Run this entry point twice against the same prepared layer-0 file.  The
baseline run has ``FLASHINFER_B12X_COOPERATIVE_FC2_M4=0`` (or unset), while
the candidate run has it set to ``1`` and consumes the baseline JSON.  Only
the current W4A4 B12X backend is measured; the heavyweight prepared-weight
loader and launch construction are shared with the established prepared B12X
probe rather than reimplemented here.

The decision metric is CUDA-graph median latency.  The M=4 balanced route must
reach the absolute serving-derived deadline, random/hot M=4 may regress by at
most two percent from the matching baseline, and every M=1 route must remain
within two percent.  Eager timing is diagnostic.  Both eager and graph output
must be finite/nonzero and graph replay must match eager numerically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import benchmark_nvfp4_a4w4_sm121 as kernel_bench
from benchmarks import benchmark_nvfp4_prepared_b12x_sm121 as prepared_bench


SCHEMA_VERSION = 1
PROBE_NAME = "prepared_nvfp4_cooperative_fc2_sm121"
ROUTINGS = ("balanced", "random", "hot")
M_VALUES = (1, 4)
COOPERATIVE_ENV = "FLASHINFER_B12X_COOPERATIVE_FC2_M4"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_opt_in(variant: str, environ: Mapping[str, str]) -> bool:
    raw = environ.get(COOPERATIVE_ENV, "0")
    if raw not in {"0", "1"}:
        raise RuntimeError(f"{COOPERATIVE_ENV} must be exactly 0 or 1; got {raw!r}")
    enabled = raw == "1"
    expected = variant == "candidate"
    if enabled != expected:
        raise RuntimeError(
            f"{variant} run requires {COOPERATIVE_ENV}={int(expected)}, got {raw}"
        )
    return enabled


def _rows_by_key(report: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    rows = report.get("results")
    if not isinstance(rows, list):
        raise RuntimeError("report results must be a list")
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("report result rows must be objects")
        key = (str(row.get("routing")), int(row.get("m", -1)))
        if key in indexed:
            raise RuntimeError(f"duplicate result row {key!r}")
        indexed[key] = row
    expected = {(routing, m) for routing in ROUTINGS for m in M_VALUES}
    if set(indexed) != expected:
        raise RuntimeError(
            f"result matrix mismatch: got {sorted(indexed)}, expected {sorted(expected)}"
        )
    return indexed


def _graph_median(row: Mapping[str, Any]) -> float:
    graph = row.get("cuda_graph")
    if not isinstance(graph, dict):
        raise RuntimeError("result row is missing cuda_graph timing")
    value = float(graph.get("median_ms", float("nan")))
    if not (value > 0.0):
        raise RuntimeError(f"invalid CUDA-graph median {value!r}")
    return value


def _matching_fingerprint(report: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = report.get("checkpoint")
    settings = report.get("settings")
    gpu = report.get("gpu")
    if not all(isinstance(value, dict) for value in (checkpoint, settings, gpu)):
        raise RuntimeError("baseline is missing checkpoint/settings/GPU metadata")
    physical = checkpoint.get("physical_validation")
    if not isinstance(physical, dict):
        raise RuntimeError("baseline is missing physical prepared-layer validation")
    return {
        "layer_file_sha256": checkpoint.get("layer_file_sha256"),
        "reference_json_sha256": physical.get("reference_json_sha256"),
        "rank0_fingerprints": physical.get("rank0_fingerprints"),
        "tp_rank": checkpoint.get("tp_rank"),
        "capability": gpu.get("capability"),
        "torch": gpu.get("torch"),
        "settings": {
            key: settings.get(key)
            for key in ("routing", "m", "seed", "warmup", "iters", "repeats")
        },
    }


def evaluate_candidate(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    balanced_m4_max_ms: float,
    max_relative_regression: float,
) -> dict[str, Any]:
    """Apply the precommitted graph-latency gate to matching reports."""

    if baseline.get("probe") != PROBE_NAME or baseline.get("variant") != "baseline":
        raise RuntimeError("baseline JSON is not a baseline from this probe")
    if candidate.get("probe") != PROBE_NAME or candidate.get("variant") != "candidate":
        raise RuntimeError("candidate report identity is invalid")
    if not bool(baseline.get("ok")):
        raise RuntimeError("baseline report did not pass correctness gates")
    baseline_fp = _matching_fingerprint(baseline)
    candidate_fp = _matching_fingerprint(candidate)
    if candidate_fp != baseline_fp:
        raise RuntimeError(
            "candidate/baseline workload fingerprint mismatch: "
            f"candidate={candidate_fp!r}, baseline={baseline_fp!r}"
        )

    baseline_rows = _rows_by_key(baseline)
    candidate_rows = _rows_by_key(candidate)
    rows: list[dict[str, Any]] = []
    passed = True
    for routing in ROUTINGS:
        for m in M_VALUES:
            baseline_ms = _graph_median(baseline_rows[(routing, m)])
            candidate_ms = _graph_median(candidate_rows[(routing, m)])
            relative_deadline_ms = baseline_ms * (1.0 + max_relative_regression)
            if routing == "balanced" and m == 4:
                deadline_ms = balanced_m4_max_ms
                gate = "absolute_serving_projection"
            else:
                deadline_ms = relative_deadline_ms
                gate = "matched_baseline_regression"
            row_passed = candidate_ms <= deadline_ms
            passed = passed and row_passed
            rows.append(
                {
                    "routing": routing,
                    "m": m,
                    "baseline_graph_median_ms": baseline_ms,
                    "candidate_graph_median_ms": candidate_ms,
                    "candidate_over_baseline": candidate_ms / baseline_ms,
                    "deadline_ms": deadline_ms,
                    "relative_deadline_ms": relative_deadline_ms,
                    "gate": gate,
                    "passed": row_passed,
                }
            )
    return {
        "balanced_m4_max_ms": balanced_m4_max_ms,
        "max_relative_regression": max_relative_regression,
        "timing_kind": "cuda_graph",
        "rows": rows,
        "passed": passed,
    }


def _output_digest(torch: Any, output: Any) -> str:
    raw = output.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def run(args: argparse.Namespace) -> int:
    import torch
    from vllm.models.deepseek_v4.nvidia.prepared_weight_loading import (
        PREPARED_FAMILY_ORDER,
        PREPARED_NAMESPACE,
        validate_prepared_layer_file,
    )

    enabled = _strict_opt_in(args.variant, os.environ)
    if tuple(PREPARED_FAMILY_ORDER) != prepared_bench.PREPARED_FAMILY_ORDER:
        raise RuntimeError("prepared family contract drifted")
    if PREPARED_NAMESPACE != prepared_bench.PREPARED_NAMESPACE:
        raise RuntimeError("prepared namespace contract drifted")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("cooperative-FC2 gate requires exactly one CUDA device")
    capability = tuple(torch.cuda.get_device_capability())
    if capability != (12, 1):
        raise RuntimeError(f"cooperative-FC2 gate requires SM121; got {capability}")
    if args.tp_rank not in (0, 1):
        raise RuntimeError("TP rank must be 0 or 1")
    if args.variant == "candidate" and args.baseline_json is None:
        raise RuntimeError("candidate run requires --baseline-json")
    if args.variant == "baseline" and args.baseline_json is not None:
        raise RuntimeError("baseline run must not consume --baseline-json")

    physical = validate_prepared_layer_file(args.layer_file, layer=0)
    shape = kernel_bench.Dsv4Shape(tp_rank=args.tp_rank)
    shape.validate()
    load_started = time.perf_counter()
    tensors = prepared_bench._load_rank(torch, args.layer_file, args.tp_rank)
    weights = prepared_bench._prepare_weights(torch, tensors, shape)
    load_seconds = time.perf_counter() - load_started

    runner_args = SimpleNamespace(
        m=M_VALUES,
        b12x_max_num_tokens=max(M_VALUES),
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        swiglu_limit=10.0,
    )
    wrapper, backend_proof = kernel_bench._make_w4a4_runner(
        torch, weights, shape, runner_args
    )
    wrapper_arena = wrapper._moe_output
    if wrapper_arena is None:
        raise RuntimeError("graph-enabled B12X wrapper has no output arena")

    # Import after runner construction, when FlashInfer's environment-backed
    # dispatch policy has been initialized, then prove the intended branch is
    # what the runtime actually parsed.
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    runtime_enabled = bool(getattr(moe_dispatch, "_COOPERATIVE_FC2_M4", False))
    if runtime_enabled != enabled:
        raise RuntimeError(
            "dispatcher cooperative-FC2 policy does not match requested variant"
        )

    failures: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    keepalive: list[Any] = [wrapper, tensors, weights]
    for routing in ROUTINGS:
        for m in M_VALUES:
            x, topk_ids, topk_weights = kernel_bench.make_routes(
                torch,
                shape,
                m,
                routing=routing,
                seed=args.seed + 1000 * ROUTINGS.index(routing) + m,
                input_rms=1.0,
            )
            launch, output = prepared_bench._b12x_launch(
                torch,
                wrapper,
                wrapper_arena,
                weights,
                x,
                topk_ids,
                topk_weights,
                direct_output=True,
            )

            # Compile once, then poison the persistent output to prove the
            # eager launch writes every element with active finite data.
            launch()
            torch.cuda.synchronize()
            output.fill_(float("nan"))
            eager_output = launch()
            torch.cuda.synchronize()
            eager_activity = kernel_bench.tensor_activity(torch, eager_output)
            eager_reference = eager_output.clone()
            if not eager_activity["passed"]:
                failures.append(
                    {
                        "kind": "output_activity",
                        "stage": "eager",
                        "routing": routing,
                        "m": m,
                        "activity": eager_activity,
                    }
                )

            eager_timing = kernel_bench.add_derived_performance(
                kernel_bench.measure_cuda_events(
                    torch,
                    launch,
                    warmup=args.warmup,
                    iters=args.iters,
                    repeats=args.repeats,
                    flush_l2=None,
                ),
                shape,
                m,
            )
            replay, graph_output, graph = kernel_bench.capture_graph(torch, launch)
            keepalive.extend((x, topk_ids, topk_weights, output, graph_output, graph))
            graph_output.fill_(float("nan"))
            replay()
            torch.cuda.synchronize()
            graph_activity = kernel_bench.tensor_activity(torch, graph_output)
            parity = kernel_bench.compare_tensors(
                torch, graph_output, eager_reference
            )
            parity_passed = kernel_bench.numeric_metrics_pass(
                parity,
                min_cosine=args.numeric_min_cosine,
                max_normalized_rmse=args.numeric_max_nrmse,
            )
            if not graph_activity["passed"]:
                failures.append(
                    {
                        "kind": "output_activity",
                        "stage": "cuda_graph",
                        "routing": routing,
                        "m": m,
                        "activity": graph_activity,
                    }
                )
            if not parity_passed:
                failures.append(
                    {
                        "kind": "numeric",
                        "comparison": "graph_vs_eager",
                        "routing": routing,
                        "m": m,
                        "cosine": parity["cosine"],
                        "normalized_rmse": parity["normalized_rmse"],
                    }
                )
            graph_timing = kernel_bench.add_derived_performance(
                kernel_bench.measure_cuda_events(
                    torch,
                    replay,
                    warmup=args.warmup,
                    iters=args.iters,
                    repeats=args.repeats,
                    flush_l2=None,
                ),
                shape,
                m,
            )
            results.append(
                {
                    "routing": routing,
                    "m": m,
                    "routed_rows": m * shape.top_k,
                    "route_stats": {
                        "unique_experts": int(torch.unique(topk_ids).numel()),
                        "maximum_multiplicity": int(
                            torch.bincount(
                                topk_ids.flatten().to(torch.int64),
                                minlength=shape.num_experts,
                            ).max().item()
                        ),
                    },
                    "eager": eager_timing,
                    "cuda_graph": graph_timing,
                    "eager_output_activity": eager_activity,
                    "cuda_graph_output_activity": graph_activity,
                    "graph_vs_eager": parity,
                    "graph_numeric_passed": parity_passed,
                    "eager_output_sha256": _output_digest(torch, eager_reference),
                    "graph_output_sha256": _output_digest(torch, graph_output),
                }
            )
            print(
                f"{routing:8s} M={m}: eager={eager_timing['median_ms']:.6f} ms "
                f"graph={graph_timing['median_ms']:.6f} ms "
                f"cosine={float(parity['cosine']):.9f} "
                f"nrmse={float(parity['normalized_rmse']):.9f}"
            )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "probe": PROBE_NAME,
        "variant": args.variant,
        "gpu": {
            "name": torch.cuda.get_device_name(),
            "capability": list(capability),
            "torch": torch.__version__,
        },
        "checkpoint": {
            "layer_file": str(args.layer_file.resolve()),
            "layer_file_sha256": _sha256_file(args.layer_file),
            "physical_validation": physical,
            "tp_rank": args.tp_rank,
            "load_and_prepare_seconds": load_seconds,
        },
        "settings": {
            "routing": list(ROUTINGS),
            "m": list(M_VALUES),
            "seed": args.seed,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "numeric_min_cosine": args.numeric_min_cosine,
            "numeric_max_nrmse": args.numeric_max_nrmse,
            "cuda_graph_required": True,
            "l2_flush_mib": 0,
        },
        "backend_proof": {
            **backend_proof,
            "cooperative_fc2_environment": COOPERATIVE_ENV,
            "cooperative_fc2_enabled": runtime_enabled,
            "direct_output_alias": True,
        },
        "results": results,
        "failures": failures,
    }
    report["ok"] = not failures

    if args.variant == "candidate":
        assert args.baseline_json is not None
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        if not isinstance(baseline, dict):
            raise RuntimeError("baseline JSON must be an object")
        gate = evaluate_candidate(
            report,
            baseline,
            balanced_m4_max_ms=args.balanced_m4_max_ms,
            max_relative_regression=args.max_relative_regression,
        )
        report["baseline_json"] = {
            "path": str(args.baseline_json.resolve()),
            "sha256": _sha256_file(args.baseline_json),
        }
        report["performance_gate"] = gate
        if not gate["passed"]:
            failures.append({"kind": "performance", **gate})
    else:
        report["performance_gate"] = {
            "applicable": False,
            "reason": "baseline measurement",
            "passed": True,
        }
    report["ok"] = not failures

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["performance_gate"], sort_keys=True))
    print(f"Wrote {args.output}")
    return 0 if report["ok"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--numeric-min-cosine", type=float, default=0.98)
    parser.add_argument("--numeric-max-nrmse", type=float, default=0.25)
    parser.add_argument("--balanced-m4-max-ms", type=float, default=0.682812)
    parser.add_argument("--max-relative-regression", type=float, default=0.02)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
