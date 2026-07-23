#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Sweep FlashInfer CUTLASS NVFP4 M=4 tactics without changing its cache.

This is a single-layer diagnostic for the prepared TP=2 DeepSeek-V4-Flash
checkpoint.  It opens the exact serving autotune cache read-only, proves the
cached M=4 tactic pair, then overrides ``AutoTuner.choose_one`` only inside
this process to measure a small reviewed tactic matrix.  The cache file is
never copied, rewritten, or used as an output path.

The serving path defaults to PDL on SM121.  Both PDL states are measured
because PDL is not part of FlashInfer's on-disk autotune key.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_nvfp4_a4w4_sm121 as kernel_bench


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
GEMM1_OP = "trtllm::fused_moe::gemm1"
GEMM2_OP = "trtllm::fused_moe::gemm2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_int_csv(text: str) -> tuple[int, ...]:
    try:
        values = tuple(dict.fromkeys(int(part) for part in text.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("tactics must be non-negative")
    return values


def _pdl_values(selection: str) -> tuple[bool, ...]:
    return {
        "true": (True,),
        "false": (False,),
        "both": (True, False),
    }[selection]


def inspect_cache(path: Path, *, m: int) -> dict[str, Any]:
    """Return only the tactic entries whose first input is the requested M."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    metadata = raw.get("_metadata")
    entries: dict[str, list[dict[str, Any]]] = {GEMM1_OP: [], GEMM2_OP: []}
    malformed: list[str] = []
    for key_text, value in raw.items():
        if key_text == "_metadata":
            continue
        try:
            key = ast.literal_eval(key_text)
            op, runner_name, profile, extras = key
            first_shape = tuple(profile[0])
            if op not in entries or not first_shape or int(first_shape[0]) != m:
                continue
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError("cache value is not [runner,tactic]")
            entries[op].append(
                {
                    "runner": str(value[0]),
                    "tactic": int(value[1]),
                    "first_shape": list(first_shape),
                    "extras": list(extras),
                    "key_sha256": hashlib.sha256(key_text.encode()).hexdigest(),
                }
            )
        except (IndexError, TypeError, ValueError, SyntaxError):
            malformed.append(hashlib.sha256(key_text.encode()).hexdigest())
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "metadata": metadata,
        "total_config_entries": len(raw) - int("_metadata" in raw),
        "m": m,
        "entries": entries,
        "malformed_key_sha256": malformed,
    }


def cache_tactics(cache: dict[str, Any], op: str) -> tuple[int, ...]:
    return tuple(
        sorted({int(entry["tactic"]) for entry in cache["entries"].get(op, [])})
    )


def build_matrix(
    gemm1_tactics: Sequence[int],
    gemm2_tactics: Sequence[int],
    pdl_values: Sequence[bool],
    *,
    service_pair: tuple[int, int],
    service_pdl: bool,
) -> tuple[tuple[int, int, bool], ...]:
    candidates = [
        (int(gemm1), int(gemm2), bool(pdl))
        for pdl in pdl_values
        for gemm1 in gemm1_tactics
        for gemm2 in gemm2_tactics
    ]
    service = (service_pair[0], service_pair[1], service_pdl)
    if service not in candidates:
        candidates.insert(0, service)
    else:
        candidates.remove(service)
        candidates.insert(0, service)
    return tuple(candidates)


def collect_tactic_inventory(native: Any) -> dict[str, Any]:
    """Enumerate every native tactic and its device occupancy.

    The native module exposes GEMM2 ids in the combined profile namespace, so
    those ids begin at ``gemm1_tactic_count``.  Zero-occupancy profiles cannot
    launch on the current device and are excluded from an exhaustive sweep.
    """

    gemm1_count = int(native.get_gemm1_tactic_count())
    gemm2_count = int(native.get_gemm2_tactic_count())
    rows: dict[str, list[dict[str, int]]] = {GEMM1_OP: [], GEMM2_OP: []}
    for tactic in range(gemm1_count):
        rows[GEMM1_OP].append(
            {
                "tactic": tactic,
                "occupancy": int(native.get_tactic_occupancy(tactic)),
            }
        )
    for tactic in range(gemm1_count, gemm1_count + gemm2_count):
        rows[GEMM2_OP].append(
            {
                "tactic": tactic,
                "occupancy": int(native.get_tactic_occupancy(tactic)),
            }
        )
    return {
        "gemm1_tactic_count": gemm1_count,
        "gemm2_tactic_count": gemm2_count,
        "profiles": rows,
    }


def occupancy_valid_tactics(inventory: dict[str, Any], op: str) -> tuple[int, ...]:
    return tuple(
        int(row["tactic"])
        for row in inventory["profiles"][op]
        if int(row["occupancy"]) > 0
    )


def unsupported_tile_phase(error: BaseException) -> str | None:
    """Classify only the native launch-time unsupported-tile rejection.

    Occupancy is necessary but not sufficient in the pinned FlashInfer module:
    several profiles report positive occupancy before the typed MoE dispatcher
    rejects their tile.  Numerical failures and every other runtime error stay
    fatal.
    """

    message = str(error)
    native_unsupported = (
        "Unsupported tile shape config" in message
        or "Failed to initialize cutlass TMA WS grouped gemm" in message
    )
    if not native_unsupported:
        return None
    if "::gemm1(" in message:
        return GEMM1_OP
    if "::gemm2(" in message:
        return GEMM2_OP
    return None


def _load_prepared_cutlass_weights(
    torch: Any,
    layer_file: Path,
    tp_rank: int,
) -> tuple[kernel_bench.PreparedWeights, dict[str, Any]]:
    from safetensors import safe_open
    from vllm.models.deepseek_v4.nvidia.prepared_weight_loading import (
        PREPARED_FAMILY_ORDER as RUNTIME_FAMILY_ORDER,
        PREPARED_NAMESPACE as RUNTIME_NAMESPACE,
        validate_prepared_layer_file,
    )

    if RUNTIME_NAMESPACE != PREPARED_NAMESPACE:
        raise RuntimeError("prepared namespace drifted from the serving loader")
    if tuple(RUNTIME_FAMILY_ORDER) != PREPARED_FAMILY_ORDER:
        raise RuntimeError("prepared family order drifted from the serving loader")
    physical = validate_prepared_layer_file(layer_file, layer=0)
    prefix = f"{PREPARED_NAMESPACE}.layers.0.experts."
    tensors: dict[str, Any] = {}
    with safe_open(str(layer_file), framework="pt", device="cpu") as handle:
        for family in PREPARED_FAMILY_ORDER:
            source = handle.get_tensor(f"{prefix}{family}")[tp_rank]
            if not source.is_contiguous():
                raise RuntimeError(f"prepared rank slice is not contiguous: {family}")
            tensors[family] = source.to("cuda:0")
    torch.cuda.synchronize()

    metadata = {
        "source": "prepared-physical-layer0-cutlass-only",
        "layer_file": str(layer_file),
        "layer_file_sha256": _sha256(layer_file),
        "physical_validation": physical,
        "source_weight_data_ptrs": {
            "w13": int(tensors["w13.weight"].data_ptr()),
            "w2": int(tensors["w2.weight"].data_ptr()),
        },
        "weight_preparation_contract": {
            "flashinfer_b12x": False,
            "flashinfer_cutlass": True,
        },
        "checkpoint_input_scale_tensor_count": 768,
        "modelopt_activation_scale_contract": {
            "loaded_from_prepared_checkpoint": True,
            "runtime_transforms": 0,
        },
    }
    weights = kernel_bench.PreparedWeights(
        w13=tensors["w13.weight"],
        w13_sf_modelopt=tensors["w13.weight_scale"],
        w13_sf_swizzled=None,
        w13_sf_mma=None,
        w2=tensors["w2.weight"],
        w2_sf_modelopt=tensors["w2.weight_scale"],
        w2_sf_swizzled=None,
        w2_sf_mma=None,
        alpha1=None,
        alpha2=None,
        fc2_input_scale=None,
        cutlass_a1_gscale=tensors["a1_gscale"],
        cutlass_a2_gscale=tensors["a2_gscale"],
        cutlass_g1_alphas=tensors["g1_alphas"],
        cutlass_g2_alphas=tensors["g2_alphas"],
        metadata=metadata,
    )
    return weights, metadata


@contextlib.contextmanager
def force_tactics_and_pdl(
    *,
    gemm1_tactic: int,
    gemm2_tactic: int,
    enable_pdl: bool,
    inventory_out: list[dict[str, Any]] | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """Override tactic selection in-process; never mutate an autotune cache."""

    from flashinfer.autotuner import AutoTuner
    from flashinfer.fused_moe import cutlass_fused_moe as real_cutlass_fused_moe
    import vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe as fi_moe

    original_choose_one = AutoTuner.choose_one
    original_cutlass_fused_moe = fi_moe.flashinfer_cutlass_fused_moe
    trace: list[dict[str, Any]] = []

    def forced_choose_one(
        self: Any,
        custom_op: str,
        runners: list[Any],
        tuning_config: Any,
        inputs: list[Any],
        **kwargs: Any,
    ) -> tuple[Any, int]:
        if custom_op not in (GEMM1_OP, GEMM2_OP):
            return original_choose_one(
                self, custom_op, runners, tuning_config, inputs, **kwargs
            )
        if len(runners) != 1:
            raise RuntimeError(f"forced {custom_op} expected exactly one runner")
        runner = runners[0]
        native = runner.fused_moe_runner
        if inventory_out is not None and not inventory_out:
            inventory_out.append(collect_tactic_inventory(native))
        gemm1_count = int(native.get_gemm1_tactic_count())
        gemm2_count = int(native.get_gemm2_tactic_count())
        tactic = gemm1_tactic if custom_op == GEMM1_OP else gemm2_tactic
        low, high = (
            (0, gemm1_count)
            if custom_op == GEMM1_OP
            else (gemm1_count, gemm1_count + gemm2_count)
        )
        if tactic < low or tactic >= high:
            raise RuntimeError(
                f"forced tactic {tactic} is outside {custom_op} range [{low},{high})"
            )
        occupancy = None
        if hasattr(native, "get_tactic_occupancy"):
            occupancy = int(native.get_tactic_occupancy(tactic))
            if occupancy <= 0:
                raise RuntimeError(
                    f"forced tactic {tactic} has zero occupancy for {custom_op}"
                )
        trace.append(
            {
                "op": custom_op,
                "tactic": tactic,
                "gemm1_tactic_count": gemm1_count,
                "gemm2_tactic_count": gemm2_count,
                "occupancy": occupancy,
            }
        )
        return runner, tactic

    def forced_cutlass_fused_moe(*args: Any, **kwargs: Any) -> Any:
        explicit = kwargs.get("enable_pdl")
        if explicit is not None and bool(explicit) != enable_pdl:
            raise RuntimeError(
                f"conflicting enable_pdl={explicit}; forced value is {enable_pdl}"
            )
        kwargs["enable_pdl"] = enable_pdl
        return real_cutlass_fused_moe(*args, **kwargs)

    AutoTuner.choose_one = forced_choose_one
    fi_moe.flashinfer_cutlass_fused_moe = forced_cutlass_fused_moe
    try:
        yield trace
    finally:
        fi_moe.flashinfer_cutlass_fused_moe = original_cutlass_fused_moe
        AutoTuner.choose_one = original_choose_one


def _time_candidate(
    torch: Any,
    *,
    runner: Any,
    weights: kernel_bench.PreparedWeights,
    shape: kernel_bench.Dsv4Shape,
    x: Any,
    topk_ids: Any,
    topk_weights: Any,
    gemm1_tactic: int,
    gemm2_tactic: int,
    enable_pdl: bool,
    reference_output: Any | None,
    warmup: int,
    iters: int,
    repeats: int,
    numeric_min_cosine: float,
    numeric_max_nrmse: float,
    inventory_out: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], Any]:
    with force_tactics_and_pdl(
        gemm1_tactic=gemm1_tactic,
        gemm2_tactic=gemm2_tactic,
        enable_pdl=enable_pdl,
        inventory_out=inventory_out,
    ) as trace:
        launch, _ = kernel_bench._make_flashinfer_cutlass_launch(
            torch, runner, weights, shape, x, topk_ids, topk_weights
        )
        eager_output = launch()
        torch.cuda.synchronize()
        eager_snapshot = eager_output.clone()
        activity = kernel_bench.tensor_activity(torch, eager_snapshot)
        if not activity["passed"]:
            raise RuntimeError("forced tactic produced an inactive eager output")
        reference = eager_snapshot if reference_output is None else reference_output
        eager_numeric = kernel_bench.compare_tensors(torch, eager_snapshot, reference)
        eager_numeric_passed = kernel_bench.numeric_metrics_pass(
            eager_numeric,
            min_cosine=numeric_min_cosine,
            max_normalized_rmse=numeric_max_nrmse,
        )
        if not eager_numeric_passed:
            raise RuntimeError(
                "forced tactic failed eager numerical parity: "
                f"cosine={eager_numeric['cosine']}, "
                f"nrmse={eager_numeric['normalized_rmse']}"
            )

        replay, graph_output, graph = kernel_bench.capture_graph(torch, launch)
        graph_output.fill_(math.nan)
        replay()
        torch.cuda.synchronize()
        graph_activity = kernel_bench.tensor_activity(torch, graph_output)
        graph_numeric = kernel_bench.compare_tensors(
            torch, graph_output, eager_snapshot
        )
        graph_numeric_passed = kernel_bench.numeric_metrics_pass(
            graph_numeric,
            min_cosine=numeric_min_cosine,
            max_normalized_rmse=numeric_max_nrmse,
        )
        if not graph_activity["passed"] or not graph_numeric_passed:
            raise RuntimeError("forced tactic failed CUDA-graph output/parity gate")
        timing = kernel_bench.measure_cuda_events(
            torch,
            replay,
            warmup=warmup,
            iters=iters,
            repeats=repeats,
            flush_l2=None,
        )
        result = {
            "gemm1_tactic": gemm1_tactic,
            "gemm2_tactic": gemm2_tactic,
            "enable_pdl": enable_pdl,
            "trace": trace[:2],
            "activity": activity,
            "eager_vs_service_reference": eager_numeric,
            "eager_numeric_passed": eager_numeric_passed,
            "graph_activity": graph_activity,
            "graph_vs_eager": graph_numeric,
            "graph_numeric_passed": graph_numeric_passed,
            "cuda_graph": timing,
        }
        # Keep only the tiny immutable output reference.  Destroy each graph
        # before the next candidate so eight captures cannot accumulate arenas.
        del graph, replay, graph_output, launch
        torch.cuda.empty_cache()
        return result, eager_snapshot


def run(args: argparse.Namespace) -> int:
    import torch
    from flashinfer.utils import device_support_pdl

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("tactic sweep requires exactly one CUDA device")
    capability = tuple(torch.cuda.get_device_capability())
    if capability != (12, 1):
        raise RuntimeError(f"tactic sweep requires SM121; got {capability}")
    if args.tp_rank not in (0, 1):
        raise RuntimeError("TP rank must be 0 or 1")

    cache = inspect_cache(args.autotune_cache, m=4)
    cached_gemm1 = cache_tactics(cache, GEMM1_OP)
    cached_gemm2 = cache_tactics(cache, GEMM2_OP)
    service_pair = (args.service_gemm1_tactic, args.service_gemm2_tactic)
    if service_pair[0] not in cached_gemm1 or service_pair[1] not in cached_gemm2:
        raise RuntimeError(
            "service cache does not prove the expected M=4 tactic pair: "
            f"expected={service_pair}, cached_gemm1={cached_gemm1}, "
            f"cached_gemm2={cached_gemm2}"
        )
    service_pdl = bool(device_support_pdl(torch.device("cuda:0")))
    if not service_pdl:
        raise RuntimeError("SM121 serving-default PDL unexpectedly resolved false")
    matrix = build_matrix(
        args.gemm1_tactics,
        args.gemm2_tactics,
        _pdl_values(args.pdl),
        service_pair=service_pair,
        service_pdl=service_pdl,
    )

    shape = kernel_bench.Dsv4Shape(tp_rank=args.tp_rank)
    shape.validate()
    weights, prepared_proof = _load_prepared_cutlass_weights(
        torch, args.layer_file, args.tp_rank
    )
    runner_args = SimpleNamespace(
        m=(4,), swiglu_alpha=1.0, swiglu_beta=0.0, swiglu_limit=10.0
    )
    runner, backend_proof = kernel_bench._make_flashinfer_cutlass_runner(
        torch, weights, shape, runner_args
    )
    x, topk_ids, topk_weights = kernel_bench.make_routes(
        torch,
        shape,
        4,
        routing=args.routing,
        seed=args.seed + 4,
        input_rms=1.0,
    )
    route_source: dict[str, Any] = {
        "kind": "synthetic",
        "routing": args.routing,
        "seed": args.seed + 4,
    }
    if args.route_ids_npy is not None:
        topk_ids = kernel_bench.load_captured_route_ids(
            torch,
            args.route_ids_npy,
            sample_index=args.route_sample_index,
            m=4,
            top_k=shape.top_k,
        )
        route_source = {
            "kind": "captured",
            "path": str(args.route_ids_npy),
            "sha256": _sha256(args.route_ids_npy),
            "sample_index": args.route_sample_index,
            "ids": topk_ids.cpu().tolist(),
        }

    results: list[dict[str, Any]] = []
    service_reference = None
    tactic_inventory: list[dict[str, Any]] = []
    if args.all_tactics:
        # One authoritative service launch discovers the native profile ranges
        # and occupancy.  Reuse its output as the numerical reference, then
        # measure the full Cartesian product of launchable profiles.
        service_result, service_reference = _time_candidate(
            torch,
            runner=runner,
            weights=weights,
            shape=shape,
            x=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            gemm1_tactic=service_pair[0],
            gemm2_tactic=service_pair[1],
            enable_pdl=service_pdl,
            reference_output=None,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
            numeric_min_cosine=args.numeric_min_cosine,
            numeric_max_nrmse=args.numeric_max_nrmse,
            inventory_out=tactic_inventory,
        )
        if len(tactic_inventory) != 1:
            raise RuntimeError("native CUTLASS tactic inventory was not captured once")
        gemm1_tactics = occupancy_valid_tactics(tactic_inventory[0], GEMM1_OP)
        gemm2_tactics = occupancy_valid_tactics(tactic_inventory[0], GEMM2_OP)
        matrix = build_matrix(
            gemm1_tactics,
            gemm2_tactics,
            _pdl_values(args.pdl),
            service_pair=service_pair,
            service_pdl=service_pdl,
        )
        results.append(service_result)
        matrix = matrix[1:]
    unsupported_tactics: dict[str, set[int]] = {GEMM1_OP: set(), GEMM2_OP: set()}
    skipped_profiles: list[dict[str, Any]] = []
    for gemm1_tactic, gemm2_tactic, enable_pdl in matrix:
        if (
            gemm1_tactic in unsupported_tactics[GEMM1_OP]
            or gemm2_tactic in unsupported_tactics[GEMM2_OP]
        ):
            continue
        print(
            f"M=4 gemm1={gemm1_tactic} gemm2={gemm2_tactic} "
            f"pdl={str(enable_pdl).lower()}",
            flush=True,
        )
        try:
            result, eager_snapshot = _time_candidate(
                torch,
                runner=runner,
                weights=weights,
                shape=shape,
                x=x,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                gemm1_tactic=gemm1_tactic,
                gemm2_tactic=gemm2_tactic,
                enable_pdl=enable_pdl,
                reference_output=service_reference,
                warmup=args.warmup,
                iters=args.iters,
                repeats=args.repeats,
                numeric_min_cosine=args.numeric_min_cosine,
                numeric_max_nrmse=args.numeric_max_nrmse,
            )
        except Exception as error:
            phase = unsupported_tile_phase(error)
            if not args.all_tactics or phase is None:
                raise
            tactic = gemm1_tactic if phase == GEMM1_OP else gemm2_tactic
            unsupported_tactics[phase].add(tactic)
            skipped_profiles.append(
                {
                    "op": phase,
                    "tactic": tactic,
                    "reason": "native_dispatch_unsupported_tile_shape",
                }
            )
            print(f"  skipped unsupported {phase} tactic={tactic}", flush=True)
            continue
        if service_reference is None:
            service_reference = eager_snapshot
        median_ms = float(result["cuda_graph"]["median_ms"])
        print(f"  graph median={median_ms * 1000:.3f} us", flush=True)
        results.append(result)

    service = results[0]
    service_ms = float(service["cuda_graph"]["median_ms"])
    best = min(results, key=lambda item: float(item["cuda_graph"]["median_ms"]))
    best_ms = float(best["cuda_graph"]["median_ms"])
    speedup = service_ms / best_ms
    material = speedup >= args.minimum_material_speedup
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "probe": "prepared_nvfp4_cutlass_tactic_pdl_sweep_sm121",
        "shape": {
            "m": 4,
            "hidden_size": shape.hidden_size,
            "intermediate_size_per_rank": shape.intermediate_size_per_rank,
            "num_experts": shape.num_experts,
            "top_k": shape.top_k,
            "tp_rank": shape.tp_rank,
            "tp_size": shape.tp_size,
        },
        "settings": {
            "routing": args.routing,
            "route_source": route_source,
            "seed": args.seed,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "gemm1_tactics": list(args.gemm1_tactics),
            "gemm2_tactics": list(args.gemm2_tactics),
            "all_tactics": args.all_tactics,
            "pdl_values": list(_pdl_values(args.pdl)),
            "service_pair": list(service_pair),
            "service_pdl": service_pdl,
            "minimum_material_speedup": args.minimum_material_speedup,
        },
        "cache_proof": cache,
        "tactic_inventory": tactic_inventory[0] if tactic_inventory else None,
        "unsupported_profiles": skipped_profiles,
        "prepared_checkpoint_proof": prepared_proof,
        "backend_proof": backend_proof,
        "results": results,
        "decision": {
            "service_median_ms": service_ms,
            "best_median_ms": best_ms,
            "best": {
                "gemm1_tactic": best["gemm1_tactic"],
                "gemm2_tactic": best["gemm2_tactic"],
                "enable_pdl": best["enable_pdl"],
            },
            "speedup_over_service": speedup,
            "material_improvement": material,
            "service_materially_optimal": not material,
        },
        "memory": {
            "allocated_gib": torch.cuda.memory_allocated() / (1 << 30),
            "reserved_gib": torch.cuda.memory_reserved() / (1 << 30),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1 << 30),
        },
        "ok": all(
            result["eager_numeric_passed"] and result["graph_numeric_passed"]
            for result in results
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["decision"], sort_keys=True))
    print(f"Wrote {args.output}")
    return 0 if report["ok"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-file", type=Path, required=True)
    parser.add_argument("--autotune-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--routing", choices=("balanced", "random", "hot"), default="balanced")
    parser.add_argument(
        "--route-ids-npy",
        type=Path,
        help=(
            "Replace synthetic route IDs with one sample from an NPY array whose "
            "last dimensions are [4, top_k]. Hidden states and route weights remain "
            "the deterministic matched fixture."
        ),
    )
    parser.add_argument("--route-sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--gemm1-tactics", type=_positive_int_csv, default=(16, 18))
    parser.add_argument("--gemm2-tactics", type=_positive_int_csv, default=(58, 59))
    parser.add_argument(
        "--all-tactics",
        action="store_true",
        help="enumerate every occupancy-valid native GEMM1/GEMM2 pair",
    )
    parser.add_argument("--pdl", choices=("true", "false", "both"), default="both")
    parser.add_argument("--service-gemm1-tactic", type=int, default=16)
    parser.add_argument("--service-gemm2-tactic", type=int, default=58)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--numeric-min-cosine", type=float, default=0.98)
    parser.add_argument("--numeric-max-nrmse", type=float, default=0.25)
    parser.add_argument("--minimum-material-speedup", type=float, default=1.03)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.layer_file.is_file():
        raise FileNotFoundError(f"prepared layer does not exist: {args.layer_file}")
    if not args.autotune_cache.is_file():
        raise FileNotFoundError(f"autotune cache does not exist: {args.autotune_cache}")
    if args.route_ids_npy is not None and not args.route_ids_npy.is_file():
        raise FileNotFoundError(f"captured route file does not exist: {args.route_ids_npy}")
    if args.route_sample_index < 0:
        raise ValueError("route sample index must be non-negative")
    if args.warmup < 0 or args.iters <= 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative; iters/repeats must be positive")
    if args.minimum_material_speedup < 1.0:
        raise ValueError("minimum material speedup must be at least 1.0")
    if not 0.0 <= args.numeric_min_cosine <= 1.0:
        raise ValueError("numeric minimum cosine must be within [0,1]")
    if args.numeric_max_nrmse < 0:
        raise ValueError("numeric maximum NRMSE must be non-negative")


def main() -> int:
    args = build_parser().parse_args()
    try:
        validate_args(args)
        return run(args)
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
