#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Gate the exact orphaned FlashInfer direct NVFP4 kernel on DSv4 M=4.

The ported module must be the byte-exact output of
``patch_flashinfer_orphan_direct_micro_dsv4.py`` applied to FlashInfer's
otherwise-unwired source.  The benchmark loads one immutable prepared layer,
compiles only that literal E4M3/K16 W4A4 kernel, and compares it with the
accepted FlashInfer CUTLASS path using identical weights, inputs, routes, and
ModelOpt scale algebra.

Pass requires eager and CUDA-graph numerical agreement plus a graph median at
or below 0.682812 ms, the measured M=4 optimization target.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import benchmark_nvfp4_a4w4_sm121 as kernel_bench
from benchmarks import benchmark_nvfp4_prepared_b12x_sm121 as prepared_bench
from benchmarks.probe_nvfp4_direct_micro_sm121 import direct_geometry
from scripts.patch_flashinfer_orphan_direct_micro_dsv4 import (
    SOURCE_SHA256,
    port_source,
)


SCHEMA_VERSION = 1
M_VALUE = 4
ROUTING = "balanced"
SWIGLU_LIMIT = 10.0
W13_LAYOUT = "w13"
DIRECT_BLOCK_DIM = 512
PORTED_SOURCE_SHA256 = (
    "ce223868f247c1abb097df2e59bf0a0ac8087924e290921e11faf9fa04e6754e"
)
DEFAULT_MAXIMUM_GRAPH_MS = 0.682812


@dataclass
class DirectRunner:
    kernel: Any
    compiled: Any
    intermediate: Any
    barrier_count: Any
    barrier_epoch: Any
    output: Any
    grid_x: int
    launch: Callable[[], Any]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_pair(source: Path, ported: Path) -> dict[str, Any]:
    source_bytes = source.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    if source_sha != SOURCE_SHA256:
        raise RuntimeError(
            f"orphan source drifted: expected {SOURCE_SHA256}, got {source_sha}"
        )
    ported_bytes = ported.read_bytes()
    ported_sha = hashlib.sha256(ported_bytes).hexdigest()
    if ported_sha != PORTED_SOURCE_SHA256:
        raise RuntimeError(
            f"ported source drifted: expected {PORTED_SOURCE_SHA256}, got {ported_sha}"
        )
    expected = port_source(source_bytes.decode("utf-8")).encode("utf-8")
    if ported_bytes != expected:
        raise RuntimeError("ported source is not the exact hash-pinned transform")
    text = ported_bytes.decode("utf-8")
    forbidden = ("fp4_dot8_dual_sum", "prefetch_global_l2", "e8m0_k32")
    leaked = [marker for marker in forbidden if marker in text]
    if leaked:
        raise RuntimeError(f"descendant-only implementation leaked in: {leaked}")
    return {
        "source": str(source.resolve()),
        "source_sha256": source_sha,
        "source_bytes": len(source_bytes),
        "ported": str(ported.resolve()),
        "ported_sha256": ported_sha,
        "ported_bytes": len(ported_bytes),
        "byte_exact_transform": True,
        "forbidden_descendant_markers_absent": list(forbidden),
    }


def load_ported_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "flashinfer_orphan_direct_micro_dsv4", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import ported source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def kernel_kwargs() -> dict[str, Any]:
    return {
        "sf_vec_size": 16,
        "mma_tiler_mn": (64, 128),
        "output_tile_count_n": 1,
        "fast_math": True,
        "activation": "silu",
        "share_input_across_experts": False,
        "share_expert_scales": False,
        "single_token": False,
        "dynamic_down_scale": False,
        "w4a16_mode": False,
        "swiglu_limit": SWIGLU_LIMIT,
        "w13_layout": W13_LAYOUT,
    }


def validate_prepared_tensors(
    torch: Any, tensors: dict[str, Any], shape: Any
) -> dict[str, Any]:
    expected = {
        "w13.weight": (
            torch.uint8,
            (
                shape.num_experts,
                2 * shape.intermediate_size_per_rank,
                shape.hidden_size // 2,
            ),
        ),
        "w2.weight": (
            torch.uint8,
            (
                shape.num_experts,
                shape.hidden_size,
                shape.intermediate_size_per_rank // 2,
            ),
        ),
        "w13.weight_scale": (
            torch.float8_e4m3fn,
            (
                shape.num_experts,
                2 * shape.intermediate_size_per_rank,
                shape.hidden_size // 16,
            ),
        ),
        "w2.weight_scale": (
            torch.float8_e4m3fn,
            (
                shape.num_experts,
                shape.hidden_size,
                shape.intermediate_size_per_rank // 16,
            ),
        ),
        "a1_gscale": (torch.float32, (shape.num_experts,)),
        "a2_gscale": (torch.float32, (shape.num_experts,)),
        "g1_alphas": (torch.float32, (shape.num_experts,)),
        "g2_alphas": (torch.float32, (shape.num_experts,)),
    }
    observed: dict[str, Any] = {}
    for name, (dtype, expected_shape) in expected.items():
        tensor = tensors[name]
        if tensor.dtype != dtype or tuple(tensor.shape) != expected_shape:
            raise RuntimeError(
                f"prepared tensor drifted for {name}: {tensor.dtype}/"
                f"{tuple(tensor.shape)} != {dtype}/{expected_shape}"
            )
        if tensor.device.type != "cuda" or not tensor.is_contiguous():
            raise RuntimeError(f"prepared tensor is not contiguous CUDA: {name}")
        observed[name] = {
            "dtype": str(dtype),
            "shape": list(expected_shape),
        }
    return {
        "tensors": observed,
        "prepared_reads": 8,
        "runtime_scale_transforms": 0,
        "block_scale_format": "E4M3 K16",
        "w13_order": "up/w3 then gate/w1",
        "a1_gscale": "1 / w13_input_scale",
        "g1_alphas": "w13_weight_scale_2 * w13_input_scale",
        "a2_gscale": "1 / w2_input_scale",
        "g2_alphas": "w2_weight_scale_2 * w2_input_scale",
    }


def compile_runner(
    torch: Any,
    kernel_class: Any,
    tensors: dict[str, Any],
    shape: Any,
    x: Any,
    topk_ids: Any,
    topk_weights: Any,
) -> DirectRunner:
    import cutlass
    import cutlass.cute as cute
    import b12x.integration.tp_moe as tp_moe
    from b12x.cute.compiler import KernelCompileSpec, compile as b12x_compile
    from cutlass.cutlass_dsl import Int32
    from flashinfer.cute_dsl.utils import current_cuda_stream, make_ptr

    if not kernel_class.is_supported(
        M_VALUE,
        shape.hidden_size,
        shape.intermediate_size_per_rank,
        shape.top_k,
        shape.num_experts,
    ):
        raise RuntimeError("literal orphan rejected the exact DSv4 M=4 shape")
    kernel = kernel_class(**kernel_kwargs())
    kernel.configure(
        M_VALUE,
        shape.hidden_size,
        shape.intermediate_size_per_rank,
        shape.top_k,
        shape.num_experts,
        device=x.device,
    )
    if (
        kernel.activation != "silu"
        or not kernel.has_swiglu_limit
        or float(kernel.swiglu_limit) != SWIGLU_LIMIT
        or kernel.w13_layout != W13_LAYOUT
        or kernel.w13_gate_first
        or kernel.w4a16_mode
    ):
        raise RuntimeError("ported literal-kernel semantics drifted")

    def dummy(dtype: Any) -> Any:
        return make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16)

    launch_ids = topk_ids.to(torch.int32).contiguous()
    barrier_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    cache_key = (
        "flashinfer_literal_orphan_dsv4_m4",
        kernel.__cache_key__,
        str(launch_ids.dtype),
        PORTED_SOURCE_SHA256,
    )
    compiled = b12x_compile(
        kernel,
        dummy(cutlass.BFloat16),
        dummy(cutlass.Uint8),
        dummy(cutlass.Uint8),
        dummy(cutlass.Float32),
        dummy(cutlass.Float32),
        dummy(cutlass.Float32),
        dummy(cutlass.Uint32),
        dummy(cutlass.Uint8),
        dummy(cutlass.Uint8),
        dummy(cutlass.Float32),
        dummy(cutlass.Int32),
        dummy(cutlass.Float32),
        dummy(cutlass.BFloat16),
        barrier_fake,
        barrier_fake,
        Int32(M_VALUE),
        Int32(1),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "benchmarks.flashinfer_literal_orphan_direct", 1, cache_key
        ),
    )
    if not tp_moe._compiled_direct_micro_accepts_block_dim(
        compiled, DIRECT_BLOCK_DIM
    ):
        raise RuntimeError("compiled literal kernel cannot launch 512 threads")

    geometry = direct_geometry(
        M_VALUE,
        shape.hidden_size,
        shape.intermediate_size_per_rank,
        shape.top_k,
    )
    cfg = kernel._cfg
    if (
        int(kernel.grid_x) <= 0
        or int(cfg.fc1_chunks) != geometry.fc1_chunks
        or int(cfg.fc2_n_chunks) != geometry.fc2_n_chunks
        or M_VALUE * int(cfg.inter_u32) != geometry.intermediate_u32
    ):
        raise RuntimeError("literal direct geometry drifted")
    intermediate = torch.empty(
        geometry.intermediate_u32, dtype=torch.uint32, device=x.device
    )
    barrier_count = torch.zeros(
        geometry.barrier_slots_per_array, dtype=torch.int32, device=x.device
    )
    barrier_epoch = torch.zeros_like(barrier_count)
    output = torch.empty_like(x)

    def launch() -> Any:
        kernel_class.launch(
            compiled,
            x=x,
            w1_fp4=tensors["w13.weight"],
            w1_blockscale=tensors["w13.weight_scale"],
            w1_alphas=tensors["g1_alphas"],
            a1_gscale=tensors["a1_gscale"],
            a2_gscale=tensors["a2_gscale"],
            inter_fp32=intermediate,
            w2_fp4=tensors["w2.weight"],
            w2_blockscale=tensors["w2.weight_scale"],
            w2_alphas=tensors["g2_alphas"],
            topk_ids=launch_ids,
            topk_weights=topk_weights,
            out=output,
            barrier_count=barrier_count,
            barrier_epoch=barrier_epoch,
            m=M_VALUE,
            grid_x=int(kernel.grid_x),
        )
        return output

    return DirectRunner(
        kernel=kernel,
        compiled=compiled,
        intermediate=intermediate,
        barrier_count=barrier_count,
        barrier_epoch=barrier_epoch,
        output=output,
        grid_x=int(kernel.grid_x),
        launch=launch,
    )


def numeric_gate(
    torch: Any, actual: Any, reference: Any, args: argparse.Namespace
) -> tuple[dict[str, Any], bool]:
    metrics = kernel_bench.compare_tensors(torch, actual, reference)
    passed = kernel_bench.numeric_metrics_pass(
        metrics,
        min_cosine=args.numeric_min_cosine,
        max_normalized_rmse=args.numeric_max_nrmse,
    )
    return metrics, passed


def validate_args(args: argparse.Namespace) -> None:
    for name in ("warmup", "iters", "repeats"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0.0 <= args.numeric_min_cosine <= 1.0:
        raise ValueError("numeric-min-cosine must be in [0,1]")
    if args.numeric_max_nrmse < 0.0 or not math.isfinite(args.numeric_max_nrmse):
        raise ValueError("numeric-max-nrmse must be finite and non-negative")
    if args.maximum_graph_ms <= 0.0 or not math.isfinite(args.maximum_graph_ms):
        raise ValueError("maximum-graph-ms must be finite and positive")


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    source_proof = validate_source_pair(args.orphan_source, args.ported_source)

    import torch
    from vllm.models.deepseek_v4.nvidia.prepared_weight_loading import (
        validate_prepared_layer_file,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("literal direct gate requires exactly one CUDA device")
    capability = tuple(torch.cuda.get_device_capability())
    if capability != (12, 1):
        raise RuntimeError(f"literal direct gate requires SM121; got {capability}")

    module = load_ported_module(args.ported_source)
    kernel_class = module.MoEDirectMicroKernel
    physical = validate_prepared_layer_file(args.layer_file, layer=0)
    if physical.get("fingerprints_match") is not True:
        raise RuntimeError("prepared physical-layer fingerprints did not match")
    shape = kernel_bench.Dsv4Shape(tp_rank=args.tp_rank)
    shape.validate()
    geometry = direct_geometry(
        M_VALUE,
        shape.hidden_size,
        shape.intermediate_size_per_rank,
        shape.top_k,
    )

    load_started = time.perf_counter()
    tensors = prepared_bench._load_rank(torch, args.layer_file, args.tp_rank)
    prepared_proof = validate_prepared_tensors(torch, tensors, shape)
    accepted_weights = prepared_bench._prepare_weights(torch, tensors, shape)
    load_seconds = time.perf_counter() - load_started

    runner_args = SimpleNamespace(m=(M_VALUE,), swiglu_limit=SWIGLU_LIMIT)
    cutlass_runner, cutlass_proof = kernel_bench._make_flashinfer_cutlass_runner(
        torch, accepted_weights, shape, runner_args
    )
    x, topk_ids, topk_weights = kernel_bench.make_routes(
        torch,
        shape,
        M_VALUE,
        routing=ROUTING,
        seed=args.seed,
        input_rms=1.0,
    )
    cutlass_launch, cutlass_output = kernel_bench._make_flashinfer_cutlass_launch(
        torch,
        cutlass_runner,
        accepted_weights,
        shape,
        x,
        topk_ids,
        topk_weights,
    )
    direct = compile_runner(
        torch, kernel_class, tensors, shape, x, topk_ids, topk_weights
    )

    failures: list[dict[str, Any]] = []
    cutlass_launch()
    direct.launch()
    torch.cuda.synchronize()
    cutlass_eager = cutlass_output.clone()
    direct_eager = direct.output.clone()
    activity = {
        "literal_orphan_direct": kernel_bench.tensor_activity(torch, direct_eager),
        "flashinfer_cutlass": kernel_bench.tensor_activity(torch, cutlass_eager),
    }
    for backend, evidence in activity.items():
        if not evidence["passed"]:
            failures.append({"kind": "output_activity", "backend": backend})
    eager_numeric, eager_passed = numeric_gate(
        torch, direct_eager, cutlass_eager, args
    )
    if not eager_passed:
        failures.append({"kind": "eager_numeric", **eager_numeric})

    pair = ("literal_orphan_direct", "flashinfer_cutlass")
    eager_timing = prepared_bench._time_orders(
        torch,
        {
            "literal_orphan_direct": direct.launch,
            "flashinfer_cutlass": cutlass_launch,
        },
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
        pair=pair,
    )

    graph_launches: dict[str, Any] = {}
    graph_outputs: dict[str, Any] = {}
    graph_keepalive: list[Any] = []
    for name, launch in (
        ("literal_orphan_direct", direct.launch),
        ("flashinfer_cutlass", cutlass_launch),
    ):
        replay, output, graph = kernel_bench.capture_graph(torch, launch)
        graph_launches[name] = replay
        graph_outputs[name] = output
        graph_keepalive.append(graph)

    graph_launches["flashinfer_cutlass"]()
    graph_launches["literal_orphan_direct"]()
    torch.cuda.synchronize()
    graph_numeric, graph_passed = numeric_gate(
        torch,
        graph_outputs["literal_orphan_direct"],
        graph_outputs["flashinfer_cutlass"],
        args,
    )
    direct_graph_vs_eager, direct_graph_eager_passed = numeric_gate(
        torch,
        graph_outputs["literal_orphan_direct"],
        direct_eager,
        args,
    )
    cutlass_graph_vs_eager, cutlass_graph_eager_passed = numeric_gate(
        torch, graph_outputs["flashinfer_cutlass"], cutlass_eager, args
    )
    if not graph_passed:
        failures.append({"kind": "graph_numeric", **graph_numeric})
    if not direct_graph_eager_passed:
        failures.append(
            {"kind": "direct_graph_vs_eager", **direct_graph_vs_eager}
        )
    if not cutlass_graph_eager_passed:
        failures.append(
            {"kind": "cutlass_graph_vs_eager", **cutlass_graph_vs_eager}
        )

    graph_timing = prepared_bench._time_orders(
        torch,
        graph_launches,
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
        pair=pair,
    )
    del graph_keepalive
    direct_graph_ms = float(
        graph_timing["combined"]["literal_orphan_direct"]["median_ms"]
    )
    cutlass_graph_ms = float(
        graph_timing["combined"]["flashinfer_cutlass"]["median_ms"]
    )
    speedup = cutlass_graph_ms / direct_graph_ms
    performance_passed = direct_graph_ms <= args.maximum_graph_ms
    if not performance_passed:
        failures.append(
            {
                "kind": "performance",
                "literal_graph_ms": direct_graph_ms,
                "maximum_graph_ms": args.maximum_graph_ms,
                "speedup_over_cutlass": speedup,
            }
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "probe": "flashinfer_literal_orphan_direct_dsv4_m4_sm121",
        "gpu": {
            "name": torch.cuda.get_device_name(),
            "capability": list(capability),
            "torch": torch.__version__,
        },
        "source_proof": source_proof,
        "checkpoint": {
            "layer_file": str(args.layer_file.resolve()),
            "physical_validation": physical,
            "tp_rank": args.tp_rank,
            "load_and_prepare_seconds": load_seconds,
        },
        "settings": {
            "m": M_VALUE,
            "routing": ROUTING,
            "seed": args.seed,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "numeric_min_cosine": args.numeric_min_cosine,
            "numeric_max_nrmse": args.numeric_max_nrmse,
        },
        "backend_proof": {
            "literal_orphan_direct": {
                "implementation": "ported MoEDirectMicroKernel",
                "constructor": kernel_kwargs(),
                "kernel_cache_key": repr(direct.kernel.__cache_key__),
                "grid_x": direct.grid_x,
                "required_block_dim": DIRECT_BLOCK_DIM,
                "geometry": vars(geometry),
                "prepared_scale_contract": prepared_proof,
                "descendant_optimizations_present": False,
                "serving_integration_claimed": False,
            },
            "flashinfer_cutlass": cutlass_proof,
        },
        "correctness": {
            "activity": activity,
            "eager_direct_vs_cutlass": eager_numeric,
            "eager_passed": eager_passed,
            "graph_direct_vs_cutlass": graph_numeric,
            "graph_passed": graph_passed,
            "direct_graph_vs_eager": direct_graph_vs_eager,
            "cutlass_graph_vs_eager": cutlass_graph_vs_eager,
        },
        "timing": {"eager": eager_timing, "cuda_graph": graph_timing},
        "performance_gate": {
            "maximum_literal_graph_ms": args.maximum_graph_ms,
            "literal_graph_ms": direct_graph_ms,
            "flashinfer_cutlass_graph_ms": cutlass_graph_ms,
            "speedup_literal_over_flashinfer_cutlass": speedup,
            "passed": performance_passed,
        },
        "failures": failures,
        "ok": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["performance_gate"], sort_keys=True))
    print(f"Wrote {args.output}")
    return 0 if not failures else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orphan-source", type=Path, required=True)
    parser.add_argument("--ported-source", type=Path, required=True)
    parser.add_argument("--layer-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--numeric-min-cosine", type=float, default=0.98)
    parser.add_argument("--numeric-max-nrmse", type=float, default=0.25)
    parser.add_argument(
        "--maximum-graph-ms", type=float, default=DEFAULT_MAXIMUM_GRAPH_MS
    )
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
