#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Real-layer M=4 gate for B12X's clamped NVFP4 direct microkernel.

This bounded probe compiles :class:`MoEMicroKernelBackend` directly instead
of entering through B12X's public NVFP4 dispatcher.  That distinction is
intentional: the public NVFP4 activation subclass does not expose DeepSeek
V4's ``SwiGLU(limit=10)`` contract, while the base kernel does.  The direct
kernel consumes the immutable prepared checkpoint's raw ModelOpt scale
algebra and is compared with the accepted FlashInfer CUTLASS W4A4 path on the
same layer, activations, routes, weights, and scales.

The measured case is fixed to balanced M=4, K=4096, I/rank=1024, E=256,
top-k=6.  Compilation, checkpoint I/O, correctness clones, and CUDA graph
capture are outside timed regions.  CUDA imports occur only in :func:`run`,
so ``--help`` and the unit contract remain usable on a development Mac.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import benchmark_nvfp4_a4w4_sm121 as kernel_bench
from benchmarks import benchmark_nvfp4_prepared_b12x_sm121 as prepared_bench


SCHEMA_VERSION = 1
M_VALUE = 4
ROUTING = "balanced"
SWIGLU_LIMIT = 10.0
W13_LAYOUT = "w13"  # B12X spelling for physical [up/w3, gate/w1].
DIRECT_BLOCK_DIM = 512
B12X_REVISION = "7dc6fb8fcc6446ea093537d1657df81985fa5f43"
B12X_MICRO_SOURCE_SHA256 = (
    "67847d6365b3707b54e5d68a89655666350029aa550c5f74742084f264d2d980"
)
B12X_TP_MOE_SOURCE_SHA256 = (
    "c2ca5aca4f9efd8ac8afb52909ef18410d1afd455d7e994debcd4e0bc13e019d"
)


@dataclass(frozen=True)
class DirectGeometry:
    m: int
    k: int
    n: int
    top_k: int
    fc1_chunks: int
    fc1_tasks: int
    fc2_tasks: int
    fc2_n_chunks: int
    intermediate_u32: int
    intermediate_bytes: int
    integration_arena_f32: int
    barrier_slots_per_array: int
    barrier_bytes_total: int
    output_bytes: int


@dataclass
class DirectRunner:
    kernel: Any
    compiled: Any
    intermediate: Any
    barrier_count: Any
    barrier_epoch: Any
    output: Any
    grid_x: int
    block_dim_guard: bool
    launch: Callable[[], Any]


def direct_geometry(m: int, k: int, n: int, top_k: int) -> DirectGeometry:
    """Mirror the pinned B12X direct workspace/task geometry in pure Python."""

    if m not in (1, 2, 4, 8):
        raise ValueError("direct microkernel M must be one of 1,2,4,8")
    if k <= 0 or k % 128 or n <= 0 or n % 16 or top_k <= 0:
        raise ValueError("invalid direct microkernel geometry")
    rows_per_chunk = max(16, 16 * m)
    fc1_chunks = max(1, n // rows_per_chunk)
    while fc1_chunks > 1 and (
        n % fc1_chunks or (n // fc1_chunks) % 16
    ):
        fc1_chunks -= 1
    fc2_n_chunks = (n // 2 + 127) // 128
    intermediate_u32 = m * top_k * fc2_n_chunks * 128
    return DirectGeometry(
        m=m,
        k=k,
        n=n,
        top_k=top_k,
        fc1_chunks=fc1_chunks,
        fc1_tasks=m * top_k * fc1_chunks,
        fc2_tasks=(k // (16 * 2) if m == 1 else (m * k) // (16 * 4)),
        fc2_n_chunks=fc2_n_chunks,
        intermediate_u32=intermediate_u32,
        intermediate_bytes=intermediate_u32 * 4,
        # The public integration planner reserves route-hidden staging plus
        # the kernel-minimum FC2 scratch. The direct probe needs only the
        # latter, but records both so the serving-memory comparison is exact.
        integration_arena_f32=m * top_k * k + intermediate_u32,
        # The current kernel touches slot zero for M>1, but the production
        # arena contract reserves routed rows plus 16 slots per token.
        barrier_slots_per_array=m * (top_k + 16),
        barrier_bytes_total=2 * m * (top_k + 16) * 4,
        output_bytes=m * k * 2,
    )


def direct_kernel_kwargs() -> dict[str, Any]:
    """Return the exact compile-time DeepSeek semantics under test."""

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
        "compile_time_phase": 0,
        "w4a16_mode": False,
        "scale_format": "e4m3_k16",
        "swiglu_limit": SWIGLU_LIMIT,
        "w13_layout": W13_LAYOUT,
    }


def prepared_scale_algebra() -> dict[str, str]:
    return {
        "a1_gscale": "1 / w13_input_scale",
        "g1_alphas": "w13_weight_scale_2 * w13_input_scale",
        "a2_gscale": "1 / w2_input_scale",
        "g2_alphas": "w2_weight_scale_2 * w2_input_scale",
        "raw_w13_weight_scale_2": "g1_alphas * a1_gscale",
        "raw_w2_weight_scale_2": "g2_alphas * a2_gscale",
        "block_scales": "raw ModelOpt E4M3 K16; no B12X bake or MMA transform",
        "w13_order": "up/w3 then gate/w1",
    }


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(symbol: Any, expected_sha256: str) -> dict[str, str]:
    source_text = inspect.getsourcefile(symbol)
    if source_text is None:
        raise RuntimeError(f"{symbol!r} has no inspectable source")
    source = Path(source_text).resolve()
    observed = _sha256_path(source)
    if observed != expected_sha256:
        raise RuntimeError(
            f"B12X source drifted for {symbol!r}: expected {expected_sha256}, "
            f"got {observed} ({source})"
        )
    return {"path": str(source), "sha256": observed}


def _validate_prepared_tensors(
    torch: Any,
    tensors: dict[str, Any],
    shape: Any,
) -> dict[str, Any]:
    expected = {
        "w13.weight": (torch.uint8, (shape.num_experts, 2 * shape.intermediate_size_per_rank, shape.hidden_size // 2)),
        "w2.weight": (torch.uint8, (shape.num_experts, shape.hidden_size, shape.intermediate_size_per_rank // 2)),
        "w13.weight_scale": (torch.float8_e4m3fn, (shape.num_experts, 2 * shape.intermediate_size_per_rank, shape.hidden_size // 16)),
        "w2.weight_scale": (torch.float8_e4m3fn, (shape.num_experts, shape.hidden_size, shape.intermediate_size_per_rank // 16)),
        "a1_gscale": (torch.float32, (shape.num_experts,)),
        "a2_gscale": (torch.float32, (shape.num_experts,)),
        "g1_alphas": (torch.float32, (shape.num_experts,)),
        "g2_alphas": (torch.float32, (shape.num_experts,)),
    }
    observed: dict[str, Any] = {}
    for name, (dtype, tensor_shape) in expected.items():
        tensor = tensors[name]
        if tensor.dtype != dtype or tuple(tensor.shape) != tensor_shape:
            raise RuntimeError(
                f"prepared tensor drifted for {name}: dtype={tensor.dtype}, "
                f"shape={tuple(tensor.shape)}, expected={dtype}/{tensor_shape}"
            )
        if not tensor.is_contiguous() or tensor.device.type != "cuda":
            raise RuntimeError(f"prepared tensor is not contiguous CUDA storage: {name}")
        observed[name] = {"dtype": str(dtype), "shape": list(tensor_shape)}

    scalar_proof: dict[str, Any] = {}
    for scale_name, alpha_name in (
        ("a1_gscale", "g1_alphas"),
        ("a2_gscale", "g2_alphas"),
    ):
        scale = tensors[scale_name]
        alpha = tensors[alpha_name]
        if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
            raise RuntimeError(f"invalid prepared reciprocal scale: {scale_name}")
        if not bool(torch.isfinite(alpha).all()) or not bool((alpha > 0).all()):
            raise RuntimeError(f"invalid prepared global alpha: {alpha_name}")
        if not torch.equal(scale, scale[0].expand_as(scale)):
            raise RuntimeError(f"prepared activation scale is not globally reduced: {scale_name}")
        recovered = alpha * scale
        if not bool(torch.isfinite(recovered).all()) or not bool((recovered > 0).all()):
            raise RuntimeError(f"recovered raw weight_scale_2 is invalid: {alpha_name}")
        scalar_proof[scale_name] = {
            "globally_constant": True,
            "value": float(scale[0].item()),
            "recovered_weight_scale_2_min": float(recovered.min().item()),
            "recovered_weight_scale_2_max": float(recovered.max().item()),
        }
    return {
        "tensors": observed,
        "algebra": prepared_scale_algebra(),
        "scalar_proof": scalar_proof,
        "runtime_scale_transforms": 0,
        "prepared_reads": 8,
    }


def _compile_direct_runner(
    torch: Any,
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
    from b12x.cute.utils import current_cuda_stream, make_ptr
    from b12x.moe.fused.micro import MoEMicroKernelBackend, _BLOCK_DIM
    from cutlass.cutlass_dsl import Int32

    if int(_BLOCK_DIM) != DIRECT_BLOCK_DIM:
        raise RuntimeError(
            f"B12X direct block dimension drifted: {_BLOCK_DIM} != {DIRECT_BLOCK_DIM}"
        )
    if not MoEMicroKernelBackend.is_supported(
        M_VALUE,
        shape.hidden_size,
        shape.intermediate_size_per_rank,
        shape.top_k,
        shape.num_experts,
    ):
        raise RuntimeError("B12X base direct kernel rejected the exact DSV4 M=4 shape")

    kernel = MoEMicroKernelBackend(**direct_kernel_kwargs())
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
        or kernel.scale_format != "e4m3_k16"
    ):
        raise RuntimeError("compiled direct-kernel DeepSeek semantics drifted")

    def dummy(dtype: Any) -> Any:
        return make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16)

    # Match the accepted native path: normalize routing IDs before compiling
    # so the fake pointer ABI and every real launch use the same Int32 type.
    launch_ids = topk_ids.to(torch.int32).contiguous()
    ids_dtype = cutlass.Int32
    barrier_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    cache_key = (
        "prepared_nvfp4_dsv4_m4_direct",
        kernel.__cache_key__,
        str(launch_ids.dtype),
    )
    # Mirror the pinned integration compiler exactly: M=2/4/8 stays a runtime
    # Int32, while the fake compile invocation supplies the supported maximum
    # value.  Only M=1 is intentionally specialized by ``m_const``.
    compile_m = int(kernel.m_const) if int(kernel.m_const) != 0 else 8
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
        dummy(ids_dtype),
        dummy(cutlass.Float32),
        dummy(cutlass.BFloat16),
        barrier_fake,
        barrier_fake,
        Int32(compile_m),
        Int32(1),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "benchmarks.probe_nvfp4_direct_micro", 1, cache_key
        ),
    )
    block_guard = bool(
        tp_moe._compiled_direct_micro_accepts_block_dim(compiled, DIRECT_BLOCK_DIM)
    )
    if not block_guard:
        raise RuntimeError(
            "compiled B12X direct kernel cannot launch its required 512 threads"
        )

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
        raise RuntimeError("B12X direct compiled geometry drifted from the probe contract")

    intermediate = torch.empty(
        geometry.intermediate_u32, dtype=torch.uint32, device=x.device
    )
    barrier_count = torch.zeros(
        geometry.barrier_slots_per_array, dtype=torch.int32, device=x.device
    )
    barrier_epoch = torch.zeros_like(barrier_count)
    output = torch.empty_like(x)

    def launch() -> Any:
        MoEMicroKernelBackend.launch(
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
        block_dim_guard=block_guard,
        launch=launch,
    )


def _numeric_gate(
    torch: Any,
    actual: Any,
    reference: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], bool]:
    metrics = kernel_bench.compare_tensors(torch, actual, reference)
    return metrics, kernel_bench.numeric_metrics_pass(
        metrics,
        min_cosine=args.numeric_min_cosine,
        max_normalized_rmse=args.numeric_max_nrmse,
    )


def _speedup(timing: dict[str, Any]) -> float:
    return float(
        timing["combined"]["speedup_b12x_direct_over_flashinfer_cutlass"]
    )


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("warmup", "iters", "repeats"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0.0 <= args.numeric_min_cosine <= 1.0:
        raise ValueError("numeric-min-cosine must be in [0, 1]")
    if args.numeric_max_nrmse < 0.0 or not math.isfinite(args.numeric_max_nrmse):
        raise ValueError("numeric-max-nrmse must be finite and non-negative")
    if args.minimum_speedup <= 0.0 or not math.isfinite(args.minimum_speedup):
        raise ValueError("minimum-speedup must be finite and positive")


def run(args: argparse.Namespace) -> int:
    _validate_args(args)

    import torch
    import b12x.integration.tp_moe as tp_moe
    from b12x.moe.fused.micro import MoEMicroKernelBackend
    from vllm.models.deepseek_v4.nvidia.prepared_weight_loading import (
        validate_prepared_layer_file,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("direct microkernel gate requires exactly one CUDA device")
    capability = tuple(torch.cuda.get_device_capability())
    if capability != (12, 1):
        raise RuntimeError(f"direct microkernel gate requires SM121; got {capability}")

    source_proof = {
        "revision": B12X_REVISION,
        "micro": _source_identity(
            MoEMicroKernelBackend, B12X_MICRO_SOURCE_SHA256
        ),
        "tp_moe": _source_identity(
            tp_moe._compiled_direct_micro_accepts_block_dim,
            B12X_TP_MOE_SOURCE_SHA256,
        ),
    }
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
    prepared_proof = _validate_prepared_tensors(torch, tensors, shape)
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
    direct = _compile_direct_runner(
        torch, tensors, shape, x, topk_ids, topk_weights
    )

    failures: list[dict[str, Any]] = []
    cutlass_launch()
    direct.launch()
    torch.cuda.synchronize()
    cutlass_eager = cutlass_output.clone()
    direct_eager = direct.output.clone()
    activity = {
        "b12x_direct": kernel_bench.tensor_activity(torch, direct_eager),
        "flashinfer_cutlass": kernel_bench.tensor_activity(torch, cutlass_eager),
    }
    for backend, evidence in activity.items():
        if not evidence["passed"]:
            failures.append({"kind": "output_activity", "backend": backend})
    eager_numeric, eager_numeric_passed = _numeric_gate(
        torch, direct_eager, cutlass_eager, args
    )
    if not eager_numeric_passed:
        failures.append({"kind": "eager_numeric", **eager_numeric})

    eager_timing = prepared_bench._time_orders(
        torch,
        {
            "b12x_direct": direct.launch,
            "flashinfer_cutlass": cutlass_launch,
        },
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
        pair=("b12x_direct", "flashinfer_cutlass"),
    )

    graph_launches: dict[str, Any] = {}
    graph_outputs: dict[str, Any] = {}
    graph_keepalive: list[Any] = []
    for name, launch in (
        ("b12x_direct", direct.launch),
        ("flashinfer_cutlass", cutlass_launch),
    ):
        replay, output, graph = kernel_bench.capture_graph(torch, launch)
        graph_launches[name] = replay
        graph_outputs[name] = output
        graph_keepalive.append(graph)
    del graph_keepalive  # Graph objects remain closed over by replay callables.

    graph_launches["flashinfer_cutlass"]()
    graph_launches["b12x_direct"]()
    torch.cuda.synchronize()
    graph_numeric, graph_numeric_passed = _numeric_gate(
        torch,
        graph_outputs["b12x_direct"],
        graph_outputs["flashinfer_cutlass"],
        args,
    )
    direct_graph_vs_eager, direct_graph_passed = _numeric_gate(
        torch, graph_outputs["b12x_direct"], direct_eager, args
    )
    cutlass_graph_vs_eager, cutlass_graph_passed = _numeric_gate(
        torch, graph_outputs["flashinfer_cutlass"], cutlass_eager, args
    )
    if not graph_numeric_passed:
        failures.append({"kind": "graph_numeric", **graph_numeric})
    if not direct_graph_passed:
        failures.append(
            {"kind": "direct_graph_vs_eager", **direct_graph_vs_eager}
        )
    if not cutlass_graph_passed:
        failures.append(
            {"kind": "cutlass_graph_vs_eager", **cutlass_graph_vs_eager}
        )

    graph_timing = prepared_bench._time_orders(
        torch,
        graph_launches,
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
        pair=("b12x_direct", "flashinfer_cutlass"),
    )
    eager_speedup = _speedup(eager_timing)
    graph_speedup = _speedup(graph_timing)
    performance_passed = (
        eager_speedup >= args.minimum_speedup
        and graph_speedup >= args.minimum_speedup
    )
    if not performance_passed:
        failures.append(
            {
                "kind": "performance",
                "eager_speedup": eager_speedup,
                "graph_speedup": graph_speedup,
                "minimum_speedup": args.minimum_speedup,
            }
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "probe": "prepared_nvfp4_b12x_base_direct_m4_sm121",
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
            "b12x_direct": {
                "implementation": "b12x.moe.fused.micro.MoEMicroKernelBackend",
                "source": source_proof,
                "constructor": direct_kernel_kwargs(),
                "kernel_cache_key": repr(direct.kernel.__cache_key__),
                "grid_x": direct.grid_x,
                "required_block_dim": DIRECT_BLOCK_DIM,
                "compiled_block_dim_guard_passed": direct.block_dim_guard,
                "geometry": vars(geometry),
                "prepared_scale_contract": prepared_proof,
                "serving_integration_claimed": False,
            },
            "flashinfer_cutlass": cutlass_proof,
        },
        "correctness": {
            "activity": activity,
            "eager_direct_vs_cutlass": eager_numeric,
            "eager_passed": eager_numeric_passed,
            "graph_direct_vs_cutlass": graph_numeric,
            "graph_passed": graph_numeric_passed,
            "direct_graph_vs_eager": direct_graph_vs_eager,
            "cutlass_graph_vs_eager": cutlass_graph_vs_eager,
        },
        "timing": {"eager": eager_timing, "cuda_graph": graph_timing},
        "performance_gate": {
            "minimum_speedup": args.minimum_speedup,
            "speedup_b12x_direct_over_flashinfer_cutlass": {
                "eager": eager_speedup,
                "cuda_graph": graph_speedup,
            },
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
    parser.add_argument("--layer-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--numeric-min-cosine", type=float, default=0.98)
    parser.add_argument("--numeric-max-nrmse", type=float, default=0.25)
    parser.add_argument("--minimum-speedup", type=float, default=1.0)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
