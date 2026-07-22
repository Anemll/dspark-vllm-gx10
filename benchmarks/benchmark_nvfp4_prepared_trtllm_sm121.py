#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Probe FlashInfer TRTLLM routed NVFP4 on one real prepared TP2 layer.

The installed vLLM backend advertises the exact ModelOpt NVFP4 scheme and
DeepSeek clamp, and the FlashInfer symbol is installed, but its device policy
admits only Blackwell family 100.  This benchmark proves that family 121 is
the sole oracle rejection, prepares the same immutable layer for TRTLLM, and
compares the routed modular kernel with the accepted FlashInfer CUTLASS path.

This is a one-layer diagnostic.  It does not modify vLLM's serving selector or
load a model.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import benchmark_nvfp4_a4w4_sm121 as kernel_bench
from benchmarks import benchmark_nvfp4_prepared_b12x_sm121 as prepared_bench


SCHEMA_VERSION = 1
TRTLLM_MODE = "flashinfer_trtllm"
CUTLASS_MODE = kernel_bench.FLASHINFER_CUTLASS_MODE


@dataclass(frozen=True)
class TrtLlmRunner:
    experts: Any
    activation: Any
    w13: Any
    w2: Any
    owner: Any


def _csv_positive_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(dict.fromkeys(int(part) for part in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("M values must be positive")
    return result


def evaluate_sm121_support_override(
    *,
    capability: tuple[int, int],
    symbol_available: bool,
    native_supported: bool,
    native_reason: str | None,
    family_only_supported: bool,
    family_only_reason: str | None,
) -> dict[str, Any]:
    """Require that overriding only the family gate admits the full config."""

    passed = bool(
        capability == (12, 1)
        and symbol_available
        and not native_supported
        and native_reason
        and "current device" in native_reason
        and family_only_supported
        and family_only_reason is None
    )
    return {
        "capability": list(capability),
        "flashinfer_symbol_available": symbol_available,
        "native_supported": native_supported,
        "native_reason": native_reason,
        "family_only_override_supported": family_only_supported,
        "family_only_override_reason": family_only_reason,
        "only_changed_predicate": "TrtLlmNvFp4ExpertsBase._supports_current_device",
        "passed": passed,
    }


def summarize_backend_call(call: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on the physical routed-TRTLLM invocation contract."""

    required = {
        "do_finalize": True,
        "top_k": 6,
        "num_experts": 256,
        "local_num_experts": 256,
        "gemm1_clamp_present": True,
        "gemm1_alpha_present": True,
        "gemm1_beta_present": True,
        "output_pointer_identity": True,
    }
    return {
        "required": required,
        "observed": call,
        "passed": all(call.get(key) == value for key, value in required.items()),
    }


def _make_parallel_config(shape: kernel_bench.Dsv4Shape) -> Any:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

    return FusedMoEParallelConfig(
        tp_size=shape.tp_size,
        tp_rank=shape.tp_rank,
        pcp_size=1,
        pcp_rank=0,
        dp_size=1,
        dp_rank=0,
        ep_size=1,
        ep_rank=0,
        sp_size=1,
        use_ep=False,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )


def _unswizzle_prepared_scales(
    torch: Any,
    weights: kernel_bench.PreparedWeights,
    shape: kernel_bench.Dsv4Shape,
) -> tuple[Any, Any]:
    def unswizzle_sf(
        sf: Any,
        *,
        rows: int,
        cols: int,
        scaling_vector_size: int = 16,
    ) -> Any:
        """Reverse FlashInfer's 128x4 scale swizzle with tensor views only.

        The installed SM121 image exposes the forward TRTLLM interleave op but
        not ``block_scale_interleave_reverse``.  This is setup-only benchmark
        work, so use FlashInfer's documented layout algebra instead of making
        the latent backend depend on an unregistered C++ utility operator.
        """

        factor = scaling_vector_size * 4
        row_tiles = (rows + 127) // 128
        col_tiles = (cols + factor - 1) // factor
        bytes_per_matrix = row_tiles * col_tiles * 128 * 4
        if sf.numel() % bytes_per_matrix:
            raise RuntimeError(
                "swizzled scale storage is not an integer number of matrices"
            )
        batch = sf.numel() // bytes_per_matrix
        linear = (
            sf.contiguous()
            .reshape(batch, row_tiles, col_tiles, 32, 4, 4)
            .transpose(2, 4)
            .reshape(batch, row_tiles * 128, col_tiles * 4)
        )
        return (
            linear[:, :rows, : cols // scaling_vector_size]
            .reshape(batch * rows, cols // scaling_vector_size)
            .contiguous()
        )

    intermediate = shape.intermediate_size_per_rank
    w13_rows = 2 * intermediate
    w13 = unswizzle_sf(
        weights.w13_sf_modelopt,
        rows=w13_rows,
        cols=shape.hidden_size,
    ).reshape(shape.num_experts, w13_rows, shape.hidden_size // 16)
    w2 = unswizzle_sf(
        weights.w2_sf_modelopt,
        rows=shape.hidden_size,
        cols=intermediate,
    ).reshape(
        shape.num_experts,
        shape.hidden_size,
        intermediate // 16,
    )
    if w13.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        raise RuntimeError("TRTLLM scale unswizzle changed the E4M3 dtype")
    return w13.contiguous(), w2.contiguous()


def _make_trtllm_runner(
    torch: Any,
    weights: kernel_bench.PreparedWeights,
    shape: kernel_bench.Dsv4Shape,
    args: argparse.Namespace,
) -> tuple[TrtLlmRunner, dict[str, Any]]:
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        RoutingMethodType,
        nvfp4_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
        TrtLlmNvFp4ExpertsModular,
    )
    from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
        prepare_static_weights_for_trtllm_fp4_moe,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kNvfp4Dynamic,
        kNvfp4Static,
    )
    from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe

    capability = tuple(torch.cuda.get_device_capability())
    symbol_available = bool(has_flashinfer_trtllm_fused_moe())
    activation = MoEActivation.SILU
    moe_config = FusedMoEConfig(
        num_experts=shape.num_experts,
        experts_per_token=shape.top_k,
        hidden_dim=shape.hidden_size,
        intermediate_size=shape.intermediate_size,
        num_local_experts=shape.num_experts,
        num_logical_experts=shape.num_experts,
        activation=activation,
        device="cuda",
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=_make_parallel_config(shape),
        in_dtype=torch.bfloat16,
        moe_backend=TRTLLM_MODE,
        max_num_tokens=max(args.m),
        skip_final_all_reduce=True,
        swiglu_limit=args.swiglu_limit,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
    )
    if moe_config.intermediate_size_per_partition != shape.intermediate_size_per_rank:
        raise RuntimeError("TRTLLM TP geometry does not match the prepared rank")

    native_supported, native_reason = TrtLlmNvFp4ExpertsModular.is_supported_config(
        TrtLlmNvFp4ExpertsModular,
        moe_config,
        kNvfp4Static,
        kNvfp4Dynamic,
        mk.FusedMoEActivationFormat.Standard,
    )
    with mock.patch.object(
        TrtLlmNvFp4ExpertsModular,
        "_supports_current_device",
        return_value=True,
    ):
        family_supported, family_reason = (
            TrtLlmNvFp4ExpertsModular.is_supported_config(
                TrtLlmNvFp4ExpertsModular,
                moe_config,
                kNvfp4Static,
                kNvfp4Dynamic,
                mk.FusedMoEActivationFormat.Standard,
            )
        )
    support = evaluate_sm121_support_override(
        capability=capability,
        symbol_available=symbol_available,
        native_supported=native_supported,
        native_reason=native_reason,
        family_only_supported=family_supported,
        family_only_reason=family_reason,
    )
    if not support["passed"]:
        raise RuntimeError(f"TRTLLM SM121 support audit failed: {support}")

    linear_w13_scale, linear_w2_scale = _unswizzle_prepared_scales(
        torch, weights, shape
    )
    trtllm_w13, trtllm_w13_scale, trtllm_w2, trtllm_w2_scale = (
        prepare_static_weights_for_trtllm_fp4_moe(
            weights.w13,
            weights.w2,
            linear_w13_scale,
            linear_w2_scale,
            hidden_size=shape.hidden_size,
            intermediate_size=shape.intermediate_size_per_rank,
            num_experts=shape.num_experts,
            is_gated_activation=True,
        )
    )

    raw_g1 = (
        weights.cutlass_g1_alphas * weights.cutlass_a1_gscale
    ).to(torch.float32).contiguous()
    raw_g2 = (
        weights.cutlass_g2_alphas * weights.cutlass_a2_gscale
    ).to(torch.float32).contiguous()
    input1 = weights.cutlass_a1_gscale.reciprocal().to(torch.float32).contiguous()
    input2 = weights.cutlass_a2_gscale.reciprocal().to(torch.float32).contiguous()

    owner = torch.nn.Module()
    for name, tensor in (
        ("w13_weight_scale_2", raw_g1),
        ("w2_weight_scale_2", raw_g2),
        ("w13_input_scale", input1),
        ("w2_input_scale", input2),
    ):
        owner.register_parameter(
            name,
            torch.nn.Parameter(tensor, requires_grad=False),
        )
    quant_config = nvfp4_moe_quant_config(
        g1_alphas=owner.w13_weight_scale_2,
        g2_alphas=owner.w2_weight_scale_2,
        a1_gscale=weights.cutlass_a1_gscale,
        a2_gscale=weights.cutlass_a2_gscale,
        w1_scale=trtllm_w13_scale,
        w2_scale=trtllm_w2_scale,
        is_scale_swizzled=False,
        gemm1_clamp_limit=args.swiglu_limit,
    )
    experts = TrtLlmNvFp4ExpertsModular(
        moe_config=moe_config,
        quant_config=quant_config,
    )
    experts.process_weights_after_loading(owner)

    expected_g1 = weights.cutlass_g1_alphas
    expected_g2 = weights.cutlass_g2_alphas
    algebra = {
        "g1_alphas_match_cutlass": bool(
            torch.allclose(
                quant_config.g1_alphas, expected_g1, rtol=1e-6, atol=0.0
            )
        ),
        "g2_alphas_match_cutlass": bool(
            torch.allclose(
                quant_config.g2_alphas, expected_g2, rtol=1e-6, atol=0.0
            )
        ),
        "g1_scale_c_match": bool(
            torch.allclose(
                experts.g1_scale_c,
                expected_g1 * weights.cutlass_a2_gscale,
                rtol=1e-6,
                atol=0.0,
            )
        ),
        "clamp_raw_space_match": bool(
            torch.allclose(
                experts.gemm1_clamp_limit,
                torch.full_like(expected_g1, args.swiglu_limit) / expected_g1,
            )
        ),
        "alpha_is_one": bool(
            torch.equal(experts.gemm1_alpha, torch.ones_like(expected_g1))
        ),
        "beta_is_zero": bool(
            torch.equal(experts.gemm1_beta, torch.zeros_like(expected_g1))
        ),
    }
    if not all(algebra.values()):
        raise RuntimeError(f"TRTLLM ModelOpt/clamp algebra failed: {algebra}")

    proof = {
        "requested": TRTLLM_MODE,
        "implementation": (
            f"{experts.__class__.__module__}.{experts.__class__.__qualname__}"
        ),
        "physical_symbol": "flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe",
        "support": support,
        "activation": activation.value,
        "activation_precision": "nvfp4",
        "weight_source": "prepared ModelOpt NVFP4 layer",
        "prepared_weight_copy_is_benchmark_only": True,
        "source_w13_layout": "w3/up, w1/gate",
        "swiglu_limit": args.swiglu_limit,
        "swiglu_alpha": 1.0,
        "swiglu_beta": 0.0,
        "is_scale_swizzled": quant_config.is_scale_swizzled,
        "tp_rank": shape.tp_rank,
        "tp_size": shape.tp_size,
        "algebra": algebra,
        "weight_shapes": {
            "w13": list(trtllm_w13.shape),
            "w13_scale": list(trtllm_w13_scale.shape),
            "w2": list(trtllm_w2.shape),
            "w2_scale": list(trtllm_w2_scale.shape),
        },
    }
    return (
        TrtLlmRunner(
            experts=experts,
            activation=activation,
            w13=trtllm_w13,
            w2=trtllm_w2,
            owner=owner,
        ),
        proof,
    )


def _make_trtllm_launch(
    torch: Any,
    runner: TrtLlmRunner,
    shape: kernel_bench.Dsv4Shape,
    x: Any,
    topk_ids: Any,
    topk_weights: Any,
) -> tuple[Callable[[], Any], Any]:
    from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

    output = torch.empty_like(x)
    workspace13 = torch.empty((0,), dtype=torch.uint8, device=x.device)
    workspace2 = torch.empty((0,), dtype=torch.uint8, device=x.device)
    quant_config = runner.experts.quant_config

    def launch() -> Any:
        a1q, a1q_scale = moe_kernel_quantize_input(
            x,
            quant_config.a1_gscale,
            quant_dtype=quant_config.quant_dtype,
            per_act_token_quant=quant_config.per_act_token_quant,
            block_shape=quant_config.block_shape,
            is_scale_swizzled=quant_config.is_scale_swizzled,
            mx_alignment=quant_config.mx_alignment,
        )
        runner.experts.apply(
            output=output,
            hidden_states=a1q,
            w1=runner.w13,
            w2=runner.w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=runner.activation,
            global_num_experts=shape.num_experts,
            expert_map=None,
            a1q_scale=a1q_scale,
            a2_scale=None,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )
        return output

    return launch, output


def _prove_physical_invocation(
    torch: Any,
    launch: Callable[[], Any],
    expected_output: Any,
) -> dict[str, Any]:
    import flashinfer

    original = flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe
    calls: list[dict[str, Any]] = []

    def traced(**kwargs: Any) -> Any:
        calls.append(
            {
                "do_finalize": bool(kwargs["do_finalize"]),
                "top_k": int(kwargs["top_k"]),
                "num_experts": int(kwargs["num_experts"]),
                "local_num_experts": int(kwargs["local_num_experts"]),
                "gemm1_clamp_present": kwargs["gemm1_clamp_limit"] is not None,
                "gemm1_alpha_present": kwargs["gemm1_alpha"] is not None,
                "gemm1_beta_present": kwargs["gemm1_beta"] is not None,
                "output_pointer_identity": int(kwargs["output"].data_ptr())
                == int(expected_output.data_ptr()),
                "hidden_states_shape": list(kwargs["hidden_states"].shape),
                "hidden_states_scale_shape": list(
                    kwargs["hidden_states_scale"].shape
                ),
            }
        )
        return original(**kwargs)

    flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe = traced
    try:
        output = launch()
        torch.cuda.synchronize()
    finally:
        flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe = original
    if int(output.data_ptr()) != int(expected_output.data_ptr()):
        raise RuntimeError("TRTLLM launch returned a different output tensor")
    if len(calls) != 1:
        raise RuntimeError(f"expected one physical TRTLLM call, observed {len(calls)}")
    proof = summarize_backend_call(calls[0])
    if not proof["passed"]:
        raise RuntimeError(f"physical TRTLLM invocation contract failed: {proof}")
    return proof


def run(args: argparse.Namespace) -> int:
    import torch
    from vllm.models.deepseek_v4.nvidia.prepared_weight_loading import (
        validate_prepared_layer_file,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("TRTLLM real-layer gate requires exactly one CUDA device")
    capability = tuple(torch.cuda.get_device_capability())
    if capability != (12, 1):
        raise RuntimeError(f"TRTLLM real-layer gate requires SM121; got {capability}")
    if args.tp_rank not in (0, 1):
        raise ValueError("TP rank must be 0 or 1")

    physical = validate_prepared_layer_file(args.layer_file, layer=0)
    shape = kernel_bench.Dsv4Shape(tp_rank=args.tp_rank)
    shape.validate()
    tensors = prepared_bench._load_rank(torch, args.layer_file, args.tp_rank)
    weights = prepared_bench._prepare_weights(torch, tensors, shape)
    runner_args = SimpleNamespace(m=args.m, swiglu_limit=args.swiglu_limit)
    cutlass_runner, cutlass_proof = kernel_bench._make_flashinfer_cutlass_runner(
        torch, weights, shape, runner_args
    )
    trtllm_runner, trtllm_proof = _make_trtllm_runner(
        torch, weights, shape, args
    )
    torch.cuda.synchronize()

    failures: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    keepalive: list[Any] = [cutlass_runner, trtllm_runner]
    for m in args.m:
        x, topk_ids, topk_weights = kernel_bench.make_routes(
            torch,
            shape,
            m,
            routing=args.routing,
            seed=args.seed + m,
            input_rms=1.0,
        )
        cutlass_launch, _cutlass_output = (
            kernel_bench._make_flashinfer_cutlass_launch(
                torch,
                cutlass_runner,
                weights,
                shape,
                x,
                topk_ids,
                topk_weights,
            )
        )
        trtllm_launch, trtllm_output = _make_trtllm_launch(
            torch,
            trtllm_runner,
            shape,
            x,
            topk_ids,
            topk_weights,
        )
        physical_call = _prove_physical_invocation(
            torch, trtllm_launch, trtllm_output
        )

        eager_outputs: dict[str, Any] = {}
        activity: dict[str, Any] = {}
        for name, launch in (
            (TRTLLM_MODE, trtllm_launch),
            (CUTLASS_MODE, cutlass_launch),
        ):
            output = launch()
            torch.cuda.synchronize()
            eager_outputs[name] = output.clone()
            activity[name] = kernel_bench.tensor_activity(torch, output)
            if not activity[name]["passed"]:
                failures.append(
                    {"kind": "output_activity", "m": m, "backend": name}
                )

        numeric = kernel_bench.compare_tensors(
            torch,
            eager_outputs[TRTLLM_MODE],
            eager_outputs[CUTLASS_MODE],
        )
        numeric_passed = kernel_bench.numeric_metrics_pass(
            numeric,
            min_cosine=args.numeric_min_cosine,
            max_normalized_rmse=args.numeric_max_nrmse,
        )
        if not numeric_passed:
            failures.append({"kind": "numeric", "m": m, **numeric})

        graph_launches: dict[str, Callable[[], Any]] = {}
        graph_status: dict[str, Any] = {}
        for name, launch in (
            (TRTLLM_MODE, trtllm_launch),
            (CUTLASS_MODE, cutlass_launch),
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
                "captured": True,
                "vs_eager": graph_numeric,
                "passed": graph_passed,
            }
            keepalive.extend((graph_output, graph))
            if not graph_passed:
                failures.append(
                    {"kind": "graph_numeric", "m": m, "backend": name}
                )

        timing = prepared_bench._time_orders(
            torch,
            graph_launches,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
            pair=(TRTLLM_MODE, CUTLASS_MODE),
        )
        speedup = float(
            timing["combined"][f"speedup_{TRTLLM_MODE}_over_{CUTLASS_MODE}"]
        )
        unique_experts, counts = torch.unique(topk_ids, return_counts=True)
        results.append(
            {
                "m": m,
                "routing": args.routing,
                "routed_rows": m * shape.top_k,
                "unique_experts": int(unique_experts.numel()),
                "maximum_expert_multiplicity": int(counts.max().item()),
                "physical_backend_call": physical_call,
                "activity": activity,
                "numeric": numeric,
                "numeric_passed": numeric_passed,
                "cuda_graph_status": graph_status,
                "cuda_graph": timing,
                f"speedup_{TRTLLM_MODE}_over_{CUTLASS_MODE}": speedup,
            }
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "probe": "prepared_nvfp4_trtllm_vs_cutlass_sm121",
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
            "m": list(args.m),
            "routing": args.routing,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "seed": args.seed,
            "swiglu_limit": args.swiglu_limit,
        },
        "backend_proof": {
            TRTLLM_MODE: trtllm_proof,
            CUTLASS_MODE: cutlass_proof,
        },
        "provenance": {
            "trtllm_experts_source": inspect.getsourcefile(
                trtllm_runner.experts.__class__
            ),
            "trtllm_experts_apply_signature": str(
                inspect.signature(trtllm_runner.experts.apply)
            ),
        },
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
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for result in results:
        combined = result["cuda_graph"]["combined"]
        print(
            f"M={result['m']} TRTLLM={combined[TRTLLM_MODE]['median_ms']:.6f} ms "
            f"CUTLASS={combined[CUTLASS_MODE]['median_ms']:.6f} ms "
            f"speedup={result[f'speedup_{TRTLLM_MODE}_over_{CUTLASS_MODE}']:.6f}x"
        )
    print(f"Wrote {args.output}")
    return 0 if not failures else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--m", type=_csv_positive_ints, default=(1, 4))
    parser.add_argument(
        "--routing",
        choices=("balanced", "random", "hot"),
        default="balanced",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--swiglu-limit", type=float, default=10.0)
    parser.add_argument("--numeric-min-cosine", type=float, default=0.98)
    parser.add_argument("--numeric-max-nrmse", type=float, default=0.25)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.tp_rank not in (0, 1):
        raise ValueError("TP rank must be 0 or 1")
    if 4 not in args.m:
        raise ValueError("TRTLLM comparison requires M=4")
    if args.warmup < 0 or args.iters <= 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative; iters/repeats must be positive")
    if not math.isfinite(args.swiglu_limit) or args.swiglu_limit <= 0:
        raise ValueError("swiglu-limit must be positive and finite")
    if not 0.0 <= args.numeric_min_cosine <= 1.0:
        raise ValueError("numeric-min-cosine must be within [0, 1]")
    if args.numeric_max_nrmse < 0:
        raise ValueError("numeric-max-nrmse must be non-negative")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        return run(args)
    except (AssertionError, FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
