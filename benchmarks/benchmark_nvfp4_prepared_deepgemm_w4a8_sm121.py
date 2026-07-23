#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Gate lossless prepared-NVFP4 collapse through DeepGEMM W4A8 on one layer.

The prepared checkpoint stores the original MXFP4 E2M1 payload plus an exact
E8M0/K32 -> E4M3/K16 scale expansion.  This probe reverses that expansion,
restores DeepGEMM's gate/up W13 order, and compares the resulting MXFP4/FP8
expert against the accepted prepared-NVFP4/FP4 CUTLASS path.  It never builds
or serves a model.  Promotion requires correctness and at least 3% lower
CUDA-graph latency at both DSpark verifier shapes M=24 and M=48.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import benchmark_nvfp4_a4w4_sm121 as kernel_bench
from benchmarks import benchmark_nvfp4_prepared_b12x_sm121 as prepared_bench
from benchmarks import benchmark_nvfp4_prepared_mxfp4_requant_sm121 as collapse_bench


SCHEMA_VERSION = 1
REQUIRED_M = (1, 4, 24, 48)
DECISION_M = (24, 48)
MINIMUM_DECISION_SPEEDUP = 1.03
DEFAULT_NUMERIC_MIN_COSINE = 0.98
DEFAULT_NUMERIC_MAX_NRMSE = 0.25


def _csv_positive_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("M values must be positive")
    return result


def swap_gate_up_halves(value: Any) -> Any:
    """Swap the two contiguous projection halves along the N dimension."""

    if getattr(value, "ndim", None) != 3 or value.shape[1] % 2:
        raise ValueError(
            "W13 payload/scale must be rank-3 with an even projection dimension"
        )
    half = value.shape[1] // 2
    return (
        value.reshape(value.shape[0], 2, half, value.shape[2])
        .flip(dims=(1,))
        .reshape(value.shape)
        .contiguous()
    )


def evaluate_decision_rows(
    rows: Mapping[int, Mapping[str, Any]],
    *,
    minimum_speedup: float = MINIMUM_DECISION_SPEEDUP,
) -> dict[str, Any]:
    missing = sorted(set(REQUIRED_M) - set(rows))
    if missing:
        raise ValueError(f"missing required M rows: {missing}")
    decision: dict[str, Any] = {}
    passed = True
    for m in DECISION_M:
        row = rows[m]
        control_ms = float(row["cutlass_graph_ms"])
        candidate_ms = float(row["deepgemm_graph_ms"])
        speedup = control_ms / candidate_ms
        row_passed = bool(
            row["numeric_passed"]
            and row["graph_passed"]
            and speedup >= minimum_speedup
        )
        decision[str(m)] = {
            "cutlass_graph_ms": control_ms,
            "deepgemm_graph_ms": candidate_ms,
            "speedup": speedup,
            "minimum_speedup": minimum_speedup,
            "passed": row_passed,
        }
        passed = passed and row_passed
    return {"shapes": decision, "passed": passed}


def _make_cutlass_weights(torch: Any, tensors: Mapping[str, Any]) -> Any:
    metadata = {
        "source": "prepared-physical-layer0",
        "source_weight_data_ptrs": {
            "w13": int(tensors["w13.weight"].data_ptr()),
            "w2": int(tensors["w2.weight"].data_ptr()),
        },
        "weight_preparation_contract": {"flashinfer_b12x": False},
        "checkpoint_input_scale_tensor_count": 3 * int(tensors["w13.weight"].shape[0]),
        "modelopt_activation_scale_contract": {
            "loaded_from_prepared_checkpoint": True,
            "raw_weight_scale_2_recovery": "g_alpha * a_gscale",
        },
    }
    return kernel_bench.PreparedWeights(
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


def _collapse_and_pack_deepgemm(
    torch: Any,
    tensors: Mapping[str, Any],
    shape: Any,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    from b12x.moe.fused.w4a16.host import unswizzle_expert_scales
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        _pack_deepgemm_mxfp4_scales,
    )

    raw_g1 = collapse_bench.recover_raw_global_scale(
        tensors["g1_alphas"], tensors["a1_gscale"]
    ).to(torch.float32)
    raw_g2 = collapse_bench.recover_raw_global_scale(
        tensors["g2_alphas"], tensors["a2_gscale"]
    ).to(torch.float32)
    w13_e8m0, w13_proof = collapse_bench.collapse_nvfp4_scale_grid(
        torch,
        tensors["w13.weight_scale"],
        raw_g1,
        rows=2 * shape.intermediate_size_per_rank,
        cols=shape.hidden_size,
        name="w13",
        unswizzle_expert_scales=unswizzle_expert_scales,
    )
    w2_e8m0, w2_proof = collapse_bench.collapse_nvfp4_scale_grid(
        torch,
        tensors["w2.weight_scale"],
        raw_g2,
        rows=shape.hidden_size,
        cols=shape.intermediate_size_per_rank,
        name="w2",
        unswizzle_expert_scales=unswizzle_expert_scales,
    )

    # Prepared CUTLASS storage is [up/w3, gate/w1]. DeepGEMM consumes
    # [gate/w1, up/w3], so both payload rows and their K32 scales are swapped.
    w13 = swap_gate_up_halves(tensors["w13.weight"])
    w13_e8m0 = swap_gate_up_halves(w13_e8m0)
    w2 = tensors["w2.weight"]
    w13_scale, w2_scale = _pack_deepgemm_mxfp4_scales(
        w13, w2, w13_e8m0, w2_e8m0
    )
    torch.cuda.synchronize()
    return w13, w2, w13_scale, w2_scale, {
        "w13": w13_proof,
        "w2": w2_proof,
        "weight_payload_transform": "projection-half swap only; no nibble change",
        "source_w13_layout": "up_gate",
        "deepgemm_w13_layout": "gate_up",
        "scale_layout": "DeepGEMM transformed E8M0/K32",
        "passed": True,
    }


def _make_deepgemm_runner(
    torch: Any,
    tensors: Mapping[str, Any],
    shape: Any,
    m_values: tuple[int, ...],
    *,
    swiglu_limit: float,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_make_prepare_finalize,
    )
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
        DeepGemmFP4Experts,
    )
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        make_mxfp4_moe_quant_config,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    w13, w2, w13_scale, w2_scale, conversion = _collapse_and_pack_deepgemm(
        torch, tensors, shape
    )
    parallel = FusedMoEParallelConfig(
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
    activation = MoEActivation.SWIGLUOAI_UNINTERLEAVE
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
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
        moe_backend="deepgemm_mxfp4",
        max_num_tokens=max(m_values),
        skip_final_all_reduce=True,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
    )
    quant_config = make_mxfp4_moe_quant_config(
        Mxfp4MoeBackend.DEEPGEMM_MXFP4,
        w13_scale,
        w2_scale,
        gemm1_alpha=1.0,
        gemm1_beta=0.0,
        swiglu_limit=swiglu_limit,
    )
    if quant_config is None:
        raise RuntimeError("DeepGEMM MXFP4 quant config was not created")

    class DeepGemmW4A8SwiGLUOAIExperts(DeepGemmFP4Experts):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.gemm1_alpha = float(quant_config.gemm1_alpha or 1.0)
            self.gemm1_beta = float(quant_config.gemm1_beta or 0.0)

        @staticmethod
        def _supports_activation(value: Any) -> bool:
            return value in (
                MoEActivation.SILU,
                MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            )

        def _act_mul_quant(
            self, input: Any, output: Any, activation: Any
        ) -> tuple[Any, Any]:
            from vllm.model_executor.layers.quantization.utils.fp8_utils import (
                per_token_group_quant_fp8,
                per_token_group_quant_fp8_packed_for_deepgemm,
                silu_mul_per_token_group_quant_fp8_colmajor,
            )
            from vllm.model_executor.layers.quantization.utils.fp8_utils import (
                silu_mul_quant_fp8_packed_triton as fused_silu_mul_fp8_quant_packed,
            )
            from vllm.utils.deep_gemm import (
                DeepGemmQuantScaleFMT,
            )

            block_k = self._ACT_BLOCK_K
            scale_fmt = DeepGemmQuantScaleFMT.from_oracle()
            m_sum, n = input.size()
            activation_out_dim = self.adjust_N_for_activation(n, activation)
            fused_gated = activation in (
                MoEActivation.SILU,
                MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            )
            if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:
                if fused_gated:
                    return fused_silu_mul_fp8_quant_packed(
                        input=input,
                        output_q=output,
                        group_size=block_k,
                        clamp_limit=self.gemm1_clamp_limit,
                        alpha=self.gemm1_alpha,
                        beta=self.gemm1_beta,
                    )
                act_out = torch.empty(
                    (m_sum, activation_out_dim),
                    dtype=input.dtype,
                    device=input.device,
                )
                self.activation(activation, act_out, input)
                return per_token_group_quant_fp8_packed_for_deepgemm(
                    act_out, block_k, out_q=output
                )
            if fused_gated:
                use_ue8m0 = (
                    scale_fmt == DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0
                )
                return silu_mul_per_token_group_quant_fp8_colmajor(
                    input=input,
                    output=output,
                    use_ue8m0=use_ue8m0,
                    clamp_limit=self.gemm1_clamp_limit,
                    group_size=block_k,
                    alpha=self.gemm1_alpha,
                    beta=self.gemm1_beta,
                )
            act_out = torch.empty(
                (m_sum, activation_out_dim),
                dtype=input.dtype,
                device=input.device,
            )
            self.activation(activation, act_out, input)
            return per_token_group_quant_fp8(
                act_out,
                block_k,
                column_major_scales=True,
                out_q=output,
            )

    init_workspace_manager(torch.device("cuda"))
    experts = DeepGemmW4A8SwiGLUOAIExperts(
        moe_config=moe_config, quant_config=quant_config
    )
    kernel = mk.FusedMoEKernel(
        prepare_finalize=maybe_make_prepare_finalize(
            moe=moe_config,
            quant_config=quant_config,
            allow_new_interface=True,
            use_monolithic=False,
        ),
        fused_experts=experts,
    )
    return kernel, w13, w2, {
        "implementation": (
            f"{experts.__class__.__module__}.{experts.__class__.__qualname__}"
        ),
        "activation_precision": "dynamic FP8/K128 (W4A8)",
        "weight_precision": "native MXFP4 E2M1 + E8M0/K32",
        "activation": "swigluoai_uninterleave(alpha=1,beta=0,limit=10)",
        "conversion": conversion,
    }


def _deepgemm_launch(
    kernel: Any,
    w13: Any,
    w2: Any,
    shape: Any,
    x: Any,
    topk_ids: Any,
    topk_weights: Any,
) -> Any:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    def launch() -> Any:
        return kernel.apply(
            hidden_states=x,
            w1=w13,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=shape.num_experts,
            activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            apply_router_weight_on_input=False,
            expert_map=None,
        )

    return launch


def _paired_graph_timing(
    torch: Any,
    launches: Mapping[str, Any],
    *,
    warmup: int,
    iters: int,
    repeats: int,
) -> dict[str, Any]:
    rounds: dict[str, Any] = {}
    for order in (("deepgemm", "cutlass"), ("cutlass", "deepgemm")):
        label = f"{order[0]}_first"
        rounds[label] = {}
        for backend in order:
            rounds[label][backend] = kernel_bench.measure_cuda_events(
                torch,
                launches[backend],
                warmup=warmup,
                iters=iters,
                repeats=repeats,
                flush_l2=None,
            )
    combined: dict[str, Any] = {}
    for backend in ("deepgemm", "cutlass"):
        medians = [
            float(rounds[label][backend]["median_ms"]) for label in rounds
        ]
        combined[backend] = {
            "order_medians_ms": medians,
            "median_ms": statistics.median(medians),
        }
    combined["deepgemm_speedup_over_cutlass"] = (
        combined["cutlass"]["median_ms"] / combined["deepgemm"]["median_ms"]
    )
    return {"rounds": rounds, "combined": combined}


def run(args: argparse.Namespace) -> int:
    if tuple(args.m) != REQUIRED_M:
        raise RuntimeError(f"one-layer decision gate requires --m 1,4,24,48")
    os.environ["VLLM_USE_DEEP_GEMM"] = "1"
    os.environ["VLLM_USE_DEEP_GEMM_E8M0"] = "1"

    import torch
    from vllm.utils import deep_gemm as deep_gemm_utils

    # Resolve the installed DeepGEMM symbols before freezing the scale-format
    # oracle. Calling init_oracle_cache() first sees a null implementation and
    # incorrectly caches FLOAT32 for the lifetime of this probe process.
    deep_gemm_utils._lazy_init()
    if deep_gemm_utils.DeepGemmQuantScaleFMT.from_oracle().name != "UE8M0":
        raise RuntimeError(
            "DeepGEMM scale oracle did not select native packed UE8M0 on SM121"
        )
    capability = torch.cuda.get_device_capability()
    if capability != (12, 1):
        raise RuntimeError(f"expected SM121 GB10, observed capability={capability}")
    shape = kernel_bench.Dsv4Shape(tp_rank=args.tp_rank)
    tensors = prepared_bench._load_rank(torch, args.layer_file, args.tp_rank)
    collapse_bench.validate_prepared_contract(torch, tensors, shape)
    control_weights = _make_cutlass_weights(torch, tensors)
    cutlass_args = SimpleNamespace(m=args.m, swiglu_limit=args.swiglu_limit)
    cutlass_runner, cutlass_proof = kernel_bench._make_flashinfer_cutlass_runner(
        torch, control_weights, shape, cutlass_args
    )
    deepgemm, deep_w13, deep_w2, deepgemm_proof = _make_deepgemm_runner(
        torch,
        tensors,
        shape,
        args.m,
        swiglu_limit=args.swiglu_limit,
    )

    rows: dict[int, Any] = {}
    failures: list[dict[str, Any]] = []
    graphs: list[Any] = []
    for m in args.m:
        x, topk_ids, topk_weights = kernel_bench.make_routes(
            torch,
            shape,
            m,
            routing=args.routing,
            seed=args.seed + m,
            input_rms=args.input_rms,
        )
        cutlass_launch, _ = kernel_bench._make_flashinfer_cutlass_launch(
            torch,
            cutlass_runner,
            control_weights,
            shape,
            x,
            topk_ids,
            topk_weights,
        )
        deep_launch = _deepgemm_launch(
            deepgemm,
            deep_w13,
            deep_w2,
            shape,
            x,
            topk_ids,
            topk_weights,
        )
        cutlass_eager = cutlass_launch().clone()
        deep_eager = deep_launch().clone()
        torch.cuda.synchronize()
        numeric = kernel_bench.compare_tensors(torch, deep_eager, cutlass_eager)
        deep_activity = kernel_bench.tensor_activity(torch, deep_eager)
        cutlass_activity = kernel_bench.tensor_activity(torch, cutlass_eager)
        numeric_passed = bool(
            numeric["finite"]
            and numeric["nonzero_activity"]
            and float(numeric["cosine"]) >= args.numeric_min_cosine
            and float(numeric["normalized_rmse"]) <= args.numeric_max_nrmse
            and deep_activity["passed"]
            and cutlass_activity["passed"]
        )

        deep_graph, deep_graph_output, deep_graph_obj = kernel_bench.capture_graph(
            torch, deep_launch
        )
        cutlass_graph, cutlass_graph_output, cutlass_graph_obj = (
            kernel_bench.capture_graph(torch, cutlass_launch)
        )
        graphs.extend((deep_graph_obj, cutlass_graph_obj))
        deep_graph_value = deep_graph().clone()
        cutlass_graph_value = cutlass_graph().clone()
        torch.cuda.synchronize()
        deep_graph_compare = kernel_bench.compare_tensors(
            torch, deep_graph_value, deep_eager
        )
        cutlass_graph_compare = kernel_bench.compare_tensors(
            torch, cutlass_graph_value, cutlass_eager
        )
        graph_passed = bool(
            deep_graph_compare["finite"]
            and deep_graph_compare["normalized_rmse"] == 0.0
            and cutlass_graph_compare["finite"]
            and cutlass_graph_compare["normalized_rmse"] == 0.0
        )
        timing = _paired_graph_timing(
            torch,
            {"deepgemm": deep_graph, "cutlass": cutlass_graph},
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        row = {
            "m": m,
            "numeric": numeric,
            "numeric_passed": numeric_passed,
            "deepgemm_activity": deep_activity,
            "cutlass_activity": cutlass_activity,
            "deepgemm_graph_vs_eager": deep_graph_compare,
            "cutlass_graph_vs_eager": cutlass_graph_compare,
            "graph_passed": graph_passed,
            "timing": timing,
            "deepgemm_graph_ms": timing["combined"]["deepgemm"]["median_ms"],
            "cutlass_graph_ms": timing["combined"]["cutlass"]["median_ms"],
        }
        rows[m] = row
        if not numeric_passed:
            failures.append({"kind": "numeric", "m": m})
        if not graph_passed:
            failures.append({"kind": "graph", "m": m})
        print(
            f"M={m:>2} DeepGEMM-W4A8={row['deepgemm_graph_ms']:.6f} ms "
            f"CUTLASS-W4A4={row['cutlass_graph_ms']:.6f} ms "
            f"speedup={timing['combined']['deepgemm_speedup_over_cutlass']:.4f}x "
            f"cos={float(numeric['cosine']):.6f} "
            f"nrmse={float(numeric['normalized_rmse']):.6f}"
        )

    decision = evaluate_decision_rows(
        rows, minimum_speedup=args.minimum_decision_speedup
    )
    if not decision["passed"]:
        failures.append({"kind": "decision_speedup", "details": decision})
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "layer_file": str(args.layer_file.resolve()),
            "tp_rank": args.tp_rank,
        },
        "settings": {
            "m": list(args.m),
            "decision_m": list(DECISION_M),
            "minimum_decision_speedup": args.minimum_decision_speedup,
            "routing": args.routing,
            "seed": args.seed,
            "input_rms": args.input_rms,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
        },
        "backend_proof": {
            "deepgemm_w4a8": deepgemm_proof,
            "cutlass_w4a4": cutlass_proof,
        },
        "rows": [rows[m] for m in args.m],
        "decision": decision,
        "failures": failures,
        "passed": not failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")
    print("PASSED" if result["passed"] else f"FAILED: {len(failures)} gate(s)")
    return 0 if result["passed"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--m", type=_csv_positive_ints, default=REQUIRED_M)
    parser.add_argument("--routing", choices=("balanced", "hot", "random"), default="balanced")
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--input-rms", type=float, default=1.0)
    parser.add_argument("--swiglu-limit", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--numeric-min-cosine", type=float, default=DEFAULT_NUMERIC_MIN_COSINE
    )
    parser.add_argument(
        "--numeric-max-nrmse", type=float, default=DEFAULT_NUMERIC_MAX_NRMSE
    )
    parser.add_argument(
        "--minimum-decision-speedup",
        type=float,
        default=MINIMUM_DECISION_SPEEDUP,
    )
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
