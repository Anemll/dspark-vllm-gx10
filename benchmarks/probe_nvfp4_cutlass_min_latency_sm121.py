#!/usr/bin/env python3
"""Probe FlashInfer's native FP4 minimum-latency MoE path on SM121.

FlashInfer 0.6.15 exposes the native implementation but its public wrapper
unconditionally rejects ``min_latency_mode`` on Blackwell.  This diagnostic
calls the generated SM121 module directly, reconstructs the token-major MoE
output from the expert-major result, and compares it with the serving path.
It never mutates an autotune cache or checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

from benchmarks import benchmark_nvfp4_a4w4_sm121 as kernel_bench
from benchmarks.benchmark_nvfp4_cutlass_tactic_sweep_sm121 import (
    _load_prepared_cutlass_weights,
)


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return values[index]


def run(args: argparse.Namespace) -> int:
    import torch
    import flashinfer.fused_moe.core as flashinfer_moe
    from flashinfer.fused_moe.core import ActivationType
    from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

    if tuple(torch.cuda.get_device_capability()) != (12, 1):
        raise RuntimeError("probe requires one SM121 GPU")
    raw_module = None
    if args.module_so is not None:
        import tvm_ffi

        raw_module = tvm_ffi.load_module(str(args.module_so))

        class _PrebuiltModuleSpec:
            def build_and_load(self):
                return raw_module

        flashinfer_moe.get_cutlass_fused_moe_module.cache_clear()
        flashinfer_moe.gen_cutlass_fused_moe_sm120_module = (
            lambda use_fast_build=False: _PrebuiltModuleSpec()
        )
    shape = kernel_bench.Dsv4Shape(tp_rank=args.tp_rank)
    shape.validate()
    weights, prepared_proof = _load_prepared_cutlass_weights(
        torch, args.layer_file, args.tp_rank
    )
    runner, backend_proof = kernel_bench._make_flashinfer_cutlass_runner(
        torch,
        weights,
        shape,
        SimpleNamespace(
            m=(args.m,), swiglu_alpha=1.0, swiglu_beta=0.0, swiglu_limit=10.0
        ),
    )
    x, topk_ids, topk_weights = kernel_bench.make_routes(
        torch,
        shape,
        args.m,
        routing="balanced",
        seed=args.seed,
        input_rms=1.0,
    )
    if args.route_ids_npy:
        topk_ids = kernel_bench.load_captured_route_ids(
            torch,
            args.route_ids_npy,
            sample_index=args.route_sample_index,
            m=args.m,
            top_k=shape.top_k,
        )

    standard_launch, _ = kernel_bench._make_flashinfer_cutlass_launch(
        torch, runner, weights, shape, x, topk_ids, topk_weights
    )
    reference = standard_launch().clone()
    torch.cuda.synchronize()

    quant_config = runner.experts.quant_config
    a1q, a1q_scale = moe_kernel_quantize_input(
        x,
        quant_config.a1_gscale,
        quant_dtype=quant_config.quant_dtype,
        per_act_token_quant=quant_config.per_act_token_quant,
        block_shape=quant_config.block_shape,
        is_scale_swizzled=quant_config.is_scale_swizzled,
        mx_alignment=quant_config.mx_alignment,
    )
    quant_scales = [
        runner.experts.a1_gscale,
        runner.experts.w1_scale.view(torch.int32),
        runner.experts.g1_alphas,
        runner.experts.a2_gscale,
        runner.experts.w2_scale.view(torch.int32),
        runner.experts.g2_alphas,
    ]
    if args.split_streams:
        if args.m < 2:
            raise RuntimeError("--split-streams requires M >= 2")
        if args.m % args.split_chunk_size != 0:
            raise RuntimeError("M must be divisible by --split-chunk-size")
        chunks = [
            (start, start + args.split_chunk_size)
            for start in range(0, args.m, args.split_chunk_size)
        ]
        split_inputs = []
        for start, end in chunks:
            row_q, row_sf = moe_kernel_quantize_input(
                x[start:end],
                quant_config.a1_gscale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape,
                is_scale_swizzled=quant_config.is_scale_swizzled,
                mx_alignment=quant_config.mx_alignment,
            )
            split_inputs.append((row_q, row_sf))
        split_runners = [
            raw_module.init(
                split_inputs[0][0].dtype,
                weights.w13.view(torch.long).dtype,
                torch.bfloat16,
                False,
                False,
                False,
                False,
                True,
            )
            for _ in chunks
        ]
        streams = [torch.cuda.Stream() for _ in chunks]
        outputs = [
            torch.empty(
                (args.split_chunk_size, shape.hidden_size),
                dtype=torch.bfloat16,
                device=x.device,
            )
            for _ in chunks
        ]

        def launch_split() -> None:
            for chunk_index, (start, end) in enumerate(chunks):
                stream = streams[chunk_index]
                with torch.cuda.stream(stream):
                    split_runners[chunk_index].run_moe(
                        outputs[chunk_index],
                        split_inputs[chunk_index][0],
                        topk_ids[start:end].to(torch.int32),
                        topk_weights[start:end],
                        weights.w13.view(torch.long),
                        None,
                        weights.w2.view(torch.long),
                        None,
                        quant_scales,
                        split_inputs[chunk_index][1],
                        None,
                        None,
                        runner.experts.gemm1_clamp_limit,
                        True,
                        2,
                        args.tp_rank,
                        1,
                        0,
                        1,
                        0,
                        False,
                        False,
                        [args.split_gemm1_tactic, args.split_gemm2_tactic],
                        True,
                        ActivationType.Swiglu,
                    )

        for _ in range(args.warmup):
            launch_split()
        torch.cuda.synchronize()
        split_samples = []
        default_stream = torch.cuda.current_stream()
        for _ in range(args.iters):
            start = torch.cuda.Event(enable_timing=True)
            done = [torch.cuda.Event() for _ in streams]
            end = torch.cuda.Event(enable_timing=True)
            start.record(default_stream)
            for stream in streams:
                stream.wait_event(start)
            launch_split()
            for stream, event in zip(streams, done, strict=True):
                event.record(stream)
                default_stream.wait_event(event)
            end.record(default_stream)
            end.synchronize()
            split_samples.append(float(start.elapsed_time(end)))
        reconstructed = torch.cat(outputs, dim=0)
        numeric = kernel_bench.compare_tensors(torch, reconstructed, reference)
        activity = kernel_bench.tensor_activity(torch, reconstructed)
        report = {
            "probe": "nvfp4_cutlass_split_stream_sm121",
            "capability": list(torch.cuda.get_device_capability()),
            "shape": {"m": args.m, "top_k": shape.top_k, "tp_rank": args.tp_rank},
            "prepared_proof": prepared_proof,
            "backend_proof": backend_proof,
            "streams": len(chunks),
            "chunk_size": args.split_chunk_size,
            "tactics": {
                "gemm1": args.split_gemm1_tactic,
                "gemm2": args.split_gemm2_tactic,
            },
            "numeric": numeric,
            "activity": activity,
            "timing_ms": {
                "median": _percentile(split_samples, 0.5),
                "p95": _percentile(split_samples, 0.95),
                "samples": split_samples,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({
            "cosine": numeric["cosine"],
            "nrmse": numeric["normalized_rmse"],
            "median_ms": report["timing_ms"]["median"],
            "output": str(args.output),
        }, sort_keys=True))
        return 0 if activity["passed"] and math.isfinite(float(numeric["cosine"])) else 2
    autotuned_tactics = None
    if args.autotune_low_latency:
        from flashinfer.autotuner import AutoTuner, autotune

        module = flashinfer_moe.get_cutlass_fused_moe_module("121")
        tuner = AutoTuner.get()
        before = set(tuner.profiling_cache)
        with autotune(tune_mode=True, tuning_buckets=(args.m,)):
            module.cutlass_fused_moe(
                torch.empty(
                    (args.m * shape.num_experts, shape.hidden_size),
                    dtype=torch.bfloat16,
                    device=x.device,
                ),
                a1q,
                topk_ids.to(torch.int32),
                topk_weights,
                weights.w13.view(torch.long),
                None,
                weights.w2.view(torch.long),
                None,
                torch.bfloat16,
                quant_scales,
                a1q_scale,
                None,
                None,
                runner.experts.gemm1_clamp_limit,
                True,
                2,
                args.tp_rank,
                1,
                0,
                1,
                0,
                enable_alltoall=False,
                use_deepseek_fp8_block_scale=False,
                use_w4_group_scaling=False,
                use_mxfp8_act_scaling=False,
                min_latency_mode=True,
                tune_max_num_tokens=args.m,
                enable_pdl=True,
                activation_type=ActivationType.Swiglu,
                use_packed_weights=False,
                use_fused_finalize=False,
            )
        selected = {}
        for key, value in tuner.profiling_cache.items():
            if key in before:
                continue
            op = getattr(key, "custom_op", "")
            print(f"LOW_LATENCY_CACHE_ENTRY={key!r} VALUE={value!r}", flush=True)
            if op in ("trtllm::fused_moe::gemm1", "trtllm::fused_moe::gemm2"):
                # FlashInfer stores (runner_id, tactic, optimization_profile).
                selected[op] = int(value[1])
        if set(selected) != {
            "trtllm::fused_moe::gemm1",
            "trtllm::fused_moe::gemm2",
        }:
            raise RuntimeError(f"missing low-latency autotune choices: {selected}")
        args.gemm1_tactic = selected["trtllm::fused_moe::gemm1"]
        args.gemm2_tactic = selected["trtllm::fused_moe::gemm2"]
        autotuned_tactics = {
            "gemm1": args.gemm1_tactic,
            "gemm2": args.gemm2_tactic,
        }
        print(f"LOW_LATENCY_AUTOTUNE={autotuned_tactics}", flush=True)
    if raw_module is None:
        raise RuntimeError("--module-so is required for the restored native path")
    raw_runner = raw_module.init(
        a1q.dtype,
        weights.w13.view(torch.long).dtype,
        torch.bfloat16,
        False,
        False,
        False,
        False,
        False,
    )

    def launch_min_latency() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output = torch.empty(
            (args.m * shape.num_experts, shape.hidden_size),
            dtype=torch.bfloat16,
            device=x.device,
        )
        num_active = torch.empty((1,), dtype=torch.int32, device=x.device)
        scores = torch.empty(
            (shape.num_experts, args.m), dtype=torch.float32, device=x.device
        )
        active_ids = torch.empty(
            (shape.num_experts,), dtype=torch.int32, device=x.device
        )
        raw_runner.run_moe_min_latency(
            output,
            a1q,
            topk_ids.to(torch.int32),
            topk_weights,
            weights.w13.view(torch.long),
            None,
            weights.w2.view(torch.long),
            None,
            quant_scales,
            a1q_scale,
            None,
            None,
            runner.experts.gemm1_clamp_limit,
            True,
            num_active,
            scores,
            active_ids,
            2,
            args.tp_rank,
            1,
            0,
            1,
            0,
            False,
            True,
            [args.gemm1_tactic, args.gemm2_tactic],
            True,
            ActivationType.Swiglu,
        )
        return output, num_active, scores, active_ids

    expanded, num_active, scores, active_ids = launch_min_latency()
    torch.cuda.synchronize()
    active = int(num_active.item())
    if not (1 <= active <= shape.num_experts):
        raise RuntimeError(f"invalid active expert count: {active}")
    # SM120 swaps A/B for the TMA warp-specialized kernels.  The unfused
    # low-latency D buffer is therefore [expert, hidden, token], even though
    # the public logical shape is [expert, token, hidden].
    expert_outputs = expanded.view(
        shape.num_experts, shape.hidden_size, args.m
    ).transpose(1, 2)
    active_scores = scores[:active, :, None]
    masked_outputs = torch.where(
        active_scores != 0,
        expert_outputs[:active].float(),
        torch.zeros((), dtype=torch.float32, device=x.device),
    )
    reconstructed = (
        masked_outputs * active_scores
    ).sum(dim=0).to(reference.dtype)
    numeric = kernel_bench.compare_tensors(torch, reconstructed, reference)
    activity = kernel_bench.tensor_activity(torch, reconstructed)

    samples: list[float] = []
    for _ in range(args.warmup):
        launch_min_latency()
    torch.cuda.synchronize()
    for _ in range(args.iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch_min_latency()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))

    report = {
        "probe": "nvfp4_cutlass_min_latency_sm121",
        "capability": list(torch.cuda.get_device_capability()),
        "shape": {"m": args.m, "top_k": shape.top_k, "tp_rank": args.tp_rank},
        "prepared_proof": prepared_proof,
        "backend_proof": backend_proof,
        "autotuned_tactics": autotuned_tactics,
        "active_experts": active,
        "active_expert_ids": active_ids[:active].cpu().tolist(),
        "active_score_nonzero_per_token": (scores[:active] != 0)
        .sum(dim=0)
        .cpu()
        .tolist(),
        "active_score_finite_per_token": torch.isfinite(scores[:active])
        .sum(dim=0)
        .cpu()
        .tolist(),
        "selected_output_nonfinite_per_token": (
            (~torch.isfinite(expert_outputs[:active]))
            & (active_scores != 0)
        )
        .sum(dim=(0, 2))
        .cpu()
        .tolist(),
        "reconstructed_nonfinite_per_token": (~torch.isfinite(reconstructed))
        .sum(dim=1)
        .cpu()
        .tolist(),
        "reconstructed_nonzero_per_token": (reconstructed != 0)
        .sum(dim=1)
        .cpu()
        .tolist(),
        "numeric": numeric,
        "activity": activity,
        "timing_ms": {
            "median": _percentile(samples, 0.5),
            "p95": _percentile(samples, 0.95),
            "samples": samples,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "active_experts": active,
        "cosine": numeric["cosine"],
        "nrmse": numeric["normalized_rmse"],
        "median_ms": report["timing_ms"]["median"],
        "output": str(args.output),
    }, sort_keys=True))
    if not activity["passed"] or not math.isfinite(float(numeric["cosine"])):
        return 2
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-file", type=Path, required=True)
    parser.add_argument("--module-so", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--route-ids-npy", type=Path)
    parser.add_argument("--route-sample-index", type=int, default=131)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--gemm1-tactic", type=int, default=16)
    parser.add_argument("--gemm2-tactic", type=int, default=58)
    parser.add_argument("--autotune-low-latency", action="store_true")
    parser.add_argument("--split-streams", action="store_true")
    parser.add_argument("--split-chunk-size", type=int, default=1)
    parser.add_argument("--split-gemm1-tactic", type=int, default=16)
    parser.add_argument("--split-gemm2-tactic", type=int, default=47)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
