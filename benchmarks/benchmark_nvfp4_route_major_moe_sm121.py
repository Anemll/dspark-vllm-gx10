#!/usr/bin/env python3
"""Real-layer route-major NVFP4 MoE prototype for SM121 decode.

This prototype keeps the official prepared W4A4 tensors and uses only
existing FlashInfer SM121 primitives.  It replaces the resident sliced-FC2
topology with two route-major grouped GEMMs:

* gather/pad routed token rows by physical expert;
* grouped W4A4 FC1;
* fused SwiGLU-OAI activation followed by batched NVFP4 quantization;
* grouped W4A4 FC2; and
* one atomics-free token reduction.

The path is decode-only (M=2..4), leaves M=1 and prefill unchanged, and is a
bounded feasibility gate rather than a serving integration.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_SCRIPT = Path(__file__).resolve()
REPO_ROOT = Path(
    os.environ.get(
        "DSPARK_REPO_ROOT",
        str(_SCRIPT.parents[1] if len(_SCRIPT.parents) > 1 else Path("/repo")),
    )
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import benchmark_nvfp4_a4w4_sm121 as kernel_bench
from benchmarks import benchmark_nvfp4_prepared_b12x_sm121 as prepared_bench


def _make_triton_kernels() -> dict[str, Any]:
    import triton
    import triton.language as tl

    @triton.jit
    def gather_padded(
        x_ptr,
        token_map_ptr,
        output_ptr,
        width: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        cols = block * BLOCK + tl.arange(0, BLOCK)
        valid_col = cols < width
        token = tl.load(token_map_ptr + row).to(tl.int32)
        value = tl.load(
            x_ptr + token * width + cols,
            mask=valid_col & (token >= 0),
            other=0.0,
        )
        tl.store(output_ptr + row * width + cols, value, mask=valid_col)

    @triton.jit
    def scatter_scale_tiles(
        source_ptr,
        destination_ptr,
        destination_base_ptr,
        tile_bytes: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        group = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < tile_bytes
        destination_base = tl.load(destination_base_ptr + group).to(tl.int64)
        values = tl.load(source_ptr + group * tile_bytes + offsets, mask=valid)
        tl.store(destination_ptr + destination_base + offsets, values, mask=valid)

    @triton.jit
    def swiglu_oai(
        fc1_ptr,
        output_ptr,
        intermediate_size: tl.constexpr,
        limit: tl.constexpr,
        alpha: tl.constexpr,
        beta: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        cols = block * BLOCK + tl.arange(0, BLOCK)
        valid = cols < intermediate_size
        base = row * (2 * intermediate_size)
        up = tl.load(fc1_ptr + base + cols, mask=valid, other=0.0).to(tl.float32)
        gate = tl.load(
            fc1_ptr + base + intermediate_size + cols,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        gate = tl.minimum(gate, limit)
        up = tl.maximum(tl.minimum(up, limit), -limit)
        sigmoid = 1.0 / (1.0 + tl.exp(-alpha * gate))
        activated = gate * sigmoid * (up + beta)
        tl.store(output_ptr + row * intermediate_size + cols, activated, mask=valid)

    @triton.jit
    def route_reduce(
        grouped_output_ptr,
        compact_expert_ptr,
        compact_row_ptr,
        topk_weights_ptr,
        local_row_base_ptr,
        output_ptr,
        hidden_size: tl.constexpr,
        top_k: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        token = tl.program_id(0)
        block = tl.program_id(1)
        cols = block * BLOCK + tl.arange(0, BLOCK)
        valid = cols < hidden_size
        accum = tl.zeros((BLOCK,), dtype=tl.float32)
        for route in range(top_k):
            pair = token * top_k + route
            local_expert = tl.load(compact_expert_ptr + pair).to(tl.int32)
            local_row = tl.load(compact_row_ptr + pair).to(tl.int32)
            physical_row = (
                tl.load(local_row_base_ptr + local_expert).to(tl.int32) + local_row
            )
            weight = tl.load(topk_weights_ptr + pair).to(tl.float32)
            value = tl.load(
                grouped_output_ptr + physical_row * hidden_size + cols,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            accum += weight * value
        tl.store(output_ptr + token * hidden_size + cols, accum, mask=valid)

    return {
        "gather": gather_padded,
        "scatter_scale": scatter_scale_tiles,
        "activation": swiglu_oai,
        "reduce": route_reduce,
        "triton": triton,
    }


def _build_static_route_plan(torch: Any, topk_ids: Any) -> dict[str, Any]:
    ids = [int(value) for value in topk_ids.detach().cpu().reshape(-1).tolist()]
    num_tokens, top_k = map(int, topk_ids.shape)
    active_experts = sorted(set(ids))
    local_index = {expert: index for index, expert in enumerate(active_experts)}
    rows_by_expert: dict[int, list[int]] = {expert: [] for expert in active_experts}
    compact_expert = []
    compact_row = []
    for pair, expert in enumerate(ids):
        token = pair // top_k
        compact_expert.append(local_index[expert])
        compact_row.append(len(rows_by_expert[expert]))
        rows_by_expert[expert].append(token)

    weight_experts = 256
    lengths = [0] * weight_experts
    for expert in active_experts:
        rows = len(rows_by_expert[expert])
        if not 1 <= rows <= num_tokens:
            raise RuntimeError(f"invalid row count for expert {expert}: {rows}")
        lengths[expert] = 4
    m_indptr = [0]
    for length in lengths:
        m_indptr.append(m_indptr[-1] + length)
    local_row_bases = [m_indptr[expert] for expert in active_experts]
    scale_row_bases = [
        ((m_indptr[expert] + expert * 127) // 128) * 128
        for expert in active_experts
    ]
    final_scale_rows = ((m_indptr[-2] + 255 * 127) // 128) * 128 + 128

    token_map = torch.full(
        (len(active_experts), 4), -1, dtype=torch.int32, device="cuda"
    )
    for local, expert in enumerate(active_experts):
        tokens = rows_by_expert[expert]
        token_map[local, : len(tokens)] = torch.tensor(
            tokens, dtype=torch.int32, device="cuda"
        )
    scale_base_bytes = torch.tensor(
        [row * (4096 // 16) for row in scale_row_bases],
        dtype=torch.int64,
        device="cuda",
    )
    # FC2 has fewer scale columns, so its destination byte bases differ.
    scale2_base_bytes = torch.tensor(
        [row * (1024 // 16) for row in scale_row_bases],
        dtype=torch.int64,
        device="cuda",
    )
    return {
        "active_experts": active_experts,
        "m_indptr": torch.tensor(m_indptr, dtype=torch.int32, device="cuda"),
        "local_row_bases": torch.tensor(
            local_row_bases, dtype=torch.int32, device="cuda"
        ),
        "scale_row_bases": scale_row_bases,
        "scale_rows": final_scale_rows,
        "scale1_base_bytes": scale_base_bytes,
        "scale2_base_bytes": scale2_base_bytes,
        "token_map": token_map,
        "compact_expert": torch.tensor(
            compact_expert, dtype=torch.int32, device="cuda"
        ),
        "compact_row": torch.tensor(compact_row, dtype=torch.int32, device="cuda"),
        "padded_rows": m_indptr[-1],
        "num_tokens": num_tokens,
        "top_k": top_k,
    }


class RouteMajorRunner:
    def __init__(
        self,
        torch: Any,
        flashinfer: Any,
        kernels: dict[str, Any],
        tensors: dict[str, Any],
        x: Any,
        topk_ids: Any,
        topk_weights: Any,
    ) -> None:
        self.torch = torch
        self.flashinfer = flashinfer
        self.kernels = kernels
        self.tensors = tensors
        self.x = x
        self.topk_weights = topk_weights.reshape(-1).contiguous().to(torch.float32)
        self.plan = _build_static_route_plan(torch, topk_ids)
        self.batch = len(self.plan["active_experts"])
        self.hidden = int(x.shape[1])
        self.intermediate = 1024
        self.padded_rows = int(self.plan["padded_rows"])
        self.active_tensor = torch.tensor(
            self.plan["active_experts"], dtype=torch.int64, device="cuda"
        )
        self.w13 = tensors["w13.weight"]
        self.w13_scale = tensors["w13.weight_scale"]
        self.w2 = tensors["w2.weight"]
        self.w2_scale = tensors["w2.weight_scale"]
        self.g1 = tensors["g1_alphas"].to(torch.float32)
        self.g2 = tensors["g2_alphas"].to(torch.float32)
        self.a1 = tensors["a1_gscale"][:1].to(torch.float32)
        self.a2 = tensors["a2_gscale"][:1].to(torch.float32)

        self.padded_input = torch.empty(
            self.batch, 4, self.hidden, dtype=torch.bfloat16, device="cuda"
        )
        self.fc1_output = torch.empty(
            self.padded_rows,
            2 * self.intermediate,
            dtype=torch.bfloat16,
            device="cuda",
        )
        self.activated = torch.empty(
            self.batch, 4, self.intermediate, dtype=torch.bfloat16, device="cuda"
        )
        self.fc2_output = torch.empty(
            self.padded_rows, self.hidden, dtype=torch.bfloat16, device="cuda"
        )
        self.output = torch.empty_like(x)
        self.fc1_scale_storage = torch.empty(
            self.plan["scale_rows"],
            self.hidden // 16,
            dtype=torch.uint8,
            device="cuda",
        )
        self.fc2_scale_storage = torch.empty(
            self.plan["scale_rows"],
            self.intermediate // 16,
            dtype=torch.uint8,
            device="cuda",
        )

    def gather(self) -> None:
        triton = self.kernels["triton"]
        self.kernels["gather"][(self.batch * 4, triton.cdiv(self.hidden, 256))](
            self.x,
            self.plan["token_map"],
            self.padded_input,
            width=self.hidden,
            BLOCK=256,
            num_warps=4,
        )

    def quantize_fc1_input(self) -> None:
        self.input_fp4, self.input_sf = self.flashinfer.nvfp4_batched_quantize(
            self.padded_input, self.a1
        )

    def scatter_fc1_scales(self) -> None:
        torch = self.torch
        triton = self.kernels["triton"]
        sf1_tile_bytes = 128 * (self.hidden // 16)
        self.kernels["scatter_scale"][(
            self.batch,
            triton.cdiv(sf1_tile_bytes, 256),
        )](
            self.input_sf.view(torch.uint8),
            self.fc1_scale_storage,
            self.plan["scale1_base_bytes"],
            tile_bytes=sf1_tile_bytes,
            BLOCK=256,
            num_warps=4,
        )

    def fc1(self) -> None:
        torch = self.torch
        self.flashinfer.gemm.group_gemm_nvfp4_nt_groupwise(
            self.input_fp4.view(torch.uint8).reshape(
                self.padded_rows, self.hidden // 2
            ),
            self.w13.view(torch.uint8),
            self.fc1_scale_storage,
            self.w13_scale.view(torch.uint8),
            self.plan["m_indptr"],
            alpha=self.g1,
            tile_m=128,
            tile_n=128,
            tile_k=128,
            swap_ab=True,
            out=self.fc1_output,
        )

    def activate(self) -> None:
        triton = self.kernels["triton"]
        self.kernels["activation"][(
            self.padded_rows,
            triton.cdiv(self.intermediate, 256),
        )](
            self.fc1_output,
            self.activated,
            intermediate_size=self.intermediate,
            limit=10.0,
            alpha=1.0,
            beta=0.0,
            BLOCK=256,
            num_warps=4,
        )

    def quantize_fc2_input(self) -> None:
        self.activated_fp4, self.activated_sf = self.flashinfer.nvfp4_batched_quantize(
            self.activated, self.a2
        )

    def scatter_fc2_scales(self) -> None:
        torch = self.torch
        triton = self.kernels["triton"]
        sf2_tile_bytes = 128 * (self.intermediate // 16)
        self.kernels["scatter_scale"][(
            self.batch,
            triton.cdiv(sf2_tile_bytes, 256),
        )](
            self.activated_sf.view(torch.uint8),
            self.fc2_scale_storage,
            self.plan["scale2_base_bytes"],
            tile_bytes=sf2_tile_bytes,
            BLOCK=256,
            num_warps=4,
        )

    def fc2(self) -> None:
        torch = self.torch
        self.flashinfer.gemm.group_gemm_nvfp4_nt_groupwise(
            self.activated_fp4.view(torch.uint8).reshape(
                self.padded_rows, self.intermediate // 2
            ),
            self.w2.view(torch.uint8),
            self.fc2_scale_storage,
            self.w2_scale.view(torch.uint8),
            self.plan["m_indptr"],
            alpha=self.g2,
            tile_m=128,
            tile_n=128,
            tile_k=128,
            swap_ab=True,
            out=self.fc2_output,
        )

    def reduce(self) -> None:
        triton = self.kernels["triton"]
        self.kernels["reduce"][(
            self.plan["num_tokens"],
            triton.cdiv(self.hidden, 256),
        )](
            self.fc2_output,
            self.plan["compact_expert"],
            self.plan["compact_row"],
            self.topk_weights,
            self.plan["local_row_bases"],
            self.output,
            hidden_size=self.hidden,
            top_k=self.plan["top_k"],
            BLOCK=256,
            num_warps=4,
        )

    def launch(self) -> Any:
        self.gather()
        self.quantize_fc1_input()
        self.scatter_fc1_scales()
        self.fc1()
        self.activate()
        self.quantize_fc2_input()
        self.scatter_fc2_scales()
        self.fc2()
        self.reduce()
        return self.output


def _measure(torch: Any, launch: Any, warmup: int, iters: int, repeats: int) -> dict[str, Any]:
    repeat_medians = []
    samples = []
    for _ in range(repeats):
        for _ in range(warmup):
            launch()
        torch.cuda.synchronize()
        current = []
        for _ in range(iters):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            launch()
            end.record()
            end.synchronize()
            current.append(float(begin.elapsed_time(end)))
        samples.extend(current)
        repeat_medians.append(statistics.median(current))
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "repeat_median_ms": repeat_medians,
        "samples": len(samples),
    }


def run(args: argparse.Namespace) -> int:
    import flashinfer
    import torch

    if not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability()) != (12, 1):
        raise RuntimeError("route-major MoE benchmark requires one SM121 GPU")
    if args.m not in (2, 4):
        raise ValueError("prototype is intentionally limited to M=2 or M=4")
    shape = kernel_bench.Dsv4Shape(tp_rank=args.tp_rank)
    tensors = prepared_bench._load_rank(torch, args.layer_file, args.tp_rank)
    weights = prepared_bench._prepare_weights(torch, tensors, shape)
    runner_args = SimpleNamespace(
        m=(args.m,), swiglu_alpha=1.0, swiglu_beta=0.0, swiglu_limit=10.0
    )
    accepted_wrapper, accepted_proof = kernel_bench._make_w4a4_runner(
        torch, weights, shape, runner_args
    )
    accepted_arena = accepted_wrapper._moe_output
    kernels = _make_triton_kernels()

    x, topk_ids, topk_weights = kernel_bench.make_routes(
        torch,
        shape,
        args.m,
        routing=args.routing,
        seed=args.seed,
        input_rms=1.0,
    )
    accepted_launch, accepted_output = prepared_bench._b12x_launch(
        torch,
        accepted_wrapper,
        accepted_arena,
        weights,
        x,
        topk_ids,
        topk_weights,
        direct_output=True,
    )
    route_runner = RouteMajorRunner(
        torch, flashinfer, kernels, tensors, x, topk_ids, topk_weights
    )
    route_output = route_runner.launch()
    accepted_launch()
    torch.cuda.synchronize()
    numeric = kernel_bench.compare_tensors(torch, route_output, accepted_output)
    numeric_passed = kernel_bench.numeric_metrics_pass(
        numeric,
        min_cosine=args.numeric_min_cosine,
        max_normalized_rmse=args.numeric_max_nrmse,
    )
    activity = kernel_bench.tensor_activity(torch, route_output)

    route_timing = _measure(
        torch, route_runner.launch, args.warmup, args.iters, args.repeats
    )
    accepted_timing = _measure(
        torch, accepted_launch, args.warmup, args.iters, args.repeats
    )
    component_methods = {
        "gather": route_runner.gather,
        "quantize_fc1_input": route_runner.quantize_fc1_input,
        "scatter_fc1_scales": route_runner.scatter_fc1_scales,
        "grouped_fc1": route_runner.fc1,
        "swiglu_oai": route_runner.activate,
        "quantize_fc2_input": route_runner.quantize_fc2_input,
        "scatter_fc2_scales": route_runner.scatter_fc2_scales,
        "grouped_fc2": route_runner.fc2,
        "route_reduce": route_runner.reduce,
    }
    component_timing = {
        name: _measure(torch, method, 2, max(10, args.iters), 1)
        for name, method in component_methods.items()
    }
    speedup = accepted_timing["median_ms"] / route_timing["median_ms"]
    report = {
        "probe": "nvfp4_route_major_moe_sm121",
        "scope": "decode-only M=2..4 prototype; M1 and prefill unchanged",
        "checkpoint": str(args.layer_file),
        "tp_rank": args.tp_rank,
        "m": args.m,
        "routing": args.routing,
        "numeric": numeric,
        "numeric_passed": numeric_passed,
        "activity": activity,
        "route_plan": {
            "active_experts": route_runner.plan["active_experts"],
            "padded_rows": route_runner.padded_rows,
            "actual_rows": args.m * shape.top_k,
            "scale_rows": route_runner.plan["scale_rows"],
        },
        "route_major": route_timing,
        "route_major_components": component_timing,
        "accepted_fused": accepted_timing,
        "speedup_route_major_over_accepted": speedup,
        "accepted_backend_proof": accepted_proof,
    }
    report["passed"] = bool(
        numeric_passed
        and activity["passed"]
        and speedup >= args.minimum_speedup
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument(
        "--routing", choices=("balanced", "random", "hot"), default="balanced"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--numeric-min-cosine", type=float, default=0.98)
    parser.add_argument("--numeric-max-nrmse", type=float, default=0.25)
    parser.add_argument("--minimum-speedup", type=float, default=1.0)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
