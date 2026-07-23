#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Compare prepared NVFP4 and abliterated MXFP4 through exact production B12X.

Both arms enter the same native MXFP4 E2M1 + E8M0/K32, BF16-activation
(``w4a16``) B12X path.  The prepared arm losslessly reverses the checkpoint's
E8M0/K32 -> E4M3/K16 scale expansion and restores production's W13
``[w1/gate, w3/up]`` order.  The control arm reads the native abliterated
checkpoint and applies the same TP=2 slicing used by serving.

This is a bounded one-layer component comparison, not a full model load.
Different checkpoints contain different weight values, so correctness is
proved per arm (finite/non-zero eager output and graph/eager identity), while
the paired timing gate asks whether identical layouts and kernels have parity.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import benchmark_nvfp4_a4w4_sm121 as kernel_bench
from benchmarks import benchmark_nvfp4_prepared_b12x_sm121 as prepared_bench
from benchmarks import benchmark_nvfp4_prepared_deepgemm_w4a8_sm121 as collapse_source
from benchmarks import benchmark_nvfp4_prepared_mxfp4_requant_sm121 as collapse_bench


SCHEMA_VERSION = 1
REQUIRED_M = (1, 4, 24, 48)
DECISION_M = (24, 48)
MAXIMUM_ABSOLUTE_DECISION_DELTA = 0.03
ABLITERATED_LAYER_PREFIX = "layers.0.ffn.experts."
ABLITERATED_FAMILIES = (
    "w1.weight",
    "w1.scale",
    "w3.weight",
    "w3.scale",
    "w2.weight",
    "w2.scale",
)


def _csv_positive_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("M values must be positive")
    return result


def tp_slice_native_expert(
    value: Any,
    *,
    family: str,
    tp_rank: int,
    tp_size: int = 2,
) -> Any:
    """Apply the production TP=2 slice to one native expert tensor."""

    if tp_rank not in range(tp_size):
        raise ValueError(f"invalid TP rank {tp_rank} for TP size {tp_size}")
    if family in ("w1.weight", "w3.weight", "w1.scale", "w3.scale"):
        if value.shape[0] % tp_size:
            raise ValueError(f"{family} output dimension is not TP divisible")
        rows = value.shape[0] // tp_size
        return value.narrow(0, tp_rank * rows, rows).contiguous()
    if family in ("w2.weight", "w2.scale"):
        if value.shape[-1] % tp_size:
            raise ValueError(f"{family} input dimension is not TP divisible")
        cols = value.shape[-1] // tp_size
        return value.narrow(-1, tp_rank * cols, cols).contiguous()
    raise ValueError(f"unknown native expert family: {family}")


def evaluate_parity(
    rows: Mapping[int, Mapping[str, Any]],
    *,
    maximum_absolute_delta: float = MAXIMUM_ABSOLUTE_DECISION_DELTA,
) -> dict[str, Any]:
    missing = sorted(set(REQUIRED_M) - set(rows))
    if missing:
        raise ValueError(f"missing required M rows: {missing}")
    decision: dict[str, Any] = {}
    passed = True
    for m in DECISION_M:
        row = rows[m]
        control_ms = float(row["abliterated_graph_ms"])
        converted_ms = float(row["converted_graph_ms"])
        delta = converted_ms / control_ms - 1.0
        row_passed = bool(
            row["graph_passed"]
            and row["activity_passed"]
            and abs(delta) <= maximum_absolute_delta + 1e-12
        )
        decision[str(m)] = {
            "abliterated_graph_ms": control_ms,
            "converted_graph_ms": converted_ms,
            "converted_delta": delta,
            "maximum_absolute_delta": maximum_absolute_delta,
            "passed": row_passed,
        }
        passed = passed and row_passed
    return {"shapes": decision, "passed": passed}


def _as_packed_u8(torch: Any, value: Any) -> Any:
    if value.element_size() != 1:
        raise TypeError(f"packed FP4/E8M0 tensor must be byte-sized, got {value.dtype}")
    return value if value.dtype == torch.uint8 else value.view(torch.uint8)


def _load_abliterated_rank(
    torch: Any,
    shard: Path,
    *,
    tp_rank: int,
    num_experts: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from safetensors import safe_open

    by_family: dict[str, list[Any]] = {family: [] for family in ABLITERATED_FAMILIES}
    shapes: dict[str, list[int]] = {}
    dtypes: dict[str, str] = {}
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        expected = {
            f"{ABLITERATED_LAYER_PREFIX}{expert}.{family}"
            for expert in range(num_experts)
            for family in ABLITERATED_FAMILIES
        }
        missing = sorted(expected - keys)
        if missing:
            raise RuntimeError(f"abliterated layer tensor(s) missing: {missing[:4]}")
        for expert in range(num_experts):
            for family in ABLITERATED_FAMILIES:
                name = f"{ABLITERATED_LAYER_PREFIX}{expert}.{family}"
                source = handle.get_tensor(name)
                shapes.setdefault(family, list(source.shape))
                dtypes.setdefault(family, str(source.dtype))
                if list(source.shape) != shapes[family] or str(source.dtype) != dtypes[family]:
                    raise RuntimeError(f"abliterated {family} contract drift at expert {expert}")
                sliced = tp_slice_native_expert(
                    source, family=family, tp_rank=tp_rank
                )
                by_family[family].append(_as_packed_u8(torch, sliced))

    stacked = {
        family: torch.stack(values, dim=0).contiguous()
        for family, values in by_family.items()
    }
    # Native production B12X consumes W13 as [w1/gate, w3/up].
    result = {
        "w13": torch.cat(
            (stacked["w1.weight"], stacked["w3.weight"]), dim=1
        ).to("cuda"),
        "w13_scale": torch.cat(
            (stacked["w1.scale"], stacked["w3.scale"]), dim=1
        ).to("cuda"),
        "w2": stacked["w2.weight"].to("cuda"),
        "w2_scale": stacked["w2.scale"].to("cuda"),
    }
    torch.cuda.synchronize()
    proof = {
        "source": str(shard.resolve()),
        "source_format": "fp4_e8m0_k32",
        "source_w13_layout": "separate-w1-w3",
        "runtime_w13_layout": "w31",
        "source_shapes": shapes,
        "source_dtypes": dtypes,
        "tp_rank": tp_rank,
        "tp_size": 2,
        "num_experts": num_experts,
        "tensor_reads": num_experts * len(ABLITERATED_FAMILIES),
    }
    return result, proof


def _load_converted_rank(
    torch: Any,
    layer_file: Path,
    *,
    tp_rank: int,
    shape: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from b12x.moe.fused.w4a16.host import unswizzle_expert_scales

    tensors = prepared_bench._load_rank(torch, layer_file, tp_rank)
    collapse_bench.validate_prepared_contract(torch, tensors, shape)
    raw_g1 = collapse_bench.recover_raw_global_scale(
        tensors["g1_alphas"], tensors["a1_gscale"]
    ).to(torch.float32)
    raw_g2 = collapse_bench.recover_raw_global_scale(
        tensors["g2_alphas"], tensors["a2_gscale"]
    ).to(torch.float32)
    w13_scale, w13_proof = collapse_bench.collapse_nvfp4_scale_grid(
        torch,
        tensors["w13.weight_scale"],
        raw_g1,
        rows=2 * shape.intermediate_size_per_rank,
        cols=shape.hidden_size,
        name="w13",
        unswizzle_expert_scales=unswizzle_expert_scales,
    )
    w2_scale, w2_proof = collapse_bench.collapse_nvfp4_scale_grid(
        torch,
        tensors["w2.weight_scale"],
        raw_g2,
        rows=shape.hidden_size,
        cols=shape.intermediate_size_per_rank,
        name="w2",
        unswizzle_expert_scales=unswizzle_expert_scales,
    )
    # Prepared CUTLASS stores [w3/up, w1/gate]. Native B12X uses [w1,w3].
    result = {
        "w13": collapse_source.swap_gate_up_halves(tensors["w13.weight"]),
        "w13_scale": collapse_source.swap_gate_up_halves(w13_scale),
        "w2": tensors["w2.weight"],
        "w2_scale": w2_scale,
    }
    torch.cuda.synchronize()
    proof = {
        "source": str(layer_file.resolve()),
        "source_format": "prepared-modelopt-nvfp4",
        "runtime_source_format": "fp4_e8m0_k32",
        "source_w13_layout": "up-gate",
        "runtime_w13_layout": "w31",
        "payload_transform": "projection-half swap only; FP4 bytes unchanged",
        "scale_transform": "exact E4M3/K16 pair collapse to E8M0/K32",
        "w13_collapse": w13_proof,
        "w2_collapse": w2_proof,
        "tp_rank": tp_rank,
        "tp_size": 2,
    }
    return result, proof


def _make_exact_b12x_runner(
    torch: Any,
    tensors: Mapping[str, Any],
    *,
    max_tokens: int,
    top_k: int,
    swiglu_limit: float,
) -> tuple[Any, dict[str, Any]]:
    from vllm.model_executor.layers.fused_moe.experts import b12x_mxfp4_moe as b12x

    num_experts = int(tensors["w13"].shape[0])
    hidden_size = int(tensors["w2"].shape[1])
    intermediate_size = int(tensors["w2"].shape[-1]) * 2
    unit = torch.ones(num_experts, dtype=torch.float32, device="cuda")
    prepared = b12x._prepare_b12x_fp4_moe_weights(
        source_format="fp4_e8m0_k32",
        w13_layout="w31",
        w1_fp4=tensors["w13"],
        w1_blockscale=tensors["w13_scale"],
        w1_global_scale=unit,
        a1_gscale=unit,
        w2_fp4=tensors["w2"],
        w2_blockscale=tensors["w2_scale"],
        w2_global_scale=unit,
        a2_gscale=unit,
        activation="silu",
        params_dtype=torch.bfloat16,
        prepare_runtime_alphas=False,
        prepare_w4a16=True,
        reuse_input_storage=True,
    )
    prepared_w4a16 = prepared.w4a16
    if prepared_w4a16 is None:
        raise RuntimeError("exact production B12X W4A16 preparation returned None")
    b12x._prewarm_b12x_route_pack(
        device=torch.device("cuda:0"),
        num_experts=num_experts,
        topk=top_k,
        max_tokens=max_tokens,
    )
    torch.cuda.synchronize()
    proof = {
        "implementation": (
            "vllm.model_executor.layers.fused_moe.experts."
            "b12x_mxfp4_moe._run_b12x_moe_fp4"
        ),
        "preparation": (
            "vllm.model_executor.layers.fused_moe.experts."
            "b12x_mxfp4_moe._prepare_b12x_fp4_moe_weights"
        ),
        "quant_mode": "w4a16",
        "activation_precision": "BF16",
        "weight_precision": "native MXFP4 E2M1 + E8M0/K32",
        "source_format": "fp4_e8m0_k32",
        "w13_layout": "w31",
        "activation": "silu",
        "swiglu_limit": swiglu_limit,
        "num_experts": num_experts,
        "hidden_size": hidden_size,
        "intermediate_size_per_rank": intermediate_size,
    }
    return (
        {
            "module": b12x,
            "tensors": tensors,
            "prepared_w4a16": prepared_w4a16,
            "unit": unit,
            "num_experts": num_experts,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "top_k": top_k,
            "swiglu_limit": swiglu_limit,
        },
        proof,
    )


def _make_launch(
    torch: Any,
    runner: Mapping[str, Any],
    x: Any,
    topk_ids: Any,
    topk_weights: Any,
) -> tuple[Any, Any, Any]:
    b12x = runner["module"]
    output = torch.empty_like(x)
    plan = b12x._plan_b12x_moe_fp4_scratch(
        tokens=int(x.shape[0]),
        weight_E=runner["num_experts"],
        k=runner["hidden_size"],
        n=runner["intermediate_size"],
        topk=runner["top_k"],
        device=x.device,
        dtype=x.dtype,
        activation="silu",
        quant_mode="w4a16",
        source_format="fp4_e8m0_k32",
        w13_layout="w31",
        swiglu_limit=runner["swiglu_limit"],
    )
    scratch = torch.empty(
        b12x._b12x_scratch_nbytes(plan), dtype=torch.uint8, device=x.device
    )
    tensors = runner["tensors"]
    unit = runner["unit"]

    def launch() -> Any:
        b12x._run_b12x_moe_fp4(
            a=x,
            a1_gscale=unit,
            w1_fp4=tensors["w13"],
            w1_blockscale=tensors["w13_scale"],
            w1_alphas=unit,
            a2_gscale=unit,
            w2_fp4=tensors["w2"],
            w2_blockscale=tensors["w2_scale"],
            w2_alphas=unit,
            output=output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=False,
            input_scales_are_reciprocal=True,
            input_scales_static=True,
            activation="silu",
            quant_mode="w4a16",
            unit_scale_contract=True,
            source_format="fp4_e8m0_k32",
            w13_layout="w31",
            prepared_w4a16=runner["prepared_w4a16"],
            swiglu_limit=runner["swiglu_limit"],
            plan=plan,
            scratch=scratch,
        )
        return output

    return launch, output, scratch


def _paired_graph_timing(
    torch: Any,
    launches: Mapping[str, Any],
    *,
    warmup: int,
    iters: int,
    repeats: int,
) -> dict[str, Any]:
    rounds: dict[str, Any] = {}
    for order in (("converted", "abliterated"), ("abliterated", "converted")):
        label = f"{order[0]}_first"
        rounds[label] = {}
        for arm in order:
            rounds[label][arm] = kernel_bench.measure_cuda_events(
                torch,
                launches[arm],
                warmup=warmup,
                iters=iters,
                repeats=repeats,
                flush_l2=None,
            )
    combined: dict[str, Any] = {}
    for arm in ("converted", "abliterated"):
        medians = [
            float(rounds[label][arm]["median_ms"]) for label in rounds
        ]
        combined[arm] = {
            "order_medians_ms": medians,
            "median_ms": statistics.median(medians),
        }
    combined["converted_delta"] = (
        combined["converted"]["median_ms"]
        / combined["abliterated"]["median_ms"]
        - 1.0
    )
    return {"rounds": rounds, "combined": combined}


def run(args: argparse.Namespace) -> int:
    if tuple(args.m) != REQUIRED_M:
        raise RuntimeError(f"exact-path gate requires --m 1,4,24,48")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exact-path B12X gate requires exactly one CUDA device")
    capability = tuple(torch.cuda.get_device_capability())
    if capability != (12, 1):
        raise RuntimeError(f"exact-path B12X gate requires SM121; got {capability}")
    shape = kernel_bench.Dsv4Shape(tp_rank=args.tp_rank)
    shape.validate()

    load_started = time.perf_counter()
    converted, converted_proof = _load_converted_rank(
        torch,
        args.prepared_layer_file,
        tp_rank=args.tp_rank,
        shape=shape,
    )
    abliterated, abliterated_proof = _load_abliterated_rank(
        torch,
        args.abliterated_shard,
        tp_rank=args.tp_rank,
        num_experts=shape.num_experts,
    )
    load_seconds = time.perf_counter() - load_started
    expected_shapes = {
        "w13": (shape.num_experts, 2 * shape.intermediate_size_per_rank, shape.hidden_size // 2),
        "w13_scale": (
            shape.num_experts,
            2 * shape.intermediate_size_per_rank,
            shape.hidden_size // 32,
        ),
        "w2": (
            shape.num_experts,
            shape.hidden_size,
            shape.intermediate_size_per_rank // 2,
        ),
        "w2_scale": (
            shape.num_experts,
            shape.hidden_size,
            shape.intermediate_size_per_rank // 32,
        ),
    }
    for arm_name, arm in (("converted", converted), ("abliterated", abliterated)):
        for family, expected in expected_shapes.items():
            if tuple(arm[family].shape) != expected:
                raise RuntimeError(
                    f"{arm_name} {family} shape drift: "
                    f"{tuple(arm[family].shape)} != {expected}"
                )
            if arm[family].dtype != torch.uint8:
                raise RuntimeError(
                    f"{arm_name} {family} dtype drift: {arm[family].dtype}"
                )

    converted_runner, converted_backend = _make_exact_b12x_runner(
        torch,
        converted,
        max_tokens=max(args.m),
        top_k=shape.top_k,
        swiglu_limit=args.swiglu_limit,
    )
    abliterated_runner, abliterated_backend = _make_exact_b12x_runner(
        torch,
        abliterated,
        max_tokens=max(args.m),
        top_k=shape.top_k,
        swiglu_limit=args.swiglu_limit,
    )

    rows: dict[int, Any] = {}
    failures: list[dict[str, Any]] = []
    keepalive: list[Any] = [converted_runner, abliterated_runner]
    for m in args.m:
        x, topk_ids, topk_weights = kernel_bench.make_routes(
            torch,
            shape,
            m,
            routing=args.routing,
            seed=args.seed + m,
            input_rms=args.input_rms,
        )
        converted_launch, _, converted_scratch = _make_launch(
            torch, converted_runner, x, topk_ids, topk_weights
        )
        abliterated_launch, _, abliterated_scratch = _make_launch(
            torch, abliterated_runner, x, topk_ids, topk_weights
        )
        keepalive.extend((x, topk_ids, topk_weights, converted_scratch, abliterated_scratch))

        eager: dict[str, Any] = {}
        activity: dict[str, Any] = {}
        graph_compare: dict[str, Any] = {}
        graph_launches: dict[str, Any] = {}
        for arm, launch in (
            ("converted", converted_launch),
            ("abliterated", abliterated_launch),
        ):
            eager[arm] = launch().clone()
            torch.cuda.synchronize()
            activity[arm] = kernel_bench.tensor_activity(torch, eager[arm])
            graph_launch, _, graph_obj = kernel_bench.capture_graph(torch, launch)
            keepalive.append(graph_obj)
            graph_value = graph_launch().clone()
            torch.cuda.synchronize()
            graph_compare[arm] = kernel_bench.compare_tensors(
                torch, graph_value, eager[arm]
            )
            graph_launches[arm] = graph_launch

        activity_passed = all(bool(value["passed"]) for value in activity.values())
        graph_passed = all(
            bool(value["finite"])
            and float(value["normalized_rmse"]) == 0.0
            and int(value["nonfinite_count"]) == 0
            for value in graph_compare.values()
        )
        timing = _paired_graph_timing(
            torch,
            graph_launches,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        row = {
            "m": m,
            "activity": activity,
            "activity_passed": activity_passed,
            "graph_vs_eager": graph_compare,
            "graph_passed": graph_passed,
            "timing": timing,
            "converted_graph_ms": timing["combined"]["converted"]["median_ms"],
            "abliterated_graph_ms": timing["combined"]["abliterated"]["median_ms"],
        }
        rows[m] = row
        if not activity_passed:
            failures.append({"kind": "activity", "m": m})
        if not graph_passed:
            failures.append({"kind": "graph", "m": m})
        print(
            f"M={m:>2} converted={row['converted_graph_ms']:.6f} ms "
            f"abliterated={row['abliterated_graph_ms']:.6f} ms "
            f"delta={timing['combined']['converted_delta']:+.2%}"
        )

    decision = evaluate_parity(
        rows, maximum_absolute_delta=args.maximum_absolute_decision_delta
    )
    if not decision["passed"]:
        failures.append({"kind": "decision_parity", "details": decision})
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "prepared_layer_file": str(args.prepared_layer_file.resolve()),
            "abliterated_shard": str(args.abliterated_shard.resolve()),
            "tp_rank": args.tp_rank,
        },
        "settings": {
            "m": list(args.m),
            "decision_m": list(DECISION_M),
            "maximum_absolute_decision_delta": args.maximum_absolute_decision_delta,
            "routing": args.routing,
            "seed": args.seed,
            "input_rms": args.input_rms,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
        },
        "load_seconds": load_seconds,
        "conversion_proof": converted_proof,
        "abliterated_load_proof": abliterated_proof,
        "backend_proof": {
            "converted": converted_backend,
            "abliterated": abliterated_backend,
            "same_exact_path": converted_backend == abliterated_backend,
        },
        "rows": [rows[m] for m in args.m],
        "decision": decision,
        "memory": {
            "allocated_gib": torch.cuda.memory_allocated() / (1024**3),
            "reserved_gib": torch.cuda.memory_reserved() / (1024**3),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        },
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
    parser.add_argument("--prepared-layer-file", type=Path, required=True)
    parser.add_argument("--abliterated-shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--m", type=_csv_positive_ints, default=REQUIRED_M)
    parser.add_argument(
        "--routing", choices=("balanced", "hot", "random"), default="balanced"
    )
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--input-rms", type=float, default=1.0)
    parser.add_argument("--swiglu-limit", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--maximum-absolute-decision-delta",
        type=float,
        default=MAXIMUM_ABSOLUTE_DECISION_DELTA,
    )
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
