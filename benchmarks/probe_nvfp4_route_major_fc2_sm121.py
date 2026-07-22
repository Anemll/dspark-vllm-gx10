#!/usr/bin/env python3
"""Bounded SM121 feasibility probe for a route-major NVFP4 FC2 launch.

This is not a serving implementation.  It answers the first structural
question for the decode-only split-kernel port: can FlashInfer's grouped
NVFP4 GEMM consume the physical 256-expert W2 tensor while inactive experts
are represented by repeated offsets in ``m_indptr``?  Active one-row routes
are padded to the primitive's required four-row granularity.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Sequence


PREPARED_PREFIX = "__dspark_tp2_nvfp4_cutlass_v1__.layers.0.experts."
PHYSICAL_EXPERTS = 256
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 1024
GROUP_ROW_ALIGNMENT = 4
SCALE_ROW_ALIGNMENT = 128

# This is the accepted end-to-end real-layer deadline, not a claim that phase
# 2 may consume the whole budget.  The grouped-GEMM-only probe must at least
# fit below it or a complete phase1 + FC2 + reduction implementation cannot
# close the measured serving gap.
FULL_PROTOTYPE_M4_MAX_MS = 0.682812


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def select_active_experts(
    routes: int,
    *,
    experts: int = PHYSICAL_EXPERTS,
    stride: int = 37,
    offset: int = 11,
) -> tuple[int, ...]:
    """Select deterministic sparse physical experts without collisions."""

    if not 1 <= routes <= experts:
        raise ValueError("routes must be in [1, experts]")
    if math.gcd(stride, experts) != 1:
        raise ValueError("expert-selection stride must be coprime to experts")
    selected = tuple(
        sorted((index * stride + offset) % experts for index in range(routes))
    )
    if len(set(selected)) != routes:
        raise AssertionError("deterministic expert selection collided")
    return selected


def group_scale_row_offset(
    group: int,
    m_offset: int,
    *,
    alignment: int = SCALE_ROW_ALIGNMENT,
) -> int:
    """Mirror FlashInfer's SM120 grouped-NVFP4 SFA pointer formula."""

    if group < 0 or m_offset < 0:
        raise ValueError("group and M offset must be non-negative")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((m_offset + group * (alignment - 1)) // alignment) * alignment


def build_sparse_group_layout(
    *,
    experts: int,
    active_experts: Sequence[int],
    rows_per_active: int = GROUP_ROW_ALIGNMENT,
) -> dict[str, tuple[int, ...] | int]:
    """Build the exact global-expert indptr and SFA storage contract.

    Packed A rows follow ascending physical-expert order in this feasibility
    probe.  Inactive experts have repeated indptr offsets.  Scale storage is
    nevertheless reserved through the complete descriptor tile of the final
    physical expert because FlashInfer builds pointers for all groups.
    """

    if experts <= 0:
        raise ValueError("experts must be positive")
    if rows_per_active <= 0 or rows_per_active % GROUP_ROW_ALIGNMENT:
        raise ValueError("rows_per_active must be a positive multiple of four")
    active = tuple(int(expert) for expert in active_experts)
    if active != tuple(sorted(active)) or len(set(active)) != len(active):
        raise ValueError("active experts must be unique and strictly ordered")
    if any(expert < 0 or expert >= experts for expert in active):
        raise ValueError("active expert is outside the physical expert range")

    lengths = [0] * experts
    for expert in active:
        lengths[expert] = rows_per_active
    m_indptr = [0]
    for length in lengths:
        m_indptr.append(m_indptr[-1] + length)
    scale_base_rows = tuple(
        group_scale_row_offset(group, m_indptr[group]) for group in range(experts)
    )
    scale_storage_rows = scale_base_rows[-1] + SCALE_ROW_ALIGNMENT

    if any(value % GROUP_ROW_ALIGNMENT for value in m_indptr):
        raise AssertionError("m_indptr lost four-row alignment")
    if any(value % SCALE_ROW_ALIGNMENT for value in scale_base_rows):
        raise AssertionError("group scale base lost 128-row alignment")
    for group, rows in enumerate(lengths[:-1]):
        if scale_base_rows[group + 1] - scale_base_rows[group] < rows:
            raise AssertionError("adjacent group scale descriptors overlap")
    if any(
        scale_base_rows[expert] + SCALE_ROW_ALIGNMENT > scale_storage_rows
        for expert in active
    ):
        raise AssertionError("active group scale descriptor is out of bounds")

    return {
        "lengths": tuple(lengths),
        "m_indptr": tuple(m_indptr),
        "scale_base_rows": scale_base_rows,
        "scale_storage_rows": scale_storage_rows,
        "repeated_offset_count": sum(
            left == right for left, right in zip(m_indptr, m_indptr[1:])
        ),
    }


def validate_prepared_w2_scale_algebra(
    a2_gscale: Sequence[float],
    g2_alphas: Sequence[float],
    *,
    experts: int = PHYSICAL_EXPERTS,
) -> dict[str, float | bool | str]:
    """Prove the offline-prepared W2 scalar contract without Torch.

    The immutable prepared format stores::

        a2_gscale = 1 / max(checkpoint w2 input_scale)
        g2_alpha  = w2.weight_scale_2 / a2_gscale

    Therefore the original activation scale and per-expert weight scale are
    reconstructible as ``1/a2_gscale`` and ``g2_alpha*a2_gscale``.  The
    grouped GEMM must receive ``g2_alpha`` directly; multiplying by either
    reconstructed scale again would double-apply the offline transform.
    """

    a2 = tuple(float(value) for value in a2_gscale)
    g2 = tuple(float(value) for value in g2_alphas)
    if len(a2) != experts or len(g2) != experts:
        raise ValueError("prepared W2 scalar vectors must match physical experts")
    if not all(math.isfinite(value) and value > 0.0 for value in a2):
        raise ValueError("a2_gscale must contain finite positive values")
    if not all(math.isfinite(value) and value > 0.0 for value in g2):
        raise ValueError("g2_alphas must contain finite positive values")
    if any(value != a2[0] for value in a2[1:]):
        raise ValueError("prepared a2_gscale must be globally reduced and constant")

    checkpoint_input_scale = 1.0 / a2[0]
    weight_scale_2 = tuple(alpha * a2[0] for alpha in g2)
    if not math.isfinite(checkpoint_input_scale) or not all(
        math.isfinite(value) and value > 0.0 for value in weight_scale_2
    ):
        raise ValueError("reconstructed prepared W2 scales are invalid")
    return {
        "contract": "a2=1/input_scale; g2=weight_scale_2/a2",
        "grouped_gemm_alpha_is_g2_direct": True,
        "a2_constant": True,
        "a2_gscale": a2[0],
        "reconstructed_checkpoint_input_scale": checkpoint_input_scale,
        "g2_alpha_min": min(g2),
        "g2_alpha_max": max(g2),
        "reconstructed_weight_scale_2_min": min(weight_scale_2),
        "reconstructed_weight_scale_2_max": max(weight_scale_2),
    }


def run(args: argparse.Namespace) -> int:
    import flashinfer
    import torch
    from safetensors import safe_open

    if not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability()) != (12, 1):
        raise RuntimeError("route-major FC2 probe requires one SM121 GPU")
    if args.tp_rank not in (0, 1):
        raise ValueError("tp-rank must be 0 or 1")
    if not 1 <= args.routes <= PHYSICAL_EXPERTS:
        raise ValueError(f"routes must be in [1, {PHYSICAL_EXPERTS}]")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be non-negative and iters must be positive")
    if not math.isfinite(args.max_phase2_median_ms) or args.max_phase2_median_ms <= 0:
        raise ValueError("max-phase2-median-ms must be finite and positive")

    started = time.perf_counter()
    with safe_open(str(args.layer_file), framework="pt", device="cpu") as handle:
        w2_cpu = handle.get_tensor(PREPARED_PREFIX + "w2.weight")[args.tp_rank]
        w2_scale_cpu = handle.get_tensor(
            PREPARED_PREFIX + "w2.weight_scale"
        )[args.tp_rank]
        a2_cpu = handle.get_tensor(PREPARED_PREFIX + "a2_gscale")[args.tp_rank]
        g2_cpu = handle.get_tensor(PREPARED_PREFIX + "g2_alphas")[args.tp_rank]

    expected_tensors = (
        (
            "w2.weight",
            w2_cpu,
            torch.uint8,
            (PHYSICAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE // 2),
        ),
        (
            "w2.weight_scale",
            w2_scale_cpu,
            torch.float8_e4m3fn,
            (PHYSICAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE // 16),
        ),
        ("a2_gscale", a2_cpu, torch.float32, (PHYSICAL_EXPERTS,)),
        ("g2_alphas", g2_cpu, torch.float32, (PHYSICAL_EXPERTS,)),
    )
    for name, tensor, dtype, shape in expected_tensors:
        if tensor.dtype != dtype or tuple(tensor.shape) != shape:
            raise RuntimeError(
                f"prepared {name} contract drift: got {tensor.dtype} "
                f"{tuple(tensor.shape)}, expected {dtype} {shape}"
            )
        if not tensor.is_contiguous():
            raise RuntimeError(f"prepared {name} must be contiguous")
    scale_algebra = validate_prepared_w2_scale_algebra(
        a2_cpu.tolist(), g2_cpu.tolist()
    )

    w2 = w2_cpu.to("cuda")
    w2_scale = w2_scale_cpu.to("cuda")
    a2 = a2_cpu.to("cuda")
    g2 = g2_cpu.to("cuda")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started

    experts, hidden_size, packed_k = map(int, w2.shape)
    intermediate_size = packed_k * 2
    if (
        experts != PHYSICAL_EXPERTS
        or hidden_size != HIDDEN_SIZE
        or intermediate_size != INTERMEDIATE_SIZE
    ):
        raise RuntimeError(f"unexpected W2 shape: {tuple(w2.shape)}")

    # One real route per active expert plus three zero rows.  Padding is not a
    # proposed serving policy; it is required by this existing primitive and
    # is measured honestly here before deciding whether the primitive is a
    # viable building block.
    active_experts = select_active_experts(args.routes, experts=experts)
    layout = build_sparse_group_layout(
        experts=experts,
        active_experts=active_experts,
        rows_per_active=GROUP_ROW_ALIGNMENT,
    )
    indptr_host = layout["m_indptr"]
    scale_base_rows = layout["scale_base_rows"]
    if not isinstance(indptr_host, tuple) or not isinstance(scale_base_rows, tuple):
        raise AssertionError("internal sparse-layout type drift")
    padded_rows = int(indptr_host[-1])
    torch.manual_seed(args.seed)
    activation = torch.zeros(
        padded_rows,
        intermediate_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    activation[::4].normal_(mean=0.0, std=0.5)
    # Spread routes across physical IDs so inactive groups occur before,
    # between, and after active groups. Packed row order is global group order.
    m_indptr = torch.tensor(indptr_host, dtype=torch.int32, device="cuda")

    # The grouped GEMM addresses one independent 128-row scale tile per
    # expert, even when that expert owns only four rows.  Its descriptor base
    # is floor((m_indptr[g] + g * 127) / 128) * 128.  A single quantization of
    # the concatenated rows would allocate only one tile and make every group
    # after zero read out of bounds.  Quantize each active segment separately
    # and place its complete swizzled tile at the exact descriptor offset;
    # allocate through expert 255 so repeated-offset inactive groups remain
    # pointer-safe as well.
    scale_rows = int(layout["scale_storage_rows"])
    a_scale = torch.zeros(
        scale_rows,
        intermediate_size // 16,
        dtype=torch.uint8,
        device="cuda",
    )
    packed_segments = []
    segment_scales = []
    for route_index, group in enumerate(active_experts):
        begin = route_index * 4
        end = begin + 4
        segment_fp4, segment_scale = flashinfer.nvfp4_quantize(
            activation[begin:end],
            a2[group : group + 1],
            sfLayout=flashinfer.SfLayout.layout_128x4,
            do_shuffle=False,
            sf_vec_size=16,
        )
        if tuple(segment_scale.shape) != (128, intermediate_size // 16):
            raise RuntimeError(
                f"unexpected per-group scale shape: {tuple(segment_scale.shape)}"
            )
        packed_segments.append(segment_fp4.view(torch.uint8))
        segment_scales.append(segment_scale.view(torch.uint8))
        scale_base = scale_base_rows[group]
        a_scale[scale_base : scale_base + 128].copy_(segment_scale.view(torch.uint8))
    a_fp4 = torch.cat(packed_segments, dim=0)
    output = torch.empty(
        padded_rows,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )

    def launch() -> None:
        flashinfer.gemm.group_gemm_nvfp4_nt_groupwise(
            a_fp4.view(torch.uint8),
            w2.view(torch.uint8),
            a_scale.view(torch.uint8),
            w2_scale.view(torch.uint8),
            m_indptr,
            alpha=g2.to(torch.float32),
            tile_m=128,
            tile_n=128,
            tile_k=128,
            swap_ab=True,
            out=output,
        )

    launch()
    torch.cuda.synchronize()
    # Compare the 256-group sparse-offset launch with independent one-group
    # launches.  This catches descriptor-address mistakes that finite/nonzero
    # checks cannot detect.
    reference_rows = []
    one_group_indptr = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
    for route_index, group in enumerate(active_experts):
        reference_rows.append(
            flashinfer.gemm.group_gemm_nvfp4_nt_groupwise(
                a_fp4[route_index * 4 : route_index * 4 + 4],
                w2[group : group + 1].view(torch.uint8),
                segment_scales[route_index],
                w2_scale[group : group + 1].view(torch.uint8),
                one_group_indptr,
                alpha=g2[group : group + 1].to(torch.float32),
                tile_m=128,
                tile_n=128,
                tile_k=128,
                swap_ab=True,
                out_dtype=torch.bfloat16,
            )
        )
    reference = torch.cat(reference_rows, dim=0)
    torch.cuda.synchronize()
    output_by_route = output.reshape(args.routes, GROUP_ROW_ALIGNMENT, hidden_size)
    reference_by_route = reference.reshape(
        args.routes, GROUP_ROW_ALIGNMENT, hidden_size
    )
    difference = (output.float() - reference.float()).abs()
    reference_rms = float(torch.sqrt(torch.mean(reference.float().square())).item())
    rmse = float(torch.sqrt(torch.mean(difference.square())).item())
    mismatch_count = int(
        torch.count_nonzero(
            output.view(torch.int16) != reference.view(torch.int16)
        ).item()
    )
    route_numeric = []
    for route_index, group in enumerate(active_experts):
        route_difference = difference.reshape(
            args.routes, GROUP_ROW_ALIGNMENT, hidden_size
        )[route_index]
        route_numeric.append(
            {
                "expert": group,
                "bitwise_equal": bool(
                    torch.equal(
                        output_by_route[route_index].view(torch.int16),
                        reference_by_route[route_index].view(torch.int16),
                    )
                ),
                "max_abs": float(route_difference.max().item()),
                "real_row_nonzero": int(
                    torch.count_nonzero(output_by_route[route_index, 0]).item()
                ),
                "padded_row_nonzero": int(
                    torch.count_nonzero(output_by_route[route_index, 1:]).item()
                ),
            }
        )
    numeric = {
        "max_abs": float(difference.max().item()),
        "rmse": rmse,
        "normalized_rmse": rmse / max(reference_rms, 1.0e-12),
        "bitwise_mismatch_count": mismatch_count,
        "bitwise_equal": mismatch_count == 0,
        "per_route": route_numeric,
    }
    real_rows = output[::4]
    padded_output = output.reshape(args.routes, 4, hidden_size)[:, 1:, :]
    activity = {
        "finite": bool(torch.isfinite(output).all().item()),
        "nonzero": int(torch.count_nonzero(output).item()),
        "real_row_nonzero": int(torch.count_nonzero(real_rows).item()),
        "padded_row_nonzero": int(torch.count_nonzero(padded_output).item()),
        "max_abs": float(output.abs().max().item()),
    }
    samples: list[float] = []
    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()
    for _ in range(args.iters):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        launch()
        end.record()
        end.synchronize()
        samples.append(float(begin.elapsed_time(end)))

    report = {
        "probe": "nvfp4_route_major_fc2_sm121",
        "checkpoint": str(args.layer_file),
        "tp_rank": args.tp_rank,
        "routes": args.routes,
        "active_experts": active_experts,
        "physical_experts": experts,
        "padded_rows": padded_rows,
        "padding_factor": 4.0,
        "inactive_experts_use_repeated_offsets": True,
        "inactive_experts": experts - args.routes,
        "repeated_indptr_offsets": int(layout["repeated_offset_count"]),
        "m_indptr_tail": int(m_indptr[-1].item()),
        "input_shape": list(a_fp4.shape),
        "input_scale_shape": list(a_scale.shape),
        "input_scale_base_rows": list(scale_base_rows),
        "weight_shape": list(w2.shape),
        "weight_scale_shape": list(w2_scale.shape),
        "prepared_w2_scale_algebra": scale_algebra,
        "load_seconds": load_seconds,
        "activity": activity,
        "sparse_vs_one_group_numeric": numeric,
        "timing_ms": {
            "median": statistics.median(samples),
            "mean": statistics.mean(samples),
            "p95": _percentile(samples, 0.95),
            "min": min(samples),
            "max": max(samples),
            "samples": len(samples),
        },
    }
    report["performance_gate"] = {
        "scope": "grouped_fc2_only_necessary_not_sufficient",
        "full_prototype_m4_deadline_ms": args.max_phase2_median_ms,
        "median_within_full_prototype_deadline": (
            report["timing_ms"]["median"] <= args.max_phase2_median_ms
        ),
    }
    report["passed"] = bool(
        activity["finite"]
        and activity["real_row_nonzero"] > 0
        and activity["padded_row_nonzero"] == 0
        and numeric["bitwise_equal"]
        and all(row["real_row_nonzero"] > 0 for row in route_numeric)
        and all(row["padded_row_nonzero"] == 0 for row in route_numeric)
        and report["m_indptr_tail"] == padded_rows
        and report["repeated_indptr_offsets"] == experts - args.routes
        and report["performance_gate"]["median_within_full_prototype_deadline"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--routes", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument(
        "--max-phase2-median-ms",
        type=float,
        default=FULL_PROTOTYPE_M4_MAX_MS,
        help=(
            "necessary upper bound for grouped FC2 alone; the complete "
            "phase1+FC2+reduction prototype must fit inside the same deadline"
        ),
    )
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
