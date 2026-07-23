#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""No-model SM121 gate for sparse 256-group NVFP4 grouped GEMM.

The route-major FC2 proposal keeps weights in physical expert order and
represents inactive experts with repeated ``m_indptr`` entries.  This probe
answers the cheapest hardware question before any checkpoint is loaded:
does FlashInfer's pinned ``group_gemm_nvfp4_nt_groupwise`` accept 256 problem
descriptors when most have M=0?

The oracle is deliberately stronger than "the kernel did not crash":

* the sparse 256-group result must be bitwise equal to independent one-group
  launches for every active expert;
* it must remain bitwise unchanged when every inactive expert's packed
  weights, scales, and alpha are replaced with different finite data; and
* it must remain numerically close to the BF16 tensors used to create the
  synthetic NVFP4 operands.

Imports of Torch and FlashInfer are local to :func:`run`, so the layout and
failure contracts remain unit-testable on a CPU-only development host.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import time
from typing import Sequence


PHYSICAL_GROUPS = 256
GROUP_ROW_ALIGNMENT = 4
SCALE_ROW_ALIGNMENT = 128
NVFP4_BLOCK_SIZE = 16


def align_up(value: int, alignment: int) -> int:
    if value < 0:
        raise ValueError("value must be non-negative")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


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


def select_sparse_groups(
    active_count: int,
    *,
    groups: int = PHYSICAL_GROUPS,
    stride: int = 37,
    offset: int = 11,
) -> tuple[int, ...]:
    """Return deterministic, sorted, collision-free physical expert IDs."""

    if not 1 <= active_count < groups:
        raise ValueError("active_count must be in [1, groups)")
    if math.gcd(stride, groups) != 1:
        raise ValueError("stride must be coprime to groups")
    selected = tuple(
        sorted((index * stride + offset) % groups for index in range(active_count))
    )
    if len(set(selected)) != active_count:
        raise AssertionError("deterministic sparse-group selection collided")
    return selected


@dataclass(frozen=True)
class SparseGroupLayout:
    groups: int
    active_groups: tuple[int, ...]
    rows_per_active: int
    lengths: tuple[int, ...]
    m_indptr: tuple[int, ...]
    scale_base_rows: tuple[int, ...]
    scale_storage_rows: int

    @property
    def active_count(self) -> int:
        return len(self.active_groups)

    @property
    def inactive_count(self) -> int:
        return self.groups - self.active_count

    @property
    def output_rows(self) -> int:
        return self.m_indptr[-1]

    @property
    def repeated_offset_count(self) -> int:
        return sum(
            left == right
            for left, right in zip(self.m_indptr, self.m_indptr[1:])
        )

    @property
    def leading_zero_groups(self) -> int:
        return self.active_groups[0]

    @property
    def trailing_zero_groups(self) -> int:
        return self.groups - self.active_groups[-1] - 1


def build_sparse_group_layout(
    *,
    groups: int,
    active_groups: Sequence[int],
    rows_per_active: int = GROUP_ROW_ALIGNMENT,
) -> SparseGroupLayout:
    """Build the exact global-group indptr and swizzled-SFA storage contract."""

    if groups <= 1:
        raise ValueError("groups must be greater than one")
    if rows_per_active <= 0 or rows_per_active % GROUP_ROW_ALIGNMENT:
        raise ValueError("rows_per_active must be a positive multiple of four")
    active = tuple(int(group) for group in active_groups)
    if not active:
        raise ValueError("at least one active group is required")
    if active != tuple(sorted(active)) or len(set(active)) != len(active):
        raise ValueError("active_groups must be unique and strictly ordered")
    if len(active) >= groups:
        raise ValueError("the sparse gate requires at least one inactive group")
    if any(group < 0 or group >= groups for group in active):
        raise ValueError("active group is outside the physical group range")

    lengths = [0] * groups
    for group in active:
        lengths[group] = rows_per_active
    m_indptr = [0]
    for length in lengths:
        m_indptr.append(m_indptr[-1] + length)
    scale_bases = tuple(
        group_scale_row_offset(group, m_indptr[group])
        for group in range(groups)
    )
    scale_rows = scale_bases[-1] + SCALE_ROW_ALIGNMENT

    if len(m_indptr) != groups + 1:
        raise AssertionError("m_indptr length drift")
    if any(value % GROUP_ROW_ALIGNMENT for value in m_indptr):
        raise AssertionError("m_indptr lost four-row alignment")
    if any(value % SCALE_ROW_ALIGNMENT for value in scale_bases):
        raise AssertionError("SFA descriptor base lost 128-row alignment")
    for group, rows in enumerate(lengths[:-1]):
        if scale_bases[group + 1] - scale_bases[group] < rows:
            raise AssertionError("adjacent SFA descriptors overlap")
    if scale_bases[-1] + SCALE_ROW_ALIGNMENT > scale_rows:
        raise AssertionError("final global-group SFA descriptor is out of bounds")

    result = SparseGroupLayout(
        groups=groups,
        active_groups=active,
        rows_per_active=rows_per_active,
        lengths=tuple(lengths),
        m_indptr=tuple(m_indptr),
        scale_base_rows=scale_bases,
        scale_storage_rows=scale_rows,
    )
    if result.repeated_offset_count != result.inactive_count:
        raise AssertionError("zero-length group accounting drift")
    return result


def gate_failures(
    *,
    finite: bool,
    nonzero_real_rows: bool,
    padded_rows_zero: bool,
    sparse_matches_independent: bool,
    inactive_poison_invariant: bool,
    cosine: float,
    normalized_rmse: float,
    minimum_cosine: float,
    maximum_nrmse: float,
) -> list[str]:
    """Return fail-closed gate names; empty means the probe passed."""

    failures = []
    checks = (
        ("finite_output", finite),
        ("nonzero_real_rows", nonzero_real_rows),
        ("zero_padded_rows", padded_rows_zero),
        ("sparse_matches_independent", sparse_matches_independent),
        ("inactive_poison_invariant", inactive_poison_invariant),
        ("minimum_cosine", math.isfinite(cosine) and cosine >= minimum_cosine),
        (
            "maximum_normalized_rmse",
            math.isfinite(normalized_rmse) and normalized_rmse <= maximum_nrmse,
        ),
    )
    failures.extend(name for name, passed in checks if not passed)
    return failures


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one sample")
    ordered = sorted(float(value) for value in values)
    index = round(fraction * (len(ordered) - 1))
    return ordered[min(len(ordered) - 1, max(0, index))]


def _distribution_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in (
        "flashinfer-python",
        "flashinfer-jit-cache",
        "flashinfer-cubin",
        "nvidia-cutlass-dsl",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def run(args: argparse.Namespace) -> int:
    import flashinfer
    import torch

    if args.groups != PHYSICAL_GROUPS:
        raise ValueError(f"hardware Gate 1 requires exactly {PHYSICAL_GROUPS} groups")
    if args.rows_per_active != GROUP_ROW_ALIGNMENT:
        raise ValueError(
            f"hardware Gate 1 requires exactly {GROUP_ROW_ALIGNMENT} padded rows"
        )
    if args.n <= 0 or args.n % SCALE_ROW_ALIGNMENT:
        raise ValueError("n must be a positive multiple of 128")
    if args.k <= 0 or args.k % 128:
        raise ValueError("k must be a positive multiple of 128")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be non-negative and iters positive")
    if not 0.0 < args.minimum_cosine <= 1.0:
        raise ValueError("minimum-cosine must be in (0,1]")
    if not math.isfinite(args.maximum_nrmse) or args.maximum_nrmse <= 0.0:
        raise ValueError("maximum-nrmse must be finite and positive")
    if not torch.cuda.is_available():
        raise RuntimeError("sparse grouped-GEMM Gate 1 requires CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("sparse grouped-GEMM Gate 1 requires exactly one GPU")
    capability = tuple(torch.cuda.get_device_capability())
    if capability != (12, 1):
        raise RuntimeError(f"sparse grouped-GEMM Gate 1 requires SM121, got {capability}")

    active_groups = select_sparse_groups(
        args.active_groups,
        groups=args.groups,
        stride=args.group_stride,
        offset=args.group_offset,
    )
    layout = build_sparse_group_layout(
        groups=args.groups,
        active_groups=active_groups,
        rows_per_active=args.rows_per_active,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    # N is 128-aligned, so quantizing the flattened [G*N,K] matrix produces
    # exactly one or more independent 128-row SFB tiles per physical group.
    b_float = (
        torch.randn(
            args.groups,
            args.n,
            args.k,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.20
    )
    b_poison_float = (
        torch.randn_like(b_float) * 0.75
        + torch.tensor(0.125, dtype=torch.bfloat16, device=device)
    )
    b_global_sf = (448.0 * 6.0) / b_float.float().abs().nan_to_num().max()
    b_fp4_flat, b_scale_flat = flashinfer.nvfp4_quantize(
        b_float.reshape(args.groups * args.n, args.k),
        b_global_sf,
        sfLayout=flashinfer.SfLayout.layout_128x4,
        do_shuffle=False,
        sf_vec_size=16,
    )
    b_poison_fp4_flat, b_poison_scale_flat = flashinfer.nvfp4_quantize(
        b_poison_float.reshape(args.groups * args.n, args.k),
        b_global_sf,
        sfLayout=flashinfer.SfLayout.layout_128x4,
        do_shuffle=False,
        sf_vec_size=16,
    )
    b_fp4 = b_fp4_flat.view(args.groups, args.n, args.k // 2)
    b_scale = b_scale_flat.view(
        args.groups, args.n, args.k // NVFP4_BLOCK_SIZE
    )
    b_poison_fp4 = b_poison_fp4_flat.view_as(b_fp4).clone()
    b_poison_scale = b_poison_scale_flat.view_as(b_scale).clone()
    active_index = torch.tensor(active_groups, dtype=torch.int64, device=device)
    b_poison_fp4.index_copy_(0, active_index, b_fp4.index_select(0, active_index))
    b_poison_scale.index_copy_(
        0, active_index, b_scale.index_select(0, active_index)
    )

    a_fp4 = torch.empty(
        layout.output_rows,
        args.k // 2,
        dtype=torch.uint8,
        device=device,
    )
    a_scale = torch.zeros(
        layout.scale_storage_rows,
        args.k // NVFP4_BLOCK_SIZE,
        dtype=torch.uint8,
        device=device,
    )
    a_float_by_group = []
    a_scale_by_group = []
    alpha = torch.ones(args.groups, dtype=torch.float32, device=device)
    for route_index, group in enumerate(active_groups):
        segment = torch.zeros(
            args.rows_per_active,
            args.k,
            dtype=torch.bfloat16,
            device=device,
        )
        # Gate 1 models the balanced C4 case: one real routed row and three
        # physical padding rows for each of 24 active experts.
        segment[0].normal_(mean=0.0, std=0.35)
        a_global_sf = torch.tensor(
            [(448.0 * 6.0) / max(float(segment[0].abs().max().item()), 1.0e-6)],
            dtype=torch.float32,
            device=device,
        )
        segment_fp4, segment_scale = flashinfer.nvfp4_quantize(
            segment,
            a_global_sf,
            sfLayout=flashinfer.SfLayout.layout_128x4,
            do_shuffle=False,
            sf_vec_size=16,
        )
        if tuple(segment_fp4.shape) != (
            args.rows_per_active,
            args.k // 2,
        ):
            raise RuntimeError(
                f"packed A shape drift: {tuple(segment_fp4.shape)}"
            )
        if tuple(segment_scale.shape) != (
            SCALE_ROW_ALIGNMENT,
            args.k // NVFP4_BLOCK_SIZE,
        ):
            raise RuntimeError(
                f"A scale shape drift: {tuple(segment_scale.shape)}"
            )
        begin = layout.m_indptr[group]
        end = begin + args.rows_per_active
        a_fp4[begin:end].copy_(segment_fp4.view(torch.uint8))
        scale_begin = layout.scale_base_rows[group]
        a_scale[scale_begin : scale_begin + SCALE_ROW_ALIGNMENT].copy_(
            segment_scale.view(torch.uint8)
        )
        alpha[group] = 1.0 / (a_global_sf * b_global_sf)
        a_float_by_group.append(segment)
        a_scale_by_group.append(segment_scale.view(torch.uint8))

    m_indptr = torch.tensor(layout.m_indptr, dtype=torch.int32, device=device)
    one_group_indptr = torch.tensor([0, args.rows_per_active], dtype=torch.int32, device=device)
    output = torch.empty(
        layout.output_rows, args.n, dtype=torch.bfloat16, device=device
    )
    poison_output = torch.empty_like(output)

    def launch(
        weight: object,
        weight_scale: object,
        group_alpha: object,
        destination: object,
        *,
        initialize_output: bool = False,
    ) -> None:
        if initialize_output:
            destination.fill_(float("nan"))
        flashinfer.gemm.group_gemm_nvfp4_nt_groupwise(
            a_fp4,
            weight,
            a_scale,
            weight_scale,
            m_indptr,
            alpha=group_alpha,
            tile_m=128,
            tile_n=128,
            tile_k=128,
            swap_ab=True,
            out=destination,
        )

    launch(
        b_fp4,
        b_scale.view(torch.uint8),
        alpha,
        output,
        initialize_output=True,
    )
    torch.cuda.synchronize()

    independent_rows = []
    for route_index, group in enumerate(active_groups):
        begin = route_index * args.rows_per_active
        end = begin + args.rows_per_active
        independent_rows.append(
            flashinfer.gemm.group_gemm_nvfp4_nt_groupwise(
                a_fp4[begin:end],
                b_fp4[group : group + 1],
                a_scale_by_group[route_index],
                b_scale[group : group + 1].view(torch.uint8),
                one_group_indptr,
                alpha=alpha[group : group + 1],
                tile_m=128,
                tile_n=128,
                tile_k=128,
                swap_ab=True,
                out_dtype=torch.bfloat16,
            )
        )
    independent = torch.cat(independent_rows, dim=0)
    torch.cuda.synchronize()

    poison_alpha = torch.full_like(alpha, -17.0)
    poison_alpha.index_copy_(0, active_index, alpha.index_select(0, active_index))
    launch(
        b_poison_fp4,
        b_poison_scale.view(torch.uint8),
        poison_alpha,
        poison_output,
        initialize_output=True,
    )
    torch.cuda.synchronize()

    bf16_reference = torch.cat(
        [
            torch.mm(segment.float(), b_float[group].float().T).to(torch.bfloat16)
            for segment, group in zip(
                a_float_by_group, active_groups, strict=True
            )
        ],
        dim=0,
    )
    output_fp32 = output.float()
    reference_fp32 = bf16_reference.float()
    difference = output_fp32 - reference_fp32
    reference_rms = float(torch.sqrt(torch.mean(reference_fp32.square())).item())
    rmse = float(torch.sqrt(torch.mean(difference.square())).item())
    cosine = float(
        torch.nn.functional.cosine_similarity(
            output_fp32.reshape(-1),
            reference_fp32.reshape(-1),
            dim=0,
        ).item()
    )
    normalized_rmse = rmse / max(reference_rms, 1.0e-12)
    output_rows = output.view(layout.active_count, args.rows_per_active, args.n)
    real_rows = output_rows[:, 0, :]
    padded_rows = output_rows[:, 1:, :]
    sparse_matches_independent = bool(
        torch.equal(output.view(torch.int16), independent.view(torch.int16))
    )
    inactive_poison_invariant = bool(
        torch.equal(output.view(torch.int16), poison_output.view(torch.int16))
    )
    finite = bool(torch.isfinite(output).all().item())
    nonzero_real_rows = int(torch.count_nonzero(real_rows).item()) > 0
    padded_rows_zero = int(torch.count_nonzero(padded_rows).item()) == 0

    for _ in range(args.warmup):
        launch(b_fp4, b_scale.view(torch.uint8), alpha, output)
    torch.cuda.synchronize()
    timing_samples = []
    for _ in range(args.iters):
        begin_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        begin_event.record()
        launch(b_fp4, b_scale.view(torch.uint8), alpha, output)
        end_event.record()
        end_event.synchronize()
        timing_samples.append(float(begin_event.elapsed_time(end_event)))

    failures = gate_failures(
        finite=finite,
        nonzero_real_rows=nonzero_real_rows,
        padded_rows_zero=padded_rows_zero,
        sparse_matches_independent=sparse_matches_independent,
        inactive_poison_invariant=inactive_poison_invariant,
        cosine=cosine,
        normalized_rmse=normalized_rmse,
        minimum_cosine=args.minimum_cosine,
        maximum_nrmse=args.maximum_nrmse,
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "probe": "nvfp4_sparse_group_gemm_sm121",
        "decision": {
            "passed": not failures,
            "failures": failures,
            "meaning": (
                "pinned grouped NVFP4 GEMM accepts sparse 256-group M=0 descriptors"
                if not failures
                else "route-major FC2 Gate 1 rejected"
            ),
        },
        "provenance": {
            "torch": torch.__version__,
            "flashinfer_file": str(Path(flashinfer.__file__).resolve()),
            "distributions": _distribution_versions(),
            "device": torch.cuda.get_device_name(),
            "capability": list(capability),
            "checkpoint_loaded": False,
        },
        "settings": {
            "seed": args.seed,
            "groups": args.groups,
            "active_groups": args.active_groups,
            "rows_per_active": args.rows_per_active,
            "n": args.n,
            "k": args.k,
            "warmup": args.warmup,
            "iters": args.iters,
            "minimum_cosine": args.minimum_cosine,
            "maximum_nrmse": args.maximum_nrmse,
            "tile_m": 128,
            "tile_n": 128,
            "tile_k": 128,
            "swap_ab": True,
        },
        "layout": {
            **asdict(layout),
            "active_count": layout.active_count,
            "inactive_count": layout.inactive_count,
            "output_rows": layout.output_rows,
            "repeated_offset_count": layout.repeated_offset_count,
            "leading_zero_groups": layout.leading_zero_groups,
            "trailing_zero_groups": layout.trailing_zero_groups,
        },
        "numeric": {
            "finite": finite,
            "real_row_nonzero_count": int(torch.count_nonzero(real_rows).item()),
            "padded_row_nonzero_count": int(torch.count_nonzero(padded_rows).item()),
            "sparse_vs_independent_bitwise_equal": sparse_matches_independent,
            "sparse_vs_independent_mismatch_count": int(
                torch.count_nonzero(
                    output.view(torch.int16) != independent.view(torch.int16)
                ).item()
            ),
            "inactive_poison_bitwise_invariant": inactive_poison_invariant,
            "inactive_poison_mismatch_count": int(
                torch.count_nonzero(
                    output.view(torch.int16) != poison_output.view(torch.int16)
                ).item()
            ),
            "bf16_reference_cosine": cosine,
            "bf16_reference_rmse": rmse,
            "bf16_reference_normalized_rmse": normalized_rmse,
            "bf16_reference_max_abs": float(difference.abs().max().item()),
        },
        "zero_length_proof": {
            "problem_count": args.groups,
            "zero_length_problem_count": layout.inactive_count,
            "m_indptr_entries": len(layout.m_indptr),
            "output_rows_equal_active_padded_rows": (
                layout.output_rows
                == layout.active_count * args.rows_per_active
            ),
            "inactive_operands_changed": True,
            "active_operands_preserved": True,
            "output_bitwise_invariant_after_inactive_poison": (
                inactive_poison_invariant
            ),
        },
        "timing_ms": {
            "median": statistics.median(timing_samples),
            "mean": statistics.mean(timing_samples),
            "p95": _percentile(timing_samples, 0.95),
            "min": min(timing_samples),
            "max": max(timing_samples),
            "samples": len(timing_samples),
        },
        "memory": {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "PASS" if not failures else "FAILED",
        f"groups={args.groups}",
        f"active={layout.active_count}",
        f"zero={layout.inactive_count}",
        f"median={report['timing_ms']['median']:.4f} ms",
        f"cosine={cosine:.6f}",
        f"nrmse={normalized_rmse:.6f}",
    )
    print(f"Wrote {args.output}")
    return 0 if not failures else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=PHYSICAL_GROUPS)
    parser.add_argument("--active-groups", type=int, default=24)
    parser.add_argument("--rows-per-active", type=int, default=GROUP_ROW_ALIGNMENT)
    parser.add_argument("--group-stride", type=int, default=37)
    parser.add_argument("--group-offset", type=int, default=11)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--minimum-cosine", type=float, default=0.97)
    parser.add_argument("--maximum-nrmse", type=float, default=0.25)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
