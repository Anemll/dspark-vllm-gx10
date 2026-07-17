#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""DeepSeek V4 Flash NVFP4 W4A4-vs-W4A16 kernel harness for SM121.

This is a single-rank, single-layer routed-MoE microbenchmark.  It deliberately
keeps checkpoint loading, route construction, weight preparation, compilation,
and correctness checks outside the timed region.  Both measured paths consume
the same packed ModelOpt NVFP4 weight tensors:

* W4A4: FlashInfer ``B12xMoEWrapper(quant_mode="nvfp4")``.
* W4A16: pinned B12X native-ModelOpt W4A16 kernel.

The W4A16 comparator comes directly from B12X because the pinned FlashInfer
W4A16 wrapper accepts only unclamped SiLU/ReLU2.  B12X's W4A16 path supports
the DeepSeek V4 clamp, allowing an activation-matched comparison with W4A4:
``swigluoai_uninterleave(alpha=1, beta=0, limit=10)`` on W4A4 is equivalent to
``silu(swiglu_limit=10)`` on W4A16.

The script imports CUDA libraries only after argument validation.  Therefore
``--help`` and ``--dry-run`` work on a non-CUDA development host.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import math
import pathlib
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any


SCHEMA_VERSION = 1
INPUT_RMS_RELATIVE_TOLERANCE = 0.01
DSV4_TP2_M8192_B12X_WRAPPER_CEILING_BYTES = 635_144_040
B12X_W13_LAYOUT = "w13"
DEFAULT_M_VALUES = (
    1,
    2,
    4,
    6,
    12,
    24,
    48,
    72,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
)
DEFAULT_CORRECTNESS_M = (1, 24, 128, 2048)


@dataclasses.dataclass(frozen=True)
class Dsv4Shape:
    hidden_size: int = 4096
    intermediate_size: int = 2048
    num_experts: int = 256
    top_k: int = 6
    tp_size: int = 2
    tp_rank: int = 0

    @property
    def intermediate_size_per_rank(self) -> int:
        return self.intermediate_size // self.tp_size

    def validate(self) -> None:
        if self.hidden_size <= 0 or self.intermediate_size <= 0:
            raise ValueError("hidden and intermediate dimensions must be positive")
        if self.num_experts <= 0 or self.top_k <= 0:
            raise ValueError("expert count and top-k must be positive")
        if self.top_k > self.num_experts:
            raise ValueError("top-k cannot exceed expert count")
        if self.tp_size <= 0 or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("invalid TP size/rank")
        if self.intermediate_size % self.tp_size:
            raise ValueError("moe_intermediate_size must be divisible by TP size")
        if self.hidden_size % 128:
            raise ValueError("SM121 B12X requires hidden_size divisible by 128")
        if self.intermediate_size_per_rank % 128:
            raise ValueError(
                "SM121 W4A4 requires per-rank intermediate size divisible by 128"
            )


def calculate_dsv4_tp2_m8192_workspace_bytes() -> dict[str, int]:
    """Reproduce the pinned B12X workspace allocation geometry in pure Python."""

    experts = 256
    hidden = 4096
    intermediate_per_rank = 1024
    top_k = 6
    max_tokens = 8192
    tile_m = tile_n = 128
    static_cutover_rows = 640
    scale_cols = ((hidden // 16 + 3) // 4) * 4
    routed_rows = max_tokens * top_k

    static_rows = min(routed_rows, static_cutover_rows)
    static_rows_padded = ((static_rows + tile_m - 1) // tile_m) * tile_m
    static = (
        experts * 4  # row_counts
        + experts * static_rows * 4  # token_map
        + experts * static_rows * 4  # token_weights
        + experts * static_rows * (hidden // 2)  # packed_input
        + experts * static_rows_padded * scale_cols  # packed_input_scale
        + 3 * 4  # barrier_count, barrier_epoch, active_expert_count
        + experts * 4  # weight_expert_ids
        + experts * 4  # global_to_local_expert
        + max(experts, static_rows) * 4  # compact_topk_ids
    )

    base_tiles = (routed_rows + tile_m - 1) // tile_m
    physical_tiles = base_tiles + min(experts, routed_rows) - 1
    dynamic_rows = physical_tiles * tile_m
    gate_tiles = (intermediate_per_rank + tile_n - 1) // tile_n
    max_tasks = physical_tiles * gate_tiles
    dynamic = (
        experts * 4  # row_counts
        + dynamic_rows * 4  # token_map
        + dynamic_rows * 4  # token_weights
        + dynamic_rows * (hidden // 2)  # packed_input
        + dynamic_rows * scale_cols  # packed_input_scale
        + 2 * 4  # barrier_count, barrier_epoch
        + experts * 4  # expert_write_rows
        + (experts + 1) * 4  # expert_tile_base
        + 5 * 4  # pair/producers/published/task head/task tail
        + 6 * max_tasks * 4  # task queue arrays
        + physical_tiles * 4  # tile_write_count
    )
    output = max_tokens * hidden * 2  # BF16 wrapper output
    return {
        "static_workspace_bytes": static,
        "dynamic_workspace_bytes": dynamic,
        "output_bytes": output,
        "total_bytes": static + dynamic + output,
    }


@dataclasses.dataclass
class PreparedWeights:
    w13: Any
    w13_sf_swizzled: Any
    w13_sf_mma: Any
    w2: Any
    w2_sf_swizzled: Any
    w2_sf_mma: Any
    alpha1: Any
    alpha2: Any
    fc2_input_scale: Any
    metadata: dict[str, Any]


def parse_positive_int_csv(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers: {value}") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all comma-separated values must be positive")
    # Preserve the requested order while avoiding accidental duplicate JIT/timing runs.
    return tuple(dict.fromkeys(parsed))


def percentile(values: Sequence[float], q: float) -> float:
    """Return a linearly interpolated percentile without NumPy."""

    if not values:
        raise ValueError("cannot compute percentile of an empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"percentile must be within [0, 1], got {q}")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_timing_runs(runs_ms: Sequence[Sequence[float]]) -> dict[str, Any]:
    if not runs_ms or any(not run for run in runs_ms):
        raise ValueError("timing runs must contain at least one non-empty repeat")
    flattened = [float(sample) for run in runs_ms for sample in run]
    repeat_medians = [statistics.median(run) for run in runs_ms]
    return {
        "samples": len(flattened),
        "repeats": len(runs_ms),
        "median_ms": statistics.median(flattened),
        "p95_ms": percentile(flattened, 0.95),
        "min_ms": min(flattened),
        "max_ms": max(flattened),
        "mean_ms": statistics.fmean(flattened),
        "repeat_median_ms": repeat_medians,
        "repeat_median_range_ms": [min(repeat_medians), max(repeat_medians)],
    }


def expected_pins(repo_root: pathlib.Path) -> dict[str, str]:
    lock_path = repo_root / "upstream.lock"
    if not lock_path.is_file():
        return {}
    result: dict[str, str] = {}
    for raw_line in lock_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key] = value.strip().strip("'\"")
    return result


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_checkpoint_contract(
    model_path: pathlib.Path,
    *,
    layer_idx: int,
    tp_size: int,
    tp_rank: int,
    require_keys: bool = True,
) -> tuple[Dsv4Shape, dict[str, Any]]:
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint config is missing: {config_path}")
    if not index_path.is_file():
        raise FileNotFoundError(f"checkpoint index is missing: {index_path}")

    raw_config = json.loads(config_path.read_text())
    config = raw_config.get("text_config", raw_config)
    shape = Dsv4Shape(
        hidden_size=int(config["hidden_size"]),
        intermediate_size=int(config["moe_intermediate_size"]),
        num_experts=int(config["n_routed_experts"]),
        top_k=int(config["num_experts_per_tok"]),
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    shape.validate()

    if str(config.get("model_type")) != "deepseek_v4":
        raise ValueError(f"expected model_type=deepseek_v4, got {config.get('model_type')!r}")
    if str(config.get("expert_dtype", "")).lower() != "fp4":
        raise ValueError(f"expected expert_dtype=fp4, got {config.get('expert_dtype')!r}")
    quant = config.get("quantization_config", {})
    if str(quant.get("moe_quant_algo", "")).upper() != "NVFP4":
        raise ValueError("checkpoint does not declare moe_quant_algo=NVFP4")
    if int(quant.get("group_size", 0)) != 16:
        raise ValueError("checkpoint does not use NVFP4 group_size=16")

    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map", {})
    if require_keys:
        required: list[str] = []
        for expert_id in (0, shape.num_experts - 1):
            prefix = f"layers.{layer_idx}.ffn.experts.{expert_id}"
            for projection in ("w1", "w3", "w2"):
                required.extend(
                    [
                        f"{prefix}.{projection}.weight",
                        f"{prefix}.{projection}.weight_scale",
                        f"{prefix}.{projection}.weight_scale_2",
                    ]
                )
        missing = [key for key in required if key not in weight_map]
        if missing:
            raise KeyError(f"checkpoint index lacks {len(missing)} required tensors: {missing[:4]}")

    metadata = {
        "model_path": str(model_path.resolve()),
        "config_sha256": _sha256_file(config_path),
        "index_sha256": _sha256_file(index_path),
        "indexed_tensor_count": len(weight_map),
        "indexed_shard_count": len(set(weight_map.values())),
        "producer": quant.get("producer"),
        "group_size": quant.get("group_size"),
        "moe_quant_algo": quant.get("moe_quant_algo"),
        "swiglu_limit": float(config.get("swiglu_limit", 10.0)),
        "layer_idx": layer_idx,
    }
    return shape, metadata


def tactic_for_shape(mode: str, m: int, top_k: int) -> str:
    routed_rows = m * top_k
    if mode == "w4a16":
        return "w4a16-native (internal micro/direct or grouped selector)"
    if routed_rows <= 40:
        return "micro"
    if routed_rows <= 640:
        return "static"
    return "dynamic"


def phase_for_m(m: int) -> str:
    """Classify the agreed routed-MoE matrix at its prefill boundary."""

    return "decode" if m < 128 else "prefill"


def evaluate_input_rms_contract(
    *,
    requested: float,
    observed_mean: float,
    observed_min: float,
    observed_max: float,
    relative_tolerance: float = INPUT_RMS_RELATIVE_TOLERANCE,
) -> dict[str, float | bool | None]:
    """Gate post-cast per-token RMS without requiring a CUDA test host."""

    finite = all(
        math.isfinite(value)
        for value in (requested, observed_mean, observed_min, observed_max)
    )
    ordered = observed_min <= observed_mean <= observed_max
    if not finite or requested <= 0 or not ordered:
        maximum_relative_error = None
    else:
        maximum_relative_error = max(
            abs(observed_min - requested),
            abs(observed_max - requested),
        ) / requested
    return {
        "requested": requested,
        "observed_mean": observed_mean if math.isfinite(observed_mean) else None,
        "observed_min": observed_min if math.isfinite(observed_min) else None,
        "observed_max": observed_max if math.isfinite(observed_max) else None,
        "relative_tolerance": relative_tolerance,
        "maximum_relative_error": maximum_relative_error,
        "finite": finite,
        "passed": bool(
            finite
            and ordered
            and relative_tolerance >= 0
            and maximum_relative_error is not None
            and maximum_relative_error <= relative_tolerance
        ),
    }


def summarize_unique_tensor_storage(
    torch: Any,
    roots: Sequence[Any],
) -> dict[str, int]:
    """Count tensor storages once, even when a workspace contains views."""

    stack = list(roots)
    visited_objects: set[int] = set()
    unique_storages: dict[tuple[str, int, int], int] = {}
    tensor_object_count = 0
    while stack:
        value = stack.pop()
        if value is None:
            continue
        object_id = id(value)
        if object_id in visited_objects:
            continue
        visited_objects.add(object_id)
        if torch.is_tensor(value):
            tensor_object_count += 1
            storage = value.untyped_storage()
            storage_bytes = int(storage.nbytes())
            storage_key = (
                str(value.device),
                int(storage.data_ptr()),
                storage_bytes,
            )
            unique_storages[storage_key] = storage_bytes
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            stack.extend(
                getattr(value, field.name) for field in dataclasses.fields(value)
            )
        elif isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            stack.extend(value)
    return {
        "tensor_object_count": tensor_object_count,
        "unique_storage_count": len(unique_storages),
        "unique_storage_bytes": sum(unique_storages.values()),
    }


def b12x_workspace_ceiling_bytes(
    shape: Dsv4Shape,
    max_num_tokens: int,
) -> int | None:
    """Return the reviewed ceiling only for the exact DSV4 TP=2 geometry."""

    if (
        shape.tp_size == 2
        and shape.hidden_size == 4096
        and shape.intermediate_size == 2048
        and shape.intermediate_size_per_rank == 1024
        and shape.num_experts == 256
        and shape.top_k == 6
        and max_num_tokens == 8192
    ):
        return DSV4_TP2_M8192_B12X_WRAPPER_CEILING_BYTES
    return None


def build_dry_run_plan(args: argparse.Namespace, repo_root: pathlib.Path) -> dict[str, Any]:
    if args.synthetic:
        shape = Dsv4Shape(
            num_experts=args.synthetic_experts or 256,
            tp_size=args.tp_size,
            tp_rank=args.tp_rank,
        )
        shape.validate()
        checkpoint = {"source": "synthetic-shape-only"}
    else:
        if args.model_path is None:
            raise ValueError("--model-path is required unless --synthetic is used")
        shape, checkpoint = read_checkpoint_contract(
            args.model_path,
            layer_idx=args.layer_idx,
            tp_size=args.tp_size,
            tp_rank=args.tp_rank,
        )

    modes = ("w4a4", "w4a16") if args.backend == "both" else (args.backend,)
    return {
        "schema_version": SCHEMA_VERSION,
        "dry_run": True,
        "shape": dataclasses.asdict(shape)
        | {"intermediate_size_per_rank": shape.intermediate_size_per_rank},
        "checkpoint": checkpoint,
        "matrix": [
            {
                "m": m,
                "phase": phase_for_m(m),
                "routed_rows": m * shape.top_k,
                "tactics": {mode: tactic_for_shape(mode, m, shape.top_k) for mode in modes},
                "correctness": m in args.correctness_m,
            }
            for m in args.m
        ],
        "activation_contract": {
            "input_rms": args.input_rms,
            "input_rms_relative_tolerance": INPUT_RMS_RELATIVE_TOLERANCE,
            "w4a4": {
                "name": "swigluoai_uninterleave",
                "alpha": args.swiglu_alpha,
                "beta": args.swiglu_beta,
                "limit": args.swiglu_limit,
            },
            "w4a16": {"name": "silu", "limit": args.swiglu_limit},
        },
        "timing": {
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "cuda_events": True,
            "cuda_graph": args.cuda_graph,
            "l2_flush_mib": args.l2_flush_mib,
        },
        "expected_pins": expected_pins(repo_root),
    }


class IndexedSafetensorLoader:
    """Minimal indexed reader, adapted from pinned B12X benchmark utilities."""

    def __init__(self, model_path: pathlib.Path):
        from safetensors import safe_open

        self.model_path = model_path
        self._safe_open = safe_open
        index = json.loads((model_path / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self._open_files: dict[str, Any] = {}

    def get_tensor(self, key: str) -> Any:
        shard = self.weight_map[key]
        handle = self._open_files.get(shard)
        if handle is None:
            handle = self._safe_open(str(self.model_path / shard), framework="pt")
            self._open_files[shard] = handle
        return handle.get_tensor(key)


def _packed_bytes(torch: Any, tensor: Any) -> Any:
    if tensor.dtype == torch.uint8:
        return tensor
    if tensor.element_size() != 1:
        raise TypeError(f"expected one-byte packed FP4 tensor, got {tensor.dtype}")
    return tensor.view(torch.uint8)


def _bake_expert_scales(torch: Any, scale: Any, global_scale: Any) -> Any:
    # Chunk by expert to avoid a temporary fp32 copy of the complete scale grid.
    for expert_id in range(scale.shape[0]):
        scale[expert_id] = (
            scale[expert_id].float() * global_scale[expert_id].float()
        ).to(torch.float8_e4m3fn)
    return scale


def _scale_to_mma(torch: Any, scale: Any, rows: int, cols: int) -> Any:
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    # ``scale`` is already the expert-leading swizzled storage.  The conversion
    # returns a strided logical view sharing that storage.
    experts, padded_rows, padded_cols = scale.shape
    if padded_rows < rows or padded_cols * 16 < cols:
        raise ValueError("swizzled scale storage is smaller than its logical matrix")
    return convert_sf_to_mma_layout(
        scale.reshape(experts * padded_rows, padded_cols),
        m=padded_rows,
        k=padded_cols * 16,
        num_groups=experts,
        sf_vec_size=16,
    )


def _sample_tensor_digest(torch: Any, tensor: Any, sample_bytes: int = 4096) -> str:
    flat = tensor.detach().view(torch.uint8).flatten()
    total = flat.numel()
    offsets = sorted({0, max(0, total // 2 - sample_bytes // 2), max(0, total - sample_bytes)})
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(str(tensor.dtype).encode())
    for offset in offsets:
        chunk = flat[offset : min(total, offset + sample_bytes)].cpu().tolist()
        digest.update(bytes(chunk))
    return digest.hexdigest()


def _finish_scale_preparation(
    torch: Any,
    *,
    w13: Any,
    w13_scale: Any,
    w13_scale_2: Any,
    w2: Any,
    w2_scale: Any,
    w2_scale_2: Any,
    shape: Dsv4Shape,
    metadata: dict[str, Any],
) -> PreparedWeights:
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        swizzle_blockscale,
    )

    # Match vLLM FlashInferB12xExperts: absorb ModelOpt's global weight scale
    # into block scales, then launch both modes with unit alphas.
    _bake_expert_scales(torch, w13_scale, w13_scale_2)
    _bake_expert_scales(torch, w2_scale, w2_scale_2)
    del w13_scale_2, w2_scale_2

    w13_sf_swizzled = swizzle_blockscale(w13_scale)
    w2_sf_swizzled = swizzle_blockscale(w2_scale)
    del w13_scale, w2_scale
    w13_sf_mma = _scale_to_mma(
        torch,
        w13_sf_swizzled,
        rows=2 * shape.intermediate_size_per_rank,
        cols=shape.hidden_size,
    )
    w2_sf_mma = _scale_to_mma(
        torch,
        w2_sf_swizzled,
        rows=shape.hidden_size,
        cols=shape.intermediate_size_per_rank,
    )
    alpha1 = torch.ones(shape.num_experts, dtype=torch.float32, device="cuda")
    alpha2 = torch.ones_like(alpha1)
    fc2_input_scale = torch.ones_like(alpha1)
    torch.cuda.synchronize()

    metadata["sample_fingerprints"] = {
        "w13": _sample_tensor_digest(torch, w13),
        "w2": _sample_tensor_digest(torch, w2),
        "w13_scale_swizzled": _sample_tensor_digest(torch, w13_sf_swizzled),
        "w2_scale_swizzled": _sample_tensor_digest(torch, w2_sf_swizzled),
    }
    metadata["same_source_weight_storage"] = True
    metadata["source_weight_data_ptrs"] = {
        "w13": int(w13.data_ptr()),
        "w2": int(w2.data_ptr()),
    }
    return PreparedWeights(
        w13=w13,
        w13_sf_swizzled=w13_sf_swizzled,
        w13_sf_mma=w13_sf_mma,
        w2=w2,
        w2_sf_swizzled=w2_sf_swizzled,
        w2_sf_mma=w2_sf_mma,
        alpha1=alpha1,
        alpha2=alpha2,
        fc2_input_scale=fc2_input_scale,
        metadata=metadata,
    )


def load_checkpoint_weights(
    torch: Any,
    model_path: pathlib.Path,
    shape: Dsv4Shape,
    *,
    layer_idx: int,
    checkpoint_metadata: dict[str, Any],
) -> PreparedWeights:
    loader = IndexedSafetensorLoader(model_path)
    device = torch.device("cuda")
    experts = shape.num_experts
    hidden = shape.hidden_size
    intermediate = shape.intermediate_size_per_rank
    tp_offset = shape.tp_rank * intermediate
    tp_packed_offset = shape.tp_rank * (intermediate // 2)
    tp_scale_offset = shape.tp_rank * (intermediate // 16)

    w1 = torch.empty(experts, intermediate, hidden // 2, dtype=torch.uint8, device=device)
    w3 = torch.empty_like(w1)
    w2 = torch.empty(experts, hidden, intermediate // 2, dtype=torch.uint8, device=device)
    s1 = torch.empty(
        experts, intermediate, hidden // 16, dtype=torch.float8_e4m3fn, device=device
    )
    s3 = torch.empty_like(s1)
    s2 = torch.empty(
        experts, hidden, intermediate // 16, dtype=torch.float8_e4m3fn, device=device
    )
    gs1 = torch.empty(experts, dtype=torch.float32, device=device)
    gs3 = torch.empty_like(gs1)
    gs2 = torch.empty_like(gs1)

    started = time.perf_counter()
    for expert_id in range(experts):
        prefix = f"layers.{layer_idx}.ffn.experts.{expert_id}"
        w1[expert_id] = _packed_bytes(
            torch, loader.get_tensor(f"{prefix}.w1.weight")
        ).narrow(0, tp_offset, intermediate).to(device)
        w3[expert_id] = _packed_bytes(
            torch, loader.get_tensor(f"{prefix}.w3.weight")
        ).narrow(0, tp_offset, intermediate).to(device)
        w2[expert_id] = _packed_bytes(
            torch, loader.get_tensor(f"{prefix}.w2.weight")
        ).narrow(1, tp_packed_offset, intermediate // 2).to(device)

        s1[expert_id] = loader.get_tensor(f"{prefix}.w1.weight_scale").narrow(
            0, tp_offset, intermediate
        ).to(device)
        s3[expert_id] = loader.get_tensor(f"{prefix}.w3.weight_scale").narrow(
            0, tp_offset, intermediate
        ).to(device)
        s2[expert_id] = loader.get_tensor(f"{prefix}.w2.weight_scale").narrow(
            1, tp_scale_offset, intermediate // 16
        ).to(device)
        gs1[expert_id] = loader.get_tensor(f"{prefix}.w1.weight_scale_2").to(device)
        gs3[expert_id] = loader.get_tensor(f"{prefix}.w3.weight_scale_2").to(device)
        gs2[expert_id] = loader.get_tensor(f"{prefix}.w2.weight_scale_2").to(device)
        if (expert_id + 1) % 32 == 0 or expert_id + 1 == experts:
            print(f"  loaded experts {expert_id + 1}/{experts}", flush=True)
    torch.cuda.synchronize()

    mismatch = float((gs1 - gs3).abs().max().item())
    if mismatch:
        print(
            f"WARNING: w1/w3 weight_scale_2 max mismatch is {mismatch:.6g}; "
            "matching vLLM by using w1 scale for fused W13",
            file=sys.stderr,
        )
    # FlashInfer B12X expects [up/w3, gate/w1] source order.
    w13 = torch.cat((w3, w1), dim=1).contiguous()
    w13_scale = torch.cat((s3, s1), dim=1).contiguous()
    del w1, w3, s1, s3, gs3
    checkpoint_metadata.update(
        {
            "load_seconds": time.perf_counter() - started,
            "w13_layout": "w13 (up/w3, gate/w1; B12X up_gate)",
            "w1_w3_scale2_max_mismatch": mismatch,
            "tp_offset": tp_offset,
        }
    )
    return _finish_scale_preparation(
        torch,
        w13=w13,
        w13_scale=w13_scale,
        w13_scale_2=gs1,
        w2=w2,
        w2_scale=s2,
        w2_scale_2=gs2,
        shape=shape,
        metadata=checkpoint_metadata,
    )


def make_synthetic_weights(
    torch: Any,
    shape: Dsv4Shape,
) -> PreparedWeights:
    device = torch.device("cuda")
    experts = shape.num_experts
    hidden = shape.hidden_size
    intermediate = shape.intermediate_size_per_rank
    w13 = torch.full(
        (experts, 2 * intermediate, hidden // 2),
        0x11,
        dtype=torch.uint8,
        device=device,
    )
    w2 = torch.full(
        (experts, hidden, intermediate // 2),
        0x11,
        dtype=torch.uint8,
        device=device,
    )
    scale_value = 2.0**-7
    w13_scale = torch.full(
        (experts, 2 * intermediate, hidden // 16),
        scale_value,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    w2_scale = torch.full(
        (experts, hidden, intermediate // 16),
        scale_value,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    ones = torch.ones(experts, dtype=torch.float32, device=device)
    return _finish_scale_preparation(
        torch,
        w13=w13,
        w13_scale=w13_scale,
        w13_scale_2=ones.clone(),
        w2=w2,
        w2_scale=w2_scale,
        w2_scale_2=ones.clone(),
        shape=shape,
        metadata={
            "source": "synthetic-shape-only",
            "packed_fill": "0x11",
            "logical_scale": scale_value,
            "w13_layout": "w13 (up/w3, gate/w1; B12X up_gate)",
        },
    )


def package_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def runtime_provenance(torch: Any, repo_root: pathlib.Path) -> dict[str, Any]:
    import b12x
    import flashinfer
    import vllm
    from flashinfer.fused_moe import B12xMoEWrapper
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
        select_sm120_moe_backend,
    )
    from b12x.moe.fused.w4a16.kernel import run_w4a16_moe

    capability = torch.cuda.get_device_capability()
    return {
        "expected_pins": expected_pins(repo_root),
        "packages": {
            "torch": {"version": torch.__version__, "module": torch.__file__},
            "vllm": {"version": getattr(vllm, "__version__", None), "module": vllm.__file__},
            "flashinfer": {
                "version": package_version("flashinfer-python", "flashinfer"),
                "module": flashinfer.__file__,
            },
            "b12x": {"version": package_version("b12x"), "module": b12x.__file__},
        },
        "cuda": {
            "torch_cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(),
            "capability": list(capability),
            "device_count": torch.cuda.device_count(),
        },
        "backend_symbols": {
            "w4a4": f"{B12xMoEWrapper.__module__}.{B12xMoEWrapper.__qualname__}",
            "w4a16": f"{run_w4a16_moe.__module__}.{run_w4a16_moe.__qualname__}",
            "selector": (
                f"{select_sm120_moe_backend.__module__}."
                f"{select_sm120_moe_backend.__qualname__}"
            ),
            "w4a4_wrapper_source": inspect.getsourcefile(B12xMoEWrapper),
            "w4a16_kernel_source": inspect.getsourcefile(run_w4a16_moe),
            "w4a4_run_signature": str(inspect.signature(B12xMoEWrapper.run)),
            "w4a16_run_signature": str(inspect.signature(run_w4a16_moe)),
        },
    }


def make_routes(
    torch: Any,
    shape: Dsv4Shape,
    m: int,
    *,
    routing: str,
    seed: int,
    input_rms: float,
) -> tuple[Any, Any, Any]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    x_fp32 = torch.randn(
        (m, shape.hidden_size),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    # A routed expert consumes post-RMSNorm activations, whose per-token RMS
    # is near one. Normalizing every row keeps M=1 and large-M correctness
    # cases equally representative and exercises the calibrated A4/clamp path
    # rather than the old 1/sqrt(K) (~0.0156) smoke-test distribution.
    row_rms = x_fp32.square().mean(dim=-1, keepdim=True).sqrt()
    x = (x_fp32 / row_rms * input_rms).to(torch.bfloat16)
    if routing == "balanced":
        token = torch.arange(m, device="cuda", dtype=torch.int64).unsqueeze(1)
        lane = torch.arange(shape.top_k, device="cuda", dtype=torch.int64).unsqueeze(0)
        ids = (token * shape.top_k + lane).remainder(shape.num_experts).to(torch.int32)
        weights = torch.full(
            (m, shape.top_k),
            1.0 / shape.top_k,
            device="cuda",
            dtype=torch.float32,
        )
    elif routing == "hot":
        ids = torch.arange(shape.top_k, device="cuda", dtype=torch.int32).expand(m, -1)
        weights = torch.full(
            (m, shape.top_k),
            1.0 / shape.top_k,
            device="cuda",
            dtype=torch.float32,
        )
    elif routing == "random":
        logits = torch.randn(
            (m, shape.num_experts),
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        selected, ids = torch.topk(logits, shape.top_k, dim=-1)
        weights = torch.softmax(selected, dim=-1)
        ids = ids.to(torch.int32)
    else:
        raise ValueError(f"unsupported routing pattern {routing!r}")
    return x.contiguous(), ids.contiguous(), weights.contiguous()


def compare_tensors(torch: Any, actual: Any, reference: Any) -> dict[str, float | int | bool]:
    actual_f = actual.float()
    reference_f = reference.float()
    diff = actual_f - reference_f
    abs_diff = diff.abs()
    rmse = diff.square().mean().sqrt()
    reference_rms = reference_f.square().mean().sqrt()
    denominator = actual_f.norm() * reference_f.norm()
    cosine = (actual_f.flatten().dot(reference_f.flatten()) / denominator.clamp_min(1e-20))
    relative = abs_diff / reference_f.abs().clamp_min(1e-5)
    finite = torch.isfinite(actual_f)
    return {
        "finite": bool(finite.all().item()),
        "nonfinite_count": int((~finite).sum().item()),
        "max_abs": float(abs_diff.max().item()),
        "mean_abs": float(abs_diff.mean().item()),
        "rmse": float(rmse.item()),
        "normalized_rmse": float((rmse / reference_rms.clamp_min(1e-20)).item()),
        "cosine": float(cosine.item()),
        "relative_p50": float(torch.quantile(relative, 0.50).item()),
        "relative_p95": float(torch.quantile(relative, 0.95).item()),
        "relative_p99": float(torch.quantile(relative, 0.99).item()),
    }


def measure_cuda_events(
    torch: Any,
    fn: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
    repeats: int,
    flush_l2: Callable[[], None] | None,
) -> dict[str, Any]:
    runs: list[list[float]] = []
    for _ in range(repeats):
        for _ in range(warmup):
            if flush_l2 is not None:
                flush_l2()
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for start, end in zip(starts, ends, strict=True):
            if flush_l2 is not None:
                flush_l2()
            start.record()
            fn()
            end.record()
        torch.cuda.synchronize()
        runs.append([start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)])
    return summarize_timing_runs(runs)


def add_derived_performance(
    stats: dict[str, Any], shape: Dsv4Shape, m: int
) -> dict[str, Any]:
    seconds = stats["median_ms"] / 1000.0
    # Gated MoE local-rank matmuls: FC1 [K -> 2I] plus FC2 [I -> K].
    flops = 6.0 * m * shape.top_k * shape.hidden_size * shape.intermediate_size_per_rank
    stats["tokens_per_second"] = m / seconds
    stats["routed_rows_per_second"] = (m * shape.top_k) / seconds
    stats["effective_tflops"] = flops / seconds / 1.0e12
    return stats


def capture_graph(torch: Any, fn: Callable[[], Any]) -> tuple[Callable[[], Any], Any, Any]:
    # Compile/cache everything before entering capture.
    for _ in range(3):
        output = fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()

    def replay() -> Any:
        graph.replay()
        return output

    replay()
    torch.cuda.synchronize()
    return replay, output, graph


def _make_w4a4_runner(
    torch: Any,
    weights: PreparedWeights,
    shape: Dsv4Shape,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    from flashinfer.fused_moe import B12xMoEWrapper

    wrapper = B12xMoEWrapper(
        num_experts=shape.num_experts,
        top_k=shape.top_k,
        hidden_size=shape.hidden_size,
        intermediate_size=shape.intermediate_size_per_rank,
        use_cuda_graph=True,
        max_num_tokens=max(args.m),
        num_local_experts=shape.num_experts,
        output_dtype=torch.bfloat16,
        device="cuda",
        activation="swigluoai_uninterleave",
        swiglu_alpha=args.swiglu_alpha,
        swiglu_beta=args.swiglu_beta,
        swiglu_limit=args.swiglu_limit,
        quant_mode="nvfp4",
        source_format="modelopt",
    )
    required_roots_present = all(
        root is not None
        for root in (
            wrapper._static_workspace,
            wrapper._dynamic_workspace,
            wrapper._moe_output,
        )
    )
    workspace_memory: dict[str, Any] = summarize_unique_tensor_storage(
        torch,
        (
            wrapper._static_workspace,
            wrapper._dynamic_workspace,
            wrapper._moe_output,
        ),
    )
    workspace_ceiling = b12x_workspace_ceiling_bytes(shape, max(args.m))
    contract_applies = workspace_ceiling is not None
    workspace_passed = None
    if workspace_ceiling is not None:
        workspace_passed = bool(
            required_roots_present
            and workspace_memory["unique_storage_bytes"] <= workspace_ceiling
        )
    workspace_memory |= {
        "contract_applies": contract_applies,
        "required_roots_present": required_roots_present,
        "ceiling_bytes": workspace_ceiling,
        "passed": workspace_passed,
    }
    proof = {
        "requested": "w4a4",
        "implementation": f"{wrapper.__class__.__module__}.{wrapper.__class__.__qualname__}",
        "normalized_quant_mode": wrapper.quant_mode,
        "activation_precision": wrapper.activation_precision,
        "activation": wrapper.activation,
        "swiglu_alpha": wrapper.swiglu_alpha,
        "swiglu_beta": wrapper.swiglu_beta,
        "swiglu_limit": wrapper.swiglu_limit,
        "source_format": wrapper.source_format,
        "static_workspace": type(wrapper._static_workspace).__name__,
        "dynamic_workspace": type(wrapper._dynamic_workspace).__name__,
        "workspace_memory": workspace_memory,
    }
    return wrapper, proof


def _prepare_w4a16(
    torch: Any,
    weights: PreparedWeights,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    from b12x.moe.fused.w4a16.prepare import prepare_w4a16_modelopt_native_weights

    prepared = prepare_w4a16_modelopt_native_weights(
        weights.w13,
        weights.w13_sf_swizzled,
        weights.alpha1,
        weights.w2,
        weights.w2_sf_swizzled,
        weights.alpha2,
        activation="silu",
        params_dtype=torch.bfloat16,
        source_format="modelopt_nvfp4",
        # FlashInfer consumes the shared physical tensor as [up/w3, gate/w1].
        # B12X calls that layout "w13"/"up_gate"; "w31" means the opposite
        # [gate, up] order and would invalidate the same-weight comparison.
        w13_layout=B12X_W13_LAYOUT,
    )
    proof = {
        "requested": "w4a16",
        "implementation": "b12x.moe.fused.w4a16.kernel.run_w4a16_moe",
        "normalized_quant_mode": "w4a16",
        "weight_layout": getattr(prepared, "weight_layout", None),
        "scale_format": getattr(prepared, "scale_format", None),
        "source_format": getattr(prepared, "source_format", None),
        "w13_layout": getattr(prepared, "w13_layout", None),
        "activation": "silu",
        "swiglu_limit": args.swiglu_limit,
        "same_source_w13": int(prepared.w13.data_ptr()) == int(weights.w13.data_ptr()),
        "same_source_w2": int(prepared.w2.data_ptr()) == int(weights.w2.data_ptr()),
    }
    return prepared, proof


def _make_w4a16_launch(
    torch: Any,
    prepared: Any,
    x: Any,
    topk_ids: Any,
    topk_weights: Any,
    args: argparse.Namespace,
) -> tuple[Callable[[], Any], Any]:
    from b12x.moe.fused.w4a16.kernel import run_w4a16_moe
    from b12x.moe.fused.w4a16.prepare import make_w4a16_packed_buffers

    buffers = make_w4a16_packed_buffers(
        prepared,
        m=x.shape[0],
        topk=topk_ids.shape[1],
        dtype=torch.bfloat16,
        device=x.device,
    )

    def launch() -> Any:
        return run_w4a16_moe(
            x,
            prepared,
            topk_weights,
            topk_ids,
            activation="silu",
            fast_math=args.fast_math,
            swiglu_limit=args.swiglu_limit,
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=buffers.output,
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
        )

    return launch, buffers


def run_benchmark(args: argparse.Namespace, repo_root: pathlib.Path) -> int:
    import torch
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
        select_sm120_moe_backend,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required (use --dry-run on a development host)")
    capability = torch.cuda.get_device_capability()
    accepted = {(12, 1)} | ({(12, 0)} if args.allow_sm120 else set())
    if capability not in accepted:
        raise RuntimeError(
            f"this harness requires SM121{'/SM120' if args.allow_sm120 else ''}; "
            f"detected compute capability {capability}"
        )

    if args.synthetic:
        shape = Dsv4Shape(
            num_experts=args.synthetic_experts or 256,
            tp_size=args.tp_size,
            tp_rank=args.tp_rank,
        )
        shape.validate()
        checkpoint_metadata = {"source": "synthetic-shape-only"}
        print("Creating synthetic real-shape packed NVFP4 weights...", flush=True)
        weights = make_synthetic_weights(torch, shape)
    else:
        if args.model_path is None:
            raise ValueError("--model-path is required unless --synthetic is used")
        shape, checkpoint_metadata = read_checkpoint_contract(
            args.model_path,
            layer_idx=args.layer_idx,
            tp_size=args.tp_size,
            tp_rank=args.tp_rank,
        )
        print(
            f"Loading NVIDIA NVFP4 layer {args.layer_idx}, TP rank {args.tp_rank}/"
            f"{args.tp_size} from {args.model_path}...",
            flush=True,
        )
        weights = load_checkpoint_weights(
            torch,
            args.model_path,
            shape,
            layer_idx=args.layer_idx,
            checkpoint_metadata=checkpoint_metadata,
        )

    provenance = runtime_provenance(torch, repo_root)
    modes = ("w4a4", "w4a16") if args.backend == "both" else (args.backend,)
    backend_proof: dict[str, Any] = {}
    w4a4_wrapper = None
    w4a16_prepared = None
    if "w4a4" in modes:
        w4a4_wrapper, backend_proof["w4a4"] = _make_w4a4_runner(
            torch, weights, shape, args
        )
    if "w4a16" in modes:
        w4a16_prepared, backend_proof["w4a16"] = _prepare_w4a16(
            torch, weights, args
        )
    torch.cuda.synchronize()

    print(
        f"GPU={provenance['cuda']['device_name']} capability={capability}; "
        f"K={shape.hidden_size}, I/rank={shape.intermediate_size_per_rank}, "
        f"E={shape.num_experts}, top-k={shape.top_k}"
    )
    print(
        "Backends: "
        + ", ".join(
            f"{mode}={backend_proof[mode]['implementation']}" for mode in modes
        )
    )
    print(
        "Activation match: W4A4 swigluoai_uninterleave(alpha="
        f"{args.swiglu_alpha:g}, beta={args.swiglu_beta:g}, limit={args.swiglu_limit:g}); "
        f"W4A16 silu(limit={args.swiglu_limit:g})"
    )

    flush_buffer = None
    flush_l2 = None
    if args.l2_flush_mib:
        flush_buffer = torch.empty(
            args.l2_flush_mib << 20, dtype=torch.uint8, device="cuda"
        )

        def flush_l2() -> None:
            assert flush_buffer is not None
            flush_buffer.bitwise_not_()

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "shape": dataclasses.asdict(shape)
        | {"intermediate_size_per_rank": shape.intermediate_size_per_rank},
        "checkpoint": weights.metadata,
        "provenance": provenance,
        "backend_proof": backend_proof,
        "settings": {
            "m": list(args.m),
            "correctness_m": list(args.correctness_m),
            "routing": args.routing,
            "seed": args.seed,
            "input_rms": args.input_rms,
            "input_rms_relative_tolerance": INPUT_RMS_RELATIVE_TOLERANCE,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "cuda_graph": args.cuda_graph,
            "fast_math": args.fast_math,
            "l2_flush_mib": args.l2_flush_mib,
            "numeric_min_cosine": args.numeric_min_cosine,
            "numeric_max_nrmse": args.numeric_max_nrmse,
        },
        "results": [],
        "failures": [],
    }
    if "w4a4" in backend_proof:
        workspace_memory = backend_proof["w4a4"]["workspace_memory"]
        if (
            workspace_memory["contract_applies"]
            and not workspace_memory["passed"]
        ):
            report["failures"].append(
                {
                    "kind": "workspace_memory",
                    "contract": workspace_memory,
                }
            )

    for m in args.m:
        x, topk_ids, topk_weights = make_routes(
            torch,
            shape,
            m,
            routing=args.routing,
            seed=args.seed + m,
            input_rms=args.input_rms,
        )
        per_token_rms = x.float().square().mean(dim=-1).sqrt()
        input_rms_contract = evaluate_input_rms_contract(
            requested=args.input_rms,
            observed_mean=float(per_token_rms.mean().item()),
            observed_min=float(per_token_rms.min().item()),
            observed_max=float(per_token_rms.max().item()),
        )
        row: dict[str, Any] = {
            "m": m,
            "phase": phase_for_m(m),
            "routed_rows": m * shape.top_k,
            "input_rms_contract": input_rms_contract,
            "modes": {},
        }
        if not input_rms_contract["passed"]:
            report["failures"].append(
                {
                    "kind": "input_rms",
                    "m": m,
                    "contract": input_rms_contract,
                }
            )
        launches: dict[str, Callable[[], Any]] = {}
        keepalive: list[Any] = []
        if w4a4_wrapper is not None:

            def launch_w4a4(
                wrapper: Any = w4a4_wrapper,
                x_local: Any = x,
                ids_local: Any = topk_ids,
                route_weights_local: Any = topk_weights,
            ) -> Any:
                return wrapper.run(
                    x=x_local,
                    w1_weight=weights.w13,
                    w1_weight_sf=weights.w13_sf_mma,
                    w1_alpha=weights.alpha1,
                    fc2_input_scale=weights.fc2_input_scale,
                    w2_weight=weights.w2,
                    w2_weight_sf=weights.w2_sf_mma,
                    w2_alpha=weights.alpha2,
                    token_selected_experts=ids_local,
                    token_final_scales=route_weights_local,
                )

            launches["w4a4"] = launch_w4a4
        if w4a16_prepared is not None:
            launch_w4a16, buffers = _make_w4a16_launch(
                torch, w4a16_prepared, x, topk_ids, topk_weights, args
            )
            launches["w4a16"] = launch_w4a16
            keepalive.append(buffers)

        # Compile and retain matched eager outputs only for requested correctness M.
        eager_outputs: dict[str, Any] = {}
        for mode, launch in launches.items():
            output = launch()
            torch.cuda.synchronize()
            if m in args.correctness_m:
                eager_outputs[mode] = output.clone()

        if "w4a4" in eager_outputs and "w4a16" in eager_outputs:
            metrics = compare_tensors(
                torch, eager_outputs["w4a4"], eager_outputs["w4a16"]
            )
            row["w4a4_vs_w4a16"] = metrics
            passed = (
                bool(metrics["finite"])
                and float(metrics["cosine"]) >= args.numeric_min_cosine
                and float(metrics["normalized_rmse"]) <= args.numeric_max_nrmse
            )
            row["numeric_gate_passed"] = passed
            if not passed:
                report["failures"].append(
                    {
                        "kind": "numeric",
                        "m": m,
                        "cosine": metrics["cosine"],
                        "normalized_rmse": metrics["normalized_rmse"],
                    }
                )

        for mode, launch in launches.items():
            if mode == "w4a4":
                selected = select_sm120_moe_backend(
                    num_tokens=m,
                    num_topk=shape.top_k,
                    quant_mode="nvfp4",
                )
                tactic = "micro" if selected == "static" and m * shape.top_k <= 40 else selected
            else:
                selected = "w4a16"
                tactic = tactic_for_shape(mode, m, shape.top_k)
            print(
                f"M={m:5d} {row['phase']:7s} {mode:5s} tactic={tactic:51s}",
                end="",
                flush=True,
            )
            eager_stats = add_derived_performance(
                measure_cuda_events(
                    torch,
                    launch,
                    warmup=args.warmup,
                    iters=args.iters,
                    repeats=args.repeats,
                    flush_l2=flush_l2,
                ),
                shape,
                m,
            )
            mode_result: dict[str, Any] = {
                "selector": selected,
                "tactic": tactic,
                "eager": eager_stats,
            }
            print(
                f" eager={eager_stats['median_ms'] * 1000:9.1f} us "
                f"p95={eager_stats['p95_ms'] * 1000:9.1f} us",
                end="",
                flush=True,
            )

            if args.cuda_graph:
                try:
                    replay, graph_output, graph = capture_graph(torch, launch)
                    keepalive.extend((graph_output, graph))
                    graph_stats = add_derived_performance(
                        measure_cuda_events(
                            torch,
                            replay,
                            warmup=args.warmup,
                            iters=args.iters,
                            repeats=args.repeats,
                            flush_l2=flush_l2,
                        ),
                        shape,
                        m,
                    )
                    mode_result["cuda_graph"] = graph_stats
                    mode_result["cuda_graph_status"] = "captured"
                    if mode in eager_outputs:
                        replay()
                        torch.cuda.synchronize()
                        mode_result["graph_vs_eager"] = compare_tensors(
                            torch, graph_output, eager_outputs[mode]
                        )
                    print(
                        f" graph={graph_stats['median_ms'] * 1000:9.1f} us "
                        f"p95={graph_stats['p95_ms'] * 1000:9.1f} us",
                        end="",
                        flush=True,
                    )
                except Exception as exc:
                    mode_result["cuda_graph_status"] = "failed"
                    mode_result["cuda_graph_error"] = f"{type(exc).__name__}: {exc}"
                    print(f" graph=FAILED({type(exc).__name__})", end="", flush=True)
                    if args.require_graphs:
                        report["failures"].append(
                            {"kind": "cuda_graph", "m": m, "mode": mode, "error": str(exc)}
                        )
            row["modes"][mode] = mode_result
            print()

        if "w4a4" in row["modes"] and "w4a16" in row["modes"]:
            row["speedup_w4a4_over_w4a16"] = {}
            for timing_kind in ("eager", "cuda_graph"):
                a4 = row["modes"]["w4a4"].get(timing_kind)
                a16 = row["modes"]["w4a16"].get(timing_kind)
                if a4 and a16:
                    row["speedup_w4a4_over_w4a16"][timing_kind] = (
                        a16["median_ms"] / a4["median_ms"]
                    )
                    print(
                        f"  W4A4 speedup ({timing_kind}): "
                        f"{row['speedup_w4a4_over_w4a16'][timing_kind]:.3f}x"
                    )
        report["results"].append(row)
        del eager_outputs, launches, keepalive, x, topk_ids, topk_weights

    report["memory"] = {
        "allocated_gib": torch.cuda.memory_allocated() / (1 << 30),
        "reserved_gib": torch.cuda.memory_reserved() / (1 << 30),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1 << 30),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {args.output}")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))

    effective_failures = [
        failure
        for failure in report["failures"]
        if failure["kind"] != "numeric" or not args.no_correctness_gate
    ]
    if effective_failures:
        print(f"FAILED: {len(effective_failures)} gate(s)", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FlashInfer/B12X NVFP4 W4A4 against same-weight B12X "
            "W4A16 at DeepSeek V4 Flash routed-MoE shapes on SM121."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model-path", type=pathlib.Path, help="NVIDIA DSV4 NVFP4 checkpoint")
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="Use full-shape synthetic packed weights (kernel smoke/perf only)",
    )
    parser.add_argument(
        "--synthetic-experts",
        type=int,
        default=None,
        help="Override E only for a faster synthetic smoke test",
    )
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--backend", choices=("both", "w4a4", "w4a16"), default="both")
    parser.add_argument(
        "--m",
        type=parse_positive_int_csv,
        default=DEFAULT_M_VALUES,
        help="Comma-separated decode/prefill token counts",
    )
    parser.add_argument(
        "--correctness-m",
        type=parse_positive_int_csv,
        default=DEFAULT_CORRECTNESS_M,
        help="Comma-separated M values for W4A4-vs-W4A16 numerical checks",
    )
    parser.add_argument("--routing", choices=("balanced", "random", "hot"), default="balanced")
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument(
        "--input-rms",
        type=float,
        default=1.0,
        help="Per-token RMS of synthetic hidden states (post-RMSNorm default: 1.0)",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-graphs", action="store_true")
    # The pinned FlashInfer public B12X wrapper fixes fast-math on.  Keep the
    # matched W4A16 comparator on the same setting instead of exposing a flag
    # that could silently make the two closures different.
    parser.set_defaults(fast_math=True)
    parser.add_argument(
        "--l2-flush-mib",
        type=int,
        default=0,
        help="Touch this many MiB before each launch; timing starts after the flush",
    )
    parser.add_argument("--swiglu-alpha", type=float, default=1.0)
    parser.add_argument("--swiglu-beta", type=float, default=0.0)
    parser.add_argument("--swiglu-limit", type=float, default=10.0)
    parser.add_argument("--numeric-min-cosine", type=float, default=0.98)
    parser.add_argument("--numeric-max-nrmse", type=float, default=0.25)
    parser.add_argument("--no-correctness-gate", action="store_true")
    parser.add_argument("--allow-sm120", action="store_true")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the matrix/checkpoint contract without importing CUDA libraries",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.synthetic_experts is not None and not args.synthetic:
        raise ValueError("--synthetic-experts requires --synthetic")
    if args.synthetic_experts is not None and args.synthetic_experts < 1:
        raise ValueError("--synthetic-experts must be positive")
    if args.layer_idx < 0:
        raise ValueError("--layer-idx must be non-negative")
    if args.warmup < 0 or args.iters <= 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative; iters/repeats must be positive")
    if args.l2_flush_mib < 0:
        raise ValueError("--l2-flush-mib must be non-negative")
    if not math.isfinite(args.input_rms) or args.input_rms <= 0:
        raise ValueError("--input-rms must be positive and finite")
    if not math.isfinite(args.swiglu_limit) or args.swiglu_limit <= 0:
        raise ValueError("--swiglu-limit must be positive and finite")
    if args.backend in {"both", "w4a16"} and (
        args.swiglu_alpha != 1.0 or args.swiglu_beta != 0.0
    ):
        raise ValueError(
            "the activation-matched W4A16 comparator requires "
            "--swiglu-alpha 1 and --swiglu-beta 0"
        )
    if not 0.0 <= args.numeric_min_cosine <= 1.0:
        raise ValueError("--numeric-min-cosine must be within [0, 1]")
    if args.numeric_max_nrmse < 0:
        raise ValueError("--numeric-max-nrmse must be non-negative")
    missing = sorted(set(args.correctness_m) - set(args.m))
    if missing:
        raise ValueError(f"--correctness-m values must be present in --m; missing {missing}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        if args.dry_run:
            print(json.dumps(build_dry_run_plan(args, repo_root), indent=2, sort_keys=True))
            return 0
        return run_benchmark(args, repo_root)
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
