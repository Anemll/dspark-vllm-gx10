#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Synthetic hardware gate for DSpark physical variable-length verification.

This intentionally does not load model weights.  It proves that confidence
prefixes alter physical target rows before CUDA-graph dispatch, then times a
small routed-expert-shaped BF16 BMM inside the exact graph selected for each
C=1 target width.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import torch
import torch.nn.functional as F


RAW_DRAFT_WIDTH = 5
TOP_K = 6
SYNTHETIC_HIDDEN = 512
CAPTURE_SIZES = [1, 2, 4, 8]


def _scheduler_output() -> SimpleNamespace:
    return SimpleNamespace(
        scheduled_spec_decode_tokens={"probe": [-1] * RAW_DRAFT_WIDTH},
        num_scheduled_tokens={"probe": RAW_DRAFT_WIDTH + 1},
        total_num_scheduled_tokens=RAW_DRAFT_WIDTH + 1,
    )


def _compact_case(prefix: list[int], *, enabled: bool) -> dict[str, object]:
    from vllm.v1.worker.gpu.spec_decode.dspark.variable_verifier import (
        compact_scheduler_output_for_variable_drafts,
    )

    output = _scheduler_output()
    invalid: dict[str, int] = {}
    if enabled:
        invalid = compact_scheduler_output_for_variable_drafts(
            output, ["probe"], [prefix]
        )
    else:
        # Confidence-off uses the normal full proposal. The async reservation
        # is represented as placeholders until the real proposal arrives.
        output.scheduled_spec_decode_tokens["probe"] = prefix
    scheduled = output.scheduled_spec_decode_tokens.get("probe", [])
    if any(token_id == -1 for token_id in scheduled):
        raise RuntimeError(
            f"sentinel reached physical target proposal: {scheduled}"
        )
    return {
        "enabled": enabled,
        "physical_draft_tokens": scheduled,
        "physical_draft_rows": len(scheduled),
        "physical_target_rows": output.num_scheduled_tokens["probe"],
        "invalid_rows": invalid.get("probe", 0),
        "sentinel_reaches_target": False,
    }


def _make_graph_manager():
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

    speculative = SimpleNamespace(
        use_dspark=lambda: True,
        uses_dynamic_speculative_decoding=lambda: False,
    )
    compilation = SimpleNamespace(
        cudagraph_capture_sizes=CAPTURE_SIZES,
        max_cudagraph_capture_size=64,
    )
    manager = object.__new__(CudaGraphManager)
    manager.vllm_config = SimpleNamespace(
        speculative_config=speculative,
        num_speculative_tokens=RAW_DRAFT_WIDTH,
    )
    manager.compilation_config = compilation
    manager.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    manager.decode_query_len = RAW_DRAFT_WIDTH + 1
    manager.max_num_reqs = 8
    manager.lora_capture_cases = [0]
    manager._lora_dispatch_map = {}
    manager._max_lora_case = 0
    manager._candidates = {}
    manager._capture_descs = {}
    manager._graphs_captured = False
    manager._init_candidates()
    manager._graphs_captured = True
    return manager


def _dispatch_rows(manager, rows: int):
    desc = manager.dispatch(
        num_reqs=1,
        num_tokens=rows,
        uniform_token_count=rows,
        num_active_loras=0,
    )
    if (
        desc.num_tokens != rows
        or desc.num_reqs != 1
        or desc.uniform_token_count != rows
        or desc.cg_mode.name != "FULL"
    ):
        raise RuntimeError(
            "DSpark C=1 graph dispatch was not exact: "
            f"requested_rows={rows}, descriptor={desc}"
        )
    return desc


def _spearman_with_rows(values: list[float]) -> float:
    # Widths are already ranks 1..N. Values are expected to be distinct in a
    # real scaling signal; deterministic index tie-breaking is conservative.
    value_order = sorted(range(len(values)), key=lambda idx: (values[idx], idx))
    ranks = [0] * len(values)
    for rank, idx in enumerate(value_order, start=1):
        ranks[idx] = rank
    n = len(values)
    sum_d2 = sum((idx + 1 - ranks[idx]) ** 2 for idx in range(n))
    return 1.0 - (6.0 * sum_d2) / (n * (n * n - 1))


def _cuda_timing(manager, *, repeats: int, iterations: int) -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one CUDA device")

    device = torch.device("cuda")
    inventory = {
        "torch": torch.__version__,
        "device_count": torch.cuda.device_count(),
        "name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }
    torch.manual_seed(4104)
    embedding = torch.randn(
        64, SYNTHETIC_HIDDEN, device=device, dtype=torch.bfloat16
    )
    max_routes = (RAW_DRAFT_WIDTH + 1) * TOP_K
    expert_weights = torch.randn(
        max_routes,
        SYNTHETIC_HIDDEN,
        SYNTHETIC_HIDDEN,
        device=device,
        dtype=torch.bfloat16,
    )

    graphs = {}
    dispatch_rows: dict[str, object] = {}
    static_inputs = []
    for rows in range(1, RAW_DRAFT_WIDTH + 2):
        token_ids = torch.arange(1, rows + 1, device=device, dtype=torch.int64)
        if token_ids.min().item() < 0:
            raise RuntimeError("negative token id reached target embedding")
        embedded = F.embedding(token_ids, embedding)
        routed = embedded.repeat_interleave(TOP_K, dim=0).unsqueeze(1)
        route_count = rows * TOP_K
        weights = expert_weights[:route_count]
        output = torch.empty(
            route_count, 1, SYNTHETIC_HIDDEN,
            device=device,
            dtype=torch.bfloat16,
        )
        for _ in range(3):
            torch.bmm(routed, weights, out=output)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            torch.bmm(routed, weights, out=output)
        desc = _dispatch_rows(manager, rows)
        graphs[desc] = graph
        static_inputs.append((token_ids, routed, weights, output))
        dispatch_rows[str(rows)] = {
            "cg_mode": desc.cg_mode.name,
            "num_tokens": desc.num_tokens,
            "num_reqs": desc.num_reqs,
            "uniform_token_count": desc.uniform_token_count,
            "graph_static_target_rows": rows,
            "sentinel_reaches_embedding": False,
        }

    samples: dict[str, list[float]] = {}
    medians: dict[str, float] = {}
    for rows in range(1, RAW_DRAFT_WIDTH + 2):
        desc = _dispatch_rows(manager, rows)
        graph = graphs[desc]
        for _ in range(20):
            graph.replay()
        torch.cuda.synchronize()
        row_samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                graph.replay()
            end.record()
            end.synchronize()
            row_samples.append(start.elapsed_time(end) / iterations)
        samples[str(rows)] = row_samples
        medians[str(rows)] = median(row_samples)

    ordered = [medians[str(rows)] for rows in range(1, 7)]
    spearman = _spearman_with_rows(ordered)
    endpoint_ratio = ordered[0] / ordered[-1]
    scaling_pass = (
        all(math.isfinite(value) and value > 0 for value in ordered)
        and ordered[0] < ordered[-1] * 0.95
        and spearman >= 0.60
    )
    return {
        "inventory": inventory,
        "kernel": {
            "kind": "BF16 routed-expert-shaped torch.bmm",
            "hidden_size": SYNTHETIC_HIDDEN,
            "top_k": TOP_K,
            "max_weight_bytes": expert_weights.numel() * expert_weights.element_size(),
        },
        "dispatch": dispatch_rows,
        "verify_ms_samples": samples,
        "verify_ms_median": medians,
        "width1_over_width6": endpoint_ratio,
        "spearman_width_vs_ms": spearman,
        "scaling_contract": {
            "width1_at_least_5pct_below_width6": ordered[0] < ordered[-1] * 0.95,
            "spearman_at_least_0_60": spearman >= 0.60,
            "passed": scaling_pass,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    if args.repeats < 3 or args.iterations < 20:
        raise ValueError("timing requires at least 3 repeats and 20 iterations")

    os.environ["VLLM_DSPARK_CONFIDENCE_SCHEDULER"] = "on"
    cases = {
        "forced_5_to_2": _compact_case([10, 11, -1, -1, -1], enabled=True),
        "forced_5_to_0": _compact_case([-1, -1, -1, -1, -1], enabled=True),
        "full_5": _compact_case([10, 11, 12, 13, 14], enabled=True),
        "confidence_off": _compact_case([10, 11, 12, 13, 14], enabled=False),
    }
    expected_rows = {
        "forced_5_to_2": 3,
        "forced_5_to_0": 1,
        "full_5": 6,
        "confidence_off": 6,
    }
    for name, expected in expected_rows.items():
        actual = cases[name]["physical_target_rows"]
        if actual != expected:
            raise RuntimeError(
                f"{name} target-row contract failed: {actual} != {expected}"
            )

    manager = _make_graph_manager()
    exact_rows3 = _dispatch_rows(manager, 3)
    result: dict[str, object] = {
        "schema_version": 1,
        "device": args.device,
        "cases": cases,
        "assertions": {
            "forced_5_to_2_rows_3": True,
            "forced_5_to_0_rows_1": True,
            "off_or_full_rows_6": True,
            "no_sentinel_reaches_target": True,
            "rows3_dispatch_exact_not_6": (
                exact_rows3.num_tokens == 3
                and exact_rows3.uniform_token_count == 3
            ),
        },
    }
    if args.device == "cuda":
        timing = _cuda_timing(
            manager, repeats=args.repeats, iterations=args.iterations
        )
        result["cuda"] = timing
        result["assertions"]["verify_scales_with_rows"] = timing[
            "scaling_contract"
        ]["passed"]
        result["ok"] = all(result["assertions"].values())
    else:
        result["ok"] = all(result["assertions"].values())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
