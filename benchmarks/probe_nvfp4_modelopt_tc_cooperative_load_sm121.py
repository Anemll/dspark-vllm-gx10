#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Run the real-layer ModelOpt TC gate with the cooperative shared loader.

This thin runner reuses the pinned scale-collapse, pointer-identity, numeric,
CUDA-graph, output-activity, memory, and latency gates from
``benchmark_nvfp4_prepared_mxfp4_requant_sm121``.  It only refreshes the exact
kernel source pin and toggles the decode-only cooperative loader specialization.
No checkpoint or serving format changes are involved.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from benchmarks import benchmark_nvfp4_prepared_mxfp4_requant_sm121 as base
from scripts import patch_b12x_w4a16_modelopt_cooperative_load as patcher


KERNEL_MODULE = "b12x.moe.fused.w4a16.kernel"
EXPECTED_KERNEL_SHA256 = patcher.PATCHED_SOURCE_SHA256
ENV_COOPERATIVE = "B12X_W4A16_MODELOPT_COOPERATIVE_LOAD"
ENV_FC1 = "B12X_W4A16_MODELOPT_FC1_TILE"
ENV_FC2 = "B12X_W4A16_MODELOPT_FC2_TILE"
WINNING_TILE = "c"
WINNING_GEOMETRY = (128, 64)


def _trace_compile_results(kernel: Any) -> tuple[list[dict[str, Any]], Any]:
    original = kernel.compile_w4a16_fused_moe
    events: list[dict[str, Any]] = []

    def traced(**kwargs: Any) -> Any:
        result = original(**kwargs)
        events.append(
            {
                "m": int(kwargs["size_m"]),
                "weight_layout": str(result.weight_layout),
                "scale_format": str(result.scale_format),
                "tc_decode_fused_sum": bool(result.tc_decode_fused_sum),
                "modelopt_cooperative_load": bool(
                    result.modelopt_cooperative_load
                ),
                "fc1_tile_k": int(result.fc1_tile_k),
                "fc1_tile_n": int(result.fc1_tile_n),
                "fc2_tile_k": int(result.fc2_tile_k),
                "fc2_tile_n": int(result.fc2_tile_n),
                "blocks_per_sm": int(result.blocks_per_sm),
            }
        )
        return result

    kernel.compile_w4a16_fused_moe = traced
    return events, original


def run(args: argparse.Namespace) -> int:
    from b12x.moe.fused.w4a16 import kernel

    suffix, old_sha = base.PINNED_SOURCE_SHA256[KERNEL_MODULE]
    base.PINNED_SOURCE_SHA256[KERNEL_MODULE] = (
        suffix,
        EXPECTED_KERNEL_SHA256,
    )
    requested = {
        ENV_COOPERATIVE: "1",
        ENV_FC1: WINNING_TILE,
        ENV_FC2: WINNING_TILE,
    }
    original_environment = {name: os.environ.get(name) for name in requested}
    for name, value in requested.items():
        os.environ[name] = value
    compile_events, original_compile = _trace_compile_results(kernel)
    try:
        base_rc = base.run(args)
    finally:
        kernel.compile_w4a16_fused_moe = original_compile
        base.PINNED_SOURCE_SHA256[KERNEL_MODULE] = (suffix, old_sha)
        for name, value in original_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    report: dict[str, Any] = json.loads(args.output.read_text())
    source_proof = (
        report.get("backend_proof", {})
        .get(base.CANDIDATE, {})
        .get("source_api_contract", {})
        .get("files", {})
        .get(KERNEL_MODULE, {})
    )
    candidate_events = [
        event
        for event in compile_events
        if event["weight_layout"] == "modelopt"
        and event["scale_format"] == "e8m0_k32"
        and event["tc_decode_fused_sum"]
    ]
    geometry_passed = bool(candidate_events) and all(
        (event["fc1_tile_k"], event["fc1_tile_n"]) == WINNING_GEOMETRY
        and (event["fc2_tile_k"], event["fc2_tile_n"]) == WINNING_GEOMETRY
        for event in candidate_events
    )
    cooperative_passed = bool(candidate_events) and all(
        event["modelopt_cooperative_load"] for event in candidate_events
    )
    compile_contract_passed = geometry_passed and cooperative_passed
    if not compile_contract_passed:
        report.setdefault("failures", []).append(
            {
                "kind": "cooperative_cc_compile_contract",
                "geometry_passed": geometry_passed,
                "cooperative_passed": cooperative_passed,
                "compile_events": candidate_events,
            }
        )
        report["ok"] = False
    report["modelopt_cooperative_load_probe"] = {
        "kernel_sha256": EXPECTED_KERNEL_SHA256,
        "environment": requested,
        "cooperative_load_requested": True,
        "winning_tactic": "C/C",
        "expected_fc1_geometry": {
            "tile_k": WINNING_GEOMETRY[0],
            "tile_n": WINNING_GEOMETRY[1],
        },
        "expected_fc2_geometry": {
            "tile_k": WINNING_GEOMETRY[0],
            "tile_n": WINNING_GEOMETRY[1],
        },
        "compile_events": candidate_events,
        "geometry_passed": geometry_passed,
        "cooperative_specialization_passed": cooperative_passed,
        "compile_contract_passed": compile_contract_passed,
        "scope": "ModelOpt direct-top-k fused-sum TC decode only",
        "global_stage": "existing cp.async canonical ModelOpt bytes",
        "pipeline_barriers_changed": False,
        "prefill_changed": False,
        "shared_loads_per_lane_before": 16,
        "shared_loads_per_lane_after": 4,
        "subgroup_shuffles_per_lane": 16,
        "extra_shared_bytes": 0,
        "source_pin_passed": source_proof.get("sha256") == EXPECTED_KERNEL_SHA256,
        "base_gate_rc": int(base_rc),
        "performance_gate": report.get("performance_gate"),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "cooperative_load": bool(args.cooperative_load),
                "winning_tactic": "C/C",
                "compile_contract_passed": compile_contract_passed,
                "base_gate_rc": int(base_rc),
                "performance_gate": report.get("performance_gate"),
            },
            sort_keys=True,
        )
    )
    return 0 if int(base_rc) == 0 and compile_contract_passed else 2


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = __doc__
    parser.add_argument("--cooperative-load", action="store_true", required=True)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
