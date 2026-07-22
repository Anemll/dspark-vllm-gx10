#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Run the real-layer C/C finite-E8M0 gate with K32 scale reuse."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from benchmarks import benchmark_nvfp4_prepared_mxfp4_requant_sm121 as base
from benchmarks import probe_nvfp4_modelopt_tc_e8m0_scale_fast_sm121 as finite_probe
from scripts import patch_b12x_w4a16_e8m0_k32_scale_reuse as patcher


KERNEL_MODULE = finite_probe.KERNEL_MODULE
EXPECTED_KERNEL_SHA256 = patcher.PATCHED_SOURCE_SHA256
ENV_FAST = finite_probe.ENV_FAST
ENV_REUSE = "B12X_W4A16_E8M0_K32_SCALE_REUSE"
ENV_FC1 = finite_probe.ENV_FC1
ENV_FC2 = finite_probe.ENV_FC2
WINNING_TILE = finite_probe.WINNING_TILE
WINNING_GEOMETRY = finite_probe.WINNING_GEOMETRY


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
                "e8m0_finite_fast": bool(result.e8m0_finite_fast),
                "e8m0_k32_scale_reuse": bool(result.e8m0_k32_scale_reuse),
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
    base.PINNED_SOURCE_SHA256[KERNEL_MODULE] = (suffix, EXPECTED_KERNEL_SHA256)
    requested = {
        ENV_FAST: "1",
        ENV_REUSE: "1",
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
    finite_fast_passed = bool(candidate_events) and all(
        event["e8m0_finite_fast"] for event in candidate_events
    )
    scale_reuse_passed = bool(candidate_events) and all(
        event["e8m0_k32_scale_reuse"] for event in candidate_events
    )
    source_pin_passed = source_proof.get("sha256") == EXPECTED_KERNEL_SHA256
    finite_scale_contract = finite_probe._finite_scale_contract(report)
    failures = finite_probe._probe_failures(report)
    if not geometry_passed:
        failures.append({"kind": "cc_compile_geometry"})
    if not finite_fast_passed:
        failures.append({"kind": "finite_e8m0_compile_specialization"})
    if not scale_reuse_passed:
        failures.append({"kind": "e8m0_k32_scale_reuse_specialization"})
    if not source_pin_passed:
        failures.append({"kind": "kernel_source_pin"})
    if not finite_scale_contract["passed"]:
        failures.append(
            {"kind": "finite_e8m0_checkpoint_contract", **finite_scale_contract}
        )

    report["modelopt_e8m0_k32_scale_reuse_probe"] = {
        "kernel_sha256": EXPECTED_KERNEL_SHA256,
        "environment": requested,
        "winning_tactic": "C/C",
        "expected_geometry": {
            "tile_k": WINNING_GEOMETRY[0],
            "tile_n": WINNING_GEOMETRY[1],
        },
        "compile_events": candidate_events,
        "geometry_passed": geometry_passed,
        "finite_fast_specialization_passed": finite_fast_passed,
        "k32_scale_reuse_specialization_passed": scale_reuse_passed,
        "source_pin_passed": source_pin_passed,
        "finite_scale_contract": finite_scale_contract,
        "scale_loads_per_k32_before": 2,
        "scale_loads_per_k32_after": 1,
        "scale_conversions_per_k32_before": 2,
        "scale_conversions_per_k32_after": 1,
        "weight_loads_changed": False,
        "global_stage_changed": False,
        "mma_or_epilogue_changed": False,
        "prefill_changed": False,
        "base_gate_rc": int(base_rc),
        "promotion_latency_gate": report.get("performance_gate"),
        "failures": failures,
        "passed": not failures,
    }
    report["ok"] = not failures
    report["failures"] = failures
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "winning_tactic": "C/C",
                "finite_fast_specialization_passed": finite_fast_passed,
                "k32_scale_reuse_specialization_passed": scale_reuse_passed,
                "promotion": report.get("performance_gate"),
                "passed": not failures,
            },
            sort_keys=True,
        )
    )
    return 0 if not failures else 2


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = __doc__
    parser.add_argument("--k32-scale-reuse", action="store_true", required=True)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
