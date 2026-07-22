#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Run the real-layer C/C ModelOpt TC gate with finite-E8M0 dequant.

The runner reuses the pinned real-checkpoint conversion, numeric, CUDA-graph,
pointer-identity, output-activity, and memory gates.  It changes only the
finite E8M0-to-BF16 scale conversion inside the ModelOpt direct-top-k decode
kernel and forces the accepted C/C geometry.  The promotion latency is
recorded but does not turn a correctness-valid microbenchmark into an invalid
probe; hardware decides whether this specialization is worth retaining.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from benchmarks import benchmark_nvfp4_prepared_mxfp4_requant_sm121 as base
from scripts import patch_b12x_w4a16_e8m0_scale_fast as patcher


KERNEL_MODULE = "b12x.moe.fused.w4a16.kernel"
EXPECTED_KERNEL_SHA256 = patcher.PATCHED_SOURCE_SHA256
ENV_FAST = "B12X_W4A16_E8M0_FINITE_FAST"
ENV_FC1 = "B12X_W4A16_MODELOPT_FC1_TILE"
ENV_FC2 = "B12X_W4A16_MODELOPT_FC2_TILE"
WINNING_TILE = "c"
WINNING_GEOMETRY = (128, 64)
MAXIMUM_FINITE_E8M0_BYTE = 247


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


def _finite_scale_contract(report: dict[str, Any]) -> dict[str, Any]:
    conversion = (
        report.get("backend_proof", {})
        .get(base.CANDIDATE, {})
        .get("conversion", {})
    )
    rows: dict[str, dict[str, Any]] = {}
    passed = True
    for name in ("w13_scale_collapse", "w2_scale_collapse"):
        source = conversion.get(name, {})
        minimum = source.get("e8m0_minimum_byte")
        maximum = source.get("e8m0_maximum_byte")
        row_passed = bool(
            source.get("passed")
            and source.get("exact_exponent_reconstruction")
            and isinstance(minimum, int)
            and isinstance(maximum, int)
            and 0 <= minimum <= maximum <= MAXIMUM_FINITE_E8M0_BYTE
        )
        rows[name] = {
            "minimum_byte": minimum,
            "maximum_byte": maximum,
            "exact_exponent_reconstruction": source.get(
                "exact_exponent_reconstruction"
            ),
            "passed": row_passed,
        }
        passed = passed and row_passed
    return {
        "maximum_allowed_byte": MAXIMUM_FINITE_E8M0_BYTE,
        "rows": rows,
        "passed": passed,
    }


def _probe_failures(report: dict[str, Any]) -> list[dict[str, Any]]:
    # Match the established stage/tactic runner: direct-path comparator rows
    # are diagnostics and deliberately may fail.  The candidate-vs-W4A4
    # numeric gate, graph, activity, pointer identity, path, and memory gates
    # remain mandatory.  Latency is the decision output, not probe validity.
    failures: list[dict[str, Any]] = []
    if not report.get("native_modelopt_tc_path_gate", {}).get("passed"):
        failures.append({"kind": "native_modelopt_tc_path"})
    if not report.get("memory_gate", {}).get("passed"):
        failures.append({"kind": "memory"})
    for row in report.get("results", []):
        m = int(row["m"])
        numeric = row.get("numeric_passed", {})
        if not numeric.get(f"{base.CANDIDATE}_vs_w4a4"):
            failures.append({"kind": "candidate_numeric", "m": m})
        graph = row.get("cuda_graph_status", {}).get(base.CANDIDATE, {})
        if not graph.get("passed"):
            failures.append({"kind": "candidate_cuda_graph", "m": m})
        activity = row.get("activity", {}).get(base.CANDIDATE, {})
        if not activity.get("passed"):
            failures.append({"kind": "candidate_output_activity", "m": m})
    pointer_proof = (
        report.get("backend_proof", {})
        .get(base.CANDIDATE, {})
        .get("conversion", {})
        .get("shared_fp4_payload_with_w4a4", {})
    )
    if pointer_proof != {
        "w13_same_data_ptr": True,
        "w2_same_data_ptr": True,
    }:
        failures.append({"kind": "single_copy_pointer_identity"})
    return failures


def run(args: argparse.Namespace) -> int:
    from b12x.moe.fused.w4a16 import kernel

    suffix, old_sha = base.PINNED_SOURCE_SHA256[KERNEL_MODULE]
    base.PINNED_SOURCE_SHA256[KERNEL_MODULE] = (
        suffix,
        EXPECTED_KERNEL_SHA256,
    )
    requested = {
        ENV_FAST: "1",
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
    fast_path_passed = bool(candidate_events) and all(
        event["e8m0_finite_fast"] for event in candidate_events
    )
    source_pin_passed = source_proof.get("sha256") == EXPECTED_KERNEL_SHA256
    finite_scale_contract = _finite_scale_contract(report)
    failures = _probe_failures(report)
    if not geometry_passed:
        failures.append({"kind": "cc_compile_geometry"})
    if not fast_path_passed:
        failures.append({"kind": "finite_e8m0_compile_specialization"})
    if not source_pin_passed:
        failures.append({"kind": "kernel_source_pin"})
    if not finite_scale_contract["passed"]:
        failures.append(
            {"kind": "finite_e8m0_checkpoint_contract", **finite_scale_contract}
        )

    report["modelopt_e8m0_finite_fast_probe"] = {
        "kernel_sha256": EXPECTED_KERNEL_SHA256,
        "environment": requested,
        "winning_tactic": "C/C",
        "expected_geometry": {
            "tile_k": WINNING_GEOMETRY[0],
            "tile_n": WINNING_GEOMETRY[1],
        },
        "compile_events": candidate_events,
        "geometry_passed": geometry_passed,
        "finite_fast_specialization_passed": fast_path_passed,
        "source_pin_passed": source_pin_passed,
        "finite_scale_contract": finite_scale_contract,
        "generic_special_value_branches_bypassed": True,
        "weight_or_scale_loads_changed": False,
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
                "finite_fast_specialization_passed": fast_path_passed,
                "finite_scale_contract_passed": finite_scale_contract["passed"],
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
    parser.add_argument("--finite-e8m0-fast", action="store_true", required=True)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
