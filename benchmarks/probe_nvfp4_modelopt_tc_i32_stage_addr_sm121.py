#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Run the real-layer C/C gate with the i32 ModelOpt stage-address path."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from benchmarks import probe_nvfp4_modelopt_tc_vector_load_sm121 as vector_probe
from scripts import patch_b12x_w4a16_modelopt_i32_stage_addr as patcher


EXPECTED_KERNEL_SHA256 = patcher.PATCHED_SOURCE_SHA256
ENV_I32 = "B12X_W4A16_MODELOPT_I32_STAGE_ADDR"


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
                "modelopt_vector_load": bool(result.modelopt_vector_load),
                "modelopt_i32_stage_addr": bool(result.modelopt_i32_stage_addr),
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
    old_environment = os.environ.get(ENV_I32)
    old_sha = vector_probe.EXPECTED_KERNEL_SHA256
    old_tracer = vector_probe._trace_compile_results
    os.environ[ENV_I32] = "1"
    vector_probe.EXPECTED_KERNEL_SHA256 = EXPECTED_KERNEL_SHA256
    vector_probe._trace_compile_results = _trace_compile_results
    try:
        base_rc = vector_probe.run(args)
    finally:
        vector_probe._trace_compile_results = old_tracer
        vector_probe.EXPECTED_KERNEL_SHA256 = old_sha
        if old_environment is None:
            os.environ.pop(ENV_I32, None)
        else:
            os.environ[ENV_I32] = old_environment

    report: dict[str, Any] = json.loads(args.output.read_text())
    parent = report.get("modelopt_vector_load_probe", {})
    events = list(parent.get("compile_events", []))
    specialization_passed = bool(events) and all(
        event.get("modelopt_i32_stage_addr") is True for event in events
    )
    failures = list(report.get("failures", []))
    if not specialization_passed:
        failures.append({"kind": "modelopt_i32_stage_addr_specialization"})

    report["modelopt_i32_stage_addr_probe"] = {
        "kernel_sha256": EXPECTED_KERNEL_SHA256,
        "environment": {ENV_I32: "1"},
        "compile_events": events,
        "i32_stage_addr_specialization_passed": specialization_passed,
        "signed_i32_maximum_vector_start": 0x7FFFFFF0,
        "fc1_total_weight_bytes": 0x80000000,
        "expert_n_k_tile_base_hoisted": True,
        "per_copy_int64_expert_multiply_removed": True,
        "weight_bytes_changed": False,
        "global_copy_count_changed": False,
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
                "i32_stage_addr_specialization_passed": specialization_passed,
                "promotion": report.get("performance_gate"),
                "passed": not failures,
            },
            sort_keys=True,
        )
    )
    return 0 if not failures else 2


def build_parser() -> argparse.ArgumentParser:
    parser = vector_probe.build_parser()
    parser.description = __doc__
    parser.add_argument("--modelopt-i32-stage-addr", action="store_true", required=True)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
