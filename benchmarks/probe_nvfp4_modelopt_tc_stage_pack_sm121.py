#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Run one exact ModelOpt/E8M0 TC tactic or stage-pack real-layer probe.

This thin runner deliberately reuses the already-audited real-layer conversion,
numeric, CUDA-graph, pointer-identity, and memory gates.  It only refreshes the
kernel source pin to the deterministic stage-pack patch, selects one FC1/FC2
tactic pair, and records the compile result that proves the requested geometry
actually ran.

Typical matrix (stage-pack disabled)::

    for pair in aa bb bc cb cc; do
      python3 benchmarks/probe_nvfp4_modelopt_tc_stage_pack_sm121.py \
        --layer-file /models/model-layer-00000.safetensors \
        --output /artifacts/modelopt-tc-${pair}.json \
        --fc1-tile "${pair%?}" --fc2-tile "${pair#?}"
    done

Then rerun the winning pair with ``--stage-pack``.  A matrix result may miss
the final 0.682812 ms promotion target without making the probe itself invalid;
this runner's exit status gates source/layout/numeric/graph/memory correctness,
while the measured promotion result remains explicit in the JSON.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from benchmarks import benchmark_nvfp4_prepared_mxfp4_requant_sm121 as base
from scripts import patch_b12x_w4a16_modelopt_stage_pack as patcher


KERNEL_MODULE = "b12x.moe.fused.w4a16.kernel"
EXPECTED_KERNEL_SHA256 = patcher.PATCHED_SOURCE_SHA256
ENV_STAGE = "B12X_W4A16_MODELOPT_STAGE_PACK"
ENV_FC1 = "B12X_W4A16_MODELOPT_FC1_TILE"
ENV_FC2 = "B12X_W4A16_MODELOPT_FC2_TILE"


def _trace_compile_results(kernel: Any) -> tuple[list[dict[str, Any]], Any]:
    original = kernel.compile_w4a16_fused_moe
    events: list[dict[str, Any]] = []

    def traced(**kwargs: Any) -> Any:
        result = original(**kwargs)
        events.append(
            {
                "m": int(kwargs["size_m"]),
                "weight_layout": str(kwargs["weight_layout"]),
                "scale_format": str(kwargs["scale_format"]),
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


def _stage_probe_failures(report: dict[str, Any]) -> list[dict[str, Any]]:
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

    suffix, _old_sha = base.PINNED_SOURCE_SHA256[KERNEL_MODULE]
    base.PINNED_SOURCE_SHA256[KERNEL_MODULE] = (
        suffix,
        EXPECTED_KERNEL_SHA256,
    )
    requested = {
        ENV_STAGE: "1" if args.stage_pack else "0",
        ENV_FC1: args.fc1_tile,
        ENV_FC2: args.fc2_tile,
    }
    original_environment = {name: os.environ.get(name) for name in requested}
    for name, value in requested.items():
        os.environ[name] = value

    events, original_compile = _trace_compile_results(kernel)
    try:
        base_rc = base.run(args)
    finally:
        kernel.compile_w4a16_fused_moe = original_compile
        for name, value in original_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    report = json.loads(args.output.read_text())
    compile_events = [
        event
        for event in events
        if event["weight_layout"] == "modelopt"
        and event["scale_format"] == "e8m0_k32"
    ]
    expected_geometry = {
        "a": (128, 128),
        "b": (64, 128),
        "c": (128, 64),
    }
    fc1_expected = expected_geometry[args.fc1_tile]
    fc2_expected = expected_geometry[args.fc2_tile]
    geometry_passed = bool(compile_events) and all(
        (event["fc1_tile_k"], event["fc1_tile_n"]) == fc1_expected
        and (event["fc2_tile_k"], event["fc2_tile_n"]) == fc2_expected
        for event in compile_events
    )
    failures = _stage_probe_failures(report)
    if not geometry_passed:
        failures.append({"kind": "compile_geometry"})
    report["modelopt_stage_pack_probe"] = {
        "kernel_sha256": EXPECTED_KERNEL_SHA256,
        "environment": requested,
        "requested_fc1_tile": {
            "name": args.fc1_tile,
            "tile_k": fc1_expected[0],
            "tile_n": fc1_expected[1],
        },
        "requested_fc2_tile": {
            "name": args.fc2_tile,
            "tile_k": fc2_expected[0],
            "tile_n": fc2_expected[1],
        },
        "compile_events": compile_events,
        "geometry_passed": geometry_passed,
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
                "stage_pack": args.stage_pack,
                "fc1_tile": args.fc1_tile,
                "fc2_tile": args.fc2_tile,
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
    parser.add_argument("--fc1-tile", choices=("a", "b", "c"), required=True)
    parser.add_argument("--fc2-tile", choices=("a", "b", "c"), required=True)
    parser.add_argument("--stage-pack", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
