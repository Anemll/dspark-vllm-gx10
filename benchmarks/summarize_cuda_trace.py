#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Attribute CUDA kernels to model-forward annotations via launch correlation.

GPU execution may lag CPU enqueue by many forwards. Do not assign kernels
using overlap with CPU time ranges. Kernel sums include stream overlap; neither
those sums nor the profiled spans are uninstrumented request latency or TPS.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path


def family(name):
    if "W4A16FusedMoeKernel" in name:
        return "moe"
    if "W4A16TopKSumKernel" in name:
        return "moe_reduce"
    if "sparse_mla" in name.lower():
        return "sparse_mla"
    if "nccl" in name.lower():
        return "nccl"
    return "other"


def summarize(events):
    contexts = [e for e in events if e.get("cat") == "user_annotation"
                and e.get("name", "").startswith("execute_context_") and "dur" in e]
    contexts.sort(key=lambda e: e["ts"])
    launches = defaultdict(list)
    for event in events:
        if event.get("cat") in ("cuda_runtime", "cuda_driver"):
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                launches[correlation].append(event)
    grouped = defaultdict(list)
    for event in events:
        if event.get("cat") != "kernel" or "dur" not in event:
            continue
        owners = set()
        for launch in launches.get(event.get("args", {}).get("correlation"), []):
            matches = [(i, c) for i, c in enumerate(contexts)
                       if c.get("pid") == launch.get("pid") and c.get("tid") == launch.get("tid")
                       and c["ts"] <= launch["ts"] < c["ts"] + c["dur"]]
            # Innermost annotation, but ambiguous correlation reuse is not
            # silently assigned to whichever launch happened to appear last.
            owners.add(min(matches, key=lambda pair: pair[1]["dur"])[0] if matches else -1)
        owner = next(iter(owners)) if len(owners) == 1 else -1
        grouped[owner].append(event)
    rows = []
    for owner, kernels in sorted(grouped.items()):
        families = defaultdict(lambda: {"count": 0, "kernel_sum_ms": 0.0})
        for kernel in kernels:
            item = families[family(kernel["name"])]
            item["count"] += 1
            item["kernel_sum_ms"] += kernel["dur"] / 1000
        rows.append({
            "context_index": owner if owner >= 0 else None,
            "annotation": contexts[owner]["name"] if owner >= 0 else "unattributed",
            "kernel_count": len(kernels),
            "kernel_sum_ms": sum(e["dur"] for e in kernels) / 1000,
            "gpu_span_ms": (max(e["ts"] + e["dur"] for e in kernels)
                            - min(e["ts"] for e in kernels)) / 1000,
            "families": dict(families),
        })
    return {"scope": "instrumented per-rank trace, not serving TPS",
            "attribution": "CPU launch correlation and forward annotation; unmatched kernels retained",
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    data = args.trace.read_bytes()
    trace = json.loads(gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data)
    result = summarize(trace["traceEvents"])
    result["trace_sha256"] = hashlib.sha256(data).hexdigest()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
