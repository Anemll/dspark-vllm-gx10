#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bind and execute every DeepSeek V4 mHC DeepGEMM split specialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
    _compute_mhc_pre_num_split,
    _select_mhc_split_representatives,
)
from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--hc-mult", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=16_384)
    parser.add_argument("--repeats", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("probe requires exactly one visible CUDA device")
    if args.repeats < 2:
        raise ValueError("at least two repeats are required to prove hot reuse")

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    num_sms = properties.multi_processor_count
    representatives = _select_mhc_split_representatives(
        max_tokens=args.max_tokens,
        hidden_size=args.hidden_size,
        hc_mult=args.hc_mult,
        num_sms=num_sms,
    )
    if not representatives:
        raise RuntimeError("no mHC split specializations were selected")

    n = args.hc_mult * 2 + args.hc_mult * args.hc_mult
    k = args.hc_mult * args.hidden_size
    max_m = max(representatives.values())
    x = torch.zeros((max_m, k), dtype=torch.bfloat16, device=device)
    fn = torch.zeros((n, k), dtype=torch.float32, device=device)
    cases: list[dict[str, object]] = []

    with torch.inference_mode():
        for repeat in range(args.repeats):
            for num_splits, num_tokens in sorted(representatives.items()):
                computed = _compute_mhc_pre_num_split(
                    num_tokens=num_tokens,
                    hidden_size=args.hidden_size,
                    hc_mult=args.hc_mult,
                    num_sms=num_sms,
                )
                if computed != num_splits:
                    raise RuntimeError(
                        f"split drift for M={num_tokens}: {computed} != {num_splits}"
                    )
                out = torch.empty(
                    (num_splits, num_tokens, n),
                    dtype=torch.float32,
                    device=device,
                )
                sqrsum = torch.empty(
                    (num_splits, num_tokens),
                    dtype=torch.float32,
                    device=device,
                )
                started = time.perf_counter()
                tf32_hc_prenorm_gemm(
                    x[:num_tokens], fn, out, sqrsum, num_splits
                )
                torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if not torch.isfinite(out).all() or not torch.isfinite(sqrsum).all():
                    raise RuntimeError(
                        f"non-finite result for M={num_tokens}, split={num_splits}"
                    )
                if torch.count_nonzero(out) or torch.count_nonzero(sqrsum):
                    raise RuntimeError(
                        f"zero-input result drift for M={num_tokens}, split={num_splits}"
                    )
                cases.append(
                    {
                        "repeat": repeat,
                        "phase": "bind" if repeat == 0 else "reuse",
                        "num_tokens": num_tokens,
                        "num_splits": num_splits,
                        "elapsed_ms": elapsed_ms,
                    }
                )

    expected_splits = sorted(representatives)
    observed_by_repeat = {
        repeat: sorted(
            int(case["num_splits"])
            for case in cases
            if case["repeat"] == repeat
        )
        for repeat in range(args.repeats)
    }
    if any(splits != expected_splits for splits in observed_by_repeat.values()):
        raise RuntimeError(
            f"coverage drift: expected={expected_splits}, observed={observed_by_repeat}"
        )

    result = {
        "ok": True,
        "gpu": properties.name,
        "capability": list(torch.cuda.get_device_capability(device)),
        "num_sms": num_sms,
        "hidden_size": args.hidden_size,
        "hc_mult": args.hc_mult,
        "k": k,
        "n": n,
        "max_tokens": args.max_tokens,
        "representatives": {
            str(num_splits): num_tokens
            for num_splits, num_tokens in sorted(representatives.items())
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
