#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure DSv4 decode and acceptance at exact prompt-token depths."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import statistics
import time
import urllib.request

from benchmark_dsv4_api import fetch_spec_metrics, spec_metrics_delta
from benchmark_prefill import (
    TOKEN_CORPUS,
    make_prompt,
    parse_positive_csv,
    prompt_sha256,
    request_json,
    tokenize,
)


DEFAULT_SIZES = "1024,2048,4096,8192,16384,32768"


@dataclass
class ContextDecodeResult:
    target_prompt_tokens: int
    trial: int
    prompt_sha256: str
    prompt_tokens: int
    completion_tokens: int
    chunks: int
    ttft_s: float
    decode_window_s: float
    elapsed_s: float
    end_to_end_tps: float
    decode_only_tps: float
    finish_reason: str | None
    spec_decode: dict[str, object]


def run_stream(
    base_url: str,
    model: str,
    prompt: list[int],
    max_tokens: int,
    timeout: float,
) -> tuple[dict[str, object], float, float, float, int, str | None]:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token: float | None = None
    last_token: float | None = None
    chunks = 0
    usage: dict[str, object] = {}
    finish_reason = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                text = choice.get("text")
                if text:
                    now = time.perf_counter()
                    first_token = first_token or now
                    last_token = now
                    chunks += 1
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    finished = time.perf_counter()
    first_token = first_token or finished
    last_token = last_token or finished
    return (
        usage,
        first_token - started,
        max(last_token - first_token, 1e-9),
        finished - started,
        chunks,
        finish_reason,
    )


def build_result(
    *,
    target_prompt_tokens: int,
    trial: int,
    prompt_digest: str,
    usage: dict[str, object],
    ttft_s: float,
    decode_window_s: float,
    elapsed_s: float,
    chunks: int,
    finish_reason: str | None,
    spec_decode: dict[str, object],
) -> ContextDecodeResult:
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    if prompt_tokens != target_prompt_tokens:
        raise RuntimeError(
            f"prompt-token drift: expected {target_prompt_tokens}, got {prompt_tokens}"
        )
    if completion_tokens <= 1:
        raise RuntimeError("decode benchmark returned fewer than two completion tokens")
    return ContextDecodeResult(
        target_prompt_tokens=target_prompt_tokens,
        trial=trial,
        prompt_sha256=prompt_digest,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        chunks=chunks,
        ttft_s=ttft_s,
        decode_window_s=decode_window_s,
        elapsed_s=elapsed_s,
        end_to_end_tps=completion_tokens / max(elapsed_s, 1e-9),
        # This intentionally matches MiaAI-Lab's published formula.
        decode_only_tps=(completion_tokens - 1) / max(decode_window_s, 1e-9),
        finish_reason=finish_reason,
        spec_decode=spec_decode,
    )


def build_summary(results: list[ContextDecodeResult]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for size in sorted({result.target_prompt_tokens for result in results}):
        rows = [result for result in results if result.target_prompt_tokens == size]
        summary.append(
            {
                "target_prompt_tokens": size,
                "trials": len(rows),
                "median_ttft_s": statistics.median(row.ttft_s for row in rows),
                "median_end_to_end_tps": statistics.median(
                    row.end_to_end_tps for row in rows
                ),
                "median_decode_only_tps": statistics.median(
                    row.decode_only_tps for row in rows
                ),
                "median_acceptance_rate": statistics.median(
                    float(row.spec_decode["aggregate_acceptance_rate"])
                    for row in rows
                ),
                "median_acceptance_length": statistics.median(
                    float(row.spec_decode["mean_acceptance_length"])
                    for row in rows
                ),
                "median_per_position_acceptance": [
                    statistics.median(
                        float(row.spec_decode["per_position_acceptance_rates"][position])
                        for row in rows
                    )
                    for position in range(
                        len(rows[0].spec_decode["per_position_acceptance_rates"])
                    )
                ],
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://spark-head.local:8888")
    parser.add_argument("--model", default="deepseek-v4-flash-dspark-abliterated")
    parser.add_argument("--sizes", default=DEFAULT_SIZES)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--expected-spec-positions", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.trials <= 0 or args.max_tokens <= 1:
        parser.error("trials must be positive and max-tokens must exceed one")
    try:
        sizes = parse_positive_csv(args.sizes, "sizes")
    except ValueError as error:
        parser.error(str(error))

    base_url = args.base_url.rstrip("/")
    version = request_json(f"{base_url}/version").get("version", "unknown")
    pool = tokenize(base_url, args.model, TOKEN_CORPUS)
    results: list[ContextDecodeResult] = []
    report: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "base_url": base_url,
        "model": args.model,
        "runtime_version": version,
        "sizes": sizes,
        "trials_per_size": args.trials,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "expected_spec_positions": args.expected_spec_positions,
        "decode_only_formula": "(completion_tokens - 1) / (t_last - t_first)",
    }

    for size in sizes:
        print(f"=== context {size} ===", flush=True)
        for trial in range(1, args.trials + 1):
            prompt = make_prompt(pool, size, trial, args.seed)
            digest = prompt_sha256(prompt)
            before = fetch_spec_metrics(base_url)
            timing = run_stream(
                base_url,
                args.model,
                prompt,
                args.max_tokens,
                args.timeout,
            )
            after = fetch_spec_metrics(base_url)
            spec = spec_metrics_delta(
                before,
                after,
                expected_positions=args.expected_spec_positions,
            )
            result = build_result(
                target_prompt_tokens=size,
                trial=trial,
                prompt_digest=digest,
                usage=timing[0],
                ttft_s=timing[1],
                decode_window_s=timing[2],
                elapsed_s=timing[3],
                chunks=timing[4],
                finish_reason=timing[5],
                spec_decode=spec,
            )
            results.append(result)
            print(
                f"  trial={trial} ttft={result.ttft_s:.3f}s "
                f"e2e={result.end_to_end_tps:.2f} tok/s "
                f"decode={result.decode_only_tps:.2f} tok/s "
                f"accept={100 * float(spec['aggregate_acceptance_rate']):.2f}% "
                f"mean_len={float(spec['mean_acceptance_length']):.3f}",
                flush=True,
            )

    report["results"] = [asdict(result) for result in results]
    report["summary"] = build_summary(results)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            json.dump(report, output, indent=2)
            output.write("\n")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
