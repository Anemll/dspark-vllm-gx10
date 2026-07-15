#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Small dependency-free streaming benchmark for the DSv4 OpenAI API."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import statistics
import time
import urllib.request


PROMPT = (
    "Write a detailed technical explanation of how speculative decoding works "
    "in an autoregressive language model. Continue until the token limit and "
    "do not use a conclusion or summary."
)


@dataclass
class StreamResult:
    request: int
    prompt_tokens: int
    completion_tokens: int
    chunks: int
    ttft_s: float
    decode_s: float
    elapsed_s: float
    token_tps: float
    chunk_tps: float
    finish_reason: str | None


def run_stream(base_url: str, model: str, max_tokens: int, request_id: int) -> StreamResult:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token = None
    last_token = None
    chunks = 0
    usage: dict[str, int] = {}
    finish_reason = None
    with urllib.request.urlopen(req, timeout=900) as response:
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
                content = choice.get("delta", {}).get("content")
                if content:
                    now = time.perf_counter()
                    first_token = first_token or now
                    last_token = now
                    chunks += 1
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    finished = time.perf_counter()
    first_token = first_token or finished
    last_token = last_token or finished
    decode_s = max(last_token - first_token, 1e-9)
    completion_tokens = int(usage.get("completion_tokens", 0))
    return StreamResult(
        request=request_id,
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=completion_tokens,
        chunks=chunks,
        ttft_s=first_token - started,
        decode_s=decode_s,
        elapsed_s=finished - started,
        token_tps=completion_tokens / decode_s,
        chunk_tps=chunks / decode_s,
        finish_reason=finish_reason,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://spark-head.local:8888")
    parser.add_argument("--model", default="deepseek-v4-flash-dspark-abliterated")
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output")
    args = parser.parse_args()

    report = {
        "base_url": args.base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "trials": [],
    }
    print(f"target {args.base_url} model {args.model}", flush=True)
    for concurrency in (int(value) for value in args.concurrency.split(",")):
        print(f"=== concurrency {concurrency} ===", flush=True)
        for trial in range(1, args.trials + 1):
            wall_started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                results = list(
                    executor.map(
                        lambda request_id: run_stream(
                            args.base_url, args.model, args.max_tokens, request_id
                        ),
                        range(concurrency),
                    )
                )
            wall_s = time.perf_counter() - wall_started
            total_tokens = sum(result.completion_tokens for result in results)
            trial_result = {
                "concurrency": concurrency,
                "trial": trial,
                "wall_s": wall_s,
                "total_tokens": total_tokens,
                "aggregate_token_tps": total_tokens / wall_s,
                "mean_ttft_s": statistics.mean(result.ttft_s for result in results),
                "streams": [asdict(result) for result in results],
            }
            report["trials"].append(trial_result)
            token_rates = ", ".join(f"{result.token_tps:.1f}" for result in results)
            chunk_rates = ", ".join(f"{result.chunk_tps:.1f}" for result in results)
            print(
                f"  trial={trial}: output [{token_rates}] tok/s | "
                f"chunks [{chunk_rates}]/s | aggregate "
                f"{trial_result['aggregate_token_tps']:.1f} tok/s | "
                f"TTFT {trial_result['mean_ttft_s']:.2f}s",
                flush=True,
            )

    print("=== BEST AGGREGATE TOKEN THROUGHPUT ===")
    for concurrency in sorted({trial["concurrency"] for trial in report["trials"]}):
        best = max(
            trial["aggregate_token_tps"]
            for trial in report["trials"]
            if trial["concurrency"] == concurrency
        )
        print(f"  x{concurrency}: {best:.1f} tok/s")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            json.dump(report, output, indent=2)
            output.write("\n")


if __name__ == "__main__":
    main()
