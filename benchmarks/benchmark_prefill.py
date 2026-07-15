#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Benchmark DSv4 prefill throughput at exact input-token lengths.

The client sends token IDs to avoid tokenizer drift and brackets every request
with Prometheus snapshots.  When no other request overlaps the trial, this
provides both client-observed TTFT and the server's request-prefill duration.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import random
import re
import statistics
import time
import urllib.request


DEFAULT_SIZES = "1024,2048,4096,8192,16384,32768"
TOKEN_CORPUS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega hardware software "
    "memory compute network storage inference benchmark deterministic matrix"
)


@dataclass
class MetricSnapshot:
    prefill_time_s: float
    prefill_requests: float
    computed_tokens: float
    cache_hit_tokens: float
    prompt_tokens: float


@dataclass
class PrefillResult:
    target_tokens: int
    trial: int
    prompt_sha256: str
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    elapsed_s: float
    client_input_tps: float
    server_prefill_s: float
    server_computed_tokens: int
    server_cache_hit_tokens: int
    server_prefill_tps: float | None
    metrics_request_delta: float
    metrics_exact: bool
    finish_reason: str | None


def request_json(
    url: str,
    body: dict[str, object] | None = None,
    timeout: float = 30,
) -> dict[str, object]:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def tokenize(base_url: str, model: str, text: str) -> list[int]:
    response = request_json(
        f"{base_url.rstrip('/')}/tokenize",
        {"model": model, "prompt": text},
    )
    return [int(token) for token in response["tokens"]]  # type: ignore[index]


def fetch_text(url: str, timeout: float = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def metric_total(
    text: str,
    metric_name: str,
    required_labels: dict[str, str] | None = None,
) -> float:
    total = 0.0
    matched = False
    pattern = re.compile(
        rf"^{re.escape(metric_name)}(?:\{{(?P<labels>[^}}]*)\}})?\s+"
        r"(?P<value>[-+0-9.eE]+)$"
    )
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        labels = dict(re.findall(r'(\w+)="((?:\\.|[^"\\])*)"', match["labels"] or ""))
        if required_labels and any(labels.get(key) != value for key, value in required_labels.items()):
            continue
        total += float(match["value"])
        matched = True
    if not matched:
        raise RuntimeError(f"metric not found: {metric_name}")
    return total


def snapshot_metrics(base_url: str) -> MetricSnapshot:
    text = fetch_text(f"{base_url.rstrip('/')}/metrics")
    return MetricSnapshot(
        prefill_time_s=metric_total(text, "vllm:request_prefill_time_seconds_sum"),
        prefill_requests=metric_total(text, "vllm:request_prefill_time_seconds_count"),
        computed_tokens=metric_total(
            text, "vllm:request_prefill_kv_computed_tokens_sum"
        ),
        cache_hit_tokens=metric_total(
            text,
            "vllm:prompt_tokens_by_source_total",
            {"source": "local_cache_hit"},
        ),
        prompt_tokens=metric_total(text, "vllm:prompt_tokens_total"),
    )


def make_prompt(pool: list[int], size: int, trial: int, seed: int) -> list[int]:
    if not pool:
        raise RuntimeError("tokenizer returned an empty benchmark token pool")
    # A different pseudo-random prefix for every size/trial prevents accidental
    # prefix-cache reuse. The same seed reproduces identical inputs before/after.
    rng = random.Random(f"dspark-prefill:{seed}:{size}:{trial}")
    return [pool[rng.randrange(len(pool))] for _ in range(size)]


def run_completion(
    base_url: str,
    model: str,
    prompt: list[int],
    timeout: float,
) -> tuple[float, float, dict[str, int], str | None]:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
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
    first_event: float | None = None
    usage: dict[str, int] = {}
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
                usage = {
                    key: int(value)
                    for key, value in event["usage"].items()
                    if isinstance(value, (int, float))
                }
            choices = event.get("choices", [])
            if choices and first_event is None:
                first_event = time.perf_counter()
            for choice in choices:
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    finished = time.perf_counter()
    first_event = first_event or finished
    return first_event - started, finished - started, usage, finish_reason


def median_or_none(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def build_summary(results: list[PrefillResult]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for size in sorted({result.target_tokens for result in results}):
        trials = [result for result in results if result.target_tokens == size]
        exact = [result for result in trials if result.metrics_exact]
        summary.append(
            {
                "target_tokens": size,
                "trials": len(trials),
                "exact_server_trials": len(exact),
                "median_ttft_s": statistics.median(result.ttft_s for result in trials),
                "median_client_input_tps": statistics.median(
                    result.client_input_tps for result in trials
                ),
                "median_server_prefill_tps": median_or_none(
                    [result.server_prefill_tps for result in exact]
                ),
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
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--label", default="unlabelled")
    parser.add_argument("--report-target", default="spark-head")
    parser.add_argument("--output")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    version = request_json(f"{base_url}/version").get("version", "unknown")
    pool = tokenize(base_url, args.model, TOKEN_CORPUS)
    sizes = [int(value) for value in args.sizes.split(",")]
    if any(size <= 0 for size in sizes) or args.trials <= 0:
        parser.error("sizes and trials must be positive")

    print(
        f"target {base_url} model {args.model} version {version} label {args.label}",
        flush=True,
    )
    results: list[PrefillResult] = []
    for size in sizes:
        print(f"=== {size:,} input tokens ===", flush=True)
        for trial in range(1, args.trials + 1):
            prompt = make_prompt(pool, size, trial, args.seed)
            prompt_hash = hashlib.sha256(
                ",".join(str(token) for token in prompt).encode()
            ).hexdigest()
            before = snapshot_metrics(base_url)
            ttft_s, elapsed_s, usage, finish_reason = run_completion(
                base_url, args.model, prompt, args.timeout
            )
            after = snapshot_metrics(base_url)

            prompt_tokens = int(usage.get("prompt_tokens", size))
            completion_tokens = int(usage.get("completion_tokens", 0))
            prefill_s = after.prefill_time_s - before.prefill_time_s
            request_delta = after.prefill_requests - before.prefill_requests
            computed_tokens = round(after.computed_tokens - before.computed_tokens)
            cache_hit_tokens = round(after.cache_hit_tokens - before.cache_hit_tokens)
            metrics_exact = abs(request_delta - 1.0) < 0.01
            server_tps = (
                computed_tokens / prefill_s
                if metrics_exact and computed_tokens > 0 and prefill_s > 0
                else None
            )
            result = PrefillResult(
                target_tokens=size,
                trial=trial,
                prompt_sha256=prompt_hash,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                ttft_s=ttft_s,
                elapsed_s=elapsed_s,
                client_input_tps=prompt_tokens / max(ttft_s, 1e-9),
                server_prefill_s=prefill_s,
                server_computed_tokens=computed_tokens,
                server_cache_hit_tokens=cache_hit_tokens,
                server_prefill_tps=server_tps,
                metrics_request_delta=request_delta,
                metrics_exact=metrics_exact,
                finish_reason=finish_reason,
            )
            results.append(result)
            server_text = f"{server_tps:.1f}" if server_tps is not None else "shared"
            cache_text = f" cache-hit={cache_hit_tokens}" if cache_hit_tokens else ""
            print(
                f"  trial={trial}: server={server_text} tok/s | "
                f"client={result.client_input_tps:.1f} tok/s | "
                f"TTFT={ttft_s:.3f}s | computed={computed_tokens}{cache_text}",
                flush=True,
            )

    report = {
        "schema_version": 1,
        "label": args.label,
        "target": args.report_target,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "version": version,
        "sizes": sizes,
        "trials_per_size": args.trials,
        "seed": args.seed,
        "token_pool_sha256": hashlib.sha256(
            ",".join(str(token) for token in pool).encode()
        ).hexdigest(),
        "measurement": {
            "server_prefill_tps": (
                "Delta of vllm request_prefill_kv_computed_tokens divided by "
                "request_prefill_time_seconds; valid only when metrics_exact is true."
            ),
            "client_input_tps": "Prompt tokens divided by client-observed TTFT.",
            "cache_control": (
                "Deterministic unique token-ID prompts prevent cross-trial prefix hits."
            ),
        },
        "results": [asdict(result) for result in results],
        "summary": build_summary(results),
    }
    print("=== MEDIANS ===")
    for row in report["summary"]:
        server = row["median_server_prefill_tps"]
        server_text = f"{server:.1f}" if isinstance(server, float) else "n/a"
        print(
            f"  {row['target_tokens']:>6,}: server {server_text} tok/s | "
            f"client {row['median_client_input_tps']:.1f} tok/s | "
            f"TTFT {row['median_ttft_s']:.3f}s"
        )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            json.dump(report, output, indent=2)
            output.write("\n")


if __name__ == "__main__":
    main()
