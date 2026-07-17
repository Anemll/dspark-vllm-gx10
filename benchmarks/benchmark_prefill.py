#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Benchmark DSv4 prefill throughput at exact input-token lengths.

The client sends token IDs to avoid tokenizer drift and brackets every
synchronized request batch with Prometheus snapshots. When no unrelated
request overlaps the trial, this provides client-observed aggregate/individual
TTFT plus the server's summed request-prefill duration.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import random
import re
import statistics
import threading
import time
import urllib.request


DEFAULT_SIZES = "1024,2048,4096,8192,16384,32768"
PREFIX_GUARD_TOKENS = 16
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
    concurrency: int = 1
    batch_prompt_sha256: str = ""
    batch_ttft_s: float = 0.0
    batch_wall_s: float = 0.0
    aggregate_input_tps: float = 0.0
    mean_ttft_s: float = 0.0
    p95_ttft_s: float = 0.0
    cache_isolated: bool = True
    prompt_lengths_exact: bool = True
    metrics_prompt_tokens_delta: int = 0
    metrics_prompt_tokens_exact: bool = True
    server_computed_tokens_exact: bool = True
    completion_lengths_exact: bool = True
    measurement_valid: bool = True
    server_mean_request_prefill_tps: float | None = None
    requests: list[PrefillRequestResult] | None = None


@dataclass
class CompletionTiming:
    started_s: float
    first_event_s: float
    finished_s: float
    usage: dict[str, int]
    finish_reason: str | None

    @property
    def ttft_s(self) -> float:
        return self.first_event_s - self.started_s

    @property
    def elapsed_s(self) -> float:
        return self.finished_s - self.started_s


@dataclass
class PrefillRequestResult:
    request: int
    prompt_sha256: str
    usage_reported: bool
    prompt_usage_reported: bool
    completion_usage_reported: bool
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    elapsed_s: float
    client_input_tps: float
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
        if required_labels and any(
            labels.get(key) != value for key, value in required_labels.items()
        ):
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


def make_prompt(
    pool: list[int],
    size: int,
    trial: int,
    seed: int,
    *,
    request: int = 0,
    concurrency: int = 1,
) -> list[int]:
    if not pool:
        raise RuntimeError("tokenizer returned an empty benchmark token pool")
    # Preserve the original concurrency-1 prompt stream exactly. Multi-request
    # batches include both concurrency and request index so no measured or
    # warm-up request can reuse another matrix row's prefix-cache entry.
    if concurrency == 1 and request == 0:
        prompt_key = f"dspark-prefill:{seed}:{size}:{trial}"
    else:
        prompt_key = (
            f"dspark-prefill:{seed}:{size}:c{concurrency}:"
            f"t{trial}:r{request}"
        )
    rng = random.Random(prompt_key)
    return [pool[rng.randrange(len(pool))] for _ in range(size)]


def prompt_sha256(prompt: list[int]) -> str:
    return hashlib.sha256(
        ",".join(str(token) for token in prompt).encode()
    ).hexdigest()


def make_prompt_batch(
    pool: list[int],
    size: int,
    trial: int,
    seed: int,
    concurrency: int,
    seen_prompt_hashes: set[str],
    seen_prefix_hashes: set[str] | None = None,
) -> tuple[list[list[int]], list[str]]:
    prompts: list[list[int]] = []
    prompt_hashes: list[str] = []
    for request in range(concurrency):
        prompt = make_prompt(
            pool,
            size,
            trial,
            seed,
            request=request,
            concurrency=concurrency,
        )
        digest = prompt_sha256(prompt)
        if digest in seen_prompt_hashes:
            raise RuntimeError(
                "benchmark prompt collision would permit prefix-cache reuse: "
                f"size={size} concurrency={concurrency} trial={trial} "
                f"request={request} sha256={digest}"
            )
        prefix_digest = prompt_sha256(prompt[:PREFIX_GUARD_TOKENS])
        if seen_prefix_hashes is not None and prefix_digest in seen_prefix_hashes:
            raise RuntimeError(
                "benchmark prompt collision in the guarded first cache block: "
                f"size={size} concurrency={concurrency} trial={trial} "
                f"request={request} prefix_tokens="
                f"{min(size, PREFIX_GUARD_TOKENS)} sha256={prefix_digest}"
            )
        seen_prompt_hashes.add(digest)
        if seen_prefix_hashes is not None:
            seen_prefix_hashes.add(prefix_digest)
        prompts.append(prompt)
        prompt_hashes.append(digest)
    return prompts, prompt_hashes


def run_completion(
    base_url: str,
    model: str,
    prompt: list[int],
    timeout: float,
    *,
    start_barrier: threading.Barrier | None = None,
    start_barrier_timeout_s: float = 30.0,
) -> CompletionTiming:
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
    # Serialize every large token-ID body before releasing a concurrent batch;
    # otherwise Python JSON work can turn a nominal x4 trial into staggered x1s.
    if start_barrier is not None:
        try:
            start_barrier.wait(timeout=start_barrier_timeout_s)
        except threading.BrokenBarrierError as error:
            raise RuntimeError("concurrent prefill start barrier failed") from error
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
    return CompletionTiming(
        started_s=started,
        first_event_s=first_event,
        finished_s=finished,
        usage=usage,
        finish_reason=finish_reason,
    )


def run_completion_batch(
    base_url: str,
    model: str,
    prompts: list[list[int]],
    timeout: float,
    *,
    start_barrier_timeout_s: float = 30.0,
) -> list[CompletionTiming]:
    if not prompts:
        raise ValueError("completion batch cannot be empty")
    if len(prompts) == 1:
        return [run_completion(base_url, model, prompts[0], timeout)]

    # Release every client at one barrier so concurrency measures one genuine
    # prefill batch rather than thread-pool startup skew.
    barrier = threading.Barrier(len(prompts) + 1)

    def run_one(prompt: list[int]) -> CompletionTiming:
        return run_completion(
            base_url,
            model,
            prompt,
            timeout,
            start_barrier=barrier,
            start_barrier_timeout_s=start_barrier_timeout_s,
        )

    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = [executor.submit(run_one, prompt) for prompt in prompts]
        try:
            barrier.wait(timeout=start_barrier_timeout_s)
        except threading.BrokenBarrierError as error:
            for future in futures:
                future.cancel()
            raise RuntimeError("concurrent prefill start barrier failed") from error
        return [future.result() for future in futures]


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * quantile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def build_prefill_result(
    *,
    target_tokens: int,
    concurrency: int,
    trial: int,
    prompt_hashes: list[str],
    timings: list[CompletionTiming],
    before: MetricSnapshot,
    after: MetricSnapshot,
) -> PrefillResult:
    if len(prompt_hashes) != concurrency or len(timings) != concurrency:
        raise ValueError("prompt/timing count does not match concurrency")

    request_results: list[PrefillRequestResult] = []
    for request, (digest, timing) in enumerate(zip(prompt_hashes, timings)):
        prompt_usage_reported = "prompt_tokens" in timing.usage
        completion_usage_reported = "completion_tokens" in timing.usage
        prompt_tokens = int(timing.usage.get("prompt_tokens", target_tokens))
        completion_tokens = int(timing.usage.get("completion_tokens", 0))
        request_results.append(
            PrefillRequestResult(
                request=request,
                prompt_sha256=digest,
                usage_reported=(
                    prompt_usage_reported and completion_usage_reported
                ),
                prompt_usage_reported=prompt_usage_reported,
                completion_usage_reported=completion_usage_reported,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                ttft_s=timing.ttft_s,
                elapsed_s=timing.elapsed_s,
                client_input_tps=prompt_tokens / max(timing.ttft_s, 1e-9),
                finish_reason=timing.finish_reason,
            )
        )

    batch_started_s = min(timing.started_s for timing in timings)
    batch_ttft_s = max(timing.first_event_s for timing in timings) - batch_started_s
    batch_wall_s = max(timing.finished_s for timing in timings) - batch_started_s
    total_prompt_tokens = sum(result.prompt_tokens for result in request_results)
    total_completion_tokens = sum(
        result.completion_tokens for result in request_results
    )
    ttfts = [result.ttft_s for result in request_results]

    prefill_s = after.prefill_time_s - before.prefill_time_s
    request_delta = after.prefill_requests - before.prefill_requests
    computed_tokens = round(after.computed_tokens - before.computed_tokens)
    cache_hit_tokens = round(after.cache_hit_tokens - before.cache_hit_tokens)
    metrics_prompt_tokens = round(after.prompt_tokens - before.prompt_tokens)
    metrics_exact = abs(request_delta - float(concurrency)) < 0.01
    cache_isolated = cache_hit_tokens == 0
    prompt_lengths_exact = all(
        result.prompt_usage_reported and result.prompt_tokens == target_tokens
        for result in request_results
    )
    completion_lengths_exact = all(
        result.completion_usage_reported and result.completion_tokens == 1
        for result in request_results
    )
    expected_prompt_tokens = target_tokens * concurrency
    metrics_prompt_tokens_exact = metrics_prompt_tokens == expected_prompt_tokens
    server_computed_tokens_exact = computed_tokens == expected_prompt_tokens
    measurement_valid = (
        metrics_exact
        and cache_isolated
        and prompt_lengths_exact
        and completion_lengths_exact
        and metrics_prompt_tokens_exact
        and server_computed_tokens_exact
        and prefill_s > 0
    )
    # This is a mean request-service rate because the Prometheus denominator is
    # the sum of per-request prefill durations. It is aggregate wall throughput
    # only at concurrency 1; aggregate_input_tps is the scaling metric for C>1.
    server_mean_request_tps = (
        computed_tokens / prefill_s
        if measurement_valid
        else None
    )

    batch_digest = hashlib.sha256("\n".join(prompt_hashes).encode()).hexdigest()
    finish_reasons = {result.finish_reason for result in request_results}
    finish_reason = next(iter(finish_reasons)) if len(finish_reasons) == 1 else None
    mean_ttft_s = statistics.mean(ttfts)
    aggregate_input_tps = total_prompt_tokens / max(batch_ttft_s, 1e-9)

    return PrefillResult(
        target_tokens=target_tokens,
        trial=trial,
        # Keep the exact historical request hash for concurrency 1. For a
        # multi-request trial this field is the deterministic batch hash, while
        # every constituent hash is retained under requests[].
        prompt_sha256=prompt_hashes[0] if concurrency == 1 else batch_digest,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        ttft_s=mean_ttft_s,
        elapsed_s=batch_wall_s,
        client_input_tps=aggregate_input_tps,
        server_prefill_s=prefill_s,
        server_computed_tokens=computed_tokens,
        server_cache_hit_tokens=cache_hit_tokens,
        # Retain the historical name as a compatibility alias.
        server_prefill_tps=server_mean_request_tps,
        metrics_request_delta=request_delta,
        metrics_exact=metrics_exact,
        finish_reason=finish_reason,
        concurrency=concurrency,
        batch_prompt_sha256=batch_digest,
        batch_ttft_s=batch_ttft_s,
        batch_wall_s=batch_wall_s,
        aggregate_input_tps=aggregate_input_tps,
        mean_ttft_s=mean_ttft_s,
        p95_ttft_s=percentile(ttfts, 0.95),
        cache_isolated=cache_isolated,
        prompt_lengths_exact=prompt_lengths_exact,
        metrics_prompt_tokens_delta=metrics_prompt_tokens,
        metrics_prompt_tokens_exact=metrics_prompt_tokens_exact,
        server_computed_tokens_exact=server_computed_tokens_exact,
        completion_lengths_exact=completion_lengths_exact,
        measurement_valid=measurement_valid,
        server_mean_request_prefill_tps=server_mean_request_tps,
        requests=request_results,
    )


def median_or_none(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def format_optional(value: object, decimals: int) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}"
    return "n/a"


def parse_positive_csv(value: str, label: str) -> list[int]:
    try:
        parsed = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise ValueError(f"{label} must be a comma-separated integer list") from error
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError(f"{label} values must be positive")
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"{label} values must be unique")
    return parsed


def build_summary(results: list[PrefillResult]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    shapes = sorted({(result.concurrency, result.target_tokens) for result in results})
    for concurrency, size in shapes:
        trials = [
            result
            for result in results
            if result.target_tokens == size and result.concurrency == concurrency
        ]
        valid = [result for result in trials if result.measurement_valid]
        # A shape is promotion/comparison eligible only when every measured
        # trial is isolated and exact. Raw trials remain in results[] for audit.
        row_valid = len(valid) == len(trials)
        eligible = trials if row_valid else []
        request_ttfts = [
            request.ttft_s
            for result in eligible
            for request in (result.requests or [])
        ]
        summary.append(
            {
                "target_tokens": size,
                "concurrency": concurrency,
                "trials": len(trials),
                "valid_trials": len(valid),
                "row_valid": row_valid,
                # Retain the historical field name, now with the stronger
                # fail-closed definition used by server rate calculations.
                "exact_server_trials": len(valid),
                "cache_isolated_trials": sum(
                    result.cache_isolated for result in trials
                ),
                "exact_length_trials": sum(
                    result.prompt_lengths_exact
                    and result.metrics_prompt_tokens_exact
                    for result in trials
                ),
                "exact_computed_token_trials": sum(
                    result.server_computed_tokens_exact for result in trials
                ),
                "median_ttft_s": median_or_none(
                    [result.ttft_s for result in eligible]
                ),
                "pooled_p95_ttft_s": (
                    percentile(request_ttfts, 0.95) if request_ttfts else None
                ),
                "median_client_input_tps": median_or_none(
                    [result.client_input_tps for result in eligible]
                ),
                "median_aggregate_input_tps": median_or_none(
                    [result.aggregate_input_tps for result in eligible]
                ),
                "median_server_prefill_tps": median_or_none(
                    [result.server_prefill_tps for result in eligible]
                ),
                "median_server_mean_request_prefill_tps": median_or_none(
                    [
                        result.server_mean_request_prefill_tps
                        for result in eligible
                    ]
                ),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://spark-head.local:8888")
    parser.add_argument("--model", default="deepseek-v4-flash-dspark-abliterated")
    parser.add_argument("--sizes", default=DEFAULT_SIZES)
    parser.add_argument(
        "--concurrency",
        default="1",
        help="Comma-separated concurrent prefill requests (default: 1)",
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--warmup-tokens", type=int, default=1024)
    parser.add_argument("--shape-warmup-trials", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--label", default="unlabelled")
    parser.add_argument("--report-target", default="spark-head")
    parser.add_argument("--output")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    version = request_json(f"{base_url}/version").get("version", "unknown")
    pool = tokenize(base_url, args.model, TOKEN_CORPUS)
    try:
        sizes = parse_positive_csv(args.sizes, "sizes")
        concurrencies = parse_positive_csv(args.concurrency, "concurrency")
    except ValueError as error:
        parser.error(str(error))
    if (
        args.trials <= 0
        or args.warmup_tokens < 0
        or args.shape_warmup_trials < 0
    ):
        parser.error(
            "sizes/trials must be positive and warm-up values non-negative"
        )

    print(
        f"target {base_url} model {args.model} version {version} label {args.label} "
        f"concurrency {','.join(str(value) for value in concurrencies)}",
        flush=True,
    )
    seen_prompt_hashes: set[str] = set()
    seen_prefix_hashes: set[str] = set()
    results: list[PrefillResult] = []
    for concurrency in concurrencies:
        if args.warmup_tokens:
            warmup_prompts, _ = make_prompt_batch(
                pool,
                args.warmup_tokens,
                0,
                args.seed,
                concurrency,
                seen_prompt_hashes,
                seen_prefix_hashes,
            )
            warmup_timings = run_completion_batch(
                base_url, args.model, warmup_prompts, args.timeout
            )
            batch_started = min(timing.started_s for timing in warmup_timings)
            warmup_ttft = (
                max(timing.first_event_s for timing in warmup_timings)
                - batch_started
            )
            print(
                f"warmup {args.warmup_tokens:,} tokens x{concurrency}: "
                f"batch TTFT {warmup_ttft:.3f}s",
                flush=True,
            )
        for size in sizes:
            for warmup_trial in range(1, args.shape_warmup_trials + 1):
                # Negative trial IDs keep shape warm-ups distinct from measured
                # prompts and from the general trial-zero warm-up.
                warmup_prompts, _ = make_prompt_batch(
                    pool,
                    size,
                    -warmup_trial,
                    args.seed,
                    concurrency,
                    seen_prompt_hashes,
                    seen_prefix_hashes,
                )
                warmup_timings = run_completion_batch(
                    base_url, args.model, warmup_prompts, args.timeout
                )
                batch_started = min(
                    timing.started_s for timing in warmup_timings
                )
                warmup_ttft = (
                    max(timing.first_event_s for timing in warmup_timings)
                    - batch_started
                )
                print(
                    f"shape warmup {size:,} tokens x{concurrency} "
                    f"({warmup_trial}/{args.shape_warmup_trials}): "
                    f"batch TTFT {warmup_ttft:.3f}s",
                    flush=True,
                )

        for size in sizes:
            print(
                f"=== {size:,} input tokens | concurrency {concurrency} ===",
                flush=True,
            )
            for trial in range(1, args.trials + 1):
                prompts, prompt_hashes = make_prompt_batch(
                    pool,
                    size,
                    trial,
                    args.seed,
                    concurrency,
                    seen_prompt_hashes,
                    seen_prefix_hashes,
                )
                before = snapshot_metrics(base_url)
                timings = run_completion_batch(
                    base_url, args.model, prompts, args.timeout
                )
                after = snapshot_metrics(base_url)
                result = build_prefill_result(
                    target_tokens=size,
                    concurrency=concurrency,
                    trial=trial,
                    prompt_hashes=prompt_hashes,
                    timings=timings,
                    before=before,
                    after=after,
                )
                results.append(result)
                server_text = (
                    f"{result.server_prefill_tps:.1f}"
                    if result.server_prefill_tps is not None
                    else "invalid"
                )
                cache_text = (
                    f" cache-hit={result.server_cache_hit_tokens}"
                    if result.server_cache_hit_tokens
                    else ""
                )
                print(
                    f"  trial={trial}: mean-service={server_text} tok/s | "
                    f"aggregate={result.aggregate_input_tps:.1f} tok/s | "
                    f"TTFT mean/p95={result.mean_ttft_s:.3f}/"
                    f"{result.p95_ttft_s:.3f}s | "
                    f"computed={result.server_computed_tokens}{cache_text} | "
                    f"valid={result.measurement_valid}",
                    flush=True,
                )

    report = {
        # Default concurrency=1 retains the original schema version and scalar
        # field meanings. Multi-concurrency reports use the additive v2 fields.
        "schema_version": 1 if concurrencies == [1] else 2,
        "label": args.label,
        "target": args.report_target,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "version": version,
        "sizes": sizes,
        "concurrencies": concurrencies,
        "trials_per_size": args.trials,
        "warmup_tokens": args.warmup_tokens,
        "shape_warmup_trials": args.shape_warmup_trials,
        "prefix_guard_tokens": PREFIX_GUARD_TOKENS,
        "seed": args.seed,
        "token_pool_sha256": hashlib.sha256(
            ",".join(str(token) for token in pool).encode()
        ).hexdigest(),
        "measurement": {
            "server_prefill_tps": (
                "Delta of vllm request_prefill_kv_computed_tokens divided by "
                "the sum of per-request request_prefill_time_seconds. This is a "
                "mean request-service rate, not aggregate wall throughput when "
                "concurrency exceeds one. It is valid only when every exactness "
                "and isolation gate passes."
            ),
            "client_input_tps": (
                "Aggregate prompt tokens divided by batch TTFT (earliest request "
                "start through the last first-token event). For concurrency 1 this "
                "is the original prompt-tokens/client-TTFT metric."
            ),
            "per_request": (
                "requests[] records prompt hash, usage, TTFT, elapsed time, input "
                "throughput, and finish reason for every concurrent request. "
                "Summary p95 pools all request TTFTs across valid trials."
            ),
            "cache_control": (
                "Deterministic unique token-ID prompts are keyed by size, "
                "concurrency, trial, and request. Full-prompt and first-16-token "
                "prefix collisions fail the run. The server cache-hit metric is "
                "the authoritative isolation gate."
            ),
        },
        "results": [asdict(result) for result in results],
        "summary": build_summary(results),
    }
    print("=== MEDIANS ===")
    for row in report["summary"]:
        print(
            f"  x{row['concurrency']} {row['target_tokens']:>6,}: "
            f"mean-service "
            f"{format_optional(row['median_server_prefill_tps'], 1)} tok/s | "
            f"aggregate "
            f"{format_optional(row['median_aggregate_input_tps'], 1)} tok/s | "
            f"TTFT median/pooled-p95 "
            f"{format_optional(row['median_ttft_s'], 3)}/"
            f"{format_optional(row['pooled_p95_ttft_s'], 3)}s | "
            f"valid {row['valid_trials']}/{row['trials']}"
        )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            json.dump(report, output, indent=2)
            output.write("\n")


if __name__ == "__main__":
    main()
