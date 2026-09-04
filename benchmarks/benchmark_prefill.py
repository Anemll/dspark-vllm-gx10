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
import math
import random
import re
import statistics
import threading
import time
import urllib.request

try:
    from .benchmark_dsv4_api import _abort_response, sse_data, write_report
except ImportError:
    from benchmark_dsv4_api import _abort_response, sse_data, write_report


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
    tokens = response.get("tokens")
    if not isinstance(tokens, list) or not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("tokenizer did not return nonempty nonnegative integer token IDs")
    if "count" in response and response["count"] != len(tokens):
        raise ValueError("tokenizer count disagrees with returned IDs")
    return tokens


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
        value = float(match["value"])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid counter: {metric_name}")
        total += value
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
    *,
    evidence: dict | None = None,
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
    evidence = evidence if evidence is not None else {}
    evidence.update(request_body=body, events=[], done_received=False, text="", status="running")
    timer = None
    try:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            remaining = timeout - (time.perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError("request deadline exceeded before response")
            timer = threading.Timer(remaining, _abort_response, args=(response,))
            timer.daemon = True
            timer.start()
            for payload in sse_data(response):
                now = time.perf_counter()
                elapsed = now - started
                if elapsed >= timeout:
                    raise TimeoutError("request deadline exceeded")
                record = {"elapsed_s": elapsed, "raw_data": payload}
                evidence["events"].append(record)
                if payload == "[DONE]":
                    evidence["done_received"] = True
                    break
                event = json.loads(payload)
                record["data"] = event
                if not isinstance(event, dict):
                    raise ValueError("SSE event is not an object")
                if event.get("error"):
                    raise ValueError(f"server error: {event['error']}")
                if event.get("usage") is not None:
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    if choice.get("index", 0) != 0:
                        raise ValueError("prefill benchmark expects exactly one completion choice")
                    text = choice.get("text") or ""
                    if text:
                        if first_event is None:
                            first_event = now
                        evidence["text"] += text
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
        elapsed = time.perf_counter() - started
        if elapsed >= timeout:
            raise TimeoutError("request deadline exceeded")
        if not evidence["done_received"]:
            raise ValueError("stream ended without [DONE]")
        if first_event is None:
            raise ValueError("no nonempty completion text; metadata is not a first token")
        if finish_reason not in {"stop", "length"}:
            raise ValueError(f"missing or unexpected finish_reason: {finish_reason!r}")
        if not isinstance(usage, dict):
            raise ValueError("usage is not an object")
        if type(usage.get("prompt_tokens")) is not int or usage["prompt_tokens"] != len(prompt):
            raise ValueError("missing or mismatched usage.prompt_tokens")
        if type(usage.get("completion_tokens")) is not int or usage["completion_tokens"] != 1:
            raise ValueError("missing or invalid usage.completion_tokens; expected one output token")
        evidence["status"] = "complete"
        return first_event - started, elapsed, usage, finish_reason
    except BaseException as exc:
        evidence["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if timer is not None:
            timer.cancel()
        evidence.update(
            elapsed_s=time.perf_counter() - started,
            ttft_s=first_event - started if first_event is not None else None,
            usage=usage, finish_reason=finish_reason,
        )


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
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--warmup-tokens", type=int, default=1024)
    parser.add_argument("--shape-warmup-trials", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--label", default="unlabelled")
    parser.add_argument("--report-target", default="spark-head")
    parser.add_argument("--output")
    args = parser.parse_args()

    try:
        sizes = [int(value) for value in args.sizes.split(",")]
    except ValueError:
        parser.error("sizes must be a comma-separated list of positive integers")
    if (
        not sizes or any(size <= 0 for size in sizes) or args.trials <= 0
        or args.warmup_tokens < 0 or args.shape_warmup_trials < 0
        or not math.isfinite(args.timeout) or args.timeout <= 0
    ):
        parser.error("sizes/trials/finite timeout must be positive; warmups non-negative")

    base_url = args.base_url.rstrip("/")
    results: list[PrefillResult] = []
    report = {
        "schema_version": 1,
        "status": "running",
        "label": args.label,
        "target": args.report_target,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "version": "unknown",
        "sizes": sizes,
        "trials_per_size": args.trials,
        "warmup_tokens": args.warmup_tokens,
        "shape_warmup_trials": args.shape_warmup_trials,
        "seed": args.seed,
        "timeout": args.timeout,
        "measurement": {
            "server_prefill_tps": (
                "Delta of vllm request_prefill_kv_computed_tokens divided by "
                "request_prefill_time_seconds; valid only when metrics_exact is true."
            ),
            "client_input_tps": "Prompt tokens divided by time to first nonempty completion text, excluding empty metadata.",
            "cache_control": (
                "Deterministic token-ID prefixes differ across sizes/trials. "
                "Reusing the same seed on a warm server may reuse previous-run prefixes; "
                "no cache flush is performed."
            ),
        },
        "requests": [],
        "results": [],
        "summary": [],
    }
    write_report(args.output, report)

    def checkpoint():
        report["results"] = [asdict(result) for result in results]
        report["summary"] = build_summary(results)
        write_report(args.output, report)

    def begin_request(prompt, phase, size, trial):
        record = {
            "phase": phase, "target_tokens": size, "trial": trial,
            "prompt_sha256": hashlib.sha256(",".join(str(token) for token in prompt).encode()).hexdigest(),
            "prompt_token_ids": prompt, "status": "running", "stream": {},
        }
        report["requests"].append(record)
        checkpoint()
        return record

    def complete_request(prompt, record):
        try:
            response = run_completion(base_url, args.model, prompt, args.timeout, evidence=record["stream"])
            record["status"] = "complete"
            return response
        except BaseException as exc:
            record["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            checkpoint()

    try:
        version = request_json(f"{base_url}/version").get("version", "unknown")
        report["version"] = version
        pool = tokenize(base_url, args.model, TOKEN_CORPUS)
        report["token_pool_sha256"] = hashlib.sha256(",".join(str(token) for token in pool).encode()).hexdigest()
        report["token_pool"] = pool
        checkpoint()
        print(f"target {base_url} model {args.model} version {version} label {args.label}", flush=True)
        if args.warmup_tokens:
            warmup = make_prompt(pool, args.warmup_tokens, 0, args.seed)
            record = begin_request(warmup, "warmup", args.warmup_tokens, 0)
            warmup_ttft, _, _, _ = complete_request(warmup, record)
            print(f"warmup {args.warmup_tokens:,} tokens: TTFT {warmup_ttft:.3f}s", flush=True)
        for size in sizes:
            for warmup_trial in range(1, args.shape_warmup_trials + 1):
                # Negative trial IDs preserve the original distinct warmup prompts.
                prompt = make_prompt(pool, size, -warmup_trial, args.seed)
                record = begin_request(prompt, "shape-warmup", size, -warmup_trial)
                warmup_ttft, _, _, _ = complete_request(prompt, record)
                print(f"shape warmup {size:,} tokens ({warmup_trial}/{args.shape_warmup_trials}): TTFT {warmup_ttft:.3f}s", flush=True)

        for size in sizes:
            print(f"=== {size:,} input tokens ===", flush=True)
            for trial in range(1, args.trials + 1):
                prompt = make_prompt(pool, size, trial, args.seed)
                record = begin_request(prompt, "measured", size, trial)
                before = snapshot_metrics(base_url)
                record["metrics_before"] = asdict(before)
                checkpoint()
                ttft_s, elapsed_s, usage, finish_reason = complete_request(prompt, record)
                after = snapshot_metrics(base_url)
                record["metrics_after"] = asdict(after)
                prompt_tokens = usage["prompt_tokens"]
                completion_tokens = usage["completion_tokens"]
                prefill_s = after.prefill_time_s - before.prefill_time_s
                request_delta = after.prefill_requests - before.prefill_requests
                prompt_delta = after.prompt_tokens - before.prompt_tokens
                computed_tokens = round(after.computed_tokens - before.computed_tokens)
                cache_hit_tokens = round(after.cache_hit_tokens - before.cache_hit_tokens)
                metrics_exact = (
                    abs(request_delta - 1.0) < 0.01
                    and abs(prompt_delta - prompt_tokens) < 0.01
                    and prefill_s >= 0 and computed_tokens >= 0
                    and 0 <= cache_hit_tokens <= prompt_tokens
                )
                server_tps = (
                    computed_tokens / prefill_s
                    if metrics_exact and computed_tokens > 0 and prefill_s > 0 else None
                )
                result = PrefillResult(
                    target_tokens=size, trial=trial,
                    prompt_sha256=record["prompt_sha256"],
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                    ttft_s=ttft_s, elapsed_s=elapsed_s,
                    client_input_tps=prompt_tokens / max(ttft_s, 1e-9),
                    server_prefill_s=prefill_s,
                    server_computed_tokens=computed_tokens,
                    server_cache_hit_tokens=cache_hit_tokens,
                    server_prefill_tps=server_tps,
                    metrics_request_delta=request_delta,
                    metrics_exact=metrics_exact, finish_reason=finish_reason,
                )
                results.append(result)
                record["measurement_complete"] = True
                checkpoint()
                server_text = f"{server_tps:.1f}" if server_tps is not None else "shared"
                cache_text = f" cache-hit={cache_hit_tokens}" if cache_hit_tokens else ""
                print(
                    f"  trial={trial}: server={server_text} tok/s | "
                    f"client={result.client_input_tps:.1f} tok/s | "
                    f"TTFT={ttft_s:.3f}s | computed={computed_tokens}{cache_text}",
                    flush=True,
                )
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        if report["requests"]:
            report["requests"][-1]["measurement_error"] = report["error"]
        raise
    finally:
        checkpoint()

    print("=== MEDIANS ===")
    for row in report["summary"]:
        server = row["median_server_prefill_tps"]
        server_text = f"{server:.1f}" if isinstance(server, float) else "n/a"
        print(
            f"  {row['target_tokens']:>6,}: server {server_text} tok/s | "
            f"client {row['median_client_input_tps']:.1f} tok/s | "
            f"TTFT {row['median_ttft_s']:.3f}s"
        )

if __name__ == "__main__":
    main()
