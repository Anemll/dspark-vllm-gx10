#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Small dependency-free streaming benchmark for the DSv4 OpenAI API."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
import statistics
import time
import urllib.request


PROMPT = (
    "Write a detailed technical explanation of how speculative decoding works "
    "in an autoregressive language model. Continue until the token limit and "
    "do not use a conclusion or summary."
)

SPEC_METRICS = {
    "num_drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
    "accepted_per_position": "vllm:spec_decode_num_accepted_tokens_per_pos_total",
}
PROM_LINE = re.compile(
    r"^(?P<name>[^\s{]+)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:\s+\d+)?$"
)
POSITION_LABEL = re.compile(r'(?:^|,)position="(?P<position>\d+)"(?:,|$)')


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
    output_sha256: str


@dataclass(frozen=True)
class SpecMetricsSnapshot:
    num_drafts: int
    draft_tokens: int
    accepted_tokens: int
    accepted_per_position: dict[int, int]


def parse_spec_metrics(text: str) -> SpecMetricsSnapshot:
    totals = {key: 0 for key in ("num_drafts", "draft_tokens", "accepted_tokens")}
    positions: dict[int, int] = {}
    recognized = 0
    reverse = {value: key for key, value in SPEC_METRICS.items()}
    for raw in text.splitlines():
        if not raw or raw.startswith("#"):
            continue
        match = PROM_LINE.match(raw)
        if match is None or match.group("name") not in reverse:
            continue
        value = float(match.group("value"))
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            raise ValueError(f"invalid speculative counter: {raw}")
        integer = int(value)
        key = reverse[match.group("name")]
        recognized += 1
        if key == "accepted_per_position":
            label_match = POSITION_LABEL.search(match.group("labels") or "")
            if label_match is None:
                raise ValueError("per-position acceptance counter lacks position label")
            position = int(label_match.group("position"))
            positions[position] = positions.get(position, 0) + integer
        else:
            totals[key] += integer
    if recognized == 0:
        raise ValueError("speculative decoding counters are absent")
    if not positions:
        raise ValueError("per-position speculative counters are absent")
    return SpecMetricsSnapshot(
        num_drafts=totals["num_drafts"],
        draft_tokens=totals["draft_tokens"],
        accepted_tokens=totals["accepted_tokens"],
        accepted_per_position=positions,
    )


def fetch_spec_metrics(base_url: str) -> SpecMetricsSnapshot:
    return parse_spec_metrics(fetch_metrics_text(base_url))


def fetch_metrics_text(base_url: str) -> str:
    with urllib.request.urlopen(
        f"{base_url.rstrip('/')}/metrics", timeout=10
    ) as response:
        return response.read().decode("utf-8", "strict")


def spec_metrics_inactive(before_text: str, after_text: str) -> dict[str, object]:
    """Prove that a no-draft arm emitted no speculative-decoding activity."""

    metric_names = tuple(SPEC_METRICS.values())
    before_present = any(name in before_text for name in metric_names)
    after_present = any(name in after_text for name in metric_names)
    if before_present != after_present:
        raise ValueError("speculative counter presence changed during no-draft arm")
    if not before_present:
        return {
            "enabled": False,
            "counter_state": "absent",
            "num_drafts": 0,
            "draft_tokens": 0,
            "accepted_tokens": 0,
        }

    before = parse_spec_metrics(before_text)
    after = parse_spec_metrics(after_text)
    if before.accepted_per_position != after.accepted_per_position:
        raise ValueError("per-position counters moved during no-draft arm")
    deltas = {
        "num_drafts": after.num_drafts - before.num_drafts,
        "draft_tokens": after.draft_tokens - before.draft_tokens,
        "accepted_tokens": after.accepted_tokens - before.accepted_tokens,
    }
    if any(value != 0 for value in deltas.values()):
        raise ValueError(f"speculative counters moved during no-draft arm: {deltas}")
    return {
        "enabled": False,
        "counter_state": "present_unchanged",
        **deltas,
    }


def spec_metrics_delta(
    before: SpecMetricsSnapshot,
    after: SpecMetricsSnapshot,
    *,
    expected_positions: int,
) -> dict[str, object]:
    deltas = {
        "num_drafts": after.num_drafts - before.num_drafts,
        "draft_tokens": after.draft_tokens - before.draft_tokens,
        "accepted_tokens": after.accepted_tokens - before.accepted_tokens,
    }
    if any(value < 0 for value in deltas.values()):
        raise ValueError("speculative counters moved backwards")
    if deltas["num_drafts"] <= 0 or deltas["draft_tokens"] <= 0:
        raise ValueError("benchmark emitted no speculative drafts")
    if deltas["accepted_tokens"] > deltas["draft_tokens"]:
        raise ValueError("accepted-token count exceeds drafted-token count")
    expected = list(range(expected_positions))
    if sorted(set(before.accepted_per_position) | set(after.accepted_per_position)) != expected:
        raise ValueError("per-position speculative metric set drifted")
    per_position_counts = [
        after.accepted_per_position[position] - before.accepted_per_position[position]
        for position in expected
    ]
    if any(value < 0 for value in per_position_counts):
        raise ValueError("per-position acceptance counters moved backwards")
    if any(value > deltas["num_drafts"] for value in per_position_counts):
        raise ValueError("per-position acceptance exceeds draft count")
    if any(
        left < right
        for left, right in zip(per_position_counts, per_position_counts[1:])
    ):
        raise ValueError("per-position acceptance is not monotonic")
    if sum(per_position_counts) != deltas["accepted_tokens"]:
        raise ValueError("per-position and total accepted-token counters disagree")
    return {
        **deltas,
        "mean_draft_length": (
            deltas["draft_tokens"] / deltas["num_drafts"]
        ),
        "accepted_excess_length": (
            deltas["accepted_tokens"] / deltas["num_drafts"]
        ),
        "aggregate_acceptance_rate": (
            deltas["accepted_tokens"] / deltas["draft_tokens"]
        ),
        "mean_acceptance_length": (
            1 + deltas["accepted_tokens"] / deltas["num_drafts"]
        ),
        "accepted_tokens_per_position": per_position_counts,
        "per_position_acceptance_rates": [
            value / deltas["num_drafts"] for value in per_position_counts
        ],
    }


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
    output_parts: list[str] = []
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
                    output_parts.append(content)
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
        output_sha256=hashlib.sha256(
            "".join(output_parts).encode("utf-8")
        ).hexdigest(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://spark-head.local:8888")
    parser.add_argument("--model", default="deepseek-v4-flash-dspark-abliterated")
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    metric_group = parser.add_mutually_exclusive_group()
    metric_group.add_argument("--require-spec-metrics", action="store_true")
    metric_group.add_argument("--require-no-spec-metrics", action="store_true")
    parser.add_argument("--expected-spec-positions", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args()

    report = {
        "base_url": args.base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "spec_decode_metric_source": (
            SPEC_METRICS
            if args.require_spec_metrics or args.require_no_spec_metrics
            else None
        ),
        "spec_decode_metric_mode": (
            "required"
            if args.require_spec_metrics
            else "inactive_required"
            if args.require_no_spec_metrics
            else None
        ),
        "trials": [],
    }
    print(f"target {args.base_url} model {args.model}", flush=True)
    for concurrency in (int(value) for value in args.concurrency.split(",")):
        print(f"=== concurrency {concurrency} ===", flush=True)
        for trial in range(1, args.trials + 1):
            metrics_before_text = (
                fetch_metrics_text(args.base_url)
                if args.require_spec_metrics or args.require_no_spec_metrics
                else None
            )
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
            if metrics_before_text is not None:
                metrics_after_text = fetch_metrics_text(args.base_url)
                if args.require_spec_metrics:
                    trial_result["spec_decode"] = {
                        "enabled": True,
                        **spec_metrics_delta(
                            parse_spec_metrics(metrics_before_text),
                            parse_spec_metrics(metrics_after_text),
                            expected_positions=args.expected_spec_positions,
                        ),
                    }
                else:
                    trial_result["spec_decode"] = spec_metrics_inactive(
                        metrics_before_text, metrics_after_text
                    )
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
            if trial_result.get("spec_decode", {}).get("enabled"):
                spec = trial_result["spec_decode"]
                print(
                    "    spec: acceptance "
                    f"{100 * spec['aggregate_acceptance_rate']:.1f}% | "
                    f"proposed {spec['mean_draft_length']:.3f} | "
                    f"effective accepted {spec['mean_acceptance_length']:.3f} | "
                    "per-position "
                    + ", ".join(
                        f"{value:.3f}"
                        for value in spec["per_position_acceptance_rates"]
                    ),
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
