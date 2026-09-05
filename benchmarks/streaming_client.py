#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Dependency-free streaming benchmark with durable per-request evidence."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import tempfile
import threading
import time
from typing import Callable
import urllib.request


PROMPT = (
    "Write a detailed technical explanation of how speculative decoding works "
    "in an autoregressive language model. Continue until the token limit and "
    "do not use a conclusion or summary."
)
METRIC_NOTES = {
    "ttft_s": "Time to first nonempty content, reasoning, or tool delta; not role metadata.",
    "content_ttft_s": "Time to first visible content delta; null for tool-only output.",
    "token_tps": "Legacy proxy: usage completion_tokens / first-to-last output-delta time; null with fewer than two output deltas.",
    "tpot_proxy_s": "First-to-last output-delta time / (completion_tokens - 1); batched SSE is not true per-token timing.",
    "chunks": "Nonempty visible-content SSE deltas, retained for legacy consumers.",
    "aggregate_token_tps": "Successfully validated completion tokens / whole trial wall time, including failed requests and client overhead.",
    "quantiles": "Nearest-rank p95 only with >=20 observations; p99 only with >=100. Small samples report median/range.",
    "cache": "Cache-state labels describe intent; cached tokens are observed only when usage exposes them. No cache flush is performed.",
}


def json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def client_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def write_report(path: str | None, report: dict) -> None:
    """Replace the report atomically, including after each completed request."""
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass
class StreamResult:
    request: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    chunks: int = 0
    ttft_s: float | None = None
    decode_s: float | None = None
    elapsed_s: float = 0
    token_tps: float | None = None
    chunk_tps: float | None = None
    finish_reason: str | None = None
    ok: bool = False
    errors: list[str] = field(default_factory=list)
    content_ttft_s: float | None = None
    output_chunks: int = 0
    tpot_proxy_s: float | None = None
    end_to_end_token_tps: float | None = None
    inter_output_chunk_s: list[float] = field(default_factory=list)
    cached_prompt_tokens: int | None = None
    computed_prompt_tokens: int | None = None
    usage: dict = field(default_factory=dict)
    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    done_received: bool = False
    request_body: dict = field(default_factory=dict)
    request_sha256: str = ""
    prompt_sha256: str = ""


def make_body(model: str, max_tokens: int, seed: int | None = None) -> dict:
    body = {
        "model": model, "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True,
        "stream": True, "stream_options": {"include_usage": True},
    }
    if seed is not None:
        body["seed"] = seed
    return body


def sse_data(response):
    """Read SSE framing, including comments, compact data: and multiline data."""
    parts = []
    for raw in response:
        line = raw.decode("utf-8", "strict").rstrip("\r\n")
        if not line:
            if parts:
                yield "\n".join(parts)
                parts = []
        elif line.startswith("data:"):
            value = line[5:]
            parts.append(value[1:] if value.startswith(" ") else value)
    if parts:
        yield "\n".join(parts)


def _abort_response(response) -> None:
    # urllib's timeout is an idle timeout. Shut down the transport as well
    # to bound streams which keep sending data beyond the whole-request budget.
    raw = getattr(getattr(response, "fp", None), "raw", None)
    transport = getattr(raw, "_sock", None)
    if transport is not None:
        try:
            transport.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def run_stream(
    base_url: str, model: str, max_tokens: int, request_id: int,
    *, body: dict | None = None, timeout: float = 900,
    expect: str = "text", expected_tool: dict | None = None,
    opener: Callable | None = None, clock: Callable = time.perf_counter,
) -> StreamResult:
    """Return failed requests as evidence; callers decide whether to continue."""
    body = body if body is not None else make_body(model, max_tokens)
    result = StreamResult(
        request=request_id, request_body=body, request_sha256=json_hash(body),
        prompt_sha256=json_hash({key: body[key] for key in ("messages", "tools") if key in body}),
    )
    started = clock()
    arrivals: list[float] = []
    tools_by_index: dict[int, dict] = {}
    timer = None
    try:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
        )
        with (opener or urllib.request.urlopen)(req, timeout=timeout) as response:
            remaining = timeout - (clock() - started)
            if remaining <= 0:
                raise TimeoutError("request deadline exceeded before response")
            timer = threading.Timer(remaining, _abort_response, args=(response,))
            timer.daemon = True
            timer.start()
            for payload in sse_data(response):
                elapsed = clock() - started
                if elapsed >= timeout:
                    raise TimeoutError("request deadline exceeded")
                if payload == "[DONE]":
                    result.done_received = True
                    result.events.append({"elapsed_s": elapsed, "data": "[DONE]"})
                    break
                raw_event = {"elapsed_s": elapsed, "raw_data": payload}
                result.events.append(raw_event)
                event = json.loads(payload)
                raw_event["data"] = event
                if not isinstance(event, dict):
                    raise ValueError("SSE event is not an object")
                if event.get("error"):
                    raise ValueError(f"server error: {event['error']}")
                if event.get("usage") is not None:
                    result.usage = event["usage"]
                meaningful = False
                for choice in event.get("choices", []):
                    if choice.get("index", 0) != 0:
                        raise ValueError("benchmark expects exactly one completion choice")
                    delta = choice.get("delta") or {}
                    content = delta.get("content") or ""
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if content:
                        result.content += content
                        result.chunks += 1
                        if result.content_ttft_s is None:
                            result.content_ttft_s = elapsed
                    result.reasoning += reasoning
                    for call in delta.get("tool_calls") or []:
                        index = call.get("index", 0)
                        merged = tools_by_index.setdefault(index, {"index": index, "id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        if call.get("id"):
                            merged["id"] += call["id"]
                        if call.get("type"):
                            merged["type"] = call["type"]
                        for key in ("name", "arguments"):
                            fragment = (call.get("function") or {}).get(key) or ""
                            merged["function"][key] += fragment
                            meaningful = meaningful or bool(fragment)
                    meaningful = meaningful or bool(content) or bool(reasoning)
                    if choice.get("finish_reason") is not None:
                        result.finish_reason = choice["finish_reason"]
                if meaningful:
                    arrivals.append(elapsed)
    except Exception as exc:
        result.errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if timer is not None:
            timer.cancel()
        result.elapsed_s = clock() - started
    if result.elapsed_s >= timeout and not any("deadline" in error for error in result.errors):
        result.errors.append("request deadline exceeded")
    result.tool_calls = [tools_by_index[index] for index in sorted(tools_by_index)]
    result.output_chunks = len(arrivals)
    if arrivals:
        result.ttft_s = arrivals[0]
    if len(arrivals) > 1 and arrivals[-1] > arrivals[0]:
        result.decode_s = arrivals[-1] - arrivals[0]
        result.inter_output_chunk_s = [b - a for a, b in zip(arrivals, arrivals[1:])]
        result.chunk_tps = result.chunks / result.decode_s
    if not isinstance(result.usage, dict):
        result.errors.append("usage is not an object")
        result.usage = {}
    for key in ("prompt_tokens", "completion_tokens"):
        value = result.usage.get(key)
        if type(value) is not int or value <= 0:
            result.errors.append(f"missing or invalid usage.{key}")
        else:
            setattr(result, key, value)
    details = result.usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    if type(cached) is int and cached >= 0:
        result.cached_prompt_tokens = cached
        if result.prompt_tokens is not None:
            if cached > result.prompt_tokens:
                result.errors.append("cached prompt tokens exceed prompt usage")
            else:
                result.computed_prompt_tokens = result.prompt_tokens - cached
    if result.completion_tokens is not None:
        if result.elapsed_s > 0:
            result.end_to_end_token_tps = result.completion_tokens / result.elapsed_s
        if result.decode_s is not None:
            result.token_tps = result.completion_tokens / result.decode_s
            if result.completion_tokens > 1:
                result.tpot_proxy_s = result.decode_s / (result.completion_tokens - 1)
    if not result.done_received:
        result.errors.append("stream ended without [DONE]")
    if result.finish_reason not in {"stop", "length", "tool_calls", "function_call"}:
        result.errors.append(f"missing or unexpected finish_reason: {result.finish_reason!r}")
    if expect == "text" and not result.content.strip():
        result.errors.append("no visible content; reasoning alone does not pass text validation")
    elif expect == "text" and (result.tool_calls or result.finish_reason not in {"stop", "length"}):
        result.errors.append("expected a text completion, received tool termination")
    elif expect == "tool":
        if result.finish_reason != "tool_calls" or not result.tool_calls:
            result.errors.append("expected parsed tool calls and tool_calls finish reason")
        for call in result.tool_calls:
            try:
                arguments = json.loads(call["function"]["arguments"])
                if not call["id"] or call["type"] != "function" or not call["function"]["name"]:
                    raise ValueError("tool call is missing id/type/name")
                if expected_tool and (call["function"]["name"] != expected_tool["name"] or arguments != expected_tool["arguments"]):
                    raise ValueError("tool name or arguments differ from the fixture")
            except (ValueError, TypeError, KeyError) as exc:
                result.errors.append(f"invalid tool call: {exc}")
        if expected_tool and len(result.tool_calls) != 1:
            result.errors.append("fixture expects exactly one tool call")
    elif expect == "any" and not arrivals:
        result.errors.append("no content, reasoning, or tool output")
    elif expect not in {"text", "tool", "any"}:
        result.errors.append(f"unknown output expectation: {expect}")
    result.ok = not result.errors
    return result


def distribution(values) -> dict:
    ordered = sorted(value for value in values if value is not None)
    count = len(ordered)
    return {
        "count": count, "median": statistics.median(ordered) if count else None,
        "min": ordered[0] if count else None, "max": ordered[-1] if count else None,
        "p95": ordered[math.ceil(0.95 * count) - 1] if count >= 20 else None,
        "p99": ordered[math.ceil(0.99 * count) - 1] if count >= 100 else None,
    }


def summarize_trials(trials: list[dict]) -> list[dict]:
    summary = []
    for concurrency in sorted({trial["concurrency"] for trial in trials}):
        group = [trial for trial in trials if trial["concurrency"] == concurrency]
        streams = [stream for trial in group for stream in trial["streams"]]
        valid = [stream for stream in streams if stream["ok"]]
        summary.append({
            "concurrency": concurrency, "trials": len(group),
            "successful_requests": len(valid), "failed_requests": len(streams) - len(valid),
            "aggregate_token_tps": distribution(trial.get("aggregate_token_tps") for trial in group if trial.get("complete")),
            **{key: distribution(stream[key] for stream in valid) for key in ("ttft_s", "content_ttft_s", "elapsed_s", "tpot_proxy_s", "end_to_end_token_tps")},
        })
    return summary


def new_report(args, workload: str) -> dict:
    return {
        "schema_version": 2, "base_url": args.base_url, "model": args.model,
        "max_tokens": args.max_tokens, "status": "running", "trials": [],
        "metric_notes": METRIC_NOTES,
        "manifest": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "client_revision": client_revision(), "workload": workload,
            "client_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "model": args.model, "model_revision": args.model_revision,
            "tokenizer_identifier": args.tokenizer_identifier or args.model,
            "tokenizer_revision": args.tokenizer_revision,
            "settings": vars(args), "cache_state_label": args.cache_state,
        },
    }


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default="http://spark-head.local:8888")
    parser.add_argument("--model", default="deepseek-v4-flash-dspark-abliterated")
    parser.add_argument("--model-revision", default="unknown")
    parser.add_argument("--tokenizer-identifier")
    parser.add_argument("--tokenizer-revision", default="unknown")
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--cache-state", default="unspecified")
    parser.add_argument("--output")


def validate_arguments(parser, args) -> list[int]:
    try:
        levels = [int(value) for value in args.concurrency.split(",")]
    except ValueError:
        parser.error("concurrency must be a comma-separated list of positive integers")
    if not levels or min(levels) <= 0 or args.trials <= 0 or args.max_tokens <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("concurrency, trials, max-tokens, and finite timeout must be positive")
    return levels


def finish_trial(trial: dict, wall_s: float) -> None:
    valid = [stream for stream in trial["streams"] if stream["ok"]]
    ttfts = [stream["ttft_s"] for stream in valid if stream["ttft_s"] is not None]
    trial.update({
        "complete": True, "wall_s": wall_s,
        "total_tokens": sum(stream["completion_tokens"] for stream in valid),
        "mean_ttft_s": statistics.mean(ttfts) if ttfts else None,
        "errors": sum(not stream["ok"] for stream in trial["streams"]),
    })
    trial["aggregate_token_tps"] = trial["total_tokens"] / wall_s if wall_s > 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    levels = validate_arguments(parser, args)
    report = new_report(args, "legacy-speculative-explanation")
    body = make_body(args.model, args.max_tokens, args.seed)
    report["manifest"]["request_body"] = body
    report["manifest"]["request_sha256"] = json_hash(body)
    write_report(args.output, report)
    failed = False
    try:
        for concurrency in levels:
            for trial_number in range(1, args.trials + 1):
                trial = {"concurrency": concurrency, "trial": trial_number, "streams": [], "complete": False}
                report["trials"].append(trial)
                started = time.perf_counter()
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    pending = [executor.submit(run_stream, args.base_url, args.model, args.max_tokens, request_id, body=body, timeout=args.timeout) for request_id in range(concurrency)]
                    for future in as_completed(pending):
                        result = future.result()
                        trial["streams"].append(asdict(result))
                        failed = failed or not result.ok
                        write_report(args.output, report)
                trial["streams"].sort(key=lambda stream: stream["request"])
                finish_trial(trial, time.perf_counter() - started)
                report["summary"] = summarize_trials(report["trials"])
                write_report(args.output, report)
                print(f"concurrency={concurrency} trial={trial_number} aggregate={trial['aggregate_token_tps']:.2f} tok/s errors={trial['errors']}", flush=True)
                if failed:
                    break
            if failed:
                break
        report["status"] = "failed" if failed else "complete"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["summary"] = summarize_trials(report["trials"])
        write_report(args.output, report)
    for summary in report["summary"]:
        print(f"concurrency={summary['concurrency']} median aggregate={summary['aggregate_token_tps']['median']} tok/s", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
