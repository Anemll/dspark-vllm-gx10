#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded text/vision content benchmark with separate latency/throughput metrics."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.request

try:
    from .streaming_client import distribution, json_hash, run_stream, write_report
    from .benchmark_prefill import metric_total
except ImportError:
    from streaming_client import distribution, json_hash, run_stream, write_report
    from benchmark_prefill import metric_total

DEFAULT_FIXTURE = Path(__file__).with_name("fixtures") / "text-content-v1.json"
METRICS = {
    "requests": "vllm:request_generation_tokens_count",
    "output_tokens": "vllm:request_generation_tokens_sum",
    "decode_seconds": "vllm:request_decode_time_seconds_sum",
    "prefill_seconds": "vllm:request_prefill_time_seconds_sum",
    "computed_tokens": "vllm:request_prefill_kv_computed_tokens_sum",
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
}


def get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def snapshot(base_url, model, timeout=10):
    with urllib.request.urlopen(base_url.rstrip("/") + "/metrics", timeout=timeout) as response:
        raw = response.read().decode()
    values = {}
    for key, name in METRICS.items():
        try:
            values[key] = metric_total(raw, name, {"model_name": model})
        except RuntimeError:
            values[key] = None
    return values


def metric_delta(before, after, requests, output_tokens):
    delta = {key: after[key] - before[key] if before.get(key) is not None and after.get(key) is not None else None for key in METRICS}
    exact = (delta["requests"] == requests and delta["output_tokens"] == output_tokens
             and before.get("running") == before.get("waiting") == 0
             and after.get("running") == after.get("waiting") == 0
             and all(value is None or value >= 0 for key, value in delta.items() if key not in {"running", "waiting"}))
    def rate(numerator, denominator):
        return numerator / denominator if exact and numerator is not None and denominator is not None and denominator > 0 else None
    return {
        "exact": exact, "delta": delta,
        "server_decode_tok_s": rate(output_tokens - requests, delta["decode_seconds"]),
        "server_prefill_tok_s": rate(delta["computed_tokens"], delta["prefill_seconds"]),
        "draft_acceptance": rate(delta["accepted_tokens"], delta["draft_tokens"]),
        "accepted_tokens_per_draft_step": rate(delta["accepted_tokens"], delta["drafts"]),
    }


def make_body(case, model, tokens, seed, fixture_dir, phase, trial, request_id):
    prompt = case["prompt"]
    # Preserve the reference prompt verbatim. Other cases have unique short
    # prefixes, saved in full, so warmup/output prefixes cannot masquerade as
    # uncached prefill. Cache state is still verified from server counters.
    if case["id"] != "upstream-explanation":
        prompt = f"Benchmark {json_hash([seed, case['id'], phase, trial, request_id])[:20]}\n" + prompt
    parts, images = [], []
    for relative in case.get("images", []):
        path = (fixture_dir / relative).resolve()
        raw = path.read_bytes()
        if len(raw) > 16 * 1024 * 1024:
            raise ValueError("image exceeds 16 MiB fixture limit")
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(path.suffix.lower())
        if mime is None:
            raise ValueError("vision fixtures must be PNG/JPEG")
        parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(raw).decode()}})
        images.append({"file": relative, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    content = parts + [{"type": "text", "text": prompt}] if parts else prompt
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "temperature": 0, "seed": seed, "max_tokens": tokens,
            "ignore_eos": "expected_json" not in case, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"thinking": False}}
    return body, images


def validate_result(result, case, tokens):
    if "expected_json" in case:
        try:
            if json.loads(result.content) != case["expected_json"]:
                result.errors.append("answer differs from the known visual/structured fixture")
        except ValueError:
            result.errors.append("answer is not valid unwrapped JSON")
    elif result.completion_tokens != tokens or result.finish_reason != "length":
        result.errors.append("fixed-length throughput request did not reach the token limit")
    result.ok = not result.errors
    return result


def summarize(waves):
    rows = []
    measured = [wave for wave in waves if wave["phase"] == "measured" and "server_metrics" in wave and "aggregate_tok_s" in wave]
    for case_id, concurrency in sorted({(w["case"], w["concurrency"]) for w in measured}):
        group = [w for w in measured if w["case"] == case_id and w["concurrency"] == concurrency]
        streams = [s for w in group for s in w["streams"]]
        valid = [s for s in streams if s["ok"]]
        row = {"case": case_id, "concurrency": concurrency, "trials": len(group),
               "successful_requests": len(valid), "failed_requests": len(streams) - len(valid),
               "aggregate_tok_s": distribution(w["aggregate_tok_s"] for w in group if w["ok"]),
               "server_decode_tok_s": distribution(w["server_metrics"]["server_decode_tok_s"] for w in group if w["ok"]),
               "draft_acceptance": distribution(w["server_metrics"]["draft_acceptance"] for w in group if w["ok"])}
        for key in ("ttft_s", "content_ttft_s", "elapsed_s", "prompt_tokens", "completion_tokens"):
            row[key] = distribution(s[key] for s in valid)
        # SSE delivers speculative groups, not individual tokens; never label
        # this estimator as true per-token GPU latency or kernel throughput.
        row["client_decode_proxy_tok_s"] = distribution((s["completion_tokens"] - 1) / s["decode_s"] for s in valid if s["decode_s"] and s["completion_tokens"] > 1)
        rows.append(row)
    return rows


def save_markdown(path, report):
    lines = ["# Content benchmark", "", f"Status: {report['status']}. Model: `{report['settings']['model']}`.", "",
             "Fixed synthetic workloads; this is a speed/transport benchmark, not a model-quality leaderboard.", "",
             "| Workload | C | Requests OK/all | Aggregate tok/s | Server decode tok/s/request | Client decode proxy tok/s | TTFT s | Draft acceptance |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "N/A" if value is None else f"{value:.3f}"
    for row in report.get("summary", []):
        values = [row[k]["median"] for k in ("aggregate_tok_s", "server_decode_tok_s", "client_decode_proxy_tok_s", "ttft_s", "draft_acceptance")]
        lines.append(f"| {row['case']} | {row['concurrency']} | {row['successful_requests']}/{row['successful_requests'] + row['failed_requests']} | " + " | ".join(map(fmt, values)) + " |")
    lines += ["", "Medians across measured trials; raw JSON includes ranges, prompts, usage, SSE events and metadata.",
              "Server decode = (output tokens − completed requests) / summed server request-decode seconds; at C>1 this is a per-request rate, not aggregate throughput.",
              "Client decode proxy uses first-to-last meaningful SSE delta time; speculative batched chunks prevent exact per-token latency measurement.",
              "TTFT includes input processing, image processing when applicable, scheduling and transport. Image encoder time is not separately available here.",
              "Draft acceptance is accepted/drafted tokens. Missing server metrics are N/A, not zero."]
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://spark-head.local:8888")
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--cases", help="comma-separated case IDs")
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--warmup-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--budget-seconds", type=float, default=1800)
    parser.add_argument("--seed", type=int, default=5205)
    parser.add_argument("--metrics", choices=("required", "off"), default="required")
    parser.add_argument("--provenance", type=Path, help="immutable image/model/config evidence JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true", help="write request manifest without network access")
    args = parser.parse_args()
    levels = [int(v) for v in args.concurrency.split(",")]
    if not levels or min(levels) < 1 or len(set(levels)) != len(levels) or min(args.trials, args.max_tokens, args.warmup_tokens) < 1:
        parser.error("concurrency must be unique and positive; trials/token limits must be positive")
    if not all(math.isfinite(v) and v > 0 for v in (args.timeout, args.budget_seconds)):
        parser.error("timeouts and budgets must be finite and positive")
    if args.output.exists():
        parser.error("output exists; use a fresh path to preserve evidence")
    fixture = json.loads(args.fixture.read_text())
    cases = fixture["cases"]
    if len({c["id"] for c in cases}) != len(cases):
        parser.error("fixture IDs must be unique")
    if args.cases:
        selected = set(args.cases.split(","))
        if not selected.issubset({c["id"] for c in cases}):
            parser.error("unknown case ID")
        cases = [c for c in cases if c["id"] in selected]
    if not cases:
        parser.error("empty case selection")
    report = {"schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
              "status": "planned", "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "fixture": fixture, "fixture_sha256": json_hash(fixture), "waves": [], "summary": [],
              "client_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "transport_sha256": hashlib.sha256(Path(__file__).with_name("streaming_client.py").read_bytes()).hexdigest()}
    if args.provenance:
        report["provenance"] = json.loads(args.provenance.read_text())
    report["examples"] = [{"case": c["id"], "body": make_body(c, args.model, args.max_tokens, args.seed, args.fixture.parent, "measured", 1, 0)[0]} for c in cases]
    write_report(str(args.output), report)
    if args.plan_only:
        print(f"Planned {len(cases)} cases; no network requests.")
        return
    started = time.perf_counter()
    deadline = started + args.budget_seconds
    def remaining(cap):
        value = min(cap, deadline - time.perf_counter())
        if value <= 0:
            raise TimeoutError("whole-suite budget exhausted")
        return value
    def snap():
        return snapshot(args.base_url, args.model, remaining(10)) if args.metrics == "required" else {k: None for k in METRICS}
    try:
        report["server"] = {endpoint: get_json(args.base_url.rstrip("/") + endpoint, remaining(10)) for endpoint in ("/version", "/v1/models")}
        report["status"] = "running"
        for case in cases:
            for concurrency in levels:
                for trial in range(args.trials + 1):
                    phase = "warmup" if trial == 0 else "measured"
                    tokens = args.warmup_tokens if trial == 0 and "expected_json" not in case else args.max_tokens
                    before = snap()
                    if args.metrics == "required" and (before["running"] != 0 or before["waiting"] != 0):
                        raise RuntimeError("server not idle before wave; exclusive access lost")
                    wave = {"case": case["id"], "category": case.get("category", case["id"]), "concurrency": concurrency,
                            "trial": trial, "phase": phase, "streams": [], "before_metrics": before, "ok": False}
                    report["waves"].append(wave)
                    write_report(str(args.output), report)
                    bodies = [make_body(case, args.model, tokens, args.seed, args.fixture.parent, phase, trial, request_id) for request_id in range(concurrency)]
                    wave["images"] = bodies[0][1]
                    completed_at = []
                    def run_one(request_id, body, timeout):
                        result = run_stream(args.base_url, args.model, tokens, request_id, body=body, timeout=timeout)
                        return result, time.perf_counter()
                    wall = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        pending = []
                        for request_id in range(concurrency):
                            pending.append(pool.submit(run_one, request_id, bodies[request_id][0], remaining(args.timeout)))
                        for future in as_completed(pending):
                            result, ended = future.result()
                            completed_at.append(ended)
                            result = validate_result(result, case, tokens)
                            wave["streams"].append(asdict(result))
                            request_path = args.output.parent / (args.output.stem + "-requests") / f"{case['id']}-c{concurrency}-t{trial}-r{result.request}.json"
                            write_report(str(request_path), asdict(result))
                    # End at the last completed HTTP stream, not after serializing
                    # the ever-growing report. Per-request artifacts stay durable.
                    wave["wall_s"] = max(completed_at) - wall
                    wave["streams"].sort(key=lambda s: s["request"])
                    outputs = sum(s["completion_tokens"] or 0 for s in wave["streams"])
                    after = snap()
                    # Exported counters may settle a short time after the last SSE.
                    settle = min(deadline, time.perf_counter() + 5)
                    while args.metrics == "required" and after["requests"] is not None and before["requests"] is not None and after["requests"] - before["requests"] < concurrency and time.perf_counter() < settle:
                        time.sleep(min(0.1, max(0, settle - time.perf_counter())))
                        after = snap()
                    wave["after_metrics"] = after
                    wave["server_metrics"] = metric_delta(before, after, concurrency, outputs)
                    wave["aggregate_tok_s"] = sum(s["completion_tokens"] for s in wave["streams"] if s["ok"]) / wave["wall_s"]
                    wave["ok"] = all(s["ok"] for s in wave["streams"]) and (args.metrics == "off" or wave["server_metrics"]["exact"])
                    report["summary"] = summarize(report["waves"])
                    report["elapsed_s"] = time.perf_counter() - started
                    write_report(str(args.output), report)
                    print(f"{phase} {case['id']} C{concurrency} trial={trial} aggregate={wave['aggregate_tok_s']:.2f} tok/s exact={wave['server_metrics']['exact']} ok={wave['ok']} elapsed={report['elapsed_s']:.0f}s", flush=True)
                    if not wave["ok"]:
                        raise RuntimeError("request correctness or exclusive server-metric gate failed")
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_s"] = time.perf_counter() - started
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["summary"] = summarize(report["waves"])
        write_report(str(args.output), report)
        save_markdown(args.output.with_suffix(".md"), report)


if __name__ == "__main__":
    main()
