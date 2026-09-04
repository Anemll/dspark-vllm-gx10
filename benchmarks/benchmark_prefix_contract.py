#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Eight C1 requests testing known-answer prefix reuse across 1024 tokens.

Require correct JSON values plus bracketed server cache-hit evidence, not
equality of free-form prose. No cache reset, tools, or constrained decoding.
Exit 0: passed; 1: failed; 2: inconclusive cache/measurement evidence.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

try:
    from .benchmark_agent import fit_coding_body, request_json, token_evidence
    from .benchmark_dsv4_api import json_hash, run_stream, write_report
    from .benchmark_prefill import MetricSnapshot, fetch_text, metric_total
except ImportError:
    from benchmark_agent import fit_coding_body, request_json, token_evidence
    from benchmark_dsv4_api import json_hash, run_stream, write_report
    from benchmark_prefill import MetricSnapshot, fetch_text, metric_total


FIXTURE = {
    "version": "prefix-contract-v1",
    "system": "Apply SET records in chronological order across user messages. The last SET for a key wins. Ignore all neutral reference notes. Return only one JSON object with exactly the keys anchor and current and their final string values. No explanation or Markdown.",
    "coding_request": "SET anchor = PINE263\nSET current = COPPER417\nReturn the final anchor and current values after reading the reference notes below.\n",
    "snippet_template": "Neutral reference note {index}: the archive contains ordinary maintenance records. This note does not change any values.\n",
}
INITIAL = {"anchor": "PINE263", "current": "COPPER417"}
SIZES = (1023, 1025)


def prepare_cases(model, max_tokens, seed, tokenize):
    cases = []
    for size in SIZES:
        body, evidence = fit_coding_body(model, size, max_tokens, seed, f"prefix:{size}", FIXTURE, tokenize)
        body["ignore_eos"] = False
        phases = [("initial", copy.deepcopy(body), INITIAL), ("exact-repeat", copy.deepcopy(body), INITIAL)]
        for phase, value in (("growing-turn", "INDIGO592"), ("divergent-suffix", "SAFFRON846")):
            branch = copy.deepcopy(body)
            # Neutral history does not repeat either answer or supply the anchor.
            branch["messages"].extend([
                {"role": "assistant", "content": "I have read the records."},
                {"role": "user", "content": f"SET current = {value}\nReturn the final anchor and current values as JSON."},
            ])
            phases.append((phase, branch, {"anchor": INITIAL["anchor"], "current": value}))
        for phase, request, expected in phases:
            cases.append({
                "family_tokens": size, "phase": phase, "body": request,
                "request_sha256": json_hash(request), "expected": dict(expected),
                "tokenization": evidence if phase in {"initial", "exact-repeat"} else token_evidence(request, tokenize),
            })
    return cases


def snapshot(base_url, timeout):
    text = fetch_text(base_url.rstrip("/") + "/metrics", timeout=timeout)
    return MetricSnapshot(
        metric_total(text, "vllm:request_prefill_time_seconds_sum"),
        metric_total(text, "vllm:request_prefill_time_seconds_count"),
        metric_total(text, "vllm:request_prefill_kv_computed_tokens_sum"),
        metric_total(text, "vllm:prompt_tokens_by_source_total", {"source": "local_cache_hit"}),
        metric_total(text, "vllm:prompt_tokens_total"),
    )


def evaluate(case, stream, before, after):
    delta = {name: getattr(after, name) - getattr(before, name) for name in asdict(before)}
    prompt_count = case["tokenization"]["count"]
    exact = (
        all(math.isfinite(value) and value >= 0 for value in delta.values())
        and abs(delta["prefill_requests"] - 1) < 0.01
        and abs(delta["prompt_tokens"] - prompt_count) < 0.01
        and delta["cache_hit_tokens"] <= prompt_count
    )
    parsed = None
    semantic_error = None
    try:
        parsed = json.loads(stream.content)
        if parsed != case["expected"]:
            semantic_error = "JSON values or keys differ from the fixture"
    except (ValueError, TypeError) as exc:
        semantic_error = f"answer is not a complete JSON object: {exc}"
    if not stream.ok:
        classification = "stream_failure"
    elif stream.prompt_tokens != prompt_count:
        classification = "tokenization_mismatch"
    elif stream.finish_reason != "stop":
        classification = "answer_did_not_finish"
    elif semantic_error:
        if case["phase"] == "initial":
            classification = "cold_answer_failure" if exact and delta["cache_hit_tokens"] == 0 else "initial_answer_failure_cache_unverified"
        else:
            classification = "replay_answer_failure_requires_investigation"
    elif not exact:
        classification = "inconclusive_metrics"
    elif case["phase"] == "initial" and delta["cache_hit_tokens"] != 0:
        classification = "inconclusive_initial_not_cold"
    elif case["phase"] != "initial" and delta["cache_hit_tokens"] <= 0:
        classification = "inconclusive_cache_reuse_not_observed"
    else:
        classification = "passed"
    return {
        "classification": classification, "semantic_correct": semantic_error is None,
        "semantic_error": semantic_error, "parsed_answer": parsed,
        "metrics_exact": exact, "metrics_delta": delta,
        "cache_hit_observed": exact and delta["cache_hit_tokens"] > 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://spark-head.local:8888")
    parser.add_argument("--model", default="deepseek-v4-flash-0731-dspark")
    parser.add_argument("--seed", type=int, default=4106)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--prepare-timeout", type=float, default=60)
    parser.add_argument("--budget-seconds", type=float, default=240)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 256 or any(not math.isfinite(value) or value <= 0 for value in (args.timeout, args.prepare_timeout, args.budget_seconds)):
        parser.error("finite positive time limits and max-tokens in 1..256 are required")
    if Path(args.output).exists():
        parser.error("output already exists; preserve previous evidence and choose a new path")
    started = time.monotonic()
    report = {
        "schema_version": 1, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "settings": vars(args), "sizes": SIZES, "concurrency": 1,
        "fixture": FIXTURE, "fixture_sha256": json_hash(FIXTURE),
        "client_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "requests": [], "cases": [],
        "interpretation": "A cold failure is a fixture/model failure. Replay answer failure merits investigation, not an automatic cache-corruption claim. Passing requires actual measured cache hits on every replay/branch. OS networking and CUDA service failure still require an outer process budget.",
    }

    def save():
        report["elapsed_s"] = time.monotonic() - started
        write_report(args.output, report)

    def remaining(limit):
        value = min(limit, args.budget_seconds - (time.monotonic() - started))
        if value <= 0:
            raise TimeoutError("prefix-contract total budget exceeded")
        return value

    save()
    try:
        preparation_deadline = time.monotonic() + args.prepare_timeout

        def tokenize(body):
            allowance = min(30, preparation_deadline - time.monotonic())
            if allowance <= 0:
                raise TimeoutError("prefix-contract preparation budget exceeded")
            return request_json(args.base_url, "/tokenize", body, timeout=remaining(allowance))

        report["cases"] = prepare_cases(args.model, args.max_tokens, args.seed, tokenize)
        report["cases_sha256"] = json_hash(report["cases"])
        save()
        for index, case in enumerate(report["cases"]):
            record = {"case": index, "family_tokens": case["family_tokens"], "phase": case["phase"], "status": "running"}
            report["requests"].append(record)
            save()
            before = snapshot(args.base_url, remaining(10))
            record["metrics_before"] = asdict(before)
            save()
            stream = run_stream(args.base_url, args.model, args.max_tokens, index, body=case["body"], timeout=remaining(args.timeout))
            record["stream"] = asdict(stream)
            save()
            after = snapshot(args.base_url, remaining(10))
            record["metrics_after"] = asdict(after)
            record.update(evaluate(case, stream, before, after))
            record["status"] = "complete"
            save()
            print(f"family={case['family_tokens']} phase={case['phase']} result={record['classification']} cached={record['metrics_delta']['cache_hit_tokens']} TTFT={stream.ttft_s}", flush=True)
            if record["classification"] != "passed":
                report["status"] = "inconclusive" if record["classification"].startswith("inconclusive_") else "failed"
                break
        else:
            report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        if report["requests"] and report["requests"][-1]["status"] == "running":
            report["requests"][-1]["status"] = "failed"
        raise
    finally:
        save()
    if report["status"] != "passed":
        raise SystemExit(2 if report["status"] == "inconclusive" else 1)


if __name__ == "__main__":
    main()
