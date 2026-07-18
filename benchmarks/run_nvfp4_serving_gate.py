#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run and validate the bounded NVIDIA NVFP4 TP=2 serving gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.request


DEFAULT_SIZES = [1024, 2048, 4096, 8192, 16384, 32768]
DEFAULT_CONCURRENCY = [1, 2, 4]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def finite_positive(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def request_json(url: str, body: dict[str, object] | None = None, timeout: float = 120):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def run_nonstream_smoke(base_url: str, model: str, timeout: float) -> dict[str, object]:
    response = request_json(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly two words: NVIDIA ready",
                }
            ],
            "max_tokens": 16,
            "temperature": 0,
            "stream": False,
        },
        timeout,
    )
    require(response.get("model") == model, "nonstream smoke returned wrong model")
    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, "nonstream choices invalid")
    choice = choices[0]
    require(isinstance(choice, dict), "nonstream choice invalid")
    message = choice.get("message")
    require(isinstance(message, dict), "nonstream message missing")
    content = message.get("content")
    require(isinstance(content, str) and content.strip(), "nonstream content empty")
    require(choice.get("finish_reason") in {"stop", "length"}, "nonstream finish invalid")
    usage = response.get("usage")
    require(isinstance(usage, dict), "nonstream usage missing")
    require(int(usage.get("prompt_tokens", 0)) > 0, "nonstream prompt usage missing")
    require(int(usage.get("completion_tokens", 0)) > 0, "nonstream completion usage missing")
    return response


def validate_decode_report(
    report: dict[str, object],
    *,
    model: str,
    concurrencies: list[int],
    trials: int,
    max_tokens: int,
) -> list[dict[str, object]]:
    require(report.get("model") == model, "decode model mismatch")
    require(report.get("max_tokens") == max_tokens, "decode token limit mismatch")
    rows = report.get("trials")
    require(isinstance(rows, list), "decode trials missing")
    require(len(rows) == len(concurrencies) * trials, "decode trial count mismatch")
    summary: list[dict[str, object]] = []
    for concurrency in concurrencies:
        selected = [row for row in rows if row.get("concurrency") == concurrency]
        require(len(selected) == trials, f"decode x{concurrency} trial count mismatch")
        require(
            sorted(int(row.get("trial", 0)) for row in selected)
            == list(range(1, trials + 1)),
            f"decode x{concurrency} trial IDs mismatch",
        )
        rates: list[float] = []
        ttfts: list[float] = []
        for row in selected:
            require(row.get("total_tokens") == concurrency * max_tokens, "decode total token mismatch")
            require(finite_positive(row.get("wall_s")), "decode wall time invalid")
            require(finite_positive(row.get("aggregate_token_tps")), "decode aggregate rate invalid")
            require(finite_positive(row.get("mean_ttft_s")), "decode TTFT invalid")
            streams = row.get("streams")
            require(isinstance(streams, list) and len(streams) == concurrency, "decode streams mismatch")
            for stream in streams:
                require(isinstance(stream, dict), "decode stream invalid")
                require(stream.get("completion_tokens") == max_tokens, "decode stream token mismatch")
                require(stream.get("finish_reason") == "length", "decode finish reason mismatch")
                require(int(stream.get("prompt_tokens", 0)) > 0, "decode prompt usage missing")
                require(int(stream.get("chunks", 0)) > 0, "decode chunks missing")
                for key in ("ttft_s", "decode_s", "elapsed_s", "token_tps", "chunk_tps"):
                    require(finite_positive(stream.get(key)), f"decode {key} invalid")
            rates.append(float(row["aggregate_token_tps"]))
            ttfts.append(float(row["mean_ttft_s"]))
        summary.append(
            {
                "concurrency": concurrency,
                "median_aggregate_output_tps": statistics.median(rates),
                "min_aggregate_output_tps": min(rates),
                "max_aggregate_output_tps": max(rates),
                "median_mean_ttft_s": statistics.median(ttfts),
            }
        )
    return summary


def validate_prefill_report(
    report: dict[str, object],
    *,
    model: str,
    sizes: list[int],
    concurrencies: list[int],
    trials: int,
    seed: int,
) -> list[dict[str, object]]:
    require(report.get("schema_version") == 2, "prefill schema mismatch")
    require(report.get("model") == model, "prefill model mismatch")
    require(report.get("sizes") == sizes, "prefill sizes mismatch")
    require(report.get("concurrencies") == concurrencies, "prefill concurrencies mismatch")
    require(report.get("trials_per_size") == trials, "prefill trial count mismatch")
    require(report.get("seed") == seed, "prefill seed mismatch")
    rows = report.get("results")
    summary = report.get("summary")
    require(isinstance(rows, list), "prefill results missing")
    require(isinstance(summary, list), "prefill summary missing")
    require(len(rows) == len(sizes) * len(concurrencies) * trials, "prefill result count mismatch")
    require(len(summary) == len(sizes) * len(concurrencies), "prefill summary count mismatch")
    prompt_hashes: set[str] = set()
    for row in rows:
        require(isinstance(row, dict), "prefill result invalid")
        concurrency = int(row.get("concurrency", 0))
        target = int(row.get("target_tokens", 0))
        require(concurrency in concurrencies and target in sizes, "prefill shape unexpected")
        require(row.get("measurement_valid") is True, "prefill measurement invalid")
        for key in (
            "metrics_exact",
            "cache_isolated",
            "prompt_lengths_exact",
            "metrics_prompt_tokens_exact",
            "server_computed_tokens_exact",
            "completion_lengths_exact",
        ):
            require(row.get(key) is True, f"prefill {key} failed")
        require(row.get("server_cache_hit_tokens") == 0, "prefill cache hit detected")
        require(abs(float(row.get("metrics_request_delta", 0)) - concurrency) < 0.01, "prefill request delta mismatch")
        require(row.get("prompt_tokens") == target * concurrency, "prefill prompt count mismatch")
        require(row.get("completion_tokens") == concurrency, "prefill completion count mismatch")
        require(row.get("server_computed_tokens") == target * concurrency, "prefill computed token mismatch")
        require(row.get("metrics_prompt_tokens_delta") == target * concurrency, "prefill metric prompt mismatch")
        for key in (
            "ttft_s",
            "elapsed_s",
            "client_input_tps",
            "server_prefill_s",
            "server_prefill_tps",
            "batch_ttft_s",
            "batch_wall_s",
            "aggregate_input_tps",
            "mean_ttft_s",
            "p95_ttft_s",
        ):
            require(finite_positive(row.get(key)), f"prefill {key} invalid")
        requests = row.get("requests")
        require(isinstance(requests, list) and len(requests) == concurrency, "prefill requests mismatch")
        for request in requests:
            require(isinstance(request, dict), "prefill request invalid")
            require(request.get("usage_reported") is True, "prefill usage missing")
            require(request.get("prompt_tokens") == target, "prefill request prompt mismatch")
            require(request.get("completion_tokens") == 1, "prefill request completion mismatch")
            require(request.get("finish_reason") == "length", "prefill finish reason mismatch")
            digest = request.get("prompt_sha256")
            require(isinstance(digest, str) and len(digest) == 64, "prefill prompt hash invalid")
            require(digest not in prompt_hashes, "prefill prompt hash reused")
            prompt_hashes.add(digest)
    validated: list[dict[str, object]] = []
    for concurrency in concurrencies:
        for target in sizes:
            matches = [
                row
                for row in summary
                if row.get("concurrency") == concurrency and row.get("target_tokens") == target
            ]
            require(len(matches) == 1, "prefill summary shape mismatch")
            row = matches[0]
            require(row.get("row_valid") is True, "prefill summary row invalid")
            require(row.get("trials") == trials, "prefill summary trials mismatch")
            require(row.get("valid_trials") == trials, "prefill valid trials mismatch")
            require(row.get("exact_server_trials") == trials, "prefill exact trials mismatch")
            require(row.get("cache_isolated_trials") == trials, "prefill cache summary mismatch")
            require(row.get("exact_length_trials") == trials, "prefill length summary mismatch")
            require(row.get("exact_computed_token_trials") == trials, "prefill computed summary mismatch")
            for key in (
                "median_ttft_s",
                "pooled_p95_ttft_s",
                "median_client_input_tps",
                "median_aggregate_input_tps",
                "median_server_prefill_tps",
                "median_server_mean_request_prefill_tps",
            ):
                require(finite_positive(row.get(key)), f"prefill summary {key} invalid")
            validated.append(dict(row))
    return validated


def run_command(command: list[str], *, timeout: float, log_path: Path) -> None:
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    elapsed = time.monotonic() - started
    require(process.returncode == 0, f"command failed rc={process.returncode}: {' '.join(command)}")
    print(f"completed in {elapsed:.1f}s: {log_path.name}", flush=True)


def parse_csv(value: str) -> list[int]:
    result = [int(item) for item in value.split(",")]
    require(result and all(item > 0 for item in result), "CSV values must be positive")
    require(len(result) == len(set(result)), "CSV values must be unique")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--sizes", default=",".join(str(value) for value in DEFAULT_SIZES))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--request-timeout", type=float, default=240)
    parser.add_argument("--decode-timeout", type=float, default=900)
    parser.add_argument("--prefill-timeout", type=float, default=1800)
    args = parser.parse_args()

    concurrencies = parse_csv(args.concurrency)
    sizes = parse_csv(args.sizes)
    require(args.trials > 0 and args.max_tokens > 0, "trials/max-tokens must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    decode_path = root / "benchmarks" / "benchmark_dsv4_api.py"
    prefill_path = root / "benchmarks" / "benchmark_prefill.py"
    decode_module = load_module(decode_path, "nvfp4_decode_benchmark")

    smoke_started = time.monotonic()
    nonstream = run_nonstream_smoke(args.base_url, args.model, args.request_timeout)
    stream = decode_module.run_stream(args.base_url, args.model, 64, 0)
    require(stream.completion_tokens == 64, "stream smoke token count mismatch")
    require(stream.finish_reason == "length", "stream smoke finish reason mismatch")
    require(stream.prompt_tokens > 0 and stream.chunks > 0, "stream smoke usage/chunks missing")
    for key in ("ttft_s", "decode_s", "elapsed_s", "token_tps", "chunk_tps"):
        require(finite_positive(getattr(stream, key)), f"stream smoke {key} invalid")
    smoke_report = {
        "elapsed_s": time.monotonic() - smoke_started,
        "nonstream": nonstream,
        "stream": asdict(stream),
    }
    (output_dir / "smoke.json").write_text(json.dumps(smoke_report, indent=2) + "\n")
    print("functional smoke passed", flush=True)

    decode_json = output_dir / "decode.json"
    run_command(
        [
            sys.executable,
            str(decode_path),
            "--base-url",
            args.base_url,
            "--model",
            args.model,
            "--concurrency",
            args.concurrency,
            "--trials",
            str(args.trials),
            "--max-tokens",
            str(args.max_tokens),
            "--output",
            str(decode_json),
        ],
        timeout=args.decode_timeout,
        log_path=output_dir / "decode.log",
    )
    decode_report = json.loads(decode_json.read_text())
    decode_summary = validate_decode_report(
        decode_report,
        model=args.model,
        concurrencies=concurrencies,
        trials=args.trials,
        max_tokens=args.max_tokens,
    )
    print("decode gate passed", flush=True)

    prefill_json = output_dir / "prefill.json"
    run_command(
        [
            sys.executable,
            str(prefill_path),
            "--base-url",
            args.base_url,
            "--model",
            args.model,
            "--sizes",
            args.sizes,
            "--concurrency",
            args.concurrency,
            "--trials",
            str(args.trials),
            "--warmup-tokens",
            "1024",
            "--shape-warmup-trials",
            "1",
            "--timeout",
            str(args.request_timeout),
            "--seed",
            str(args.seed),
            "--label",
            "cutlass-71d0a3c",
            "--report-target",
            "gx10-tp2",
            "--output",
            str(prefill_json),
        ],
        timeout=args.prefill_timeout,
        log_path=output_dir / "prefill.log",
    )
    prefill_report = json.loads(prefill_json.read_text())
    prefill_summary = validate_prefill_report(
        prefill_report,
        model=args.model,
        sizes=sizes,
        concurrencies=concurrencies,
        trials=args.trials,
        seed=args.seed,
    )
    print("prefill gate passed", flush=True)

    summary = {
        "schema_version": 1,
        "base_url": args.base_url,
        "model": args.model,
        "decode": decode_summary,
        "prefill": prefill_summary,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("=== DECODE MEDIAN AGGREGATE OUTPUT TOK/S ===")
    for row in decode_summary:
        print(f"C{row['concurrency']}: {row['median_aggregate_output_tps']:.2f}")
    print("=== PREFILL MEDIAN AGGREGATE INPUT TOK/S ===")
    for row in prefill_summary:
        print(
            f"C{row['concurrency']} L{row['target_tokens']}: "
            f"{row['median_aggregate_input_tps']:.2f} "
            f"TTFT={row['median_ttft_s']:.3f}s"
        )
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
