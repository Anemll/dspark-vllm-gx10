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
DEFAULT_MINIMUM_SPEC_ACCEPTANCE_RATE = 0.30
DEFAULT_MINIMUM_SPEC_ACCEPTANCE_LENGTH = 2.50
DEFAULT_MINIMUM_SPEC_LAST_POSITION_RATE = 0.06


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
    require_spec_metrics: bool = False,
    expected_spec_positions: int = 5,
    minimum_spec_acceptance_rate: float = DEFAULT_MINIMUM_SPEC_ACCEPTANCE_RATE,
    minimum_spec_acceptance_length: float = DEFAULT_MINIMUM_SPEC_ACCEPTANCE_LENGTH,
    minimum_spec_last_position_rate: float = DEFAULT_MINIMUM_SPEC_LAST_POSITION_RATE,
) -> list[dict[str, object]]:
    require(report.get("model") == model, "decode model mismatch")
    require(report.get("max_tokens") == max_tokens, "decode token limit mismatch")
    rows = report.get("trials")
    require(isinstance(rows, list), "decode trials missing")
    require(len(rows) == len(concurrencies) * trials, "decode trial count mismatch")
    if require_spec_metrics:
        require(
            report.get("spec_decode_metric_source")
            == {
                "num_drafts": "vllm:spec_decode_num_drafts_total",
                "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
                "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
                "accepted_per_position": (
                    "vllm:spec_decode_num_accepted_tokens_per_pos_total"
                ),
            },
            "decode speculative metric source mismatch",
        )
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
        acceptance_rates: list[float] = []
        acceptance_lengths: list[float] = []
        per_position_rates: list[list[float]] = []
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
            if require_spec_metrics:
                spec = row.get("spec_decode")
                require(isinstance(spec, dict), "decode speculative metrics missing")
                require(int(spec.get("num_drafts", 0)) > 0, "decode draft count missing")
                require(int(spec.get("draft_tokens", 0)) > 0, "decode draft tokens missing")
                require(
                    minimum_spec_acceptance_rate
                    <= float(spec.get("aggregate_acceptance_rate", -1))
                    <= 1,
                    "decode acceptance rate below gate",
                )
                require(
                    minimum_spec_acceptance_length
                    <= float(spec.get("mean_acceptance_length", 0))
                    <= expected_spec_positions + 1,
                    "decode acceptance length below gate",
                )
                positions = spec.get("per_position_acceptance_rates")
                require(
                    isinstance(positions, list)
                    and len(positions) == expected_spec_positions,
                    "decode per-position acceptance missing",
                )
                require(
                    all(
                        isinstance(value, (int, float)) and 0 <= value <= 1
                        for value in positions
                    ),
                    "decode per-position acceptance invalid",
                )
                require(
                    all(left >= right for left, right in zip(positions, positions[1:])),
                    "decode per-position acceptance is not monotonic",
                )
                require(
                    float(positions[-1]) >= minimum_spec_last_position_rate,
                    "decode final-position acceptance below gate",
                )
                acceptance_rates.append(float(spec["aggregate_acceptance_rate"]))
                acceptance_lengths.append(float(spec["mean_acceptance_length"]))
                per_position_rates.append([float(value) for value in positions])
        row_summary: dict[str, object] = {
            "concurrency": concurrency,
            "median_aggregate_output_tps": statistics.median(rates),
            "min_aggregate_output_tps": min(rates),
            "max_aggregate_output_tps": max(rates),
            "median_mean_ttft_s": statistics.median(ttfts),
        }
        if require_spec_metrics:
            row_summary.update(
                {
                    "median_spec_acceptance_rate": statistics.median(acceptance_rates),
                    "median_spec_acceptance_length": statistics.median(
                        acceptance_lengths
                    ),
                    "median_spec_acceptance_per_position": [
                        statistics.median(row[position] for row in per_position_rates)
                        for position in range(expected_spec_positions)
                    ],
                }
            )
        summary.append(row_summary)
    return summary


def compare_decode_summaries(
    candidate: list[dict[str, object]],
    baseline: list[dict[str, object]],
    *,
    minimum_output_tps_ratio: float,
    minimum_spec_retention_ratio: float,
    minimum_spec_position_retention_ratio: float,
) -> list[dict[str, object]]:
    require(
        [row["concurrency"] for row in candidate]
        == [row["concurrency"] for row in baseline],
        "candidate/baseline decode concurrency mismatch",
    )
    comparisons: list[dict[str, object]] = []
    for candidate_row, baseline_row in zip(candidate, baseline):
        concurrency = int(candidate_row["concurrency"])
        candidate_tps = float(candidate_row["median_aggregate_output_tps"])
        baseline_tps = float(baseline_row["median_aggregate_output_tps"])
        output_ratio = candidate_tps / baseline_tps
        require(
            output_ratio >= minimum_output_tps_ratio,
            f"decode x{concurrency} output throughput below baseline gate",
        )

        candidate_acceptance = float(candidate_row["median_spec_acceptance_rate"])
        baseline_acceptance = float(baseline_row["median_spec_acceptance_rate"])
        acceptance_retention = candidate_acceptance / baseline_acceptance
        require(
            acceptance_retention >= minimum_spec_retention_ratio,
            f"decode x{concurrency} acceptance retention below gate",
        )

        candidate_length = float(candidate_row["median_spec_acceptance_length"])
        baseline_length = float(baseline_row["median_spec_acceptance_length"])
        accepted_excess_retention = (candidate_length - 1) / (baseline_length - 1)
        require(
            accepted_excess_retention >= minimum_spec_retention_ratio,
            f"decode x{concurrency} accepted-length retention below gate",
        )

        candidate_positions = [
            float(value)
            for value in candidate_row["median_spec_acceptance_per_position"]
        ]
        baseline_positions = [
            float(value)
            for value in baseline_row["median_spec_acceptance_per_position"]
        ]
        require(
            len(candidate_positions) == len(baseline_positions),
            "candidate/baseline acceptance position mismatch",
        )
        position_retentions = [
            candidate_value / baseline_value
            for candidate_value, baseline_value in zip(
                candidate_positions, baseline_positions
            )
        ]
        require(
            all(
                value >= minimum_spec_position_retention_ratio
                for value in position_retentions
            ),
            f"decode x{concurrency} per-position acceptance retention below gate",
        )
        comparisons.append(
            {
                "concurrency": concurrency,
                "candidate_median_output_tps": candidate_tps,
                "baseline_median_output_tps": baseline_tps,
                "output_tps_ratio": output_ratio,
                "acceptance_retention_ratio": acceptance_retention,
                "accepted_excess_length_retention_ratio": accepted_excess_retention,
                "per_position_acceptance_retention_ratios": position_retentions,
            }
        )
    return comparisons


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
    parser.add_argument("--label", required=True)
    parser.add_argument("--expected-content", default="NVIDIA ready")
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--sizes", default=",".join(str(value) for value in DEFAULT_SIZES))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--require-spec-metrics", action="store_true")
    parser.add_argument("--expected-spec-positions", type=int, default=5)
    parser.add_argument(
        "--minimum-spec-acceptance-rate",
        type=float,
        default=DEFAULT_MINIMUM_SPEC_ACCEPTANCE_RATE,
    )
    parser.add_argument(
        "--minimum-spec-acceptance-length",
        type=float,
        default=DEFAULT_MINIMUM_SPEC_ACCEPTANCE_LENGTH,
    )
    parser.add_argument(
        "--minimum-spec-last-position-rate",
        type=float,
        default=DEFAULT_MINIMUM_SPEC_LAST_POSITION_RATE,
    )
    parser.add_argument("--baseline-decode-json", type=Path)
    parser.add_argument("--minimum-output-tps-ratio", type=float, default=1.0)
    parser.add_argument("--minimum-spec-retention-ratio", type=float, default=0.8)
    parser.add_argument(
        "--minimum-spec-position-retention-ratio", type=float, default=0.5
    )
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument("--seed", type=int, default=4104)
    parser.add_argument("--request-timeout", type=float, default=240)
    parser.add_argument("--decode-timeout", type=float, default=900)
    parser.add_argument("--prefill-timeout", type=float, default=1800)
    args = parser.parse_args()

    concurrencies = parse_csv(args.concurrency)
    sizes = parse_csv(args.sizes)
    require(args.trials > 0 and args.max_tokens > 0, "trials/max-tokens must be positive")
    require(args.expected_spec_positions > 0, "expected spec positions must be positive")
    require(
        0 < args.minimum_spec_acceptance_rate <= 1,
        "minimum speculative acceptance rate must be in (0, 1]",
    )
    require(
        1 < args.minimum_spec_acceptance_length <= args.expected_spec_positions + 1,
        "minimum speculative acceptance length is invalid",
    )
    require(
        0 < args.minimum_spec_last_position_rate <= 1,
        "minimum final-position acceptance rate must be in (0, 1]",
    )
    require(
        args.minimum_output_tps_ratio > 0,
        "minimum output throughput ratio must be positive",
    )
    require(
        0 < args.minimum_spec_retention_ratio <= 1,
        "minimum speculative retention ratio must be in (0, 1]",
    )
    require(
        0 < args.minimum_spec_position_retention_ratio <= 1,
        "minimum per-position retention ratio must be in (0, 1]",
    )
    require(
        args.baseline_decode_json is None or args.require_spec_metrics,
        "baseline decode comparison requires speculative metrics",
    )
    require(args.label.strip() == args.label and bool(args.label), "label must be nonempty and trimmed")
    require(
        args.expected_content.strip() == args.expected_content and bool(args.expected_content),
        "expected content must be nonempty and trimmed",
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    decode_path = root / "benchmarks" / "benchmark_dsv4_api.py"
    prefill_path = root / "benchmarks" / "benchmark_prefill.py"
    decode_module = load_module(decode_path, "nvfp4_decode_benchmark")

    smoke_started = time.monotonic()
    nonstream = run_nonstream_smoke(args.base_url, args.model, args.request_timeout)
    nonstream_content = nonstream["choices"][0]["message"]["content"]
    require(
        " ".join(nonstream_content.split()) == args.expected_content,
        "nonstream smoke content mismatch",
    )
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
    decode_command = [
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
    ]
    if args.require_spec_metrics:
        decode_command.extend(
            [
                "--require-spec-metrics",
                "--expected-spec-positions",
                str(args.expected_spec_positions),
            ]
        )
    run_command(
        decode_command,
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
        require_spec_metrics=args.require_spec_metrics,
        expected_spec_positions=args.expected_spec_positions,
        minimum_spec_acceptance_rate=args.minimum_spec_acceptance_rate,
        minimum_spec_acceptance_length=args.minimum_spec_acceptance_length,
        minimum_spec_last_position_rate=args.minimum_spec_last_position_rate,
    )
    print("decode gate passed", flush=True)

    decode_comparison: list[dict[str, object]] | None = None
    if args.baseline_decode_json is not None:
        baseline_path = args.baseline_decode_json.resolve()
        baseline_report = json.loads(baseline_path.read_text())
        baseline_rows = baseline_report.get("trials")
        require(isinstance(baseline_rows, list), "baseline decode trials missing")
        selected_baseline_rows = [
            row for row in baseline_rows if row.get("concurrency") in concurrencies
        ]
        baseline_subset = dict(baseline_report)
        baseline_subset["trials"] = selected_baseline_rows
        baseline_summary = validate_decode_report(
            baseline_subset,
            model=str(baseline_report.get("model")),
            concurrencies=concurrencies,
            trials=args.trials,
            max_tokens=args.max_tokens,
            require_spec_metrics=True,
            expected_spec_positions=args.expected_spec_positions,
            minimum_spec_acceptance_rate=args.minimum_spec_acceptance_rate,
            minimum_spec_acceptance_length=args.minimum_spec_acceptance_length,
            minimum_spec_last_position_rate=args.minimum_spec_last_position_rate,
        )
        decode_comparison = compare_decode_summaries(
            decode_summary,
            baseline_summary,
            minimum_output_tps_ratio=args.minimum_output_tps_ratio,
            minimum_spec_retention_ratio=args.minimum_spec_retention_ratio,
            minimum_spec_position_retention_ratio=(
                args.minimum_spec_position_retention_ratio
            ),
        )
        print("decode production comparison passed", flush=True)

    prefill_summary: list[dict[str, object]] = []
    if not args.skip_prefill:
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
                args.label,
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
        "decode_comparison": decode_comparison,
        "prefill": prefill_summary,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("=== DECODE MEDIAN AGGREGATE OUTPUT TOK/S ===")
    for row in decode_summary:
        print(f"C{row['concurrency']}: {row['median_aggregate_output_tps']:.2f}")
    if prefill_summary:
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
