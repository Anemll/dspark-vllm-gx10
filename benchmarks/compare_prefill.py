#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Render a before/after Markdown table from two prefill benchmark reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_report(path: str) -> dict[str, object]:
    with open(path, encoding="utf-8") as source:
        return json.load(source)


def rows_by_shape(
    report: dict[str, object],
) -> dict[tuple[int, int], dict[str, object]]:
    return {
        (int(row.get("concurrency", 1)), int(row["target_tokens"])): row
        for row in report["summary"]  # type: ignore[index]
    }


def rows_by_size(report: dict[str, object]) -> dict[int, dict[str, object]]:
    """Retain the original concurrency-1 helper for external callers."""
    return {
        size: row
        for (concurrency, size), row in rows_by_shape(report).items()
        if concurrency == 1
    }


def request_prompt_fingerprints(
    report: dict[str, object],
) -> dict[tuple[int, int, int], tuple[str, ...]]:
    fingerprints: dict[tuple[int, int, int], tuple[str, ...]] = {}
    for row in report["results"]:  # type: ignore[index]
        requests = row.get("requests")
        if isinstance(requests, list):
            hashes = tuple(str(request["prompt_sha256"]) for request in requests)
        else:
            hashes = (str(row["prompt_sha256"]),)
        key = (
            int(row.get("concurrency", 1)),
            int(row["target_tokens"]),
            int(row["trial"]),
        )
        fingerprints[key] = hashes
    return fingerprints


def prompt_fingerprints(report: dict[str, object]) -> dict[tuple[int, int], str]:
    """Retain the original concurrency-1 fingerprint helper."""
    return {
        (int(row["target_tokens"]), int(row["trial"])): str(row["prompt_sha256"])
        for row in report["results"]  # type: ignore[index]
        if int(row.get("concurrency", 1)) == 1
    }


def number(value: object, digits: int = 1) -> str:
    return f"{float(value):,.{digits}f}" if value is not None else "n/a"


def row_is_valid(row: dict[str, object]) -> bool:
    """Old reports predate row_valid; new reports fail closed explicitly."""
    return bool(row.get("row_valid", True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--output")
    parser.add_argument(
        "--allow-unmatched-prompts",
        action="store_true",
        help="render an explicitly caveated aggregate comparison when prompt fingerprints differ",
    )
    args = parser.parse_args()

    before = load_report(args.before)
    after = load_report(args.after)
    before_rows = rows_by_shape(before)
    after_rows = rows_by_shape(after)
    matching_prompts = request_prompt_fingerprints(before) == request_prompt_fingerprints(after)
    if not matching_prompts and not args.allow_unmatched_prompts:
        parser.error("reports did not use identical token prompts")
    shapes = sorted(before_rows.keys() & after_rows.keys())
    if not shapes:
        parser.error("reports do not contain any common prompt shapes")
    multi_concurrency = any(concurrency != 1 for concurrency, _size in shapes)

    before_label = str(before.get("label") or Path(args.before).stem)
    after_label = str(after.get("label") or Path(args.after).stem)
    lines = [
        f"Before: `{before_label}` / `{before.get('version', 'unknown')}`  ",
        f"After: `{after_label}` / `{after.get('version', 'unknown')}`",
    ]
    if not matching_prompts:
        lines.extend(
            [
                "",
                "Comparison caveat: prompt fingerprints and/or trial counts differ; "
                "this is a same-size aggregate comparison, not a paired prompt-matched A/B.",
            ]
        )
    artifact = before.get("model_artifact")
    if isinstance(artifact, dict):
        lines.extend(
            [
                "",
                "Model: "
                f"[{artifact['huggingface_repo']}]"
                f"(https://huggingface.co/{artifact['huggingface_repo']}), "
                f"{artifact['architecture']}. The checkpoint contains "
                f"{artifact['safetensors_shards']} FP8 Safetensors shards totaling "
                f"{artifact['safetensors_gib']} GiB on each node; the serving KV "
                f"cache uses `{artifact['kv_cache_dtype']}`.",
            ]
        )
    single_node = after.get("single_node_reference")
    if isinstance(single_node, dict):
        lines.extend(
            [
                "",
                "Single-node reference: "
                f"{single_node['reason']} See the "
                f"[TP=1 fit check]({single_node['report']}); no single-node "
                "throughput samples are valid.",
            ]
        )
    if multi_concurrency:
        lines.extend(
            [
                "",
                "| Concurrency | Input tokens | Valid | "
                "Before aggregate tok/s | After aggregate tok/s | Gain | "
                "Before median TTFT | After median TTFT | "
                "Before pooled p95 | After pooled p95 |",
                "|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "| Input tokens | Before server tok/s | After server tok/s | "
                "Gain | Before TTFT | After TTFT |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
    for concurrency, size in shapes:
        old = before_rows[(concurrency, size)]
        new = after_rows[(concurrency, size)]
        if multi_concurrency:
            valid = row_is_valid(old) and row_is_valid(new)
            old_aggregate = (
                old.get(
                    "median_aggregate_input_tps",
                    old.get("median_client_input_tps"),
                )
                if valid
                else None
            )
            new_aggregate = (
                new.get(
                    "median_aggregate_input_tps",
                    new.get("median_client_input_tps"),
                )
                if valid
                else None
            )
            gain = (
                (float(new_aggregate) / float(old_aggregate) - 1) * 100
                if old_aggregate is not None
                and new_aggregate is not None
                and float(old_aggregate) != 0
                else None
            )
            gain_text = f"{gain:+.1f}%" if gain is not None else "n/a"
            old_ttft = old.get("median_ttft_s") if valid else None
            new_ttft = new.get("median_ttft_s") if valid else None
            old_p95 = old.get("pooled_p95_ttft_s") if valid else None
            new_p95 = new.get("pooled_p95_ttft_s") if valid else None
            lines.append(
                f"| {concurrency} | {size:,} | {'yes' if valid else 'no'} | "
                f"{number(old_aggregate)} | {number(new_aggregate)} | "
                f"{gain_text} | {number(old_ttft, 3)}s | "
                f"{number(new_ttft, 3)}s | {number(old_p95, 3)}s | "
                f"{number(new_p95, 3)}s |"
            )
        else:
            old_tps = old.get("median_server_prefill_tps")
            new_tps = new.get("median_server_prefill_tps")
            gain = (
                (float(new_tps) / float(old_tps) - 1) * 100
                if old_tps is not None
                and new_tps is not None
                and float(old_tps) != 0
                else None
            )
            gain_text = f"{gain:+.1f}%" if gain is not None else "n/a"
            lines.append(
                f"| {size:,} | {number(old_tps)} | {number(new_tps)} | "
                f"{gain_text} | {number(old['median_ttft_s'], 3)}s | "
                f"{number(new['median_ttft_s'], 3)}s |"
            )
    if multi_concurrency:
        footer = (
            "Warmed steady-state comparison on two GX10 nodes (TP=2): "
            f"{before.get('trials_per_size', 'unknown')} trials per shape, "
            f"seed {before.get('seed', 'unknown')}, one output token, matching "
            "per-request prompt hashes, and no unrelated requests. A row is "
            "valid only when every trial has exact usage/Prometheus/computed "
            "token counts and zero prefix-cache hits."
        )
    else:
        # Preserve the legacy concurrency-1 rendering verbatim.
        if matching_prompts:
            footer = (
                "Warmed steady-state comparison on two GX10 nodes (TP=2): "
                f"{before.get('trials_per_size', 'unknown')} trials per size, "
                f"seed {before.get('seed', 'unknown')}, one output token, zero "
                "prefix-cache hits, no overlapping requests, and matching prompt "
                "hashes across versions."
            )
        else:
            footer = (
                "Same-size aggregate comparison on two GX10 nodes (TP=2). "
                "Each report retains its own seed, trial count, cache-isolation, "
                "and exact-token validity checks; see the caveat above."
            )
    lines.extend(["", footer])
    rendered = "\n".join(lines) + "\n"
    print(rendered, end="")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            output.write(rendered)


if __name__ == "__main__":
    main()
