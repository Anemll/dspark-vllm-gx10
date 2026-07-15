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


def rows_by_size(report: dict[str, object]) -> dict[int, dict[str, object]]:
    return {
        int(row["target_tokens"]): row
        for row in report["summary"]  # type: ignore[index]
    }


def prompt_fingerprints(report: dict[str, object]) -> dict[tuple[int, int], str]:
    return {
        (int(row["target_tokens"]), int(row["trial"])): str(row["prompt_sha256"])
        for row in report["results"]  # type: ignore[index]
    }


def number(value: object, digits: int = 1) -> str:
    return f"{float(value):,.{digits}f}" if value is not None else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--output")
    args = parser.parse_args()

    before = load_report(args.before)
    after = load_report(args.after)
    before_rows = rows_by_size(before)
    after_rows = rows_by_size(after)
    if prompt_fingerprints(before) != prompt_fingerprints(after):
        parser.error("reports did not use identical token prompts")
    sizes = sorted(before_rows.keys() & after_rows.keys())
    if not sizes:
        parser.error("reports do not contain any common prompt sizes")

    before_label = str(before.get("label") or Path(args.before).stem)
    after_label = str(after.get("label") or Path(args.after).stem)
    lines = [
        f"Before: `{before_label}` / `{before.get('version', 'unknown')}`  ",
        f"After: `{after_label}` / `{after.get('version', 'unknown')}`",
        "",
        "| Input tokens | Before server tok/s | After server tok/s | Gain | "
        "Before TTFT | After TTFT |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for size in sizes:
        old = before_rows[size]
        new = after_rows[size]
        old_tps = old.get("median_server_prefill_tps")
        new_tps = new.get("median_server_prefill_tps")
        gain = (
            (float(new_tps) / float(old_tps) - 1) * 100
            if old_tps is not None and new_tps is not None and float(old_tps) != 0
            else None
        )
        gain_text = f"{gain:+.1f}%" if gain is not None else "n/a"
        lines.append(
            f"| {size:,} | {number(old_tps)} | {number(new_tps)} | "
            f"{gain_text} | {number(old['median_ttft_s'], 3)}s | "
            f"{number(new['median_ttft_s'], 3)}s |"
        )
    rendered = "\n".join(lines) + "\n"
    print(rendered, end="")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            output.write(rendered)


if __name__ == "__main__":
    main()
