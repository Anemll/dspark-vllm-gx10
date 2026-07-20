#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Start T8 real-prompt capture immediately after API readiness.

This runner deliberately keeps the readiness-to-first-prompt path local and
HTTP-only.  It writes a minimal readiness record, starts the first prompt, and
defers heavyweight node evidence collection to the caller's post-prompt phase.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request


PROMPTS = {
    "short_code": (
        "Write a correct, tested Python implementation of a thread-safe LRU "
        "cache. Explain the invariants briefly, then give the code."
    ),
    "long_code_html": (
        "Create a complete single-file HTML canvas space game with keyboard "
        "controls, scoring, collision detection, particle effects, "
        "pause/restart, and responsive layout. Continue implementing until "
        "the token limit; do not repeat sections."
    ),
    "tool_agentic": (
        "You are diagnosing a two-node inference service. Produce a concise "
        "tool-driven investigation plan, including exact observations to "
        "collect, decision branches, rollback points, and a final evidence "
        "checklist."
    ),
    "json_structured": (
        "Return only valid JSON describing a deployment plan with keys "
        "assumptions, stages, checks, rollback, and metrics. Each stage must "
        "have name, commands, success_criteria, and failure_action."
    ),
    "long_context_retrieval": (
        "Background record: "
        + "alpha beta gamma delta " * 500
        + "The recovery code is ORCHID-7319. "
        + "epsilon zeta eta theta " * 500
        + "Question: state the recovery code, then explain in two sentences "
        "where it appeared."
    ),
}

RESTORE_EXIT_CODE = 75


class RestoreRequired(RuntimeError):
    """The outage caller must stop measurement and begin restoration."""


@dataclass(frozen=True)
class ReadyEvent:
    wall_time: str
    monotonic: float
    attempts: int
    http_status: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _get_status(url: str, *, timeout: float) -> int:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return int(response.status)


def wait_for_readiness(
    base_url: str,
    *,
    timeout_seconds: float,
    poll_seconds: float,
    request_timeout_seconds: float,
) -> ReadyEvent:
    deadline = time.monotonic() + timeout_seconds
    attempts = 0
    health_url = base_url.rstrip("/") + "/health"
    while time.monotonic() < deadline:
        attempts += 1
        try:
            status = _get_status(health_url, timeout=request_timeout_seconds)
        except (OSError, urllib.error.URLError):
            status = 0
        if status == 200:
            return ReadyEvent(
                wall_time=_utc_now(),
                monotonic=time.monotonic(),
                attempts=attempts,
                http_status=status,
            )
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(poll_seconds, remaining))
    raise RestoreRequired("API missed the readiness deadline")


def _write_restore_marker(output_dir: Path, reason: str) -> None:
    _atomic_json(
        output_dir / "RESTORE_REQUIRED.json",
        {
            "schema_version": 1,
            "restore_required": True,
            "reason": reason,
            "recorded_at": _utc_now(),
        },
    )


def _run_prompt(
    *,
    base_url: str,
    model: str,
    name: str,
    prompt: str,
    max_tokens: int,
    request_timeout_seconds: float,
) -> dict[str, object]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(
        request, timeout=request_timeout_seconds
    ) as response:
        payload = json.load(response)
    elapsed = time.monotonic() - started
    text = payload["choices"][0]["message"]["content"]
    return {
        "name": name,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "output": text,
        "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "elapsed_s": elapsed,
        "usage": payload.get("usage", {}),
        "finish_reason": payload["choices"][0].get("finish_reason"),
    }


def run_capture(
    *,
    base_url: str,
    model: str,
    output_dir: Path,
    readiness_timeout_seconds: float,
    readiness_poll_seconds: float,
    readiness_request_timeout_seconds: float,
    ready_to_first_prompt_budget_seconds: float,
    prompt_timeout_seconds: float,
    max_tokens: int,
) -> dict[str, object]:
    if ready_to_first_prompt_budget_seconds <= 0:
        raise ValueError("ready-to-first-prompt budget must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    restore_marker = output_dir / "RESTORE_REQUIRED.json"
    if restore_marker.exists():
        raise RestoreRequired("a restore-required marker already exists")

    try:
        ready = wait_for_readiness(
            base_url,
            timeout_seconds=readiness_timeout_seconds,
            poll_seconds=readiness_poll_seconds,
            request_timeout_seconds=readiness_request_timeout_seconds,
        )
        _atomic_json(
            output_dir / "readiness.json",
            {
                "schema_version": 1,
                "ready_at": ready.wall_time,
                "health_http_status": ready.http_status,
                "readiness_attempts": ready.attempts,
                "pre_prompt_privileged_commands": 0,
                "heavy_evidence_deferred_until_after_prompts": True,
            },
        )

        first_prompt_started = time.monotonic()
        first_prompt_started_at = _utc_now()
        ready_to_first_prompt = first_prompt_started - ready.monotonic
        if ready_to_first_prompt > ready_to_first_prompt_budget_seconds:
            raise RestoreRequired(
                "readiness-to-first-prompt delay "
                f"{ready_to_first_prompt:.6f}s exceeded "
                f"{ready_to_first_prompt_budget_seconds:.6f}s"
            )

        rows = []
        for name, prompt in PROMPTS.items():
            print(f"running {name}", flush=True)
            rows.append(
                _run_prompt(
                    base_url=base_url,
                    model=model,
                    name=name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    request_timeout_seconds=prompt_timeout_seconds,
                )
            )
        with urllib.request.urlopen(
            urllib.request.Request(
                base_url.rstrip("/") + "/metrics", method="GET"
            ),
            timeout=readiness_request_timeout_seconds,
        ) as response:
            metrics_text = response.read().decode("utf-8", "strict")
        _atomic_text(output_dir / "metrics-after-prompts.txt", metrics_text)
        result = {
            "schema_version": 1,
            "base_url": base_url,
            "model": model,
            "ready_at": ready.wall_time,
            "first_prompt_started_at": first_prompt_started_at,
            "ready_to_first_prompt_s": ready_to_first_prompt,
            "ready_to_first_prompt_budget_s": (
                ready_to_first_prompt_budget_seconds
            ),
            "pre_prompt_privileged_commands": 0,
            "heavy_evidence_deferred_until_after_prompts": True,
            "metrics_after_prompts_sha256": hashlib.sha256(
                metrics_text.encode("utf-8")
            ).hexdigest(),
            "rows": rows,
        }
        _atomic_json(output_dir / "prompts.json", result)
        return result
    except BaseException as error:
        _write_restore_marker(output_dir, f"{type(error).__name__}: {error}")
        raise


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--readiness-timeout-seconds", type=_positive_float, default=480.0
    )
    parser.add_argument(
        "--readiness-poll-seconds", type=_positive_float, default=1.0
    )
    parser.add_argument(
        "--readiness-request-timeout-seconds",
        type=_positive_float,
        default=2.0,
    )
    parser.add_argument(
        "--ready-to-first-prompt-budget-seconds",
        type=_positive_float,
        default=5.0,
    )
    parser.add_argument(
        "--prompt-timeout-seconds", type=_positive_float, default=900.0
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")

    try:
        result = run_capture(
            base_url=args.base_url,
            model=args.model,
            output_dir=args.output_dir,
            readiness_timeout_seconds=args.readiness_timeout_seconds,
            readiness_poll_seconds=args.readiness_poll_seconds,
            readiness_request_timeout_seconds=(
                args.readiness_request_timeout_seconds
            ),
            ready_to_first_prompt_budget_seconds=(
                args.ready_to_first_prompt_budget_seconds
            ),
            prompt_timeout_seconds=args.prompt_timeout_seconds,
            max_tokens=args.max_tokens,
        )
    except RestoreRequired as error:
        print(f"RESTORE_REQUIRED: {error}", file=sys.stderr)
        return RESTORE_EXIT_CODE
    print(
        "T8_PROMPT_CAPTURE_PASS "
        f"ready_to_first_prompt_s={result['ready_to_first_prompt_s']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
