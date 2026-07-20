#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prove DSpark worker metrics cross into the real API ``/metrics`` route.

This is intentionally a process-and-HTTP integration probe, not an isolated
CollectorRegistry unit test.  It first reproduces the old missing-series
behavior without multiprocess storage, then starts the same API route and a
separate emitting worker with the CLI bootstrap enabled and requires every
confidence metric family in the HTTP scrape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


METRIC_PREFIX = "vllm:dspark_confidence_"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_text(url: str, timeout: float = 2.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")
        return response.read().decode("utf-8")


def _wait_ready(port: int, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"API process exited rc={process.returncode}: {stdout}\n{stderr}"
            )
        try:
            if _http_text(f"http://127.0.0.1:{port}/health") == "ok":
                return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.05)
    raise TimeoutError("API process did not expose /health within 20 seconds")


def _stop(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _run_api(port: int) -> None:
    from fastapi import FastAPI
    from fastapi.responses import PlainTextResponse
    import uvicorn

    from vllm.entrypoints.serve.instrumentator.metrics import attach_router

    app = FastAPI()

    @app.get("/health", response_class=PlainTextResponse)
    async def health() -> PlainTextResponse:
        return PlainTextResponse("ok")

    attach_router(app)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def _run_worker(threshold: float) -> None:
    import torch

    from vllm.v1.worker.gpu.spec_decode.dspark.confidence import (
        get_confidence_metrics,
    )

    probabilities = torch.tensor(
        [
            [0.92, 0.78, 0.61, 0.35, 0.18],
            [0.88, 0.52, 0.27, 0.16, 0.08],
        ],
        dtype=torch.float32,
    )
    metrics = get_confidence_metrics(threshold)
    observed = metrics.observe(torch.logit(probabilities))
    if observed["prefix_lengths"] != [3, 2]:
        raise RuntimeError(f"unexpected confidence prefix: {observed}")
    metrics.observe_physical_target_rows([4, 3])
    metrics.observe_d2h_copy_completion(fallback_wait=False)
    metrics.observe_d2h_copy_completion(fallback_wait=True)
    metrics.observe_dropped_batch()


def _metric_lines(exposition: str, name: str) -> list[str]:
    prefix = name + "{"
    return [line for line in exposition.splitlines() if line.startswith(prefix)]


def _numeric_value(line: str) -> float:
    return float(line.rsplit(" ", 1)[1])


def _assert_fixed_exposition(exposition: str) -> dict[str, object]:
    probability_counts: dict[str, float] = {}
    probability_buckets: dict[str, int] = {}
    for position in range(5):
        label = f'position="{position}"'
        counts = [
            line
            for line in _metric_lines(
                exposition, METRIC_PREFIX + "probability_count"
            )
            if label in line and 'threshold="0.4"' in line
        ]
        buckets = [
            line
            for line in _metric_lines(
                exposition, METRIC_PREFIX + "probability_bucket"
            )
            if label in line and 'threshold="0.4"' in line
        ]
        if len(counts) != 1 or _numeric_value(counts[0]) != 2.0:
            raise AssertionError(f"missing/non-empty p{position} count: {counts}")
        if not buckets or max(_numeric_value(line) for line in buckets) != 2.0:
            raise AssertionError(f"missing/non-empty p{position} buckets")
        probability_counts[str(position)] = _numeric_value(counts[0])
        probability_buckets[str(position)] = len(buckets)

    required = {
        "below_threshold": METRIC_PREFIX + "below_threshold_total",
        "position_exposed": METRIC_PREFIX + "position_exposed_total",
        "prefix_length": METRIC_PREFIX + "prefix_length_count",
        "physical_target_rows": METRIC_PREFIX + "physical_target_rows_count",
        "d2h_copy_completion": METRIC_PREFIX + "d2h_copy_completion_total",
        "telemetry_dropped": METRIC_PREFIX + "telemetry_dropped_batches_total",
    }
    lines_by_family = {
        key: _metric_lines(exposition, metric) for key, metric in required.items()
    }
    missing = [key for key, lines in lines_by_family.items() if not lines]
    if missing:
        raise AssertionError(f"missing confidence metric families: {missing}")

    d2h = lines_by_family["d2h_copy_completion"]
    for result in ("ready", "fallback_wait"):
        matches = [line for line in d2h if f'result="{result}"' in line]
        if len(matches) != 1 or _numeric_value(matches[0]) != 1.0:
            raise AssertionError(f"bad D2H {result} series: {matches}")

    return {
        "probability_count_by_position": probability_counts,
        "probability_bucket_series_by_position": probability_buckets,
        "families": {key: len(lines) for key, lines in lines_by_family.items()},
    }


def _start_api_and_worker(
    *,
    env: dict[str, str],
    threshold: float,
) -> tuple[str, int, int]:
    port = _free_port()
    api: subprocess.Popen[str] | None = None
    try:
        api = subprocess.Popen(
            [sys.executable, __file__, "--api", "--port", str(port)],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _wait_ready(port, api)
        worker = subprocess.Popen(
            [
                sys.executable,
                __file__,
                "--worker",
                "--threshold",
                str(threshold),
            ],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            worker_stdout, worker_stderr = worker.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker_stdout, worker_stderr = worker.communicate(timeout=5)
            raise TimeoutError("confidence metric worker exceeded 30 seconds")
        if worker.returncode != 0:
            raise RuntimeError(
                f"worker failed rc={worker.returncode}: "
                f"{worker_stdout}\n{worker_stderr}"
            )
        exposition = _http_text(f"http://127.0.0.1:{port}/metrics", timeout=5)
        return exposition, int(api.pid), int(worker.pid)
    finally:
        _stop(api)


def _run_controller(output: Path) -> None:
    control_env = os.environ.copy()
    control_env.pop("PROMETHEUS_MULTIPROC_DIR", None)
    control, control_api_pid, control_worker_pid = _start_api_and_worker(
        env=control_env,
        threshold=0.4,
    )
    if METRIC_PREFIX in control:
        raise AssertionError("control unexpectedly exposed worker confidence metrics")

    from vllm.entrypoints.cli.main import (
        _cleanup_owned_prometheus_multiprocess_dir,
        _setup_prometheus_multiprocess_for_serve,
    )

    old_env = os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
    try:
        multiprocess_dir = _setup_prometheus_multiprocess_for_serve(
            ["vllm", "serve"]
        )
        if multiprocess_dir is None or not Path(multiprocess_dir).is_dir():
            raise AssertionError("CLI bootstrap did not create multiprocess storage")
        fixed_env = os.environ.copy()
        fixed, fixed_api_pid, fixed_worker_pid = _start_api_and_worker(
            env=fixed_env,
            threshold=0.4,
        )
        summary = _assert_fixed_exposition(fixed)
        files = sorted(path.name for path in Path(multiprocess_dir).iterdir())
        if not any(name.startswith("histogram_") for name in files):
            raise AssertionError(f"worker histogram mmap missing: {files}")
        if not any(name.startswith("counter_") for name in files):
            raise AssertionError(f"worker counter mmap missing: {files}")

        result = {
            "schema_version": 1,
            "ok": True,
            "control": {
                "api_pid": control_api_pid,
                "worker_pid": control_worker_pid,
                "worker_series_absent": True,
            },
            "fixed": {
                "api_pid": fixed_api_pid,
                "worker_pid": fixed_worker_pid,
                "multiprocess_dir": multiprocess_dir,
                "multiprocess_files": files,
                "http_metrics_sha256": hashlib.sha256(fixed.encode()).hexdigest(),
                **summary,
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True))
    finally:
        _cleanup_owned_prometheus_multiprocess_dir()
        if old_env is None:
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
        else:
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = old_env


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--api", action="store_true")
    mode.add_argument("--worker", action="store_true")
    parser.add_argument("--port", type=int)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.api:
        if args.port is None:
            parser.error("--api requires --port")
        _run_api(args.port)
        return 0
    if args.worker:
        _run_worker(args.threshold)
        return 0
    if args.output is None:
        parser.error("controller mode requires --output")
    _run_controller(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
