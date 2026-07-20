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
OVERLAP_PREFIX = "vllm:dspark_overlap_"


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
    from types import SimpleNamespace

    import numpy as np
    import torch

    from vllm.v1.worker.gpu.spec_decode.dspark.confidence import (
        get_confidence_metrics,
        observe_engine_compaction_telemetry,
    )
    from vllm.v1.worker.gpu.spec_decode.dspark.overlap_trace import (
        observe_engine_overlap_trace,
    )
    from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

    os.environ["VLLM_DSPARK_CONFIDENCE_SCHEDULER"] = "on"
    os.environ["VLLM_DSPARK_CONFIDENCE_THRESHOLD"] = str(threshold)

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

    class CopyEvent:
        def __init__(self, ready: bool):
            self.ready = ready
            self.synchronized = False

        def query(self) -> bool:
            return self.ready

        def synchronize(self) -> None:
            self.synchronized = True
            self.ready = True

    def compact_and_export(
        req_ids: list[str],
        proposals: list[list[int]],
        *,
        ready: bool,
    ) -> tuple[dict[str, int], list[int], bool]:
        handler = DraftTokensHandler.__new__(DraftTokensHandler)
        handler.variable_draft_lengths = True
        handler.req_ids = req_ids
        handler.draft_tokens_np = np.asarray(proposals, dtype=np.int32)
        handler.copy_event = CopyEvent(ready)
        handler.copy_wait_fallbacks = 0
        handler.last_physical_target_rows = None
        handler.last_d2h_copy_fallback = None
        output = SimpleNamespace(
            scheduled_spec_decode_tokens={req_id: [-1] * 5 for req_id in req_ids},
            num_scheduled_tokens={req_id: 6 for req_id in req_ids},
            total_num_scheduled_tokens=6 * len(req_ids),
        )
        invalid = handler.compact_scheduler_output(output)
        rows, fallback = handler.get_last_compaction_telemetry()
        if rows is None or fallback is None:
            raise RuntimeError("real compaction did not produce telemetry evidence")
        observe_engine_compaction_telemetry(rows, fallback)
        return invalid, rows, fallback

    ready_result = compact_and_export(
        ["a", "b"],
        [[10, 11, 12, -1, -1], [20, 21, -1, -1, -1]],
        ready=True,
    )
    if ready_result != ({"a": 2, "b": 3}, [4, 3], False):
        raise RuntimeError(f"bad ready compaction evidence: {ready_result}")
    fallback_result = compact_and_export(
        ["c"],
        [[30, 31, -1, -1, -1]],
        ready=False,
    )
    if fallback_result != ({"c": 3}, [3], True):
        raise RuntimeError(f"bad fallback compaction evidence: {fallback_result}")
    metrics.observe_dropped_batch()
    traces = (
        (
            (10.0, 20.0, 3.0, 1.0, 5.0, 39.0),
            (11.0, 21.0, 4.0, 2.0, 6.0, 44.0),
        ),
        (
            (12.0, 22.0, 5.0, 1.5, 7.0, 47.5),
            (13.0, 23.0, 6.0, 2.5, 8.0, 52.5),
        ),
    )
    phase_names = ("draft", "verify", "commit", "nccl_wait", "overhead", "total")
    for block in traces:
        observe_engine_overlap_trace(
            {
                "schema_version": 1,
                "world_size": 2,
                "rank_traces": [
                    {
                        "rank": rank,
                        **dict(zip(phase_names, values, strict=True)),
                    }
                    for rank, values in enumerate(block)
                ],
            }
        )


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

    physical_count = lines_by_family["physical_target_rows"]
    if len(physical_count) != 1 or _numeric_value(physical_count[0]) != 3.0:
        raise AssertionError(
            f"physical-row count did not cross the real compaction path: "
            f"{physical_count}"
        )
    physical_sum = [
        line
        for line in _metric_lines(
            exposition, METRIC_PREFIX + "physical_target_rows_sum"
        )
        if 'threshold="0.4"' in line
    ]
    if len(physical_sum) != 1 or _numeric_value(physical_sum[0]) != 10.0:
        raise AssertionError(
            f"physical-row sum did not preserve [4, 3, 3]: {physical_sum}"
        )

    d2h = lines_by_family["d2h_copy_completion"]
    for result in ("ready", "fallback_wait"):
        matches = [line for line in d2h if f'result="{result}"' in line]
        if len(matches) != 1 or _numeric_value(matches[0]) != 1.0:
            raise AssertionError(f"bad D2H {result} series: {matches}")

    overlap_phases = (
        "draft",
        "verify",
        "commit",
        "nccl_wait",
        "overhead",
        "total",
    )
    overlap_counts: dict[str, float] = {}
    block_counts: dict[str, float] = {}
    for rank in (0, 1):
        rank_label = f'rank="{rank}"'
        blocks = [
            line
            for line in _metric_lines(exposition, OVERLAP_PREFIX + "blocks_total")
            if rank_label in line
        ]
        if len(blocks) != 1 or _numeric_value(blocks[0]) != 2.0:
            raise AssertionError(
                f"overlap block count drift for rank {rank}: {blocks}"
            )
        block_counts[str(rank)] = _numeric_value(blocks[0])
        for phase in overlap_phases:
            counts = [
                line
                for line in _metric_lines(
                    exposition, OVERLAP_PREFIX + "phase_ms_count"
                )
                if rank_label in line and f'phase="{phase}"' in line
            ]
            if len(counts) != 1 or _numeric_value(counts[0]) != 2.0:
                raise AssertionError(
                    f"overlap {phase} count drift for rank {rank}: {counts}"
                )
            overlap_counts[f"rank{rank}:{phase}"] = _numeric_value(counts[0])

    return {
        "probability_count_by_position": probability_counts,
        "probability_bucket_series_by_position": probability_buckets,
        "physical_target_rows_count": _numeric_value(physical_count[0]),
        "physical_target_rows_sum": _numeric_value(physical_sum[0]),
        "families": {key: len(lines) for key, lines in lines_by_family.items()},
        "overlap_blocks_by_rank": block_counts,
        "overlap_phase_counts": overlap_counts,
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
    if METRIC_PREFIX in control or OVERLAP_PREFIX in control:
        raise AssertionError("control unexpectedly exposed worker DSpark metrics")

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
