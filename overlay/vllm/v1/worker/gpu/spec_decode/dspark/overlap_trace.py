# SPDX-License-Identifier: Apache-2.0
"""Opt-in, lossless per-block DSpark draft/verify timing telemetry.

The trace is deliberately disabled by default.  When enabled it records CUDA
events around the target verifier, sampler/commit, and draft proposal.  A tiny
TP all-reduce after the target forward measures rank-arrival slack plus the
collective itself (``nccl_wait_ms``).  At block completion, a fixed-size vector
is gathered across TP ranks and serialized through ``ModelRunnerOutput``.

The final event synchronization and two tiny TP collectives add diagnostic
overhead.  They are part of the opt-in experiment and are never enabled in the
normal serving configuration.
"""

from __future__ import annotations

import json
import math
import os
import time
from threading import Lock
from typing import Any

import torch


TRACE_ENV = "VLLM_DSPARK_OVERLAP_TRACE"
TRACE_JSONL_ENV = "VLLM_DSPARK_OVERLAP_TRACE_JSONL"
TRACE_SCHEMA_VERSION = 1
PHASES = ("draft", "verify", "commit", "nccl_wait", "overhead", "total")

_metrics_lock = Lock()
_metrics_instance: DSparkOverlapMetrics | None = None
_signal_lock = Lock()
_signals: dict[str, torch.Tensor] = {}
_jsonl_lock = Lock()
_jsonl_sequence = 0


def overlap_trace_enabled(environment: dict[str, str] | None = None) -> bool:
    values = os.environ if environment is None else environment
    raw = values.get(TRACE_ENV, "0")
    if raw not in ("0", "1"):
        raise ValueError(f"{TRACE_ENV} must be exactly 0 or 1, got {raw!r}")
    return raw == "1"


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def _tp_signal(device: torch.device) -> torch.Tensor:
    key = str(device)
    with _signal_lock:
        signal = _signals.get(key)
        if signal is None:
            signal = torch.ones(1, dtype=torch.float32, device=device)
            _signals[key] = signal
        return signal


class DSparkOverlapBlockTrace:
    """One real decode block, measured on the current CUDA stream."""

    def __init__(self, *, device: torch.device):
        from vllm.distributed.parallel_state import get_tp_group

        if device.type != "cuda":
            raise ValueError(f"DSpark overlap tracing requires CUDA, got {device}")
        group = get_tp_group()
        self.device = device
        self.rank = int(group.rank_in_group)
        self.world_size = int(group.world_size)
        self.wall_start = time.perf_counter()
        self.block_start = _event()
        self.verify_start = _event()
        self.verify_end = _event()
        self.nccl_start = _event()
        self.nccl_end = _event()
        self.commit_start = _event()
        self.commit_end = _event()
        self.draft_start = _event()
        self.draft_end = _event()
        self.block_end = _event()
        self._nccl_result: torch.Tensor | None = None
        self._state = "created"
        self.block_start.record()

    def begin_verify(self) -> None:
        if self._state != "created":
            raise RuntimeError(f"verify start after {self._state}")
        self.verify_start.record()
        self._state = "verify"

    def end_verify_and_measure_rank_wait(self) -> None:
        from vllm.distributed.parallel_state import get_tp_group

        if self._state != "verify":
            raise RuntimeError(f"verify end after {self._state}")
        self.verify_end.record()
        self.nccl_start.record()
        # This diagnostic collective begins only after each rank has enqueued
        # its target work.  Its duration therefore exposes rank-arrival slack
        # plus the small collective latency; it is not claimed to be the sum of
        # every NCCL call inside the target model.
        self._nccl_result = get_tp_group().all_reduce(_tp_signal(self.device))
        self.nccl_end.record()
        self._state = "verified"

    def begin_commit(self) -> None:
        if self._state != "verified":
            raise RuntimeError(f"commit start after {self._state}")
        self.commit_start.record()
        self._state = "commit"

    def end_commit(self) -> None:
        if self._state != "commit":
            raise RuntimeError(f"commit end after {self._state}")
        self.commit_end.record()
        self._state = "committed"

    def begin_draft(self) -> None:
        if self._state != "committed":
            raise RuntimeError(f"draft start after {self._state}")
        self.draft_start.record()
        self._state = "draft"

    def end_draft_and_gather(self) -> dict[str, Any]:
        from vllm.distributed.parallel_state import get_tp_group

        if self._state != "draft":
            raise RuntimeError(f"draft end after {self._state}")
        self.draft_end.record()
        self.block_end.record()
        self.block_end.synchronize()
        wall_ms = (time.perf_counter() - self.wall_start) * 1000.0
        values = {
            "draft": float(self.draft_start.elapsed_time(self.draft_end)),
            "verify": float(self.verify_start.elapsed_time(self.verify_end)),
            "commit": float(self.commit_start.elapsed_time(self.commit_end)),
            "nccl_wait": float(self.nccl_start.elapsed_time(self.nccl_end)),
        }
        measured = sum(values.values())
        values["total"] = max(wall_ms, measured)
        values["overhead"] = values["total"] - measured
        local = torch.tensor(
            [values[phase] for phase in PHASES],
            dtype=torch.float32,
            device=self.device,
        )
        gathered = get_tp_group().all_gather(local, dim=0)
        rows = gathered.reshape(self.world_size, len(PHASES)).cpu().tolist()
        rank_traces = []
        for rank, row in enumerate(rows):
            record = {phase: float(row[index]) for index, phase in enumerate(PHASES)}
            if any(not math.isfinite(value) or value < 0.0 for value in record.values()):
                raise RuntimeError(f"invalid overlap timing for rank {rank}: {record}")
            rank_traces.append({"rank": rank, **record})
        self._state = "finished"
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "world_size": self.world_size,
            "rank_traces": rank_traces,
        }


def maybe_begin_overlap_trace(
    *,
    device: torch.device,
    dummy_run: bool,
    has_speculator: bool,
    is_verifier_block: bool,
) -> DSparkOverlapBlockTrace | None:
    if not overlap_trace_enabled() or dummy_run or not is_verifier_block:
        return None
    if not has_speculator:
        raise RuntimeError(
            f"{TRACE_ENV}=1 requires a speculative decoder on every real block"
        )
    return DSparkOverlapBlockTrace(device=device)


class DSparkOverlapMetrics:
    """Engine-process Prometheus collectors for gathered TP timing records."""

    def __init__(self, *, registry: Any | None = None):
        from prometheus_client import Counter, Histogram

        metric_kwargs = {} if registry is None else {"registry": registry}
        buckets = (
            0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
            10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0,
        )
        self.phase_ms = Histogram(
            "vllm:dspark_overlap_phase_ms",
            "Per-block DSpark CUDA phase and wall-overhead timing by TP rank.",
            ("phase", "rank"),
            buckets=buckets,
            **metric_kwargs,
        )
        self.blocks = Counter(
            "vllm:dspark_overlap_blocks",
            "Complete DSpark overlap trace records exported by TP rank.",
            ("rank",),
            **metric_kwargs,
        )

    def observe(self, trace: dict[str, Any]) -> None:
        if trace.get("schema_version") != TRACE_SCHEMA_VERSION:
            raise RuntimeError(f"unsupported overlap trace schema: {trace!r}")
        world_size = int(trace.get("world_size", 0))
        rows = trace.get("rank_traces")
        if not isinstance(rows, list) or len(rows) != world_size or world_size < 1:
            raise RuntimeError(
                f"incomplete overlap rank handoff: world_size={world_size}, rows={rows!r}"
            )
        seen: set[int] = set()
        for row in rows:
            rank = int(row["rank"])
            if rank in seen or not 0 <= rank < world_size:
                raise RuntimeError(f"invalid overlap rank record: {row!r}")
            seen.add(rank)
            rank_label = str(rank)
            values: dict[str, float] = {}
            for phase in PHASES:
                value = float(row[phase])
                if not math.isfinite(value) or value < 0.0:
                    raise RuntimeError(
                        f"invalid overlap {phase} for rank {rank}: {value}"
                    )
                values[phase] = value
            component_total = sum(values[phase] for phase in PHASES[:-1])
            total = values["total"]
            if not math.isclose(total, component_total, rel_tol=1e-5, abs_tol=0.05):
                raise RuntimeError(
                    "overlap phase conservation drift for rank "
                    f"{rank}: components={component_total}, total={total}"
                )
            for phase, value in values.items():
                self.phase_ms.labels(phase=phase, rank=rank_label).observe(value)
            self.blocks.labels(rank=rank_label).inc()
        if seen != set(range(world_size)):
            raise RuntimeError(f"overlap rank set drift: {sorted(seen)}")


def get_overlap_metrics() -> DSparkOverlapMetrics:
    global _metrics_instance
    with _metrics_lock:
        if _metrics_instance is None:
            _metrics_instance = DSparkOverlapMetrics()
        return _metrics_instance


def _append_trace_jsonl(trace: dict[str, Any]) -> None:
    """Persist one validated block handoff without buffering or aggregation."""

    path = os.environ.get(TRACE_JSONL_ENV)
    if path is None:
        return
    if not path or not os.path.isabs(path):
        raise ValueError(
            f"{TRACE_JSONL_ENV} must be an absolute path when set, got {path!r}"
        )
    parent = os.path.dirname(path)
    if not os.path.isdir(parent):
        raise FileNotFoundError(
            f"{TRACE_JSONL_ENV} parent directory does not exist: {parent}"
        )

    global _jsonl_sequence
    with _jsonl_lock:
        _jsonl_sequence += 1
        record = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "sequence": _jsonl_sequence,
            "observed_unix_ns": time.time_ns(),
            "trace": trace,
        }
        payload = (
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC,
            0o600,
        )
        try:
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise OSError(
                    f"short overlap trace write: {written} of {len(payload)} bytes"
                )
        finally:
            os.close(descriptor)


def observe_engine_overlap_trace(trace: dict[str, Any] | None) -> None:
    if trace is not None:
        get_overlap_metrics().observe(trace)
        _append_trace_jsonl(trace)
