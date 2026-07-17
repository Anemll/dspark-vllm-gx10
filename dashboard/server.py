#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Small, dependency-free live monitor for a vLLM server.

The dashboard keeps the Prometheus endpoint private: browsers fetch a curated,
same-origin snapshot rather than polling vLLM directly.  It samples at most
twice per second regardless of how many dashboard tabs are open.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import threading
import time
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


APP_DIR = Path(__file__).resolve().parent
METRICS_URL = os.environ.get("VLLM_METRICS_URL", "http://127.0.0.1:8888/metrics")
VERSION_URL = os.environ.get(
    "VLLM_VERSION_URL", f"{METRICS_URL.rsplit('/', 1)[0]}/version"
)
BIND_ADDRESS = os.environ.get("DASHBOARD_BIND", "127.0.0.1")
PORT = int(os.environ.get("DASHBOARD_PORT", "11001"))
POLL_CACHE_SECONDS = float(os.environ.get("DASHBOARD_POLL_CACHE_SECONDS", "0.45"))
HARDWARE_CACHE_SECONDS = float(os.environ.get("DASHBOARD_HARDWARE_CACHE_SECONDS", "2"))
LOAD_CACHE_SECONDS = 2.0
LOAD_LOG_TAIL = int(os.environ.get("DASHBOARD_LOAD_LOG_TAIL", "160"))
VERSION_CACHE_SECONDS = 10.0
HEAD_NODE_LABEL = os.environ.get("DASHBOARD_HEAD_LABEL", "SPARK-head")
WORKER_NODE_LABEL = os.environ.get("DASHBOARD_WORKER_LABEL", "SPARK-worker")
WORKER_SSH = os.environ.get("DASHBOARD_WORKER_SSH", "")
WORKER_HOST_KEY_ALIAS = os.environ.get("DASHBOARD_WORKER_HOST_KEY_ALIAS", "")
WORKER_IDENTITY_FILE = os.environ.get("DASHBOARD_WORKER_IDENTITY_FILE", "")
CONTAINER_NAME = os.environ.get(
    "DASHBOARD_CONTAINER_NAME", "dspark-vllm-gx10-vllm-dspark-1"
)
XFLASH_DEVICE = os.environ.get("DASHBOARD_NVME_DEVICE", "/dev/nvme0")

METRIC_LINE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{([^}]*)\})?\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|NaN|[+-]?Inf)$"
)
LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:\\.|[^"\\])*)"')


COUNTER_METRICS = {
    "vllm:generation_tokens_total": "generated_tokens",
    "vllm:prompt_tokens_total": "prompt_tokens",
    "vllm:spec_decode_num_accepted_tokens_total": "accepted_tokens",
    "vllm:spec_decode_num_draft_tokens_total": "draft_tokens",
    "vllm:request_success_total": "completed_requests",
    "vllm:time_to_first_token_seconds_sum": "ttft_sum",
    "vllm:time_to_first_token_seconds_count": "ttft_count",
    "vllm:inter_token_latency_seconds_sum": "itl_sum",
    "vllm:inter_token_latency_seconds_count": "itl_count",
    "vllm:e2e_request_latency_seconds_sum": "e2e_sum",
    "vllm:e2e_request_latency_seconds_count": "e2e_count",
}

GAUGE_METRICS = {
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
}


def _empty_metrics() -> dict[str, Any]:
    return {
        "model": "vLLM",
        "generated_tokens": 0.0,
        "prompt_tokens": 0.0,
        "accepted_tokens": 0.0,
        "draft_tokens": 0.0,
        "completed_requests": 0.0,
        "errors": 0.0,
        "ttft_sum": 0.0,
        "ttft_count": 0.0,
        "itl_sum": 0.0,
        "itl_count": 0.0,
        "e2e_sum": 0.0,
        "e2e_count": 0.0,
        "running": 0.0,
        "waiting": 0.0,
        "waiting_capacity": 0.0,
        "waiting_deferred": 0.0,
        "kv_cache_pct": 0.0,
    }


def _labels(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    return {
        match.group(1): bytes(match.group(2), "utf-8").decode("unicode_escape")
        for match in LABEL.finditer(raw)
    }


def parse_prometheus(payload: str) -> dict[str, Any]:
    metrics = _empty_metrics()
    for line in payload.splitlines():
        if not line or line.startswith("#"):
            continue
        match = METRIC_LINE.match(line)
        if not match:
            continue
        name, raw_labels, raw_value = match.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        labels = _labels(raw_labels)
        if labels.get("model_name"):
            metrics["model"] = labels["model_name"]

        counter_key = COUNTER_METRICS.get(name)
        if counter_key:
            metrics[counter_key] += value
            if name == "vllm:request_success_total" and labels.get("finished_reason") == "error":
                metrics["errors"] += value
            continue

        gauge_key = GAUGE_METRICS.get(name)
        if gauge_key:
            metrics[gauge_key] += value
            continue

        if name == "vllm:kv_cache_usage_perc":
            # This is a per-engine gauge.  A TP deployment reports one engine;
            # max() also remains meaningful if a future server reports several.
            metrics["kv_cache_pct"] = max(metrics["kv_cache_pct"], value * 100.0)
        elif name == "vllm:num_requests_waiting_by_reason":
            reason = labels.get("reason")
            if reason == "capacity":
                metrics["waiting_capacity"] += value
            elif reason == "deferred":
                metrics["waiting_deferred"] += value
    return metrics


GPU_QUERY = (
    "--query-gpu=temperature.gpu,power.draw,power.draw.instant,utilization.gpu",
    "--format=csv,noheader,nounits",
)
XFLASH_SMART_QUERY = ("sudo", "-n", "/usr/sbin/nvme", "smart-log", XFLASH_DEVICE)


def _number(raw: str) -> float | None:
    try:
        value = float(raw.strip())
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _parse_gpu_telemetry(output: str) -> dict[str, float | None] | None:
    """Reduce nvidia-smi CSV output for all GPUs visible on a node."""
    temperatures: list[float] = []
    powers: list[float] = []
    instant_powers: list[float] = []
    utilizations: list[float] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 4:
            continue
        temperature, power, instant_power, utilization = (_number(field) for field in fields[:4])
        if temperature is not None:
            temperatures.append(temperature)
        if power is not None:
            powers.append(power)
        if instant_power is not None:
            instant_powers.append(instant_power)
        if utilization is not None:
            utilizations.append(utilization)
    if not temperatures and not powers:
        return None
    return {
        "temperatureC": max(temperatures) if temperatures else None,
        "powerW": sum(powers) if powers else None,
        "powerInstantW": sum(instant_powers) if instant_powers else None,
        "utilizationPct": max(utilizations) if utilizations else None,
    }


def _parse_nvme_temperature(output: str) -> float | None:
    """Return the NVMe SMART composite temperature in Celsius."""
    match = re.search(r"^temperature\s*:\s*(\d+)\s*°C", output, re.MULTILINE)
    return float(match.group(1)) if match else None


def _worker_ssh_prefix() -> list[str]:
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=3",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        "StrictHostKeyChecking=yes",
    ]
    if WORKER_HOST_KEY_ALIAS:
        command.extend(["-o", f"HostKeyAlias={WORKER_HOST_KEY_ALIAS}"])
    if WORKER_IDENTITY_FILE:
        command.extend(["-i", WORKER_IDENTITY_FILE, "-o", "IdentitiesOnly=yes"])
    return command


class HardwareSampler:
    """Caches low-cost GB10 telemetry from both TP nodes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: dict[str, Any] | None = None
        self._last_fetch = 0.0

    @staticmethod
    def _read_node(label: str, command: list[str]) -> tuple[str, dict[str, Any]]:
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=4,
            )
            telemetry = _parse_gpu_telemetry(completed.stdout)
            if telemetry:
                return label, {"available": True, **telemetry}
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
        return label, {"available": False}

    @staticmethod
    def _read_xflash() -> dict[str, Any]:
        try:
            completed = subprocess.run(
                XFLASH_SMART_QUERY,
                check=True,
                capture_output=True,
                text=True,
                timeout=4,
            )
            temperature = _parse_nvme_temperature(completed.stdout)
            if temperature is not None:
                return {"available": True, "temperatureC": temperature}
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
        return {"available": False}

    def _fetch(self) -> dict[str, Any]:
        commands: list[tuple[str, list[str]]] = [
            (HEAD_NODE_LABEL, ["nvidia-smi", *GPU_QUERY]),
        ]
        if WORKER_SSH:
            commands.append(
                (
                    WORKER_NODE_LABEL,
                    [
                        *_worker_ssh_prefix(),
                        WORKER_SSH,
                        "nvidia-smi",
                        *GPU_QUERY,
                    ],
                )
            )

        nodes: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=len(commands) + 1) as pool:
            futures = [pool.submit(self._read_node, label, command) for label, command in commands]
            xflash_future = pool.submit(self._read_xflash)
            for future in futures:
                label, telemetry = future.result()
                nodes[label] = telemetry
            xflash = xflash_future.result()

        reported = [telemetry for telemetry in nodes.values() if telemetry["available"]]
        temperatures = [telemetry["temperatureC"] for telemetry in reported if telemetry.get("temperatureC") is not None]
        powers = [telemetry["powerW"] for telemetry in reported if telemetry.get("powerW") is not None]
        instant_powers = [telemetry["powerInstantW"] for telemetry in reported if telemetry.get("powerInstantW") is not None]
        return {
            "available": bool(reported),
            "reportedNodes": len(reported),
            "expectedNodes": len(commands),
            "sampledAt": datetime.now(UTC).isoformat(),
            "maxTemperatureC": max(temperatures) if temperatures else None,
            "averageTemperatureC": (sum(temperatures) / len(temperatures)) if temperatures else None,
            "combinedGpuPowerW": sum(powers) if powers else None,
            "combinedGpuInstantPowerW": sum(instant_powers) if instant_powers else None,
            "nodes": nodes,
            "xflash": xflash,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            if self._latest and now - self._last_fetch < HARDWARE_CACHE_SECONDS:
                return self._latest
            self._latest = self._fetch()
            self._last_fetch = now
            return self._latest


HARDWARE_SAMPLER = HardwareSampler()


class VersionSampler:
    """Reads the runtime version from vLLM's lightweight /version endpoint."""

    def __init__(self) -> None:
        self._latest: str | None = None
        self._at = 0.0
        self._lock = threading.Lock()

    def snapshot(self) -> str | None:
        with self._lock:
            now = time.monotonic()
            if self._latest and now - self._at < VERSION_CACHE_SECONDS:
                return self._latest
            try:
                request = urllib.request.Request(
                    VERSION_URL,
                    headers={"Accept": "application/json", "User-Agent": "dspark-live-dashboard/1.0"},
                )
                with urllib.request.urlopen(request, timeout=2) as response:
                    payload = json.loads(response.read().decode("utf-8", errors="replace"))
                version = payload.get("version")
                if isinstance(version, str) and version:
                    self._latest = version
                    self._at = now
            except Exception:
                pass
            return self._latest


VERSION_SAMPLER = VersionSampler()

class LoadSampler:
    """Read startup state and weight-ingress diagnostics from container logs."""

    _shards = re.compile(
        r"Loading safetensors checkpoint shards:\s*(\d+)% Completed \| (\d+)/(\d+)"
    )
    _direct_load = re.compile(r"Loading weights took ([0-9]+(?:\.[0-9]+)?) seconds")
    _diagnostic_line = re.compile(r"DSPARK_WEIGHT_LOAD\s+([^\r\n]+)")
    _draft_markers = ("Loading drafter model", "DSpark draft model loaded")

    def __init__(self) -> None:
        self._latest: dict[str, Any] | None = None
        self._weight_load_latest: dict[str, dict[str, Any]] = {}
        self._completed_by_mode: dict[str, dict[str, Any]] = {}
        self._at = 0.0
        self._lock = threading.Lock()

    @staticmethod
    def _fields(raw: str) -> dict[str, str]:
        fields: dict[str, str] = {}
        for item in raw.split():
            if "=" in item:
                key, value = item.split("=", 1)
                fields[key] = value
        return fields

    @staticmethod
    def _rollup_weight_load(diagnostic: dict[str, Any]) -> dict[str, Any]:
        phases = diagnostic.get("phases", [])
        completed = [phase for phase in phases if phase.get("state") == "complete"]
        failed = [phase for phase in phases if phase.get("state") == "failed"]
        active = [phase for phase in phases if phase.get("state") == "start"]
        numeric_ids = sorted(
            int(phase["id"])
            for phase in phases
            if isinstance(phase.get("id"), int)
        )
        phase_sequence_complete = bool(numeric_ids) and numeric_ids == list(
            range(numeric_ids[-1] + 1)
        )
        diagnostic["phaseSequenceComplete"] = phase_sequence_complete
        diagnostic["state"] = (
            "failed"
            if failed
            else "loading"
            if active
            else "complete"
            if phase_sequence_complete
            else "partial"
        )
        diagnostic["phaseCount"] = len(completed)
        elapsed = sum(float(phase.get("elapsedSeconds", 0.0)) for phase in completed)
        diagnostic["elapsedSeconds"] = round(elapsed, 6)
        if diagnostic.get("mode") == "roce_tp":
            source = sum(int(phase.get("sourceBytes", 0)) for phase in completed)
            traffic = sum(int(phase.get("trafficBytes", 0)) for phase in completed)
            diagnostic["sourceBytes"] = source
            diagnostic["trafficBytes"] = traffic
            diagnostic["tensors"] = sum(
                int(phase.get("tensors", 0)) for phase in completed
            )
            diagnostic["batches"] = sum(
                int(phase.get("batches", 0)) for phase in completed
            )
            diagnostic["directBytes"] = sum(
                int(phase.get("directBytes", 0)) for phase in completed
            )
            diagnostic["stagedBytes"] = sum(
                int(phase.get("stagedBytes", 0)) for phase in completed
            )
            diagnostic["maxFrameBytes"] = max(
                (int(phase.get("maxFrameBytes", 0)) for phase in completed),
                default=0,
            )
            diagnostic["maxWriteBytes"] = max(
                (int(phase.get("maxWriteBytes", 0)) for phase in completed),
                default=0,
            )
            diagnostic["releaseCount"] = sum(
                int(phase.get("releaseCount", 0)) for phase in completed
            )
            diagnostic["maxPendingReleaseBytes"] = max(
                (
                    int(phase.get("maxPendingReleaseBytes", 0))
                    for phase in completed
                ),
                default=0,
            )
            diagnostic["releasedReservedBytes"] = sum(
                int(phase.get("releasedReservedBytes", 0))
                for phase in completed
            )
            if source > 0:
                diagnostic["payloadRatio"] = traffic / source
            else:
                diagnostic.pop("payloadRatio", None)
            if elapsed > 0 and traffic > 0:
                diagnostic["throughputBytesPerSecond"] = traffic / elapsed
            else:
                diagnostic.pop("throughputBytesPerSecond", None)
        if failed:
            diagnostic["error"] = failed[-1].get("error", "weight load failed")
        else:
            diagnostic.pop("error", None)
        return diagnostic

    @classmethod
    def _parse_structured_weight_load(cls, text: str) -> dict[str, Any] | None:
        events = [
            cls._fields(match.group(1))
            for match in cls._diagnostic_line.finditer(text)
        ]
        events = [
            event for event in events if event.get("mode") in {"direct", "roce_tp"}
        ]
        if not events:
            return None

        mode = events[-1]["mode"]
        events = [event for event in events if event.get("mode") == mode]
        run_id = events[-1].get("run")
        if run_id:
            events = [event for event in events if event.get("run") == run_id]
        process_id = events[-1].get("pid")
        if process_id:
            events = [event for event in events if event.get("pid") == process_id]

        phases: dict[str, dict[str, Any]] = {}
        rank: int | None = None
        role: str | None = None
        buffer_bytes: int | None = None
        release_watermark_bytes: int | None = None
        protocol: int | None = None
        transport: str | None = None
        for event in events:
            load_id = event.get("id", "0")
            phase = phases.setdefault(
                load_id,
                {
                    "id": int(load_id) if load_id.isdigit() else load_id,
                    "name": event.get("phase", "model"),
                    "state": "start",
                },
            )
            phase["name"] = event.get("phase", phase["name"])
            phase["state"] = event.get("event", phase["state"])
            if event.get("rank", "").isdigit():
                rank = int(event["rank"])
            role = event.get("role", role)
            if event.get("buffer_bytes", "").isdigit():
                buffer_bytes = int(event["buffer_bytes"])
            if event.get("release_watermark_bytes", "").isdigit():
                release_watermark_bytes = int(event["release_watermark_bytes"])
            if event.get("protocol", "").isdigit():
                protocol = int(event["protocol"])
            transport = event.get("transport", transport)
            for source_key, result_key in (
                ("tensors", "tensors"),
                ("batches", "batches"),
                ("source_bytes", "sourceBytes"),
                ("traffic_bytes", "trafficBytes"),
                ("direct_bytes", "directBytes"),
                ("staged_bytes", "stagedBytes"),
                ("max_frame_bytes", "maxFrameBytes"),
                ("max_write_bytes", "maxWriteBytes"),
                ("releases", "releaseCount"),
                ("max_pending_release_bytes", "maxPendingReleaseBytes"),
                ("released_reserved_bytes", "releasedReservedBytes"),
            ):
                if event.get(source_key, "").isdigit():
                    phase[result_key] = int(event[source_key])
            try:
                if "elapsed_s" in event:
                    phase["elapsedSeconds"] = float(event["elapsed_s"])
            except ValueError:
                pass
            if "error_type" in event:
                phase["error"] = event["error_type"]

        diagnostic: dict[str, Any] = {
            "mode": mode,
            "runId": run_id,
            "processId": (
                int(process_id)
                if process_id and process_id.isdigit()
                else process_id
            ),
            "rank": rank,
            "role": role,
            "phases": list(phases.values()),
            "timingComparable": True,
            "timerKind": "synchronized_ram",
        }
        if buffer_bytes is not None:
            diagnostic["bufferBytes"] = buffer_bytes
        if release_watermark_bytes is not None:
            diagnostic["releaseWatermarkBytes"] = release_watermark_bytes
        if protocol is not None:
            diagnostic["protocol"] = protocol
        if transport is not None:
            diagnostic["transport"] = transport
        return cls._rollup_weight_load(diagnostic)

    @classmethod
    def _parse_weight_load(cls, text: str) -> dict[str, Any] | None:
        structured = cls._parse_structured_weight_load(text)
        if structured:
            return structured
        matches = list(cls._direct_load.finditer(text))
        if not matches:
            return None
        phases: list[dict[str, Any]] = []
        for index, match in enumerate(matches):
            after_drafter_start = any(
                text.rfind(marker, 0, match.start()) >= 0
                for marker in cls._draft_markers
            )
            phase_id = 1 if after_drafter_start else index
            phase_name = (
                "target"
                if phase_id == 0
                else "drafter"
                if phase_id == 1
                else f"model-{phase_id + 1}"
            )
            phases.append(
                {
                    "id": phase_id,
                    "name": phase_name,
                    "state": "complete",
                    "elapsedSeconds": float(match.group(1)),
                }
            )
        diagnostic = {
            "mode": "direct",
            "role": "local_reader",
            "phases": phases,
            "timingComparable": False,
            "timerKind": "vllm_reported",
        }
        return cls._rollup_weight_load(diagnostic)

    @classmethod
    def _merge_weight_load(
        cls, previous: dict[str, Any], current: dict[str, Any]
    ) -> dict[str, Any]:
        if previous.get("mode") != current.get("mode"):
            return current
        if current.get("runId") or previous.get("runId"):
            if previous.get("runId") != current.get("runId"):
                return current
            if previous.get("processId") != current.get("processId"):
                return current
        elif current.get("mode") == "roce_tp" and previous.get(
            "processId"
        ) != current.get("processId"):
            return current

        current_ids = {str(phase.get("id")) for phase in current.get("phases", [])}
        if (
            current.get("mode") == "direct"
            and current.get("timingComparable") is False
            and current_ids == {"0"}
            and int(previous.get("phaseCount", 0)) > 1
        ):
            return current

        phases: dict[str, dict[str, Any]] = {}
        for diagnostic in (previous, current):
            for phase in diagnostic.get("phases", []):
                phases[str(phase.get("id"))] = dict(phase)
        merged = dict(current)
        merged["phases"] = list(phases.values())
        for key in (
            "bufferBytes",
            "releaseWatermarkBytes",
            "runId",
            "processId",
        ):
            if merged.get(key) is None and previous.get(key) is not None:
                merged[key] = previous[key]
        return cls._rollup_weight_load(merged)

    def _parse(self, text: str) -> dict[str, Any]:
        shard_matches = list(self._shards.finditer(text))
        weight_load = self._parse_weight_load(text)
        result: dict[str, Any]
        if "SafetensorError" in text or "Engine core initialization failed" in text:
            result = {"state": "failed", "detail": text.splitlines()[-1]}
        elif "Uvicorn running" in text or "Application startup complete" in text:
            result = {"state": "ready"}
        elif "Kernel JIT monitor activated" in text or "Graph capturing finished" in text:
            result = {"state": "active_tp_rank_1"}
        elif text.count("Loading weights took") >= 2:
            result = {"state": "initializing"}
        elif any(marker in text for marker in self._draft_markers):
            result = {"state": "drafter"}
        elif "Loading weights took" in text:
            result = {"state": "target_loaded"}
        elif shard_matches:
            pct, done, total = shard_matches[-1].groups()
            result = {
                "state": "target_weights",
                "percent": int(pct),
                "done": int(done),
                "total": int(total),
            }
        elif "Filesystem type for checkpoints" in text:
            result = {"state": "target_weights"}
        elif "Starting to load model" in text:
            result = {"state": "initializing"}
        else:
            result = {"state": "waiting"}
        if weight_load:
            result["weightLoad"] = weight_load
        elif result["state"] == "failed":
            result["weightLoad"] = {
                "mode": "unknown",
                "state": "failed",
                "error": "engine initialization failed",
                "timingComparable": False,
            }
        return result

    @staticmethod
    def _mark_ready(nodes: dict[str, Any]) -> dict[str, Any]:
        ready: dict[str, Any] = {}
        for label, rank in ((HEAD_NODE_LABEL, 0), (WORKER_NODE_LABEL, 1)):
            record = dict(nodes.get(label, {}))
            record.update({"state": "ready", "detail": f"TP rank {rank} · serving"})
            ready[label] = record
        return ready

    @staticmethod
    def _summarize(nodes: dict[str, Any]) -> dict[str, Any]:
        diagnostics = [
            (label, node["weightLoad"])
            for label, node in nodes.items()
            if isinstance(node.get("weightLoad"), dict)
        ]
        if not diagnostics:
            return {"mode": "unknown", "state": "waiting", "nodes": {}}

        modes = {item.get("mode", "unknown") for _, item in diagnostics}
        if len(modes) != 1:
            return {
                "mode": "mixed",
                "state": "inconsistent",
                "nodes": {
                    label: {
                        "mode": item.get("mode"),
                        "rank": item.get("rank"),
                        "state": item.get("state"),
                    }
                    for label, item in diagnostics
                },
            }

        mode = modes.pop()
        states = {item.get("state") for _, item in diagnostics}
        ranks = {item.get("rank") for _, item in diagnostics}
        state = (
            "failed"
            if "failed" in states
            else "loading"
            if "loading" in states
            else "partial"
            if "partial" in states or len(diagnostics) < 2 or ranks != {0, 1}
            else "complete"
        )
        keys = (
            "rank",
            "runId",
            "role",
            "protocol",
            "transport",
            "bufferBytes",
            "state",
            "phaseCount",
            "elapsedSeconds",
            "sourceBytes",
            "trafficBytes",
            "directBytes",
            "stagedBytes",
            "maxFrameBytes",
            "maxWriteBytes",
            "tensors",
            "batches",
            "throughputBytesPerSecond",
            "payloadRatio",
            "timingComparable",
            "timerKind",
            "phaseSequenceComplete",
            "error",
        )
        node_summary = {
            label: {key: item[key] for key in keys if item.get(key) is not None}
            for label, item in diagnostics
        }
        elapsed_values = [
            float(item["elapsedSeconds"])
            for _, item in diagnostics
            if item.get("elapsedSeconds") is not None
        ]
        phase_elapsed: dict[str, list[float]] = {}
        for _, item in diagnostics:
            for phase in item.get("phases", []):
                if (
                    phase.get("state") == "complete"
                    and isinstance(phase.get("elapsedSeconds"), (int, float))
                ):
                    phase_elapsed.setdefault(str(phase.get("id")), []).append(
                        float(phase["elapsedSeconds"])
                    )
        critical_elapsed = (
            sum(max(values) for values in phase_elapsed.values())
            if phase_elapsed
            else max(elapsed_values)
            if elapsed_values
            else None
        )
        result: dict[str, Any] = {
            "mode": mode,
            "state": state,
            "nodes": node_summary,
            "criticalElapsedSeconds": critical_elapsed,
            "phaseCount": max(
                (int(item.get("phaseCount", 0)) for _, item in diagnostics),
                default=0,
            ),
            "timingComparable": all(
                item.get("timingComparable") is True for _, item in diagnostics
            ),
        }
        complete = [
            item for _, item in diagnostics if item.get("state") == "complete"
        ]
        if len(complete) == 2:
            phase_maps = [
                {str(phase.get("id")): phase for phase in item.get("phases", [])}
                for item in complete
            ]
            if phase_maps[0] or phase_maps[1]:
                ranks_agree = phase_maps[0].keys() == phase_maps[1].keys()
                for phase_id in phase_maps[0].keys() & phase_maps[1].keys():
                    left = phase_maps[0][phase_id]
                    right = phase_maps[1][phase_id]
                    ranks_agree = ranks_agree and (
                        left.get("name") == right.get("name")
                    )
                    if mode == "roce_tp":
                        ranks_agree = ranks_agree and all(
                            left.get(key) == right.get(key)
                            for key in (
                                "sourceBytes",
                                "trafficBytes",
                                "directBytes",
                                "stagedBytes",
                                "maxFrameBytes",
                                "maxWriteBytes",
                                "tensors",
                                "batches",
                            )
                        )
                if mode == "roce_tp":
                    ranks_agree = ranks_agree and all(
                        complete[0].get(key) == complete[1].get(key)
                        for key in ("protocol", "transport", "bufferBytes")
                    )
                result["ranksAgree"] = ranks_agree
            else:
                agreement_keys = ["phaseCount"]
                if mode == "roce_tp":
                    agreement_keys.extend(
                        [
                            "sourceBytes",
                            "trafficBytes",
                            "directBytes",
                            "stagedBytes",
                            "maxFrameBytes",
                            "maxWriteBytes",
                            "protocol",
                            "transport",
                            "bufferBytes",
                            "tensors",
                            "batches",
                        ]
                    )
                result["ranksAgree"] = all(
                    complete[0].get(key) == complete[1].get(key)
                    for key in agreement_keys
                )
        if mode == "roce_tp":
            receiver = next(
                (item for _, item in diagnostics if item.get("role") == "receiver"),
                diagnostics[-1][1],
            )
            for key in (
                "sourceBytes",
                "trafficBytes",
                "directBytes",
                "stagedBytes",
                "maxFrameBytes",
                "maxWriteBytes",
                "protocol",
                "transport",
                "bufferBytes",
                "tensors",
                "batches",
                "throughputBytesPerSecond",
                "payloadRatio",
            ):
                if receiver.get(key) is not None:
                    result[key] = receiver[key]
            reader = next(
                (item for _, item in diagnostics if item.get("role") == "reader"),
                diagnostics[0][1],
            )
            for key in (
                "releaseCount",
                "releaseWatermarkBytes",
                "maxPendingReleaseBytes",
                "releasedReservedBytes",
            ):
                if reader.get(key) is not None:
                    result[key] = reader[key]
            traffic = result.get("trafficBytes")
            critical = result.get("criticalElapsedSeconds")
            if (
                isinstance(traffic, (int, float))
                and isinstance(critical, (int, float))
                and traffic > 0
                and critical > 0
            ):
                result["throughputBytesPerSecond"] = traffic / critical
        errors = [item.get("error") for _, item in diagnostics if item.get("error")]
        if errors:
            result["error"] = errors[-1]
        return result

    def weight_summary(
        self, nodes: dict[str, Any], *, api_ready: bool = False
    ) -> dict[str, Any]:
        with self._lock:
            current = self._summarize(nodes)
            mode = current.get("mode")
            if (
                api_ready
                and mode in {"direct", "roce_tp"}
                and current.get("state") == "complete"
                and current.get("timingComparable") is True
                and current.get("ranksAgree") is True
            ):
                self._completed_by_mode[str(mode)] = {
                    key: value
                    for key, value in current.items()
                    if key not in {"nodes", "observed"}
                }
            current["observed"] = dict(self._completed_by_mode)
            direct = self._completed_by_mode.get("direct")
            roce = self._completed_by_mode.get("roce_tp")
            if direct and roce:
                direct_time = direct.get("criticalElapsedSeconds")
                roce_time = roce.get("criticalElapsedSeconds")
                if (
                    isinstance(direct_time, (int, float))
                    and isinstance(roce_time, (int, float))
                    and roce_time > 0
                ):
                    current["directVsRoceSpeedup"] = direct_time / roce_time
                    current["directVsRoceSavedSeconds"] = direct_time - roce_time
            return current

    def snapshot(self, api_ready: bool = False) -> dict[str, Any]:
        with self._lock:
            if self._latest and time.monotonic() - self._at < LOAD_CACHE_SECONDS:
                if not api_ready or all(
                    node.get("state") == "ready" for node in self._latest.values()
                ):
                    return self._latest
            local = [
                "sudo",
                "-n",
                "/usr/bin/docker",
                "logs",
                "--tail",
                str(LOAD_LOG_TAIL),
                CONTAINER_NAME,
            ]
            commands = {HEAD_NODE_LABEL: local}
            if WORKER_SSH:
                commands[WORKER_NODE_LABEL] = [
                    *_worker_ssh_prefix(),
                    WORKER_SSH,
                    *local,
                ]

            def read(cmd: list[str]) -> tuple[bool, str]:
                try:
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=5
                    )
                    return result.returncode == 0, result.stdout + result.stderr
                except Exception:
                    return False, ""

            with ThreadPoolExecutor(max_workers=len(commands)) as pool:
                futures = {
                    label: pool.submit(read, command)
                    for label, command in commands.items()
                }
                outputs = {
                    label: future.result() for label, future in futures.items()
                }
            parsed = {
                label: self._parse(text)
                if successful
                else {"state": "unavailable", "detail": "container log unavailable"}
                for label, (successful, text) in outputs.items()
            }
            parsed.setdefault(WORKER_NODE_LABEL, {"state": "unavailable"})
            for label, rank in ((HEAD_NODE_LABEL, 0), (WORKER_NODE_LABEL, 1)):
                diagnostic = parsed[label].get("weightLoad")
                if isinstance(diagnostic, dict):
                    diagnostic.setdefault("rank", rank)
                    if label in self._weight_load_latest:
                        diagnostic = self._merge_weight_load(
                            self._weight_load_latest[label], diagnostic
                        )
                        parsed[label]["weightLoad"] = diagnostic
                    self._weight_load_latest[label] = diagnostic
                elif parsed[label].get("state") in {
                    "initializing",
                    "target_weights",
                    "drafter",
                }:
                    self._weight_load_latest.pop(label, None)
                elif (
                    outputs.get(label, (False, ""))[0]
                    and label in self._weight_load_latest
                ):
                    parsed[label]["weightLoad"] = self._weight_load_latest[label]

            # API readiness implies both TP ranks serve, but it must not erase
            # startup diagnostics collected from the headless worker.
            self._latest = self._mark_ready(parsed) if api_ready else parsed
            self._at = time.monotonic()
            return self._latest

LOAD_SAMPLER = LoadSampler()


class MetricsSampler:
    """Fetches and reduces Prometheus counters into a compact live snapshot."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._previous: tuple[dict[str, Any], float] | None = None
        self._latest: dict[str, Any] | None = None
        self._last_fetch = 0.0
        self._history: deque[dict[str, Any]] = deque(maxlen=121)

    @staticmethod
    def _rate(current: float, previous: float, elapsed: float) -> tuple[float | None, bool]:
        if elapsed <= 0:
            return None, False
        delta = current - previous
        if delta < 0:
            return 0.0, True
        return delta / elapsed, False

    @staticmethod
    def _recent_mean(current_sum: float, current_count: float, previous_sum: float, previous_count: float) -> float | None:
        count_delta = current_count - previous_count
        sum_delta = current_sum - previous_sum
        if count_delta <= 0 or sum_delta < 0:
            return None
        return sum_delta / count_delta

    def _fetch(self) -> tuple[dict[str, Any], float]:
        started = time.monotonic()
        request = urllib.request.Request(
            METRICS_URL,
            headers={"Accept": "text/plain", "User-Agent": "dspark-live-dashboard/1.0"},
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            payload = response.read().decode("utf-8", errors="replace")
        return parse_prometheus(payload), (time.monotonic() - started) * 1000.0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            if self._latest and now - self._last_fetch < POLL_CACHE_SECONDS:
                return self._latest

            try:
                current, scrape_ms = self._fetch()
            except Exception:
                load = LOAD_SAMPLER.snapshot(api_ready=False)
                unavailable = dict(self._latest or {})
                unavailable.update(
                    {
                        "healthy": False,
                        "message": "vLLM metrics are temporarily unavailable",
                        "sampledAt": datetime.now(UTC).isoformat(),
                        "history": list(self._history),
                        "headNodeLabel": HEAD_NODE_LABEL,
                        "workerNodeLabel": WORKER_NODE_LABEL,
                        "load": load,
                        "weightLoad": LOAD_SAMPLER.weight_summary(
                            load, api_ready=False
                        ),
                        "vllmVersion": VERSION_SAMPLER.snapshot(),
                    }
                )
                self._latest = unavailable
                self._last_fetch = now
                return unavailable

            previous = self._previous
            generated_tps: float | None = None
            prefill_tps: float | None = None
            dspark_acceptance: float | None = None
            ttft_seconds: float | None = None
            itl_seconds: float | None = None
            e2e_seconds: float | None = None
            counter_reset = False

            if previous:
                old, old_at = previous
                elapsed = now - old_at
                generated_tps, reset = self._rate(
                    current["generated_tokens"], old["generated_tokens"], elapsed
                )
                counter_reset = counter_reset or reset
                prefill_tps, reset = self._rate(
                    current["prompt_tokens"], old["prompt_tokens"], elapsed
                )
                counter_reset = counter_reset or reset
                accepted_delta = current["accepted_tokens"] - old["accepted_tokens"]
                draft_delta = current["draft_tokens"] - old["draft_tokens"]
                if accepted_delta >= 0 and draft_delta > 0:
                    dspark_acceptance = (accepted_delta / draft_delta) * 100.0
                ttft_seconds = self._recent_mean(
                    current["ttft_sum"], current["ttft_count"], old["ttft_sum"], old["ttft_count"]
                )
                itl_seconds = self._recent_mean(
                    current["itl_sum"], current["itl_count"], old["itl_sum"], old["itl_count"]
                )
                e2e_seconds = self._recent_mean(
                    current["e2e_sum"], current["e2e_count"], old["e2e_sum"], old["e2e_count"]
                )

            if counter_reset:
                self._history.clear()

            point = {
                "time": int(time.time() * 1000),
                "generationTps": generated_tps,
                "prefillTps": prefill_tps,
                "running": current["running"],
                "waiting": current["waiting"],
                "kvCachePct": current["kv_cache_pct"],
            }
            self._history.append(point)
            hardware = HARDWARE_SAMPLER.snapshot()
            load = LOAD_SAMPLER.snapshot(api_ready=True)
            snapshot = {
                "healthy": True,
                "message": "live",
                "sampledAt": datetime.now(UTC).isoformat(),
                "scrapeMs": round(scrape_ms, 1),
                "warmup": previous is None,
                "counterReset": counter_reset,
                "model": current["model"],
                "vllmVersion": VERSION_SAMPLER.snapshot(),
                "headNodeLabel": HEAD_NODE_LABEL,
                "workerNodeLabel": WORKER_NODE_LABEL,
                "generationTps": generated_tps,
                "prefillTps": prefill_tps,
                "dsparkAcceptancePct": dspark_acceptance,
                "running": current["running"],
                "waiting": current["waiting"],
                "waitingCapacity": current["waiting_capacity"],
                "waitingDeferred": current["waiting_deferred"],
                "kvCachePct": current["kv_cache_pct"],
                "generatedTokens": current["generated_tokens"],
                "promptTokens": current["prompt_tokens"],
                "completedRequests": current["completed_requests"],
                "errors": current["errors"],
                "ttftMs": None if ttft_seconds is None else ttft_seconds * 1000.0,
                "itlMs": None if itl_seconds is None else itl_seconds * 1000.0,
                "e2eMs": None if e2e_seconds is None else e2e_seconds * 1000.0,
                "hardware": hardware,
                "load": load,
                "weightLoad": LOAD_SAMPLER.weight_summary(load, api_ready=True),
                "history": list(self._history),
            }
            self._previous = (current, now)
            self._latest = snapshot
            self._last_fetch = now
            return snapshot


SAMPLER = MetricsSampler()


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(APP_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        if not self.path.startswith("/api/"):
            super().log_message(format, *args)

    def _json(self, value: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/snapshot":
            self._json(SAMPLER.snapshot())
            return
        if path == "/health":
            snapshot = SAMPLER.snapshot()
            self._json({"ok": bool(snapshot.get("healthy")), "message": snapshot.get("message")})
            return
        if path == "/":
            self.path = "/index.html"
        super().do_GET()


def main() -> None:
    server = ThreadingHTTPServer((BIND_ADDRESS, PORT), DashboardHandler)
    server.daemon_threads = True
    print(f"DSpark live dashboard listening on http://{BIND_ADDRESS}:{PORT}", flush=True)
    server.serve_forever(poll_interval=0.5)


if __name__ == "__main__":
    main()
