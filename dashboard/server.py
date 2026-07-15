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
    """Reads vLLM startup state from each container's recent log lines."""
    _shards = re.compile(r"Loading safetensors checkpoint shards:\s*(\d+)% Completed \| (\d+)/(\d+)")
    def __init__(self) -> None:
        self._latest: dict[str, Any] | None = None; self._at = 0.0; self._lock = threading.Lock()
    def _parse(self, text: str) -> dict[str, Any]:
        match = list(self._shards.finditer(text))
        if "SafetensorError" in text or "Engine core initialization failed" in text:
            return {"state": "failed", "detail": text.splitlines()[-1]}
        if "Uvicorn running" in text or "Application startup complete" in text:
            return {"state": "ready"}
        if "Kernel JIT monitor activated" in text or "Graph capturing finished" in text:
            return {"state": "active_tp_rank_1"}
        if text.count("Loading weights took") >= 2:
            return {"state": "initializing"}
        if "Loading drafter model" in text:
            return {"state": "drafter"}
        if "Loading weights took" in text:
            return {"state": "target_loaded"}
        if match:
            pct, done, total = match[-1].groups()
            return {"state": "target_weights", "percent": int(pct), "done": int(done), "total": int(total)}
        if "Filesystem type for checkpoints" in text:
            return {"state": "target_weights"}
        if "Starting to load model" in text: return {"state": "initializing"}
        return {"state": "waiting"}
    def snapshot(self, api_ready: bool = False) -> dict[str, Any]:
        with self._lock:
            if self._latest and time.monotonic() - self._at < LOAD_CACHE_SECONDS:
                if not api_ready or all(node.get("state") == "ready" for node in self._latest.values()):
                    return self._latest
            local = ["sudo", "-n", "/usr/bin/docker", "logs", "--tail", "160", CONTAINER_NAME]
            commands = {HEAD_NODE_LABEL: local}
            if WORKER_SSH:
                commands[WORKER_NODE_LABEL] = [*_worker_ssh_prefix(), WORKER_SSH, *local]
            def read(cmd: list[str]) -> str:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    return result.stdout + result.stderr
                except Exception: return ""
            with ThreadPoolExecutor(max_workers=len(commands)) as pool:
                futures = {label: pool.submit(read, command) for label, command in commands.items()}
                self._latest = {label: self._parse(future.result()) for label, future in futures.items()}
            self._latest.setdefault(WORKER_NODE_LABEL, {"state": "unavailable"})
            # A live metrics endpoint means EngineCore is serving with both TP
            # ranks. Headless rank 1 never emits the API server's final ready
            # line, and old ready lines eventually roll out of the log tail.
            if api_ready:
                self._latest = {
                    HEAD_NODE_LABEL: {"state": "ready", "detail": "TP rank 0 · serving"},
                    WORKER_NODE_LABEL: {"state": "ready", "detail": "TP rank 1 · serving"},
                }
            self._at = time.monotonic(); return self._latest

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
                unavailable = dict(self._latest or {})
                unavailable.update(
                    {
                        "healthy": False,
                        "message": "vLLM metrics are temporarily unavailable",
                        "sampledAt": datetime.now(UTC).isoformat(),
                        "history": list(self._history),
                        "load": LOAD_SAMPLER.snapshot(api_ready=False),
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
