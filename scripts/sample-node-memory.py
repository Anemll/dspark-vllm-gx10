#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded, read-only Linux memory evidence; JSONL on stdout, no GPU imports.

Reports interval counters, never vmstat's since-boot first-line averages.
Does not kill processes, alter caches/sysctls, or decide experimental gates.
Keep stdout as a private artifact: requested process/cgroup paths are recorded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time


VM_COUNTERS = ("pswpin", "pswpout", "pgmajfault", "pgscan_direct", "pgsteal_direct", "oom_kill")
MEM_FIELDS = ("MemTotal", "MemAvailable", "MemFree", "Cached", "SReclaimable",
              "AnonPages", "Mapped", "Unevictable", "Mlocked", "Dirty", "Writeback",
              "SwapTotal", "SwapFree", "SwapCached")
PROCESS_FIELDS = ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmSwap", "VmLck")


def key_values(raw: str) -> dict[str, int]:
    """Parse procfs integer fields; colon and kB suffix are optional."""
    result = {}
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            result[parts[0].removesuffix(":")] = int(parts[1])
    return result


def pressure(raw: str) -> dict:
    result = {}
    for line in raw.splitlines():
        parts = line.split()
        if parts and parts[0] in ("some", "full"):
            values = dict(item.split("=", 1) for item in parts[1:])
            result[parts[0]] = {
                "total_us": int(values["total"]),
                **{key: float(values[key]) for key in ("avg10", "avg60", "avg300")},
            }
    return result


def snapshot(*, proc_root=Path("/proc"), disk_path=Path("."), pids=(), cgroup_dir=None):
    errors = []

    def read(path, parser):
        try:
            return parser(path.read_text())
        except (OSError, ValueError, KeyError) as error:
            errors.append({"path": str(path), "error": type(error).__name__, "message": str(error)})
            return None

    begun = time.monotonic()
    mem = read(proc_root / "meminfo", key_values)
    vm = read(proc_root / "vmstat", key_values)
    psi = read(proc_root / "pressure/memory", pressure)
    processes = {}
    for pid in pids:
        values = read(proc_root / str(pid) / "status", key_values)
        processes[str(pid)] = None if values is None else {key: values.get(key) for key in PROCESS_FIELDS}
    disk_free = None
    try:
        disk = os.statvfs(disk_path)
        disk_free = disk.f_bavail * disk.f_frsize
    except OSError as error:
        errors.append({"path": str(disk_path), "error": type(error).__name__, "message": str(error)})
    cgroup = None
    if cgroup_dir is not None:
        cgroup = {"path": str(cgroup_dir),
                  "events": read(cgroup_dir / "memory.events", key_values),
                  "current_bytes": read(cgroup_dir / "memory.current", lambda raw: int(raw.strip())),
                  "swap_current_bytes": read(cgroup_dir / "memory.swap.current", lambda raw: int(raw.strip()))}
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "monotonic_s": begun,
        "collection_s": time.monotonic() - begun,
        "boot_id": read(proc_root / "sys/kernel/random/boot_id", str.strip),
        "page_size_bytes": os.sysconf("SC_PAGE_SIZE"),
        "mem_kib": None if mem is None else {key: mem.get(key) for key in MEM_FIELDS},
        "vm_pages_or_events": None if vm is None else {key: vm.get(key) for key in VM_COUNTERS},
        "memory_psi": psi,
        "process_kib": processes,
        "cgroup": cgroup,
        "disk_free_bytes": disk_free,
        "errors": errors,
    }


def interval(previous: dict | None, current: dict) -> dict:
    """Missing/reset counters stay unknown; they never become zero or a rate."""
    if previous is None:
        return {"valid": False, "reason": "first_sample", "seconds": None}
    elapsed = current["monotonic_s"] - previous["monotonic_s"]
    if not math.isfinite(elapsed) or elapsed <= 0:
        return {"valid": False, "reason": "invalid_time", "seconds": elapsed if math.isfinite(elapsed) else None}
    if not current.get("boot_id") or current["boot_id"] != previous.get("boot_id"):
        return {"valid": False, "reason": "boot_unknown_or_changed", "seconds": elapsed}
    if current.get("page_size_bytes") != previous.get("page_size_bytes"):
        return {"valid": False, "reason": "page_size_changed", "seconds": elapsed}
    errors = []

    def delta(old, new, label):
        if old is None or new is None:
            errors.append(label + ":missing")
            return None
        if new < old:
            errors.append(label + ":reset")
            return None
        return new - old

    old_vm = previous.get("vm_pages_or_events") or {}
    new_vm = current.get("vm_pages_or_events") or {}
    counters = {key: delta(old_vm.get(key), new_vm.get(key), key) for key in VM_COUNTERS}
    old_mem = previous.get("mem_kib") or {}
    new_mem = current.get("mem_kib") or {}
    swap_used_delta = None
    if all(item is not None for item in (old_mem.get("SwapTotal"), old_mem.get("SwapFree"),
                                        new_mem.get("SwapTotal"), new_mem.get("SwapFree"))):
        swap_used_delta = (new_mem["SwapTotal"] - new_mem["SwapFree"]) - (old_mem["SwapTotal"] - old_mem["SwapFree"])
    psi_pct = {}
    for kind in ("some", "full"):
        old = ((previous.get("memory_psi") or {}).get(kind) or {}).get("total_us")
        new = ((current.get("memory_psi") or {}).get(kind) or {}).get("total_us")
        diff = delta(old, new, "psi_" + kind)
        psi_pct[kind] = None if diff is None else 100 * diff / (elapsed * 1_000_000)
    cgroup_delta = None
    old_cgroup, new_cgroup = previous.get("cgroup"), current.get("cgroup")
    if old_cgroup is not None or new_cgroup is not None:
        if old_cgroup is None or new_cgroup is None or old_cgroup["path"] != new_cgroup["path"]:
            errors.append("cgroup:changed")
        else:
            old_events, new_events = old_cgroup.get("events") or {}, new_cgroup.get("events") or {}
            cgroup_delta = {key: delta(old_events.get(key), new_events.get(key), "cgroup_" + key)
                            for key in ("high", "max", "oom", "oom_kill")}
    return {
        "valid": not errors,
        "seconds": elapsed,
        "swap_in_bytes_s": None if counters["pswpin"] is None else counters["pswpin"] * current["page_size_bytes"] / elapsed,
        "swap_out_bytes_s": None if counters["pswpout"] is None else counters["pswpout"] * current["page_size_bytes"] / elapsed,
        "swap_used_delta_kib": swap_used_delta,
        "counter_deltas": counters,
        "memory_stall_percent": psi_pct,
        "cgroup_event_deltas": cgroup_delta,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--disk-path", type=Path, default=Path("."))
    parser.add_argument("--pid", type=int, action="append", default=[])
    parser.add_argument("--cgroup-dir", type=Path)
    args = parser.parse_args()
    if not 1 <= args.samples <= 3600 or not math.isfinite(args.interval) or not 0.1 <= args.interval <= 60:
        parser.error("require 1–3600 samples and interval 0.1–60 seconds")
    if (args.samples - 1) * args.interval > 3600 or any(pid <= 0 for pid in args.pid):
        parser.error("sample duration must be <=3600 seconds and PIDs positive")
    previous = None
    start = time.monotonic()
    for index in range(args.samples):
        if index:
            time.sleep(max(0, start + index * args.interval - time.monotonic()))
        current = snapshot(disk_path=args.disk_path, pids=args.pid, cgroup_dir=args.cgroup_dir)
        record = {"sample": index, "snapshot": current, "interval": interval(previous, current)}
        print(json.dumps(record, allow_nan=False), flush=True)
        previous = current


if __name__ == "__main__":
    main()
