#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""No-model cold-I/O probe for the prepared NVFP4 rank-range reader."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def _mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def _raw_sha256(tensor: Any, torch_module: Any) -> str:
    raw = tensor.contiguous().view(torch_module.uint8).reshape(-1)
    return hashlib.sha256(memoryview(raw.numpy())).hexdigest()


def _evict_rank_ranges(fd: int, ranges: dict[str, Any]) -> bool:
    advise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advise is None or dontneed is None:
        return False
    for item in ranges.values():
        advise(fd, item.offset, item.nbytes, dontneed)
    return True


def run_probe(
    checkpoint: Path,
    *,
    manifest_sha256: str,
    layer: int,
    tp_rank: int,
    mode: str,
    minimum_mem_available_gib: float,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from vllm.models.deepseek_v4.nvidia import prepared_weight_loading as helper

    before_mem = _mem_available_bytes()
    minimum_bytes = int(minimum_mem_available_gib * (1 << 30))
    if before_mem < minimum_bytes:
        raise RuntimeError(
            f"MemAvailable is below the probe floor: {before_mem} < {minimum_bytes}"
        )
    contract = helper.inspect_prepared_checkpoint(
        checkpoint,
        environ={
            helper.PREPARED_LOAD_ENV: "1",
            helper.PREPARED_MANIFEST_SHA256_ENV: manifest_sha256,
        },
    )
    if contract is None:
        raise RuntimeError("Prepared checkpoint contract was not selected")
    path = contract.checkpoint / contract.layer_files[layer]
    before_stat = path.stat()
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        ranges = helper._parse_prepared_rank_ranges(
            fd,
            path=path,
            layer=layer,
            tp_rank=tp_rank,
        )
        evicted = _evict_rank_ranges(fd, ranges)
    finally:
        os.close(fd)

    prefix = f"{helper.PREPARED_NAMESPACE}.layers.{layer}.experts."
    shapes = helper._destination_shapes()
    dtypes = helper._family_dtypes(torch)
    family_rows = []
    if mode == "direct":
        reader = helper.PreparedSafetensorsDirectReader(
            torch_module=torch,
            contract=contract,
            tp_rank=tp_rank,
        )
        with safe_open(path, framework="pt", device="cpu") as handle:
            for family in helper.PREPARED_FAMILY_ORDER:
                destination = torch.empty(shapes[family], dtype=dtypes[family])
                reader.copy_into(
                    layer=layer,
                    family=family,
                    destination=destination,
                )
                reference = handle.get_tensor(f"{prefix}{family}")[tp_rank]
                bitwise = torch.equal(
                    destination.view(torch.uint8),
                    reference.view(torch.uint8),
                )
                if not bitwise:
                    raise RuntimeError(
                        f"Direct rank-range bytes differ for layer={layer} {family}"
                    )
                row = dict(reader.layer_stats(layer))
                family_rows.append(
                    {
                        "family": family,
                        "bytes": int(ranges[family].nbytes),
                        "destination_sha256": _raw_sha256(destination, torch),
                        "bitwise_match": True,
                        "cumulative_read_seconds": float(row["read_seconds"]),
                        "cumulative_copy_seconds": float(row["copy_seconds"]),
                    }
                )
                del reference, destination
        summary = dict(reader.summary())
        reader.finish()
        elapsed_seconds = float(summary["read_seconds"]) + float(
            summary["copy_seconds"]
        )
    elif mode == "mmap":
        started = time.perf_counter()
        with safe_open(path, framework="pt", device="cpu") as handle:
            for family in helper.PREPARED_FAMILY_ORDER:
                destination = torch.empty(shapes[family], dtype=dtypes[family])
                family_started = time.perf_counter()
                reference = handle.get_tensor(f"{prefix}{family}")[tp_rank]
                destination.copy_(reference)
                family_seconds = time.perf_counter() - family_started
                family_rows.append(
                    {
                        "family": family,
                        "bytes": int(ranges[family].nbytes),
                        "destination_sha256": _raw_sha256(destination, torch),
                        "copy_seconds": family_seconds,
                    }
                )
                del reference, destination
        elapsed_seconds = time.perf_counter() - started
        summary = {
            "ranges": len(ranges),
            "syscalls": None,
            "bytes": sum(item.nbytes for item in ranges.values()),
            "read_seconds": None,
            "copy_seconds": elapsed_seconds,
        }
    else:
        raise ValueError(f"Unknown probe mode: {mode}")

    after_stat = path.stat()
    if (
        before_stat.st_dev,
        before_stat.st_ino,
        before_stat.st_size,
        before_stat.st_mtime_ns,
    ) != (
        after_stat.st_dev,
        after_stat.st_ino,
        after_stat.st_size,
        after_stat.st_mtime_ns,
    ):
        raise RuntimeError("Prepared layer file changed while the probe ran")
    total_bytes = int(summary["bytes"])
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "probe": "deepseek_v4_nvfp4_prepared_rank_io",
        "model_loaded": False,
        "gpu_required": False,
        "mode": mode,
        "layer": layer,
        "tp_rank": tp_rank,
        "checkpoint": str(contract.checkpoint),
        "layer_file": str(path),
        "manifest_sha256": contract.manifest_sha256,
        "cache_evict_advised": evicted,
        "elapsed_seconds": elapsed_seconds,
        "gib_per_second": total_bytes / (1 << 30) / elapsed_seconds,
        "summary": summary,
        "families": family_rows,
        "memory": {
            "before_available_bytes": before_mem,
            "after_available_bytes": _mem_available_bytes(),
            "minimum_available_bytes": minimum_bytes,
        },
        "file_unchanged": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tp-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--mode", choices=("direct", "mmap"), required=True)
    parser.add_argument("--minimum-mem-available-gib", type=float, default=4.0)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_probe(
        args.checkpoint,
        manifest_sha256=args.manifest_sha256,
        layer=args.layer,
        tp_rank=args.tp_rank,
        mode=args.mode,
        minimum_mem_available_gib=args.minimum_mem_available_gib,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, args.output)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
