#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read runtime provenance without importing vLLM or loading model weights.

Source declarations and requested settings are deliberately separate from
observed log evidence. A Python dispatch entry does not prove a CUDA binary
is present or that a live request selected it.
"""

from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


PACKAGES = (
    "vllm", "flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache",
    "b12x", "torch", "triton", "nvidia-cutlass-dsl", "apache-tvm-ffi",
    "cuda-python", "transformers",
)
SERVING_FLAGS = frozenset({
    "--served-model-name", "--model", "--tensor-parallel-size",
    "--pipeline-parallel-size", "--kv-cache-dtype", "--block-size",
    "--max-model-len", "--max-num-seqs", "--max-num-batched-tokens",
    "--max-cudagraph-capture-size", "--gpu-memory-utilization",
    "--speculative-config", "--compilation-config", "--attention-backend",
    "--attention-config", "--moe-backend", "--tokenizer-mode",
    "--tool-call-parser", "--reasoning-parser", "--jit-monitor-mode",
    "--enable-prefix-caching", "--no-enable-prefix-caching",
    "--enable-chunked-prefill", "--async-scheduling", "--enforce-eager",
    "--nnodes", "--node-rank",
})
ENV_NAMES = (
    "VLLM_USE_B12X_MOE", "VLLM_USE_B12X_WO_PROJECTION",
    "VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM", "VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M",
    "VLLM_B12X_W4A16_FORCE_TILE_CONFIG", "VLLM_DSPARK_CONFIDENCE_THRESHOLD",
    "VLLM_DSPARK_CONFIDENCE_SCHEDULER", "VLLM_DSPARK_LOCAL_ARGMAX",
    "VLLM_DSPARK_SPARSE_DECODE_TOPKS", "FLASHINFER_DISABLE_VERSION_CHECK",
    "FLASHINFER_CUDA_ARCH_LIST", "TORCH_CUDA_ARCH_LIST", "JIT_MONITOR_MODE",
)
SOURCE_FILES = {
    "dspark_loader": "vllm/models/deepseek_v4/nvidia/dspark.py",
    "sparse_wrapper": "vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py",
    "sparse_metadata": "vllm/v1/attention/backends/mla/sparse_swa.py",
    "indexer": "vllm/v1/attention/backends/mla/indexer.py",
    "prefix_manager": "vllm/v1/core/kv_cache_manager.py",
    "scheduler": "vllm/v1/core/sched/scheduler.py",
    "b12x_adapter": "vllm/model_executor/layers/fused_moe/experts/b12x_mxfp4_moe.py",
    "flashinfer_dispatch": "flashinfer/mla/_sparse_mla_sm120.py",
    "flashinfer_decode_cuda": "flashinfer/data/csrc/sparse_mla_sm120_decode_dsv4.cu",
    "flashinfer_binding_cuda": "flashinfer/data/csrc/sparse_mla_sm120_jit_binding.cu",
    "flashinfer_prefill_cuda": "flashinfer/data/csrc/sparse_mla_sm120_prefill.cu",
    "flashinfer_prefill_header": "flashinfer/data/include/flashinfer/attention/sparse_mla_sm120/prefill_kernel.cuh",
}
LOG_PATTERNS = (
    r"decode_backend=", r"Using .*attention backend", r"attention backend:",
    r"DSA indexer decode path:", r"Prewarmed B12X route-pack",
    r"DSpark draft model loaded", r"Capturing CUDA graph", r"cudagraph_mode",
    r"enable_adaptive_verification", r"FLASHINFER_MLA_SPARSE_DSV4",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def declared_constant(source: str, name: str):
    """Extract a literal constant without executing package code."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        value = node.value
        if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and value.func.id in {"frozenset", "set", "tuple"}
                and len(value.args) == 1 and not value.keywords):
            value = value.args[0]
        try:
            literal = ast.literal_eval(value)
            return sorted(literal) if isinstance(literal, (set, frozenset)) else literal
        except (ValueError, TypeError):
            return None
    return None


def requested_flags(argv: list[str]) -> dict:
    result = {}
    for i, argument in enumerate(argv):
        flag, separator, value = argument.partition("=")
        if flag not in SERVING_FLAGS:
            continue
        if not separator:
            value = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else True
        if isinstance(value, str) and flag.endswith("-config"):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        result[flag.removeprefix("--")] = value
    return result


def package_info(name: str) -> dict:
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None}
    info = {"installed": True, "version": dist.version}
    direct = dist.read_text("direct_url.json")
    if direct:
        try:
            metadata = json.loads(direct)
            # Do not export URLs: they can contain authentication information.
            info["source_commit"] = metadata.get("vcs_info", {}).get("commit_id")
            info["archive_hashes"] = metadata.get("archive_info", {}).get("hashes", {})
        except json.JSONDecodeError:
            info["provenance_error"] = "invalid direct_url.json"
    return info


def locate_source(relative: str, source_roots: list[Path]) -> Path | None:
    for root in source_roots:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    package = "flashinfer-python" if relative.startswith("flashinfer/") else "vllm"
    try:
        candidate = Path(importlib.metadata.distribution(package).locate_file(relative))
        return candidate if candidate.is_file() else None
    except importlib.metadata.PackageNotFoundError:
        return None


def inspect_sources(roots: list[Path]) -> dict:
    result = {}
    for key, relative in SOURCE_FILES.items():
        path = locate_source(relative, roots)
        source = read_text(path) if path else None
        entry = {"available": source is not None, "relative_path": relative}
        if source is not None:
            entry["sha256"] = hashlib.sha256(source.encode()).hexdigest()
            if key == "flashinfer_dispatch":
                entry["declared_decode_shapes"] = declared_constant(source, "_DECODE_DSV4_DISPATCH")
                entry["declared_packed_bytes"] = declared_constant(source, "_BPT_DSV4")
                entry["declared_decode_token_limit"] = declared_constant(source, "_DECODE_MAX_TOKENS")
            elif key == "sparse_wrapper":
                entry["declared_decode_topks"] = declared_constant(source, "_FLASHINFER_DSV4_DECODE_TOPKS")
                entry["uses_dsv4_sparse_api"] = "flashinfer_trtllm_batch_decode_sparse_mla_dsv4(" in source
            elif key == "dspark_loader":
                entry["confidence_weights_skipped_marker"] = "confidence head is not wired into inference yet" in source
            elif key == "prefix_manager":
                entry["draft_window_recompute_marker"] = "request.num_tokens - 1 - self.dspark_window_size" in source
            elif key == "b12x_adapter":
                entry["route_pack_warmup_marker"] = "_prewarm_b12x_route_pack" in source
        result[key] = entry
    return result


def model_manifest(directory: Path) -> dict:
    if not directory.is_dir():
        return {"available": False, "error": "model directory not found"}
    files = {}
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors.index.json"):
        path = directory / name
        if path.is_file():
            files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    shards = sorted(directory.glob("*.safetensors"))
    config_text = read_text(directory / "config.json")
    try:
        config = json.loads(config_text) if config_text else {}
        selected = {key: config[key] for key in (
            "architectures", "model_type", "torch_dtype", "dtype", "sliding_window",
            "n_mtp_layers", "dspark_target_layer_ids", "enable_confidence_head",
            "max_position_embeddings",
        ) if key in config}
    except json.JSONDecodeError:
        selected = {"error": "invalid config.json"}
    return {"available": True, "metadata": files, "config": selected,
            "shard_count": len(shards), "shard_bytes": sum(path.stat().st_size for path in shards),
            "weights_content_hashed": False}


def observed_log_evidence(paths: list[Path]) -> dict:
    pattern = re.compile("|".join(LOG_PATTERNS), re.IGNORECASE)
    result = {"provided": bool(paths), "matches": [], "unreadable": []}
    for path in paths:
        try:
            with path.open(encoding="utf-8", errors="replace") as stream:
                matches = []
                for number, line in enumerate(stream, 1):
                    if pattern.search(line):
                        # Only report short matching lines, not complete logs/configs.
                        matches.append({"line": number, "text": line.strip()[:800]})
                        matches = matches[-100:]
                result["matches"].append({"file": path.name, "evidence": matches})
        except OSError:
            result["unreadable"].append(path.name)
    return result


def gpu_inventory() -> dict:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False}
    fields = ["name", "uuid", "compute_cap", "driver_version", "memory.total", "memory.used", "temperature.gpu"]
    try:
        proc = subprocess.run([executable, "--query-gpu=" + ",".join(fields),
                               "--format=csv,noheader,nounits"],
                              text=True, capture_output=True, timeout=10, check=True)
        rows = [{key: value.strip() for key, value in zip(fields, row)}
                for row in csv.reader(io.StringIO(proc.stdout))]
        return {"available": True, "devices": rows}
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "error_type": type(error).__name__}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", default=[], type=Path,
                        help="Source roots, searched in order before installed packages")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--server-log", action="append", default=[], type=Path)
    parser.add_argument("--command-json", type=Path, help="JSON argv list, e.g. Docker Config.Cmd")
    parser.add_argument("--server-pid", type=int, help="Read argv from this exact /proc PID")
    parser.add_argument("--gpu", action="store_true", help="Read nvidia-smi without allocating GPU memory")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command_json and args.server_pid is not None:
        parser.error("choose --command-json or --server-pid")
    argv = []
    if args.command_json:
        argv = json.loads(args.command_json.read_text())
        if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
            parser.error("--command-json must contain an argv list of strings")
    if args.server_pid is not None:
        if args.server_pid < 1:
            parser.error("--server-pid must be positive")
        argv = Path(f"/proc/{args.server_pid}/cmdline").read_bytes().decode().strip("\0").split("\0")
    report = {
        "schema_version": 1, "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "system": platform.system(),
        "architecture": platform.machine(),
        "packages": {name: package_info(name) for name in PACKAGES},
        "requested": {"flags": requested_flags(argv),
                      "inspector_environment": {name: os.environ[name] for name in ENV_NAMES if name in os.environ}},
        "source": inspect_sources(args.source_root),
        "observed_logs": observed_log_evidence(args.server_log),
        "limitations": [
            "No model weights loaded and no attention kernels executed.",
            "Source-declared capabilities are not GPU correctness or dispatch validation.",
            "Requested configuration and inspector environment are not proof of effective server settings.",
            "Log evidence may come from an earlier process; correlate with container identity and start time.",
        ],
    }
    if args.model_dir:
        report["model"] = model_manifest(args.model_dir)
    if args.gpu:
        report["gpu"] = gpu_inventory()
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
