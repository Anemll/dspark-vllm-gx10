#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prebuild and run the isolated sparse-width component diagnostic.

Use an outer container/process timeout: Python deadline checks cannot interrupt
NVCC, CUDA synchronization, or a failed GPU. No model or service is started.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time


def file_record(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="New directory on a persistent output mount; must not exist")
    parser.add_argument("--cuda-arch", default="12.1a")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER,
                        help="Options for benchmark_sparse_mla.py after --")
    args = parser.parse_args()
    forwarded = args.benchmark_args
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    if any(value == "--output" or value.startswith("--output=") for value in forwarded):
        parser.error("benchmark output is fixed under --run-dir")
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    os.environ["FLASHINFER_WORKSPACE_BASE"] = str(run_dir / "jit-workspace")
    os.environ["FLASHINFER_CUDA_ARCH_LIST"] = args.cuda_arch
    os.environ["FLASHINFER_JIT_DEBUG"] = "0"
    os.environ["FLASHINFER_JIT_VERBOSE"] = "1"
    os.environ.setdefault("FLASHINFER_NVCC_THREADS", "1")
    os.environ.setdefault("MAX_JOBS", "2")
    root = Path(__file__).resolve().parent
    manifest_path = root / "patches" / "runtime-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    report = {"status": "initializing", "started_at": datetime.now(timezone.utc).isoformat(),
              "patch_sha256": manifest["patch_sha256"],
              "target_flashinfer_commit": manifest["target_commit"],
              "upstream_commit": manifest["upstream_commit"],
              "environment": {name: os.environ[name] for name in
                              ("FLASHINFER_WORKSPACE_BASE", "FLASHINFER_CUDA_ARCH_LIST",
                               "FLASHINFER_JIT_DEBUG", "FLASHINFER_JIT_VERBOSE",
                               "FLASHINFER_NVCC_THREADS", "MAX_JOBS")}}
    start = time.monotonic()

    def save() -> None:
        report["elapsed_s"] = time.monotonic() - start
        (run_dir / "runtime-provenance.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        import torch
        import flashinfer
        from flashinfer.jit import env as jit_env
        from flashinfer.jit.mla import gen_sparse_mla_sm120_module

        capability = torch.cuda.get_device_capability()
        if capability != (12, 1) and args.cuda_arch == "12.1a":
            raise RuntimeError(f"default GB10 diagnostic requires SM121; found {capability}")
        package = Path(flashinfer.__file__).resolve().parent
        records = []
        for entry in manifest["files"]:
            record = file_record(package / entry["installed"])
            if record["sha256"] != entry["new_sha256"]:
                raise RuntimeError(f"patched runtime hash mismatch: {entry['installed']}")
            records.append(record)
        # This pinned FlashInfer prefers AOT binaries even with a fresh workspace.
        # Override its module attribute before any sparse JitSpec is requested.
        jit_env.FLASHINFER_AOT_DIR = run_dir / "empty-aot"
        jit_env.FLASHINFER_AOT_DIR.mkdir()
        spec = gen_sparse_mla_sm120_module()
        if spec.is_aot or not spec.jit_library_path.resolve().is_relative_to(run_dir):
            raise RuntimeError("sparse module is not isolated to the diagnostic run")
        report.update({"status": "compiling", "runtime_files": records,
                       "flashinfer_package": str(package),
                       "flashinfer_version": flashinfer.__version__,
                       "flashinfer_distribution_version": importlib.metadata.version("flashinfer-python"),
                       "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                       "gpu": {"name": torch.cuda.get_device_name(), "capability": capability},
                       "aot_dir": str(jit_env.FLASHINFER_AOT_DIR), "is_aot": spec.is_aot,
                       "library_path": str(spec.jit_library_path),
                       "extra_cuda_cflags": spec.extra_cuda_cflags,
                       "nvcc_version": subprocess.check_output(["nvcc", "--version"], text=True),
                       "image_build_provenance": json.loads((root / "build-provenance.json").read_text())})
        save()
        print(json.dumps({"status": "compiling", "library": str(spec.jit_library_path)}), flush=True)
        compile_start = time.monotonic()
        spec.build_and_load()
        torch.cuda.synchronize()
        report.update({"status": "compiled", "compile_elapsed_s": time.monotonic() - compile_start,
                       "library": file_record(spec.jit_library_path),
                       "ninja": file_record(spec.ninja_path)})
        save()
        print(json.dumps({"status": "compiled", "compile_elapsed_s": report["compile_elapsed_s"],
                          "library_sha256": report["library"]["sha256"]}), flush=True)
        if not args.compile_only:
            sys.argv = [str(root / "benchmark_sparse_mla.py"), *forwarded,
                        "--output", str(run_dir / "benchmark.json")]
            runpy.run_path(str(root / "benchmark_sparse_mla.py"), run_name="__main__")
            report["status"] = "passed"
        else:
            report["status"] = "compile_only_passed"
    except BaseException as error:
        report.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
