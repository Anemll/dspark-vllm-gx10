#!/usr/bin/env python3
"""Fail closed when FlashInfer Python and binary artifacts are incoherent."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Callable


VersionGetter = Callable[[str], str]


def _distribution_version(name: str, get_version: VersionGetter) -> str | None:
    try:
        return get_version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def validate_version_contract(
    *,
    expected_version: str,
    expected_cuda_suffix: str,
    require_jit_cache: bool,
    require_cubin: bool,
    get_version: VersionGetter = importlib.metadata.version,
    version_check_override: str | None = None,
) -> dict[str, str | None]:
    """Validate package versions without importing CUDA or FlashInfer."""

    override = (
        os.getenv("FLASHINFER_DISABLE_VERSION_CHECK")
        if version_check_override is None
        else version_check_override
    )
    if override:
        raise RuntimeError(
            "FLASHINFER_DISABLE_VERSION_CHECK must be unset or empty; "
            f"got {override!r}. Values such as '0' still disable FlashInfer's check."
        )

    versions = {
        "flashinfer-python": _distribution_version("flashinfer-python", get_version),
        "flashinfer-jit-cache": _distribution_version(
            "flashinfer-jit-cache", get_version
        ),
        "flashinfer-cubin": _distribution_version("flashinfer-cubin", get_version),
    }
    if versions["flashinfer-python"] != expected_version:
        raise RuntimeError(
            "flashinfer-python mismatch: "
            f"expected {expected_version}, got {versions['flashinfer-python']}"
        )

    expected_jit = f"{expected_version}+{expected_cuda_suffix}"
    jit_version = versions["flashinfer-jit-cache"]
    if require_jit_cache and jit_version is None:
        raise RuntimeError("flashinfer-jit-cache is required but not installed")
    if jit_version is not None and jit_version != expected_jit:
        raise RuntimeError(
            f"flashinfer-jit-cache mismatch: expected {expected_jit}, got {jit_version}"
        )

    cubin_version = versions["flashinfer-cubin"]
    if require_cubin and cubin_version is None:
        raise RuntimeError("flashinfer-cubin is required but not installed")
    if cubin_version is not None and cubin_version != expected_version:
        raise RuntimeError(
            f"flashinfer-cubin mismatch: expected {expected_version}, got {cubin_version}"
        )
    return versions


def inspect_aot(*, require_fused_moe_120: bool) -> dict[str, object]:
    import flashinfer
    from flashinfer.jit import env as jit_env

    aot_dir = Path(jit_env.FLASHINFER_AOT_DIR)
    fused_moe = aot_dir / "fused_moe_120" / "fused_moe_120.so"
    if require_fused_moe_120 and not fused_moe.is_file():
        raise RuntimeError(f"required SM120/SM121 fused-MoE AOT is missing: {fused_moe}")
    return {
        "flashinfer_file": str(flashinfer.__file__),
        "aot_dir": str(aot_dir),
        "aot_dir_exists": aot_dir.is_dir(),
        "fused_moe_120": str(fused_moe),
        "fused_moe_120_exists": fused_moe.is_file(),
    }


def _exercise_fused_moe_init(module: object, torch_module: object) -> dict[str, str]:
    """Call the exact NVFP4 ABI twice so the tuple is unit-testable without CUDA."""

    results: dict[str, str] = {}
    base_args = (
        torch_module.bfloat16,
        # FlashInfer's SM120 NVFP4 AOT accepts packed weights through an int64
        # view. uint8 with every mode flag false selects no native dispatch.
        torch_module.int64,
        torch_module.bfloat16,
        False,
        False,
        False,
        False,
    )
    for use_fused_finalize in (True, False):
        runner = module.init(*base_args, use_fused_finalize)
        results[str(use_fused_finalize).lower()] = type(runner).__name__
    return results


def probe_fused_moe_init() -> dict[str, object]:
    """Exercise both forms of the v0.6.15 eight-argument native ABI."""

    import torch
    from flashinfer.fused_moe.core import gen_cutlass_fused_moe_sm120_module

    module = gen_cutlass_fused_moe_sm120_module(False).build_and_load()
    return {
        "module": repr(module),
        "use_fused_finalize": _exercise_fused_moe_init(module, torch),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", default="0.6.15")
    parser.add_argument("--expected-cuda-suffix", default="cu130")
    parser.add_argument("--require-jit-cache", action="store_true")
    parser.add_argument("--require-cubin", action="store_true")
    parser.add_argument("--require-fused-moe-120", action="store_true")
    parser.add_argument("--probe-fused-moe-init", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report: dict[str, object] = {
        "expected_version": args.expected_version,
        "expected_cuda_suffix": args.expected_cuda_suffix,
        "version_check_override": os.getenv("FLASHINFER_DISABLE_VERSION_CHECK"),
    }
    try:
        report["versions"] = validate_version_contract(
            expected_version=args.expected_version,
            expected_cuda_suffix=args.expected_cuda_suffix,
            require_jit_cache=args.require_jit_cache,
            require_cubin=args.require_cubin,
        )
        report["artifacts"] = inspect_aot(
            require_fused_moe_120=args.require_fused_moe_120
        )
        if args.probe_fused_moe_init:
            report["fused_moe_init"] = probe_fused_moe_init()
        report["ok"] = True
    except Exception as exc:  # Fail closed while preserving machine-readable evidence.
        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
