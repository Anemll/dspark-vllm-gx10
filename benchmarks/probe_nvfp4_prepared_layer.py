#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed physical boundary probe for prepared NVFP4 layer 0.

The probe is intentionally independent of a completed checkpoint manifest. It
pins the physical layer file and optional resumable build state before and
after invoking vLLM's shared prepared-layer validator, then emits one
JSON-serializable report. No model is constructed and no GPU is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import re
import stat
import traceback
from typing import Any, Callable, Sequence


SCHEMA_VERSION = 1
LAYER = 0
LAYER_FILENAME = "model-layer-00000.safetensors"
BUILD_STATE_FILENAME = ".dspark-nvfp4-prepared-build-state.json"
PREPARED_SCHEMA = "dspark.deepseek_v4.nvfp4.tp2_cutlass_prepared.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_expected_sha(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be an exact lowercase SHA-256")
    return value


def _file_snapshot(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a direct regular file: {path}")
    stat = path.stat()
    return {
        "path": str(path),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "sha256": _sha256_file(path),
    }


def _direct_regular_input(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        mode = absolute.lstat().st_mode
    except OSError as error:
        raise RuntimeError(f"Cannot stat {label} {absolute}: {error}") from error
    if not stat.S_ISREG(mode):
        raise RuntimeError(f"{label} must be a direct regular file: {absolute}")
    return absolute.resolve()


def _read_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read prepared build state {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError("Prepared build state must be a JSON object")
    return value


def _cross_check_state_row(
    state: dict[str, Any], layer_snapshot: dict[str, Any]
) -> dict[str, Any]:
    contract = state.get("contract")
    if (
        not isinstance(contract, dict)
        or contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("format") != PREPARED_SCHEMA
    ):
        raise RuntimeError("Prepared build-state contract drifted")
    files = state.get("files")
    row = files.get(LAYER_FILENAME) if isinstance(files, dict) else None
    if (
        not isinstance(row, dict)
        or row.get("path") != LAYER_FILENAME
        or row.get("layer") != LAYER
        or row.get("size") != layer_snapshot["size"]
        or row.get("sha256") != layer_snapshot["sha256"]
    ):
        raise RuntimeError("Prepared build-state layer0 row drifted")
    return {
        "path": row["path"],
        "layer": row["layer"],
        "size": row["size"],
        "sha256": row["sha256"],
    }


def _assert_unchanged(
    before: dict[str, Any], after: dict[str, Any], label: str
) -> None:
    if before != after:
        raise RuntimeError(f"{label} changed while the read-only probe ran")


def run_probe(
    layer_file: Path,
    *,
    expected_file_sha256: str | None = None,
    build_state: Path | None = None,
    expected_build_state_sha256: str | None = None,
    validator_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    expected_file_sha256 = _parse_expected_sha(
        expected_file_sha256, "expected file digest"
    )
    expected_build_state_sha256 = _parse_expected_sha(
        expected_build_state_sha256, "expected build-state digest"
    )
    if expected_build_state_sha256 is not None and build_state is None:
        raise ValueError("Expected build-state digest requires --build-state")
    layer_input = Path(os.path.abspath(os.fspath(layer_file)))
    layer_path = _direct_regular_input(layer_input, "prepared layer0")
    if layer_path.name != LAYER_FILENAME:
        raise RuntimeError(f"Layer file must be named exactly {LAYER_FILENAME}")
    before_layer = _file_snapshot(layer_path, "prepared layer0")
    if (
        expected_file_sha256 is not None
        and before_layer["sha256"] != expected_file_sha256
    ):
        raise RuntimeError("Prepared layer0 digest differs from its expected pin")

    state_path: Path | None = None
    before_state: dict[str, Any] | None = None
    state_row: dict[str, Any] | None = None
    if build_state is not None:
        state_input = Path(os.path.abspath(os.fspath(build_state)))
        expected_state_input = layer_input.parent / BUILD_STATE_FILENAME
        if state_input != expected_state_input:
            raise RuntimeError(
                "Prepared build-state path must be exactly beside the supplied "
                f"layer file: {expected_state_input}"
            )
        state_path = _direct_regular_input(state_input, "prepared build state")
        if state_path.parent != layer_path.parent:
            raise RuntimeError(
                "Prepared build state must be beside the prepared layer file"
            )
        before_state = _file_snapshot(state_path, "prepared build state")
        if (
            expected_build_state_sha256 is not None
            and before_state["sha256"] != expected_build_state_sha256
        ):
            raise RuntimeError(
                "Prepared build-state digest differs from its expected pin"
            )
        state_row = _cross_check_state_row(
            _read_state(state_path), before_layer
        )

    if validator_fn is None:
        from vllm.models.deepseek_v4.nvidia.prepared_weight_loading import (
            validate_prepared_layer_file,
        )

        validator_fn = validate_prepared_layer_file
    physical = validator_fn(layer_path, layer=LAYER)

    after_layer = _file_snapshot(layer_path, "prepared layer0")
    _assert_unchanged(before_layer, after_layer, "Prepared layer0")
    after_state: dict[str, Any] | None = None
    if state_path is not None:
        assert before_state is not None
        after_state = _file_snapshot(state_path, "prepared build state")
        _assert_unchanged(before_state, after_state, "Prepared build state")
        after_row = _cross_check_state_row(_read_state(state_path), after_layer)
        if after_row != state_row:
            raise RuntimeError("Prepared build-state layer0 row changed")
    if physical.get("ok") is not True:
        raise RuntimeError("Shared prepared physical-layer validator did not pass")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "probe": "deepseek_v4_nvfp4_prepared_physical_layer0",
        "model_loaded": False,
        "gpu_required": False,
        "layer_file": {
            "before": before_layer,
            "after": after_layer,
            "expected_sha256": expected_file_sha256,
            "unchanged": True,
        },
        "build_state": None
        if before_state is None
        else {
            "before": before_state,
            "after": after_state,
            "expected_sha256": expected_build_state_sha256,
            "layer0_row": state_row,
            "unchanged": True,
        },
        "physical_validation": physical,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-file", required=True, type=Path)
    parser.add_argument("--expected-file-sha256")
    parser.add_argument("--build-state", type=Path)
    parser.add_argument("--expected-build-state-sha256")
    parser.add_argument("--output", type=Path)
    return parser


def _emit(report: dict[str, Any], output: Path | None) -> None:
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, output)
    print(encoded, end="")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_probe(
            args.layer_file,
            expected_file_sha256=args.expected_file_sha256,
            build_state=args.build_state,
            expected_build_state_sha256=args.expected_build_state_sha256,
        )
    except Exception as error:
        report = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ok": False,
            "probe": "deepseek_v4_nvfp4_prepared_physical_layer0",
            "model_loaded": False,
            "gpu_required": False,
            "failures": [
                {
                    "kind": "exception",
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc().splitlines(),
                }
            ],
        }
    _emit(report, args.output)
    return 0 if report.get("ok") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
