#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate TP-rank target-route artifacts and summarize C=4 collisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rank_artifact(manifest_path: Path) -> tuple[dict[str, Any], np.ndarray]:
    manifest = json.loads(manifest_path.read_text())
    required = {
        "schema_version",
        "status",
        "rank",
        "world_size",
        "shape",
        "dtype",
        "data_file",
        "data_size",
        "data_sha256",
        "raw_tensor_sha256",
        "layer_name_sha256",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        raise RuntimeError(f"route manifest is missing fields: {missing}")
    if manifest["schema_version"] != 1 or manifest["status"] != "complete":
        raise RuntimeError(f"route manifest is not complete schema v1: {manifest_path}")
    data_name = manifest["data_file"]
    if Path(data_name).name != data_name:
        raise RuntimeError(f"route data_file must be a basename: {data_name!r}")
    data_path = manifest_path.parent / data_name
    if data_path.stat().st_size != manifest["data_size"]:
        raise RuntimeError(f"route data size mismatch: {data_path}")
    digest = sha256_file(data_path)
    if digest != manifest["data_sha256"]:
        raise RuntimeError(f"route data SHA-256 mismatch: {data_path}")
    data = np.load(data_path, allow_pickle=False)
    if list(data.shape) != manifest["shape"] or str(data.dtype) != manifest["dtype"]:
        raise RuntimeError(f"route data shape/dtype mismatch: {data_path}")
    raw_digest = hashlib.sha256(data.tobytes(order="C")).hexdigest()
    if raw_digest != manifest["raw_tensor_sha256"]:
        raise RuntimeError(f"route raw tensor SHA-256 mismatch: {data_path}")
    if data.ndim != 4 or data.shape[1:] != (4, 43, 6):
        raise RuntimeError(f"unexpected target route tensor shape: {data.shape}")
    return manifest, data


def analyze_rank_pair(
    first_manifest_path: Path, second_manifest_path: Path
) -> dict[str, Any]:
    first_manifest, first = load_rank_artifact(first_manifest_path)
    second_manifest, second = load_rank_artifact(second_manifest_path)
    manifests = sorted(
        ((int(first_manifest["rank"]), first_manifest, first),
         (int(second_manifest["rank"]), second_manifest, second)),
        key=lambda row: row[0],
    )
    if [row[0] for row in manifests] != [0, 1]:
        observed_ranks = [row[0] for row in manifests]
        raise RuntimeError(
            f"expected rank artifacts 0 and 1, got {observed_ranks}"
        )
    rank0_manifest, rank0 = manifests[0][1], manifests[0][2]
    rank1_manifest, rank1 = manifests[1][1], manifests[1][2]
    if rank0_manifest["world_size"] != 2 or rank1_manifest["world_size"] != 2:
        raise RuntimeError("rank artifacts do not prove TP world_size=2")
    for field in ("shape", "dtype", "layer_name_sha256", "steps", "warmup_steps"):
        if rank0_manifest.get(field) != rank1_manifest.get(field):
            raise RuntimeError(f"rank route metadata differs for {field}")
    if not np.array_equal(rank0, rank1):
        mismatch = np.argwhere(rank0 != rank1)[0].tolist()
        raise RuntimeError(
            "logical target routes differ across TP ranks at "
            f"{mismatch}: rank0={int(rank0[tuple(mismatch)])}, "
            f"rank1={int(rank1[tuple(mismatch)])}"
        )

    steps = rank0.shape[0]
    active = np.empty((steps, 43), dtype=np.int32)
    max_multiplicity = np.empty((steps, 43), dtype=np.int32)
    multiplicity_histogram = {str(value): 0 for value in range(1, 5)}
    for step in range(steps):
        for layer in range(43):
            counts = np.bincount(
                rank0[step, :, layer, :].reshape(-1), minlength=256
            )
            nonzero = counts[counts > 0]
            active[step, layer] = nonzero.size
            max_multiplicity[step, layer] = int(nonzero.max())
            for value in range(1, 5):
                multiplicity_histogram[str(value)] += int(
                    np.count_nonzero(nonzero == value)
                )
    collisions = 24 - active
    percentiles = (0, 10, 25, 50, 75, 90, 100)
    active_percentiles = np.percentile(active, percentiles)
    return {
        "schema_version": 1,
        "rank_pair_equal": True,
        "world_size": 2,
        "shape": list(rank0.shape),
        "rank_data_sha256": {
            "0": rank0_manifest["data_sha256"],
            "1": rank1_manifest["data_sha256"],
        },
        "logical_route_raw_sha256": rank0_manifest["raw_tensor_sha256"],
        "active_experts": {
            "maximum_possible": 24,
            "mean": float(active.mean()),
            "minimum": int(active.min()),
            "maximum": int(active.max()),
            "percentiles": {
                str(percentile): float(value)
                for percentile, value in zip(percentiles, active_percentiles)
            },
        },
        "cross_token_collisions": {
            "mean": float(collisions.mean()),
            "mean_fraction": float(collisions.mean() / 24.0),
            "minimum": int(collisions.min()),
            "maximum": int(collisions.max()),
        },
        "expert_row_multiplicity": {
            "maximum": int(max_multiplicity.max()),
            "mean_of_layer_step_maximum": float(max_multiplicity.mean()),
            "layer_step_maximum_histogram": {
                str(value): int(np.count_nonzero(max_multiplicity == value))
                for value in range(1, 5)
            },
            "active_expert_histogram": multiplicity_histogram,
        },
        "per_layer": [
            {
                "layer": layer,
                "active_mean": float(active[:, layer].mean()),
                "active_min": int(active[:, layer].min()),
                "active_max": int(active[:, layer].max()),
                "collision_mean": float(collisions[:, layer].mean()),
            }
            for layer in range(43)
        ],
        "per_step_active_mean": [float(row.mean()) for row in active],
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-0-manifest", type=Path, required=True)
    parser.add_argument("--rank-1-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze_rank_pair(args.rank_0_manifest, args.rank_1_manifest)
    if args.output is None:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        atomic_json(args.output, result)
        print(args.output)


if __name__ == "__main__":
    main()
