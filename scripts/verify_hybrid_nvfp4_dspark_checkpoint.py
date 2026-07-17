#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Verify a staged NVIDIA-NVFP4 + native-MXFP4 DSpark checkpoint.

``metadata-only`` verifies the seven files which are safe to copy to a TP
worker before a RoCE-backed load. ``runnable`` additionally requires all 48
weight shards to be materialized as regular files. The verifier never opens a
weight shard unless its provenance record contains a SHA-256 digest.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import build_hybrid_nvfp4_dspark_checkpoint as builder


METADATA_FILES = (
    builder.CONFIG_NAME,
    builder.INDEX_NAME,
    builder.PROVENANCE_NAME,
    *builder.RUNTIME_METADATA_FILES,
)
ARTIFACT_FILES = tuple(
    sorted((builder.CONFIG_NAME, builder.INDEX_NAME, *builder.RUNTIME_METADATA_FILES))
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class VerificationContract:
    """Immutable identities and structural contract for one hybrid artifact."""

    sources: builder.HybridContract
    merged_index: builder.IndexContract
    expected_merged_config_sha256: str | None
    expected_merged_index_sha256: str | None
    expected_dspark_config_sha256: str | None
    expected_dspark_index_sha256: str | None
    expected_artifacts: Mapping[str, Mapping[str, Any]]


PINNED_ARTIFACTS: dict[str, dict[str, Any]] = {
    "config.json": {
        "sha256": "bb0d2286d6761439e41d3cef31d16489411b816ed8688922f59730bbd5567cdb",
        "size": 7_036,
    },
    "generation_config.json": {
        "sha256": "5fccff80f55a4d455bbe516bdd552edf3e9623df95e99fbf2a3c3389fdf91af0",
        "size": 170,
    },
    "hf_quant_config.json": {
        "sha256": "1fbdb2bd9831a11cb822bc048fd45180927bcfb821ba32a3a7050f96a37b0b5f",
        "size": 4_546,
    },
    "model.safetensors.index.json": {
        "sha256": "819d3eb30b47ab6ccee24dcf22729effd59327a1082cf089a53a7779ff1c8669",
        "size": 11_437_404,
    },
    "tokenizer.json": {
        "sha256": "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf",
        "size": 6_367_146,
    },
    "tokenizer_config.json": {
        "sha256": "6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547",
        "size": 801,
    },
}

PINNED_MERGED_INDEX = builder.IndexContract(
    total_size=175_535_844_088,
    tensor_count=138_365,
    non_mtp_count=133_660,
    mtp_counts=((0, 1_568), (1, 1_565), (2, 1_572)),
    shard_count=48,
    stage_shards=((0, 46), (1, 47), (2, 48)),
)

PINNED_VERIFICATION_CONTRACT = VerificationContract(
    sources=builder.PINNED_CONTRACT,
    merged_index=PINNED_MERGED_INDEX,
    expected_merged_config_sha256=PINNED_ARTIFACTS[builder.CONFIG_NAME]["sha256"],
    expected_merged_index_sha256=PINNED_ARTIFACTS[builder.INDEX_NAME]["sha256"],
    expected_dspark_config_sha256=(
        "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023"
    ),
    expected_dspark_index_sha256=(
        "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
    ),
    expected_artifacts=PINNED_ARTIFACTS,
)


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise builder.ContractError(message)


def _regular_file(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise builder.ContractError(f"{label} is missing: {path}") from error
    except OSError as error:
        raise builder.ContractError(
            f"cannot stat {label} at {path}: {error}"
        ) from error
    _expect(
        not stat.S_ISLNK(metadata.st_mode),
        f"{label} must not be a symlink: {path}",
    )
    _expect(stat.S_ISREG(metadata.st_mode), f"{label} must be a regular file: {path}")


def _directory(path: Path) -> Path:
    lexical = path.expanduser()
    _expect(
        not lexical.is_symlink(),
        f"checkpoint directory must not be a symlink: {lexical}",
    )
    try:
        _expect(lexical.is_dir(), f"checkpoint directory is missing: {lexical}")
        return lexical.resolve(strict=True)
    except OSError as error:
        raise builder.ContractError(
            f"cannot resolve checkpoint directory {lexical}: {error}"
        ) from error


def _expected_summary(contract: builder.IndexContract) -> dict[str, Any]:
    return {
        "total_size": contract.total_size,
        "tensor_count": contract.tensor_count,
        "non_mtp_count": contract.non_mtp_count,
        "mtp_counts": {str(stage): count for stage, count in contract.mtp_counts},
        "shard_count": contract.shard_count,
    }


def _validate_config(
    config: Mapping[str, Any], contract: VerificationContract
) -> None:
    source_contract = contract.sources
    _expect(
        tuple(config.get("compress_ratios", ()))
        == source_contract.dspark_compress_ratios,
        "hybrid compress_ratios do not match the pinned DSpark layout",
    )
    for name, value in source_contract.dspark_fields.items():
        _expect(
            config.get(name) == value,
            f"hybrid {name} is {config.get(name)!r}, expected {value!r}",
        )

    # The target quantization validator expects the shorter NVIDIA source
    # compress-ratio vector. Normalize only that grafted field, then reuse its
    # exact architecture, producer, ignore-map, and per-layer NVFP4 checks.
    target_view = copy.deepcopy(dict(config))
    target_view["compress_ratios"] = list(source_contract.nvidia_compress_ratios)
    builder._validate_nvidia_config(target_view, source_contract)


def _validate_hash_identity(
    actual: str, expected: str | None, label: str
) -> None:
    if expected is not None:
        _expect(
            actual == expected,
            f"{label} SHA-256 does not match the pinned artifact",
        )


def _validate_artifacts(
    root: Path,
    provenance: Mapping[str, Any],
    contract: VerificationContract,
) -> dict[str, str]:
    merged = provenance.get("merged")
    _expect(isinstance(merged, dict), "provenance merged must be an object")
    assert isinstance(merged, dict)
    artifacts = merged.get("artifacts")
    _expect(
        isinstance(artifacts, dict),
        "provenance merged.artifacts must be an object",
    )
    assert isinstance(artifacts, dict)
    _expect(
        set(artifacts) == set(ARTIFACT_FILES),
        f"provenance artifacts must contain exactly {list(ARTIFACT_FILES)}",
    )
    _expect(
        dict(artifacts) == {
            name: dict(contract.expected_artifacts[name]) for name in ARTIFACT_FILES
        },
        "provenance metadata artifact identities do not match the pinned contract",
    )

    hashes: dict[str, str] = {}
    for name in ARTIFACT_FILES:
        path = root / name
        _regular_file(path, f"metadata artifact {name}")
        record = artifacts[name]
        _expect(
            isinstance(record, dict),
            f"artifact record for {name} must be an object",
        )
        assert isinstance(record, dict)
        _expect(
            set(record) == {"sha256", "size"},
            f"artifact record for {name} has unexpected fields",
        )
        _expect(
            path.stat().st_size == record["size"],
            f"metadata artifact {name} size does not match provenance",
        )
        digest = builder._sha256(path)
        _expect(
            digest == record["sha256"],
            f"metadata artifact {name} SHA-256 does not match provenance",
        )
        hashes[name] = digest
    return hashes


def _validate_provenance(
    provenance: Mapping[str, Any], contract: VerificationContract
) -> list[dict[str, Any]]:
    expected_top_level = {
        "schema",
        "builder",
        "materialization",
        "runnable_view",
        "shard_sha256_computed",
        "sources",
        "merged",
        "shards",
    }
    _expect(
        set(provenance) == expected_top_level,
        "provenance top-level fields do not match schema v1",
    )
    _expect(
        provenance.get("schema") == "anemll.hybrid-checkpoint-provenance.v1",
        "unsupported provenance schema",
    )
    _expect(
        provenance.get("builder")
        == "scripts/build_hybrid_nvfp4_dspark_checkpoint.py",
        "unexpected checkpoint builder identity",
    )
    _expect(
        provenance.get("materialization")
        in {"symlink", "hardlink", "copy", "manifest"},
        "invalid provenance materialization",
    )
    _expect(
        isinstance(provenance.get("runnable_view"), bool),
        "provenance runnable_view must be boolean",
    )
    _expect(
        provenance["runnable_view"]
        == (provenance["materialization"] != "manifest"),
        "provenance runnable_view conflicts with materialization",
    )
    computed = provenance.get("shard_sha256_computed")
    _expect(isinstance(computed, bool), "shard_sha256_computed must be boolean")

    sources = provenance.get("sources")
    _expect(
        isinstance(sources, dict) and set(sources) == {"nvidia", "dspark"},
        "provenance sources must contain exactly nvidia and dspark",
    )
    assert isinstance(sources, dict)
    nvidia = sources["nvidia"]
    dspark = sources["dspark"]
    _expect(isinstance(nvidia, dict), "NVIDIA provenance source must be an object")
    _expect(isinstance(dspark, dict), "DSpark provenance source must be an object")
    assert isinstance(nvidia, dict) and isinstance(dspark, dict)
    _expect(
        set(nvidia)
        == {
            "path",
            "expected_revision",
            "observed_revision",
            "config_sha256",
            "index_sha256",
            "contract",
        },
        "NVIDIA provenance source has unexpected fields",
    )
    _expect(
        set(dspark) == {"path", "config_sha256", "index_sha256", "contract"},
        "DSpark provenance source has unexpected fields",
    )
    _expect(
        isinstance(nvidia["path"], str) and nvidia["path"],
        "invalid NVIDIA source path",
    )
    _expect(
        isinstance(dspark["path"], str) and dspark["path"],
        "invalid DSpark source path",
    )
    _expect(
        nvidia["expected_revision"] == contract.sources.expected_nvidia_revision,
        "NVIDIA expected revision does not match the pinned contract",
    )
    _expect(
        nvidia["observed_revision"]
        in {None, contract.sources.expected_nvidia_revision},
        "NVIDIA observed revision conflicts with the pinned contract",
    )
    _validate_hash_identity(
        nvidia["config_sha256"],
        contract.sources.expected_nvidia_config_sha256,
        "NVIDIA source config",
    )
    _validate_hash_identity(
        nvidia["index_sha256"],
        contract.sources.expected_nvidia_index_sha256,
        "NVIDIA source index",
    )
    _validate_hash_identity(
        dspark["config_sha256"],
        contract.expected_dspark_config_sha256,
        "DSpark source config",
    )
    _validate_hash_identity(
        dspark["index_sha256"],
        contract.expected_dspark_index_sha256,
        "DSpark source index",
    )
    _expect(
        nvidia["contract"] == _expected_summary(contract.sources.nvidia),
        "NVIDIA source summary does not match the pinned contract",
    )
    _expect(
        dspark["contract"] == _expected_summary(contract.sources.dspark),
        "DSpark source summary does not match the pinned contract",
    )

    merged = provenance["merged"]
    _expect(isinstance(merged, dict), "provenance merged must be an object")
    assert isinstance(merged, dict)
    _expect(
        set(merged)
        == {
            "artifacts",
            "config_base",
            "copied_dspark_config_fields",
            "draft_stages",
            "index_total_size",
            "tensor_count",
            "shard_count",
            "target_layers",
            "target_quantization",
            "draft_quantization",
        },
        "merged provenance has unexpected fields",
    )
    expected_merged = {
        "config_base": "nvidia",
        "copied_dspark_config_fields": list(builder.DSPARK_CONFIG_FIELDS),
        "draft_stages": [stage for stage, _count in contract.merged_index.mtp_counts],
        "index_total_size": contract.merged_index.total_size,
        "tensor_count": contract.merged_index.tensor_count,
        "shard_count": contract.merged_index.shard_count,
        "target_layers": [0, contract.sources.num_hidden_layers - 1],
        "target_quantization": "NVFP4 W4A4",
        "draft_quantization": "native MXFP4",
    }
    for name, value in expected_merged.items():
        _expect(
            merged.get(name) == value,
            f"merged provenance {name} is {merged.get(name)!r}, expected {value!r}",
        )

    shards = provenance.get("shards")
    _expect(isinstance(shards, list), "provenance shards must be a list")
    assert isinstance(shards, list)
    _expect(
        len(shards) == contract.merged_index.shard_count,
        f"provenance must describe exactly {contract.merged_index.shard_count} shards",
    )
    expected_destinations = {
        builder._shard_name(number, contract.merged_index.shard_count)
        for number in range(1, contract.merged_index.shard_count + 1)
    }
    records: dict[str, dict[str, Any]] = {}
    hashes_present: list[bool] = []
    stage_shards = contract.merged_index.stage_shard_map
    stage_by_shard = {shard: stage for stage, shard in stage_shards.items()}
    target_count = contract.merged_index.target_shard_count
    for record in shards:
        _expect(
            isinstance(record, dict),
            "every provenance shard must be an object",
        )
        assert isinstance(record, dict)
        _expect(
            set(record)
            == {
                "destination",
                "role",
                "source_name",
                "source_size",
                "sha256",
                "hub_etag",
                "hub_revision",
            },
            "provenance shard record has unexpected fields",
        )
        destination = record["destination"]
        _expect(
            isinstance(destination, str) and destination in expected_destinations,
            f"invalid shard destination {destination!r}",
        )
        _expect(destination not in records, f"duplicate shard destination {destination}")
        number, total = builder._parse_shard_name(destination, "hybrid provenance")
        _expect(
            total == contract.merged_index.shard_count,
            "wrong destination shard total",
        )
        if number <= target_count:
            expected_role = "nvidia-target"
            expected_source = builder._shard_name(
                number, contract.sources.nvidia.shard_count
            )
            _expect(
                record["hub_revision"]
                in {None, contract.sources.expected_nvidia_revision},
                f"NVIDIA shard {destination} has a conflicting Hub revision",
            )
        else:
            _expect(
                number in stage_by_shard,
                f"shard {destination} has no draft stage",
            )
            expected_role = f"dspark-mtp.{stage_by_shard[number]}"
            expected_source = destination
            _expect(
                record["hub_etag"] is None and record["hub_revision"] is None,
                f"DSpark shard {destination} must not claim NVIDIA Hub metadata",
            )
        _expect(record["role"] == expected_role, f"wrong role for shard {destination}")
        _expect(
            record["source_name"] == expected_source,
            f"wrong source name for shard {destination}",
        )
        _expect(
            isinstance(record["source_size"], int) and record["source_size"] > 0,
            f"invalid source size for shard {destination}",
        )
        digest = record["sha256"]
        _expect(
            digest is None or (isinstance(digest, str) and SHA256_RE.fullmatch(digest)),
            f"invalid SHA-256 for shard {destination}",
        )
        hashes_present.append(digest is not None)
        records[destination] = record
    _expect(set(records) == expected_destinations, "provenance shard set is incomplete")
    _expect(
        all(hashes_present) if computed else not any(hashes_present),
        "shard_sha256_computed does not agree with shard records",
    )
    return [records[name] for name in sorted(records)]


def _metadata_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in METADATA_FILES:
        path = root / name
        _regular_file(path, f"metadata file {name}")
        hashes[name] = builder._sha256(path)
    return hashes


def _compare_reference(root: Path, reference: Path) -> dict[str, str]:
    reference_root = _directory(reference)
    actual = _metadata_hashes(root)
    expected = _metadata_hashes(reference_root)
    for name in METADATA_FILES:
        _expect(
            actual[name] == expected[name],
            f"metadata file {name} does not match reference {reference_root}",
        )
    return expected


def verify_checkpoint(
    directory: Path,
    mode: str,
    reference: Path | None = None,
    contract: VerificationContract = PINNED_VERIFICATION_CONTRACT,
) -> dict[str, Any]:
    """Verify a hybrid directory without importing any ML dependencies."""

    _expect(mode in {"runnable", "metadata-only"}, f"unsupported mode {mode!r}")
    root = _directory(directory)
    try:
        entries = {entry.name: entry for entry in root.iterdir()}
    except OSError as error:
        raise builder.ContractError(
            f"cannot list checkpoint directory {root}: {error}"
        ) from error
    _expect(
        set(METADATA_FILES).issubset(entries),
        "checkpoint is missing one or more of the seven required metadata files",
    )
    for name in METADATA_FILES:
        _regular_file(entries[name], f"metadata file {name}")
    if mode == "metadata-only":
        _expect(
            set(entries) == set(METADATA_FILES),
            f"metadata-only directory must contain exactly {len(METADATA_FILES)} files",
        )
    else:
        _expect(
            len(entries) == len(METADATA_FILES) + contract.merged_index.shard_count,
            "runnable directory has the wrong number of entries",
        )
        for name, path in entries.items():
            if name not in METADATA_FILES:
                _expect(
                    builder.SHARD_RE.fullmatch(name) is not None,
                    f"runnable directory has unexpected entry {name!r}",
                )
                _regular_file(path, f"weight shard {name}")

    index = builder._read_json(root / builder.INDEX_NAME, "hybrid index")
    config = builder._read_json(root / builder.CONFIG_NAME, "hybrid config")
    provenance = builder._read_json(
        root / builder.PROVENANCE_NAME, "hybrid provenance"
    )

    _validate_config(config, contract)
    config_hash = builder._sha256(root / builder.CONFIG_NAME)
    index_hash = builder._sha256(root / builder.INDEX_NAME)
    _validate_hash_identity(
        config_hash, contract.expected_merged_config_sha256, "merged config"
    )
    _validate_hash_identity(
        index_hash, contract.expected_merged_index_sha256, "merged index"
    )
    summary = builder._index_summary(index, "Hybrid")
    builder._validate_index_contract(
        index, summary, contract.merged_index, "Hybrid"
    )
    shard_records = _validate_provenance(provenance, contract)
    artifact_hashes = _validate_artifacts(root, provenance, contract)

    shard_names = {record["destination"] for record in shard_records}
    expected_entries = set(METADATA_FILES)
    if mode == "runnable":
        _expect(
            provenance["runnable_view"] is True,
            "runnable mode requires provenance runnable_view=true",
        )
        expected_entries.update(shard_names)
    _expect(
        set(entries) == expected_entries,
        f"{mode} directory entries differ: expected {len(expected_entries)} exact entries, "
        f"found {len(entries)}",
    )
    hashed_shards = 0
    if mode == "runnable":
        for record in shard_records:
            name = record["destination"]
            path = entries[name]
            _regular_file(path, f"weight shard {name}")
            _expect(
                path.stat().st_size == record["source_size"],
                f"weight shard {name} size does not match provenance",
            )
            if record["sha256"] is not None:
                _expect(
                    builder._sha256(path) == record["sha256"],
                    f"weight shard {name} SHA-256 does not match provenance",
                )
                hashed_shards += 1

    reference_hashes = None
    if reference is not None:
        reference_hashes = _compare_reference(root, reference)

    metadata_hashes = {
        **artifact_hashes,
        builder.PROVENANCE_NAME: builder._sha256(root / builder.PROVENANCE_NAME),
    }
    return {
        "status": "valid",
        "mode": mode,
        "directory": str(root),
        "metadata_file_count": len(METADATA_FILES),
        "tensor_count": summary.tensor_count,
        "index_total_size": summary.total_size,
        "shard_count": len(shard_records),
        "materialized_shard_count": len(shard_records) if mode == "runnable" else 0,
        "payload_sha256_verified": hashed_shards,
        "metadata_sha256": dict(sorted(metadata_hashes.items())),
        "reference": str(reference.resolve()) if reference is not None else None,
        "reference_sha256": (
            dict(sorted(reference_hashes.items())) if reference_hashes else None
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a pinned NVIDIA NVFP4 target + native-MXFP4 DSpark "
            "checkpoint or its metadata-only worker view."
        )
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("runnable", "metadata-only"), required=True
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="optional checkpoint whose seven metadata hashes must match",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = verify_checkpoint(args.checkpoint_dir, args.mode, args.reference)
    except (builder.ContractError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
