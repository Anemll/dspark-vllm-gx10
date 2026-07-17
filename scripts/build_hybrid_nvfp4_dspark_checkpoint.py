#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Build a strict NVIDIA-NVFP4 target + native-MXFP4 DSpark view.

The builder deliberately accepts only the pinned DeepSeek V4 Flash layouts
validated by this repository.  It never rewrites checkpoint payloads: target
shards 1..45 come from NVIDIA and DSpark draft shards 46..48 come from the
native-MXFP4 checkpoint.  The default ``symlink`` materialization is a cheap
local view which must be transferred with a dereferencing copy such as
``rsync -L``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


INDEX_NAME = "model.safetensors.index.json"
PROVENANCE_NAME = "checkpoint.provenance.json"
CONFIG_NAME = "config.json"
RUNTIME_METADATA_FILES = (
    "generation_config.json",
    "hf_quant_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
DSPARK_CONFIG_FIELDS = (
    "compress_ratios",
    "dspark_block_size",
    "dspark_noise_token_id",
    "dspark_target_layer_ids",
    "dspark_markov_rank",
)
SHARD_RE = re.compile(r"^model-(\d{5})-of-(\d{5})\.safetensors$")
MTP_RE = re.compile(r"^mtp\.(\d+)\.")
MAX_SAFETENSORS_HEADER_BYTES = 128 * 1024 * 1024


class ContractError(ValueError):
    """A source checkpoint does not satisfy the pinned structural contract."""


@dataclass(frozen=True)
class IndexContract:
    total_size: int
    tensor_count: int
    non_mtp_count: int
    mtp_counts: tuple[tuple[int, int], ...]
    shard_count: int
    stage_shards: tuple[tuple[int, int], ...]

    @property
    def mtp_count_map(self) -> dict[int, int]:
        return dict(self.mtp_counts)

    @property
    def stage_shard_map(self) -> dict[int, int]:
        return dict(self.stage_shards)

    @property
    def target_shard_count(self) -> int:
        return min(self.stage_shard_map.values()) - 1


@dataclass(frozen=True)
class HybridContract:
    nvidia: IndexContract
    dspark: IndexContract
    num_hidden_layers: int
    nvidia_compress_ratios: tuple[int, ...]
    dspark_compress_ratios: tuple[int, ...]
    dspark_fields: Mapping[str, Any]
    expected_nvidia_revision: str | None = None
    expected_nvidia_config_sha256: str | None = None
    expected_nvidia_index_sha256: str | None = None


NVIDIA_COMPRESS_RATIOS = (
    0,
    0,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    0,
)
DSPARK_COMPRESS_RATIOS = NVIDIA_COMPRESS_RATIOS + (0, 0)

PINNED_CONTRACT = HybridContract(
    nvidia=IndexContract(
        total_size=168_266_793_544,
        tensor_count=135_235,
        non_mtp_count=133_660,
        mtp_counts=((0, 1_575),),
        shard_count=46,
        stage_shards=((0, 46),),
    ),
    dspark=IndexContract(
        total_size=166_878_536_440,
        tensor_count=72_317,
        non_mtp_count=67_612,
        mtp_counts=((0, 1_568), (1, 1_565), (2, 1_572)),
        shard_count=48,
        stage_shards=((0, 46), (1, 47), (2, 48)),
    ),
    num_hidden_layers=43,
    nvidia_compress_ratios=NVIDIA_COMPRESS_RATIOS,
    dspark_compress_ratios=DSPARK_COMPRESS_RATIOS,
    dspark_fields={
        "dspark_block_size": 5,
        "dspark_noise_token_id": 128_799,
        "dspark_target_layer_ids": [40, 41, 42],
        "dspark_markov_rank": 256,
    },
    expected_nvidia_revision="e3cd60e7de98e9867116860d522499a728de1cf9",
    expected_nvidia_config_sha256=(
        "0c5dc7303ff322d73e0cd5caf9cc1b65d6efeff68fab53514531c2e959b1d616"
    ),
    expected_nvidia_index_sha256=(
        "2d83d58754cff11724f117d20d95e31803a48512d29f8e00463b2501905d6d72"
    ),
)


@dataclass(frozen=True)
class IndexSummary:
    total_size: int
    tensor_count: int
    non_mtp_count: int
    mtp_counts: Mapping[int, int]
    shards: tuple[str, ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "total_size": self.total_size,
            "tensor_count": self.tensor_count,
            "non_mtp_count": self.non_mtp_count,
            "mtp_counts": {
                str(stage): self.mtp_counts[stage]
                for stage in sorted(self.mtp_counts)
            },
            "shard_count": len(self.shards),
        }


@dataclass(frozen=True)
class SafetensorsHeader:
    tensor_names: frozenset[str]
    payload_bytes: int


@dataclass
class ValidatedSources:
    nvidia_dir: Path
    dspark_dir: Path
    nvidia_config: dict[str, Any]
    dspark_config: dict[str, Any]
    nvidia_index: dict[str, Any]
    dspark_index: dict[str, Any]
    nvidia_summary: IndexSummary
    dspark_summary: IndexSummary
    nvidia_stage_payloads: dict[int, int]
    dspark_stage_payloads: dict[int, int]
    nvidia_revision: str | None
    contract: HybridContract

    @property
    def merged_total_size(self) -> int:
        removed = sum(self.nvidia_stage_payloads.values())
        added = sum(self.dspark_stage_payloads.values())
        return self.nvidia_summary.total_size - removed + added

    @property
    def merged_tensor_count(self) -> int:
        return self.nvidia_summary.non_mtp_count + sum(
            self.dspark_summary.mtp_counts.values()
        )

    def summary_json(self) -> dict[str, Any]:
        return {
            "status": "valid",
            "nvidia": self.nvidia_summary.as_json(),
            "dspark": self.dspark_summary.as_json(),
            "merged": {
                "total_size": self.merged_total_size,
                "tensor_count": self.merged_tensor_count,
                "shard_count": self.contract.dspark.shard_count,
                "target_quantization": "NVFP4 W4A4",
                "draft_quantization": "native MXFP4",
                "draft_stages": sorted(self.dspark_summary.mtp_counts),
            },
            "nvidia_revision": self.nvidia_revision,
        }


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as error:
        raise ContractError(f"{label} is missing: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read {label} at {path}: {error}") from error
    _expect(isinstance(value, dict), f"{label} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _shard_name(index: int, total: int) -> str:
    return f"model-{index:05d}-of-{total:05d}.safetensors"


def _parse_shard_name(name: str, label: str) -> tuple[int, int]:
    match = SHARD_RE.fullmatch(name)
    _expect(match is not None, f"{label} has invalid shard reference {name!r}")
    assert match is not None
    return int(match.group(1)), int(match.group(2))


def _index_summary(index: Mapping[str, Any], label: str) -> IndexSummary:
    metadata = index.get("metadata")
    weight_map = index.get("weight_map")
    _expect(isinstance(metadata, dict), f"{label} index metadata must be an object")
    _expect(isinstance(weight_map, dict), f"{label} weight_map must be an object")
    assert isinstance(metadata, dict) and isinstance(weight_map, dict)

    total_size = metadata.get("total_size")
    _expect(
        isinstance(total_size, int) and total_size > 0,
        f"{label} metadata.total_size must be a positive integer",
    )

    stage_counts: Counter[int] = Counter()
    non_mtp_count = 0
    shards: set[str] = set()
    for tensor_name, shard_name in weight_map.items():
        _expect(
            isinstance(tensor_name, str) and isinstance(shard_name, str),
            f"{label} weight_map keys and values must be strings",
        )
        _parse_shard_name(shard_name, label)
        shards.add(shard_name)
        match = MTP_RE.match(tensor_name)
        if match:
            stage_counts[int(match.group(1))] += 1
        else:
            _expect(
                not tensor_name.startswith("mtp."),
                f"{label} has malformed MTP tensor name {tensor_name!r}",
            )
            non_mtp_count += 1

    return IndexSummary(
        total_size=total_size,
        tensor_count=len(weight_map),
        non_mtp_count=non_mtp_count,
        mtp_counts=dict(stage_counts),
        shards=tuple(sorted(shards)),
    )


def _validate_index_contract(
    index: Mapping[str, Any],
    summary: IndexSummary,
    expected: IndexContract,
    label: str,
) -> None:
    if label == "DSpark" and set(summary.mtp_counts) == {0}:
        raise ContractError(
            "DSpark source is an NVIDIA-only one-stage artifact; expected "
            "native-MXFP4 mtp.0, mtp.1, and mtp.2 stages"
        )

    _expect(
        summary.total_size == expected.total_size,
        f"{label} total_size is {summary.total_size}, expected {expected.total_size}",
    )
    _expect(
        summary.tensor_count == expected.tensor_count,
        f"{label} tensor count is {summary.tensor_count}, expected "
        f"{expected.tensor_count}",
    )
    _expect(
        summary.non_mtp_count == expected.non_mtp_count,
        f"{label} non-MTP tensor count is {summary.non_mtp_count}, expected "
        f"{expected.non_mtp_count}",
    )
    _expect(
        dict(summary.mtp_counts) == expected.mtp_count_map,
        f"{label} MTP tensor counts are {dict(summary.mtp_counts)}, expected "
        f"{expected.mtp_count_map}",
    )
    expected_shards = {
        _shard_name(index_number, expected.shard_count)
        for index_number in range(1, expected.shard_count + 1)
    }
    _expect(
        set(summary.shards) == expected_shards,
        f"{label} index must reference exactly shards 1..{expected.shard_count}",
    )

    weight_map = index["weight_map"]
    assert isinstance(weight_map, dict)
    stage_shards = expected.stage_shard_map
    for tensor_name, shard_name in weight_map.items():
        shard_number, shard_total = _parse_shard_name(shard_name, label)
        _expect(
            shard_total == expected.shard_count,
            f"{label} shard {shard_name} has the wrong total",
        )
        match = MTP_RE.match(tensor_name)
        if match:
            stage = int(match.group(1))
            _expect(
                stage in stage_shards,
                f"{label} contains unexpected MTP stage {stage}",
            )
            _expect(
                shard_number == stage_shards[stage],
                f"{label} mtp.{stage} tensor {tensor_name} is in shard "
                f"{shard_number}, expected {stage_shards[stage]}",
            )
        else:
            _expect(
                shard_number <= expected.target_shard_count,
                f"{label} non-MTP tensor {tensor_name} appears in draft shard "
                f"{shard_number}",
            )


def _validate_common_config(
    config: Mapping[str, Any], contract: HybridContract, label: str
) -> None:
    _expect(
        config.get("architectures") == ["DeepseekV4ForCausalLM"],
        f"{label} architectures must be ['DeepseekV4ForCausalLM']",
    )
    _expect(
        config.get("model_type") == "deepseek_v4",
        f"{label} model_type must be deepseek_v4",
    )
    _expect(
        config.get("num_hidden_layers") == contract.num_hidden_layers,
        f"{label} num_hidden_layers must be {contract.num_hidden_layers}",
    )
    _expect(config.get("num_hash_layers") == 3, f"{label} num_hash_layers must be 3")
    _expect(
        config.get("num_nextn_predict_layers") == 1,
        f"{label} num_nextn_predict_layers must be 1",
    )
    _expect(config.get("expert_dtype") == "fp4", f"{label} expert_dtype must be fp4")


def _validate_nvidia_config(
    config: Mapping[str, Any], contract: HybridContract
) -> None:
    _validate_common_config(config, contract, "NVIDIA")
    _expect(
        tuple(config.get("compress_ratios", ())) == contract.nvidia_compress_ratios,
        "NVIDIA compress_ratios do not match the pinned one-stage layout",
    )
    quantization = config.get("quantization_config")
    _expect(
        isinstance(quantization, dict),
        "NVIDIA quantization_config must be an object",
    )
    assert isinstance(quantization, dict)
    _expect(
        quantization.get("moe_quant_algo") == "NVFP4",
        "NVIDIA target experts must declare moe_quant_algo=NVFP4",
    )
    _expect(
        quantization.get("quant_algo") == "MIXED_PRECISION",
        "NVIDIA checkpoint must declare quant_algo=MIXED_PRECISION",
    )
    _expect(
        quantization.get("group_size") == 16,
        "NVIDIA NVFP4 group_size must be 16",
    )
    producer = quantization.get("producer")
    _expect(
        producer == {"name": "modelopt", "version": "dsv4-nvfp4-experts"},
        "NVIDIA producer must be modelopt/dsv4-nvfp4-experts",
    )
    ignore = quantization.get("ignore")
    _expect(
        isinstance(ignore, list) and "mtp.*" in ignore,
        "NVIDIA quantization_config must explicitly ignore mtp.*",
    )
    quantized_layers = quantization.get("quantized_layers")
    _expect(
        isinstance(quantized_layers, dict),
        "NVIDIA quantized_layers must be an object",
    )
    assert isinstance(quantized_layers, dict)
    expected_layers = {
        f"layers.{layer}.ffn.experts" for layer in range(contract.num_hidden_layers)
    }
    _expect(
        set(quantized_layers) == expected_layers,
        "NVIDIA quantized_layers must cover exactly target layers 0.."
        f"{contract.num_hidden_layers - 1}",
    )
    _expect(
        all(
            value == {"group_size": 16, "quant_algo": "NVFP4"}
            for value in quantized_layers.values()
        ),
        "every NVIDIA target layer must use NVFP4 group_size 16",
    )


def _validate_dspark_config(
    config: Mapping[str, Any], contract: HybridContract
) -> None:
    _validate_common_config(config, contract, "DSpark")
    _expect(
        tuple(config.get("compress_ratios", ())) == contract.dspark_compress_ratios,
        "DSpark compress_ratios do not match the pinned three-stage layout",
    )
    for field_name, expected_value in contract.dspark_fields.items():
        _expect(
            config.get(field_name) == expected_value,
            f"DSpark {field_name} is {config.get(field_name)!r}, expected "
            f"{expected_value!r}",
        )
    quantization = config.get("quantization_config")
    _expect(
        isinstance(quantization, dict),
        "DSpark quantization_config must be an object",
    )
    assert isinstance(quantization, dict)
    _expect(
        quantization.get("moe_quant_algo") != "NVFP4"
        and "quantized_layers" not in quantization,
        "DSpark draft source must be native MXFP4, not NVIDIA NVFP4",
    )


def _read_safetensors_header(path: Path, label: str) -> SafetensorsHeader:
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            prefix = handle.read(8)
            _expect(len(prefix) == 8, f"{label} has a truncated header prefix")
            header_size = struct.unpack("<Q", prefix)[0]
            _expect(
                0 < header_size <= MAX_SAFETENSORS_HEADER_BYTES,
                f"{label} has unreasonable header size {header_size}",
            )
            _expect(
                8 + header_size <= file_size,
                f"{label} header extends past the end of the file",
            )
            header_bytes = handle.read(header_size)
            _expect(
                len(header_bytes) == header_size,
                f"{label} has a truncated JSON header",
            )
    except OSError as error:
        raise ContractError(f"cannot read {label} at {path}: {error}") from error

    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{label} contains an invalid JSON header: {error}") from error
    _expect(isinstance(header, dict), f"{label} header must be a JSON object")
    assert isinstance(header, dict)
    tensor_names: set[str] = set()
    payload_bytes = 0
    for tensor_name, tensor_metadata in header.items():
        if tensor_name == "__metadata__":
            continue
        _expect(
            isinstance(tensor_name, str) and isinstance(tensor_metadata, dict),
            f"{label} contains malformed tensor metadata",
        )
        offsets = tensor_metadata.get("data_offsets")
        _expect(
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(isinstance(value, int) for value in offsets),
            f"{label} tensor {tensor_name} has invalid data_offsets",
        )
        start, end = offsets
        _expect(
            0 <= start <= end,
            f"{label} tensor {tensor_name} has invalid data_offsets {offsets}",
        )
        payload_bytes = max(payload_bytes, end)
        tensor_names.add(tensor_name)
    _expect(tensor_names, f"{label} contains no tensors")
    available_payload = file_size - 8 - header_size
    _expect(
        payload_bytes <= available_payload,
        f"{label} data_offsets exceed the file payload",
    )
    return SafetensorsHeader(frozenset(tensor_names), payload_bytes)


def _validate_stage_files(
    directory: Path,
    index: Mapping[str, Any],
    contract: IndexContract,
    label: str,
) -> dict[int, int]:
    weight_map = index["weight_map"]
    assert isinstance(weight_map, dict)
    payloads: dict[int, int] = {}
    for stage, shard_number in sorted(contract.stage_shard_map.items()):
        shard_name = _shard_name(shard_number, contract.shard_count)
        expected_names = {
            tensor_name
            for tensor_name, referenced_shard in weight_map.items()
            if referenced_shard == shard_name
        }
        expected_names = {
            tensor_name
            for tensor_name in expected_names
            if MTP_RE.match(tensor_name)
            and int(MTP_RE.match(tensor_name).group(1)) == stage  # type: ignore[union-attr]
        }
        header = _read_safetensors_header(
            directory / shard_name, f"{label} mtp.{stage} shard"
        )
        _expect(
            header.tensor_names == expected_names,
            f"{label} mtp.{stage} shard header names do not match its index "
            f"({len(header.tensor_names)} vs {len(expected_names)} tensors)",
        )
        payloads[stage] = header.payload_bytes
    return payloads


def _read_hf_metadata(directory: Path, filename: str) -> dict[str, str] | None:
    path = directory / ".cache" / "huggingface" / "download" / f"{filename}.metadata"
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ContractError(f"cannot read Hugging Face metadata {path}: {error}") from error
    if len(lines) < 2:
        raise ContractError(f"Hugging Face metadata {path} is malformed")
    return {"revision": lines[0], "etag": lines[1]}


def _check_source_file(path: Path, label: str) -> None:
    try:
        _expect(path.is_file(), f"{label} is missing: {path}")
        _expect(path.stat().st_size > 0, f"{label} is empty: {path}")
    except OSError as error:
        raise ContractError(f"cannot stat {label} at {path}: {error}") from error


def validate_sources(
    nvidia_dir: Path,
    dspark_dir: Path,
    contract: HybridContract = PINNED_CONTRACT,
) -> ValidatedSources:
    """Validate both inputs without reading bulk tensor payloads."""

    nvidia_dir = nvidia_dir.expanduser().resolve()
    dspark_dir = dspark_dir.expanduser().resolve()
    _expect(nvidia_dir != dspark_dir, "NVIDIA and DSpark sources must be different")

    nvidia_config = _read_json(nvidia_dir / CONFIG_NAME, "NVIDIA config")
    dspark_config = _read_json(dspark_dir / CONFIG_NAME, "DSpark config")
    nvidia_index = _read_json(nvidia_dir / INDEX_NAME, "NVIDIA index")
    dspark_index = _read_json(dspark_dir / INDEX_NAME, "DSpark index")
    _validate_nvidia_config(nvidia_config, contract)
    _validate_dspark_config(dspark_config, contract)

    nvidia_summary = _index_summary(nvidia_index, "NVIDIA")
    dspark_summary = _index_summary(dspark_index, "DSpark")
    _validate_index_contract(nvidia_index, nvidia_summary, contract.nvidia, "NVIDIA")
    _validate_index_contract(dspark_index, dspark_summary, contract.dspark, "DSpark")

    for shard_number in range(1, contract.nvidia.target_shard_count + 1):
        name = _shard_name(shard_number, contract.nvidia.shard_count)
        _check_source_file(nvidia_dir / name, f"NVIDIA target shard {shard_number}")
    for filename in RUNTIME_METADATA_FILES:
        _check_source_file(nvidia_dir / filename, f"NVIDIA runtime metadata {filename}")

    nvidia_stage_payloads = _validate_stage_files(
        nvidia_dir, nvidia_index, contract.nvidia, "NVIDIA"
    )
    dspark_stage_payloads = _validate_stage_files(
        dspark_dir, dspark_index, contract.dspark, "DSpark"
    )

    nvidia_config_sha256 = _sha256(nvidia_dir / CONFIG_NAME)
    nvidia_index_sha256 = _sha256(nvidia_dir / INDEX_NAME)
    if contract.expected_nvidia_config_sha256 is not None:
        _expect(
            nvidia_config_sha256 == contract.expected_nvidia_config_sha256,
            "NVIDIA config SHA-256 does not match the pinned artifact",
        )
    if contract.expected_nvidia_index_sha256 is not None:
        _expect(
            nvidia_index_sha256 == contract.expected_nvidia_index_sha256,
            "NVIDIA index SHA-256 does not match the pinned artifact",
        )

    revision_metadata = _read_hf_metadata(nvidia_dir, CONFIG_NAME)
    nvidia_revision = revision_metadata["revision"] if revision_metadata else None
    if nvidia_revision is not None and contract.expected_nvidia_revision is not None:
        _expect(
            nvidia_revision == contract.expected_nvidia_revision,
            f"NVIDIA source revision is {nvidia_revision}, expected "
            f"{contract.expected_nvidia_revision}",
        )
    elif contract.expected_nvidia_revision is not None:
        _expect(
            contract.expected_nvidia_config_sha256 is not None
            and contract.expected_nvidia_index_sha256 is not None,
            "NVIDIA Hub revision metadata is missing and the contract has no "
            "immutable config/index SHA-256 identity gate",
        )

    return ValidatedSources(
        nvidia_dir=nvidia_dir,
        dspark_dir=dspark_dir,
        nvidia_config=nvidia_config,
        dspark_config=dspark_config,
        nvidia_index=nvidia_index,
        dspark_index=dspark_index,
        nvidia_summary=nvidia_summary,
        dspark_summary=dspark_summary,
        nvidia_stage_payloads=nvidia_stage_payloads,
        dspark_stage_payloads=dspark_stage_payloads,
        nvidia_revision=nvidia_revision,
        contract=contract,
    )


def _merged_config(sources: ValidatedSources) -> dict[str, Any]:
    merged = copy.deepcopy(sources.nvidia_config)
    for field_name in DSPARK_CONFIG_FIELDS:
        _expect(
            field_name in sources.dspark_config,
            f"DSpark config is missing required graft field {field_name}",
        )
        merged[field_name] = copy.deepcopy(sources.dspark_config[field_name])
    _expect(
        merged["quantization_config"]
        == sources.nvidia_config["quantization_config"],
        "internal error: NVIDIA quantization_config was modified",
    )
    return merged


def _merged_index(sources: ValidatedSources) -> dict[str, Any]:
    nvidia_map = sources.nvidia_index["weight_map"]
    dspark_map = sources.dspark_index["weight_map"]
    assert isinstance(nvidia_map, dict) and isinstance(dspark_map, dict)
    output_total = sources.contract.dspark.shard_count
    merged_map: dict[str, str] = {}
    for tensor_name, source_shard in nvidia_map.items():
        if MTP_RE.match(tensor_name):
            continue
        shard_number, shard_total = _parse_shard_name(source_shard, "NVIDIA")
        _expect(
            shard_total == sources.contract.nvidia.shard_count
            and shard_number <= sources.contract.nvidia.target_shard_count,
            f"NVIDIA target tensor {tensor_name} has unexpected shard {source_shard}",
        )
        merged_map[tensor_name] = _shard_name(shard_number, output_total)
    for tensor_name, source_shard in dspark_map.items():
        if not MTP_RE.match(tensor_name):
            continue
        _expect(
            tensor_name not in merged_map,
            f"duplicate tensor name while grafting DSpark: {tensor_name}",
        )
        merged_map[tensor_name] = source_shard

    _expect(
        len(merged_map) == sources.merged_tensor_count,
        f"merged tensor count is {len(merged_map)}, expected "
        f"{sources.merged_tensor_count}",
    )
    metadata = copy.deepcopy(sources.nvidia_index.get("metadata", {}))
    metadata["total_size"] = sources.merged_total_size
    return {
        "metadata": metadata,
        "weight_map": dict(sorted(merged_map.items())),
    }


def _source_shards(sources: ValidatedSources) -> list[tuple[str, str, Path]]:
    result: list[tuple[str, str, Path]] = []
    output_total = sources.contract.dspark.shard_count
    for shard_number in range(1, sources.contract.nvidia.target_shard_count + 1):
        source_name = _shard_name(shard_number, sources.contract.nvidia.shard_count)
        destination_name = _shard_name(shard_number, output_total)
        result.append((destination_name, "nvidia-target", sources.nvidia_dir / source_name))
    for stage, shard_number in sorted(sources.contract.dspark.stage_shard_map.items()):
        name = _shard_name(shard_number, output_total)
        result.append((name, f"dspark-mtp.{stage}", sources.dspark_dir / name))
    return result


def _materialize_file(source: Path, destination: Path, mode: str) -> None:
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError as error:
            raise ContractError(
                f"cannot hardlink {source} to {destination}; use symlink or copy: "
                f"{error}"
            ) from error
    elif mode == "copy":
        shutil.copyfile(source, destination)
    elif mode != "manifest":
        raise ContractError(f"unknown materialization mode {mode!r}")


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _validate_output_path(
    output: Path, sources: ValidatedSources | None, force: bool
) -> Path:
    lexical_output = Path(os.path.abspath(os.fspath(output.expanduser())))
    _expect(
        not lexical_output.is_symlink(),
        f"refusing symbolic-link output path: {lexical_output}",
    )
    resolved_output = lexical_output.resolve(strict=False)
    _expect(
        resolved_output != Path(resolved_output.anchor),
        "refusing to use a filesystem root as output",
    )
    if sources is not None:
        for source in (sources.nvidia_dir, sources.dspark_dir):
            _expect(
                not _paths_overlap(resolved_output, source),
                f"output {lexical_output} must not overlap source {source}",
            )
    if lexical_output.exists() and not lexical_output.is_dir():
        raise ContractError(f"output must be a directory path, not {lexical_output}")
    if lexical_output.is_dir() and any(lexical_output.iterdir()) and not force:
        raise ContractError(
            f"output directory is nonempty: {lexical_output}; pass --force to replace it"
        )
    return lexical_output


def _publish_staging(staging: Path, output: Path, force: bool) -> None:
    if not output.exists():
        os.replace(staging, output)
        return
    _expect(output.is_dir() and not output.is_symlink(), f"unsafe output path {output}")
    nonempty = any(output.iterdir())
    _expect(
        force or not nonempty,
        f"output directory became nonempty: {output}; rerun with --force",
    )
    if not nonempty:
        output.rmdir()
        os.replace(staging, output)
        return

    backup = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.backup-", dir=str(output.parent))
    )
    backup.rmdir()
    os.replace(output, backup)
    try:
        os.replace(staging, output)
    except BaseException:
        os.replace(backup, output)
        raise
    shutil.rmtree(backup)


def _build_provenance(
    sources: ValidatedSources,
    staging: Path,
    mode: str,
    hash_shards: bool,
) -> dict[str, Any]:
    shard_records: list[dict[str, Any]] = []
    for destination_name, role, source_path in _source_shards(sources):
        hf_metadata = (
            _read_hf_metadata(sources.nvidia_dir, source_path.name)
            if role == "nvidia-target"
            else None
        )
        if (
            hf_metadata
            and sources.contract.expected_nvidia_revision
            and hf_metadata["revision"] != sources.contract.expected_nvidia_revision
        ):
            raise ContractError(
                f"NVIDIA shard {source_path.name} revision is "
                f"{hf_metadata['revision']}, expected "
                f"{sources.contract.expected_nvidia_revision}"
            )
        shard_records.append(
            {
                "destination": destination_name,
                "role": role,
                "source_name": source_path.name,
                "source_size": source_path.stat().st_size,
                "sha256": _sha256(source_path) if hash_shards else None,
                "hub_etag": hf_metadata["etag"] if hf_metadata else None,
                "hub_revision": hf_metadata["revision"] if hf_metadata else None,
            }
        )

    artifact_names = [CONFIG_NAME, INDEX_NAME, *RUNTIME_METADATA_FILES]
    artifacts = {
        name: {"sha256": _sha256(staging / name), "size": (staging / name).stat().st_size}
        for name in sorted(artifact_names)
    }
    return {
        "schema": "anemll.hybrid-checkpoint-provenance.v1",
        "builder": "scripts/build_hybrid_nvfp4_dspark_checkpoint.py",
        "materialization": mode,
        "runnable_view": mode != "manifest",
        "shard_sha256_computed": hash_shards,
        "sources": {
            "nvidia": {
                "path": str(sources.nvidia_dir),
                "expected_revision": sources.contract.expected_nvidia_revision,
                "observed_revision": sources.nvidia_revision,
                "config_sha256": _sha256(sources.nvidia_dir / CONFIG_NAME),
                "index_sha256": _sha256(sources.nvidia_dir / INDEX_NAME),
                "contract": sources.nvidia_summary.as_json(),
            },
            "dspark": {
                "path": str(sources.dspark_dir),
                "config_sha256": _sha256(sources.dspark_dir / CONFIG_NAME),
                "index_sha256": _sha256(sources.dspark_dir / INDEX_NAME),
                "contract": sources.dspark_summary.as_json(),
            },
        },
        "merged": {
            "artifacts": artifacts,
            "config_base": "nvidia",
            "copied_dspark_config_fields": list(DSPARK_CONFIG_FIELDS),
            "draft_stages": sorted(sources.dspark_summary.mtp_counts),
            "index_total_size": sources.merged_total_size,
            "tensor_count": sources.merged_tensor_count,
            "shard_count": sources.contract.dspark.shard_count,
            "target_layers": [0, sources.contract.num_hidden_layers - 1],
            "target_quantization": "NVFP4 W4A4",
            "draft_quantization": "native MXFP4",
        },
        "shards": shard_records,
    }


def build_hybrid_view(
    sources: ValidatedSources,
    output: Path,
    mode: str = "symlink",
    force: bool = False,
    hash_shards: bool = False,
) -> dict[str, Any]:
    """Build the hybrid metadata and optional shard view transactionally."""

    output = _validate_output_path(output, sources, force)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    try:
        _write_json(staging / CONFIG_NAME, _merged_config(sources))
        _write_json(staging / INDEX_NAME, _merged_index(sources))
        for filename in RUNTIME_METADATA_FILES:
            shutil.copyfile(sources.nvidia_dir / filename, staging / filename)
        if mode != "manifest":
            for destination_name, _role, source_path in _source_shards(sources):
                _materialize_file(source_path, staging / destination_name, mode)
        provenance = _build_provenance(sources, staging, mode, hash_shards)
        _write_json(staging / PROVENANCE_NAME, provenance)
        _publish_staging(staging, output, force)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        **sources.summary_json(),
        "output": str(output),
        "materialization": mode,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or validate the pinned NVIDIA NVFP4 target + native-MXFP4 "
            "three-stage DSpark checkpoint view."
        )
    )
    parser.add_argument("--nvidia-dir", type=Path, required=True)
    parser.add_argument("--dspark-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--materialize",
        choices=("symlink", "hardlink", "copy", "manifest"),
        default="symlink",
        help=(
            "shard materialization (default: symlink); manifest writes only "
            "metadata and is not directly runnable"
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate source metadata and stage shard headers without writing output",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="explicitly replace a nonempty output directory",
    )
    parser.add_argument(
        "--hash-shards",
        action="store_true",
        help="read every selected shard and add SHA-256 values (slow for full model)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.validate_only:
        if args.output is not None:
            parser.error("--output cannot be used with --validate-only")
        if args.force:
            parser.error("--force cannot be used with --validate-only")
        if args.hash_shards:
            parser.error("--hash-shards requires an output manifest")
    elif args.output is None:
        parser.error("--output is required unless --validate-only is used")

    try:
        if not args.validate_only:
            _validate_output_path(args.output, None, args.force)
        sources = validate_sources(args.nvidia_dir, args.dspark_dir)
        if args.validate_only:
            result = sources.summary_json()
        else:
            result = build_hybrid_view(
                sources,
                args.output,
                mode=args.materialize,
                force=args.force,
                hash_shards=args.hash_shards,
            )
    except (ContractError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
