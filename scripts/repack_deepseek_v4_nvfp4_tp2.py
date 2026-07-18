#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Repack DeepSeek V4 ModelOpt NVFP4 experts into TP=2 layer layouts.

This is an offline format converter, not a generic safetensors optimizer.  It
accepts only a digest-pinned DeepSeek V4 NVFP4 checkpoint and writes a custom,
deliberately incompatible namespace.  A stock vLLM loader cannot silently use
the result: the future loader must opt into ``LOADER_CONTRACT`` and verify the
sidecar manifest before reading any payload.

Each routed layer becomes one physical safetensors file.  Its 3,072 NVIDIA
per-expert tensors are fused into eight payload families:

* rank-major W13 and W2 packed weights;
* rank-major W13 and W2 block scales;
* shared W13 and W2 global scales;
* shared W13 and W2 activation input scales.

W13 is stored in raw checkpoint order ``[w1/gate, w3/up]``.  This is important:
the serving CUTLASS path performs the one authoritative gate/up swap during
post-load processing.  Repacking must not apply that swap a second time.

The default ``build`` command preserves this raw v1 behavior.  The separate
``build-prepared`` command creates an incompatible CUTLASS-ready format.  It
applies a pinned, byte-exact NumPy implementation of the audited vLLM
preparation semantics once per layer/rank while offline, stores the final W13
order and swizzled block scales, and replaces the raw ModelOpt scale inputs with
the four final kernel scale tensors.  The matching serving loader therefore
performs exactly eight bulk H2D copies and no reorder, reduction, swizzle, or
scale algebra at model-load time.

All non-routed-expert tensors retain their original names, dtype, shape, and
payload bytes.  Raw v1 conversion streams source ranges without importing
torch.  Prepared conversion bounds host residency to one complete layer,
publishes each completed file atomically into a resumable partial directory,
and only renames the complete verified directory to its final name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


INDEX_NAME = "model.safetensors.index.json"
MANIFEST_NAME = "dspark-nvfp4-tp2-repack.json"
MANIFEST_DIGEST_NAME = f"{MANIFEST_NAME}.sha256"
NAMESPACE = "__dspark_tp2_nvfp4_v1__"
SCHEMA = "dspark.deepseek_v4.nvfp4.tp2_layer_fused.v1"
LOADER_CONTRACT = "deepseek_v4_nvfp4_tp2_layer_fused_v1"
PAYLOAD_STAGE = "raw_modelopt_tp2_fused"
REQUIRED_BACKEND = "FLASHINFER_CUTLASS"
# These pins define the post-load transformations expected by the raw format.
# A dependency update must produce a new schema after an equivalence audit; it
# must not silently reinterpret an existing repack.
VLLM_LAYOUT_PIN = "752a3a504485790a2e8491cacbb35c137339ad34"
FLASHINFER_LAYOUT_PIN = "0472b9b3f2fba11b463f8526f390297d52a8aad7"
RESERVED_PREPARED_STAGE = "flashinfer_cutlass_prepared_v1"
PREPARED_NAMESPACE = "__dspark_tp2_nvfp4_cutlass_v1__"
PREPARED_SCHEMA = "dspark.deepseek_v4.nvfp4.tp2_cutlass_prepared.v1"
PREPARED_LOADER_CONTRACT = "deepseek_v4_nvfp4_tp2_cutlass_prepared_v1"
PREPARED_PAYLOAD_STAGE = RESERVED_PREPARED_STAGE
PREPARED_STATE_NAME = ".dspark-nvfp4-prepared-build-state.json"
PREPARED_SCHEMA_VERSION = 1
PREPARED_ENGINE = "cpu_numpy_exact_v1"
PINNED_PREPARATION_SOURCE_SHA256 = {
    "flashinfer_fp4_moe": "7a98da73bebad0168fbb19ecd96232d4bed0c3586af882a6409e8dabb4b60b9d",
    "nvfp4_oracle": "746e6a5569696fe07329e13aeb397ae2152453d6a972640c7e7cf29efd173350",
    "modelopt": "e39a867fdbefd46ad25a51dace9c294c2c0b079206f285eb08e092aefc0d77d5",
    "flashinfer_experts": "d90f5215a6972c742be60ff8e9786432ab544570273483daa8faf317ba2d3ab5",
    "nvfp4_utils": "ed665537e42580e82ae71bb4f2ce8a699c0ffe8a042947c4eb600107c0b924ba",
}
LAYER0_RANK0_REFERENCE_JSON_SHA256 = (
    "b393a257791c2964d29c6762ad27658ab34b1a4de71d0b9a06a60974a0686ba6"
)
LAYER0_RANK0_REFERENCE_FINGERPRINTS = {
    "w13.weight": "f02bb1c5778d151fbc210d57fe14c232a3dcb5b3ef213366a466ddf8ce875e55",
    "w2.weight": "24b3299b6f60cd66f9b9209294503d7d8c18f5565790c52b1f2b5012270b586d",
    "w13.weight_scale": "b76d300f85af4d71e6df85b2910cd1f6da319c97d7875d022c08c62e67ae8a0a",
    "w2.weight_scale": "4a2c3041d99597af244183af181d3ca99edf738df4563f2c06a3b65fa67b7156",
    "a1_gscale": "ca75efb8ecb87d5b545fb8e0acdbe4db69c2683b34073738ed3132c7fed8755a",
    "a2_gscale": "4fe0db93441a8df2f59f74d196fdf5511db258c02dababb9bd9cd95a0b6f2887",
    "g1_alphas": "42fd9084021adf41c496f6696244ac24c853d114ece6143660b428a5c4a1e193",
    "g2_alphas": "e67aa7dc665954c5a922e9a74d645ef903240c935e9bc2b09f37770ac5ed0615",
}
TP_SIZE = 2
MAX_HEADER_BYTES = 128 * 1024 * 1024
DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024

EXPERT_RE = re.compile(
    r"^(?P<root>(?:model\.)?layers)\.(?P<layer>[0-9]+)\.ffn\.experts\."
    r"(?P<expert>[0-9]+)\.(?P<projection>w[123])\."
    r"(?P<suffix>weight|weight_scale|weight_scale_2|input_scale)$"
)
LAYER_RE = re.compile(r"^(?:model\.)?layers\.(?P<layer>[0-9]+)\.")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

FAMILY_ORDER = (
    "w13.weight",
    "w2.weight",
    "w13.weight_scale",
    "w2.weight_scale",
    "w13.weight_scale_2",
    "w2.weight_scale_2",
    "w13.input_scale",
    "w2.input_scale",
)

PREPARED_FAMILY_ORDER = (
    "w13.weight",
    "w2.weight",
    "w13.weight_scale",
    "w2.weight_scale",
    "a1_gscale",
    "a2_gscale",
    "g1_alphas",
    "g2_alphas",
)

PREPARED_DTYPES = {
    "w13.weight": "U8",
    "w2.weight": "U8",
    "w13.weight_scale": "F8_E4M3",
    "w2.weight_scale": "F8_E4M3",
    "a1_gscale": "F32",
    "a2_gscale": "F32",
    "g1_alphas": "F32",
    "g2_alphas": "F32",
}

PREPARED_SOURCE_DTYPES = {
    "w13.weight": "U8",
    "w2.weight": "U8",
    "w13.weight_scale": "F8_E4M3",
    "w2.weight_scale": "F8_E4M3",
    "w13.weight_scale_2": "F32",
    "w2.weight_scale_2": "F32",
    "w13.input_scale": "F32",
    "w2.input_scale": "F32",
}


class ContractError(ValueError):
    """The source or repacked checkpoint violates the reviewed contract."""


@dataclass(frozen=True)
class TensorRecord:
    name: str
    dtype: str
    shape: tuple[int, ...]
    shard_name: str
    shard_path: Path
    payload_start: int
    byte_length: int

    @property
    def numel(self) -> int:
        return math.prod(self.shape) if self.shape else 1

    @property
    def element_bytes(self) -> int:
        if self.numel <= 0 or self.byte_length % self.numel:
            raise ContractError(
                f"{self.name} has non-integral storage width: "
                f"shape={self.shape}, bytes={self.byte_length}"
            )
        return self.byte_length // self.numel


@dataclass(frozen=True)
class ContiguousPiece:
    tensor: TensorRecord
    relative_offset: int
    byte_length: int


@dataclass(frozen=True)
class StridedRowsPiece:
    tensor: TensorRecord
    rows: int
    source_row_bytes: int
    column_offset: int
    column_bytes: int

    @property
    def byte_length(self) -> int:
        return self.rows * self.column_bytes


Piece = ContiguousPiece | StridedRowsPiece


@dataclass(frozen=True)
class OutputTensorPlan:
    name: str
    dtype: str
    shape: tuple[int, ...]
    pieces: tuple[Piece, ...]
    source_names: tuple[str, ...]
    kind: str
    layer: int | None
    family: str | None

    @property
    def byte_length(self) -> int:
        return sum(piece.byte_length for piece in self.pieces)


@dataclass(frozen=True)
class SourceCatalog:
    config: Mapping[str, Any]
    index: Mapping[str, Any]
    tensors: Mapping[str, TensorRecord]
    shards: tuple[Path, ...]
    config_sha256: str
    index_sha256: str
    shard_stats: Mapping[str, tuple[int, int, int]]


class SourceReader:
    """Small descriptor cache with exact pread semantics."""

    def __init__(self) -> None:
        self._descriptors: dict[Path, int] = {}

    def close(self) -> None:
        for descriptor in self._descriptors.values():
            os.close(descriptor)
        self._descriptors.clear()

    def __enter__(self) -> SourceReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def read_exact(self, path: Path, offset: int, byte_length: int) -> bytes:
        descriptor = self._descriptors.get(path)
        if descriptor is None:
            descriptor = os.open(path, os.O_RDONLY)
            self._descriptors[path] = descriptor
        data = os.pread(descriptor, byte_length, offset)
        if len(data) != byte_length:
            raise OSError(
                f"short read from {path.name}: requested {byte_length} bytes "
                f"at {offset}, received {len(data)}"
            )
        return data


def sha256_file(path: Path, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_sha256(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if SHA256_RE.fullmatch(normalized) is None:
        raise ContractError(f"{label} must be an exact 64-character SHA-256")
    return normalized


def parse_git_revision(value: str, label: str = "source revision") -> str:
    normalized = value.strip().lower()
    if GIT_REVISION_RE.fullmatch(normalized) is None:
        raise ContractError(f"{label} must be an exact 40-character git SHA")
    return normalized


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{label} must contain a JSON object")
    return value


def _safe_shard_name(value: Any) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ContractError(f"unsafe safetensors shard name in index: {value!r}")
    if not value.endswith(".safetensors"):
        raise ContractError(f"indexed payload is not safetensors: {value!r}")
    return value


def _read_safetensors_header(
    path: Path,
) -> tuple[dict[str, Any], int, dict[str, tuple[int, int, int]]]:
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise ContractError(f"truncated safetensors prefix: {path}")
            (header_length,) = struct.unpack("<Q", raw_length)
            if not 2 <= header_length <= MAX_HEADER_BYTES:
                raise ContractError(
                    f"unsafe safetensors header length {header_length}: {path}"
                )
            raw_header = handle.read(header_length)
            if len(raw_header) != header_length:
                raise ContractError(f"truncated safetensors header: {path}")
    except OSError as error:
        raise ContractError(
            f"cannot inspect safetensors shard {path}: {error}"
        ) from error

    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid safetensors header in {path}: {error}") from error
    if not isinstance(header, dict):
        raise ContractError(f"safetensors header must be an object: {path}")

    payload_base = 8 + header_length
    ranges: list[tuple[int, int, str]] = []
    parsed: dict[str, tuple[int, int, int]] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            if not isinstance(entry, dict):
                raise ContractError(f"invalid __metadata__ in {path}")
            continue
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ContractError(f"invalid tensor header entry in {path}")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if not isinstance(dtype, str):
            raise ContractError(f"{name} has no dtype in {path}")
        if (
            not isinstance(shape, list)
            or any(not isinstance(item, int) or item < 0 for item in shape)
        ):
            raise ContractError(f"{name} has invalid shape in {path}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(item, int) for item in offsets)
        ):
            raise ContractError(f"{name} has invalid data_offsets in {path}")
        start, end = offsets
        if start < 0 or end <= start:
            raise ContractError(f"{name} has empty/invalid payload range in {path}")
        parsed[name] = (payload_base + start, end - start, len(shape))
        ranges.append((start, end, name))

    ranges.sort()
    cursor = 0
    for start, end, name in ranges:
        if start != cursor:
            raise ContractError(
                f"safetensors payload ranges are not contiguous before {name} in {path}"
            )
        cursor = end
    if payload_base + cursor != file_size:
        raise ContractError(
            f"safetensors file size does not match header ranges: {path}"
        )
    return header, payload_base, parsed


def inspect_source(
    source: Path,
    *,
    expected_config_sha256: str,
    expected_index_sha256: str,
) -> SourceCatalog:
    source = source.resolve()
    config_path = source / "config.json"
    index_path = source / INDEX_NAME
    if not source.is_dir() or not config_path.is_file() or not index_path.is_file():
        raise ContractError(
            f"source must contain config.json and {INDEX_NAME}: {source}"
        )

    config_digest = sha256_file(config_path)
    index_digest = sha256_file(index_path)
    if config_digest != parse_sha256(expected_config_sha256, "config digest"):
        raise ContractError(
            f"config digest mismatch: observed {config_digest}, "
            f"expected {expected_config_sha256}"
        )
    if index_digest != parse_sha256(expected_index_sha256, "index digest"):
        raise ContractError(
            f"index digest mismatch: observed {index_digest}, "
            f"expected {expected_index_sha256}"
        )

    config = _read_json(config_path, "config")
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise ContractError("config.text_config must be an object when present")
    quant = text_config.get("quantization_config")
    if str(text_config.get("model_type")) != "deepseek_v4":
        raise ContractError("source config must declare model_type=deepseek_v4")
    if str(text_config.get("expert_dtype", "")).lower() != "fp4":
        raise ContractError("source config must declare expert_dtype=fp4")
    if not isinstance(quant, dict):
        raise ContractError("source config lacks quantization_config")
    if str(quant.get("moe_quant_algo", "")).upper() != "NVFP4":
        raise ContractError("source must declare moe_quant_algo=NVFP4")
    if int(quant.get("group_size", 0)) != 16:
        raise ContractError("source must declare NVFP4 group_size=16")

    index = _read_json(index_path, "checkpoint index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ContractError("checkpoint index weight_map must be non-empty")
    if any(not isinstance(name, str) for name in weight_map):
        raise ContractError("checkpoint index tensor names must be strings")

    shard_names = tuple(
        sorted({_safe_shard_name(value) for value in weight_map.values()})
    )
    actual_shards = tuple(sorted(path.name for path in source.glob("*.safetensors")))
    if actual_shards != shard_names:
        raise ContractError(
            "indexed safetensors shards do not exactly match physical shards: "
            f"index={shard_names}, files={actual_shards}"
        )

    tensors: dict[str, TensorRecord] = {}
    shard_stats: dict[str, tuple[int, int, int]] = {}
    for shard_name in shard_names:
        shard_path = source / shard_name
        if not shard_path.is_file():
            raise ContractError(f"indexed shard is missing: {shard_path}")
        stat = shard_path.stat()
        shard_stats[shard_name] = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        header, _, parsed = _read_safetensors_header(shard_path)
        for name, (payload_start, byte_length, _) in parsed.items():
            if name in tensors:
                raise ContractError(f"tensor appears in more than one shard: {name}")
            entry = header[name]
            tensors[name] = TensorRecord(
                name=name,
                dtype=str(entry["dtype"]),
                shape=tuple(int(item) for item in entry["shape"]),
                shard_name=shard_name,
                shard_path=shard_path,
                payload_start=payload_start,
                byte_length=byte_length,
            )

    indexed_names = set(weight_map)
    physical_names = set(tensors)
    if indexed_names != physical_names:
        missing = sorted(indexed_names - physical_names)
        extra = sorted(physical_names - indexed_names)
        raise ContractError(
            "index/header tensor sets differ: "
            f"missing={missing[:4]}, extra={extra[:4]}"
        )
    for name, shard_name in weight_map.items():
        if tensors[name].shard_name != shard_name:
            raise ContractError(
                f"index maps {name} to {shard_name}, header found it in "
                f"{tensors[name].shard_name}"
            )

    total_payload = sum(tensor.byte_length for tensor in tensors.values())
    metadata = index.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ContractError("checkpoint index metadata must be an object")
    declared_total = metadata.get("total_size")
    if not isinstance(declared_total, int) or declared_total != total_payload:
        raise ContractError(
            "index total_size mismatch: "
            f"declared={declared_total}, actual={total_payload}"
        )

    return SourceCatalog(
        config=config,
        index=index,
        tensors=tensors,
        shards=tuple(source / name for name in shard_names),
        config_sha256=config_digest,
        index_sha256=index_digest,
        shard_stats=shard_stats,
    )


def _config_int(config: Mapping[str, Any], name: str) -> int:
    text_config = config.get("text_config", config)
    if not isinstance(text_config, Mapping):
        raise ContractError("config.text_config is not an object")
    value = text_config.get(name)
    if not isinstance(value, int) or value <= 0:
        raise ContractError(f"config {name} must be a positive integer")
    return value


def _expect_tensor(
    records: Mapping[tuple[int, str, str], TensorRecord],
    *,
    expert: int,
    projection: str,
    suffix: str,
    shape: tuple[int, ...],
    element_bytes: int,
) -> TensorRecord:
    key = (expert, projection, suffix)
    tensor = records.get(key)
    if tensor is None:
        raise ContractError(f"missing expert tensor {key}")
    if tensor.shape != shape:
        raise ContractError(
            f"{tensor.name} shape {tensor.shape} does not match required {shape}"
        )
    if tensor.element_bytes != element_bytes:
        raise ContractError(
            f"{tensor.name} uses {tensor.element_bytes} storage bytes/element, "
            f"expected {element_bytes}"
        )
    return tensor


def _contiguous_rows(
    tensor: TensorRecord, row_start: int, row_count: int
) -> ContiguousPiece:
    if len(tensor.shape) != 2:
        raise ContractError(f"row slice requires rank-2 tensor: {tensor.name}")
    row_bytes = tensor.byte_length // tensor.shape[0]
    return ContiguousPiece(
        tensor=tensor,
        relative_offset=row_start * row_bytes,
        byte_length=row_count * row_bytes,
    )


def _column_slice(
    tensor: TensorRecord, column_start: int, column_count: int
) -> StridedRowsPiece:
    if len(tensor.shape) != 2:
        raise ContractError(f"column slice requires rank-2 tensor: {tensor.name}")
    element_bytes = tensor.element_bytes
    return StridedRowsPiece(
        tensor=tensor,
        rows=tensor.shape[0],
        source_row_bytes=tensor.shape[1] * element_bytes,
        column_offset=column_start * element_bytes,
        column_bytes=column_count * element_bytes,
    )


def _direct_piece(tensor: TensorRecord) -> ContiguousPiece:
    return ContiguousPiece(
        tensor=tensor, relative_offset=0, byte_length=tensor.byte_length
    )


def _make_plan(
    *,
    name: str,
    dtype: str,
    shape: tuple[int, ...],
    pieces: Iterable[Piece],
    kind: str,
    layer: int | None,
    family: str | None,
) -> OutputTensorPlan:
    pieces_tuple = tuple(pieces)
    source_names = tuple(piece.tensor.name for piece in pieces_tuple)
    plan = OutputTensorPlan(
        name=name,
        dtype=dtype,
        shape=shape,
        pieces=pieces_tuple,
        source_names=source_names,
        kind=kind,
        layer=layer,
        family=family,
    )
    if not pieces_tuple or plan.byte_length <= 0:
        raise ContractError(f"output tensor {name} has no payload")
    if kind != "bitwise_nonexpert":
        widths = {piece.tensor.element_bytes for piece in pieces_tuple}
        if len(widths) != 1:
            raise ContractError(f"output tensor {name} mixes storage widths")
        expected_bytes = math.prod(shape) * next(iter(widths))
        if plan.byte_length != expected_bytes:
            raise ContractError(
                f"output tensor {name} shape implies {expected_bytes} bytes, "
                f"pieces provide {plan.byte_length}"
            )
    return plan


def plan_repack(
    catalog: SourceCatalog, namespace: str
) -> dict[int | None, list[OutputTensorPlan]]:
    if namespace != NAMESPACE:
        raise ContractError(
            f"custom namespace must be exactly {NAMESPACE!r}; got {namespace!r}"
        )
    hidden = _config_int(catalog.config, "hidden_size")
    intermediate = _config_int(catalog.config, "moe_intermediate_size")
    experts = _config_int(catalog.config, "n_routed_experts")
    layers = _config_int(catalog.config, "num_hidden_layers")
    if hidden % 16 or intermediate % (16 * TP_SIZE):
        raise ContractError(
            "hidden_size must be divisible by 16 and moe_intermediate_size "
            f"by {16 * TP_SIZE}"
        )

    by_layer: dict[int, dict[tuple[int, str, str], TensorRecord]] = {
        layer: {} for layer in range(layers)
    }
    nonexpert_by_layer: dict[int | None, list[TensorRecord]] = {
        layer: [] for layer in range(layers)
    }
    nonexpert_by_layer[None] = []
    expert_names: set[str] = set()
    roots: set[str] = set()

    for name, tensor in catalog.tensors.items():
        match = EXPERT_RE.fullmatch(name)
        if match is not None:
            layer = int(match.group("layer"))
            expert = int(match.group("expert"))
            if layer not in by_layer:
                raise ContractError(
                    f"expert tensor uses out-of-range layer {layer}: {name}"
                )
            if not 0 <= expert < experts:
                raise ContractError(f"expert id is out of range in {name}")
            key = (expert, match.group("projection"), match.group("suffix"))
            if key in by_layer[layer]:
                raise ContractError(f"duplicate expert tensor semantic key {key}")
            by_layer[layer][key] = tensor
            expert_names.add(name)
            roots.add(match.group("root"))
            continue
        layer_match = LAYER_RE.match(name)
        if layer_match is None:
            nonexpert_by_layer[None].append(tensor)
        else:
            layer = int(layer_match.group("layer"))
            if layer not in by_layer:
                raise ContractError(f"tensor uses out-of-range layer {layer}: {name}")
            nonexpert_by_layer[layer].append(tensor)

    if len(roots) != 1:
        raise ContractError(
            f"expert tensor roots must be uniform, observed {sorted(roots)}"
        )
    expected_expert_tensors = layers * experts * 3 * 4
    if len(expert_names) != expected_expert_tensors:
        raise ContractError(
            f"expected {expected_expert_tensors} routed expert tensors, "
            f"observed {len(expert_names)}"
        )

    planned: dict[int | None, list[OutputTensorPlan]] = {}
    for layer in range(layers):
        records = by_layer[layer]
        w1: dict[tuple[int, str], TensorRecord] = {}
        w3: dict[tuple[int, str], TensorRecord] = {}
        w2: dict[tuple[int, str], TensorRecord] = {}
        for expert in range(experts):
            for projection, destination in (("w1", w1), ("w3", w3), ("w2", w2)):
                weight_shape = (
                    (intermediate, hidden // 2)
                    if projection != "w2"
                    else (hidden, intermediate // 2)
                )
                scale_shape = (
                    (intermediate, hidden // 16)
                    if projection != "w2"
                    else (hidden, intermediate // 16)
                )
                destination[(expert, "weight")] = _expect_tensor(
                    records,
                    expert=expert,
                    projection=projection,
                    suffix="weight",
                    shape=weight_shape,
                    element_bytes=1,
                )
                destination[(expert, "weight_scale")] = _expect_tensor(
                    records,
                    expert=expert,
                    projection=projection,
                    suffix="weight_scale",
                    shape=scale_shape,
                    element_bytes=1,
                )
                for suffix in ("weight_scale_2", "input_scale"):
                    tensor = records.get((expert, projection, suffix))
                    if tensor is None:
                        raise ContractError(
                            f"missing expert tensor {(expert, projection, suffix)}"
                        )
                    if tensor.numel != 1:
                        raise ContractError(f"{tensor.name} must contain one scalar")
                    destination[(expert, suffix)] = tensor

        for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
            dtypes = {
                table[(expert, suffix)].dtype
                for table in (w1, w3, w2)
                for expert in range(experts)
            }
            widths = {
                table[(expert, suffix)].element_bytes
                for table in (w1, w3, w2)
                for expert in range(experts)
            }
            if len(dtypes) != 1 or len(widths) != 1:
                raise ContractError(
                    f"layer {layer} {suffix} dtypes/storage widths are not uniform: "
                    f"dtypes={sorted(dtypes)}, widths={sorted(widths)}"
                )

        prefix = f"{namespace}.layers.{layer}.experts"
        layer_plans: list[OutputTensorPlan] = []

        w13_weight_pieces: list[Piece] = []
        w13_scale_pieces: list[Piece] = []
        w2_weight_pieces: list[Piece] = []
        w2_scale_pieces: list[Piece] = []
        rank_intermediate = intermediate // TP_SIZE
        rank_packed_intermediate = (intermediate // 2) // TP_SIZE
        rank_scale_intermediate = (intermediate // 16) // TP_SIZE
        for rank in range(TP_SIZE):
            for expert in range(experts):
                # Raw W13 order is deliberately w1/gate then w3/up.
                for table in (w1, w3):
                    w13_weight_pieces.append(
                        _contiguous_rows(
                            table[(expert, "weight")],
                            rank * rank_intermediate,
                            rank_intermediate,
                        )
                    )
                    w13_scale_pieces.append(
                        _contiguous_rows(
                            table[(expert, "weight_scale")],
                            rank * rank_intermediate,
                            rank_intermediate,
                        )
                    )
                w2_weight_pieces.append(
                    _column_slice(
                        w2[(expert, "weight")],
                        rank * rank_packed_intermediate,
                        rank_packed_intermediate,
                    )
                )
                w2_scale_pieces.append(
                    _column_slice(
                        w2[(expert, "weight_scale")],
                        rank * rank_scale_intermediate,
                        rank_scale_intermediate,
                    )
                )

        scalar_w13_scale = [
            _direct_piece(table[(expert, "weight_scale_2")])
            for expert in range(experts)
            for table in (w1, w3)
        ]
        scalar_w2_scale = [
            _direct_piece(w2[(expert, "weight_scale_2")])
            for expert in range(experts)
        ]
        scalar_w13_input = [
            _direct_piece(table[(expert, "input_scale")])
            for expert in range(experts)
            for table in (w1, w3)
        ]
        scalar_w2_input = [
            _direct_piece(w2[(expert, "input_scale")])
            for expert in range(experts)
        ]

        layer_plans.extend(
            (
                _make_plan(
                    name=f"{prefix}.w13.weight",
                    dtype=w1[(0, "weight")].dtype,
                    shape=(
                        TP_SIZE,
                        experts,
                        2 * rank_intermediate,
                        hidden // 2,
                    ),
                    pieces=w13_weight_pieces,
                    kind="tp2_rank_major_w13",
                    layer=layer,
                    family="w13.weight",
                ),
                _make_plan(
                    name=f"{prefix}.w2.weight",
                    dtype=w2[(0, "weight")].dtype,
                    shape=(TP_SIZE, experts, hidden, rank_packed_intermediate),
                    pieces=w2_weight_pieces,
                    kind="tp2_rank_major_w2",
                    layer=layer,
                    family="w2.weight",
                ),
                _make_plan(
                    name=f"{prefix}.w13.weight_scale",
                    dtype=w1[(0, "weight_scale")].dtype,
                    shape=(
                        TP_SIZE,
                        experts,
                        2 * rank_intermediate,
                        hidden // 16,
                    ),
                    pieces=w13_scale_pieces,
                    kind="tp2_rank_major_w13",
                    layer=layer,
                    family="w13.weight_scale",
                ),
                _make_plan(
                    name=f"{prefix}.w2.weight_scale",
                    dtype=w2[(0, "weight_scale")].dtype,
                    shape=(TP_SIZE, experts, hidden, rank_scale_intermediate),
                    pieces=w2_scale_pieces,
                    kind="tp2_rank_major_w2",
                    layer=layer,
                    family="w2.weight_scale",
                ),
                _make_plan(
                    name=f"{prefix}.w13.weight_scale_2",
                    dtype=w1[(0, "weight_scale_2")].dtype,
                    shape=(experts, 2),
                    pieces=scalar_w13_scale,
                    kind="shared_w13_scalars",
                    layer=layer,
                    family="w13.weight_scale_2",
                ),
                _make_plan(
                    name=f"{prefix}.w2.weight_scale_2",
                    dtype=w2[(0, "weight_scale_2")].dtype,
                    shape=(experts,),
                    pieces=scalar_w2_scale,
                    kind="shared_w2_scalars",
                    layer=layer,
                    family="w2.weight_scale_2",
                ),
                _make_plan(
                    name=f"{prefix}.w13.input_scale",
                    dtype=w1[(0, "input_scale")].dtype,
                    shape=(experts, 2),
                    pieces=scalar_w13_input,
                    kind="shared_w13_scalars",
                    layer=layer,
                    family="w13.input_scale",
                ),
                _make_plan(
                    name=f"{prefix}.w2.input_scale",
                    dtype=w2[(0, "input_scale")].dtype,
                    shape=(experts,),
                    pieces=scalar_w2_input,
                    kind="shared_w2_scalars",
                    layer=layer,
                    family="w2.input_scale",
                ),
            )
        )
        if tuple(plan.family for plan in layer_plans) != FAMILY_ORDER:
            raise AssertionError("internal fused-family order drifted")

        for tensor in sorted(nonexpert_by_layer[layer], key=lambda item: item.name):
            layer_plans.append(
                _make_plan(
                    name=tensor.name,
                    dtype=tensor.dtype,
                    shape=tensor.shape,
                    pieces=(_direct_piece(tensor),),
                    kind="bitwise_nonexpert",
                    layer=layer,
                    family=None,
                )
            )
        planned[layer] = layer_plans

    residual = []
    for tensor in sorted(nonexpert_by_layer[None], key=lambda item: item.name):
        residual.append(
            _make_plan(
                name=tensor.name,
                dtype=tensor.dtype,
                shape=tensor.shape,
                pieces=(_direct_piece(tensor),),
                kind="bitwise_nonexpert",
                layer=None,
                family=None,
            )
        )
    if residual:
        planned[None] = residual

    input_bytes = sum(tensor.byte_length for tensor in catalog.tensors.values())
    output_bytes = sum(
        plan.byte_length for plans in planned.values() for plan in plans
    )
    if output_bytes != input_bytes:
        raise ContractError(
            f"repack must preserve payload bytes exactly: input={input_bytes}, "
            f"output={output_bytes}"
        )
    return planned


def _safetensors_header(
    plans: Sequence[OutputTensorPlan],
    *,
    layer: int | None,
    contract_metadata: Mapping[str, str] | None = None,
) -> tuple[bytes, dict[str, tuple[int, int]]]:
    metadata = (
        dict(contract_metadata)
        if contract_metadata is not None
        else {
            "format": "pt",
            "dspark_schema": SCHEMA,
            "dspark_loader_contract": LOADER_CONTRACT,
            "dspark_namespace": NAMESPACE,
            "dspark_payload_stage": PAYLOAD_STAGE,
            "dspark_required_backend": REQUIRED_BACKEND,
            "dspark_standard_loader_compatible": "false",
            "dspark_layer": "residual" if layer is None else str(layer),
        }
    )
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()):
        raise ContractError("safetensors contract metadata must contain strings")
    header: dict[str, Any] = {"__metadata__": metadata}
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for plan in plans:
        if plan.name in header:
            raise ContractError(f"duplicate output tensor name: {plan.name}")
        end = cursor + plan.byte_length
        header[plan.name] = {
            "dtype": plan.dtype,
            "shape": list(plan.shape),
            "data_offsets": [cursor, end],
        }
        offsets[plan.name] = (cursor, end)
        cursor = end
    raw = json.dumps(
        header, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    padding = (-len(raw)) % 8
    raw += b" " * padding
    if len(raw) > MAX_HEADER_BYTES:
        raise ContractError(f"output safetensors header is too large: {len(raw)}")
    return struct.pack("<Q", len(raw)) + raw, offsets


def _iter_piece_chunks(
    reader: SourceReader,
    piece: Piece,
    *,
    chunk_bytes: int,
) -> Iterator[bytes]:
    if isinstance(piece, ContiguousPiece):
        remaining = piece.byte_length
        offset = piece.tensor.payload_start + piece.relative_offset
        while remaining:
            size = min(remaining, chunk_bytes)
            yield reader.read_exact(piece.tensor.shard_path, offset, size)
            offset += size
            remaining -= size
        return

    rows_per_block = max(1, chunk_bytes // piece.source_row_bytes)
    for row_start in range(0, piece.rows, rows_per_block):
        row_count = min(rows_per_block, piece.rows - row_start)
        source_offset = piece.tensor.payload_start + row_start * piece.source_row_bytes
        source = reader.read_exact(
            piece.tensor.shard_path,
            source_offset,
            row_count * piece.source_row_bytes,
        )
        output = bytearray(row_count * piece.column_bytes)
        for relative_row in range(row_count):
            source_start = (
                relative_row * piece.source_row_bytes + piece.column_offset
            )
            output_start = relative_row * piece.column_bytes
            output[output_start : output_start + piece.column_bytes] = source[
                source_start : source_start + piece.column_bytes
            ]
        yield bytes(output)


def _write_safetensors(
    path: Path,
    plans: Sequence[OutputTensorPlan],
    *,
    layer: int | None,
    chunk_bytes: int,
    contract_metadata: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    prefix, _ = _safetensors_header(
        plans, layer=layer, contract_metadata=contract_metadata
    )
    file_digest = hashlib.sha256()
    tensor_rows: list[dict[str, Any]] = []
    with SourceReader() as reader, path.open("xb") as output:
        output.write(prefix)
        file_digest.update(prefix)
        for plan in plans:
            tensor_digest = hashlib.sha256()
            written = 0
            for piece in plan.pieces:
                for chunk in _iter_piece_chunks(
                    reader, piece, chunk_bytes=chunk_bytes
                ):
                    output.write(chunk)
                    file_digest.update(chunk)
                    tensor_digest.update(chunk)
                    written += len(chunk)
            if written != plan.byte_length:
                raise OSError(
                    f"short output tensor write for {plan.name}: "
                    f"{written} != {plan.byte_length}"
                )
            source_semantics = [
                {
                    "name": piece.tensor.name,
                    "dtype": piece.tensor.dtype,
                    "shape": list(piece.tensor.shape),
                }
                for piece in plan.pieces
            ]
            tensor_rows.append(
                {
                    "name": plan.name,
                    "dtype": plan.dtype,
                    "shape": list(plan.shape),
                    "bytes": plan.byte_length,
                    "sha256": tensor_digest.hexdigest(),
                    "kind": plan.kind,
                    "family": plan.family,
                    "source_piece_count": len(plan.pieces),
                    "source_names_sha256": canonical_sha256(plan.source_names),
                    "source_semantics_sha256": canonical_sha256(source_semantics),
                    "first_source_name": plan.source_names[0],
                    "last_source_name": plan.source_names[-1],
                    "source_payload_sha256": (
                        tensor_digest.hexdigest()
                        if plan.kind == "bitwise_nonexpert"
                        else None
                    ),
                }
            )
        output.flush()
        os.fsync(output.fileno())
    return {
        "path": path.name,
        "layer": layer,
        "size": path.stat().st_size,
        "sha256": file_digest.hexdigest(),
        "payload_bytes": sum(plan.byte_length for plan in plans),
        "tensor_count": len(plans),
        "tensors": tensor_rows,
    }


def _copy_metadata_files(
    source: Path, destination: Path, source_shard_names: set[str]
) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    reserved = {
        INDEX_NAME,
        MANIFEST_NAME,
        MANIFEST_DIGEST_NAME,
    } | source_shard_names
    for child in sorted(source.iterdir(), key=lambda path: path.name):
        if child.name in reserved or child.is_dir():
            continue
        if not child.is_file():
            raise ContractError(f"unsupported checkpoint metadata entry: {child}")
        target = destination / child.name
        shutil.copyfile(child, target)
        digest = sha256_file(target)
        if digest != sha256_file(child):
            raise OSError(f"metadata copy digest mismatch: {child.name}")
        copied.append(
            {"path": child.name, "size": target.stat().st_size, "sha256": digest}
        )
    if not (destination / "config.json").is_file():
        raise ContractError("config.json was not copied")
    return copied


def _output_filename(layer: int | None) -> str:
    return (
        "model-nonlayer.safetensors"
        if layer is None
        else f"model-layer-{layer:05d}.safetensors"
    )


def _source_stats_unchanged(source: Path, catalog: SourceCatalog) -> None:
    for shard_name, expected in catalog.shard_stats.items():
        stat = (source / shard_name).stat()
        observed = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if observed != expected:
            raise ContractError(
                f"source shard changed during repack: {shard_name}; "
                f"before={expected}, after={observed}"
            )


def _loader_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "loader_contract": LOADER_CONTRACT,
        "namespace": NAMESPACE,
        "payload_stage": PAYLOAD_STAGE,
        "required_backend": REQUIRED_BACKEND,
        "vllm_layout_pin": VLLM_LAYOUT_PIN,
        "flashinfer_layout_pin": FLASHINFER_LAYOUT_PIN,
        "standard_vllm_compatible": False,
        "fail_closed_without_exact_loader_contract": True,
        "cutlass_serving_layout_ready": False,
        "tp_size": TP_SIZE,
        "matrix_rank_axis": 0,
        "matrix_expert_axis": 1,
        "scalar_families_shared_across_tp_ranks": True,
        "w13_raw_projection_order": ["w1", "w3"],
        "w13_semantic_projection_order": ["gate", "up"],
        "cutlass_final_projection_order": ["w3", "w1"],
        "serving_postload_swap_count": 1,
        "required_runtime_transforms": [
            "reduce_checkpoint_input_scales",
            "combine_weight_scale_2_with_input_scale",
            "swizzle_block_scales_for_flashinfer_cutlass",
            "reorder_w13_once_from_w1_w3_to_w3_w1",
        ],
        "reserved_payload_stages": {
            RESERVED_PREPARED_STAGE: {
                "implemented": False,
                "required_backend": REQUIRED_BACKEND,
                "requires_exact_vllm_and_flashinfer_layout_pins": True,
                "required_final_projection_order": ["w3", "w1"],
                "required_runtime_w13_reorder_count": 0,
                "required_proof": (
                    "bitwise-or-numeric equivalence against freshly prepared "
                    "real checkpoint tensors"
                ),
            }
        },
        "families": list(FAMILY_ORDER),
        "num_hidden_layers": _config_int(config, "num_hidden_layers"),
        "n_routed_experts": _config_int(config, "n_routed_experts"),
    }


def build_repacked_checkpoint(
    source: Path,
    output: Path,
    *,
    namespace: str,
    expected_config_sha256: str,
    expected_index_sha256: str,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> dict[str, Any]:
    if chunk_bytes <= 0:
        raise ContractError("chunk_bytes must be positive")
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise ContractError(f"output must not already exist: {output}")
    if source == output or source in output.parents:
        raise ContractError("output must not be the source or a child of the source")
    output.parent.mkdir(parents=True, exist_ok=True)

    catalog = inspect_source(
        source,
        expected_config_sha256=expected_config_sha256,
        expected_index_sha256=expected_index_sha256,
    )
    plans_by_layer = plan_repack(catalog, namespace)
    payload_bytes = sum(
        plan.byte_length for plans in plans_by_layer.values() for plan in plans
    )
    free_bytes = shutil.disk_usage(output.parent).free
    required_free = payload_bytes + (1 << 30)
    if free_bytes < required_free:
        raise ContractError(
            f"insufficient free space: free={free_bytes}, required={required_free}"
        )

    temp_path = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent)
    )
    published = False
    try:
        source_shard_digests = [
            {
                "path": shard.name,
                "size": shard.stat().st_size,
                "sha256": sha256_file(shard, chunk_bytes=chunk_bytes),
            }
            for shard in catalog.shards
        ]
        copied_metadata = _copy_metadata_files(
            source, temp_path, {path.name for path in catalog.shards}
        )

        output_files: list[dict[str, Any]] = []
        weight_map: dict[str, str] = {}
        for layer in sorted(
            plans_by_layer, key=lambda value: (-1 if value is None else value)
        ):
            plans = plans_by_layer[layer]
            filename = _output_filename(layer)
            result = _write_safetensors(
                temp_path / filename,
                plans,
                layer=layer,
                chunk_bytes=chunk_bytes,
            )
            output_files.append(result)
            for plan in plans:
                if plan.name in weight_map:
                    raise ContractError(f"duplicate output index key: {plan.name}")
                weight_map[plan.name] = filename

        index = {
            "metadata": {
                "total_size": payload_bytes,
                "dspark_schema": SCHEMA,
                "dspark_loader_contract": LOADER_CONTRACT,
                "dspark_namespace": NAMESPACE,
                "dspark_payload_stage": PAYLOAD_STAGE,
                "dspark_required_backend": REQUIRED_BACKEND,
                "dspark_vllm_layout_pin": VLLM_LAYOUT_PIN,
                "dspark_flashinfer_layout_pin": FLASHINFER_LAYOUT_PIN,
                "dspark_standard_loader_compatible": False,
                "source_index_sha256": catalog.index_sha256,
            },
            "weight_map": weight_map,
        }
        index_path = temp_path / INDEX_NAME
        index_path.write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        output_index_sha256 = sha256_file(index_path)

        source_payload_bytes = sum(
            tensor.byte_length for tensor in catalog.tensors.values()
        )
        manifest = {
            "schema_version": 1,
            "format": SCHEMA,
            "loader": _loader_contract(catalog.config),
            "source": {
                "checkpoint_name": source.name,
                "config_sha256": catalog.config_sha256,
                "index_sha256": catalog.index_sha256,
                "indexed_tensor_count": len(catalog.tensors),
                "payload_bytes": source_payload_bytes,
                "shards": source_shard_digests,
            },
            "output": {
                "index_sha256": output_index_sha256,
                "payload_bytes": payload_bytes,
                "payload_bytes_preserved": payload_bytes == source_payload_bytes,
                "tensor_count": len(weight_map),
                "layer_file_count": _config_int(
                    catalog.config, "num_hidden_layers"
                ),
                "files": output_files,
                "copied_metadata_files": copied_metadata,
            },
            "integrity": {
                "source_shards_hashed": True,
                "output_files_hashed": True,
                "output_tensors_hashed": True,
                "nonexpert_payloads_bitwise_copied": True,
                "atomic_directory_publication": True,
            },
        }
        manifest_path = temp_path / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest_digest = sha256_file(manifest_path)
        (temp_path / MANIFEST_DIGEST_NAME).write_text(
            f"{manifest_digest}  {MANIFEST_NAME}\n", encoding="ascii"
        )

        _source_stats_unchanged(source, catalog)
        verification = verify_repacked_checkpoint(temp_path)
        if not verification["ok"]:
            raise ContractError("internal output verification did not pass")
        os.rename(temp_path, output)
        published = True
        return {
            "ok": True,
            "output": str(output),
            "manifest_sha256": manifest_digest,
            "output_index_sha256": output_index_sha256,
            "source_payload_bytes": source_payload_bytes,
            "output_payload_bytes": payload_bytes,
            "layer_file_count": manifest["output"]["layer_file_count"],
            "output_tensor_count": len(weight_map),
            "loader_contract": LOADER_CONTRACT,
            "namespace": NAMESPACE,
        }
    finally:
        if not published and temp_path.exists():
            shutil.rmtree(temp_path)


def _tensor_payload_digest(path: Path, start: int, byte_length: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = byte_length
        while remaining:
            chunk = handle.read(min(remaining, DEFAULT_CHUNK_BYTES))
            if not chunk:
                raise ContractError(f"truncated tensor payload in {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def verify_repacked_checkpoint(output: Path) -> dict[str, Any]:
    output = output.resolve()
    manifest_path = output / MANIFEST_NAME
    digest_path = output / MANIFEST_DIGEST_NAME
    if not manifest_path.is_file() or not digest_path.is_file():
        raise ContractError("repacked checkpoint manifest/digest sidecar is missing")
    words = digest_path.read_text(encoding="ascii").strip().split()
    if len(words) != 2 or words[1] != MANIFEST_NAME:
        raise ContractError("manifest digest sidecar has an invalid contract")
    expected_manifest_digest = parse_sha256(words[0], "manifest digest")
    observed_manifest_digest = sha256_file(manifest_path)
    if observed_manifest_digest != expected_manifest_digest:
        raise ContractError("manifest digest mismatch")
    manifest = _read_json(manifest_path, "repack manifest")
    loader = manifest.get("loader")
    if not isinstance(loader, dict) or loader != _loader_contract(
        _read_json(output / "config.json", "output config")
    ):
        raise ContractError("manifest does not declare the exact loader contract")
    if loader.get("standard_vllm_compatible") is not False:
        raise ContractError("repack must remain incompatible with a standard loader")

    output_section = manifest.get("output")
    if not isinstance(output_section, dict):
        raise ContractError("manifest output section is missing")
    files = output_section.get("files")
    if not isinstance(files, list) or not files:
        raise ContractError("manifest output files are missing")
    copied_metadata = output_section.get("copied_metadata_files")
    if not isinstance(copied_metadata, list):
        raise ContractError("manifest copied metadata file list is missing")
    copied_names: set[str] = set()
    for row in copied_metadata:
        if not isinstance(row, dict):
            raise ContractError("invalid copied metadata manifest row")
        filename = row.get("path")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or filename in copied_names
        ):
            raise ContractError("unsafe/duplicate copied metadata path")
        copied_names.add(filename)
        path = output / filename
        if path.stat().st_size != row.get("size") or sha256_file(path) != row.get(
            "sha256"
        ):
            raise ContractError(f"copied metadata digest mismatch: {filename}")
    source_section = manifest.get("source")
    if not isinstance(source_section, dict):
        raise ContractError("manifest source section is missing")
    if sha256_file(output / "config.json") != source_section.get("config_sha256"):
        raise ContractError("copied config digest differs from source manifest")

    index_path = output / INDEX_NAME
    if sha256_file(index_path) != output_section.get("index_sha256"):
        raise ContractError("output index digest mismatch")
    index = _read_json(index_path, "output index")
    metadata = index.get("metadata")
    weight_map = index.get("weight_map")
    if not isinstance(metadata, dict) or not isinstance(weight_map, dict):
        raise ContractError("output index is malformed")
    if (
        metadata.get("dspark_schema") != SCHEMA
        or metadata.get("dspark_loader_contract") != LOADER_CONTRACT
        or metadata.get("dspark_namespace") != NAMESPACE
        or metadata.get("dspark_payload_stage") != PAYLOAD_STAGE
        or metadata.get("dspark_required_backend") != REQUIRED_BACKEND
        or metadata.get("dspark_vllm_layout_pin") != VLLM_LAYOUT_PIN
        or metadata.get("dspark_flashinfer_layout_pin") != FLASHINFER_LAYOUT_PIN
        or metadata.get("dspark_standard_loader_compatible") is not False
    ):
        raise ContractError("output index loader contract is missing or altered")

    seen_names: set[str] = set()
    payload_bytes = 0
    manifest_file_names: set[str] = set()
    for row in files:
        if not isinstance(row, dict):
            raise ContractError("invalid output file manifest entry")
        filename = row.get("path")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ContractError("unsafe output file path in manifest")
        path = output / filename
        if filename in manifest_file_names:
            raise ContractError(f"duplicate output file manifest entry: {filename}")
        manifest_file_names.add(filename)
        if path.stat().st_size != row.get("size"):
            raise ContractError(f"output file size mismatch: {filename}")
        if sha256_file(path) != row.get("sha256"):
            raise ContractError(f"output file digest mismatch: {filename}")
        header, _, parsed = _read_safetensors_header(path)
        header_metadata = header.get("__metadata__")
        if not isinstance(header_metadata, dict) or (
            header_metadata.get("dspark_schema") != SCHEMA
            or header_metadata.get("dspark_loader_contract") != LOADER_CONTRACT
            or header_metadata.get("dspark_namespace") != NAMESPACE
            or header_metadata.get("dspark_payload_stage") != PAYLOAD_STAGE
            or header_metadata.get("dspark_required_backend") != REQUIRED_BACKEND
            or header_metadata.get("dspark_standard_loader_compatible") != "false"
        ):
            raise ContractError(f"safetensors loader contract mismatch: {filename}")
        tensor_rows = row.get("tensors")
        if not isinstance(tensor_rows, list) or len(tensor_rows) != len(parsed):
            raise ContractError(f"tensor manifest count mismatch: {filename}")
        for tensor_row in tensor_rows:
            if not isinstance(tensor_row, dict):
                raise ContractError(f"invalid tensor manifest row: {filename}")
            name = tensor_row.get("name")
            if not isinstance(name, str) or name in seen_names or name not in parsed:
                raise ContractError(f"invalid/duplicate output tensor: {name!r}")
            seen_names.add(name)
            start, byte_length, _ = parsed[name]
            entry = header[name]
            if (
                entry.get("dtype") != tensor_row.get("dtype")
                or entry.get("shape") != tensor_row.get("shape")
                or byte_length != tensor_row.get("bytes")
            ):
                raise ContractError(f"tensor metadata mismatch: {name}")
            if _tensor_payload_digest(path, start, byte_length) != tensor_row.get(
                "sha256"
            ):
                raise ContractError(f"tensor payload digest mismatch: {name}")
            if tensor_row.get("kind") == "bitwise_nonexpert" and tensor_row.get(
                "source_payload_sha256"
            ) != tensor_row.get("sha256"):
                raise ContractError(f"nonexpert bitwise digest proof failed: {name}")
            if weight_map.get(name) != filename:
                raise ContractError(f"output index maps {name} to the wrong file")
            payload_bytes += byte_length

    if seen_names != set(weight_map):
        raise ContractError("manifest and output index tensor sets differ")
    if payload_bytes != output_section.get("payload_bytes"):
        raise ContractError("verified output payload byte count differs from manifest")
    if payload_bytes != metadata.get("total_size"):
        raise ContractError("verified output payload byte count differs from index")
    if output_section.get("payload_bytes_preserved") is not True:
        raise ContractError("manifest does not prove payload byte preservation")
    if payload_bytes != source_section.get("payload_bytes"):
        raise ContractError("source/output payload byte counts differ")
    physical_safetensors = {
        path.name for path in output.glob("*.safetensors") if path.is_file()
    }
    if physical_safetensors != manifest_file_names:
        raise ContractError("physical and manifested output safetensors files differ")

    layer_count = int(loader["num_hidden_layers"])
    output_config = _read_json(output / "config.json", "output config")
    hidden = _config_int(output_config, "hidden_size")
    intermediate = _config_int(
        output_config, "moe_intermediate_size"
    )
    experts = int(loader["n_routed_experts"])
    expected_shapes = {
        "w13.weight": [TP_SIZE, experts, intermediate, hidden // 2],
        "w2.weight": [
            TP_SIZE,
            experts,
            hidden,
            (intermediate // 2) // TP_SIZE,
        ],
        "w13.weight_scale": [TP_SIZE, experts, intermediate, hidden // 16],
        "w2.weight_scale": [
            TP_SIZE,
            experts,
            hidden,
            (intermediate // 16) // TP_SIZE,
        ],
        "w13.weight_scale_2": [experts, 2],
        "w2.weight_scale_2": [experts],
        "w13.input_scale": [experts, 2],
        "w2.input_scale": [experts],
    }
    for layer in range(layer_count):
        prefix = f"{NAMESPACE}.layers.{layer}.experts."
        observed = tuple(
            name[len(prefix) :]
            for name in weight_map
            if name.startswith(prefix)
        )
        if set(observed) != set(FAMILY_ORDER) or len(observed) != len(FAMILY_ORDER):
            raise ContractError(
                f"layer {layer} does not contain exactly eight fused families"
            )
        layer_filename = _output_filename(layer)
        for family in FAMILY_ORDER:
            name = f"{prefix}{family}"
            if weight_map.get(name) != layer_filename:
                raise ContractError(
                    f"layer {layer} fused payload is not in its one physical file"
                )
            file_row = next(row for row in files if row["path"] == layer_filename)
            tensor_row = next(
                row for row in file_row["tensors"] if row["name"] == name
            )
            if tensor_row.get("shape") != expected_shapes[family]:
                raise ContractError(f"fused tensor shape contract mismatch: {name}")
    if any(EXPERT_RE.fullmatch(name) is not None for name in weight_map):
        raise ContractError("original per-expert keys remain in repacked index")

    return {
        "ok": True,
        "manifest_sha256": observed_manifest_digest,
        "file_count": len(files),
        "tensor_count": len(seen_names),
        "payload_bytes": payload_bytes,
        "loader_contract": LOADER_CONTRACT,
        "namespace": NAMESPACE,
    }


def _prepared_shapes(config: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """Return the exact rank-major SM121 CUTLASS payload shapes."""

    hidden = _config_int(config, "hidden_size")
    intermediate = _config_int(config, "moe_intermediate_size")
    experts = _config_int(config, "n_routed_experts")
    if hidden % 16 or intermediate % (16 * TP_SIZE):
        raise ContractError("prepared layout dimensions violate NVFP4 TP=2")
    return {
        "w13.weight": (TP_SIZE, experts, intermediate, hidden // 2),
        "w2.weight": (
            TP_SIZE,
            experts,
            hidden,
            (intermediate // 2) // TP_SIZE,
        ),
        "w13.weight_scale": (
            TP_SIZE,
            experts,
            intermediate,
            hidden // 16,
        ),
        "w2.weight_scale": (
            TP_SIZE,
            experts,
            hidden,
            (intermediate // 16) // TP_SIZE,
        ),
        "a1_gscale": (TP_SIZE, experts),
        "a2_gscale": (TP_SIZE, experts),
        "g1_alphas": (TP_SIZE, experts),
        "g2_alphas": (TP_SIZE, experts),
    }


def _prepared_header_metadata(layer: int | None) -> dict[str, str]:
    return {
        "format": "pt",
        "dspark_schema": PREPARED_SCHEMA,
        "dspark_loader_contract": PREPARED_LOADER_CONTRACT,
        "dspark_namespace": PREPARED_NAMESPACE,
        "dspark_payload_stage": PREPARED_PAYLOAD_STAGE,
        "dspark_required_backend": REQUIRED_BACKEND,
        "dspark_standard_loader_compatible": "false",
        "dspark_layer": "residual" if layer is None else str(layer),
    }


def _validate_preparation_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    required_strings = (
        "implementation",
        "engine",
        "backend",
        "vllm_layout_pin",
        "flashinfer_layout_pin",
        "numpy_version",
        "transform_spec_sha256",
        "repacker_script_path",
        "repacker_script_sha256",
        "source_revision",
    )
    result = dict(value)
    for name in required_strings:
        if not isinstance(result.get(name), str) or not result[name]:
            raise ContractError(f"prepared identity field {name!r} is missing")
    if result["implementation"] != (
        "scripts.repack_deepseek_v4_nvfp4_tp2._cpu_prepare_rank"
    ) or result["engine"] != PREPARED_ENGINE:
        raise ContractError("prepared identity implementation drifted")
    if result["backend"] != REQUIRED_BACKEND:
        raise ContractError("prepared identity backend drifted")
    if result["vllm_layout_pin"] != VLLM_LAYOUT_PIN:
        raise ContractError("prepared identity vLLM pin drifted")
    if result["flashinfer_layout_pin"] != FLASHINFER_LAYOUT_PIN:
        raise ContractError("prepared identity FlashInfer pin drifted")
    result["transform_spec_sha256"] = parse_sha256(
        result["transform_spec_sha256"], "transform spec digest"
    )
    result["repacker_script_sha256"] = parse_sha256(
        result["repacker_script_sha256"], "repacker script digest"
    )
    result["source_revision"] = parse_git_revision(result["source_revision"])
    source_hashes = result.get("pinned_preparation_source_sha256")
    if source_hashes != PINNED_PREPARATION_SOURCE_SHA256:
        raise ContractError("prepared identity pinned source hashes drifted")
    if result.get("is_act_and_mul") is not True:
        raise ContractError("prepared identity must pin is_act_and_mul=true")
    return result


def _prepared_loader_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": PREPARED_SCHEMA,
        "loader_contract": PREPARED_LOADER_CONTRACT,
        "namespace": PREPARED_NAMESPACE,
        "payload_stage": PREPARED_PAYLOAD_STAGE,
        "required_backend": REQUIRED_BACKEND,
        "vllm_layout_pin": VLLM_LAYOUT_PIN,
        "flashinfer_layout_pin": FLASHINFER_LAYOUT_PIN,
        "standard_vllm_compatible": False,
        "fail_closed_without_exact_loader_contract": True,
        "cutlass_serving_layout_ready": True,
        "tp_size": TP_SIZE,
        "matrix_rank_axis": 0,
        "matrix_expert_axis": 1,
        "families": list(PREPARED_FAMILY_ORDER),
        "w13_final_projection_order": ["w3", "w1"],
        "required_runtime_transforms": [],
        "runtime_h2d_calls_per_layer": len(PREPARED_FAMILY_ORDER),
        "runtime_source_reads_per_layer": len(PREPARED_FAMILY_ORDER),
        "scalar_rank_copies_required_bitwise_equal": True,
        "final_scale_fields": {
            "a1_gscale": "1 / max(all checkpoint w1,w3 input_scale)",
            "a2_gscale": "1 / max(all checkpoint w2 input_scale)",
            "g1_alphas": "w1.weight_scale_2 * reciprocal(a1_gscale)",
            "g2_alphas": "w2.weight_scale_2 * reciprocal(a2_gscale)",
        },
        "offline_transforms": [
            "tp2_slice",
            "reorder_w13_once_from_w1_w3_to_w3_w1",
            "reduce_checkpoint_input_scales_globally",
            "swizzle_block_scales_for_flashinfer_cutlass",
            "compute_final_cutlass_global_scales_and_alphas",
        ],
        "num_hidden_layers": _config_int(config, "num_hidden_layers"),
        "n_routed_experts": _config_int(config, "n_routed_experts"),
    }


PreparedRankFn = Callable[
    [Mapping[str, Any], int, Mapping[str, Any]],
    tuple[Mapping[str, Any], Mapping[str, Any]],
]


def _cpu_transform_spec() -> dict[str, Any]:
    return {
        "engine": PREPARED_ENGINE,
        "source_layout": "raw ModelOpt TP=2 rank slice [w1,w3]",
        "final_w13_order": ["w3", "w1"],
        "swizzle": {
            "reshape": ["B", "M/128", 4, 32, "K/4", 4],
            "transpose": [0, 1, 4, 3, 2, 5],
            "padding_allowed": False,
        },
        "activation_reduction": "float32 max across all experts/projections",
        "w13_scale2": "require w1/w3 bitwise equal; use w1",
        "a_gscale": "float32 reciprocal(reduced input scale)",
        "g_alpha": "float32 weight_scale_2 * reduced input scale",
        "runtime_transforms": [],
    }


def _cpu_preparation_identity(source_revision: str) -> dict[str, Any]:
    import numpy as np

    return _validate_preparation_identity(
        {
            "implementation": (
                "scripts.repack_deepseek_v4_nvfp4_tp2._cpu_prepare_rank"
            ),
            "engine": PREPARED_ENGINE,
            "backend": REQUIRED_BACKEND,
            "vllm_layout_pin": VLLM_LAYOUT_PIN,
            "flashinfer_layout_pin": FLASHINFER_LAYOUT_PIN,
            "numpy_version": str(np.__version__),
            "transform_spec_sha256": canonical_sha256(_cpu_transform_spec()),
            "repacker_script_path": str(Path(__file__).resolve()),
            "repacker_script_sha256": sha256_file(Path(__file__).resolve()),
            "source_revision": parse_git_revision(source_revision),
            "pinned_preparation_source_sha256": dict(
                PINNED_PREPARATION_SOURCE_SHA256
            ),
            "is_act_and_mul": True,
        }
    )


def _swizzle_blockscale_bytes(array: Any) -> Any:
    """Exact CPU byte permutation of pinned vLLM swizzle_blockscale."""

    import numpy as np

    if array.dtype != np.uint8 or array.ndim != 3:
        raise ContractError("block-scale swizzle requires a rank-3 uint8 array")
    batches, rows, columns = array.shape
    if rows % 128 or columns % 4:
        raise ContractError(
            "prepared format forbids implicit block-scale padding: "
            f"shape={array.shape}"
        )
    return (
        array.reshape(batches, rows // 128, 4, 32, columns // 4, 4)
        .transpose(0, 1, 4, 3, 2, 5)
        .copy()
        .reshape(batches, rows, columns)
    )


def _numpy_bitwise_equal(left: Any, right: Any) -> bool:
    """Compare exact array storage, including signed zero and NaN payloads."""

    import numpy as np

    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if (
        left_array.shape != right_array.shape
        or left_array.dtype != right_array.dtype
    ):
        return False
    left_bytes = np.ascontiguousarray(left_array).view(np.uint8).reshape(-1)
    right_bytes = np.ascontiguousarray(right_array).view(np.uint8).reshape(-1)
    return bool(np.array_equal(left_bytes, right_bytes))


def _cpu_prepare_rank(
    raw: Mapping[str, Any],
    rank: int,
    context: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Prepare one already-TP-sliced rank with exact NumPy byte semantics."""

    import numpy as np

    if rank not in range(TP_SIZE):
        raise ContractError(f"invalid prepared TP rank: {rank}")
    if set(raw) != set(FAMILY_ORDER):
        raise ContractError("CPU prepared input family set drifted")
    w13_raw = np.asarray(raw["w13.weight"], dtype=np.uint8)
    w13_scale_raw = np.asarray(raw["w13.weight_scale"], dtype=np.uint8)
    if w13_raw.ndim != 3 or w13_raw.shape[1] % 2:
        raise ContractError("raw W13 weight shape cannot be reordered")
    if w13_scale_raw.ndim != 3 or w13_scale_raw.shape[1] % 2:
        raise ContractError("raw W13 scale shape cannot be reordered")
    weight_half = w13_raw.shape[1] // 2
    scale_half = w13_scale_raw.shape[1] // 2
    w13 = np.concatenate(
        (w13_raw[:, weight_half:], w13_raw[:, :weight_half]), axis=1
    )
    w13_scale_linear = np.concatenate(
        (
            w13_scale_raw[:, scale_half:],
            w13_scale_raw[:, :scale_half],
        ),
        axis=1,
    )
    w2 = np.ascontiguousarray(raw["w2.weight"], dtype=np.uint8)
    w2_scale_linear = np.ascontiguousarray(
        raw["w2.weight_scale"], dtype=np.uint8
    )
    w13_scale = _swizzle_blockscale_bytes(w13_scale_linear)
    w2_scale = _swizzle_blockscale_bytes(w2_scale_linear)

    w13_scale2 = np.asarray(raw["w13.weight_scale_2"], dtype=np.float32)
    w2_scale2 = np.asarray(raw["w2.weight_scale_2"], dtype=np.float32)
    w13_input = np.asarray(raw["w13.input_scale"], dtype=np.float32)
    w2_input = np.asarray(raw["w2.input_scale"], dtype=np.float32)
    if w13_scale2.ndim != 2 or w13_scale2.shape[1] != 2:
        raise ContractError("raw W13 scale_2 must have [E,2] shape")
    if w13_input.shape != w13_scale2.shape:
        raise ContractError("raw W13 input-scale shape drifted")
    if w2_scale2.ndim != 1 or w2_input.shape != w2_scale2.shape:
        raise ContractError("raw W2 scalar shape drifted")
    if not _numpy_bitwise_equal(w13_scale2[:, 0], w13_scale2[:, 1]):
        raise ContractError("checkpoint w1/w3 weight_scale_2 bytes differ")
    for name, value in (
        ("w13 input_scale", w13_input),
        ("w2 input_scale", w2_input),
        ("w13 weight_scale_2", w13_scale2),
        ("w2 weight_scale_2", w2_scale2),
    ):
        if not bool(np.isfinite(value).all()):
            raise ContractError(f"{name} contains non-finite values")
    if not bool((w13_input > 0).all()) or not bool((w2_input > 0).all()):
        raise ContractError("checkpoint input scales must be positive")
    a13 = np.float32(np.max(w13_input))
    a2 = np.float32(np.max(w2_input))
    experts = w13_scale2.shape[0]
    prepared = {
        "w13.weight": np.ascontiguousarray(w13),
        "w2.weight": w2,
        "w13.weight_scale": w13_scale,
        "w2.weight_scale": w2_scale,
        "a1_gscale": np.full(
            experts, np.float32(1.0) / a13, dtype=np.float32
        ),
        "a2_gscale": np.full(
            experts, np.float32(1.0) / a2, dtype=np.float32
        ),
        "g1_alphas": np.multiply(
            w13_scale2[:, 0], a13, dtype=np.float32
        ),
        "g2_alphas": np.multiply(w2_scale2, a2, dtype=np.float32),
    }
    proof = {
        "rank": rank,
        "engine": PREPARED_ENGINE,
        "w13_scale_2_columns_bitwise_equal": True,
        "a13_global_scale": float(a13),
        "a2_global_scale": float(a2),
        "transform_spec_sha256": canonical_sha256(_cpu_transform_spec()),
    }
    return prepared, proof


def _sample_numpy_digest(array: Any, family: str) -> str:
    """Match the immutable GPU benchmark's sampled tensor fingerprint."""

    import numpy as np

    dtype_text = {
        "w13.weight": "torch.uint8",
        "w2.weight": "torch.uint8",
        "w13.weight_scale": "torch.float8_e4m3fn",
        "w2.weight_scale": "torch.float8_e4m3fn",
        "a1_gscale": "torch.float32",
        "a2_gscale": "torch.float32",
        "g1_alphas": "torch.float32",
        "g2_alphas": "torch.float32",
    }[family]
    contiguous = np.ascontiguousarray(array)
    flat = contiguous.view(np.uint8).reshape(-1)
    sample_bytes = flat.size if family in PREPARED_FAMILY_ORDER[4:] else 4096
    offsets = sorted(
        {
            0,
            max(0, flat.size // 2 - sample_bytes // 2),
            max(0, flat.size - sample_bytes),
        }
    )
    digest = hashlib.sha256()
    digest.update(str(tuple(contiguous.shape)).encode())
    digest.update(dtype_text.encode())
    for offset in offsets:
        digest.update(flat[offset : min(flat.size, offset + sample_bytes)].tobytes())
    return digest.hexdigest()


def _numpy_safetensors_views(path: Path) -> dict[str, Any]:
    import numpy as np

    header, _, parsed = _read_safetensors_header(path)
    result: dict[str, Any] = {}
    for name, (start, _byte_length, _) in parsed.items():
        entry = header[name]
        dtype_name = str(entry["dtype"])
        if dtype_name in ("U8", "F8_E4M3"):
            dtype = np.uint8
        elif dtype_name == "F32":
            dtype = np.dtype("<f4")
        else:
            raise ContractError(
                f"unsupported CPU prepared scratch dtype {dtype_name}: {name}"
            )
        result[name] = np.memmap(
            path,
            mode="r",
            dtype=dtype,
            offset=start,
            shape=tuple(entry["shape"]),
            order="C",
        )
    return result


def _prepared_routed_filename(layer: int) -> str:
    return f"model-layer-{layer:05d}.safetensors"


def _prepared_nonexpert_filename(layer: int | None) -> str:
    return (
        "model-nonlayer.safetensors"
        if layer is None
        else f"model-layer-{layer:05d}-nonexpert.safetensors"
    )


def _atomic_write_json(path: Path, value: Any) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_safetensors(
    path: Path,
    plans: Sequence[OutputTensorPlan],
    *,
    layer: int | None,
    chunk_bytes: int,
    contract_metadata: Mapping[str, str],
) -> dict[str, Any]:
    """Publish one non-expert safetensors file with crash-safe replacement."""

    temporary = path.with_name(f".{path.name}.partial")
    if temporary.exists():
        temporary.unlink()
    try:
        row = _write_safetensors(
            temporary,
            plans,
            layer=layer,
            chunk_bytes=chunk_bytes,
            contract_metadata=contract_metadata,
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    row["path"] = path.name
    return row


def _prepared_tensor_name(layer: int, family: str) -> str:
    return f"{PREPARED_NAMESPACE}.layers.{layer}.experts.{family}"


def _physical_file_manifest(
    path: Path,
    *,
    layer: int | None,
    kinds: Mapping[str, tuple[str, str | None]],
) -> dict[str, Any]:
    header, _, parsed = _read_safetensors_header(path)
    metadata = header.get("__metadata__")
    if metadata != _prepared_header_metadata(layer):
        raise ContractError(f"prepared safetensors metadata drifted: {path.name}")
    if set(parsed) != set(kinds):
        raise ContractError(f"prepared physical tensor set drifted: {path.name}")
    rows: list[dict[str, Any]] = []
    for name in sorted(parsed):
        start, byte_length, _ = parsed[name]
        entry = header[name]
        kind, family = kinds[name]
        rows.append(
            {
                "name": name,
                "dtype": entry["dtype"],
                "shape": entry["shape"],
                "bytes": byte_length,
                "sha256": _tensor_payload_digest(path, start, byte_length),
                "kind": kind,
                "family": family,
                "source_payload_sha256": (
                    _tensor_payload_digest(path, start, byte_length)
                    if kind == "bitwise_nonexpert"
                    else None
                ),
            }
        )
    return {
        "path": path.name,
        "layer": layer,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "payload_bytes": sum(row["bytes"] for row in rows),
        "tensor_count": len(rows),
        "tensors": rows,
    }


def _write_prepared_arrays(
    path: Path,
    *,
    layer: int,
    rank_outputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Write two rank outputs without relying on float8 NumPy dtype support."""

    import numpy as np

    if len(rank_outputs) != TP_SIZE:
        raise ContractError("prepared output requires exactly two rank rows")
    descriptors: list[tuple[str, str, tuple[int, ...], int, str]] = []
    for family in PREPARED_FAMILY_ORDER:
        first = np.asarray(rank_outputs[0][family])
        descriptors.append(
            (
                _prepared_tensor_name(layer, family),
                PREPARED_DTYPES[family],
                (TP_SIZE, *first.shape),
                TP_SIZE * int(first.nbytes),
                family,
            )
        )
    header: dict[str, Any] = {"__metadata__": _prepared_header_metadata(layer)}
    cursor = 0
    for name, dtype, shape, byte_length, _family in descriptors:
        end = cursor + byte_length
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [cursor, end],
        }
        cursor = end
    raw_header = json.dumps(
        header, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    raw_header += b" " * ((-len(raw_header)) % 8)
    if len(raw_header) > MAX_HEADER_BYTES:
        raise ContractError("prepared safetensors header is too large")
    prefix = struct.pack("<Q", len(raw_header)) + raw_header
    file_digest = hashlib.sha256(prefix)
    tensor_rows: list[dict[str, Any]] = []
    with path.open("xb") as handle:
        handle.write(prefix)
        for name, dtype, shape, byte_length, family in descriptors:
            tensor_digest = hashlib.sha256()
            written = 0
            for rank in range(TP_SIZE):
                array = np.asarray(rank_outputs[rank][family])
                view = memoryview(array).cast("B")
                for offset in range(0, len(view), DEFAULT_CHUNK_BYTES):
                    chunk = view[offset : offset + DEFAULT_CHUNK_BYTES]
                    handle.write(chunk)
                    file_digest.update(chunk)
                    tensor_digest.update(chunk)
                    written += len(chunk)
                del view
            if written != byte_length:
                raise ContractError(f"prepared tensor write was short: {name}")
            tensor_rows.append(
                {
                    "name": name,
                    "dtype": dtype,
                    "shape": list(shape),
                    "bytes": byte_length,
                    "sha256": tensor_digest.hexdigest(),
                    "kind": "tp2_rank_major_cutlass_prepared",
                    "family": family,
                    "source_payload_sha256": None,
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": path.name,
        "layer": layer,
        "size": path.stat().st_size,
        "sha256": file_digest.hexdigest(),
        "payload_bytes": sum(row["bytes"] for row in tensor_rows),
        "tensor_count": len(tensor_rows),
        "tensors": tensor_rows,
    }


def _validate_resumable_file(root: Path, row: Mapping[str, Any]) -> bool:
    path = root / str(row.get("path"))
    return (
        path.is_file()
        and path.stat().st_size == row.get("size")
        and sha256_file(path) == row.get("sha256")
    )


def build_prepared_checkpoint(
    source: Path,
    output: Path,
    *,
    expected_config_sha256: str,
    expected_index_sha256: str,
    source_revision: str,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    stop_after_layer: int | None = None,
    prepare_rank: PreparedRankFn | None = None,
    preparation_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the resumable, final CUTLASS-ready TP=2 checkpoint."""

    if chunk_bytes <= 0:
        raise ContractError("chunk_bytes must be positive")
    if stop_after_layer not in (None, 0):
        raise ContractError(
            "prepared conversion currently supports only --stop-after-layer 0"
        )
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise ContractError(f"output must not already exist: {output}")
    if source == output or source in output.parents:
        raise ContractError("output must not be the source or a child of the source")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.prepared-partial")

    catalog = inspect_source(
        source,
        expected_config_sha256=expected_config_sha256,
        expected_index_sha256=expected_index_sha256,
    )
    plans_by_layer = plan_repack(catalog, NAMESPACE)
    shapes = _prepared_shapes(catalog.config)
    layers = _config_int(catalog.config, "num_hidden_layers")
    experts = _config_int(catalog.config, "n_routed_experts")
    source_revision = parse_git_revision(source_revision)

    if prepare_rank is None:
        identity = _cpu_preparation_identity(source_revision)
        prepare_rank = _cpu_prepare_rank
    else:
        if preparation_identity is None:
            raise ContractError("injected preparer requires an exact identity")
        identity = _validate_preparation_identity(preparation_identity)
        if identity["source_revision"] != source_revision:
            raise ContractError("injected preparer source revision drifted")

    source_stats = {
        name: list(values) for name, values in sorted(catalog.shard_stats.items())
    }
    state_contract = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "format": PREPARED_SCHEMA,
        "source_realpath": str(source),
        "source_config_sha256": catalog.config_sha256,
        "source_index_sha256": catalog.index_sha256,
        "source_shard_stats": source_stats,
        "preparation_identity": identity,
    }
    state_path = partial / PREPARED_STATE_NAME
    if partial.exists():
        if not partial.is_dir() or not state_path.is_file():
            raise ContractError(
                f"prepared partial path lacks resumable state: {partial}"
            )
        state = _read_json(state_path, "prepared build state")
        if state.get("contract") != state_contract:
            raise ContractError("prepared partial state contract drifted")
        files_by_name = state.get("files")
        if not isinstance(files_by_name, dict):
            raise ContractError("prepared partial file state is malformed")
        copied_metadata = state.get("copied_metadata_files")
        source_shard_digests = state.get("source_shards")
        if not isinstance(copied_metadata, list) or not isinstance(
            source_shard_digests, list
        ):
            raise ContractError("prepared partial integrity state is incomplete")
        for row in copied_metadata:
            if not isinstance(row, dict) or not _validate_resumable_file(partial, row):
                raise ContractError("prepared partial metadata file drifted")
        for filename, row in files_by_name.items():
            if not isinstance(row, dict) or row.get("path") != filename:
                raise ContractError("prepared partial output-file state is malformed")
            if not _validate_resumable_file(partial, row):
                raise ContractError(f"prepared partial file drifted: {filename}")
        if stop_after_layer == 0:
            later_routed = {
                _prepared_routed_filename(layer) for layer in range(1, layers)
            }
            state_later = later_routed.intersection(files_by_name)
            physical_later = later_routed.intersection(
                path.name for path in partial.glob("model-layer-*.safetensors")
            )
            if state_later or physical_later:
                raise ContractError(
                    "layer-0 gate cannot run after later routed layers were "
                    f"published: state={sorted(state_later)} "
                    f"physical={sorted(physical_later)}"
                )
    else:
        partial.mkdir()
        source_shard_digests = [
            {
                "path": shard.name,
                "size": shard.stat().st_size,
                "sha256": sha256_file(shard, chunk_bytes=chunk_bytes),
            }
            for shard in catalog.shards
        ]
        copied_metadata = _copy_metadata_files(
            source, partial, {path.name for path in catalog.shards}
        )
        output_config_path = partial / "config.json"
        output_config = _read_json(output_config_path, "copied output config")
        if "dspark_nvfp4_prepared" in output_config:
            raise ContractError("source config already declares a prepared contract")
        output_config["dspark_nvfp4_prepared"] = {
            "schema": PREPARED_SCHEMA,
            "loader_contract": PREPARED_LOADER_CONTRACT,
            "namespace": PREPARED_NAMESPACE,
            "payload_stage": PREPARED_PAYLOAD_STAGE,
            "required_backend": REQUIRED_BACKEND,
            "vllm_layout_pin": VLLM_LAYOUT_PIN,
            "flashinfer_layout_pin": FLASHINFER_LAYOUT_PIN,
            "tp_size": TP_SIZE,
            "manifest": MANIFEST_NAME,
            "manifest_digest": MANIFEST_DIGEST_NAME,
        }
        _atomic_write_json(output_config_path, output_config)
        for row in copied_metadata:
            if row["path"] == "config.json":
                row["size"] = output_config_path.stat().st_size
                row["sha256"] = sha256_file(output_config_path)
                break
        else:
            raise ContractError("prepared metadata copy omitted config.json")
        files_by_name: dict[str, Any] = {}
        state = {
            "contract": state_contract,
            "source_shards": source_shard_digests,
            "copied_metadata_files": copied_metadata,
            "files": files_by_name,
            "layer_rank_proofs": {},
        }
        _atomic_write_json(state_path, state)

    raw_source_payload = sum(tensor.byte_length for tensor in catalog.tensors.values())
    expected_prepared_payload = raw_source_payload + layers * experts * 2 * 4
    largest_raw_layer = max(
        sum(plan.byte_length for plan in plans_by_layer[layer][: len(FAMILY_ORDER)])
        for layer in range(layers)
    )
    completed_payload = sum(
        int(row.get("payload_bytes", 0)) for row in files_by_name.values()
    )
    required_free = (
        expected_prepared_payload - completed_payload + largest_raw_layer + (1 << 30)
    )
    if shutil.disk_usage(partial).free < required_free:
        raise ContractError(
            "insufficient free space for resumable prepared conversion: "
            f"required={required_free}"
        )

    import numpy as np

    expected_numpy_dtypes = {
        "w13.weight": np.dtype("uint8"),
        "w2.weight": np.dtype("uint8"),
        "w13.weight_scale": np.dtype("uint8"),
        "w2.weight_scale": np.dtype("uint8"),
        "a1_gscale": np.dtype("float32"),
        "a2_gscale": np.dtype("float32"),
        "g1_alphas": np.dtype("float32"),
        "g2_alphas": np.dtype("float32"),
    }
    scalar_families = PREPARED_FAMILY_ORDER[4:]
    for layer in range(layers):
        plans = plans_by_layer[layer]
        raw_plans = plans[: len(FAMILY_ORDER)]
        if tuple(plan.family for plan in raw_plans) != FAMILY_ORDER:
            raise ContractError(f"layer {layer} raw family plan drifted")
        for plan in raw_plans:
            assert plan.family is not None
            expected_dtype = PREPARED_SOURCE_DTYPES[plan.family]
            if plan.dtype != expected_dtype:
                raise ContractError(
                    f"prepared source dtype drifted for layer {layer} "
                    f"{plan.family}: {plan.dtype} != {expected_dtype}"
                )
        nonexpert_plans = plans[len(FAMILY_ORDER) :]

        routed_filename = _prepared_routed_filename(layer)
        if routed_filename not in files_by_name:
            scratch = partial / f".raw-layer-{layer:05d}.safetensors.partial"
            if scratch.exists():
                scratch.unlink()
            try:
                _write_safetensors(
                    scratch,
                    raw_plans,
                    layer=layer,
                    chunk_bytes=chunk_bytes,
                )
                rank_outputs: list[Mapping[str, Any]] = []
                rank_proofs: list[Mapping[str, Any]] = []
                views = _numpy_safetensors_views(scratch)
                for rank in range(TP_SIZE):
                    raw: dict[str, Any] = {}
                    for family in FAMILY_ORDER:
                        value = views[
                            f"{NAMESPACE}.layers.{layer}.experts.{family}"
                        ]
                        raw[family] = (
                            value[rank]
                            if family in FAMILY_ORDER[:4]
                            else value
                        )
                    prepared, proof = prepare_rank(
                        raw,
                        rank,
                        {
                            "layer": layer,
                            "config": catalog.config,
                            "identity": identity,
                        },
                    )
                    if tuple(prepared) != PREPARED_FAMILY_ORDER:
                        raise ContractError(
                            f"layer {layer} rank {rank} prepared family order drifted"
                        )
                    for family in PREPARED_FAMILY_ORDER:
                        tensor = np.asarray(prepared[family])
                        if tuple(tensor.shape) != shapes[family][1:]:
                            raise ContractError(
                                f"prepared shape drifted for layer {layer} rank "
                                f"{rank} {family}: {tuple(tensor.shape)}"
                            )
                        if tensor.dtype != expected_numpy_dtypes[family]:
                            raise ContractError(
                                f"prepared dtype drifted for {family}: {tensor.dtype}"
                            )
                        if not tensor.flags.c_contiguous:
                            raise ContractError(
                                f"prepared tensor {family} is not contiguous"
                            )
                    rank_outputs.append(prepared)
                    rank_proofs.append(dict(proof))
                del views
                for family in scalar_families:
                    if not _numpy_bitwise_equal(
                        rank_outputs[0][family], rank_outputs[1][family]
                    ):
                        raise ContractError(
                            f"prepared scalar rank copies differ: layer={layer} {family}"
                        )
                if (
                    layer == 0
                    and catalog.config_sha256
                    == "0c5dc7303ff322d73e0cd5caf9cc1b65d6efeff68fab53514531c2e959b1d616"
                    and catalog.index_sha256
                    == "2d83d58754cff11724f117d20d95e31803a48512d29f8e00463b2501905d6d72"
                ):
                    observed_fingerprints = {
                        family: _sample_numpy_digest(
                            rank_outputs[0][family], family
                        )
                        for family in PREPARED_FAMILY_ORDER
                    }
                    if observed_fingerprints != LAYER0_RANK0_REFERENCE_FINGERPRINTS:
                        raise ContractError(
                            "prepared layer0/rank0 fingerprints differ from the "
                            "immutable GPU reference"
                        )
                    rank_proofs[0]["immutable_reference"] = {
                        "source_json_sha256": (
                            LAYER0_RANK0_REFERENCE_JSON_SHA256
                        ),
                        "fingerprints": observed_fingerprints,
                        "passed": True,
                    }
                temporary = partial / f".{routed_filename}.partial"
                if temporary.exists():
                    temporary.unlink()
                row = _write_prepared_arrays(
                    temporary,
                    layer=layer,
                    rank_outputs=rank_outputs,
                )
                os.replace(temporary, partial / routed_filename)
                row["path"] = routed_filename
                files_by_name[routed_filename] = row
                state["layer_rank_proofs"][str(layer)] = rank_proofs
                _atomic_write_json(state_path, state)
                del rank_outputs
            finally:
                if scratch.exists():
                    scratch.unlink()

        if nonexpert_plans:
            nonexpert_filename = _prepared_nonexpert_filename(layer)
            if nonexpert_filename not in files_by_name:
                row = _atomic_write_safetensors(
                    partial / nonexpert_filename,
                    nonexpert_plans,
                    layer=layer,
                    chunk_bytes=chunk_bytes,
                    contract_metadata=_prepared_header_metadata(layer),
                )
                files_by_name[nonexpert_filename] = row
                _atomic_write_json(state_path, state)

        if stop_after_layer == layer:
            _source_stats_unchanged(source, catalog)
            routed_row = files_by_name.get(routed_filename)
            if not isinstance(routed_row, dict) or not _validate_resumable_file(
                partial, routed_row
            ):
                raise ContractError(
                    "layer-0 gate did not publish a verified routed file"
                )
            return {
                "ok": True,
                "complete": False,
                "paused_after_layer": layer,
                "resumable": True,
                "partial_checkpoint": str(partial),
                "routed_file": routed_filename,
                "routed_file_size": routed_row["size"],
                "routed_file_sha256": routed_row["sha256"],
                "build_state_sha256": sha256_file(state_path),
                "source_revision": source_revision,
                "preparation_identity": identity,
            }

    residual_plans = plans_by_layer.get(None, [])
    if residual_plans:
        residual_filename = _prepared_nonexpert_filename(None)
        if residual_filename not in files_by_name:
            row = _atomic_write_safetensors(
                partial / residual_filename,
                residual_plans,
                layer=None,
                chunk_bytes=chunk_bytes,
                contract_metadata=_prepared_header_metadata(None),
            )
            files_by_name[residual_filename] = row
            _atomic_write_json(state_path, state)

    _source_stats_unchanged(source, catalog)
    weight_map: dict[str, str] = {}
    output_files = [files_by_name[name] for name in sorted(files_by_name)]
    for row in output_files:
        for tensor_row in row["tensors"]:
            name = tensor_row["name"]
            if name in weight_map:
                raise ContractError(f"duplicate prepared output tensor: {name}")
            weight_map[name] = row["path"]
    output_payload = sum(int(row["payload_bytes"]) for row in output_files)
    if output_payload != expected_prepared_payload:
        raise ContractError(
            "prepared output payload arithmetic drifted: "
            f"observed={output_payload} expected={expected_prepared_payload}"
        )
    index = {
        "metadata": {
            "total_size": output_payload,
            "dspark_schema": PREPARED_SCHEMA,
            "dspark_loader_contract": PREPARED_LOADER_CONTRACT,
            "dspark_namespace": PREPARED_NAMESPACE,
            "dspark_payload_stage": PREPARED_PAYLOAD_STAGE,
            "dspark_required_backend": REQUIRED_BACKEND,
            "dspark_vllm_layout_pin": VLLM_LAYOUT_PIN,
            "dspark_flashinfer_layout_pin": FLASHINFER_LAYOUT_PIN,
            "dspark_standard_loader_compatible": False,
            "source_index_sha256": catalog.index_sha256,
        },
        "weight_map": weight_map,
    }
    index_path = partial / INDEX_NAME
    _atomic_write_json(index_path, index)
    output_index_sha256 = sha256_file(index_path)
    output_config_sha256 = sha256_file(partial / "config.json")
    manifest = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "format": PREPARED_SCHEMA,
        "loader": _prepared_loader_contract(catalog.config),
        "preparation": {
            "identity": identity,
            "layer_rank_proofs": state["layer_rank_proofs"],
        },
        "source": {
            "checkpoint_name": source.name,
            "config_sha256": catalog.config_sha256,
            "index_sha256": catalog.index_sha256,
            "indexed_tensor_count": len(catalog.tensors),
            "payload_bytes": raw_source_payload,
            "shards": source_shard_digests,
        },
        "output": {
            "config_sha256": output_config_sha256,
            "index_sha256": output_index_sha256,
            "payload_bytes": output_payload,
            "source_payload_delta_bytes": output_payload - raw_source_payload,
            "tensor_count": len(weight_map),
            "layer_file_count": layers,
            "files": output_files,
            "copied_metadata_files": copied_metadata,
        },
        "integrity": {
            "source_shards_hashed": True,
            "output_files_hashed": True,
            "output_tensors_hashed": True,
            "nonexpert_payloads_bitwise_copied": True,
            "scalar_rank_copies_bitwise_equal": True,
            "atomic_file_publication": True,
            "resumable_partial_directory": True,
            "atomic_directory_publication": True,
        },
    }
    manifest_path = partial / MANIFEST_NAME
    _atomic_write_json(manifest_path, manifest)
    manifest_digest = sha256_file(manifest_path)
    digest_temporary = partial / f".{MANIFEST_DIGEST_NAME}.tmp"
    digest_temporary.write_text(
        f"{manifest_digest}  {MANIFEST_NAME}\n", encoding="ascii"
    )
    os.replace(digest_temporary, partial / MANIFEST_DIGEST_NAME)
    verification = verify_prepared_checkpoint(partial, allow_build_state=True)
    if not verification["ok"]:
        raise ContractError("internal prepared verification did not pass")
    state_path.unlink()
    os.rename(partial, output)
    return {
        "ok": True,
        "output": str(output),
        "manifest_sha256": manifest_digest,
        "output_index_sha256": output_index_sha256,
        "source_payload_bytes": raw_source_payload,
        "output_payload_bytes": output_payload,
        "source_payload_delta_bytes": output_payload - raw_source_payload,
        "layer_file_count": layers,
        "output_tensor_count": len(weight_map),
        "loader_contract": PREPARED_LOADER_CONTRACT,
        "namespace": PREPARED_NAMESPACE,
    }


def verify_prepared_checkpoint(
    output: Path, *, allow_build_state: bool = False
) -> dict[str, Any]:
    """Verify every prepared payload byte and the fail-closed loader contract."""

    output = output.resolve()
    manifest_path = output / MANIFEST_NAME
    digest_path = output / MANIFEST_DIGEST_NAME
    if not manifest_path.is_file() or not digest_path.is_file():
        raise ContractError("prepared manifest/digest sidecar is missing")
    words = digest_path.read_text(encoding="ascii").strip().split()
    if len(words) != 2 or words[1] != MANIFEST_NAME:
        raise ContractError("prepared manifest digest sidecar is malformed")
    expected_manifest_digest = parse_sha256(words[0], "manifest digest")
    observed_manifest_digest = sha256_file(manifest_path)
    if observed_manifest_digest != expected_manifest_digest:
        raise ContractError("prepared manifest digest mismatch")
    manifest = _read_json(manifest_path, "prepared manifest")
    if (
        manifest.get("schema_version") != PREPARED_SCHEMA_VERSION
        or manifest.get("format") != PREPARED_SCHEMA
    ):
        raise ContractError("prepared manifest schema drifted")

    output_config = _read_json(output / "config.json", "prepared config")
    marker = output_config.get("dspark_nvfp4_prepared")
    expected_marker = {
        "schema": PREPARED_SCHEMA,
        "loader_contract": PREPARED_LOADER_CONTRACT,
        "namespace": PREPARED_NAMESPACE,
        "payload_stage": PREPARED_PAYLOAD_STAGE,
        "required_backend": REQUIRED_BACKEND,
        "vllm_layout_pin": VLLM_LAYOUT_PIN,
        "flashinfer_layout_pin": FLASHINFER_LAYOUT_PIN,
        "tp_size": TP_SIZE,
        "manifest": MANIFEST_NAME,
        "manifest_digest": MANIFEST_DIGEST_NAME,
    }
    if marker != expected_marker:
        raise ContractError("prepared config marker drifted")
    loader = manifest.get("loader")
    if loader != _prepared_loader_contract(output_config):
        raise ContractError("prepared manifest loader contract drifted")
    preparation = manifest.get("preparation")
    if not isinstance(preparation, dict):
        raise ContractError("prepared provenance is missing")
    identity = preparation.get("identity")
    if not isinstance(identity, dict):
        raise ContractError("prepared engine identity is missing")
    _validate_preparation_identity(identity)

    source_section = manifest.get("source")
    output_section = manifest.get("output")
    integrity = manifest.get("integrity")
    if not all(
        isinstance(value, dict)
        for value in (source_section, output_section, integrity)
    ):
        raise ContractError("prepared manifest sections are incomplete")
    assert isinstance(source_section, dict)
    assert isinstance(output_section, dict)
    assert isinstance(integrity, dict)
    required_integrity = {
        "source_shards_hashed": True,
        "output_files_hashed": True,
        "output_tensors_hashed": True,
        "nonexpert_payloads_bitwise_copied": True,
        "scalar_rank_copies_bitwise_equal": True,
        "atomic_file_publication": True,
        "resumable_partial_directory": True,
        "atomic_directory_publication": True,
    }
    if integrity != required_integrity:
        raise ContractError("prepared integrity contract drifted")
    if sha256_file(output / "config.json") != output_section.get("config_sha256"):
        raise ContractError("prepared output config digest mismatch")

    copied_metadata = output_section.get("copied_metadata_files")
    if not isinstance(copied_metadata, list):
        raise ContractError("prepared copied metadata list is missing")
    copied_names: set[str] = set()
    for row in copied_metadata:
        if not isinstance(row, dict):
            raise ContractError("prepared copied metadata row is malformed")
        name = row.get("path")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in copied_names
        ):
            raise ContractError("prepared copied metadata path is unsafe")
        copied_names.add(name)
        path = output / name
        if path.stat().st_size != row.get("size") or sha256_file(path) != row.get(
            "sha256"
        ):
            raise ContractError(f"prepared metadata digest mismatch: {name}")

    index_path = output / INDEX_NAME
    if sha256_file(index_path) != output_section.get("index_sha256"):
        raise ContractError("prepared output index digest mismatch")
    index = _read_json(index_path, "prepared output index")
    metadata = index.get("metadata")
    weight_map = index.get("weight_map")
    if not isinstance(metadata, dict) or not isinstance(weight_map, dict):
        raise ContractError("prepared output index is malformed")
    exact_index_contract = {
        "dspark_schema": PREPARED_SCHEMA,
        "dspark_loader_contract": PREPARED_LOADER_CONTRACT,
        "dspark_namespace": PREPARED_NAMESPACE,
        "dspark_payload_stage": PREPARED_PAYLOAD_STAGE,
        "dspark_required_backend": REQUIRED_BACKEND,
        "dspark_vllm_layout_pin": VLLM_LAYOUT_PIN,
        "dspark_flashinfer_layout_pin": FLASHINFER_LAYOUT_PIN,
        "dspark_standard_loader_compatible": False,
        "source_index_sha256": source_section.get("index_sha256"),
    }
    for key, expected in exact_index_contract.items():
        if metadata.get(key) != expected:
            raise ContractError(f"prepared index contract field drifted: {key}")

    files = output_section.get("files")
    if not isinstance(files, list) or not files:
        raise ContractError("prepared output file manifest is empty")
    seen_names: set[str] = set()
    manifest_file_names: set[str] = set()
    payload_bytes = 0
    file_rows: dict[str, Mapping[str, Any]] = {}
    for row in files:
        if not isinstance(row, dict):
            raise ContractError("prepared output file row is malformed")
        filename = row.get("path")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or filename in manifest_file_names
        ):
            raise ContractError("prepared output file path is unsafe/duplicate")
        manifest_file_names.add(filename)
        file_rows[filename] = row
        path = output / filename
        if path.stat().st_size != row.get("size") or sha256_file(path) != row.get(
            "sha256"
        ):
            raise ContractError(f"prepared output file digest mismatch: {filename}")
        header, _, parsed = _read_safetensors_header(path)
        layer_value = row.get("layer")
        if layer_value is not None and not isinstance(layer_value, int):
            raise ContractError("prepared file layer field is malformed")
        if header.get("__metadata__") != _prepared_header_metadata(layer_value):
            raise ContractError(f"prepared file contract drifted: {filename}")
        tensor_rows = row.get("tensors")
        if not isinstance(tensor_rows, list) or len(tensor_rows) != len(parsed):
            raise ContractError(f"prepared tensor manifest count drifted: {filename}")
        for tensor_row in tensor_rows:
            if not isinstance(tensor_row, dict):
                raise ContractError("prepared tensor row is malformed")
            name = tensor_row.get("name")
            if not isinstance(name, str) or name in seen_names or name not in parsed:
                raise ContractError(f"prepared tensor name is invalid: {name!r}")
            seen_names.add(name)
            start, byte_length, _ = parsed[name]
            entry = header[name]
            if (
                entry.get("dtype") != tensor_row.get("dtype")
                or entry.get("shape") != tensor_row.get("shape")
                or byte_length != tensor_row.get("bytes")
            ):
                raise ContractError(f"prepared tensor metadata drifted: {name}")
            digest = _tensor_payload_digest(path, start, byte_length)
            if digest != tensor_row.get("sha256"):
                raise ContractError(f"prepared tensor payload digest mismatch: {name}")
            if tensor_row.get("kind") == "bitwise_nonexpert" and tensor_row.get(
                "source_payload_sha256"
            ) != digest:
                raise ContractError(f"prepared nonexpert copy proof failed: {name}")
            if weight_map.get(name) != filename:
                raise ContractError(f"prepared index file mapping drifted: {name}")
            payload_bytes += byte_length

    if seen_names != set(weight_map):
        raise ContractError("prepared manifest/index tensor sets differ")
    if payload_bytes != output_section.get("payload_bytes") or payload_bytes != metadata.get(
        "total_size"
    ):
        raise ContractError("prepared payload-byte total drifted")
    physical = {path.name for path in output.glob("*.safetensors") if path.is_file()}
    if physical != manifest_file_names:
        raise ContractError("prepared physical/manifested file sets differ")
    if not allow_build_state and (output / PREPARED_STATE_NAME).exists():
        raise ContractError("published prepared checkpoint retained build state")
    if any(".partial" in child.name for child in output.iterdir()):
        raise ContractError("prepared checkpoint retained a partial artifact")

    layer_count = int(loader["num_hidden_layers"])
    experts = int(loader["n_routed_experts"])
    expected_shapes = _prepared_shapes(output_config)
    for layer in range(layer_count):
        prefix = f"{PREPARED_NAMESPACE}.layers.{layer}.experts."
        names = {
            name[len(prefix) :]
            for name in weight_map
            if name.startswith(prefix)
        }
        if names != set(PREPARED_FAMILY_ORDER):
            raise ContractError(
                f"prepared layer {layer} does not contain exactly eight families"
            )
        filename = _prepared_routed_filename(layer)
        row = file_rows.get(filename)
        if row is None:
            raise ContractError(f"prepared routed layer file is missing: {layer}")
        for family in PREPARED_FAMILY_ORDER:
            name = _prepared_tensor_name(layer, family)
            if weight_map.get(name) != filename:
                raise ContractError(f"prepared family is not layer-local: {name}")
            tensor_row = next(
                value for value in row["tensors"] if value["name"] == name
            )
            if (
                tensor_row.get("shape") != list(expected_shapes[family])
                or tensor_row.get("dtype") != PREPARED_DTYPES[family]
                or tensor_row.get("kind")
                != "tp2_rank_major_cutlass_prepared"
            ):
                raise ContractError(f"prepared family contract drifted: {name}")
    if any(EXPERT_RE.fullmatch(name) is not None for name in weight_map):
        raise ContractError("prepared index retained original per-expert names")
    if any(name.startswith(NAMESPACE + ".") for name in weight_map):
        raise ContractError("prepared index retained raw-v1 expert names")
    source_payload = source_section.get("payload_bytes")
    expected_delta = layer_count * experts * 2 * 4
    if (
        not isinstance(source_payload, int)
        or payload_bytes != source_payload + expected_delta
        or output_section.get("source_payload_delta_bytes") != expected_delta
    ):
        raise ContractError("prepared source/output payload arithmetic drifted")

    proofs = preparation.get("layer_rank_proofs")
    if not isinstance(proofs, dict) or set(proofs) != {
        str(layer) for layer in range(layer_count)
    }:
        raise ContractError("prepared per-layer provenance is incomplete")
    if (
        source_section.get("config_sha256")
        == "0c5dc7303ff322d73e0cd5caf9cc1b65d6efeff68fab53514531c2e959b1d616"
        and source_section.get("index_sha256")
        == "2d83d58754cff11724f117d20d95e31803a48512d29f8e00463b2501905d6d72"
    ):
        layer0 = proofs.get("0")
        if (
            not isinstance(layer0, list)
            or len(layer0) != TP_SIZE
            or layer0[0].get("immutable_reference")
            != {
                "source_json_sha256": LAYER0_RANK0_REFERENCE_JSON_SHA256,
                "fingerprints": LAYER0_RANK0_REFERENCE_FINGERPRINTS,
                "passed": True,
            }
        ):
            raise ContractError("immutable layer0/rank0 GPU reference proof is missing")
    return {
        "ok": True,
        "manifest_sha256": observed_manifest_digest,
        "file_count": len(files),
        "tensor_count": len(seen_names),
        "payload_bytes": payload_bytes,
        "loader_contract": PREPARED_LOADER_CONTRACT,
        "namespace": PREPARED_NAMESPACE,
        "runtime_h2d_calls_per_layer": len(PREPARED_FAMILY_ORDER),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser(
        "build", help="build and verify a repacked checkpoint"
    )
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--namespace", required=True)
    build.add_argument("--expected-config-sha256", required=True)
    build.add_argument("--expected-index-sha256", required=True)
    build.add_argument("--chunk-mib", type=int, default=8)
    prepared = subparsers.add_parser(
        "build-prepared",
        help="build/resume and verify the CPU-prepared CUTLASS checkpoint",
    )
    prepared.add_argument("--source", type=Path, required=True)
    prepared.add_argument("--output", type=Path, required=True)
    prepared.add_argument("--expected-config-sha256", required=True)
    prepared.add_argument("--expected-index-sha256", required=True)
    prepared.add_argument("--source-revision", required=True)
    prepared.add_argument("--chunk-mib", type=int, default=8)
    prepared.add_argument(
        "--stop-after-layer",
        type=int,
        choices=(0,),
        help=(
            "publish and record physical layer 0, then exit cleanly so its "
            "serialization boundary can be verified before resuming"
        ),
    )
    verify = subparsers.add_parser("verify", help="verify a completed repack")
    verify.add_argument("--checkpoint", type=Path, required=True)
    verify_prepared = subparsers.add_parser(
        "verify-prepared", help="verify a completed prepared repack"
    )
    verify_prepared.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "build":
            result = build_repacked_checkpoint(
                args.source,
                args.output,
                namespace=args.namespace,
                expected_config_sha256=args.expected_config_sha256,
                expected_index_sha256=args.expected_index_sha256,
                chunk_bytes=args.chunk_mib * 1024 * 1024,
            )
        elif args.command == "build-prepared":
            result = build_prepared_checkpoint(
                args.source,
                args.output,
                expected_config_sha256=args.expected_config_sha256,
                expected_index_sha256=args.expected_index_sha256,
                source_revision=args.source_revision,
                chunk_bytes=args.chunk_mib * 1024 * 1024,
                stop_after_layer=args.stop_after_layer,
            )
        elif args.command == "verify":
            result = verify_repacked_checkpoint(args.checkpoint)
        else:
            result = verify_prepared_checkpoint(args.checkpoint)
    except (ContractError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
