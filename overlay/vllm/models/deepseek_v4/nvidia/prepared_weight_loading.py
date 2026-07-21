# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed loader for the offline-prepared DeepSeek V4 NVFP4 format.

The prepared checkpoint is deliberately incompatible with the ordinary
per-expert ModelOpt loader.  Each target layer contains eight rank-major
tensors already in the final FlashInfer CUTLASS layout.  This module validates
the small immutable metadata contract before checkpoint payload iteration,
then performs one blocking H2D copy for each family.  Its default direct reader
uses explicit rank-range ``preadv`` calls instead of faulting cold mmap pages
inside those copies; setting its dedicated environment flag to ``0`` restores
the mmap path for diagnosis.  The post-load hook only constructs the quant
config and kernel; it must not reorder, reduce, swizzle, or modify scale values.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import os
import re
import stat
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


logger = logging.getLogger(__name__)

PREPARED_LOAD_ENV = "VLLM_DSV4_NVFP4_CUTLASS_PREPARED_LOAD"
PREPARED_MANIFEST_SHA256_ENV = (
    "VLLM_DSV4_NVFP4_CUTLASS_PREPARED_MANIFEST_SHA256"
)
PREPARED_DIRECT_READ_ENV = "VLLM_DSV4_NVFP4_CUTLASS_PREPARED_DIRECT_READ"

INDEX_NAME = "model.safetensors.index.json"
MANIFEST_NAME = "dspark-nvfp4-tp2-repack.json"
MANIFEST_DIGEST_NAME = f"{MANIFEST_NAME}.sha256"

PREPARED_NAMESPACE = "__dspark_tp2_nvfp4_cutlass_v1__"
PREPARED_SCHEMA = "dspark.deepseek_v4.nvfp4.tp2_cutlass_prepared.v1"
PREPARED_LOADER_CONTRACT = "deepseek_v4_nvfp4_tp2_cutlass_prepared_v1"
PREPARED_PAYLOAD_STAGE = "flashinfer_cutlass_prepared_v1"
PREPARED_BACKEND = "FLASHINFER_CUTLASS"
PREPARED_B12X_BACKEND = "FLASHINFER_B12X"
PREPARED_SCHEMA_VERSION = 1
PREPARED_ENGINE = "cpu_numpy_exact_v1"
VLLM_LAYOUT_PIN = "752a3a504485790a2e8491cacbb35c137339ad34"
FLASHINFER_LAYOUT_PIN = "0472b9b3f2fba11b463f8526f390297d52a8aad7"
SOURCE_CONFIG_SHA256 = (
    "0c5dc7303ff322d73e0cd5caf9cc1b65d6efeff68fab53514531c2e959b1d616"
)
SOURCE_INDEX_SHA256 = (
    "2d83d58754cff11724f117d20d95e31803a48512d29f8e00463b2501905d6d72"
)
PINNED_PREPARATION_SOURCE_SHA256 = {
    "flashinfer_fp4_moe": (
        "7a98da73bebad0168fbb19ecd96232d4bed0c3586af882a6409e8dabb4b60b9d"
    ),
    "nvfp4_oracle": (
        "746e6a5569696fe07329e13aeb397ae2152453d6a972640c7e7cf29efd173350"
    ),
    "modelopt": (
        "e39a867fdbefd46ad25a51dace9c294c2c0b079206f285eb08e092aefc0d77d5"
    ),
    "flashinfer_experts": (
        "d90f5215a6972c742be60ff8e9786432ab544570273483daa8faf317ba2d3ab5"
    ),
    "flashinfer_b12x_experts": (
        "4a6728752e7653a45c3afe65b88e9041e70cb95d1c44458b44b39d6e63231229"
    ),
    "nvfp4_utils": (
        "ed665537e42580e82ae71bb4f2ce8a699c0ffe8a042947c4eb600107c0b924ba"
    ),
}
LAYER0_RANK0_REFERENCE_JSON_SHA256 = (
    "b393a257791c2964d29c6762ad27658ab34b1a4de71d0b9a06a60974a0686ba6"
)
LAYER0_RANK0_REFERENCE_FINGERPRINTS = {
    "w13.weight": (
        "f02bb1c5778d151fbc210d57fe14c232a3dcb5b3ef213366a466ddf8ce875e55"
    ),
    "w2.weight": (
        "24b3299b6f60cd66f9b9209294503d7d8c18f5565790c52b1f2b5012270b586d"
    ),
    "w13.weight_scale": (
        "b76d300f85af4d71e6df85b2910cd1f6da319c97d7875d022c08c62e67ae8a0a"
    ),
    "w2.weight_scale": (
        "4a2c3041d99597af244183af181d3ca99edf738df4563f2c06a3b65fa67b7156"
    ),
    "a1_gscale": (
        "ca75efb8ecb87d5b545fb8e0acdbe4db69c2683b34073738ed3132c7fed8755a"
    ),
    "a2_gscale": (
        "4fe0db93441a8df2f59f74d196fdf5511db258c02dababb9bd9cd95a0b6f2887"
    ),
    "g1_alphas": (
        "42fd9084021adf41c496f6696244ac24c853d114ece6143660b428a5c4a1e193"
    ),
    "g2_alphas": (
        "e67aa7dc665954c5a922e9a74d645ef903240c935e9bc2b09f37770ac5ed0615"
    ),
}
REQUIRED_INTEGRITY = {
    "source_shards_hashed": True,
    "output_files_hashed": True,
    "output_tensors_hashed": True,
    "nonexpert_payloads_bitwise_copied": True,
    "scalar_rank_copies_bitwise_equal": True,
    "atomic_file_publication": True,
    "resumable_partial_directory": True,
    "atomic_directory_publication": True,
}

EXPECTED_LAYERS = 43
EXPECTED_EXPERTS = 256
EXPECTED_TP_SIZE = 2
EXPECTED_HIDDEN_SIZE = 4_096
EXPECTED_INTERMEDIATE_PER_RANK = 1_024
EXPECTED_H2D_CALLS_PER_LAYER = 8
EXPECTED_RANK_BYTES = 1_811_943_424

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
FINAL_SCALE_FIELDS = {
    "a1_gscale": "1 / max(all checkpoint w1,w3 input_scale)",
    "a2_gscale": "1 / max(all checkpoint w2 input_scale)",
    "g1_alphas": "w1.weight_scale_2 * reciprocal(a1_gscale)",
    "g2_alphas": "w2.weight_scale_2 * reciprocal(a2_gscale)",
}
OFFLINE_TRANSFORMS = [
    "tp2_slice",
    "reorder_w13_once_from_w1_w3_to_w3_w1",
    "reduce_checkpoint_input_scales_globally",
    "swizzle_block_scales_for_flashinfer_cutlass",
    "compute_final_cutlass_global_scales_and_alphas",
]

_FAMILY_TO_PARAMETER = {
    "w13.weight": "w13_weight",
    "w2.weight": "w2_weight",
    "w13.weight_scale": "w13_weight_scale",
    "w2.weight_scale": "w2_weight_scale",
    "a1_gscale": "w13_input_scale",
    "a2_gscale": "w2_input_scale",
    "g1_alphas": "w13_weight_scale_2",
    "g2_alphas": "w2_weight_scale_2",
}
_PREPARED_NAME_RE = re.compile(
    rf"^{re.escape(PREPARED_NAMESPACE)}\.layers\."
    r"(?P<layer>[0-9]+)\.experts\."
    r"(?P<family>w13\.weight|w2\.weight|w13\.weight_scale|"
    r"w2\.weight_scale|a1_gscale|a2_gscale|g1_alphas|g2_alphas)$"
)
_ORIGINAL_ROUTED_RE = re.compile(
    r"^(?:model\.)?layers\.[0-9]+\.ffn\.experts\.[0-9]+\.w[123]\."
    r"(?:weight|weight_scale|weight_scale_2|input_scale)$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def _strict_flag(name: str, environ: Mapping[str, str]) -> bool:
    value = environ.get(name, "0")
    if value == "0":
        return False
    if value == "1":
        return True
    raise ValueError(f"{name} must be exactly '0' or '1'; got {value!r}")


def prepared_load_requested(
    environ: Mapping[str, str] | None = None,
) -> bool:
    return _strict_flag(
        PREPARED_LOAD_ENV, os.environ if environ is None else environ
    )


def prepared_direct_read_requested(
    environ: Mapping[str, str] | None = None,
) -> bool:
    source = os.environ if environ is None else environ
    if PREPARED_DIRECT_READ_ENV not in source:
        return True
    return _strict_flag(PREPARED_DIRECT_READ_ENV, source)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read prepared NVFP4 {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Prepared NVFP4 {label} must be a JSON object")
    return value


def _expected_config_marker() -> dict[str, Any]:
    return {
        "schema": PREPARED_SCHEMA,
        "loader_contract": PREPARED_LOADER_CONTRACT,
        "namespace": PREPARED_NAMESPACE,
        "payload_stage": PREPARED_PAYLOAD_STAGE,
        "required_backend": PREPARED_BACKEND,
        "vllm_layout_pin": VLLM_LAYOUT_PIN,
        "flashinfer_layout_pin": FLASHINFER_LAYOUT_PIN,
        "tp_size": EXPECTED_TP_SIZE,
        "manifest": MANIFEST_NAME,
        "manifest_digest": MANIFEST_DIGEST_NAME,
    }


def _expected_loader_contract() -> dict[str, Any]:
    return {
        "schema": PREPARED_SCHEMA,
        "loader_contract": PREPARED_LOADER_CONTRACT,
        "namespace": PREPARED_NAMESPACE,
        "payload_stage": PREPARED_PAYLOAD_STAGE,
        "required_backend": PREPARED_BACKEND,
        "vllm_layout_pin": VLLM_LAYOUT_PIN,
        "flashinfer_layout_pin": FLASHINFER_LAYOUT_PIN,
        "standard_vllm_compatible": False,
        "fail_closed_without_exact_loader_contract": True,
        "cutlass_serving_layout_ready": True,
        "tp_size": EXPECTED_TP_SIZE,
        "matrix_rank_axis": 0,
        "matrix_expert_axis": 1,
        "families": list(PREPARED_FAMILY_ORDER),
        "w13_final_projection_order": ["w3", "w1"],
        "required_runtime_transforms": [],
        "runtime_h2d_calls_per_layer": EXPECTED_H2D_CALLS_PER_LAYER,
        "runtime_source_reads_per_layer": EXPECTED_H2D_CALLS_PER_LAYER,
        "scalar_rank_copies_required_bitwise_equal": True,
        "final_scale_fields": dict(FINAL_SCALE_FIELDS),
        "offline_transforms": list(OFFLINE_TRANSFORMS),
        "num_hidden_layers": EXPECTED_LAYERS,
        "n_routed_experts": EXPECTED_EXPERTS,
    }


def _expected_index_metadata() -> dict[str, Any]:
    return {
        "dspark_schema": PREPARED_SCHEMA,
        "dspark_loader_contract": PREPARED_LOADER_CONTRACT,
        "dspark_namespace": PREPARED_NAMESPACE,
        "dspark_payload_stage": PREPARED_PAYLOAD_STAGE,
        "dspark_required_backend": PREPARED_BACKEND,
        "dspark_vllm_layout_pin": VLLM_LAYOUT_PIN,
        "dspark_flashinfer_layout_pin": FLASHINFER_LAYOUT_PIN,
        "dspark_standard_loader_compatible": False,
        "source_index_sha256": SOURCE_INDEX_SHA256,
    }


def _validate_preparation_identity(identity: Any) -> None:
    if not isinstance(identity, dict):
        raise RuntimeError("Prepared NVFP4 preparation identity is missing")
    required_keys = {
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
        "pinned_preparation_source_sha256",
        "is_act_and_mul",
    }
    if set(identity) != required_keys:
        raise RuntimeError("Prepared NVFP4 preparation identity fields drifted")
    exact = {
        "implementation": (
            "scripts.repack_deepseek_v4_nvfp4_tp2._cpu_prepare_rank"
        ),
        "engine": PREPARED_ENGINE,
        "backend": PREPARED_BACKEND,
        "vllm_layout_pin": VLLM_LAYOUT_PIN,
        "flashinfer_layout_pin": FLASHINFER_LAYOUT_PIN,
        "pinned_preparation_source_sha256": PINNED_PREPARATION_SOURCE_SHA256,
        "is_act_and_mul": True,
    }
    for key, expected in exact.items():
        if identity.get(key) != expected:
            raise RuntimeError(
                f"Prepared NVFP4 preparation identity {key!r} drifted"
            )
    for key in ("numpy_version", "repacker_script_path"):
        if not isinstance(identity.get(key), str) or not identity[key]:
            raise RuntimeError(
                f"Prepared NVFP4 preparation identity {key!r} is missing"
            )
    for key in ("transform_spec_sha256", "repacker_script_sha256"):
        if _SHA256_RE.fullmatch(str(identity.get(key, ""))) is None:
            raise RuntimeError(
                f"Prepared NVFP4 preparation identity {key!r} is malformed"
            )
    if _GIT_REVISION_RE.fullmatch(str(identity.get("source_revision", ""))) is None:
        raise RuntimeError("Prepared NVFP4 source revision is malformed")


@dataclass(frozen=True)
class PreparedCheckpointContract:
    checkpoint: Path
    manifest_sha256: str
    output_index_sha256: str
    layer_files: tuple[str, ...]


@dataclass(frozen=True)
class PreparedRankRange:
    family: str
    offset: int
    nbytes: int


_SAFETENSORS_HEADER_LIMIT = 1 << 20
_PREPARED_DTYPE_TOKENS = {
    "w13.weight": ("U8", 1),
    "w2.weight": ("U8", 1),
    "w13.weight_scale": ("F8_E4M3", 1),
    "w2.weight_scale": ("F8_E4M3", 1),
    "a1_gscale": ("F32", 4),
    "a2_gscale": ("F32", 4),
    "g1_alphas": ("F32", 4),
    "g2_alphas": ("F32", 4),
}


def _pread_exact_bytes(fd: int, size: int, offset: int, *, label: str) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    cursor = offset
    while remaining:
        chunk = os.pread(fd, remaining, cursor)
        if not chunk:
            raise RuntimeError(f"Prepared NVFP4 {label} was truncated")
        chunks.append(chunk)
        cursor += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _parse_prepared_rank_ranges(
    fd: int,
    *,
    path: Path,
    layer: int,
    tp_rank: int,
    source_shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> dict[str, PreparedRankRange]:
    """Parse and validate the eight exact rank slices without touching payload."""

    if tp_rank not in range(EXPECTED_TP_SIZE):
        raise RuntimeError(f"Prepared NVFP4 TP rank must be 0 or 1; got {tp_rank}")
    shapes = dict(_source_shapes() if source_shapes is None else source_shapes)
    prefix = _pread_exact_bytes(fd, 8, 0, label=f"header prefix {path}")
    header_size = int.from_bytes(prefix, byteorder="little", signed=False)
    if not 2 <= header_size <= _SAFETENSORS_HEADER_LIMIT:
        raise RuntimeError(
            f"Prepared NVFP4 safetensors header length drifted for {path}: "
            f"{header_size}"
        )
    raw_header = _pread_exact_bytes(
        fd, header_size, 8, label=f"header JSON {path}"
    )
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Prepared NVFP4 safetensors header is invalid for {path}: {error}"
        ) from error
    if not isinstance(header, dict):
        raise RuntimeError(f"Prepared NVFP4 safetensors header is not an object: {path}")
    metadata = header.get("__metadata__")
    if metadata != _expected_prepared_header_metadata(layer):
        raise RuntimeError(f"Prepared NVFP4 safetensors metadata drifted: {path}")

    prefix_name = f"{PREPARED_NAMESPACE}.layers.{layer}.experts."
    expected_names = {f"{prefix_name}{family}" for family in PREPARED_FAMILY_ORDER}
    observed_names = set(header) - {"__metadata__"}
    if observed_names != expected_names:
        raise RuntimeError(f"Prepared NVFP4 safetensors tensor set drifted: {path}")

    payload_base = 8 + header_size
    file_size = os.fstat(fd).st_size
    ranges: dict[str, PreparedRankRange] = {}
    all_offsets: list[tuple[int, int]] = []
    for family in PREPARED_FAMILY_ORDER:
        name = f"{prefix_name}{family}"
        row = header.get(name)
        expected_dtype, element_size = _PREPARED_DTYPE_TOKENS[family]
        expected_shape = shapes[family]
        expected_total_bytes = math.prod(expected_shape) * element_size
        offsets = row.get("data_offsets") if isinstance(row, dict) else None
        if (
            not isinstance(row, dict)
            or row.get("dtype") != expected_dtype
            or row.get("shape") != list(expected_shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[1] - offsets[0] != expected_total_bytes
        ):
            raise RuntimeError(
                f"Prepared NVFP4 safetensors tensor metadata drifted: {name}"
            )
        rank_bytes, remainder = divmod(expected_total_bytes, EXPECTED_TP_SIZE)
        if remainder:
            raise RuntimeError(f"Prepared NVFP4 rank byte split drifted: {name}")
        rank_offset = payload_base + offsets[0] + tp_rank * rank_bytes
        if rank_offset < payload_base or rank_offset + rank_bytes > file_size:
            raise RuntimeError(f"Prepared NVFP4 rank range exceeds file: {name}")
        ranges[family] = PreparedRankRange(
            family=family,
            offset=rank_offset,
            nbytes=rank_bytes,
        )
        all_offsets.append((offsets[0], offsets[1]))

    ordered = sorted(all_offsets)
    if ordered[0][0] != 0 or any(
        previous[1] != current[0]
        for previous, current in zip(ordered, ordered[1:])
    ):
        raise RuntimeError(f"Prepared NVFP4 payload ranges are not gapless: {path}")
    if payload_base + ordered[-1][1] != file_size:
        raise RuntimeError(f"Prepared NVFP4 payload/file size drifted: {path}")
    return ranges


def _preadv_exact_into(
    fd: int,
    target: memoryview,
    offset: int,
    *,
    preadv_fn: Callable[[int, list[memoryview], int], int] | None = None,
) -> int:
    """Fill a writable byte view, tolerating EINTR/partial preadv results."""

    if target.readonly:
        raise RuntimeError("Prepared NVFP4 direct-read target must be writable")
    preadv = os.preadv if preadv_fn is None else preadv_fn
    total = 0
    calls = 0
    while total < len(target):
        try:
            observed = preadv(fd, [target[total:]], offset + total)
        except InterruptedError:
            continue
        calls += 1
        if observed <= 0:
            raise RuntimeError(
                "Prepared NVFP4 direct read ended before the rank range was full"
            )
        total += observed
    return calls


def _posix_fadvise_if_supported(
    fd: int, offset: int, length: int, advice_name: str
) -> None:
    advise = getattr(os, "posix_fadvise", None)
    advice = getattr(os, advice_name, None)
    if advise is None or advice is None:
        return
    try:
        advise(fd, offset, length, advice)
    except OSError as error:
        # Advisory cache policy is never part of the payload correctness
        # contract.  The explicit preadv path remains valid without it.
        logger.debug("Prepared NVFP4 %s was unavailable: %s", advice_name, error)


def inspect_prepared_checkpoint(
    checkpoint: str | os.PathLike[str],
    *,
    environ: Mapping[str, str] | None = None,
) -> PreparedCheckpointContract | None:
    """Validate the metadata-only contract before any tensor is consumed.

    The manifest and output index are small and cryptographically pinned.  The
    per-file payload digests recorded by that immutable manifest are verified
    offline and after transfer; this runtime check intentionally performs only
    file inventory/stat checks rather than rehashing the 168 GB payload.
    """

    source = os.environ if environ is None else environ
    requested = prepared_load_requested(source)
    root = Path(checkpoint).resolve()
    config_path = root / "config.json"
    index_path = root / INDEX_NAME
    if not config_path.is_file() or not index_path.is_file():
        if requested:
            raise RuntimeError(
                "Prepared NVFP4 loading requires a local checkpoint directory "
                f"with config.json and {INDEX_NAME}: {root}"
            )
        return None

    config = _read_json(config_path, "config")
    marker = config.get("dspark_nvfp4_prepared")
    index = _read_json(index_path, "index")
    metadata = index.get("metadata")
    expected_index = _expected_index_metadata()
    index_declares_prepared = isinstance(metadata, dict) and any(
        metadata.get(key) == value for key, value in expected_index.items()
    )
    config_declares_prepared = marker is not None
    declared = config_declares_prepared or index_declares_prepared
    if requested != declared:
        raise RuntimeError(
            "Prepared NVFP4 env/checkpoint declaration mismatch: "
            f"requested={requested}, checkpoint_declares_prepared={declared}"
        )
    if not requested:
        return None
    if not isinstance(metadata, dict):
        raise RuntimeError("Prepared NVFP4 index metadata must be an object")
    if marker != _expected_config_marker():
        raise RuntimeError("Prepared NVFP4 config marker drifted from its exact contract")
    for key, expected in expected_index.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"Prepared NVFP4 index metadata {key!r} drifted: "
                f"observed={metadata.get(key)!r}, expected={expected!r}"
            )

    expected_manifest_sha = source.get(PREPARED_MANIFEST_SHA256_ENV, "").lower()
    if _SHA256_RE.fullmatch(expected_manifest_sha) is None:
        raise RuntimeError(
            f"{PREPARED_MANIFEST_SHA256_ENV} must pin an exact lowercase SHA-256"
        )
    manifest_path = root / MANIFEST_NAME
    digest_path = root / MANIFEST_DIGEST_NAME
    if not manifest_path.is_file() or not digest_path.is_file():
        raise RuntimeError("Prepared NVFP4 manifest or digest sidecar is missing")
    words = digest_path.read_text(encoding="ascii").strip().split()
    if len(words) != 2 or words[1] != MANIFEST_NAME:
        raise RuntimeError("Prepared NVFP4 manifest digest sidecar is malformed")
    observed_manifest_sha = _sha256_file(manifest_path)
    if words[0] != observed_manifest_sha or observed_manifest_sha != expected_manifest_sha:
        raise RuntimeError(
            "Prepared NVFP4 manifest SHA-256 does not match sidecar/env pin"
        )
    manifest = _read_json(manifest_path, "manifest")
    if (
        manifest.get("schema_version") != PREPARED_SCHEMA_VERSION
        or manifest.get("format") != PREPARED_SCHEMA
    ):
        raise RuntimeError("Prepared NVFP4 manifest schema drifted")
    if manifest.get("loader") != _expected_loader_contract():
        raise RuntimeError("Prepared NVFP4 manifest loader contract drifted")
    preparation = manifest.get("preparation")
    if not isinstance(preparation, dict):
        raise RuntimeError("Prepared NVFP4 preparation provenance is missing")
    _validate_preparation_identity(preparation.get("identity"))
    proofs = preparation.get("layer_rank_proofs")
    if not isinstance(proofs, dict) or set(proofs) != {
        str(layer) for layer in range(EXPECTED_LAYERS)
    }:
        raise RuntimeError("Prepared NVFP4 per-layer provenance is incomplete")
    for layer in range(EXPECTED_LAYERS):
        ranks = proofs[str(layer)]
        if (
            not isinstance(ranks, list)
            or len(ranks) != EXPECTED_TP_SIZE
            or any(not isinstance(row, dict) for row in ranks)
            or [row.get("rank") for row in ranks] != list(range(EXPECTED_TP_SIZE))
            or any(row.get("engine") != PREPARED_ENGINE for row in ranks)
            or any(
                row.get("w13_scale_2_columns_bitwise_equal") is not True
                for row in ranks
            )
            or any(
                _SHA256_RE.fullmatch(
                    str(row.get("transform_spec_sha256", ""))
                )
                is None
                for row in ranks
            )
        ):
            raise RuntimeError(
                f"Prepared NVFP4 rank provenance drifted for layer {layer}"
            )
    source_section = manifest.get("source")
    if not isinstance(source_section, dict) or (
        source_section.get("config_sha256") != SOURCE_CONFIG_SHA256
        or source_section.get("index_sha256") != SOURCE_INDEX_SHA256
    ):
        raise RuntimeError("Prepared NVFP4 source checkpoint identity drifted")
    output = manifest.get("output")
    if not isinstance(output, dict):
        raise RuntimeError("Prepared NVFP4 manifest output section is missing")
    if manifest.get("integrity") != REQUIRED_INTEGRITY:
        raise RuntimeError("Prepared NVFP4 integrity contract drifted")
    output_config_sha = _sha256_file(config_path)
    if output.get("config_sha256") != output_config_sha:
        raise RuntimeError("Prepared NVFP4 output config digest drifted")
    output_index_sha = _sha256_file(index_path)
    if output.get("index_sha256") != output_index_sha:
        raise RuntimeError("Prepared NVFP4 output index digest drifted")

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("Prepared NVFP4 index weight_map must be an object")
    prepared_names: set[str] = set()
    layer_files: list[str] = []
    for layer in range(EXPECTED_LAYERS):
        filenames: set[str] = set()
        for family in PREPARED_FAMILY_ORDER:
            name = f"{PREPARED_NAMESPACE}.layers.{layer}.experts.{family}"
            filename = weight_map.get(name)
            if (
                not isinstance(filename, str)
                or not filename.endswith(".safetensors")
                or Path(filename).name != filename
            ):
                raise RuntimeError(f"Prepared NVFP4 weight_map lacks {name!r}")
            prepared_names.add(name)
            filenames.add(filename)
        if len(filenames) != 1:
            raise RuntimeError(
                f"Prepared NVFP4 layer {layer} is not contained in one file"
            )
        layer_files.append(next(iter(filenames)))
    if len(set(layer_files)) != EXPECTED_LAYERS:
        raise RuntimeError("Prepared NVFP4 layers do not use distinct physical files")
    observed_prepared = {
        name for name in weight_map if name.startswith(f"{PREPARED_NAMESPACE}.")
    }
    if observed_prepared != prepared_names:
        raise RuntimeError("Prepared NVFP4 namespace contains missing/extra tensors")
    if any(_ORIGINAL_ROUTED_RE.fullmatch(name) for name in weight_map):
        raise RuntimeError("Prepared NVFP4 index still contains per-expert target tensors")

    file_rows = output.get("files")
    if not isinstance(file_rows, list):
        raise RuntimeError("Prepared NVFP4 output file inventory is missing")
    rows_by_name: dict[str, dict[str, Any]] = {}
    for row in file_rows:
        filename = row.get("path") if isinstance(row, dict) else None
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.endswith(".safetensors")
            or filename in rows_by_name
        ):
            raise RuntimeError("Prepared NVFP4 output file inventory is malformed")
        rows_by_name[filename] = row
    physical_files = {
        path.name for path in root.glob("*.safetensors") if path.is_file()
    }
    if physical_files != set(rows_by_name):
        raise RuntimeError("Prepared NVFP4 physical/manifest file sets differ")
    manifest_tensor_names: set[str] = set()
    manifest_payload_bytes = 0
    for filename, row in rows_by_name.items():
        path = root / filename
        tensor_rows = row.get("tensors") if isinstance(row, dict) else None
        if (
            not isinstance(row, dict)
            or not path.is_file()
            or path.stat().st_size != row.get("size")
            or _SHA256_RE.fullmatch(str(row.get("sha256", ""))) is None
            or not isinstance(tensor_rows, list)
            or row.get("tensor_count") != len(tensor_rows)
        ):
            raise RuntimeError(
                f"Prepared NVFP4 physical file inventory drifted for {filename}"
            )
        file_payload_bytes = 0
        for tensor_row in tensor_rows:
            name = tensor_row.get("name") if isinstance(tensor_row, dict) else None
            tensor_bytes = (
                tensor_row.get("bytes") if isinstance(tensor_row, dict) else None
            )
            if (
                not isinstance(name, str)
                or name in manifest_tensor_names
                or weight_map.get(name) != filename
                or not isinstance(tensor_bytes, int)
                or tensor_bytes <= 0
                or _SHA256_RE.fullmatch(
                    str(tensor_row.get("sha256", ""))
                )
                is None
            ):
                raise RuntimeError(
                    f"Prepared NVFP4 tensor inventory drifted in {filename}"
                )
            manifest_tensor_names.add(name)
            file_payload_bytes += tensor_bytes
        if row.get("payload_bytes") != file_payload_bytes:
            raise RuntimeError(
                f"Prepared NVFP4 payload byte inventory drifted for {filename}"
            )
        manifest_payload_bytes += file_payload_bytes
    if manifest_tensor_names != set(weight_map):
        raise RuntimeError("Prepared NVFP4 manifest/index tensor sets differ")
    if (
        output.get("tensor_count") != len(weight_map)
        or output.get("layer_file_count") != EXPECTED_LAYERS
        or output.get("payload_bytes") != manifest_payload_bytes
        or metadata.get("total_size") != manifest_payload_bytes
    ):
        raise RuntimeError("Prepared NVFP4 output payload arithmetic drifted")
    dtype_names = {
        "w13.weight": "U8",
        "w2.weight": "U8",
        "w13.weight_scale": "F8_E4M3",
        "w2.weight_scale": "F8_E4M3",
        "a1_gscale": "F32",
        "a2_gscale": "F32",
        "g1_alphas": "F32",
        "g2_alphas": "F32",
    }
    source_shapes = _source_shapes()
    for layer, filename in enumerate(layer_files):
        tensor_rows = rows_by_name[filename].get("tensors")
        by_tensor = {
            row.get("name"): row for row in tensor_rows if isinstance(row, dict)
        }
        for family in PREPARED_FAMILY_ORDER:
            name = f"{PREPARED_NAMESPACE}.layers.{layer}.experts.{family}"
            row = by_tensor.get(name)
            if (
                not isinstance(row, dict)
                or row.get("family") != family
                or row.get("kind") != "tp2_rank_major_cutlass_prepared"
                or row.get("dtype") != dtype_names[family]
                or row.get("shape") != list(source_shapes[family])
                or _SHA256_RE.fullmatch(str(row.get("sha256", ""))) is None
            ):
                raise RuntimeError(
                    f"Prepared NVFP4 tensor metadata drifted for {name}"
                )
    return PreparedCheckpointContract(
        checkpoint=root,
        manifest_sha256=observed_manifest_sha,
        output_index_sha256=output_index_sha,
        layer_files=tuple(layer_files),
    )


def _device_type(tensor: Any) -> str | None:
    return getattr(getattr(tensor, "device", None), "type", None)


def _tensor_bytes(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _source_shapes() -> dict[str, tuple[int, ...]]:
    e = EXPECTED_EXPERTS
    i = EXPECTED_INTERMEDIATE_PER_RANK
    k = EXPECTED_HIDDEN_SIZE
    return {
        "w13.weight": (EXPECTED_TP_SIZE, e, 2 * i, k // 2),
        "w2.weight": (EXPECTED_TP_SIZE, e, k, i // 2),
        "w13.weight_scale": (EXPECTED_TP_SIZE, e, 2 * i, k // 16),
        "w2.weight_scale": (EXPECTED_TP_SIZE, e, k, i // 16),
        "a1_gscale": (EXPECTED_TP_SIZE, e),
        "a2_gscale": (EXPECTED_TP_SIZE, e),
        "g1_alphas": (EXPECTED_TP_SIZE, e),
        "g2_alphas": (EXPECTED_TP_SIZE, e),
    }


def _destination_shapes() -> dict[str, tuple[int, ...]]:
    return {family: shape[1:] for family, shape in _source_shapes().items()}


def _family_dtypes(torch_module: Any) -> dict[str, Any]:
    return {
        "w13.weight": torch_module.uint8,
        "w2.weight": torch_module.uint8,
        "w13.weight_scale": torch_module.float8_e4m3fn,
        "w2.weight_scale": torch_module.float8_e4m3fn,
        "a1_gscale": torch_module.float32,
        "a2_gscale": torch_module.float32,
        "g1_alphas": torch_module.float32,
        "g2_alphas": torch_module.float32,
    }


def _expected_prepared_header_metadata(layer: int) -> dict[str, str]:
    return {
        "format": "pt",
        "dspark_schema": PREPARED_SCHEMA,
        "dspark_loader_contract": PREPARED_LOADER_CONTRACT,
        "dspark_namespace": PREPARED_NAMESPACE,
        "dspark_payload_stage": PREPARED_PAYLOAD_STAGE,
        "dspark_required_backend": PREPARED_BACKEND,
        "dspark_standard_loader_compatible": "false",
        "dspark_layer": str(layer),
    }


def _sample_prepared_tensor_digest(
    tensor: Any, family: str, torch_module: Any
) -> str:
    """Match the immutable Gate-2/Gate-3 sampled tensor fingerprint."""

    if family not in PREPARED_FAMILY_ORDER or not bool(tensor.is_contiguous()):
        raise RuntimeError("Prepared NVFP4 fingerprint input contract drifted")
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
    flat = tensor.view(torch_module.uint8).reshape(-1)
    flat_size = int(flat.numel())
    sample_bytes = (
        flat_size if family in PREPARED_FAMILY_ORDER[4:] else 4096
    )
    offsets = sorted(
        {
            0,
            max(0, flat_size // 2 - sample_bytes // 2),
            max(0, flat_size - sample_bytes),
        }
    )
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(dtype_text.encode())
    for offset in offsets:
        length = min(sample_bytes, flat_size - offset)
        sample = flat.narrow(0, offset, length)
        if hasattr(sample, "numpy"):
            payload = sample.numpy().tobytes()
        else:
            payload = bytes(sample.tolist())
        digest.update(payload)
    return digest.hexdigest()


def validate_prepared_layer_file(
    path: str | os.PathLike[str],
    *,
    layer: int = 0,
    torch_module: Any | None = None,
    safe_open_fn: Callable[..., Any] | None = None,
    fingerprint_fn: Callable[[Any, str, Any], str] | None = None,
) -> dict[str, Any]:
    """Validate one physical prepared layer without a completed manifest.

    This is the boundary gate for a resumable layer-0 pilot.  It opens only
    the requested safetensors file read-only, exercises the same source
    shape/dtype/rank contract as :class:`Nvfp4PreparedLayerLoader`, and checks
    all eight rank-0 fingerprints against the immutable hardware reference.
    """

    if layer != 0:
        raise RuntimeError("Immutable prepared fingerprints are pinned only for layer 0")
    input_path = Path(os.path.abspath(os.fspath(path)))
    try:
        input_mode = input_path.lstat().st_mode
    except OSError as error:
        raise RuntimeError(
            f"Cannot stat prepared NVFP4 physical layer {input_path}: {error}"
        ) from error
    if not stat.S_ISREG(input_mode):
        raise RuntimeError(
            "Prepared NVFP4 physical layer must be a direct regular file"
        )
    source_path = input_path.resolve()
    expected_filename = f"model-layer-{layer:05d}.safetensors"
    if (
        source_path.name != expected_filename
        or not source_path.is_file()
        or source_path.stat().st_size <= 0
    ):
        raise RuntimeError(
            f"Prepared NVFP4 physical layer path must be {expected_filename}"
        )
    if torch_module is None:
        import torch as torch_module
    if safe_open_fn is None:
        from safetensors import safe_open as safe_open_fn
    if fingerprint_fn is None:
        fingerprint_fn = _sample_prepared_tensor_digest

    shapes = _source_shapes()
    dtypes = _family_dtypes(torch_module)
    dtype_names = {
        "w13.weight": "U8",
        "w2.weight": "U8",
        "w13.weight_scale": "F8_E4M3",
        "w2.weight_scale": "F8_E4M3",
        "a1_gscale": "F32",
        "a2_gscale": "F32",
        "g1_alphas": "F32",
        "g2_alphas": "F32",
    }
    prefix = f"{PREPARED_NAMESPACE}.layers.{layer}.experts."
    expected_names = {f"{prefix}{family}" for family in PREPARED_FAMILY_ORDER}
    expected_metadata = _expected_prepared_header_metadata(layer)
    fingerprints: dict[str, str] = {}
    rank_bytes = [0 for _ in range(EXPECTED_TP_SIZE)]
    contiguous = {str(rank): True for rank in range(EXPECTED_TP_SIZE)}
    with safe_open_fn(
        str(source_path), framework="pt", device="cpu"
    ) as handle:
        if handle.metadata() != expected_metadata:
            raise RuntimeError("Prepared NVFP4 physical header metadata drifted")
        observed_names = set(handle.keys())
        if observed_names != expected_names:
            raise RuntimeError("Prepared NVFP4 physical layer tensor names drifted")
        for family in PREPARED_FAMILY_ORDER:
            tensor = handle.get_tensor(f"{prefix}{family}")
            if (
                _device_type(tensor) != "cpu"
                or tuple(tensor.shape) != shapes[family]
                or tensor.dtype != dtypes[family]
            ):
                raise RuntimeError(
                    f"Prepared NVFP4 physical source contract drifted for {family}"
                )
            for rank in range(EXPECTED_TP_SIZE):
                rank_slice = tensor[rank]
                if (
                    tuple(rank_slice.shape) != shapes[family][1:]
                    or not bool(rank_slice.is_contiguous())
                ):
                    contiguous[str(rank)] = False
                    raise RuntimeError(
                        f"Prepared NVFP4 rank slice drifted for {family}/rank{rank}"
                    )
                rank_bytes[rank] += _tensor_bytes(rank_slice)
                if rank == 0:
                    fingerprints[family] = fingerprint_fn(
                        rank_slice, family, torch_module
                    )
    if rank_bytes != [EXPECTED_RANK_BYTES] * EXPECTED_TP_SIZE:
        raise RuntimeError(
            f"Prepared NVFP4 physical rank-byte contract drifted: {rank_bytes}"
        )
    if fingerprints != LAYER0_RANK0_REFERENCE_FINGERPRINTS:
        raise RuntimeError("Prepared NVFP4 physical fingerprints drifted")
    return {
        "ok": True,
        "path": str(source_path),
        "file_size": source_path.stat().st_size,
        "layer": layer,
        "tensor_count": len(PREPARED_FAMILY_ORDER),
        "header_metadata": expected_metadata,
        "families": list(PREPARED_FAMILY_ORDER),
        "source_shapes": {
            family: list(shapes[family]) for family in PREPARED_FAMILY_ORDER
        },
        "dtypes": dict(dtype_names),
        "tp_size": EXPECTED_TP_SIZE,
        "rank_slices_contiguous": contiguous,
        "rank_bytes": {str(rank): rank_bytes[rank] for rank in range(EXPECTED_TP_SIZE)},
        "rank0_fingerprints": fingerprints,
        "reference_json_sha256": LAYER0_RANK0_REFERENCE_JSON_SHA256,
        "fingerprints_match": True,
    }


@dataclass
class PreparedPostloadState:
    layer: int
    loaded: bool = False
    finalized: bool = False
    aborted: bool = False


def _finalize_prepared_cutlass(
    quant_method: Any,
    routed_layer: Any,
    state: PreparedPostloadState,
    *,
    quant_config_factory: Callable[..., Any] | None = None,
    kernel_factory: Callable[..., Any] | None = None,
    expected_backend: Any | None = None,
) -> None:
    """Initialize CUTLASS around final tensors without scale preparation."""

    if state.aborted or not state.loaded or state.finalized:
        raise RuntimeError(
            "Prepared NVFP4 post-load state is not exactly loaded/unfinalized"
        )
    if (
        quant_config_factory is None
        or kernel_factory is None
        or expected_backend is None
    ):
        from vllm.model_executor.layers.fused_moe.config import (
            nvfp4_moe_quant_config,
        )
        from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
            NvFp4MoeBackend,
            make_nvfp4_moe_kernel,
        )

        quant_config_factory = nvfp4_moe_quant_config
        kernel_factory = make_nvfp4_moe_kernel
        expected_backend = NvFp4MoeBackend.FLASHINFER_CUTLASS
    if quant_method.nvfp4_backend != expected_backend:
        raise RuntimeError("Prepared NVFP4 post-load requires FLASHINFER_CUTLASS")
    if quant_method.moe_quant_config is not None or getattr(
        quant_method, "moe_kernel", None
    ) is not None:
        raise RuntimeError("Prepared NVFP4 quant method was already initialized")

    quant_method.moe_quant_config = quant_config_factory(
        g1_alphas=routed_layer.w13_weight_scale_2,
        g2_alphas=routed_layer.w2_weight_scale_2,
        a1_gscale=routed_layer.w13_input_scale,
        a2_gscale=routed_layer.w2_input_scale,
        w1_scale=routed_layer.w13_weight_scale,
        w2_scale=routed_layer.w2_weight_scale,
        is_scale_swizzled=True,
        gemm1_clamp_limit=getattr(routed_layer, "swiglu_limit", None),
    )
    if quant_method.experts_cls is None:
        raise RuntimeError("Prepared NVFP4 CUTLASS experts class is missing")
    quant_method.moe_kernel = kernel_factory(
        moe_quant_config=quant_method.moe_quant_config,
        moe_config=quant_method.moe,
        experts_cls=quant_method.experts_cls,
        backend=quant_method.nvfp4_backend,
        routing_tables=routed_layer._expert_routing_tables(),
        layer=routed_layer,
    )
    state.finalized = True
    logger.info(
        "NVFP4_PREPARED event=postload layer=%d transforms=0 backend=%s",
        state.layer,
        PREPARED_BACKEND,
    )


def _finalize_prepared_b12x(
    quant_method: Any,
    routed_layer: Any,
    state: PreparedPostloadState,
    *,
    quant_config_factory: Callable[..., Any] | None = None,
    kernel_factory: Callable[..., Any] | None = None,
    expected_backend: Any | None = None,
) -> None:
    """Convert the prepared CUTLASS scale contract in place for B12X.

    Packed weights and expert-major block-scale storage are shared by both
    FlashInfer backends. The prepared artifact stores CUTLASS alphas as
    ``weight_scale_2 * activation_global`` and the reciprocal activation
    global separately. Recover ModelOpt's per-expert ``weight_scale_2`` in
    FP32, then let the pinned B12X expert bake it into the block scales and
    construct its MMA-layout views. No packed-weight copy or repack occurs.
    """

    if state.aborted or not state.loaded or state.finalized:
        raise RuntimeError(
            "Prepared NVFP4 post-load state is not exactly loaded/unfinalized"
        )
    if quant_config_factory is None or kernel_factory is None or expected_backend is None:
        from vllm.model_executor.layers.fused_moe.config import (
            nvfp4_moe_quant_config,
        )
        from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
            NvFp4MoeBackend,
            make_nvfp4_moe_kernel,
        )

        quant_config_factory = nvfp4_moe_quant_config
        kernel_factory = make_nvfp4_moe_kernel
        expected_backend = NvFp4MoeBackend.FLASHINFER_B12X
    if quant_method.nvfp4_backend != expected_backend:
        raise RuntimeError("Prepared NVFP4 B12X conversion requires FLASHINFER_B12X")
    if quant_method.moe_quant_config is not None or getattr(
        quant_method, "moe_kernel", None
    ) is not None:
        raise RuntimeError("Prepared NVFP4 quant method was already initialized")

    scalar_pairs = (
        (routed_layer.w13_weight_scale_2, routed_layer.w13_input_scale),
        (routed_layer.w2_weight_scale_2, routed_layer.w2_input_scale),
    )
    for alpha, reciprocal_activation_global in scalar_pairs:
        if tuple(alpha.shape) != tuple(reciprocal_activation_global.shape):
            raise RuntimeError("Prepared NVFP4 B12X scalar shape contract drifted")
        # CUTLASS prepared: alpha = raw_weight_scale_2 * activation_global.
        # Multiplication by reciprocal_activation_global recovers the scalar
        # contract consumed by FlashInferB12xExperts' ordinary post-load.
        alpha.data.mul_(reciprocal_activation_global)

    quant_method.moe_quant_config = quant_config_factory(
        g1_alphas=routed_layer.w13_weight_scale_2,
        g2_alphas=routed_layer.w2_weight_scale_2,
        a1_gscale=routed_layer.w13_input_scale,
        a2_gscale=routed_layer.w2_input_scale,
        w1_scale=routed_layer.w13_weight_scale,
        w2_scale=routed_layer.w2_weight_scale,
        is_scale_swizzled=True,
        gemm1_clamp_limit=getattr(routed_layer, "swiglu_limit", None),
    )
    if quant_method.experts_cls is None:
        raise RuntimeError("Prepared NVFP4 B12X experts class is missing")
    quant_method.moe_kernel = kernel_factory(
        moe_quant_config=quant_method.moe_quant_config,
        moe_config=quant_method.moe,
        experts_cls=quant_method.experts_cls,
        backend=quant_method.nvfp4_backend,
        routing_tables=routed_layer._expert_routing_tables(),
        layer=routed_layer,
    )
    postload = getattr(
        quant_method.moe_kernel, "process_weights_after_loading", None
    )
    if not callable(postload):
        raise RuntimeError("Prepared NVFP4 B12X kernel lacks post-load conversion")
    postload(routed_layer)
    state.finalized = True
    logger.info(
        "NVFP4_PREPARED event=postload layer=%d transforms=scale_recover,bake,mma "
        "backend=%s",
        state.layer,
        PREPARED_B12X_BACKEND,
    )


def _install_prepared_postload_hook(routed_layer: Any, state: PreparedPostloadState) -> None:
    quant_method = routed_layer.quant_method
    if hasattr(quant_method, "_dspark_prepared_original_postload"):
        raise RuntimeError("Prepared NVFP4 post-load hook was installed twice")
    if quant_method.__class__.__name__ != "ModelOptNvFp4FusedMoE":
        raise RuntimeError(
            "Prepared NVFP4 requires exact ModelOptNvFp4FusedMoE; got "
            f"{quant_method.__class__.__name__}"
        )
    backend = getattr(getattr(quant_method, "nvfp4_backend", None), "value", None)
    if backend not in (PREPARED_BACKEND, PREPARED_B12X_BACKEND):
        raise RuntimeError(
            "Prepared NVFP4 requires FLASHINFER_CUTLASS or "
            f"FLASHINFER_B12X; got {backend!r}"
        )
    experts_cls = getattr(quant_method, "experts_cls", None)
    expected_experts = {
        PREPARED_BACKEND: (
            "FlashInferExperts",
            "vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe",
        ),
        PREPARED_B12X_BACKEND: (
            "FlashInferB12xExperts",
            "vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe",
        ),
    }[backend]
    if (
        getattr(experts_cls, "__name__", None),
        getattr(experts_cls, "__module__", None),
    ) != expected_experts:
        raise RuntimeError("Prepared NVFP4 FlashInfer experts identity drifted")
    original = quant_method.process_weights_after_loading
    original_function = getattr(original, "__func__", None)
    if (
        getattr(original_function, "__module__", None)
        != "vllm.model_executor.layers.quantization.modelopt"
        or getattr(original_function, "__qualname__", None)
        != "ModelOptNvFp4FusedMoE.process_weights_after_loading"
    ):
        raise RuntimeError("Prepared NVFP4 ModelOpt post-load identity drifted")
    def prepared_postload(method_self: Any, candidate_layer: Any) -> None:
        if candidate_layer is not routed_layer:
            raise RuntimeError("Prepared NVFP4 post-load received the wrong layer")
        if backend == PREPARED_BACKEND:
            _finalize_prepared_cutlass(method_self, candidate_layer, state)
        else:
            _finalize_prepared_b12x(method_self, candidate_layer, state)

    quant_method._dspark_prepared_original_postload = original
    quant_method.process_weights_after_loading = types.MethodType(
        prepared_postload, quant_method
    )


def _validate_runtime_transform_sources(routed_layer: Any) -> None:
    """Pin the ModelOpt and selected expert implementations."""

    quant_method = routed_layer.quant_method
    experts_cls = getattr(quant_method, "experts_cls", None)
    backend = getattr(getattr(quant_method, "nvfp4_backend", None), "value", None)
    expert_contract = {
        PREPARED_BACKEND: (
            "FlashInferExperts",
            "vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe",
            PINNED_PREPARATION_SOURCE_SHA256["flashinfer_experts"],
        ),
        PREPARED_B12X_BACKEND: (
            "FlashInferB12xExperts",
            "vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe",
            PINNED_PREPARATION_SOURCE_SHA256["flashinfer_b12x_experts"],
        ),
    }.get(backend)
    if expert_contract is None:
        raise RuntimeError(f"Prepared NVFP4 runtime backend drifted: {backend!r}")
    source_contract = (
        (
            quant_method.__class__,
            "ModelOptNvFp4FusedMoE",
            "vllm.model_executor.layers.quantization.modelopt",
            PINNED_PREPARATION_SOURCE_SHA256["modelopt"],
        ),
        (experts_cls, *expert_contract),
    )
    for candidate, expected_name, expected_module, expected_sha in source_contract:
        if (
            candidate is None
            or getattr(candidate, "__name__", None) != expected_name
            or getattr(candidate, "__module__", None) != expected_module
        ):
            raise RuntimeError(
                f"Prepared NVFP4 runtime class identity drifted: {expected_name}"
            )
        source_path = inspect.getsourcefile(candidate)
        if source_path is None or _sha256_file(Path(source_path)) != expected_sha:
            raise RuntimeError(
                f"Prepared NVFP4 runtime source digest drifted: {expected_module}"
            )


class PreparedSafetensorsDirectReader:
    """Read exact rank payload ranges with large preadv calls.

    The ordinary safetensors iterator still supplies metadata-only tensor
    views, preserving AutoWeightsLoader ordering and mapping.  Payload bytes
    bypass the mmap view so a cold checkpoint does not devolve into one block
    read per page fault while ``copy_`` is holding the CUDA destination.
    """

    def __init__(
        self,
        *,
        torch_module: Any,
        contract: PreparedCheckpointContract,
        tp_rank: int,
        source_shapes: Mapping[str, tuple[int, ...]] | None = None,
        preadv_fn: Callable[[int, list[memoryview], int], int] | None = None,
        copy_bytes_fn: Callable[[Any, bytearray, int], None] | None = None,
    ) -> None:
        self._torch = torch_module
        self._contract = contract
        self._tp_rank = tp_rank
        self._source_shapes = dict(
            _source_shapes() if source_shapes is None else source_shapes
        )
        self._preadv_fn = preadv_fn
        self._copy_bytes_fn = copy_bytes_fn
        self._buffer: bytearray | None = None
        self._buffer_view: memoryview | None = None
        self._active_layer: int | None = None
        self._active_fd: int | None = None
        self._active_ranges: dict[str, PreparedRankRange] = {}
        self._seen: set[str] = set()
        self._stats: dict[int, dict[str, float | int]] = {}
        self._closed = False

    def _close_active(self) -> None:
        fd = self._active_fd
        self._active_fd = None
        self._active_layer = None
        self._active_ranges = {}
        self._seen = set()
        if fd is not None:
            os.close(fd)

    def _open_layer(self, layer: int) -> None:
        if self._closed:
            raise RuntimeError("Prepared NVFP4 direct reader is closed")
        if self._active_fd is not None:
            raise RuntimeError("Prepared NVFP4 direct reader layer is interleaved")
        try:
            filename = self._contract.layer_files[layer]
        except IndexError as error:
            raise RuntimeError(
                f"Prepared NVFP4 direct reader lacks layer file {layer}"
            ) from error
        path = self._contract.checkpoint / filename
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(path, flags)
        try:
            ranges = _parse_prepared_rank_ranges(
                fd,
                path=path,
                layer=layer,
                tp_rank=self._tp_rank,
                source_shapes=self._source_shapes,
            )
            _posix_fadvise_if_supported(
                fd, 0, 0, "POSIX_FADV_SEQUENTIAL"
            )
        except BaseException:
            os.close(fd)
            raise
        self._active_layer = layer
        self._active_fd = fd
        self._active_ranges = ranges
        max_bytes = max(item.nbytes for item in ranges.values())
        if self._buffer is None:
            self._buffer = bytearray(max_bytes)
            self._buffer_view = memoryview(self._buffer)
        elif len(self._buffer) < max_bytes:
            raise RuntimeError("Prepared NVFP4 direct-read buffer size drifted")
        self._stats[layer] = {
            "ranges": 0,
            "syscalls": 0,
            "bytes": 0,
            "read_seconds": 0.0,
            "copy_seconds": 0.0,
        }

    def _copy_bytes(self, destination: Any, nbytes: int) -> None:
        if self._buffer is None:
            raise RuntimeError("Prepared NVFP4 direct-read buffer is missing")
        if self._copy_bytes_fn is not None:
            self._copy_bytes_fn(destination, self._buffer, nbytes)
            return
        host = self._torch.frombuffer(
            self._buffer, dtype=self._torch.uint8, count=nbytes
        )
        target = destination.data.view(self._torch.uint8).reshape(-1)
        if int(target.numel()) != nbytes:
            raise RuntimeError("Prepared NVFP4 raw destination byte size drifted")
        target.copy_(host)

    def copy_into(self, *, layer: int, family: str, destination: Any) -> None:
        if self._active_layer is None:
            self._open_layer(layer)
        elif self._active_layer != layer:
            raise RuntimeError(
                "Prepared NVFP4 direct reader layers are interleaved: "
                f"active={self._active_layer}, observed={layer}"
            )
        if family in self._seen:
            raise RuntimeError(
                f"Prepared NVFP4 direct reader saw duplicate family {family!r}"
            )
        rank_range = self._active_ranges.get(family)
        if rank_range is None or self._active_fd is None or self._buffer_view is None:
            raise RuntimeError(
                f"Prepared NVFP4 direct reader lacks range for {family!r}"
            )
        view = self._buffer_view[: rank_range.nbytes]
        read_started = time.perf_counter()
        calls = _preadv_exact_into(
            self._active_fd,
            view,
            rank_range.offset,
            preadv_fn=self._preadv_fn,
        )
        read_seconds = time.perf_counter() - read_started
        copy_started = time.perf_counter()
        self._copy_bytes(destination, rank_range.nbytes)
        copy_seconds = time.perf_counter() - copy_started

        _posix_fadvise_if_supported(
            self._active_fd,
            rank_range.offset,
            rank_range.nbytes,
            "POSIX_FADV_DONTNEED",
        )
        row = self._stats[layer]
        row["ranges"] = int(row["ranges"]) + 1
        row["syscalls"] = int(row["syscalls"]) + calls
        row["bytes"] = int(row["bytes"]) + rank_range.nbytes
        row["read_seconds"] = float(row["read_seconds"]) + read_seconds
        row["copy_seconds"] = float(row["copy_seconds"]) + copy_seconds
        self._seen.add(family)
        if len(self._seen) == len(PREPARED_FAMILY_ORDER):
            self._close_active()

    def layer_stats(self, layer: int) -> Mapping[str, float | int]:
        return dict(self._stats.get(layer, {}))

    def summary(self) -> Mapping[str, float | int]:
        return {
            "ranges": sum(int(row["ranges"]) for row in self._stats.values()),
            "syscalls": sum(
                int(row["syscalls"]) for row in self._stats.values()
            ),
            "bytes": sum(int(row["bytes"]) for row in self._stats.values()),
            "read_seconds": sum(
                float(row["read_seconds"]) for row in self._stats.values()
            ),
            "copy_seconds": sum(
                float(row["copy_seconds"]) for row in self._stats.values()
            ),
        }

    def finish(self) -> None:
        if self._active_fd is not None:
            raise RuntimeError("Prepared NVFP4 direct reader ended mid-layer")
        self.close()

    def close(self) -> None:
        self._close_active()
        if self._buffer_view is not None:
            self._buffer_view.release()
        self._buffer_view = None
        self._buffer = None
        self._closed = True


class Nvfp4PreparedLayerLoader:
    """Consume exactly eight final tensors for each target layer."""

    def __init__(
        self,
        *,
        torch_module: Any,
        tp_rank: int,
        parameters: dict[int, dict[str, Any]],
        states: dict[int, PreparedPostloadState],
        expected_source_shapes: dict[str, tuple[int, ...]] | None = None,
        expected_rank_bytes: int = EXPECTED_RANK_BYTES,
        direct_reader: PreparedSafetensorsDirectReader | None = None,
    ) -> None:
        self._torch = torch_module
        self._tp_rank = tp_rank
        self._parameters = parameters
        self._states = states
        self._source_shapes = expected_source_shapes or _source_shapes()
        self._dtypes = _family_dtypes(torch_module)
        self._expected_rank_bytes = expected_rank_bytes
        self._direct_reader = direct_reader
        self._seen: dict[int, set[str]] = {layer: set() for layer in parameters}
        self._completed: set[int] = set()
        self._copies = 0
        self._active_layer: int | None = None
        self._active_started = 0.0
        self._started = time.perf_counter()
        if tp_rank not in range(EXPECTED_TP_SIZE):
            raise RuntimeError(f"Prepared NVFP4 TP rank must be 0 or 1; got {tp_rank}")

    @property
    def completed_layers(self) -> frozenset[int]:
        return frozenset(self._completed)

    @property
    def total_h2d_calls(self) -> int:
        return self._copies

    @property
    def is_complete(self) -> bool:
        return self._completed == set(self._parameters) and self._active_layer is None

    def consume(self, name: str, loaded_weight: Any) -> str | None:
        if not name.startswith(f"{PREPARED_NAMESPACE}."):
            return None
        match = _PREPARED_NAME_RE.fullmatch(name)
        if match is None:
            raise RuntimeError(f"Unknown prepared NVFP4 tensor name: {name!r}")
        layer = int(match.group("layer"))
        family = match.group("family")
        if layer not in self._parameters:
            raise RuntimeError(f"Prepared NVFP4 tensor targets non-local layer {layer}")
        if layer in self._completed or family in self._seen[layer]:
            raise RuntimeError(f"Duplicate/repeated prepared NVFP4 tensor {name!r}")
        if self._active_layer is None:
            self._active_layer = layer
            self._active_started = time.perf_counter()
        elif self._active_layer != layer:
            raise RuntimeError(
                "Prepared NVFP4 layer tensors are interleaved: "
                f"active={self._active_layer}, observed={layer}"
            )
        if _device_type(loaded_weight) != "cpu":
            raise RuntimeError(f"Prepared NVFP4 source {name!r} must be on CPU")
        if tuple(loaded_weight.shape) != self._source_shapes[family]:
            raise RuntimeError(
                f"Prepared NVFP4 source {name!r} shape drifted: "
                f"{tuple(loaded_weight.shape)} vs {self._source_shapes[family]}"
            )
        if loaded_weight.dtype != self._dtypes[family]:
            raise RuntimeError(f"Prepared NVFP4 source {name!r} dtype drifted")
        rank_source = loaded_weight[self._tp_rank]
        if not bool(rank_source.is_contiguous()):
            raise RuntimeError(f"Prepared NVFP4 rank slice {name!r} is not contiguous")
        parameter = self._parameters[layer][family]
        if (
            _device_type(parameter) != "cuda"
            or tuple(parameter.shape) != tuple(rank_source.shape)
            or parameter.dtype != rank_source.dtype
        ):
            raise RuntimeError(f"Prepared NVFP4 destination contract drifted for {name!r}")
        if self._direct_reader is None:
            parameter.data.copy_(rank_source)
        else:
            self._direct_reader.copy_into(
                layer=layer,
                family=family,
                destination=parameter,
            )
        self._copies += 1
        self._seen[layer].add(family)
        if len(self._seen[layer]) == len(PREPARED_FAMILY_ORDER):
            observed_bytes = sum(
                _tensor_bytes(self._parameters[layer][item])
                for item in PREPARED_FAMILY_ORDER
            )
            if observed_bytes != self._expected_rank_bytes:
                raise RuntimeError(
                    "Prepared NVFP4 rank bytes drifted: "
                    f"observed={observed_bytes}, expected={self._expected_rank_bytes}"
                )
            self._states[layer].loaded = True
            self._completed.add(layer)
            self._active_layer = None
            direct = (
                self._direct_reader.layer_stats(layer)
                if self._direct_reader is not None
                else {}
            )
            logger.info(
                "NVFP4_PREPARED event=layer_load layer=%d reads=%d copies=%d "
                "bytes=%d seconds=%.6f io_mode=%s read_syscalls=%d "
                "read_seconds=%.6f copy_seconds=%.6f",
                layer,
                EXPECTED_H2D_CALLS_PER_LAYER,
                EXPECTED_H2D_CALLS_PER_LAYER,
                observed_bytes,
                time.perf_counter() - self._active_started,
                "preadv" if direct else "mmap",
                int(direct.get("syscalls", 0)),
                float(direct.get("read_seconds", 0.0)),
                float(direct.get("copy_seconds", 0.0)),
            )
        basename = _FAMILY_TO_PARAMETER[family]
        return f"layers.{layer}.ffn.experts.routed_experts.{basename}"

    def finish(self) -> None:
        if self._active_layer is not None:
            missing = set(PREPARED_FAMILY_ORDER) - self._seen[self._active_layer]
            raise RuntimeError(
                f"Prepared NVFP4 layer {self._active_layer} is incomplete: "
                f"{sorted(missing)}"
            )
        expected = set(self._parameters)
        if self._completed != expected:
            raise RuntimeError(
                "Prepared NVFP4 load missed layers: "
                f"{sorted(expected - self._completed)}"
            )
        expected_copies = len(expected) * EXPECTED_H2D_CALLS_PER_LAYER
        if self._copies != expected_copies:
            raise RuntimeError(
                f"Prepared NVFP4 H2D count drifted: {self._copies} vs {expected_copies}"
            )
        direct = (
            self._direct_reader.summary()
            if self._direct_reader is not None
            else {}
        )
        if self._direct_reader is not None:
            self._direct_reader.finish()
        logger.info(
            "NVFP4_PREPARED event=complete layers=%d reads=%d copies=%d "
            "elapsed_seconds=%.6f io_mode=%s read_syscalls=%d "
            "read_seconds=%.6f copy_seconds=%.6f",
            len(expected),
            expected_copies,
            expected_copies,
            time.perf_counter() - self._started,
            "preadv" if direct else "mmap",
            int(direct.get("syscalls", 0)),
            float(direct.get("read_seconds", 0.0)),
            float(direct.get("copy_seconds", 0.0)),
        )

    def abort(self) -> None:
        if self._direct_reader is not None:
            self._direct_reader.close()
        for state in self._states.values():
            if not state.finalized:
                state.aborted = True
        self._active_layer = None


class Nvfp4PreparedLoadSession:
    def __init__(self) -> None:
        self._active = False
        self._requested = False
        self._loader: Nvfp4PreparedLayerLoader | None = None
        self._nested_load_calls = 0
        self._completed_noop_calls = 0

    @property
    def active(self) -> bool:
        return self._active

    @property
    def nested_load_calls(self) -> int:
        return self._nested_load_calls

    @property
    def completed_noop_calls(self) -> int:
        return self._completed_noop_calls

    def begin(
        self,
        loader: Nvfp4PreparedLayerLoader | None,
        *,
        prepared_requested: bool,
    ) -> None:
        if self._active:
            raise RuntimeError("Prepared NVFP4 load session reentry is not allowed")
        if (loader is not None) != prepared_requested:
            raise RuntimeError("Prepared NVFP4 session loader/request drifted")
        self._active = True
        self._requested = prepared_requested
        self._loader = loader
        self._nested_load_calls = 0
        self._completed_noop_calls = 0

    def loader_for_nested_load(
        self, *, prepared_requested: bool
    ) -> Nvfp4PreparedLayerLoader | None:
        if not self._active:
            if prepared_requested:
                raise RuntimeError("Prepared NVFP4 nested load lacks outer session")
            return None
        if prepared_requested != self._requested:
            raise RuntimeError("Prepared NVFP4 opt-in changed during loading")
        self._nested_load_calls += 1
        if self._loader is not None and self._loader.is_complete:
            # AutoWeightsLoader may enter DeepseekV4Model.load_weights again
            # for a trailing non-expert prefix group.  Once all prepared
            # layers are complete, returning None makes that invocation use
            # only the ordinary loader and structurally prevents a second
            # prepared commit or hook installation.
            self._completed_noop_calls += 1
            return None
        return self._loader

    def finish(self) -> None:
        if not self._active:
            raise RuntimeError("No active prepared NVFP4 session")
        if self._loader is not None:
            self._loader.finish()
        self._reset()

    def abort(self) -> None:
        try:
            if self._loader is not None:
                self._loader.abort()
        finally:
            self._reset()

    def _reset(self) -> None:
        self._active = False
        self._requested = False
        self._loader = None


def maybe_create_nvfp4_prepared_loader(
    *,
    torch_module: Any,
    checkpoint: str | os.PathLike[str],
    routed_layers: dict[int, Any],
    start_layer: int,
    end_layer: int,
    num_hidden_layers: int,
    num_routed_experts: int,
    tp_size: int,
    tp_rank: int,
    use_mega_moe: bool,
    enable_expert_parallel: bool,
    num_redundant_experts: int,
    load_format: str,
    quant_config: Any,
    environ: Mapping[str, str] | None = None,
    replace_parameter_fn: Callable[[Any, str, Any], None] | None = None,
) -> Nvfp4PreparedLayerLoader | None:
    source = os.environ if environ is None else environ
    contract = inspect_prepared_checkpoint(checkpoint, environ=source)
    if contract is None:
        return None
    direct_read = prepared_direct_read_requested(source)
    if str(load_format).lower().rsplit(".", 1)[-1] == "roce_tp":
        raise RuntimeError("Prepared NVFP4 loading does not support roce_tp")
    if use_mega_moe or enable_expert_parallel or num_redundant_experts != 0:
        raise RuntimeError("Prepared NVFP4 requires TP-only MoE with EP/EPLB disabled")
    if (
        num_hidden_layers != EXPECTED_LAYERS
        or start_layer != 0
        or end_layer != EXPECTED_LAYERS
        or num_routed_experts != EXPECTED_EXPERTS
        or tp_size != EXPECTED_TP_SIZE
        or set(routed_layers) != set(range(EXPECTED_LAYERS))
    ):
        raise RuntimeError("Prepared NVFP4 model topology drifted")
    if (
        quant_config.__class__.__name__ != "DeepseekV4FP8Config"
        or getattr(quant_config, "expert_dtype", None) != "fp4"
        or getattr(quant_config, "moe_quant_algo", None) != "NVFP4"
    ):
        raise RuntimeError("Prepared NVFP4 quantization config drifted")
    if replace_parameter_fn is None:
        from vllm.model_executor.utils import replace_parameter

        replace_parameter_fn = replace_parameter

    shapes = _destination_shapes()
    dtypes = _family_dtypes(torch_module)
    parameters: dict[int, dict[str, Any]] = {}
    states: dict[int, PreparedPostloadState] = {}
    if len({id(routed) for routed in routed_layers.values()}) != EXPECTED_LAYERS:
        raise RuntimeError("Prepared NVFP4 routed-layer ownership is not unique")
    if (
        len({id(routed.quant_method) for routed in routed_layers.values()})
        != EXPECTED_LAYERS
    ):
        raise RuntimeError("Prepared NVFP4 quant-method ownership is not per-layer")
    _validate_runtime_transform_sources(routed_layers[0])
    for layer, routed in routed_layers.items():
        for name in ("w13_weight_scale_2", "w13_input_scale"):
            old = getattr(routed, name)
            if (
                tuple(old.shape) != (EXPECTED_EXPERTS, 2)
                or old.dtype != torch_module.float32
                or _device_type(old) != "cuda"
            ):
                raise RuntimeError(f"Prepared NVFP4 raw scalar parameter drifted: {name}")
            replacement = torch_module.empty(
                (EXPECTED_EXPERTS,), dtype=torch_module.float32, device=old.device
            )
            replace_parameter_fn(routed, name, replacement)

        layer_parameters: dict[str, Any] = {}
        for family, basename in _FAMILY_TO_PARAMETER.items():
            parameter = getattr(routed, basename)
            if (
                tuple(parameter.shape) != shapes[family]
                or parameter.dtype != dtypes[family]
                or _device_type(parameter) != "cuda"
            ):
                raise RuntimeError(
                    f"Prepared NVFP4 destination {layer}/{family} drifted"
                )
            layer_parameters[family] = parameter
        state = PreparedPostloadState(layer)
        setattr(routed, "_dspark_nvfp4_prepared_state", state)
        _install_prepared_postload_hook(routed, state)
        states[layer] = state
        parameters[layer] = layer_parameters
    logger.info(
        "NVFP4_PREPARED event=enabled layers=%d reads_per_layer=%d "
        "copies_per_layer=%d rank_bytes=%d manifest_sha256=%s io_mode=%s",
        len(parameters),
        EXPECTED_H2D_CALLS_PER_LAYER,
        EXPECTED_H2D_CALLS_PER_LAYER,
        EXPECTED_RANK_BYTES,
        contract.manifest_sha256,
        "preadv" if direct_read else "mmap",
    )
    direct_reader = (
        PreparedSafetensorsDirectReader(
            torch_module=torch_module,
            contract=contract,
            tp_rank=tp_rank,
        )
        if direct_read
        else None
    )
    return Nvfp4PreparedLayerLoader(
        torch_module=torch_module,
        tp_rank=tp_rank,
        parameters=parameters,
        states=states,
        direct_reader=direct_reader,
    )
