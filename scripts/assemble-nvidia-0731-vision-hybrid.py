#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Assemble the pinned NVIDIA 0731 NVFP4 + Vision-Exp test checkpoint.

The NVIDIA shards are hard-linked into a new directory.  Only the 316
Vision-Exp-specific tensors are copied into a small additional safetensors
shard.  Neither source checkpoint is modified.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
from typing import BinaryIO


NVIDIA_MODEL_ID = "nvidia/DeepSeek-V4-Flash-0731-NVFP4"
NVIDIA_REVISION = "f1caa71142bd0be02f728c79f75042ac1e461579"
VISION_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
VISION_REVISION = "6821d6ad3681a4b137b066b76094fa82ebd0a380"
NVIDIA_CONFIG_SHA256 = "bb0d2286d6761439e41d3cef31d16489411b816ed8688922f59730bbd5567cdb"
NVIDIA_INDEX_SHA256 = "5d2ad3076e04081d6c0728cb4b004dc832850ec5ae732f3adb06cf87c4b437a5"
VISION_CONFIG_SHA256 = "6cd841bdd6702f5e2ac34671bc78047ed80817102465525ae2a41c502abbcd75"
VISION_INDEX_SHA256 = "507977e3d3818865264e68c0fdab139aa7f3929d0d0cf693dacc47428da56395"
VISION_TENSOR_COUNT = 316
VISION_TENSOR_NAMES_SHA256 = "7a941d04caeb96f06925d08d815a081596142e64dfbca7bd98c0cb78cc3663b7"
VISION_PAYLOAD_BYTES = 932_836_352
DONOR_SHARD = "model-vision-donor.safetensors"
MANIFEST_FILE = "HYBRID_MANIFEST.json"

VISION_CONFIG_FIELDS = (
    "vision_dim",
    "vision_downsample_ratio",
    "vision_inter_dim",
    "vision_max_n_token",
    "vision_max_wh_ratio",
    "vision_min_pixels",
    "vision_n_heads",
    "vision_n_layers",
    "vision_patch_size",
    "vision_rope_theta",
)
SUPPORT_FILES = (
    ".gitattributes",
    "LICENSE",
    "generation_config.json",
    "hf_quant_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


@dataclass(frozen=True)
class Pins:
    nvidia_config: str = NVIDIA_CONFIG_SHA256
    nvidia_index: str = NVIDIA_INDEX_SHA256
    vision_config: str = VISION_CONFIG_SHA256
    vision_index: str = VISION_INDEX_SHA256
    tensor_count: int = VISION_TENSOR_COUNT
    tensor_names: str = VISION_TENSOR_NAMES_SHA256
    payload_bytes: int = VISION_PAYLOAD_BYTES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require_sha256(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}: {actual} != {expected}")


def is_vision_tensor(name: str) -> bool:
    return (
        name.startswith(("vision.", "aligner."))
        or name in {"image_start", "image_pad", "image_newline", "image_end"}
        or name.endswith(".ffn.gate.bias_vl")
        or name
        in {
            "layers.0.ffn.gate.bias",
            "layers.1.ffn.gate.bias",
            "layers.2.ffn.gate.bias",
        }
    )


def tensor_names_digest(names: list[str]) -> str:
    return hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest()


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    size = path.stat().st_size
    with path.open("rb") as source:
        raw = source.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated safetensors prefix: {path}")
        header_len = struct.unpack("<Q", raw)[0]
        if header_len > 32 * 1024 * 1024 or 8 + header_len > size:
            raise ValueError(f"invalid safetensors header length: {path}")
        header = json.loads(source.read(header_len))
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors header: {path}")
    return header, 8 + header_len


def copy_range(source: BinaryIO, target: BinaryIO, count: int) -> None:
    remaining = count
    while remaining:
        chunk = source.read(min(8 * 1024 * 1024, remaining))
        if not chunk:
            raise ValueError("source safetensors payload ended early")
        target.write(chunk)
        remaining -= len(chunk)


def collect_selected_tensors(
    vision_dir: Path, vision_weight_map: dict[str, str], selected: list[str]
) -> tuple[dict[str, dict], int]:
    headers: dict[str, tuple[dict, int]] = {}
    entries: dict[str, dict] = {}
    payload_bytes = 0
    for name in selected:
        shard = vision_weight_map[name]
        if not isinstance(shard, str) or Path(shard).name != shard:
            raise ValueError(f"unsafe donor shard name for {name}: {shard!r}")
        if shard not in headers:
            headers[shard] = read_safetensors_header(vision_dir / shard)
        header, _ = headers[shard]
        info = header.get(name)
        if not isinstance(info, dict):
            raise ValueError(f"missing donor tensor header: {name}")
        offsets = info.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(x, int) for x in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
        ):
            raise ValueError(f"invalid donor offsets: {name}")
        data_start = headers[shard][1]
        if data_start + offsets[1] > (vision_dir / shard).stat().st_size:
            raise ValueError(f"donor tensor outside shard: {name}")
        entries[name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "source_shard": shard,
            "source_offsets": offsets,
            "source_data_start": data_start,
        }
        payload_bytes += offsets[1] - offsets[0]
    return entries, payload_bytes


def write_donor_shard(
    vision_dir: Path, destination: Path, entries: dict[str, dict]
) -> tuple[str, int]:
    output_header: dict[str, dict] = {
        "__metadata__": {
            "format": "pt",
            "source": VISION_MODEL_ID,
            "source_revision": VISION_REVISION,
        }
    }
    offset = 0
    for name in sorted(entries):
        item = entries[name]
        length = item["source_offsets"][1] - item["source_offsets"][0]
        output_header[name] = {
            "dtype": item["dtype"],
            "shape": item["shape"],
            "data_offsets": [offset, offset + length],
        }
        offset += length

    encoded = json.dumps(output_header, separators=(",", ":"), ensure_ascii=False).encode()
    encoded += b" " * (-len(encoded) % 8)
    with destination.open("xb") as target:
        target.write(struct.pack("<Q", len(encoded)))
        target.write(encoded)
        for name in sorted(entries):
            item = entries[name]
            with (vision_dir / item["source_shard"]).open("rb") as source:
                source.seek(item["source_data_start"] + item["source_offsets"][0])
                copy_range(
                    source,
                    target,
                    item["source_offsets"][1] - item["source_offsets"][0],
                )
    return sha256_file(destination), offset


def validate_sources(
    nvidia_dir: Path, vision_dir: Path, pins: Pins
) -> tuple[dict, dict, dict, dict, list[str], dict[str, dict], int]:
    for directory in (nvidia_dir, vision_dir):
        if not directory.is_dir():
            raise ValueError(f"checkpoint directory not found: {directory}")
    for path, expected in (
        (nvidia_dir / "config.json", pins.nvidia_config),
        (nvidia_dir / "model.safetensors.index.json", pins.nvidia_index),
        (vision_dir / "config.json", pins.vision_config),
        (vision_dir / "model.safetensors.index.json", pins.vision_index),
    ):
        require_sha256(path, expected)

    nvidia_config = load_json(nvidia_dir / "config.json")
    vision_config = load_json(vision_dir / "config.json")
    nvidia_index = load_json(nvidia_dir / "model.safetensors.index.json")
    vision_index = load_json(vision_dir / "model.safetensors.index.json")
    nvidia_map = nvidia_index.get("weight_map")
    vision_map = vision_index.get("weight_map")
    if not isinstance(nvidia_map, dict) or not isinstance(vision_map, dict):
        raise ValueError("checkpoint indexes must contain weight_map objects")

    selected = sorted(name for name in vision_map if is_vision_tensor(name))
    if len(selected) != pins.tensor_count:
        raise ValueError(f"unexpected vision tensor count: {len(selected)}")
    if tensor_names_digest(selected) != pins.tensor_names:
        raise ValueError("vision tensor-name manifest does not match the pinned donor")
    collisions = sorted(set(selected) & set(nvidia_map))
    if collisions:
        raise ValueError(f"vision tensors already present in NVIDIA checkpoint: {collisions[:3]}")

    for field in (
        "hidden_size",
        "num_hidden_layers",
        "n_routed_experts",
        "vocab_size",
        "hc_mult",
        "dspark_block_size",
        "dspark_target_layer_ids",
    ):
        if nvidia_config.get(field) != vision_config.get(field):
            raise ValueError(f"structural mismatch in {field}")
    entries, payload_bytes = collect_selected_tensors(
        vision_dir, vision_map, selected
    )
    if payload_bytes != pins.payload_bytes:
        raise ValueError(f"unexpected donor payload size: {payload_bytes}")

    shard_names = sorted(set(nvidia_map.values()))
    if len(shard_names) != 48:
        raise ValueError(f"unexpected NVIDIA shard count: {len(shard_names)}")
    for shard in shard_names:
        if not isinstance(shard, str) or Path(shard).name != shard:
            raise ValueError(f"unsafe NVIDIA shard name: {shard!r}")
        if not (nvidia_dir / shard).is_file():
            raise ValueError(f"missing NVIDIA shard: {shard}")
    return (
        nvidia_config,
        vision_config,
        nvidia_index,
        vision_index,
        selected,
        entries,
        payload_bytes,
    )


def build_config(nvidia_config: dict, vision_config: dict) -> dict:
    result = dict(nvidia_config)
    for field in VISION_CONFIG_FIELDS:
        if field not in vision_config:
            raise ValueError(f"vision donor config is missing {field}")
        result[field] = vision_config[field]
    result["anemll_hybrid"] = {
        "experimental": True,
        "language_model": NVIDIA_MODEL_ID,
        "language_revision": NVIDIA_REVISION,
        "vision_model": VISION_MODEL_ID,
        "vision_revision": VISION_REVISION,
        "vision_tensor_count": VISION_TENSOR_COUNT,
    }
    # These language/draft choices are intentionally inherited from NVIDIA.
    result["rms_norm_eps"] = nvidia_config["rms_norm_eps"]
    result["num_nextn_predict_layers"] = nvidia_config[
        "num_nextn_predict_layers"
    ]
    return result


def assemble(
    nvidia_dir: Path,
    vision_dir: Path,
    output_dir: Path,
    pins: Pins = Pins(),
    *,
    dry_run: bool = False,
) -> dict:
    (
        nvidia_config,
        vision_config,
        nvidia_index,
        _vision_index,
        selected,
        entries,
        payload_bytes,
    ) = validate_sources(nvidia_dir, vision_dir, pins)

    summary = {
        "status": "validated" if dry_run else "assembled",
        "nvidia_model": NVIDIA_MODEL_ID,
        "nvidia_revision": NVIDIA_REVISION,
        "vision_model": VISION_MODEL_ID,
        "vision_revision": VISION_REVISION,
        "vision_tensor_count": len(selected),
        "vision_tensor_names_sha256": tensor_names_digest(selected),
        "vision_payload_bytes": payload_bytes,
        "nvidia_shard_count": len(set(nvidia_index["weight_map"].values())),
        "link_mode": "hardlink",
    }
    if dry_run:
        return summary
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite candidate: {output_dir}")
    if output_dir.parent.stat().st_dev != nvidia_dir.stat().st_dev:
        raise ValueError("output and NVIDIA checkpoint must share a filesystem")

    temporary = output_dir.parent / f".{output_dir.name}.assembling-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"temporary assembly path exists: {temporary}")
    temporary.mkdir(mode=0o755)
    try:
        nvidia_map = nvidia_index["weight_map"]
        for shard in sorted(set(nvidia_map.values())):
            os.link(nvidia_dir / shard, temporary / shard)
        for name in SUPPORT_FILES:
            source = nvidia_dir / name
            if source.is_file():
                shutil.copy2(source, temporary / name)

        donor_sha256, written_payload = write_donor_shard(
            vision_dir, temporary / DONOR_SHARD, entries
        )
        if written_payload != payload_bytes:
            raise ValueError("donor shard payload length changed during extraction")

        hybrid_config = build_config(nvidia_config, vision_config)
        (temporary / "config.json").write_text(
            json.dumps(hybrid_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        hybrid_map = dict(nvidia_map)
        hybrid_map.update({name: DONOR_SHARD for name in selected})
        total_size = int(nvidia_index.get("metadata", {}).get("total_size", 0))
        hybrid_index = {
            "metadata": {"total_size": total_size + payload_bytes},
            "weight_map": hybrid_map,
        }
        (temporary / "model.safetensors.index.json").write_text(
            json.dumps(hybrid_index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary.update(
            {
                "donor_shard": DONOR_SHARD,
                "donor_shard_sha256": donor_sha256,
                "config_sha256": sha256_file(temporary / "config.json"),
                "index_sha256": sha256_file(
                    temporary / "model.safetensors.index.json"
                ),
                "total_tensor_payload_bytes": total_size + payload_bytes,
            }
        )
        (temporary / MANIFEST_FILE).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvidia-dir", type=Path, required=True)
    parser.add_argument("--vision-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = assemble(
            args.nvidia_dir.resolve(),
            args.vision_dir.resolve(),
            args.output_dir.resolve(),
            dry_run=args.dry_run,
        )
    except Exception as error:
        print(f"assembly failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
