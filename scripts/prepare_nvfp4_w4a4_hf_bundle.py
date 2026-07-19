#!/usr/bin/env python3
"""Build or verify the one-download prepared-NVFP4 + DSpark distribution.

The prepared target remains at the repository root.  The native three-stage
DSpark checkpoint is materialized below ``dspark/`` with a filtered index that
references only the three original MTP shards.  Shard payloads are copied
byte-for-byte; no tensor is converted or renamed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Any


BUNDLE_SCHEMA = "anemll.deepseek_v4.nvfp4_tp2_w4a4_dspark_bundle.v1"
BUNDLE_MANIFEST = "bundle-manifest.json"
BUNDLE_DIGEST = BUNDLE_MANIFEST + ".sha256"
PREPARED_MANIFEST = "dspark-nvfp4-tp2-repack.json"
PREPARED_DIGEST = PREPARED_MANIFEST + ".sha256"
PREPARED_BASE_SHA256 = (
    "972ba797456da80e586324a5a8c29af42bac86510ceff983e674de41d31e6f26"
)
DRAFT_CONFIG_SHA256 = (
    "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023"
)
DRAFT_INDEX_SHA256 = (
    "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
)
STAGES = {
    0: {
        "file": "model-00046-of-00048.safetensors",
        "count": 1568,
        "size": 3_610_455_184,
        "sha256": "14810f274692bb771c3970e8cba45846c4aa2213dcfb0025ffebe788d229e18d",
    },
    1: {
        "file": "model-00047-of-00048.safetensors",
        "count": 1565,
        "size": 3_560_111_960,
        "sha256": "7a44164698d90648a35c030c5eb369256d2c469306bfbf2b1ae27f35b6e57889",
    },
    2: {
        "file": "model-00048-of-00048.safetensors",
        "count": 1572,
        "size": 3_692_775_244,
        "sha256": "a0bbb24f36d2ef6107250088e0f020f93aec0677cd24be3e9e69589547a7656f",
    },
}
COPY_METADATA = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


class BundleError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BundleError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def safetensors_names_and_payload(path: Path) -> tuple[list[str], int]:
    with path.open("rb", buffering=0) as source:
        prefix = source.read(8)
        require(len(prefix) == 8, f"short safetensors prefix: {path}")
        header_len = struct.unpack("<Q", prefix)[0]
        require(2 <= header_len <= 128 * 1024 * 1024, f"bad header length: {path}")
        raw = source.read(header_len)
    require(len(raw) == header_len, f"short safetensors header: {path}")
    header = json.loads(raw)
    require(isinstance(header, dict), f"bad safetensors header: {path}")
    names = [name for name in header if name != "__metadata__"]
    spans: list[tuple[int, int]] = []
    for name in names:
        row = header[name]
        require(isinstance(row, dict), f"bad tensor row: {name}")
        offsets = row.get("data_offsets")
        require(
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(isinstance(v, int) for v in offsets),
            f"bad tensor offsets: {name}",
        )
        start, end = offsets
        require(0 <= start <= end, f"bad tensor span: {name}")
        spans.append((start, end))
    spans.sort()
    cursor = 0
    for start, end in spans:
        require(start == cursor, f"non-gapless payload: {path.name}")
        cursor = end
    require(8 + header_len + cursor == path.stat().st_size, f"payload size drift: {path}")
    return names, cursor


def validate_sources(bundle: Path, draft: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    prepared = bundle / PREPARED_MANIFEST
    require(prepared.is_file(), "prepared manifest missing")
    require((bundle / PREPARED_DIGEST).is_file(), "prepared manifest digest missing")
    require(sha256(prepared) == PREPARED_BASE_SHA256, "prepared base manifest drifted")
    require(sha256(draft / "config.json") == DRAFT_CONFIG_SHA256, "draft config drifted")
    require(sha256(draft / "model.safetensors.index.json") == DRAFT_INDEX_SHA256, "draft index drifted")
    source_index = read_json(draft / "model.safetensors.index.json")
    weight_map = source_index.get("weight_map")
    require(isinstance(weight_map, dict), "draft weight_map missing")
    mtp = {name: shard for name, shard in weight_map.items() if name.startswith("mtp.")}
    require(len(mtp) == 4705, "draft MTP tensor count drifted")
    require(
        Counter(name.split(".")[1] for name in mtp) == Counter({"0": 1568, "1": 1565, "2": 1572}),
        "draft MTP stage counts drifted",
    )
    payload_total = 0
    rows: dict[str, Any] = {}
    for stage, expected in STAGES.items():
        path = draft / str(expected["file"])
        require(path.is_file() and not path.is_symlink(), f"draft shard missing: {path}")
        require(path.stat().st_size == expected["size"], f"draft shard size drifted: {path.name}")
        require(sha256(path) == expected["sha256"], f"draft shard digest drifted: {path.name}")
        names, payload = safetensors_names_and_payload(path)
        expected_names = {name for name, shard in mtp.items() if shard == path.name}
        require(len(names) == expected["count"], f"draft stage count drifted: {stage}")
        require(set(names) == expected_names, f"draft header/index parity failed: {stage}")
        require(all(name.startswith(f"mtp.{stage}.") for name in names), f"mixed draft shard: {stage}")
        payload_total += payload
        rows[str(stage)] = {
            "file": path.name,
            "size": path.stat().st_size,
            "sha256": expected["sha256"],
            "tensor_count": len(names),
            "payload_bytes": payload,
        }
    return source_index, {"tensor_count": len(mtp), "payload_bytes": payload_total, "stages": rows}


def replace_card_and_rebind_manifest(bundle: Path, card: Path) -> tuple[str, str]:
    manifest_path = bundle / PREPARED_MANIFEST
    old_sha = sha256(manifest_path)
    require(old_sha == PREPARED_BASE_SHA256, "prepared base manifest changed before card replacement")
    shutil.copy2(card, bundle / "README.md")
    manifest = read_json(manifest_path)
    rows = manifest.get("output", {}).get("copied_metadata_files")
    require(isinstance(rows, list), "prepared copied metadata list missing")
    matches = [row for row in rows if isinstance(row, dict) and row.get("path") == "README.md"]
    require(len(matches) == 1, "prepared README manifest row missing")
    readme = bundle / "README.md"
    matches[0]["size"] = readme.stat().st_size
    matches[0]["sha256"] = sha256(readme)
    write_json_atomic(manifest_path, manifest)
    new_sha = sha256(manifest_path)
    (bundle / PREPARED_DIGEST).write_text(f"{new_sha}  {PREPARED_MANIFEST}\n", encoding="ascii")
    return old_sha, new_sha


def build(args: argparse.Namespace) -> dict[str, Any]:
    bundle = args.bundle.resolve()
    draft = args.draft_source.resolve()
    card = args.card.resolve()
    require(bundle.is_dir(), "bundle root missing")
    require(card.is_file(), "placeholder card missing")
    require(not (bundle / "dspark").exists(), "dspark directory already exists")
    partial = bundle / "dspark.partial"
    require(not partial.exists(), "dspark partial already exists")
    source_index, draft_proof = validate_sources(bundle, draft)
    partial.mkdir()
    try:
        for name in COPY_METADATA:
            source = draft / name
            require(source.is_file() and not source.is_symlink(), f"draft metadata missing: {name}")
            shutil.copy2(source, partial / name)
        mtp_map = {
            name: shard
            for name, shard in source_index["weight_map"].items()
            if name.startswith("mtp.")
        }
        filtered = {
            "metadata": {
                "total_size": draft_proof["payload_bytes"],
                "format": "deepseek_v4_dspark_mtp_only_v1",
                "source_index_sha256": DRAFT_INDEX_SHA256,
            },
            "weight_map": mtp_map,
        }
        write_json_atomic(partial / "model.safetensors.index.json", filtered)
        for expected in STAGES.values():
            source = draft / str(expected["file"])
            destination = partial / source.name
            shutil.copyfile(source, destination)
            shutil.copystat(source, destination)
            require(destination.stat().st_size == expected["size"], f"copy size mismatch: {source.name}")
            require(sha256(destination) == expected["sha256"], f"copy digest mismatch: {source.name}")
        os.replace(partial, bundle / "dspark")
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    old_manifest_sha, new_manifest_sha = replace_card_and_rebind_manifest(bundle, card)
    result = verify_bundle(bundle)
    result["prepared_base_manifest_sha256"] = old_manifest_sha
    result["prepared_release_manifest_sha256"] = new_manifest_sha
    write_json_atomic(bundle / BUNDLE_MANIFEST, result)
    digest = sha256(bundle / BUNDLE_MANIFEST)
    (bundle / BUNDLE_DIGEST).write_text(f"{digest}  {BUNDLE_MANIFEST}\n", encoding="ascii")
    result["bundle_manifest_sha256"] = digest
    return result


def verify_bundle(bundle: Path) -> dict[str, Any]:
    bundle = bundle.resolve()
    draft = bundle / "dspark"
    require(bundle.is_dir() and draft.is_dir(), "bundle layout missing")
    manifest = read_json(bundle / PREPARED_MANIFEST)
    prepared_sha = sha256(bundle / PREPARED_MANIFEST)
    sidecar = (bundle / PREPARED_DIGEST).read_text(encoding="ascii").split()
    require(len(sidecar) == 2 and sidecar[0] == prepared_sha, "prepared manifest sidecar drifted")
    copied = manifest.get("output", {}).get("copied_metadata_files")
    require(isinstance(copied, list), "prepared metadata manifest missing")
    for row in copied:
        path = bundle / row["path"]
        require(path.stat().st_size == row["size"], f"prepared metadata size drifted: {path.name}")
        require(sha256(path) == row["sha256"], f"prepared metadata digest drifted: {path.name}")
    source_index = read_json(draft / "model.safetensors.index.json")
    weight_map = source_index.get("weight_map")
    require(isinstance(weight_map, dict) and len(weight_map) == 4705, "filtered draft index drifted")
    require(all(name.startswith("mtp.") for name in weight_map), "non-MTP draft key present")
    rows: dict[str, Any] = {}
    payload_total = 0
    for stage, expected in STAGES.items():
        path = draft / str(expected["file"])
        require(path.is_file() and not path.is_symlink(), f"bundle draft shard missing: {path.name}")
        require(path.stat().st_size == expected["size"], f"bundle draft size drifted: {path.name}")
        observed_sha = sha256(path)
        require(observed_sha == expected["sha256"], f"bundle draft hash drifted: {path.name}")
        names, payload = safetensors_names_and_payload(path)
        expected_names = {name for name, shard in weight_map.items() if shard == path.name}
        require(set(names) == expected_names, f"bundle draft header/index mismatch: {stage}")
        payload_total += payload
        rows[str(stage)] = {
            "file": path.name,
            "size": path.stat().st_size,
            "sha256": observed_sha,
            "tensor_count": len(names),
            "payload_bytes": payload,
        }
    metadata = source_index.get("metadata")
    require(isinstance(metadata, dict), "filtered draft metadata missing")
    require(metadata.get("total_size") == payload_total, "filtered draft total_size drifted")
    target_output = manifest.get("output")
    require(isinstance(target_output, dict), "prepared output manifest missing")
    return {
        "schema": BUNDLE_SCHEMA,
        "name": bundle.name,
        "layout": {"target": ".", "draft": "dspark"},
        "target": {
            "format": manifest.get("format"),
            "manifest": PREPARED_MANIFEST,
            "manifest_sha256": prepared_sha,
            "tensor_count": target_output.get("tensor_count"),
            "payload_bytes": target_output.get("payload_bytes"),
            "root_safetensors_files": len(target_output.get("files", [])),
        },
        "draft": {
            "format": metadata.get("format"),
            "source_config_sha256": DRAFT_CONFIG_SHA256,
            "source_index_sha256": DRAFT_INDEX_SHA256,
            "filtered_index_sha256": sha256(draft / "model.safetensors.index.json"),
            "tensor_count": len(weight_map),
            "payload_bytes": payload_total,
            "stages": rows,
        },
        "runtime": {
            "target_path": ".",
            "draft_path": "dspark",
            "tp_size": 2,
            "target_precision": "NVFP4 W4A4",
            "draft_precision": "native MXFP4",
        },
        "status": {
            "prepared_target_validated": True,
            "target_only_prefill_validated": True,
            "combined_dspark_serving": "experimental_pending_final_validation",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--bundle", type=Path, required=True)
    build_parser.add_argument("--draft-source", type=Path, required=True)
    build_parser.add_argument("--card", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "build":
            result = build(args)
        else:
            result = verify_bundle(args.bundle)
    except (BundleError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
