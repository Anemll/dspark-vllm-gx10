#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify a prepared NVFP4 target + separate native DSpark draft contract.

The verifier reads only small JSON metadata and the JSON headers of the three
native MTP safetensors shards. It never reads tensor payload bytes. Besides
cross-checking the two configs, it derives a conservative TP=2 resident-byte
projection: routed-expert payload is partitioned across TP ranks while every
non-expert MTP tensor is charged in full to each rank.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from typing import Any, Mapping


PREPARED_SCHEMA = "dspark.deepseek_v4.nvfp4.tp2_cutlass_prepared.v1"
PREPARED_LOADER_CONTRACT = "deepseek_v4_nvfp4_tp2_cutlass_prepared_v1"
PREPARED_BACKEND = "FLASHINFER_CUTLASS"
PREPARED_MANIFEST = "dspark-nvfp4-tp2-repack.json"
PREPARED_MANIFEST_DIGEST = PREPARED_MANIFEST + ".sha256"
PINNED_VLLM = "752a3a504485790a2e8491cacbb35c137339ad34"
EXPECTED_DRAFT_TOTAL_SIZE = 166_878_536_440
EXPECTED_DRAFT_TENSOR_COUNT = 72_317
EXPECTED_STAGE_COUNTS = {0: 1_568, 1: 1_565, 2: 1_572}
EXPECTED_STAGE_SHARDS = {
    0: "model-00046-of-00048.safetensors",
    1: "model-00047-of-00048.safetensors",
    2: "model-00048-of-00048.safetensors",
}
EXPECTED_DRAFT_SHARD_SHA256 = {
    0: "14810f274692bb771c3970e8cba45846c4aa2213dcfb0025ffebe788d229e18d",
    1: "7a44164698d90648a35c030c5eb369256d2c469306bfbf2b1ae27f35b6e57889",
    2: "a0bbb24f36d2ef6107250088e0f020f93aec0677cd24be3e9e69589547a7656f",
}
EXPECTED_EXPERT_TENSORS_PER_STAGE = 1_536
EXPECTED_EXPERT_SUFFIX_COUNTS = Counter({"weight": 768, "scale": 768})
DTYPE_BYTES = {
    "BF16": 2,
    "F32": 4,
    "F8_E4M3": 1,
    "F8_E8M0": 1,
    "I8": 1,
}
EXPECTED_DSPARK_FIELDS = {
    "dspark_block_size": 5,
    "dspark_noise_token_id": 128_799,
    "dspark_target_layer_ids": [40, 41, 42],
    "dspark_markov_rank": 256,
}
ALLOWED_CONFIG_DIFFERENCES = {
    "compress_ratios",
    "dspark_block_size",
    "dspark_noise_token_id",
    "dspark_target_layer_ids",
    "dspark_markov_rank",
    "dspark_nvfp4_prepared",
    "quantization_config",
}
MTP_RE = re.compile(r"^mtp\.([0-2])\.")
MAX_HEADER_BYTES = 128 * 1024 * 1024
GIB = 1 << 30


class ContractError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sha256_sidecar(path: Path, expected_name: str) -> str:
    """Read either a bare digest or standard sha256sum output."""
    require(path.is_file() and not path.is_symlink(), "prepared manifest digest is invalid")
    try:
        fields = path.read_text(encoding="ascii").strip().split()
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"cannot read prepared manifest digest: {exc}") from exc
    require(len(fields) in {1, 2}, "prepared manifest digest sidecar is malformed")
    digest = fields[0].lower()
    require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, "prepared manifest digest is malformed")
    if len(fields) == 2:
        require(fields[1].lstrip("*") == expected_name, "prepared manifest digest filename drifted")
    return digest


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{label} is not a file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {label}: {exc}") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    require(path.is_file() and not path.is_symlink(), f"draft shard invalid: {path}")
    file_size = path.stat().st_size
    with path.open("rb", buffering=0) as source:
        prefix = source.read(8)
        require(len(prefix) == 8, f"short safetensors prefix: {path.name}")
        header_len = struct.unpack("<Q", prefix)[0]
        require(2 <= header_len <= MAX_HEADER_BYTES, f"invalid header size: {path.name}")
        require(8 + header_len <= file_size, f"header exceeds shard: {path.name}")
        raw = source.read(header_len)
        require(len(raw) == header_len, f"short safetensors header: {path.name}")
    try:
        header = json.loads(raw, object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid safetensors header {path.name}: {exc}") from exc
    require(isinstance(header, dict), f"header is not an object: {path.name}")
    return header, file_size - 8 - header_len


def _validate_prepared_target(
    target: Path, expected_manifest_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_path = target / "config.json"
    index_path = target / "model.safetensors.index.json"
    manifest_path = target / PREPARED_MANIFEST
    digest_path = target / PREPARED_MANIFEST_DIGEST
    config = read_json(config_path, "target config")
    index = read_json(index_path, "target index")
    manifest = read_json(manifest_path, "target prepared manifest")
    observed_manifest_sha = sha256(manifest_path)
    require(
        observed_manifest_sha == expected_manifest_sha256,
        "prepared manifest SHA-256 drifted",
    )
    require(
        read_sha256_sidecar(digest_path, PREPARED_MANIFEST) == observed_manifest_sha,
        "prepared manifest digest sidecar drifted",
    )
    marker = config.get("dspark_nvfp4_prepared")
    require(isinstance(marker, dict), "target lacks prepared marker")
    expected_marker = {
        "schema": PREPARED_SCHEMA,
        "loader_contract": PREPARED_LOADER_CONTRACT,
        "required_backend": PREPARED_BACKEND,
        "tp_size": 2,
        "vllm_layout_pin": PINNED_VLLM,
        "manifest": PREPARED_MANIFEST,
        "manifest_digest": PREPARED_MANIFEST_DIGEST,
    }
    for key, value in expected_marker.items():
        require(marker.get(key) == value, f"prepared marker drifted: {key}")
    loader = manifest.get("loader")
    output = manifest.get("output")
    integrity = manifest.get("integrity")
    require(isinstance(loader, dict), "prepared manifest loader is missing")
    require(isinstance(output, dict), "prepared manifest output is missing")
    require(isinstance(integrity, dict), "prepared manifest integrity is missing")
    require(manifest.get("format") == PREPARED_SCHEMA, "prepared manifest format drifted")
    require(output.get("config_sha256") == sha256(config_path), "manifest/config binding drifted")
    require(output.get("index_sha256") == sha256(index_path), "manifest/index binding drifted")
    require(output.get("tensor_count") == len(index.get("weight_map", {})), "manifest tensor count drifted")
    require(output.get("layer_file_count") == 43, "prepared layer-file count drifted")
    require(integrity.get("output_files_hashed") is True, "prepared file hashing proof missing")
    require(integrity.get("output_tensors_hashed") is True, "prepared tensor hashing proof missing")
    require(loader.get("required_backend") == PREPARED_BACKEND, "backend drifted")
    require(loader.get("required_runtime_transforms") == [], "runtime transforms present")
    require(loader.get("runtime_h2d_calls_per_layer") == 8, "H2D contract drifted")
    require(loader.get("runtime_source_reads_per_layer") == 8, "read contract drifted")
    quant = config.get("quantization_config")
    require(isinstance(quant, dict), "target quantization config missing")
    require(quant.get("moe_quant_algo") == "NVFP4", "target is not NVFP4")
    weight_map = index.get("weight_map")
    require(isinstance(weight_map, dict) and weight_map, "target weight map missing")
    return config, index, {
        "config_sha256": sha256(config_path),
        "index_sha256": sha256(index_path),
        "manifest_sha256": observed_manifest_sha,
        "physical_tensor_count": len(weight_map),
        "required_backend": PREPARED_BACKEND,
        "tp_size": 2,
    }


def _validate_draft(
    draft: Path,
    expected_config_sha256: str,
    expected_index_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_path = draft / "config.json"
    index_path = draft / "model.safetensors.index.json"
    config = read_json(config_path, "draft config")
    index = read_json(index_path, "draft index")
    require(sha256(config_path) == expected_config_sha256, "draft config SHA-256 drifted")
    require(sha256(index_path) == expected_index_sha256, "draft index SHA-256 drifted")
    metadata = index.get("metadata")
    weight_map = index.get("weight_map")
    require(isinstance(metadata, dict), "draft index metadata missing")
    require(isinstance(weight_map, dict), "draft weight map missing")
    require(metadata.get("total_size") == EXPECTED_DRAFT_TOTAL_SIZE, "draft total size drifted")
    require(len(weight_map) == EXPECTED_DRAFT_TENSOR_COUNT, "draft tensor count drifted")

    names_by_stage: dict[int, list[str]] = {stage: [] for stage in EXPECTED_STAGE_COUNTS}
    for name in weight_map:
        match = MTP_RE.match(name)
        if match:
            names_by_stage[int(match.group(1))].append(name)
        elif name.startswith("mtp."):
            raise ContractError(f"unexpected MTP stage: {name}")

    stage_rows: dict[str, Any] = {}
    total_expert_bytes = 0
    total_nonexpert_bytes = 0
    for stage, expected_count in EXPECTED_STAGE_COUNTS.items():
        names = names_by_stage[stage]
        require(len(names) == expected_count, f"mtp.{stage} tensor count drifted")
        shards = {weight_map[name] for name in names}
        expected_shard = EXPECTED_STAGE_SHARDS[stage]
        require(shards == {expected_shard}, f"mtp.{stage} shard placement drifted")
        shard_path = draft / expected_shard
        shard_sha = sha256(shard_path)
        require(
            shard_sha == EXPECTED_DRAFT_SHARD_SHA256[stage],
            f"mtp.{stage} shard SHA-256 drifted",
        )
        header, available_payload = read_safetensors_header(shard_path)
        tensor_names = {name for name in header if name != "__metadata__"}
        require(tensor_names == set(names), f"mtp.{stage} header/index parity failed")
        expert_bytes = 0
        nonexpert_bytes = 0
        expert_suffixes: Counter[str] = Counter()
        expert_dtypes: dict[str, set[str]] = {"weight": set(), "scale": set()}
        ranges: list[tuple[int, int, str]] = []
        for name in names:
            row = header.get(name)
            require(isinstance(row, dict), f"invalid header row: {name}")
            offsets = row.get("data_offsets")
            require(
                isinstance(offsets, list)
                and len(offsets) == 2
                and all(isinstance(value, int) for value in offsets)
                and 0 <= offsets[0] <= offsets[1],
                f"invalid offsets: {name}",
            )
            payload_bytes = offsets[1] - offsets[0]
            require(payload_bytes > 0, f"empty tensor payload: {name}")
            require(offsets[1] <= available_payload, f"tensor exceeds shard: {name}")
            dtype = row.get("dtype")
            shape = row.get("shape")
            require(dtype in DTYPE_BYTES, f"unsupported dtype: {name}")
            require(
                isinstance(shape, list)
                and all(isinstance(dim, int) and dim >= 0 for dim in shape),
                f"invalid shape: {name}",
            )
            expected_bytes = DTYPE_BYTES[str(dtype)] * math.prod(shape)
            require(payload_bytes == expected_bytes, f"dtype/shape byte drift: {name}")
            ranges.append((offsets[0], offsets[1], name))
            if ".ffn.experts." in name:
                expert_bytes += payload_bytes
                suffix = name.rsplit(".", 1)[-1]
                expert_suffixes[suffix] += 1
                if suffix in expert_dtypes:
                    expert_dtypes[suffix].add(str(dtype))
            else:
                nonexpert_bytes += payload_bytes
        require(
            sum(expert_suffixes.values()) == EXPECTED_EXPERT_TENSORS_PER_STAGE,
            f"mtp.{stage} expert count drifted",
        )
        require(
            expert_suffixes == EXPECTED_EXPERT_SUFFIX_COUNTS,
            f"mtp.{stage} expert family drifted",
        )
        require(expert_dtypes == {"weight": {"I8"}, "scale": {"F8_E8M0"}},
                f"mtp.{stage} expert dtype drifted")
        cursor = 0
        for start, end, name in sorted(ranges):
            require(start == cursor, f"non-gapless payload before {name}")
            cursor = end
        require(cursor == available_payload, f"mtp.{stage} payload coverage drifted")
        total_expert_bytes += expert_bytes
        total_nonexpert_bytes += nonexpert_bytes
        stage_rows[str(stage)] = {
            "tensor_count": len(names),
            "shard": expected_shard,
            "shard_sha256": shard_sha,
            "payload_gapless": True,
            "dtype_shape_bytes_exact": True,
            "expert_tensor_count": sum(expert_suffixes.values()),
            "expert_payload_bytes": expert_bytes,
            "nonexpert_payload_bytes": nonexpert_bytes,
            "payload_bytes": expert_bytes + nonexpert_bytes,
            "expert_suffix_counts": dict(sorted(expert_suffixes.items())),
            "expert_dtypes": {key: sorted(value) for key, value in expert_dtypes.items()},
        }
    return config, index, {
        "config_sha256": expected_config_sha256,
        "index_sha256": expected_index_sha256,
        "tensor_count": len(weight_map),
        "total_size": metadata["total_size"],
        "stages": stage_rows,
        "expert_payload_bytes": total_expert_bytes,
        "nonexpert_payload_bytes": total_nonexpert_bytes,
        "mtp_payload_bytes": total_expert_bytes + total_nonexpert_bytes,
    }


def _validate_cross_config(target: Mapping[str, Any], draft: Mapping[str, Any]) -> dict[str, Any]:
    differing: list[str] = []
    for key in sorted(set(target) | set(draft)):
        if target.get(key) != draft.get(key):
            differing.append(key)
            require(key in ALLOWED_CONFIG_DIFFERENCES, f"incompatible config field: {key}")
    require(set(differing) == ALLOWED_CONFIG_DIFFERENCES, "expected config differences drifted")
    for key, value in EXPECTED_DSPARK_FIELDS.items():
        require(key not in target, f"prepared target unexpectedly owns {key}")
        require(draft.get(key) == value, f"draft field drifted: {key}")
    target_ratios = target.get("compress_ratios")
    draft_ratios = draft.get("compress_ratios")
    require(isinstance(target_ratios, list) and len(target_ratios) == 44,
            "target compress ratios drifted")
    require(isinstance(draft_ratios, list) and len(draft_ratios) == 46,
            "draft compress ratios drifted")
    require(draft_ratios[:44] == target_ratios and draft_ratios[44:] == [0, 0],
            "draft compress-ratio extension drifted")
    draft_quant = draft.get("quantization_config")
    require(isinstance(draft_quant, dict), "draft quantization config missing")
    require(not draft_quant.get("moe_quant_algo"), "draft unexpectedly requests NVFP4")
    require(target.get("expert_dtype") == draft.get("expert_dtype") == "fp4",
            "expert dtype drifted")
    return {
        "passed": True,
        "equal_fields": sorted((set(target) | set(draft)) - set(differing)),
        "allowed_differences": differing,
        "draft_config_source_is_distinct": True,
        "target_quantization": "ModelOpt NVFP4 / prepared CUTLASS",
        "draft_quantization": "native MXFP4",
    }


def _memory_projection(
    draft_summary: Mapping[str, Any],
    *,
    tp_size: int,
    usable_memory_gib_per_rank: float,
    target_only_model_gib: float,
    observed_target_only_kv_gib: float,
    loader_overhead_fraction: float,
    graph_workspace_reserve_gib: float,
    system_safety_reserve_gib: float,
    minimum_kv_gib: float,
) -> dict[str, Any]:
    require(tp_size == 2, "only the proven TP=2 contract is supported")
    require(
        usable_memory_gib_per_rank > 0
        and target_only_model_gib > 0
        and observed_target_only_kv_gib > 0,
        "target memory evidence missing",
    )
    require(0 <= loader_overhead_fraction <= 1, "invalid loader overhead fraction")
    require(
        graph_workspace_reserve_gib >= 0
        and system_safety_reserve_gib >= 0
        and minimum_kv_gib > 0,
        "invalid reserve",
    )
    expert_bytes = int(draft_summary["expert_payload_bytes"])
    nonexpert_bytes = int(draft_summary["nonexpert_payload_bytes"])
    require(expert_bytes % tp_size == 0, "expert payload is not TP divisible")
    rank_parameter_bytes = expert_bytes // tp_size + nonexpert_bytes
    overhead_bytes = math.ceil(rank_parameter_bytes * loader_overhead_fraction)
    usable_bytes = math.floor(usable_memory_gib_per_rank * GIB)
    target_model_bytes = math.ceil(target_only_model_gib * GIB)
    graph_workspace_bytes = math.ceil(graph_workspace_reserve_gib * GIB)
    safety_bytes = math.ceil(system_safety_reserve_gib * GIB)
    minimum_kv_bytes = math.ceil(minimum_kv_gib * GIB)
    incremental_bytes = rank_parameter_bytes + overhead_bytes
    projected_kv_bytes = (
        usable_bytes
        - target_model_bytes
        - incremental_bytes
        - graph_workspace_bytes
        - safety_bytes
    )
    passed = projected_kv_bytes >= minimum_kv_bytes
    require(passed, "projected KV reserve is below the fail-closed floor")
    observed_kv_bytes = math.floor(observed_target_only_kv_gib * GIB)
    return {
        "passed": True,
        "method": "usable per-rank envelope minus target, TP2 draft, workspace, and safety reserves",
        "usable_memory_gib_per_rank": usable_memory_gib_per_rank,
        "target_only_model_gib_per_rank": target_only_model_gib,
        "observed_target_only_kv_allocation_gib": observed_target_only_kv_gib,
        "observed_target_only_kv_is_hardware_ceiling": False,
        "expert_payload_partition": "exactly half per rank under pinned TP2 fused-MoE loader",
        "nonexpert_payload_partition": "charged fully to each rank",
        "rank_parameter_bytes": rank_parameter_bytes,
        "rank_parameter_gib": rank_parameter_bytes / GIB,
        "loader_overhead_fraction": loader_overhead_fraction,
        "loader_overhead_bytes": overhead_bytes,
        "graph_workspace_reserve_gib": graph_workspace_reserve_gib,
        "system_safety_reserve_gib": system_safety_reserve_gib,
        "incremental_draft_budget_bytes": incremental_bytes,
        "projected_remaining_kv_bytes": projected_kv_bytes,
        "projected_remaining_kv_gib": projected_kv_bytes / GIB,
        "minimum_kv_reserve_gib": minimum_kv_gib,
        "recommended_kv_cache_memory_bytes": minimum_kv_bytes,
        "recommended_kv_cache_memory_gib": minimum_kv_gib,
        "recommended_allocation_mode": "explicit --kv-cache-memory-bytes",
        "configuration_retune_required": observed_kv_bytes < minimum_kv_bytes,
        "configured_kv_gap_to_minimum_gib": max(
            0.0, (minimum_kv_bytes - observed_kv_bytes) / GIB
        ),
        "phase_c_requires_observed_kv_at_or_above_floor": True,
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    target = args.target_dir.resolve()
    draft = args.draft_dir.resolve()
    require(target != draft, "target and draft checkpoints must be distinct")
    require(args.num_speculative_tokens > 0, "speculative token count must be positive")
    target_config, _target_index, target_summary = _validate_prepared_target(
        target, args.expected_target_manifest_sha256
    )
    draft_config, _draft_index, draft_summary = _validate_draft(
        draft,
        args.expected_draft_config_sha256,
        args.expected_draft_index_sha256,
    )
    compatibility = _validate_cross_config(target_config, draft_config)
    memory = _memory_projection(
        draft_summary,
        tp_size=args.tp_size,
        usable_memory_gib_per_rank=args.usable_memory_gib_per_rank,
        target_only_model_gib=args.target_only_model_gib,
        observed_target_only_kv_gib=args.observed_target_only_kv_gib,
        loader_overhead_fraction=args.loader_overhead_fraction,
        graph_workspace_reserve_gib=args.graph_workspace_reserve_gib,
        system_safety_reserve_gib=args.system_safety_reserve_gib,
        minimum_kv_gib=args.minimum_kv_gib,
    )
    return {
        "schema_version": 1,
        "ok": True,
        "contract": "prepared_nvfp4_target_plus_separate_native_dspark_draft",
        "target": {"path": str(target), **target_summary},
        "draft": {"path": str(draft), **draft_summary},
        "compatibility": compatibility,
        "memory": memory,
        "runtime": {
            "target_model_path": "/models/dsv4-abliterated",
            "draft_model_path": "/models/dspark-draft",
            "speculative_method": "dspark",
            "num_speculative_tokens": args.num_speculative_tokens,
            "native_mtp_stage_count": 3,
            "draft_model_is_explicit": True,
            "prepared_target_skips_mtp": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--draft-dir", type=Path, required=True)
    parser.add_argument("--expected-target-manifest-sha256", required=True)
    parser.add_argument("--expected-draft-config-sha256", required=True)
    parser.add_argument("--expected-draft-index-sha256", required=True)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--num-speculative-tokens", type=int, default=5)
    parser.add_argument("--usable-memory-gib-per-rank", type=float, required=True)
    parser.add_argument("--target-only-model-gib", type=float, required=True)
    parser.add_argument("--observed-target-only-kv-gib", type=float, required=True)
    parser.add_argument("--loader-overhead-fraction", type=float, default=0.15)
    parser.add_argument("--graph-workspace-reserve-gib", type=float, default=4.0)
    parser.add_argument("--system-safety-reserve-gib", type=float, default=2.0)
    parser.add_argument("--minimum-kv-gib", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = verify(args)
    except ContractError as exc:
        raise SystemExit(f"FAILED: {exc}") from exc
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
