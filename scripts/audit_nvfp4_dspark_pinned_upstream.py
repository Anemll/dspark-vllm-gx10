#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Pin the upstream split-draft call sites and acceptance metric source."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


PINNED_REVISION = "752a3a504485790a2e8491cacbb35c137339ad34"
EXPECTED_FILES = {
    "vllm/config/speculative.py": "3f1abd1ca3042fba239e7bf98b08f645f3e950c16ab510fbc99a49c5c507721f",
    "vllm/v1/worker/gpu/spec_decode/dspark/utils.py": "457c44aec45fe00780748c34288a091bd212d18b40b0d901a8f628640c0a2d24",
    "vllm/models/deepseek_v4/nvidia/dspark.py": "efe33c32d37ed7f26d869d94626f1415906d31218ec0ee44d79bb2b815b8cf39",
    "vllm/models/deepseek_v4/nvidia/model.py": "a0cbb88b7a0ac5ba9419e07f8922bb84c861f41611596d719efdd86ce95a2e50",
    "vllm/v1/spec_decode/metrics.py": "c1c6b20bbf0ae3427dc331bbf8032193cf0b1ab9f3fe552257da6e4d51f2f4f6",
}
ACCEPTANCE_METRICS = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
    "accepted_per_position": "vllm:spec_decode_num_accepted_tokens_per_pos_total",
}


class AuditError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_class(tree: ast.AST, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AuditError(f"class not found: {name}")


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AuditError(f"function not found: {name}")


def _is_attr(node: ast.AST, *parts: str) -> bool:
    observed: list[str] = []
    while isinstance(node, ast.Attribute):
        observed.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        observed.append(node.id)
    return tuple(reversed(observed)) == parts


def audit_speculative_config(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    cls = _find_class(tree, "SpeculativeConfig")
    model_field = any(
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "model"
        for node in cls.body
    )
    post = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
    )
    separate_model_config = False
    default_same_path = False
    for node in ast.walk(post):
        if isinstance(node, ast.Call) and (
            isinstance(node.func, ast.Name) and node.func.id == "ModelConfig"
        ):
            separate_model_config |= any(
                keyword.arg == "model" and _is_attr(keyword.value, "self", "model")
                for keyword in node.keywords
            )
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            default_same_path |= any(
                _is_attr(target, "self", "model")
                and _is_attr(value, "self", "target_model_config", "model")
                for target in targets
            )
    require(model_field, "SpeculativeConfig.model field is missing")
    require(separate_model_config, "explicit model does not construct a separate ModelConfig")
    require(default_same_path, "same-checkpoint DSpark default path is missing")
    return {
        "explicit_model_field": True,
        "explicit_model_builds_separate_draft_model_config": True,
        "omitted_model_defaults_to_target_path": True,
    }


def audit_dspark_loader(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    fn = _find_function(tree, "load_dspark_model")
    passes_draft_config = False
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "get_model":
            passes_draft_config |= any(
                keyword.arg == "model_config"
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "draft_model_config"
                for keyword in node.keywords
            )
    require(passes_draft_config, "load_dspark_model does not pass draft_model_config")
    return {"get_model_receives_draft_model_config": True}


def audit_dspark_weights(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    cls = _find_class(tree, "DSparkDeepseekV4ForCausalLM")
    load = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "load_weights"
    )
    remap = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_remap_dspark_name"
    )
    uses_remap = any(
        isinstance(node, ast.Call) and _is_attr(node.func, "self", "_remap_dspark_name")
        for node in ast.walk(load)
    )
    mtp_match = any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("mtp\\.")
        for node in ast.walk(remap)
    )
    returns_none = any(
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
        for node in ast.walk(remap)
    )
    require(uses_remap and mtp_match and returns_none, "draft MTP-only filtering drifted")
    return {
        "load_weights_uses_mtp_remapper": True,
        "non_mtp_weights_are_skipped": True,
    }


def audit_metrics(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    strings = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    base_names = {name.removesuffix("_total") for name in ACCEPTANCE_METRICS.values()}
    require(base_names <= strings, "speculative acceptance Prometheus counters drifted")
    require("position" in strings, "per-position acceptance label is missing")
    return {
        "prometheus_counters": ACCEPTANCE_METRICS,
        "aggregate_acceptance_formula": "delta(accepted_tokens) / delta(draft_tokens)",
        "mean_acceptance_length_formula": "1 + delta(accepted_tokens) / delta(drafts)",
        "per_position_formula": "delta(accepted_per_position) / delta(drafts)",
    }


def audit_tp_expert_partition(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    cls = _find_class(tree, "DeepseekV4MoE")
    init = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_init_fused_moe_experts"
    )
    local_division = False
    rank_offset = False
    for node in ast.walk(init):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if _is_attr(target, "self", "n_local_physical_experts"):
                    value = node.value
                    local_division |= (
                        isinstance(value, ast.BinOp)
                        and isinstance(value.op, ast.FloorDiv)
                        and _is_attr(value.left, "self", "n_physical_experts")
                        and _is_attr(value.right, "self", "tp_size")
                    )
                if _is_attr(target, "self", "experts_start_idx"):
                    value = node.value
                    rank_offset |= (
                        isinstance(value, ast.BinOp)
                        and isinstance(value.op, ast.Mult)
                        and _is_attr(value.left, "self", "tp_rank")
                        and _is_attr(value.right, "self", "n_local_experts")
                    )
    require(local_division, "fused-MoE local expert division drifted")
    require(rank_offset, "fused-MoE TP rank expert offset drifted")
    return {
        "path": "DeepseekV4MoE._init_fused_moe_experts",
        "non_eplb_physical_experts_divided_by_tp_size": True,
        "rank_owns_contiguous_expert_range": True,
        "projection_assumption": "no redundant experts and TP=2 imply exactly half of routed-expert payload per rank",
    }


def audit(checkout: Path, revision: str) -> dict[str, Any]:
    require(revision == PINNED_REVISION, "vLLM revision drifted")
    sources: dict[str, str] = {}
    identities: dict[str, str] = {}
    for relative, expected in EXPECTED_FILES.items():
        path = checkout / relative
        require(path.is_file(), f"pinned upstream file missing: {relative}")
        observed = sha256(path)
        require(observed == expected, f"pinned upstream SHA drifted: {relative}")
        sources[relative] = path.read_text(encoding="utf-8")
        identities[relative] = observed
    return {
        "schema_version": 1,
        "ok": True,
        "vllm_revision": revision,
        "file_sha256": identities,
        "split_model_config": audit_speculative_config(sources["vllm/config/speculative.py"]),
        "draft_model_construction": audit_dspark_loader(
            sources["vllm/v1/worker/gpu/spec_decode/dspark/utils.py"]
        ),
        "draft_weight_filter": audit_dspark_weights(
            sources["vllm/models/deepseek_v4/nvidia/dspark.py"]
        ),
        "draft_tp_expert_partition": audit_tp_expert_partition(
            sources["vllm/models/deepseek_v4/nvidia/model.py"]
        ),
        "acceptance_metric_source": audit_metrics(
            sources["vllm/v1/spec_decode/metrics.py"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    try:
        revision = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        report = audit(checkout, revision)
    except (AuditError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"FAILED: {exc}") from exc
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
