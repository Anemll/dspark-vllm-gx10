# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve an explicit mixed-format expert manifest without touching tensors."""

import re


def expert_quant_algo(hf_config, prefix: str, default: str) -> str:
    quant = getattr(hf_config, "quantization_config", None) or {}
    if default != "NVFP4" or "quantized_layers" not in quant:
        return default
    manifest = quant["quantized_layers"]
    if not isinstance(manifest, dict):
        raise ValueError("DeepSeek V4 quantized_layers must be a mapping")
    # Target and DSpark constructors add model/wrapper prefixes. The draft
    # constructor numbers mtp.0-2 after the target's final decoder layer.
    match = re.search(
        r"(?:^|\.)(layers\.(\d+)\.ffn\.experts)(?:\.routed_experts)?$", prefix
    )
    if match is None:
        raise ValueError(f"Unrecognized DeepSeek V4 mixed expert prefix: {prefix}")
    name, layer_id = match.group(1), int(match.group(2))
    if name in manifest:
        entry = manifest[name]
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid expert quantization entry: {name}")
        algo = entry.get("quant_algo", "").upper()
        expected_group = {"NVFP4": 16, "MXFP4": 32}.get(algo)
        if expected_group is None or entry.get("group_size") != expected_group:
            raise ValueError(f"Unsupported expert quantization entry: {name}: {entry}")
        return algo
    target_layers = hf_config.num_hidden_layers
    draft_layers = getattr(hf_config, "n_mtp_layers", None) or 3
    # NVIDIA explicitly excludes mtp.* and keeps those MXFP4 tensors intact.
    # Missing target entries are errors, not a silent MXFP4 fallback. Do not
    # infer that all globally NVFP4 checkpoints must have MXFP4 draft weights.
    if (
        target_layers <= layer_id < target_layers + draft_layers
        and "mtp.*" in quant.get("ignore", [])
        and not any(key.startswith("mtp.") for key in manifest)
    ):
        return "MXFP4"
    raise ValueError(f"Missing explicit expert quantization entry for {prefix}")
