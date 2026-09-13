#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Add NVIDIA-target/MXFP4-draft dispatch to the pinned B12X Vision runtime."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sysconfig


DEFAULT_TARGET = (
    Path(sysconfig.get_paths()["purelib"])
    / "vllm/models/deepseek_v4/quant_config.py"
)

INSERT_AFTER = """    @property
    def moe_quant_algo(self) -> str:
        self._resolve_moe_overrides()
        return self._resolved_moe_quant_algo or ""
"""

PREFIX_RESOLVER = """

    def _moe_quant_algo_for_prefix(self, prefix: str) -> str:
        algo = self.moe_quant_algo
        if algo != "NVFP4":
            return algo
        from .mixed_expert_quant import expert_quant_algo

        hf_config = get_current_vllm_config().model_config.hf_config
        return expert_quant_algo(hf_config, prefix, algo)
"""

OLD_SELECTION = '                if self.moe_quant_algo == "NVFP4":\n'
NEW_SELECTION = '                if self._moe_quant_algo_for_prefix(prefix) == "NVFP4":\n'

MXFP4_QUERY = """

    def is_mxfp4_quant(self, prefix, layer):
        if not isinstance(layer, RoutedExperts) or self.expert_dtype != "fp4":
            return False
        return self._moe_quant_algo_for_prefix(prefix) != "NVFP4"
"""


def patch(source: bytes, expected_sha256: str) -> bytes:
    actual = hashlib.sha256(source).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"refusing to patch unknown DeepSeek V4 quant config: {actual} "
            f"(expected {expected_sha256})"
        )
    text = source.decode()
    if text.count(INSERT_AFTER) != 1:
        raise ValueError("expected one MoE algorithm property insertion point")
    if text.count(OLD_SELECTION) != 1:
        raise ValueError("expected one global NVIDIA NVFP4 selection")
    if "def _moe_quant_algo_for_prefix" in text or "def is_mxfp4_quant" in text:
        raise ValueError("mixed expert dispatch already exists")
    text = text.replace(INSERT_AFTER, INSERT_AFTER + PREFIX_RESOLVER, 1)
    text = text.replace(OLD_SELECTION, NEW_SELECTION, 1)
    text += MXFP4_QUERY
    compile(text, "quant_config.py", "exec")
    return text.encode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    try:
        result = patch(args.target.read_bytes(), args.expected_sha256)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    args.target.write_bytes(result)
    print(hashlib.sha256(result).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
