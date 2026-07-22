#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Apply the pinned vLLM modular-MoE output-alias extension.

The overlay changes only one backend (FlashInfer B12X), but its caller lives
in vLLM's large ``modular_kernel.py``.  Keep that upstream file out of the
overlay by applying two exact, hash-pinned source transformations at image
build time.  Any upstream drift fails closed instead of silently patching a
different implementation.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


PINNED_SOURCE_SHA256 = (
    "0722a23b0526c141206d17aa6472a610aca77d24f80fa72194c44d320a133687"
)
PINNED_RESULT_SHA256 = (
    "ce4b85fe464a31a0171daedcb0a498310290f1b40969728ebbeb7f6db13290fd"
)
DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/"
    "model_executor/layers/fused_moe/modular_kernel.py"
)

_PROPERTY_ANCHOR = '''\
        return False

    @staticmethod
    @abstractmethod
    def activation_format() -> FusedMoEActivationFormat:
'''

_PROPERTY_REPLACEMENT = '''\
        return False

    @property
    def supports_output_alias(self) -> bool:
        """Whether this expert can write directly into the final output."""
        return False

    @staticmethod
    @abstractmethod
    def activation_format() -> FusedMoEActivationFormat:
'''

_ALIAS_ANCHOR = '''\
        # If caller's output buffer already matches fused_out shape/dtype, alias
        # to skip the redundant copy in TopKWeightAndReduceNoOP.apply downstream.
        # This eliminates ~94% of __amd_rocclr_copyBuffer events (Copy 2 of the
        # double-copy MoE write-back path).
        if current_platform.is_rocm():
'''

_ALIAS_REPLACEMENT = '''\
        # A fused expert that already applies router weights and reduction may
        # opt into writing directly to the modular kernel's final output.  The
        # default is deliberately false; each backend must prove this contract.
        if (
            self.fused_experts.supports_output_alias
            and output_alias is not None
            and output_alias.shape == fused_out.shape
            and output_alias.dtype == fused_out.dtype
            and output_alias.device == fused_out.device
            and output_alias.is_contiguous()
        ):
            fused_out = output_alias

        # If caller's output buffer already matches fused_out shape/dtype, alias
        # to skip the redundant copy in TopKWeightAndReduceNoOP.apply downstream.
        # This eliminates ~94% of __amd_rocclr_copyBuffer events (Copy 2 of the
        # double-copy MoE write-back path).
        if current_platform.is_rocm():
'''


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch_source(source: str) -> str:
    """Return the exactly patched source, rejecting missing/duplicate anchors."""
    for label, anchor in (
        ("experts property", _PROPERTY_ANCHOR),
        ("modular output alias", _ALIAS_ANCHOR),
    ):
        count = source.count(anchor)
        if count != 1:
            raise RuntimeError(f"expected one {label} anchor, found {count}")
    source = source.replace(_PROPERTY_ANCHOR, _PROPERTY_REPLACEMENT, 1)
    source = source.replace(_ALIAS_ANCHOR, _ALIAS_REPLACEMENT, 1)
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()

    original = args.target.read_bytes()
    original_sha = _sha256(original)
    if original_sha == PINNED_RESULT_SHA256:
        print(
            "vLLM modular output alias already patched: "
            f"result={original_sha}"
        )
        return 0
    if original_sha != PINNED_SOURCE_SHA256:
        raise RuntimeError(
            "pinned modular_kernel.py SHA-256 mismatch: "
            f"expected {PINNED_SOURCE_SHA256}, got {original_sha}"
        )
    patched = patch_source(original.decode("utf-8")).encode("utf-8")
    args.target.write_bytes(patched)
    print(
        "patched vLLM modular output alias: "
        f"source={original_sha} result={_sha256(patched)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
