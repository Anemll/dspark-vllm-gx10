#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Enable the pinned SM121 B12X single-token shared-input fast path.

Prepared DeepSeek V4 B12X layers bake their expert-global scales into the
block scales, leaving the two activation global-scale vectors uniformly 1.0.
For M=1 the same hidden-state row is routed to all six experts, so quantizing
that row once is sufficient.  FlashInfer's microkernel already implements the
gated shared-input path, but the pinned dispatcher only enables it for ReLU2
and the wrapper does not expose the now-shared activation scales separately
from its per-expert weight-alpha views.

This image-time patch is deliberately narrow and source-pinned:

* an adapter-owned opt-in attribute slices only the activation-scale call
  arguments to one element after full per-expert weight views are cached; and
* the dispatcher permits its existing shared-input implementation for any
  single-token activation when those call arguments are scalar.

Multi-token decode and prefill are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


PINNED_WRAPPER_SHA256 = (
    "bcac806795c035decd0773f4f801d477e7ebf14c1d67c3e49eee42ee0579c0a4"
)
PINNED_DISPATCH_SHA256 = (
    "cba2d0966631a47a576747e8322b57116122f2c8e5e868f8efb3f5ea692391a4"
)
DEFAULT_WRAPPER_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/"
    "cute_dsl/b12x_moe.py"
)
DEFAULT_DISPATCH_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/"
    "cute_dsl/blackwell_sm12x/moe_dispatch.py"
)

_WRAPPER_RETURN_ANCHOR = """\
        return launch_sm120_moe(
            a=x,
"""
_WRAPPER_RETURN_REPLACEMENT = """\
        shared_unit_scales = getattr(self, "_dspark_unit_input_scales", False)
        input_gs = w1_alpha[:1] if shared_unit_scales else w1_alpha
        down_input_scale = (
            fc2_input_scale[:1]
            if shared_unit_scales and fc2_input_scale is not None
            else fc2_input_scale
        )

        return launch_sm120_moe(
            a=x,
"""
_WRAPPER_ARGS_ANCHOR = """\
            w1_alpha=w1_alpha,
            fc2_input_scale=fc2_input_scale,
"""
_WRAPPER_ARGS_REPLACEMENT = """\
            w1_alpha=input_gs,
            fc2_input_scale=down_input_scale,
"""

_DISPATCH_SHARE_INPUT_ANCHOR = """\
    share_input_across_experts = (
        activation == "relu2"
        and num_tokens == 1
        and input_gs_is_shared
        and _MICRO_SHARE_INPUT_ACROSS_EXPERTS
    )
"""
_DISPATCH_SHARE_INPUT_REPLACEMENT = """\
    share_input_across_experts = (
        num_tokens == 1
        and input_gs_is_shared
        and _MICRO_SHARE_INPUT_ACROSS_EXPERTS
    )
"""
_DISPATCH_SHARE_SCALE_ANCHOR = """\
    share_expert_scales = (
        activation == "relu2" and input_gs_is_shared and down_input_scale_is_shared
    )
"""
_DISPATCH_SHARE_SCALE_REPLACEMENT = """\
    share_expert_scales = input_gs_is_shared and down_input_scale_is_shared
"""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _replace_once(source: str, anchor: str, replacement: str, label: str) -> str:
    count = source.count(anchor)
    if count != 1:
        raise RuntimeError(f"expected one {label} anchor, found {count}")
    return source.replace(anchor, replacement, 1)


def patch_wrapper_source(source: str) -> str:
    source = _replace_once(
        source,
        _WRAPPER_RETURN_ANCHOR,
        _WRAPPER_RETURN_REPLACEMENT,
        "wrapper launch",
    )
    return _replace_once(
        source,
        _WRAPPER_ARGS_ANCHOR,
        _WRAPPER_ARGS_REPLACEMENT,
        "wrapper activation scales",
    )


def patch_dispatch_source(source: str) -> str:
    source = _replace_once(
        source,
        _DISPATCH_SHARE_INPUT_ANCHOR,
        _DISPATCH_SHARE_INPUT_REPLACEMENT,
        "dispatcher shared input",
    )
    return _replace_once(
        source,
        _DISPATCH_SHARE_SCALE_ANCHOR,
        _DISPATCH_SHARE_SCALE_REPLACEMENT,
        "dispatcher shared scales",
    )


def _patch_file(path: Path, expected_sha: str, transform, label: str) -> str:
    original = path.read_bytes()
    actual_sha = _sha256(original)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"pinned {label} SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
        )
    patched = transform(original.decode("utf-8")).encode("utf-8")
    path.write_bytes(patched)
    return _sha256(patched)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper-target", type=Path, default=DEFAULT_WRAPPER_TARGET)
    parser.add_argument("--dispatch-target", type=Path, default=DEFAULT_DISPATCH_TARGET)
    args = parser.parse_args()

    wrapper_result = _patch_file(
        args.wrapper_target,
        PINNED_WRAPPER_SHA256,
        patch_wrapper_source,
        "FlashInfer B12X wrapper",
    )
    dispatch_result = _patch_file(
        args.dispatch_target,
        PINNED_DISPATCH_SHA256,
        patch_dispatch_source,
        "FlashInfer B12X dispatcher",
    )
    print(
        "patched FlashInfer B12X unit-scale shared input: "
        f"wrapper_result={wrapper_result} dispatch_result={dispatch_result}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
