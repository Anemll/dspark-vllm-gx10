#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Enable the B12X NVFP4 direct microkernel to consume packed E8M0/K32 scales.

The arithmetic remains W4A4.  Only the weight-scale representation changes:
paired power-of-two E4M3/K16 scales are collapsed to one E8M0/K32 byte.  The
direct kernel already implements the E8M0 load/decode path; upstream only
artificially restricts it to W4A16 and does not expose the scale-format choice
from the NVFP4 microkernel factory.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


MICRO_SHA256 = "67847d6365b3707b54e5d68a89655666350029aa550c5f74742084f264d2d980"
TP_MOE_SHA256 = "49cd151aa80f4fdfa603eafe21b792b51a6483fa6f39452892ed2240fd79da34"
SILU_SHA256 = "397bc15e2691ab33be5ebe0c052e5a4a79da586148f9645f4028afb8eae0011a"

MICRO_DEFAULT = Path(
    "/usr/local/lib/python3.12/dist-packages/b12x/moe/fused/micro.py"
)
TP_MOE_DEFAULT = Path(
    "/usr/local/lib/python3.12/dist-packages/b12x/integration/tp_moe.py"
)
SILU_DEFAULT = Path(
    "/usr/local/lib/python3.12/dist-packages/b12x/moe/fused/silu.py"
)

_MICRO_ANCHOR = """\
        if scale_format == "e8m0_k32" and not w4a16_mode:
            raise ValueError("e8m0_k32 scales are only supported in W4A16 micro mode")
"""
_MICRO_REPLACEMENT = """\
        # E8M0/K32 is also valid for W4A4 when the caller supplies the exact
        # collapsed weight-scale sidecar.  The existing kernel branches decode
        # each K32 byte and reuse it for the corresponding two K16 dot groups.
"""
_ACTIVATION_SCALE_ANCHOR = (
    "self._scale_byte_to_f32(cvt_f32_to_e4m3(q_scale))"
)
_ACTIVATION_SCALE_REPLACEMENT = (
    "cvt_e4m3_to_f32_via_f16(cvt_f32_to_e4m3(q_scale))"
)

_FACTORY_ANCHOR = """\
    kernel = activation_spec.make_micro_kernel(
        sf_vec_size=16,
"""
_FACTORY_REPLACEMENT = """\
    micro_scale_format = os.environ.get(
        "B12X_NVFP4_MICRO_SCALE_FORMAT", "e4m3_k16"
    )
    if quant_mode != "nvfp4" and micro_scale_format != "e4m3_k16":
        raise ValueError(
            "B12X_NVFP4_MICRO_SCALE_FORMAT applies only to quant_mode='nvfp4'"
        )
    kernel = activation_spec.make_micro_kernel(
        sf_vec_size=16,
"""

_KWARGS_ANCHOR = """\
        dynamic_down_scale=dynamic_down_scale,
    )
"""
_KWARGS_REPLACEMENT = """\
        dynamic_down_scale=dynamic_down_scale,
        scale_format=micro_scale_format,
    )
"""

_SILU_SIGNATURE_ANCHOR = """\
        dynamic_down_scale: bool = False,
    ):
"""
_SILU_SIGNATURE_REPLACEMENT = """\
        dynamic_down_scale: bool = False,
        scale_format: str = "e4m3_k16",
    ):
"""
_SILU_FORWARD_ANCHOR = """\
            dynamic_down_scale=dynamic_down_scale,
        )
"""
_SILU_FORWARD_REPLACEMENT = """\
            dynamic_down_scale=dynamic_down_scale,
            scale_format=scale_format,
        )
"""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch_micro(source: str) -> str:
    if source.count(_MICRO_ANCHOR) != 1:
        raise RuntimeError("B12X micro E8M0 restriction anchor drifted")
    if source.count(_ACTIVATION_SCALE_ANCHOR) != 4:
        raise RuntimeError(
            "B12X W4A4 activation-scale decode anchor drifted: "
            f"expected 4, got {source.count(_ACTIVATION_SCALE_ANCHOR)}"
        )
    source = source.replace(_MICRO_ANCHOR, _MICRO_REPLACEMENT, 1)
    return source.replace(
        _ACTIVATION_SCALE_ANCHOR, _ACTIVATION_SCALE_REPLACEMENT
    )


def patch_tp_moe(source: str) -> str:
    if source.count(_FACTORY_ANCHOR) != 1:
        raise RuntimeError("B12X micro factory anchor drifted")
    source = source.replace(_FACTORY_ANCHOR, _FACTORY_REPLACEMENT, 1)
    # This anchor occurs in static and dynamic factories too.  Patch only the
    # occurrence after the newly inserted micro_scale_format declaration.
    start = source.index("    micro_scale_format = os.environ.get(")
    tail = source[start:]
    if tail.count(_KWARGS_ANCHOR) < 1:
        raise RuntimeError("B12X micro factory kwargs anchor drifted")
    tail = tail.replace(_KWARGS_ANCHOR, _KWARGS_REPLACEMENT, 1)
    return source[:start] + tail


def patch_silu(source: str) -> str:
    if source.count(_SILU_SIGNATURE_ANCHOR) < 1:
        raise RuntimeError("B12X SiLU micro signature anchor drifted")
    if source.count(_SILU_FORWARD_ANCHOR) < 1:
        raise RuntimeError("B12X SiLU micro forwarding anchor drifted")
    source = source.replace(
        _SILU_SIGNATURE_ANCHOR, _SILU_SIGNATURE_REPLACEMENT, 1
    )
    return source.replace(
        _SILU_FORWARD_ANCHOR, _SILU_FORWARD_REPLACEMENT, 1
    )


def _patch_file(path: Path, expected_sha: str, patcher) -> str:
    original = path.read_bytes()
    observed = _sha256(original)
    if observed != expected_sha:
        raise RuntimeError(
            f"{path} SHA-256 mismatch: expected {expected_sha}, got {observed}"
        )
    patched = patcher(original.decode()).encode()
    path.write_bytes(patched)
    return _sha256(patched)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro-target", type=Path, default=MICRO_DEFAULT)
    parser.add_argument("--tp-moe-target", type=Path, default=TP_MOE_DEFAULT)
    parser.add_argument("--silu-target", type=Path, default=SILU_DEFAULT)
    args = parser.parse_args()
    micro_result = _patch_file(args.micro_target, MICRO_SHA256, patch_micro)
    tp_result = _patch_file(args.tp_moe_target, TP_MOE_SHA256, patch_tp_moe)
    silu_result = _patch_file(args.silu_target, SILU_SHA256, patch_silu)
    print(
        "enabled NVFP4 E8M0/K32 direct micro scale path: "
        f"micro={micro_result} tp_moe={tp_result} silu={silu_result}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
