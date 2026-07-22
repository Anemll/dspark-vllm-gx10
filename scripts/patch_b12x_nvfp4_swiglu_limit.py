#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Enable the pinned B12X NVFP4 direct microkernel's SwiGLU clamp.

The pinned microkernel already implements ``swiglu_limit`` and includes it in
its compile cache key.  The public TP-MoE integration currently rejects the
option for NVFP4 and does not forward it to that microkernel.  Apply the small
integration-only change at image build time, failing closed on source drift.

Only the compact NVFP4 direct-micro path is enabled.  Static fallback and
dynamic kernels continue to reject the clamp rather than silently changing
DeepSeek V4 semantics.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


PINNED_SOURCE_SHA256 = (
    "49cd151aa80f4fdfa603eafe21b792b51a6483fa6f39452892ed2240fd79da34"
)
DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/b12x/integration/tp_moe.py"
)

_REPLACEMENTS = (
    (
        """\
    activation: str = "silu",
    device: torch.device | None = None,
    quant_mode: str = "nvfp4",
):
""",
        """\
    activation: str = "silu",
    swiglu_limit: float | None = None,
    device: torch.device | None = None,
    quant_mode: str = "nvfp4",
):
""",
        "micro signature",
    ),
    (
        """\
        single_token=single_token,
        dynamic_down_scale=dynamic_down_scale,
    )
""",
        """\
        single_token=single_token,
        dynamic_down_scale=dynamic_down_scale,
        swiglu_limit=swiglu_limit,
    )
""",
        "micro construction",
    ),
    (
        """\
    quant_mode: str = "nvfp4",
    unit_scale_contract: bool = False,
) -> None:
""",
        """\
    quant_mode: str = "nvfp4",
    unit_scale_contract: bool = False,
    swiglu_limit: float | None = None,
) -> None:
""",
        "compact-static signature",
    ),
    (
        """\
    if use_micro_direct:
        if flat_ids.dtype == torch.int32 and flat_ids.is_contiguous():
""",
        """\
    if swiglu_limit is not None and not use_micro_direct:
        raise NotImplementedError(
            "NVFP4 swiglu_limit requires the compact direct microkernel"
        )
    if use_micro_direct:
        if flat_ids.dtype == torch.int32 and flat_ids.is_contiguous():
""",
        "direct-only guard",
    ),
    (
        """\
            single_token=(m == 1),
            activation=activation,
            device=a.device,
""",
        """\
            single_token=(m == 1),
            activation=activation,
            swiglu_limit=swiglu_limit,
            device=a.device,
""",
        "micro forwarding",
    ),
    (
        """\
    if swiglu_limit is not None and quant_mode != "w4a16":
        raise NotImplementedError("swiglu_limit is implemented only for W4A16 MoE")
""",
        """\
    if (
        swiglu_limit is not None
        and quant_mode == "nvfp4"
        and activation != "silu"
    ):
        raise ValueError("NVFP4 swiglu_limit requires activation='silu'")
""",
        "public NVFP4 clamp gate",
    ),
    (
        """\
            quant_mode=quant_mode,
            unit_scale_contract=unit_scale_contract,
        )
""",
        """\
            quant_mode=quant_mode,
            unit_scale_contract=unit_scale_contract,
            swiglu_limit=swiglu_limit,
        )
""",
        "compact-static forwarding",
    ),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch_source(source: str) -> str:
    for anchor, replacement, label in _REPLACEMENTS:
        count = source.count(anchor)
        if count != 1:
            raise RuntimeError(f"expected one {label} anchor, found {count}")
        source = source.replace(anchor, replacement, 1)
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()

    original = args.target.read_bytes()
    original_sha = _sha256(original)
    if original_sha != PINNED_SOURCE_SHA256:
        raise RuntimeError(
            "pinned b12x tp_moe.py SHA-256 mismatch: "
            f"expected {PINNED_SOURCE_SHA256}, got {original_sha}"
        )
    patched = patch_source(original.decode("utf-8")).encode("utf-8")
    args.target.write_bytes(patched)
    print(
        "patched B12X NVFP4 swiglu_limit forwarding: "
        f"source={original_sha} result={_sha256(patched)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
