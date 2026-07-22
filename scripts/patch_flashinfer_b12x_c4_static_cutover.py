#!/usr/bin/env python3
"""Select FlashInfer's compact-static kernel at the real C4 decode shape.

This follows the already-patched FlashInfer dispatcher in the validated W4A4
image.  C1--C3 retain compact micro dispatch; C4 (24 routed rows) moves to
the existing compact-static implementation, measured faster on the prepared
layer without changing routing, quantization, MMA, or reduction semantics.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


PINNED_PATCHED_DISPATCH_SHA256 = (
    "253cc2f26d465adc37e48c4eee53bdb534bf6fb371a3823a5923cc8d45e2d0d3"
)
DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/"
    "cute_dsl/blackwell_sm12x/moe_dispatch.py"
)
ANCHOR = "_MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK = 40\n"
REPLACEMENT = "_MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK = 18\n"


def patch_source(source: str) -> str:
    if source.count(ANCHOR) != 1:
        raise RuntimeError("expected exactly one multi-top-k micro cutover anchor")
    return source.replace(ANCHOR, REPLACEMENT, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()
    original = args.target.read_bytes()
    actual = hashlib.sha256(original).hexdigest()
    if actual != PINNED_PATCHED_DISPATCH_SHA256:
        raise RuntimeError(
            "pinned patched dispatcher SHA-256 mismatch: "
            f"expected {PINNED_PATCHED_DISPATCH_SHA256}, got {actual}"
        )
    patched = patch_source(original.decode("utf-8")).encode("utf-8")
    args.target.write_bytes(patched)
    print("patched FlashInfer C4 compact-static cutover: " + hashlib.sha256(patched).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
