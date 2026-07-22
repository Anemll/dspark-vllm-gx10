#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Enable B12X TC-decode on its existing single-copy ModelOpt layout.

The pinned W4A16 tensor-core kernel already implements both ``packed`` and
``modelopt`` weight loaders all the way through FC1 and FC2.  Its direct-top-k
TC-decode policy nevertheless admits only ``packed`` weights.  Remove only
those three policy exclusions so a benchmark can measure the same fused-sum
epilogue directly on the immutable ModelOpt NVFP4 storage.

This is intentionally a benchmark patch.  It does not change TP-MoE serving
policy, checkpoint preparation, or the default-off TC-decode environment gate.
The exact pinned source and exact patched result are both content-addressed so
upstream drift fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


PINNED_SOURCE_SHA256 = (
    "28872ab5e474f13212a9e57b22a06e1ccba8fef012ba183bf69208d8f8c9e677"
)
# Filled from the exact pinned source after applying _REPLACEMENTS.  Keeping
# this separate from the input pin proves the patch itself is deterministic.
PATCHED_SOURCE_SHA256 = (
    "c4eaa91d8a6f90b8ec6f6abf87c0f2ecb8d73dd4df6b8ae15fba18c0f1b623cd"
)
DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/"
    "b12x/moe/fused/w4a16/kernel.py"
)


_REPLACEMENTS = (
    (
        """\
# TC-decode: a small-M decode specialization that runs on the PACKED W4A16
# object (the same weights/scales the prefill GEMM uses). It reuses the packed
# tensor-core MMA inner loop but folds the top-k sum into the FC2 store
""",
        """\
# TC-decode: a small-M decode specialization that runs on packed or single-copy
# ModelOpt W4A16 weights. It reuses the tensor-core MMA inner loop but folds the
# top-k sum into the FC2 store
""",
        "module TC-decode layout contract",
    ),
    (
        """\
    if direct_topk_routes and (
        int(size_m) > direct_topk_m_cap
        or weight_layout != "packed"
        or bool(zero_fc2_output)
    ):
        raise ValueError(
            "direct_topk_routes is only valid for small-M packed W4A16 without expert_map"
        )
""",
        """\
    if direct_topk_routes and (
        int(size_m) > direct_topk_m_cap or bool(zero_fc2_output)
    ):
        raise ValueError(
            "direct_topk_routes is only valid for small-M W4A16 without expert_map"
        )
""",
        "compile direct-top-k layout guard",
    ),
    (
        """\
    # TC-decode: small-M packed decode that folds the top-k sum into the FC2
    # store epilogue. Reuses the packed tensor-core MMA path; only the launch
    # scheduling/epilogue changes. Requires the packed object, bf16 gated
""",
        """\
    # TC-decode: small-M packed or ModelOpt decode that folds the top-k sum into
    # the FC2 store epilogue. Reuses the tensor-core MMA path; only the launch
    # scheduling/epilogue changes. Requires a prepared packed or ModelOpt
    # object, bf16 gated
""",
        "runtime TC-decode layout contract",
    ),
    (
        """\
    use_tc_decode = bool(
        _tc_decode_enabled()
        and (fused_launch is None or preplanned_tc_decode)
        and weight_layout == "packed"
        and expert_map is None
        and is_gated
""",
        """\
    use_tc_decode = bool(
        _tc_decode_enabled()
        and (fused_launch is None or preplanned_tc_decode)
        and weight_layout in _WEIGHT_LAYOUTS
        and expert_map is None
        and is_gated
""",
        "TC-decode ModelOpt eligibility",
    ),
    (
        """\
        (m <= _MAX_DIRECT_TOPK_ROUTE_M or use_tc_decode)
        and weight_layout == "packed"
        and expert_map is None
""",
        """\
        (m <= _MAX_DIRECT_TOPK_ROUTE_M or use_tc_decode)
        and weight_layout in _WEIGHT_LAYOUTS
        and expert_map is None
""",
        "direct-top-k ModelOpt eligibility",
    ),
    (
        """\
            "preplanned W4A16 direct top-k routing requires small-M packed "
            "int32 topk_ids without expert_map"
""",
        """\
            "preplanned W4A16 direct top-k routing requires small-M "
            "packed or ModelOpt int32 topk_ids without expert_map"
""",
        "direct-top-k preplanned error contract",
    ),
    (
        """\
            "preplanned TC-decode W4A16 launch requires small-M packed bf16 "
            f"decode (m <= {_TC_DECODE_MAX_M}, cuda int32/int64 topk_ids, "
""",
        """\
            "preplanned TC-decode W4A16 launch requires small-M "
            "packed or ModelOpt bf16 decode "
            f"(m <= {_TC_DECODE_MAX_M}, cuda int32/int64 topk_ids, "
""",
        "TC-decode preplanned error contract",
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
            "pinned B12X W4A16 kernel SHA-256 mismatch: "
            f"expected {PINNED_SOURCE_SHA256}, got {original_sha}"
        )
    patched = patch_source(original.decode("utf-8")).encode("utf-8")
    patched_sha = _sha256(patched)
    if patched_sha != PATCHED_SOURCE_SHA256:
        raise RuntimeError(
            "deterministic B12X W4A16 ModelOpt TC-decode result mismatch: "
            f"expected {PATCHED_SOURCE_SHA256}, got {patched_sha}"
        )
    args.target.write_bytes(patched)
    print(
        "patched B12X W4A16 ModelOpt TC-decode benchmark path: "
        f"source={original_sha} result={patched_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
