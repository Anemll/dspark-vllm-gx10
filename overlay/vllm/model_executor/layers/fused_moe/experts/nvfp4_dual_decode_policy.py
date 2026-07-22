# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure policy helpers for NVFP4 CUTLASS/W4A16 dual decode."""

from __future__ import annotations


def validate_dual_decode_bounds(minimum: int, maximum: int) -> tuple[int, int]:
    minimum = int(minimum)
    maximum = int(maximum)
    if minimum < 2:
        raise ValueError(
            "dual-decode minimum M must be >= 2 so M=1 remains CUTLASS W4A4, "
            f"got {minimum}"
        )
    if maximum < minimum:
        raise ValueError(
            "dual-decode maximum M must be >= minimum M, got "
            f"{maximum} < {minimum}"
        )
    return minimum, maximum


def use_w4a16_decode(
    num_tokens: int,
    bounds: tuple[int, int],
    *,
    uniform_decode: bool,
) -> bool:
    """Select W4A16 only inside the explicit small-M decode interval."""

    minimum, maximum = validate_dual_decode_bounds(*bounds)
    return bool(uniform_decode) and minimum <= int(num_tokens) <= maximum
