#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the real-layer FlashInfer benchmark with a preserved autotune cache.

This deliberately wraps the upstream CUTLASS harness rather than the B12X
comparator.  The service cache only selects FlashInfer fused-MoE tactics; the
B12X adapter has an independent compile/runtime ABI and must not be pulled into
this control measurement.
"""

from __future__ import annotations

import os
from pathlib import Path

from flashinfer.autotuner import autotune

import benchmark_nvfp4_a4w4_sm121 as benchmark


def main() -> int:
    cache = Path(os.environ["NVFP4_FLASHINFER_AUTOTUNE_CACHE"])
    if not cache.is_file():
        raise FileNotFoundError(f"autotune cache does not exist: {cache}")
    with autotune(False, cache=str(cache)):
        return benchmark.main()


if __name__ == "__main__":
    raise SystemExit(main())
