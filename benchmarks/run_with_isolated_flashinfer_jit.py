#!/usr/bin/env python3
"""Run a Python program while forcing FlashInfer to use an isolated JIT module.

The production image installs a matching FlashInfer AOT cache.  Native source
experiments must not silently load that AOT shared object, otherwise the source
mounted into ``flashinfer/data/csrc`` is never compiled.  This wrapper imports
the FlashInfer environment first, redirects the AOT lookup to a deliberately
empty directory, and then executes the requested program in the same process.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--empty-aot-dir", type=Path, required=True)
    parser.add_argument("program", type=Path)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    ns = parser.parse_args()

    ns.empty_aot_dir.mkdir(parents=True, exist_ok=True)
    if any(ns.empty_aot_dir.iterdir()):
        raise RuntimeError(f"AOT override directory is not empty: {ns.empty_aot_dir}")

    from flashinfer.jit import env as jit_env

    jit_env.FLASHINFER_AOT_DIR = ns.empty_aot_dir.resolve()
    if jit_env.FLASHINFER_AOT_DIR.exists() is False:
        raise RuntimeError("failed to create the empty AOT override directory")

    sys.argv = [str(ns.program), *ns.args]
    runpy.run_path(str(ns.program), run_name="__main__")


if __name__ == "__main__":
    main()
