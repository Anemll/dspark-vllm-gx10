#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Package the once-compiled, component-tested SM121 library into an image."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--patch-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.library.read_bytes()).hexdigest() != args.sha256:
        raise ValueError("component-tested library SHA-256 mismatch")
    from flashinfer.jit import env

    target = env.FLASHINFER_AOT_DIR / "sparse_mla_sm120" / "sparse_mla_sm120.so"
    target.parent.mkdir(parents=True, exist_ok=True)
    # Image-layer operation only: never write a host's shared serving cache.
    shutil.copyfile(args.library, target)
    record = json.loads(args.patch_report.read_text())
    record.update(library_path=str(target), library_sha256=args.sha256,
                  cuda_arch="12.1a", provenance="once-compiled component-tested binary")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
