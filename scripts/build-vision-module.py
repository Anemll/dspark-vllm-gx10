# SPDX-License-Identifier: MIT
"""Prebuild ONLY the image-prefill module; existing text binaries stay untouched."""
import hashlib
import json
import os
from pathlib import Path
import shutil

from vllm.models.deepseek_v4.nvidia import vision_sparse


def main():
    archs = os.environ.get("FLASHINFER_CUDA_ARCH_LIST", "").split()
    if not archs or any(arch not in ("12.0a", "12.1a") for arch in archs):
        raise SystemExit("Set FLASHINFER_CUDA_ARCH_LIST explicitly to 12.0a and/or 12.1a")
    spec = vision_sparse.get_vision_sparse_spec()
    spec.build(verbose=True)
    output = Path(vision_sparse.__file__).with_name("vision_kernels")
    binary = output / "dspark_dsv4_vision512.so"
    shutil.copyfile(spec.jit_library_path, binary)
    manifest = {
        "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "cuda_archs": archs,
        "source_revision": os.environ.get("SOURCE_REVISION", "unknown"),
        "sources": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in spec.sources},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
