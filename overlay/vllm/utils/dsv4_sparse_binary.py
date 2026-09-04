# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail closed when native sparse widths meet an old AOT/JIT binary."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path


@lru_cache(maxsize=None)
def verify_native_sparse_binary(num_heads: int) -> None:
    import flashinfer
    from flashinfer.jit.mla import gen_sparse_mla_sm120_module
    from flashinfer.mla._sparse_mla_sm120 import _DECODE_DSV4_DISPATCH

    provenance = Path("/opt/ifa26/native-sparse/provenance.json")
    if not provenance.is_file():
        raise RuntimeError("Native sparse widths require the IFA native-sparse image")
    record = json.loads(provenance.read_text())
    package = Path(flashinfer.__file__).resolve().parent
    for entry in record["files"]:
        path = package / entry["installed"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["new_sha256"]:
            raise RuntimeError(f"Native sparse source mismatch: {path}")
    for width in (128, 192, 256, 512, 1024):
        if (num_heads, width) not in _DECODE_DSV4_DISPATCH:
            raise RuntimeError(f"Missing native sparse dispatch {(num_heads, width)}")
    spec = gen_sparse_mla_sm120_module()
    if not spec.is_aot:
        raise RuntimeError("Native sparse module must use its packaged AOT library")
    library = spec.aot_path
    if str(library) != record["library_path"]:
        raise RuntimeError("Native sparse library path changed")
    if hashlib.sha256(library.read_bytes()).hexdigest() != record["library_sha256"]:
        raise RuntimeError("Native sparse library SHA-256 mismatch")
