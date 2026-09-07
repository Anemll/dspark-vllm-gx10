# SPDX-License-Identifier: MIT
"""Extract one checksum-pinned ARM module; never upgrade an entire runtime cache."""
import argparse
from email.parser import Parser
import hashlib
import json
from pathlib import Path
import re
import zipfile


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract(wheel, lock, output, source_revision, python_root):
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("A full source commit is required")
    if sha256(python_root / "fused_moe/core.py") != lock["python_core_sha256"]:
        raise ValueError("Installed FlashInfer Python source does not match the lock")
    if sha256(wheel) != lock["wheel_sha256"]:
        raise ValueError("Wheel SHA256 mismatch")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        member = lock["module_member"]
        if names.count(member) != 1 or names.count(lock["metadata_member"]) != 1:
            raise ValueError("Missing or duplicate pinned wheel members")
        metadata = Parser().parsestr(archive.read(lock["metadata_member"]).decode())
        if metadata["Version"] != lock["wheel_version"]:
            raise ValueError("Wheel metadata version mismatch")
        info = archive.getinfo(member)
        if info.file_size != lock["module_bytes"] or info.file_size > 128 * 1024**2:
            raise ValueError("Unexpected module size")
        binary = archive.read(member)
    if hashlib.sha256(binary).hexdigest() != lock["module_sha256"]:
        raise ValueError("Module SHA256 mismatch")
    # ELF64 little-endian, e_machine=EM_AARCH64 (183).
    if binary[:6] != b"\x7fELF\x02\x01" or binary[18:20] != b"\xb7\x00":
        raise ValueError("Expected a little-endian ARM64 ELF module")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "fused_moe_120.so").open("xb") as stream:
        stream.write(binary)
    manifest = {**lock, "source_revision": source_revision,
                "scope": "Only fused_moe_120.so replaced; other compiled modules retained",
                "abi_and_gpu_validation": "required_after_build"}
    with (output / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--python-root", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    print(json.dumps(extract(args.wheel, lock, args.output, args.source_revision,
                             args.python_root)), flush=True)


if __name__ == "__main__":
    main()
