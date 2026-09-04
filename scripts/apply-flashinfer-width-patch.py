#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Apply the five manifest-pinned runtime changes inside a diagnostic image.

This helper intentionally accepts only ordinary exact-context unified hunks.
It has no fuzz, offsets, binary patch, file creation, or dependency installation.
Every old/new file hash is verified before any file is written.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import stat
import tempfile


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in text.splitlines(keepends=True):
        if line.startswith("diff --git "):
            match = re.fullmatch(r"diff --git a/(\S+) b/(\S+)\n", line)
            if match is None or match[1] != match[2] or match[1] in sections:
                raise ValueError("unsupported or duplicate patch file section")
            current = sections[match[1]] = []
        elif current is not None:
            current.append(line)
        else:
            raise ValueError("patch content before first file section")
    return sections


def apply_exact(original: bytes, section: list[str]) -> bytes:
    old = original.decode("utf-8").splitlines(keepends=True)
    result: list[str] = []
    cursor = 0
    index = 0
    hunks = 0
    while index < len(section):
        line = section[index]
        if not line.startswith("@@ "):
            if not line.startswith(("index ", "--- a/", "+++ b/")):
                raise ValueError(f"unexpected patch metadata: {line!r}")
            index += 1
            continue
        match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if match is None:
            raise ValueError("malformed hunk header")
        start = int(match[1]) - 1
        old_count = int(match[2] or "1")
        new_count = int(match[4] or "1")
        if start < cursor or start > len(old):
            raise ValueError("hunk old-line offset is outside the original file")
        result.extend(old[cursor:start])
        cursor = start
        old_seen = new_seen = 0
        index += 1
        while index < len(section) and not section[index].startswith("@@ "):
            entry = section[index]
            if entry[:1] not in (" ", "+", "-"):
                raise ValueError(f"unsupported hunk line: {entry!r}")
            text = entry[1:]
            if entry[0] in (" ", "-"):
                if cursor >= len(old) or old[cursor] != text:
                    raise ValueError(f"exact patch context mismatch at old line {cursor + 1}")
                cursor += 1
                old_seen += 1
            if entry[0] in (" ", "+"):
                result.append(text)
                new_seen += 1
            index += 1
        if (old_seen, new_seen) != (old_count, new_count):
            raise ValueError("hunk line counts disagree with the header")
        hunks += 1
    if not hunks:
        raise ValueError("no text hunks found")
    result.extend(old[cursor:])
    return "".join(result).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--tested-image")
    parser.add_argument("--source-revision")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    patch = args.manifest.parent / manifest["patch_file"]
    patch_bytes = patch.read_bytes()
    if sha256(patch_bytes) != manifest["patch_sha256"]:
        raise ValueError("patch SHA-256 mismatch")
    sections = patch_sections(patch_bytes.decode("utf-8"))
    if len(manifest["files"]) != 5:
        raise ValueError("expected exactly five runtime files")
    package = args.package_dir
    if package is None:
        spec = importlib.util.find_spec("flashinfer")
        if spec is None or spec.origin is None:
            raise ValueError("installed flashinfer package was not found")
        package = Path(spec.origin).parent
    package = package.resolve(strict=True)
    pending: list[tuple[Path, bytes, int]] = []
    records = []
    seen = set()
    for entry in manifest["files"]:
        relative = Path(entry["installed"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("runtime path must be a non-traversing relative path")
        unresolved = package / relative
        if any(path.is_symlink() for path in (unresolved, *unresolved.parents)
               if path != package and path.is_relative_to(package)):
            raise ValueError("runtime target or parent is a symlink")
        target = unresolved.resolve(strict=True)
        if not target.is_relative_to(package) or target in seen:
            raise ValueError("duplicate or out-of-package runtime target")
        seen.add(target)
        original = target.read_bytes()
        if sha256(original) != entry["old_sha256"]:
            raise ValueError(f"original SHA-256 mismatch: {entry['installed']}")
        modified = apply_exact(original, sections[entry["source"]])
        if sha256(modified) != entry["new_sha256"]:
            raise ValueError(f"patched SHA-256 mismatch: {entry['installed']}")
        pending.append((target, modified, stat.S_IMODE(target.stat().st_mode)))
        records.append({**entry, "path": str(target)})
    if not args.check:
        for target, modified, mode in pending:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                temp = Path(handle.name)
                handle.write(modified)
            try:
                temp.chmod(mode)
                temp.replace(target)
            finally:
                temp.unlink(missing_ok=True)
    report = {"status": "checked" if args.check else "applied",
              "tested_image": args.tested_image, "source_revision": args.source_revision,
              "target_commit": manifest["target_commit"],
              "upstream_commit": manifest["upstream_commit"],
              "patch_sha256": manifest["patch_sha256"], "files": records}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "files": len(records),
                      "patch_sha256": report["patch_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
