# SPDX-License-Identifier: MIT
"""Run component tests from a small tar stream without editing node source.

The archive contains overlay/vllm/** and tests/**. No extraction, package
installation, GPU allocation, or service mutation occurs. Intended for a
disposable CPU-only container based on the immutable text image.
"""
import importlib.abc
import importlib.util
import io
import linecache
import os
import sys
import tarfile


files = {}
with tarfile.open(fileobj=sys.stdin.buffer, mode="r|*") as archive:
    for entry in archive:
        if entry.isfile() and entry.size <= 2_000_000:
            files[entry.name.removeprefix("./")] = archive.extractfile(entry).read()


class OverlayLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        key = "overlay/" + fullname.replace(".", "/") + ".py"
        if key in files:
            return importlib.util.spec_from_loader(fullname, self, origin=key)

    def create_module(self, spec):
        return None

    def get_source(self, fullname):
        return files["overlay/" + fullname.replace(".", "/") + ".py"].decode()

    def exec_module(self, module):
        module.__file__ = module.__spec__.origin
        source = files[module.__file__].decode()
        linecache.cache[module.__file__] = (
            len(source), None, source.splitlines(True), module.__file__
        )
        exec(compile(files[module.__file__], module.__file__, "exec"), module.__dict__)


sys.meta_path.insert(0, OverlayLoader())
key = os.environ.get("VISION_TEST_FILE", "tests/vision_components.py")
if key not in ("tests/vision_components.py", "tests/vision_gpu_components.py"):
    raise ValueError("Unsupported test entrypoint")
exec(compile(files[key], key, "exec"), {"__name__": "__main__", "__file__": key})
