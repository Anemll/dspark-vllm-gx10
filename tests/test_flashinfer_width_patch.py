# SPDX-License-Identifier: MIT
"""Exact backport and fail-before-write checks; no GPU or network required."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "apply-flashinfer-width-patch.py"
MANIFEST = ROOT / "patches" / "flashinfer" / "runtime-manifest.json"
spec = importlib.util.spec_from_file_location("width_patch", HELPER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExactHunkTests(unittest.TestCase):
    def test_no_fuzz_or_line_offset(self):
        section = ["--- a/file\n", "+++ b/file\n", "@@ -2,2 +2,2 @@\n",
                   " beta\n", "-gamma\n", "+delta\n"]
        self.assertEqual(module.apply_exact(b"alpha\nbeta\ngamma\n", section),
                         b"alpha\nbeta\ndelta\n")
        with self.assertRaisesRegex(ValueError, "context mismatch"):
            module.apply_exact(b"extra\nalpha\nbeta\ngamma\n", section)

    def test_bad_hunk_counts_fail(self):
        with self.assertRaisesRegex(ValueError, "line counts"):
            module.apply_exact(b"alpha\n", ["@@ -1,2 +1,1 @@\n", "-alpha\n", "+beta\n"])


class PinnedRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST.read_text())
        cls.reference = ROOT / ".build" / "flashinfer-upstream"
        check = subprocess.run(["git", "-C", str(cls.reference), "cat-file", "-e",
                                cls.manifest["target_commit"]], capture_output=True)
        if check.returncode:
            raise unittest.SkipTest("exact pinned FlashInfer reference is not available")
        cls.originals = {}
        for entry in cls.manifest["files"]:
            data = subprocess.check_output(["git", "-C", str(cls.reference), "show",
                                            cls.manifest["target_commit"] + ":" + entry["source"]])
            if hashlib.sha256(data).hexdigest() != entry["old_sha256"]:
                raise AssertionError("reference file does not match manifest")
            cls.originals[entry["installed"]] = data

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / "flashinfer"
        for relative, data in self.originals.items():
            target = self.package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

    def invoke(self, manifest=MANIFEST, *extra):
        return subprocess.run([sys.executable, str(HELPER), "--manifest", str(manifest),
                               "--package-dir", str(self.package), "--report",
                               str(self.root / "report.json"), *extra],
                              text=True, capture_output=True)

    def assert_originals(self, except_path=None):
        for relative, expected in self.originals.items():
            if relative != except_path:
                self.assertEqual((self.package / relative).read_bytes(), expected, relative)

    def mutated_manifest(self, mutate):
        manifest = json.loads(MANIFEST.read_text())
        mutate(manifest)
        path = self.root / "runtime-manifest.json"
        path.write_text(json.dumps(manifest))
        patch = MANIFEST.parent / manifest["patch_file"]
        (self.root / patch.name).write_bytes(patch.read_bytes())
        return path

    def test_exact_pinned_runtime_check_and_apply(self):
        check = self.invoke(MANIFEST, "--check")
        self.assertEqual(check.returncode, 0, check.stderr)
        self.assert_originals()
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        for entry in self.manifest["files"]:
            actual = (self.package / entry["installed"]).read_bytes()
            self.assertEqual(hashlib.sha256(actual).hexdigest(), entry["new_sha256"])
        second = self.invoke()
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("original SHA-256 mismatch", second.stderr)

    def test_last_file_mismatch_causes_no_earlier_writes(self):
        last = self.manifest["files"][-1]["installed"]
        (self.package / last).write_bytes(self.originals[last] + b"// mismatch\n")
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("original SHA-256 mismatch", result.stderr)
        self.assert_originals(except_path=last)
        self.assertEqual((self.package / last).read_bytes(), self.originals[last] + b"// mismatch\n")

    def test_traversal_is_refused_before_writes(self):
        manifest = self.mutated_manifest(
            lambda value: value["files"][-1].update(installed="../outside.cuh"))
        result = self.invoke(manifest)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("non-traversing relative path", result.stderr)
        self.assert_originals()

    def test_symlink_target_is_refused_before_writes(self):
        last = self.manifest["files"][-1]["installed"]
        target = self.package / last
        outside = self.root / "outside.cuh"
        outside.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(outside)
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("symlink", result.stderr)
        self.assert_originals()
        self.assertEqual(outside.read_bytes(), self.originals[last])

    def test_new_hash_mismatch_causes_no_writes(self):
        manifest = self.mutated_manifest(
            lambda value: value["files"][-1].update(new_sha256="0" * 64))
        result = self.invoke(manifest)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("patched SHA-256 mismatch", result.stderr)
        self.assert_originals()


if __name__ == "__main__":
    unittest.main()
