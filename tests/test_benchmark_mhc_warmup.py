# SPDX-License-Identifier: MIT
"""CPU-only checks for canary cache isolation and provenance helpers."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "benchmarks/benchmark_mhc_warmup.py"
spec = importlib.util.spec_from_file_location("mhc_canary_under_test", SOURCE)
canary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canary)


class CanaryHelpersTests(unittest.TestCase):
    def test_import_and_help_do_not_import_gpu_packages(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import runpy,sys; runpy.run_path(sys.argv[1]); "
             "assert 'torch' not in sys.modules; assert 'tilelang' not in sys.modules",
             str(SOURCE)],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(
            [sys.executable, str(SOURCE), "--help"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--baseline-source", result.stdout)

    def test_unique_cache_env_preserves_home_and_parent_environment(self):
        before = dict(os.environ)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = canary.prepare_environment(root)
            for name in canary.CACHE_NAMES:
                self.assertTrue(Path(env[name]).is_relative_to(root / "cache"))
                self.assertTrue(Path(env[name]).is_dir())
            self.assertEqual(env.get("HOME"), before.get("HOME"))
            self.assertEqual(env["TILELANG_DISABLE_CACHE"], "0")
            self.assertEqual(env["TILELANG_CLEAR_CACHE"], "0")
            self.assertEqual(canary.cache_snapshot(root / "cache"), {})
            with self.assertRaises(FileExistsError):
                canary.prepare_environment(root)
        self.assertEqual(dict(os.environ), before)

    def test_cache_snapshots_hash_changes_and_refuse_external_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "kernel.so"
            artifact.write_bytes(b"first")
            before = canary.cache_snapshot(root)
            artifact.write_bytes(b"second")
            after = canary.cache_snapshot(root)
            self.assertNotEqual(before["kernel.so"]["sha256"],
                                after["kernel.so"]["sha256"])
            (root / "outside").symlink_to(SOURCE)
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                canary.cache_snapshot(root)

    def test_live_adapter_library_is_hashed_and_scoped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / "generated.so"
            library.write_bytes(b"generated binary")
            cache = SimpleNamespace(_memory_cache={
                "key": SimpleNamespace(
                    adapter=SimpleNamespace(libpath=str(library)),
                    kernel_source="synthetic source",
                ),
            })
            rows = canary.tilelang_binary_evidence(cache, root)
            self.assertTrue(rows[0]["library_inside_isolated_root"])
            self.assertEqual(rows[0]["library"]["sha256"], canary.sha256(library))
            cache._memory_cache["key"].adapter.libpath = str(SOURCE)
            rows = canary.tilelang_binary_evidence(cache, root)
            self.assertFalse(rows[0]["library_inside_isolated_root"])

    def test_invalid_timeout_is_rejected_before_output_or_imports(self):
        result = subprocess.run(
            [sys.executable, str(SOURCE), "--baseline-source", str(SOURCE),
             "--warmup-source", str(SOURCE), "--output-dir", "unused-canary-output",
             "--timeout", "601"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("timeout must be 1..600", result.stderr)


if __name__ == "__main__":
    unittest.main()
