# SPDX-License-Identifier: MIT
"""CPU-only checks for canary cache isolation and provenance helpers."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import FunctionType, SimpleNamespace
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

    def test_selects_dispatcher_cache_not_unused_base_singleton(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / "generated.so"
            library.write_bytes(b"live TVMFFI binary")
            unused_base = SimpleNamespace(_memory_cache={})
            live_cache = SimpleNamespace(
                _memory_cache={
                    "compiled": SimpleNamespace(
                        adapter=SimpleNamespace(libpath=str(library)),
                        kernel_source="source",
                    ),
                },
                _get_cache_root=lambda: str(root),
            )
            other_cache = SimpleNamespace(
                _memory_cache={}, _get_cache_root=lambda: str(root)
            )
            dispatch = {"tvm_ffi": live_cache, "nvrtc": other_cache}

            def cached_stub():
                pass

            module = SimpleNamespace(
                _dispatch_map=dispatch,
                cached=FunctionType(cached_stub.__code__, {"_dispatch_map": dispatch}),
                KernelCache=lambda: unused_base,
            )
            selected = canary.select_live_tilelang_caches(module)
            self.assertIs(selected["tvm_ffi"], live_cache)
            self.assertIs(selected["nvrtc"], other_cache)
            self.assertEqual(unused_base._memory_cache, {})
            rows = canary.live_tilelang_binary_evidence(selected, root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["execution_backend"], "tvm_ffi")
            self.assertEqual(rows[0]["library"]["sha256"], canary.sha256(library))
            self.assertTrue(rows[0]["library_inside_isolated_root"])

    def test_rejects_unbound_or_missing_live_dispatcher(self):
        with self.assertRaisesRegex(RuntimeError, "no auditable cache dispatcher"):
            canary.select_live_tilelang_caches(SimpleNamespace(_dispatch_map={}))
        with self.assertRaisesRegex(RuntimeError, "does not reference"):
            canary.select_live_tilelang_caches(
                SimpleNamespace(_dispatch_map={"tvm_ffi": object()}, cached=lambda: None)
            )

    def test_tvmffi_provenance_tracks_exact_live_export_without_libpath(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()

            class TVMFFIKernelCache:
                kernel_lib_path = "executable.so"

                def __init__(self):
                    self._memory_cache = {}

                def _get_cache_path(self, key):
                    return str(root / key)

                def _save_so_cubin_to_disk(self, kernel, cache_path, verbose=False):
                    (Path(cache_path) / self.kernel_lib_path).write_bytes(b"exported")

            cache = TVMFFIKernelCache()
            caches = {"tvm_ffi": cache}
            exports, restore = canary.observe_tilelang_executable_exports(caches, root)
            kernel = SimpleNamespace(adapter=SimpleNamespace(executable=object()))
            staging = root / "staging"
            staging.mkdir()
            cache._save_so_cubin_to_disk(kernel, str(staging))
            staging.rename(root / "key")
            cache._memory_cache["key"] = kernel
            rows = canary.live_tilelang_binary_evidence(caches, root, exports)
            self.assertTrue(rows[0]["library_provenance_verified"])
            self.assertTrue(rows[0]["library_inside_isolated_root"])
            self.assertEqual(rows[0]["library_provenance"],
                             "observed_live_executable_export")
            (root / "key" / "executable.so").write_bytes(b"different binary")
            rows = canary.live_tilelang_binary_evidence(caches, root, exports)
            self.assertFalse(rows[0]["library_provenance_verified"])
            kernel.adapter.executable = object()
            rows = canary.live_tilelang_binary_evidence(caches, root, exports)
            self.assertNotIn("library_provenance_verified", rows[0])
            for instance, original in restore:
                instance._save_so_cubin_to_disk = original

    @staticmethod
    def valid_probe():
        return {
            "finite": True, "cache_changed": [],
            "compiler_process_observations": [],
            "tilelang_binaries": [{
                "library_inside_isolated_root": True,
                "library_provenance_verified": True,
                "library": {"sha256": "a" * 64},
            }],
        }

    def test_probe_passes_empty_cache_and_compiler_deltas(self):
        probe = self.valid_probe()
        canary.verify_post_warmup_probe(probe)
        self.assertTrue(probe["no_new_compilation_observed"])
        # Profile compile()/cached() lookups are not compiler process events.
        probe["compile_entry_observations"] = [{"function": "compile", "calls": 1}]
        probe["compiler_process_observations"] = [{"is_compiler_invocation": False}]
        canary.verify_post_warmup_probe(probe)
        self.assertTrue(probe["no_new_compilation_observed"])

    def test_probe_fails_cache_changes_and_actual_compiler(self):
        for key, value, message in (
            ("cache_changed", ["new.cubin"], "changed the isolated cache"),
            ("compiler_process_observations", [{"is_compiler_invocation": True}],
             "invoked an observed compiler"),
        ):
            with self.subTest(key=key):
                probe = self.valid_probe()
                probe[key] = value
                with self.assertRaisesRegex(RuntimeError, message):
                    canary.verify_post_warmup_probe(probe)
                self.assertFalse(probe["no_new_compilation_observed"])

    def test_probe_fails_missing_or_invalid_library_provenance(self):
        for libraries in ([], [{"library_inside_isolated_root": False}],
                          [{"library_inside_isolated_root": True,
                            "library_provenance_verified": False}],
                          [{"library_inside_isolated_root": True,
                            "library_provenance_verified": True}]):
            probe = self.valid_probe()
            probe["tilelang_binaries"] = libraries
            with self.assertRaisesRegex(RuntimeError, "lost verified library"):
                canary.verify_post_warmup_probe(probe)
            self.assertFalse(probe["no_new_compilation_observed"])

    def test_cache_delta_includes_removed_as_well_as_changed_files(self):
        previous = {"removed": {"sha256": "a"}, "changed": {"sha256": "b"},
                    "same": {"sha256": "c"}}
        current = {"added": {"sha256": "d"}, "changed": {"sha256": "e"},
                   "same": {"sha256": "c"}}
        self.assertEqual(canary.changed_cache_entries(previous, current),
                         ["added", "changed", "removed"])

    def test_compiler_processes_are_distinct_from_queries_and_profile_calls(self):
        for argv, compiling in (
            (["/cuda/bin/nvcc", "--cubin", "kernel.cu", "-o", "kernel.cubin"], True),
            (["g++", "-shared", "kernel.c", "-o", "kernel.so"], True),
            (["which", "nvcc"], False),
            (["g++", "-dumpmachine"], False),
            (["nvcc", "--version"], False),
        ):
            with self.subTest(argv=argv):
                row = canary.compiler_process_observation(
                    "subprocess.Popen", (argv[0], argv, None, {})
                )
                self.assertEqual(row["is_compiler_invocation"], compiling)
        row = canary.compiler_process_observation(
            "os.system", (b"/cuda/bin/ptxas kernel.ptx -o kernel.cubin",)
        )
        self.assertTrue(row["is_compiler_invocation"])
        self.assertIsNone(canary.compiler_process_observation("call", ("compile",)))

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
