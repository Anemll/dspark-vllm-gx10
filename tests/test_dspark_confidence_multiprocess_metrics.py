# SPDX-License-Identifier: MIT

import ast
import os
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
CLI_MAIN = ROOT / "overlay/vllm/entrypoints/cli/main.py"
PROBE = ROOT / "scripts/probe_dspark_confidence_multiprocess_metrics.py"
DOCKERFILE = ROOT / "docker/Dockerfile.nvfp4-aot-overlay"
CONFIDENCE_DOCKERFILE = ROOT / "docker/Dockerfile.dspark-confidence-overlay"
RUNTIME_DOCKERFILE = ROOT / "docker/Dockerfile.runtime"
DOCKERIGNORE = ROOT / "docker/Dockerfile.nvfp4-aot-overlay.dockerignore"
CONFIDENCE_DOCKERIGNORE = (
    ROOT / "docker/Dockerfile.dspark-confidence-overlay.dockerignore"
)


class PrometheusBootstrapTests(unittest.TestCase):
    def _load_cli(self, argv: list[str]):
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.__path__ = []
        fake_logger = types.ModuleType("vllm.logger")
        fake_logger.init_logger = lambda _name: mock.Mock()
        old_modules = {
            name: sys.modules.get(name) for name in ("vllm", "vllm.logger")
        }
        old_argv = sys.argv
        try:
            sys.modules["vllm"] = fake_vllm
            sys.modules["vllm.logger"] = fake_logger
            sys.argv = argv
            return runpy.run_path(str(CLI_MAIN), run_name="confidence_cli_probe")
        finally:
            sys.argv = old_argv
            for name, value in old_modules.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value

    def test_serve_bootstrap_creates_empty_directory_before_vllm_imports(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
            namespace = self._load_cli(["vllm", "serve"])
            path = Path(os.environ["PROMETHEUS_MULTIPROC_DIR"])
            self.assertTrue(path.is_dir())
            self.assertEqual(list(path.iterdir()), [])
            namespace["_cleanup_owned_prometheus_multiprocess_dir"]()
            self.assertFalse(path.exists())
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)

    def test_non_serve_command_does_not_enable_multiprocess_mode(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
            self._load_cli(["vllm", "bench"])
            self.assertNotIn("PROMETHEUS_MULTIPROC_DIR", os.environ)

    def test_explicit_directory_must_exist_and_be_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "stale.db").write_text("stale")
            with mock.patch.dict(
                os.environ,
                {"PROMETHEUS_MULTIPROC_DIR": directory},
                clear=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "must be empty"):
                    self._load_cli(["vllm", "serve"])

    def test_inherited_active_directory_can_contain_worker_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "histogram_123.db").write_text("current run")
            with mock.patch.dict(
                os.environ,
                {
                    "PROMETHEUS_MULTIPROC_DIR": directory,
                    "VLLM_PROMETHEUS_MULTIPROC_ACTIVE_DIR": directory,
                },
                clear=False,
            ):
                namespace = self._load_cli(["vllm", "serve"])
                self.assertEqual(
                    namespace["_setup_prometheus_multiprocess_for_serve"](
                        ["vllm", "serve"]
                    ),
                    directory,
                )

    def test_bootstrap_precedes_first_vllm_import(self):
        tree = ast.parse(CLI_MAIN.read_text())
        setup_call = next(
            index
            for index, node in enumerate(tree.body)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None)
            == "_setup_prometheus_multiprocess_for_serve"
        )
        first_vllm_import = next(
            index
            for index, node in enumerate(tree.body)
            if isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("vllm")
        )
        self.assertLess(setup_call, first_vllm_import)


class CrossProcessProbeContractTests(unittest.TestCase):
    def test_probe_uses_real_vllm_api_metrics_router_and_http_scrape(self):
        source = PROBE.read_text()
        self.assertIn(
            "vllm.entrypoints.serve.instrumentator.metrics import attach_router",
            source,
        )
        self.assertIn("urllib.request.urlopen", source)
        self.assertIn("subprocess.Popen", source)
        self.assertIn("control unexpectedly exposed worker confidence metrics", source)
        self.assertIn("probability_bucket", source)
        self.assertIn("physical_target_rows", source)
        self.assertIn("d2h_copy_completion", source)
        self.assertIn("DraftTokensHandler.__new__", source)
        self.assertIn("handler.compact_scheduler_output(output)", source)
        self.assertIn("handler.get_last_compaction_telemetry()", source)
        self.assertIn("observe_engine_compaction_telemetry(rows, fallback)", source)

    def test_probe_compiles_and_help_parses_without_vllm_dependencies(self):
        with mock.patch.object(sys, "argv", [str(PROBE), "--help"]):
            with self.assertRaises(SystemExit) as raised:
                runpy.run_path(str(PROBE), run_name="__main__")
        self.assertEqual(raised.exception.code, 0)

    def test_overlay_image_bakes_cross_process_probe(self):
        for dockerfile in (DOCKERFILE, CONFIDENCE_DOCKERFILE, RUNTIME_DOCKERFILE):
            source = dockerfile.read_text()
            self.assertIn(
                "scripts/probe_dspark_confidence_multiprocess_metrics.py",
                source,
            )
            self.assertIn(
                "/usr/local/bin/dspark-probe-confidence-multiprocess-metrics",
                source,
            )
        self.assertIn(
            "!scripts/probe_dspark_confidence_multiprocess_metrics.py",
            DOCKERIGNORE.read_text(),
        )
        self.assertIn(
            "!scripts/probe_dspark_confidence_multiprocess_metrics.py",
            CONFIDENCE_DOCKERIGNORE.read_text(),
        )
        confidence_image = CONFIDENCE_DOCKERFILE.read_text()
        self.assertIn(
            "COPY overlay/vllm/entrypoints/cli/main.py ", confidence_image
        )
        self.assertIn("BASE_CLI_MAIN_SHA256", confidence_image)
        self.assertIn(
            "!overlay/vllm/entrypoints/cli/main.py",
            CONFIDENCE_DOCKERIGNORE.read_text(),
        )


if __name__ == "__main__":
    unittest.main()
