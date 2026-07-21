# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from benchmarks import benchmark_nvfp4_prepared_b12x_sm121 as probe


class PreparedB12xBenchmarkTest(unittest.TestCase):
    def test_default_matrix_contains_only_decode_shapes(self) -> None:
        parser = probe.build_parser()
        args = parser.parse_args(
            ["--layer-file", "/tmp/layer", "--output", "/tmp/result.json"]
        )

        self.assertEqual(args.m, (1, 4))
        self.assertTrue(all(m < 128 for m in args.m))

    def test_matrix_parser_rejects_empty_zero_and_negative_values(self) -> None:
        for value in ("", "0", "1,0", "-1,4"):
            with self.subTest(value=value), self.assertRaises(
                argparse.ArgumentTypeError
            ):
                probe._csv_positive_ints(value)

    def test_prepared_contract_is_exactly_eight_families(self) -> None:
        self.assertEqual(len(probe.PREPARED_FAMILY_ORDER), 8)
        self.assertEqual(
            probe.PREPARED_FAMILY_ORDER[:2], ("w13.weight", "w2.weight")
        )

    def test_candidate_image_bakes_exact_gate_and_overlay(self) -> None:
        root = Path(__file__).resolve().parents[1]
        dockerfile = (
            root / "docker" / "Dockerfile.nvfp4-b12x-decode-overlay"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "COPY overlay/vllm/ /usr/local/lib/python3.12/dist-packages/vllm/",
            dockerfile,
        )
        self.assertIn(
            "COPY benchmarks/benchmark_nvfp4_prepared_b12x_sm121.py",
            dockerfile,
        )
        dockerignore = (
            root
            / "docker"
            / "Dockerfile.nvfp4-b12x-decode-overlay.dockerignore"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "!benchmarks/benchmark_nvfp4_prepared_b12x_sm121.py",
            dockerignore,
        )

    def test_baked_sibling_layout_imports_without_repo_package(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            install_dir = Path(directory) / "usr" / "local" / "bin"
            install_dir.mkdir(parents=True)
            prepared = install_dir / "dspark-benchmark-nvfp4-prepared-b12x-sm121"
            kernel = install_dir / "dspark-benchmark-nvfp4-a4w4-sm121"
            shutil.copy2(
                root / "benchmarks" / "benchmark_nvfp4_prepared_b12x_sm121.py",
                prepared,
            )
            shutil.copy2(
                root / "benchmarks" / "benchmark_nvfp4_a4w4_sm121.py",
                kernel,
            )
            environment = dict(os.environ)
            environment.pop("PYTHONPATH", None)
            completed = subprocess.run(
                [sys.executable, str(prepared), "--help"],
                cwd=directory,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--layer-file", completed.stdout)


if __name__ == "__main__":
    unittest.main()
