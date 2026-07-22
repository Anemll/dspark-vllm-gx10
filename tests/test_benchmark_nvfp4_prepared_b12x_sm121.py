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
from unittest.mock import patch

from benchmarks import benchmark_nvfp4_prepared_b12x_sm121 as probe
from scripts import patch_b12x_nvfp4_swiglu_limit as b12x_clamp_patch
from scripts import patch_vllm_b12x_output_alias as output_alias_patch


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

    def test_order_balanced_timing_keeps_requested_speedup_direction(self) -> None:
        launches = {
            "direct_output": lambda: 1.0,
            "legacy_two_copy": lambda: 2.0,
        }

        def fake_measure(_torch, launch, **_kwargs):
            return {"median_ms": launch()}

        with patch.object(
            probe.kernel_bench,
            "measure_cuda_events",
            side_effect=fake_measure,
        ):
            result = probe._time_orders(
                object(),
                launches,
                warmup=1,
                iters=1,
                repeats=1,
                pair=("direct_output", "legacy_two_copy"),
            )

        self.assertEqual(
            result["combined"]["speedup_direct_output_over_legacy_two_copy"],
            2.0,
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
        self.assertIn("dspark-patch-vllm-b12x-output-alias", dockerfile)
        self.assertIn("dspark-patch-b12x-nvfp4-swiglu-limit", dockerfile)
        self.assertIn("pairwise-tiny-decode-v2", dockerfile)
        dockerignore = (
            root
            / "docker"
            / "Dockerfile.nvfp4-b12x-decode-overlay.dockerignore"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "!benchmarks/benchmark_nvfp4_prepared_b12x_sm121.py",
            dockerignore,
        )
        self.assertIn("!scripts/patch_vllm_b12x_output_alias.py", dockerignore)
        self.assertIn(
            "!scripts/patch_b12x_nvfp4_swiglu_limit.py", dockerignore
        )

    def test_pinned_modular_patch_adds_only_explicit_alias_contract(self) -> None:
        source = (
            "prefix\n"
            + output_alias_patch._PROPERTY_ANCHOR
            + "middle\n"
            + output_alias_patch._ALIAS_ANCHOR
            + "suffix\n"
        )
        patched = output_alias_patch.patch_source(source)

        self.assertEqual(patched.count("def supports_output_alias"), 1)
        self.assertEqual(
            patched.count("self.fused_experts.supports_output_alias"), 1
        )
        self.assertEqual(patched.count("if current_platform.is_rocm():"), 1)
        self.assertRegex(
            output_alias_patch.PINNED_SOURCE_SHA256, r"^[0-9a-f]{64}$"
        )
        self.assertRegex(
            output_alias_patch.PINNED_RESULT_SHA256, r"^[0-9a-f]{64}$"
        )

    def test_pinned_modular_patch_rejects_anchor_drift(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "experts property"):
            output_alias_patch.patch_source("not the pinned source")

    def test_pinned_b12x_patch_enables_only_direct_micro_clamp(self) -> None:
        source = "\n".join(
            anchor for anchor, _replacement, _label in b12x_clamp_patch._REPLACEMENTS
        )
        patched = b12x_clamp_patch.patch_source(source)

        self.assertIn("swiglu_limit=swiglu_limit,\n            device=a.device", patched)
        self.assertIn(
            "NVFP4 swiglu_limit requires the compact direct microkernel",
            patched,
        )
        self.assertNotIn(
            "swiglu_limit is implemented only for W4A16 MoE", patched
        )
        self.assertRegex(
            b12x_clamp_patch.PINNED_SOURCE_SHA256, r"^[0-9a-f]{64}$"
        )

    def test_pinned_b12x_patch_rejects_anchor_drift(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "micro signature"):
            b12x_clamp_patch.patch_source("not the pinned source")

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
