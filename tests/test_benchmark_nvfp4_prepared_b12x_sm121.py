# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from __future__ import annotations

import argparse
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


if __name__ == "__main__":
    unittest.main()
