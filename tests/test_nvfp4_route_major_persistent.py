#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU/static gates for the isolated persistent M=4 route-major probe."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

from benchmarks.nvfp4_route_major_persistent import (
    PINNED_DYNAMIC_SOURCE_SHA256,
    readiness_bank,
    simulate_expert_publication,
    transform_dynamic_source,
)
from benchmarks.probe_nvfp4_route_major_persistent_sm121 import build_parser


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "benchmarks/nvfp4_route_major_persistent.py"
PROBE = ROOT / "benchmarks/probe_nvfp4_route_major_persistent_sm121.py"


class PersistentPublicationTests(unittest.TestCase):
    def test_each_expert_publishes_after_its_final_route(self) -> None:
        publication = simulate_expert_publication((8, 3, 8, 12, 3))
        self.assertEqual(publication, ((2, 8), (3, 12), (4, 3)))
        self.assertEqual(len(publication), 3)

    def test_hot_expert_publishes_once(self) -> None:
        self.assertEqual(
            simulate_expert_publication((4, 4, 4, 4)),
            ((3, 4),),
        )

    def test_publication_input_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            simulate_expert_publication(())
        with self.assertRaisesRegex(ValueError, "non-negative"):
            simulate_expert_publication((0, -1))

    def test_two_physical_readiness_banks_alternate(self) -> None:
        self.assertEqual([readiness_bank(i) for i in range(6)], [0, 1, 0, 1, 0, 1])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            readiness_bank(-1)


class PersistentSourceContractTests(unittest.TestCase):
    def test_source_and_probe_parse(self) -> None:
        ast.parse(CORE.read_text())
        ast.parse(PROBE.read_text())

    def test_transform_rejects_any_unpinned_source(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "SHA drift"):
            transform_dynamic_source("class MoEDynamicKernel:\n    pass\n")

    def test_source_pins_true_overlap_invariants(self) -> None:
        source = CORE.read_text()
        self.assertEqual(
            PINNED_DYNAMIC_SOURCE_SHA256,
            "94b4dd2c25b2b01604a74c8ab4b5708fdf235c56467ebf8b12808dc52b69d106",
        )
        for required in (
            "full_tile_publish_enabled = Int32(1)",
            "expected_rows = row_counts[expert_id]",
            "if cutlass.const_expr(self.ab_stage != 2):",
            "num_stages=self.ab_stage",
            "phase2_pipeline.producer_acquire(",
            "phase2_pipeline.consumer_wait(",
            "all_work_published",
            "Int32(65536)",
            "limit=self.swiglu_limit",
            "tempfile.NamedTemporaryFile(",
            'compile(transformed, str(runtime_path), "exec")',
            "Path(module.__file__).unlink(missing_ok=True)",
        ):
            self.assertIn(required, source)
        self.assertIn("route_compute_barrier_removed=True", source)

    def test_probe_has_hardware_correctness_and_latency_gates(self) -> None:
        source = PROBE.read_text()
        for required in (
            '"no_runtime_overlap"',
            '"double_buffer_numeric"',
            '"graph_numeric"',
            '"performance"',
            '"prefill_changed": False',
            'moe_dispatch._DYNAMIC_SLICE_CHUNK',
            'moe_dispatch._level_tile_n("fp4")',
        ):
            self.assertIn(required, source)

        parser = build_parser()
        args = parser.parse_args(
            ["--layer-file", "/tmp/layer.safetensors", "--output", "/tmp/out.json"]
        )
        self.assertEqual(args.m, 4)
        self.assertEqual(args.m4_max_ms, 0.682812)
        self.assertEqual(args.numeric_min_cosine, 0.98)
        self.assertEqual(args.numeric_max_nrmse, 0.25)


if __name__ == "__main__":
    unittest.main()
