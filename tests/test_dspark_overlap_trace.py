# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "overlay/vllm/v1/worker/gpu/spec_decode/dspark/overlap_trace.py"
)
SPEC = importlib.util.spec_from_file_location("dspark_overlap_trace", MODULE_PATH)
assert SPEC and SPEC.loader
overlap = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = overlap
SPEC.loader.exec_module(overlap)


def trace(world_size: int = 2) -> dict[str, object]:
    return {
        "schema_version": 1,
        "world_size": world_size,
        "rank_traces": [
            {
                "rank": rank,
                "draft": 10.0 + rank,
                "verify": 20.0 + rank,
                "commit": 3.0 + rank,
                "nccl_wait": 1.0 + rank,
                "overhead": 5.0 + rank,
                "total": 39.0 + 5 * rank,
            }
            for rank in range(world_size)
        ],
    }


class OverlapTraceConfigTests(unittest.TestCase):
    def test_trace_is_strictly_opt_in(self) -> None:
        self.assertFalse(overlap.overlap_trace_enabled({}))
        self.assertFalse(
            overlap.overlap_trace_enabled({overlap.TRACE_ENV: "0"})
        )
        self.assertTrue(
            overlap.overlap_trace_enabled({overlap.TRACE_ENV: "1"})
        )
        for value in ("true", "on", "2", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                overlap.overlap_trace_enabled({overlap.TRACE_ENV: value})

    def test_metrics_preserve_every_phase_and_rank_count(self) -> None:
        try:
            from prometheus_client import CollectorRegistry, generate_latest
        except ModuleNotFoundError:
            self.skipTest("prometheus_client unavailable")
        registry = CollectorRegistry()
        metrics = overlap.DSparkOverlapMetrics(registry=registry)
        metrics.observe(trace())
        metrics.observe(trace())
        text = generate_latest(registry).decode()
        for rank in (0, 1):
            self.assertIn(
                f'vllm:dspark_overlap_blocks_total{{rank="{rank}"}} 2.0', text
            )
            for phase in overlap.PHASES:
                self.assertIn(
                    "vllm:dspark_overlap_phase_ms_count"
                    f'{{phase="{phase}",rank="{rank}"}} 2.0',
                    text,
                )

    def test_metrics_fail_closed_on_incomplete_or_invalid_rank_rows(self) -> None:
        try:
            from prometheus_client import CollectorRegistry
        except ModuleNotFoundError:
            self.skipTest("prometheus_client unavailable")
        metrics = overlap.DSparkOverlapMetrics(registry=CollectorRegistry())
        incomplete = trace()
        incomplete["rank_traces"] = incomplete["rank_traces"][:1]
        with self.assertRaisesRegex(RuntimeError, "incomplete overlap"):
            metrics.observe(incomplete)
        duplicate = trace()
        duplicate["rank_traces"][1]["rank"] = 0
        with self.assertRaisesRegex(RuntimeError, "invalid overlap rank"):
            metrics.observe(duplicate)
        invalid = trace()
        invalid["rank_traces"][0]["draft"] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "invalid overlap draft"):
            metrics.observe(invalid)
        nonconserving = trace()
        nonconserving["rank_traces"][0]["total"] = 999.0
        with self.assertRaisesRegex(RuntimeError, "phase conservation drift"):
            metrics.observe(nonconserving)


if __name__ == "__main__":
    unittest.main()
