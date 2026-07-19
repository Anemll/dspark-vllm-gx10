# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
PATH = BENCHMARKS / "benchmark_dsv4_context_decode.py"
SPEC = importlib.util.spec_from_file_location("benchmark_dsv4_context_decode", PATH)
assert SPEC and SPEC.loader
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


class ContextDecodeTests(unittest.TestCase):
    def test_result_uses_mia_decode_formula(self) -> None:
        result = bench.build_result(
            target_prompt_tokens=1024,
            trial=1,
            prompt_digest="abc",
            usage={"prompt_tokens": 1024, "completion_tokens": 512},
            ttft_s=2.0,
            decode_window_s=10.0,
            elapsed_s=12.0,
            chunks=100,
            finish_reason="length",
            spec_decode={
                "aggregate_acceptance_rate": 0.5,
                "mean_acceptance_length": 2.5,
                "per_position_acceptance_rates": [0.8, 0.5, 0.2],
            },
        )
        self.assertAlmostEqual(result.decode_only_tps, 51.1)
        self.assertAlmostEqual(result.end_to_end_tps, 512 / 12)

    def test_prompt_length_drift_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "prompt-token drift"):
            bench.build_result(
                target_prompt_tokens=1024,
                trial=1,
                prompt_digest="abc",
                usage={"prompt_tokens": 1023, "completion_tokens": 512},
                ttft_s=1.0,
                decode_window_s=10.0,
                elapsed_s=11.0,
                chunks=100,
                finish_reason="length",
                spec_decode={},
            )

    def test_summary_preserves_per_position_medians(self) -> None:
        rows = []
        for trial, decode, acceptance in ((1, 50.0, 0.4), (2, 60.0, 0.6)):
            rows.append(
                bench.ContextDecodeResult(
                    target_prompt_tokens=1024,
                    trial=trial,
                    prompt_sha256=str(trial),
                    prompt_tokens=1024,
                    completion_tokens=512,
                    chunks=100,
                    ttft_s=float(trial),
                    decode_window_s=511 / decode,
                    elapsed_s=12.0,
                    end_to_end_tps=512 / 12,
                    decode_only_tps=decode,
                    finish_reason="length",
                    spec_decode={
                        "aggregate_acceptance_rate": acceptance,
                        "mean_acceptance_length": 2 + acceptance,
                        "per_position_acceptance_rates": [0.8, acceptance, 0.2],
                    },
                )
            )
        summary = bench.build_summary(rows)[0]
        self.assertEqual(summary["median_decode_only_tps"], 55.0)
        self.assertEqual(summary["median_acceptance_rate"], 0.5)
        self.assertEqual(summary["median_per_position_acceptance"], [0.8, 0.5, 0.2])


if __name__ == "__main__":
    unittest.main()
