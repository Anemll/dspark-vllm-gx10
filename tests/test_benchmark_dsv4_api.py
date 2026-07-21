# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "benchmarks" / "benchmark_dsv4_api.py"
SPEC = importlib.util.spec_from_file_location("benchmark_dsv4_api", PATH)
assert SPEC and SPEC.loader
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


def metrics(drafts: int, drafted: int, accepted: int, positions: list[int]) -> str:
    rows = [
        f'vllm:spec_decode_num_drafts_total{{model_name="m"}} {drafts}',
        f'vllm:spec_decode_num_draft_tokens_total{{model_name="m"}} {drafted}',
        f'vllm:spec_decode_num_accepted_tokens_total{{model_name="m"}} {accepted}',
    ]
    rows.extend(
        'vllm:spec_decode_num_accepted_tokens_per_pos_total'
        f'{{model_name="m",position="{position}"}} {value}'
        for position, value in enumerate(positions)
    )
    return "\n".join(rows) + "\n"


class SpecMetricsTests(unittest.TestCase):
    def test_load_prompt_defaults_to_canonical_prompt_and_records_hash(self) -> None:
        prompt, digest = bench.load_prompt(None)
        self.assertEqual(prompt, bench.PROMPT)
        self.assertEqual(digest, hashlib.sha256(prompt.encode()).hexdigest())

    def test_load_prompt_file_preserves_exact_text_and_rejects_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt.txt"
            path.write_text("exact agentic prompt\n", encoding="utf-8")
            prompt, digest = bench.load_prompt(str(path))
            self.assertEqual(prompt, "exact agentic prompt\n")
            self.assertEqual(digest, hashlib.sha256(prompt.encode()).hexdigest())
            path.write_text(" \n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                bench.load_prompt(str(path))

    def test_tool_agentic_preset_has_the_archived_prompt_hash(self) -> None:
        prompt, digest = bench.load_prompt(None, "tool_agentic")
        self.assertEqual(prompt, bench.TOOL_AGENTIC_PROMPT)
        self.assertEqual(
            digest,
            "6173a7ae0ea3c64b364d0c405be28808efb8486c68a7011e966d31ce222c1736",
        )

    def test_parse_and_delta_per_position_acceptance(self) -> None:
        before = bench.parse_spec_metrics(metrics(10, 30, 18, [9, 6, 3]))
        after = bench.parse_spec_metrics(metrics(20, 60, 39, [19, 13, 7]))
        delta = bench.spec_metrics_delta(before, after, expected_positions=3)
        self.assertEqual(delta["num_drafts"], 10)
        self.assertEqual(delta["draft_tokens"], 30)
        self.assertEqual(delta["accepted_tokens"], 21)
        self.assertEqual(delta["accepted_tokens_per_position"], [10, 7, 4])
        self.assertEqual(delta["per_position_acceptance_rates"], [1.0, 0.7, 0.4])
        self.assertAlmostEqual(delta["mean_draft_length"], 3.0)
        self.assertAlmostEqual(delta["accepted_excess_length"], 2.1)
        self.assertAlmostEqual(delta["aggregate_acceptance_rate"], 0.7)
        self.assertAlmostEqual(delta["mean_acceptance_length"], 3.1)

    def test_missing_metrics_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "counters are absent"):
            bench.parse_spec_metrics("vllm:request_success_total 2\n")

    def test_missing_position_fails(self) -> None:
        before = bench.parse_spec_metrics(metrics(0, 0, 0, [0, 0]))
        after = bench.parse_spec_metrics(metrics(10, 30, 15, [9, 6]))
        with self.assertRaisesRegex(ValueError, "metric set drifted"):
            bench.spec_metrics_delta(before, after, expected_positions=3)

    def test_inconsistent_position_total_fails(self) -> None:
        before = bench.parse_spec_metrics(metrics(0, 0, 0, [0, 0, 0]))
        after = bench.parse_spec_metrics(metrics(10, 30, 20, [10, 7, 4]))
        with self.assertRaisesRegex(ValueError, "counters disagree"):
            bench.spec_metrics_delta(before, after, expected_positions=3)

    def test_nonmonotonic_position_counts_fail(self) -> None:
        before = bench.parse_spec_metrics(metrics(0, 0, 0, [0, 0, 0]))
        after = bench.parse_spec_metrics(metrics(10, 30, 20, [8, 9, 3]))
        with self.assertRaisesRegex(ValueError, "not monotonic"):
            bench.spec_metrics_delta(before, after, expected_positions=3)

    def test_no_draft_accepts_absent_or_unchanged_counters(self) -> None:
        absent = bench.spec_metrics_inactive(
            "vllm:request_success_total 2\n",
            "vllm:request_success_total 3\n",
        )
        self.assertEqual(absent["counter_state"], "absent")
        self.assertEqual(absent["num_drafts"], 0)

        snapshot = metrics(10, 30, 18, [9, 6, 3])
        unchanged = bench.spec_metrics_inactive(snapshot, snapshot)
        self.assertEqual(unchanged["counter_state"], "present_unchanged")
        self.assertEqual(unchanged["draft_tokens"], 0)

    def test_no_draft_rejects_counter_movement(self) -> None:
        before = metrics(10, 30, 18, [9, 6, 3])
        after = metrics(11, 33, 20, [10, 7, 3])
        with self.assertRaisesRegex(ValueError, "moved during no-draft"):
            bench.spec_metrics_inactive(before, after)


if __name__ == "__main__":
    unittest.main()
