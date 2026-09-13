#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Contract checks for the dashboard's live request-capacity display."""

from __future__ import annotations

import unittest
from pathlib import Path

from dashboard.server import CapacitySampler, parse_max_num_seqs, parse_prometheus


ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "dashboard" / "index.html").read_text()
SERVER = (ROOT / "dashboard" / "server.py").read_text()


class DashboardCapacityTests(unittest.TestCase):
    def test_parses_space_separated_value(self):
        self.assertEqual(parse_max_num_seqs("vllm serve /model --max-num-seqs 10 --port 8888"), 10)

    def test_parses_equals_value(self):
        self.assertEqual(parse_max_num_seqs("vllm serve /model --max-num-seqs=4"), 4)

    def test_returns_none_when_setting_is_absent(self):
        self.assertIsNone(parse_max_num_seqs("vllm serve /model --port 8888"))

    def test_capacity_sampler_uses_direct_docker_access(self):
        constants = CapacitySampler.snapshot.__code__.co_consts
        self.assertIn("/usr/bin/docker", constants)
        self.assertNotIn("sudo", constants)

    def test_dashboard_renders_live_capacity(self):
        self.assertIn('id="maxActiveRequests"', INDEX)
        self.assertIn('integer(data.maxActiveRequests)', INDEX)
        self.assertIn("FP8 DS-MLA cache utilization", INDEX)
        self.assertNotIn("NVFP4 DS MLA cache utilization", INDEX)

    def test_aggregate_rate_shows_active_request_count(self):
        self.assertIn('id="aggregateRequests"', INDEX)
        self.assertIn('data.running === 1 ? "1 active request"', INDEX)
        self.assertIn('`${integer(data.running)} active requests`', INDEX)

    def test_prefill_histogram_sums_are_collected(self):
        metrics = parse_prometheus(
            "\n".join(
                [
                    'vllm:request_prefill_time_seconds_sum{model_name="m"} 2.5',
                    'vllm:request_prefill_time_seconds_count{model_name="m"} 2',
                    'vllm:request_prefill_kv_computed_tokens_sum{model_name="m"} 5000',
                    'vllm:request_prefill_kv_computed_tokens_count{model_name="m"} 2',
                ]
            )
        )
        self.assertEqual(metrics["prefill_time_sum"], 2.5)
        self.assertEqual(metrics["prefill_kv_tokens_sum"], 5000.0)

    def test_dashboard_renders_completed_prefill_rate(self):
        self.assertIn('id="completedPrefillTps"', INDEX)
        self.assertIn('value(data.completedPrefillTps, " t/s")', INDEX)
        self.assertIn(
            '"completedPrefillTps": completed_prefill_tps,\n                "running"',
            SERVER,
        )
        self.assertIn(
            '"completedPrefillTps": self._last_completed_prefill_tps,\n'
            '                "dsparkAcceptancePct"',
            SERVER,
        )
        self.assertIn("self._last_completed_prefill_tps = completed_prefill_tps", SERVER)


if __name__ == "__main__":
    unittest.main()
