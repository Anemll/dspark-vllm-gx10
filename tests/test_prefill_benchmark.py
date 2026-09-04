# SPDX-License-Identifier: MIT
"""Offline coverage for exact-token prefill measurements and partial failures."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from benchmarks import benchmark_prefill as prefill


def event(text="", finish=None):
    return {"choices": [{"index": 0, "text": text, "finish_reason": finish}]}


def usage(prompt=3, completion=1):
    return {"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


def fake_stream(events):
    return io.BytesIO("".join("data:" + (item if isinstance(item, str) else json.dumps(item)) + "\n\n" for item in events).encode())


class PrefillStreamTests(unittest.TestCase):
    def run_fixture(self, events, timeout=30, evidence=None):
        times = iter(index / 10 for index in range(100))
        with patch.object(prefill.urllib.request, "urlopen", return_value=fake_stream(events)), patch.object(prefill.time, "perf_counter", side_effect=lambda: next(times)):
            return prefill.run_completion("http://fixture.invalid", "fixture", [1, 2, 3], timeout, evidence=evidence)

    def test_empty_metadata_does_not_start_ttft(self):
        evidence = {}
        ttft, elapsed, counters, finish = self.run_fixture([event(), event("x", "length"), usage(), "[DONE]"], evidence=evidence)
        self.assertAlmostEqual(ttft, 0.3)
        self.assertGreater(elapsed, ttft)
        self.assertEqual(counters["completion_tokens"], 1)
        self.assertEqual(finish, "length")
        self.assertEqual(evidence["status"], "complete")
        self.assertTrue(evidence["done_received"])

    def test_whitespace_is_a_real_nonempty_token(self):
        ttft, _, _, _ = self.run_fixture([event(" ", "length"), usage(), "[DONE]"])
        self.assertGreater(ttft, 0)

    def test_metadata_only_stream_fails(self):
        with self.assertRaisesRegex(ValueError, "nonempty completion text"):
            self.run_fixture([event("", "length"), usage(), "[DONE]"])

    def test_usage_is_required_and_exact(self):
        for counters in (None, usage(prompt=4), usage(completion=0), usage(completion=2)):
            events = [event("x", "length")]
            if counters:
                events.append(counters)
            events.append("[DONE]")
            with self.subTest(counters=counters), self.assertRaisesRegex(ValueError, "usage"):
                self.run_fixture(events)

    def test_finish_and_done_are_required(self):
        with self.assertRaisesRegex(ValueError, "finish_reason"):
            self.run_fixture([event("x"), usage(), "[DONE]"])
        with self.assertRaisesRegex(ValueError, r"\[DONE\]"):
            self.run_fixture([event("x", "length"), usage()])

    def test_deadline_is_not_reset_by_stream_activity(self):
        evidence = {}
        with self.assertRaisesRegex(TimeoutError, "deadline"):
            self.run_fixture([event(), event("x", "length"), usage(), "[DONE]"], timeout=0.25, evidence=evidence)
        self.assertEqual(evidence["status"], "failed")
        self.assertTrue(evidence["events"])

    def test_http_failure_retains_evidence(self):
        evidence = {}
        with patch.object(prefill.urllib.request, "urlopen", side_effect=urllib.error.URLError("offline")):
            with self.assertRaises(urllib.error.URLError):
                prefill.run_completion("http://fixture.invalid", "fixture", [1], 30, evidence=evidence)
        self.assertEqual(evidence["status"], "failed")
        self.assertEqual(evidence["request_body"]["prompt"], [1])

    def test_tokenizer_rejects_invalid_ids(self):
        with patch.object(prefill, "request_json", return_value={"tokens": [1, 2.5]}):
            with self.assertRaises(ValueError):
                prefill.tokenize("http://fixture.invalid", "fixture", "text")

    def test_prompt_generation_retains_unique_reproducible_trials(self):
        first = prefill.make_prompt(list(range(100)), 256, 1, 4104)
        self.assertEqual(first, prefill.make_prompt(list(range(100)), 256, 1, 4104))
        self.assertNotEqual(first[:64], prefill.make_prompt(list(range(100)), 256, 2, 4104)[:64])


class PrefillReportTests(unittest.TestCase):
    def test_failed_second_request_preserves_first_result(self):
        count = 0

        def complete(base, model, prompt, timeout, *, evidence=None):
            nonlocal count
            count += 1
            evidence["request_body"] = {"prompt": prompt}
            if count == 2:
                evidence["error"] = "fixture failure"
                raise ValueError("fixture failure")
            return 1.0, 1.1, {"prompt_tokens": len(prompt), "completion_tokens": 1}, "length"

        snapshots = [
            prefill.MetricSnapshot(10, 1, 3, 0, 3),
            prefill.MetricSnapshot(11, 2, 6, 0, 6),
            prefill.MetricSnapshot(11, 2, 6, 0, 6),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "prefill.json")
            command = ["prefill", "--sizes", "3", "--trials", "2", "--warmup-tokens", "0", "--shape-warmup-trials", "0", "--output", output]
            with patch("sys.argv", command), patch.object(prefill, "request_json", return_value={"version": "fixture"}), patch.object(prefill, "tokenize", return_value=list(range(20))), patch.object(prefill, "run_completion", side_effect=complete), patch.object(prefill, "snapshot_metrics", side_effect=snapshots), patch("builtins.print"):
                with self.assertRaisesRegex(ValueError, "fixture failure"):
                    prefill.main()
            report = json.loads(Path(output).read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(len(report["results"]), 1)
            self.assertEqual(len(report["requests"]), 2)
            self.assertEqual(report["requests"][1]["status"], "failed")
            self.assertEqual(report["results"][0]["server_prefill_tps"], 3)
            self.assertTrue(report["results"][0]["metrics_exact"])

    def test_metadata_failure_still_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "prefill.json")
            with patch("sys.argv", ["prefill", "--output", output]), patch.object(prefill, "request_json", side_effect=ValueError("offline")):
                with self.assertRaises(ValueError):
                    prefill.main()
            report = json.loads(Path(output).read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["results"], [])


if __name__ == "__main__":
    unittest.main()
