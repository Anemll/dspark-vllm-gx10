# SPDX-License-Identifier: MIT
import unittest
from benchmarks.summarize_cuda_trace import summarize


def context(ts, name="execute_context_test", tid=1):
    return {"cat": "user_annotation", "name": name, "ts": ts,
            "dur": 10, "pid": 1, "tid": tid}


def launch(ts, correlation=4, tid=1):
    return {"cat": "cuda_runtime", "ts": ts, "pid": 1, "tid": tid,
            "args": {"correlation": correlation}}


def kernel(ts=100, correlation=4):
    return {"cat": "kernel", "name": "W4A16FusedMoeKernel", "ts": ts,
            "dur": 1000, "args": {"correlation": correlation}}


class TraceSummaryTests(unittest.TestCase):
    def test_gpu_lag_does_not_change_forward_owner(self):
        result = summarize([context(0), context(100), launch(3), kernel()])
        row = result["rows"][0]
        self.assertEqual(row["context_index"], 0)
        self.assertEqual(row["families"]["moe"]["kernel_sum_ms"], 1)

    def test_other_thread_is_not_attributed(self):
        result = summarize([context(0), launch(3, tid=2), kernel()])
        self.assertIsNone(result["rows"][0]["context_index"])

    def test_reused_ambiguous_correlation_is_not_guessed(self):
        result = summarize([context(0), context(20), launch(3), launch(23), kernel()])
        self.assertIsNone(result["rows"][0]["context_index"])

    def test_missing_launch_retained_and_empty_trace_supported(self):
        self.assertEqual(summarize([])["rows"], [])
        result = summarize([kernel()])
        self.assertEqual(result["rows"][0]["annotation"], "unattributed")
        self.assertEqual(result["rows"][0]["kernel_count"], 1)
