# SPDX-License-Identifier: MIT
import json
import unittest

from benchmarks import benchmark_prefix_contract as prefix
from benchmarks.benchmark_dsv4_api import StreamResult
from benchmarks.benchmark_prefill import MetricSnapshot


class PrefixContractTests(unittest.TestCase):
    @staticmethod
    def tokenize(body):
        ids = [ord(character) for character in json.dumps(body["messages"], sort_keys=True)]
        return {"tokens": ids, "count": len(ids)}

    def setUp(self):
        self.cases = prefix.prepare_cases("fixture", 64, 4106, self.tokenize)
        self.before = MetricSnapshot(10, 1, 100, 0, 100)

    def evaluate(self, phase=0, *, cached=0, content=None, requests=1):
        case = self.cases[phase]
        count = case["tokenization"]["count"]
        stream = StreamResult(request=phase, ok=True, prompt_tokens=count, completion_tokens=20, finish_reason="stop", content=json.dumps(case["expected"]) if content is None else content)
        after = MetricSnapshot(11, 1 + requests, 100 + count - cached, cached, 100 + count)
        return prefix.evaluate(case, stream, self.before, after)

    def test_exact_families_and_fixed_history(self):
        self.assertEqual(len(self.cases), 8)
        self.assertEqual([self.cases[index]["tokenization"]["count"] for index in (0, 1, 4, 5)], [1023, 1023, 1025, 1025])
        self.assertEqual(self.cases[0]["request_sha256"], self.cases[1]["request_sha256"])
        self.assertNotEqual(self.cases[0]["request_sha256"], self.cases[4]["request_sha256"])
        self.assertEqual(self.cases[2]["body"]["messages"][:-1], self.cases[3]["body"]["messages"][:-1])
        self.assertNotIn("PINE263", self.cases[2]["body"]["messages"][-2]["content"])
        self.assertNotEqual(self.cases[2]["expected"], self.cases[3]["expected"])
        self.assertFalse(self.cases[0]["body"]["ignore_eos"])

    def test_correct_cold_and_cached_replay_pass(self):
        self.assertEqual(self.evaluate()["classification"], "passed")
        result = self.evaluate(phase=1, cached=768)
        self.assertEqual(result["classification"], "passed")
        self.assertTrue(result["cache_hit_observed"])

    def test_correct_replay_without_hit_is_inconclusive(self):
        self.assertEqual(self.evaluate(phase=1)["classification"], "inconclusive_cache_reuse_not_observed")

    def test_wrong_cold_is_not_labelled_cache_corruption(self):
        self.assertEqual(self.evaluate(content='{"anchor":"wrong","current":"wrong"}')["classification"], "cold_answer_failure")

    def test_stale_branch_answer_fails_semantics(self):
        result = self.evaluate(phase=2, cached=768, content=json.dumps(prefix.INITIAL))
        self.assertEqual(result["classification"], "replay_answer_failure_requires_investigation")
        self.assertFalse(result["semantic_correct"])

    def test_shared_metrics_cannot_prove_a_cache_hit(self):
        result = self.evaluate(phase=1, cached=768, requests=2)
        self.assertEqual(result["classification"], "inconclusive_metrics")
        self.assertFalse(result["cache_hit_observed"])

    def test_prewarmed_initial_is_not_a_cold_control(self):
        self.assertEqual(self.evaluate(cached=768)["classification"], "inconclusive_initial_not_cold")

    def test_json_key_order_and_whitespace_do_not_affect_semantics(self):
        text = ' { "current": "COPPER417", "anchor": "PINE263" } '
        self.assertEqual(self.evaluate(content=text)["classification"], "passed")


if __name__ == "__main__":
    unittest.main()
