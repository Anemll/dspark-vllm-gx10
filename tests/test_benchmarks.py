# SPDX-License-Identifier: MIT
"""Offline fixtures: no model, GPU, network, or optional packages required."""

from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import urllib.error

from benchmarks import benchmark_agent as agent
from benchmarks import benchmark_dsv4_api as benchmark


def delta(content=None, reason=None, tools=None, finish=None):
    body = {}
    if content is not None:
        body["content"] = content
    if reason is not None:
        body["reasoning_content"] = reason
    if tools is not None:
        body["tool_calls"] = tools
    return {"choices": [{"index": 0, "delta": body, "finish_reason": finish}]}


def usage(prompt=100, completion=6):
    return {"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "prompt_tokens_details": {"cached_tokens": 32}}}


def stream_result(events, **kwargs):
    payload = "".join("data:" + (event if isinstance(event, str) else json.dumps(event)) + "\n\n" for event in events)
    counter = iter(index / 10 for index in range(100))
    return benchmark.run_stream(
        "http://fixture.invalid", "fixture", 6, 0,
        opener=lambda request, timeout: io.BytesIO(payload.encode()),
        clock=lambda: next(counter), **kwargs,
    )


class StreamingTests(unittest.TestCase):
    def test_batched_tokens_are_not_counted_as_chunks(self):
        result = stream_result([delta("one two three"), delta(" four five six", finish="length"), usage(), "[DONE]"])
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.chunks, 2)
        self.assertEqual(result.completion_tokens, 6)
        self.assertEqual(len(result.events), 4)
        self.assertEqual(result.cached_prompt_tokens, 32)
        self.assertEqual(len(result.inter_output_chunk_s), 1)
        self.assertAlmostEqual(result.token_tps / result.chunk_tps, 3)

    def test_one_chunk_has_no_first_last_decode_rate(self):
        result = stream_result([delta("all six tokens in one chunk", finish="length"), usage(), "[DONE]"])
        self.assertTrue(result.ok, result.errors)
        self.assertIsNone(result.decode_s)
        self.assertIsNone(result.token_tps)
        self.assertIsNone(result.tpot_proxy_s)

    def test_missing_usage_is_failure_not_zero_tokens(self):
        result = stream_result([delta("output", finish="stop"), "[DONE]"])
        self.assertFalse(result.ok)
        self.assertIsNone(result.completion_tokens)
        self.assertTrue(any("usage.completion_tokens" in error for error in result.errors))

    def test_reasoning_only_is_not_a_text_answer(self):
        result = stream_result([delta(reason="thinking", finish="length"), usage(), "[DONE]"])
        self.assertFalse(result.ok)
        self.assertIsNotNone(result.ttft_s)
        self.assertIsNone(result.content_ttft_s)
        self.assertEqual(result.reasoning, "thinking")

    def test_reasoning_before_content_has_distinct_ttft(self):
        result = stream_result([delta(reason="thinking"), delta("answer", finish="stop"), usage(), "[DONE]"])
        self.assertTrue(result.ok, result.errors)
        self.assertLess(result.ttft_s, result.content_ttft_s)

    def test_fragmented_tool_arguments(self):
        calls = [
            delta(tools=[{"index": 0, "id": "call1", "type": "function", "function": {"name": "read_file", "arguments": '{"path":'}}]),
            delta(tools=[{"index": 0, "function": {"arguments": '"src/cache.py"}'}}], finish="tool_calls"),
            usage(), "[DONE]",
        ]
        result = stream_result(calls, expect="tool", expected_tool={"name": "read_file", "arguments": {"path": "src/cache.py"}})
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.chunks, 0)
        self.assertEqual(result.output_chunks, 2)
        wrong = stream_result(calls, expect="tool", expected_tool={"name": "read_file", "arguments": {"path": "wrong.py"}})
        self.assertFalse(wrong.ok)

    def test_missing_done_and_finish_are_failures(self):
        result = stream_result([delta("partial"), usage()])
        self.assertFalse(result.ok)
        self.assertTrue(any("[DONE]" in error for error in result.errors))
        self.assertTrue(any("finish_reason" in error for error in result.errors))

    def test_http_failure_retains_request(self):
        def fail(request, timeout):
            raise urllib.error.HTTPError(request.full_url, 503, "unavailable", {}, None)
        result = benchmark.run_stream("http://fixture.invalid", "fixture", 6, 7, opener=fail)
        self.assertFalse(result.ok)
        self.assertEqual(result.request, 7)
        self.assertEqual(result.request_body["model"], "fixture")
        self.assertIn("503", result.errors[0])

    def test_deadline_fails_even_when_events_keep_arriving(self):
        result = stream_result([delta("a"), delta("b", finish="stop"), usage(), "[DONE]"], timeout=0.25)
        self.assertFalse(result.ok)
        self.assertTrue(any("deadline" in error for error in result.errors))

    def test_malformed_json_is_recorded_as_failure(self):
        result = stream_result(["not JSON"])
        self.assertFalse(result.ok)
        self.assertIn("JSONDecodeError", result.errors[0])
        self.assertEqual(result.events[0]["raw_data"], "not JSON")

    def test_sse_comments_and_multiline_data(self):
        payload = b': heartbeat\r\ndata: {"choices": [],\r\ndata: "usage": null}\r\n\r\ndata:[DONE]\n\n'
        values = list(benchmark.sse_data(io.BytesIO(payload)))
        self.assertEqual(json.loads(values[0]), {"choices": [], "usage": None})
        self.assertEqual(values[1], "[DONE]")

    def test_small_samples_do_not_publish_tail_quantiles(self):
        self.assertIsNone(benchmark.distribution([1, 2])["p95"])
        self.assertEqual(benchmark.distribution(range(1, 21))["p95"], 19)
        self.assertIsNone(benchmark.distribution(range(20))["p99"])
        self.assertEqual(benchmark.distribution(range(1, 101))["p99"], 99)

    def test_failed_streams_are_excluded_from_successful_tokens(self):
        good = stream_result([delta("answer", finish="stop"), usage(), "[DONE]"])
        bad = stream_result([delta(reason="thinking", finish="length"), usage(), "[DONE]"])
        trial = {"concurrency": 2, "trial": 1, "streams": [asdict(good), asdict(bad)]}
        benchmark.finish_trial(trial, 2)
        self.assertEqual(trial["total_tokens"], 6)
        self.assertEqual(trial["aggregate_token_tps"], 3)
        self.assertEqual(benchmark.summarize_trials([trial])[0]["failed_requests"], 1)

    def test_cli_writes_partial_failure_and_stops(self):
        failure = stream_result([delta("partial")])
        with tempfile.TemporaryDirectory() as directory:
            target = str(Path(directory) / "result.json")
            with patch("sys.argv", ["benchmark", "--concurrency", "1", "--trials", "3", "--output", target]), patch.object(benchmark, "run_stream", return_value=failure), patch("builtins.print"):
                with self.assertRaises(SystemExit) as exit_context:
                    benchmark.main()
            report = json.loads(Path(target).read_text())
            self.assertEqual(exit_context.exception.code, 1)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(len(report["trials"]), 1)
            self.assertEqual(report["trials"][0]["streams"][0]["content"], "partial")


class AgentFixtureTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads(agent.FIXTURE_PATH.read_text())

    @staticmethod
    def tokenizer(body):
        # Deterministic fake tokenizer includes message framing and content.
        text = json.dumps(body["messages"], sort_keys=True)
        tokens = [ord(character) for character in text]
        return {"count": len(tokens), "tokens": tokens}

    def test_exact_length_and_prefix_uniqueness(self):
        a, evidence_a = agent.fit_coding_body("fixture", 2048, 32, 4104, "1:1:0", self.fixture, self.tokenizer)
        repeated, evidence_repeat = agent.fit_coding_body("fixture", 2048, 32, 4104, "1:1:0", self.fixture, self.tokenizer)
        b, evidence_b = agent.fit_coding_body("fixture", 2048, 32, 4104, "1:2:0", self.fixture, self.tokenizer)
        self.assertEqual(evidence_a["count"], 2048)
        self.assertEqual(a, repeated)
        self.assertEqual(evidence_a, evidence_repeat)
        self.assertNotEqual(a["messages"][1]["content"][:40], b["messages"][1]["content"][:40])
        self.assertNotEqual(evidence_a["token_ids_sha256"], evidence_b["token_ids_sha256"])

    def test_exact_fit_fails_instead_of_claiming_approximate_count(self):
        def impossible(body):
            return {"count": 10, "tokens": [1] * 10}
        with self.assertRaisesRegex(ValueError, "outside"):
            agent.fit_coding_body("fixture", 20, 32, 1, "case", self.fixture, impossible)

    def test_count_and_token_ids_must_agree(self):
        with self.assertRaisesRegex(ValueError, "disagrees"):
            agent.token_evidence({"model": "fixture", "messages": []}, lambda body: {"count": 2, "tokens": [1]})

    def test_fixed_history_repeats_and_diverges(self):
        base = benchmark.make_body("fixture", 32)
        turns = agent.conversation_bodies(base, "multi-turn", self.fixture)
        self.assertEqual(turns[0][2], turns[1][2])
        self.assertEqual(turns[2][2]["messages"][:-1], turns[3][2]["messages"][:-1])
        self.assertNotEqual(turns[2][2]["messages"][-1], turns[3][2]["messages"][-1])
        self.assertEqual(len(base["messages"]), 1)

    def test_usage_mismatch_stops_conversation_and_keeps_result(self):
        result = stream_result([delta("answer", finish="stop"), usage(prompt=99), "[DONE]"])
        case = {"conversation": 0, "turn": 0, "phase": "initial", "cache_intent": "unique-prefix-intended", "body": benchmark.make_body("fixture", 32), "tokenization": {"count": 100}, "expect": "text", "expected_tool": None}
        args = types.SimpleNamespace(base_url="http://fixture.invalid", model="fixture", max_tokens=32, timeout=30)
        saved = []
        outcome = agent.run_conversation([case, case], args, saved.append, runner=lambda *args, **kwargs: result)
        self.assertFalse(outcome)
        self.assertEqual(len(saved), 1)
        self.assertIn("inference prompt usage", saved[0]["errors"][-1])

    def test_agent_cli_and_exact_replay_with_fake_http(self):
        def fake_json(base, endpoint, body=None, timeout=30):
            if endpoint == "/tokenize":
                return self.tokenizer(body)
            return {"fixture": endpoint}

        def fake_stream(*args, **kwargs):
            body = kwargs["body"]
            prompt_count = self.tokenizer(body)["count"]
            result = stream_result([delta("fixed answer", finish="length"), usage(prompt=prompt_count), "[DONE]"])
            result.request_body = body
            result.request_sha256 = benchmark.json_hash(body)
            return result

        with tempfile.TemporaryDirectory() as directory:
            first = str(Path(directory) / "control.json")
            second = str(Path(directory) / "candidate.json")
            commands = [
                ["agent", "--mode", "multi-turn", "--concurrency", "1", "--trials", "1", "--target-tokens", "2048", "--max-tokens", "6", "--output", first],
                ["agent", "--replay-report", first, "--max-tokens", "6", "--output", second],
            ]
            for command in commands:
                with patch("sys.argv", command), patch.object(agent, "request_json", side_effect=fake_json), patch.object(agent, "run_stream", side_effect=fake_stream), patch("builtins.print"):
                    agent.main()
            control = json.loads(Path(first).read_text())
            candidate = json.loads(Path(second).read_text())
            self.assertEqual(control["status"], "complete")
            self.assertEqual(candidate["status"], "complete")
            self.assertEqual(control["manifest"]["cases_sha256"], candidate["manifest"]["cases_sha256"])
            self.assertEqual(len(control["trials"][0]["streams"]), 4)
            self.assertEqual(len(control["phase_summary"]), 4)


if __name__ == "__main__":
    unittest.main()
