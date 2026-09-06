# SPDX-License-Identifier: MIT
import io
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.benchmark_content import make_body, metric_delta, METRICS, summarize, validate_result
from benchmarks.streaming_client import run_stream, StreamResult


class Response(io.BytesIO):
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()


class ContentTests(unittest.TestCase):
    def test_upstream_prompt_unchanged(self):
        case = {"id": "upstream-explanation", "prompt": "Original prompt"}
        body, images = make_body(case, "m", 512, 42, Path("."), "measured", 1, 0)
        self.assertEqual(body["messages"][0]["content"], "Original prompt")
        self.assertFalse(body["chat_template_kwargs"]["thinking"])
        self.assertTrue(body["ignore_eos"])
        self.assertEqual(images, [])

    def test_content_identity_reproducible_and_unique(self):
        case = {"id": "code", "prompt": "Write code"}
        def build(phase, trial, request):
            return make_body(case, "m", 512, 42, Path("."), phase, trial, request)[0]
        self.assertEqual(build("measured", 1, 0), build("measured", 1, 0))
        self.assertNotEqual(build("measured", 1, 0), build("warmup", 1, 0))
        self.assertNotEqual(build("measured", 1, 0), build("measured", 1, 1))

    def test_metric_isolation(self):
        before = dict.fromkeys(METRICS, 0.0)
        after = before | {"requests": 2, "output_tokens": 1024, "decode_seconds": 40,
                          "drafts": 100, "draft_tokens": 500, "accepted_tokens": 250,
                          "prefill_seconds": 1, "computed_tokens": 100}
        result = metric_delta(before, after, 2, 1024)
        self.assertTrue(result["exact"])
        self.assertAlmostEqual(result["server_decode_tok_s"], 1022 / 40)
        self.assertEqual(result["draft_acceptance"], 0.5)
        self.assertFalse(metric_delta(before, after | {"requests": 3}, 2, 1024)["exact"])
        self.assertFalse(metric_delta(before | {"running": 1}, after, 2, 1024)["exact"])
        self.assertIsNone(metric_delta(before, after | {"decode_seconds": 0}, 2, 1024)["server_decode_tok_s"])

    def test_missing_metrics_are_unavailable(self):
        missing = dict.fromkeys(METRICS)
        result = metric_delta(missing, missing, 1, 128)
        self.assertFalse(result["exact"])
        self.assertIsNone(result["server_decode_tok_s"])

    def test_short_generation_fails_fixed_length_gate(self):
        result = StreamResult(0, completion_tokens=20, finish_reason="stop", ok=True)
        self.assertFalse(validate_result(result, {}, 512).ok)

    def test_semantic_json_gate(self):
        case = {"expected_json": {"answer": 42}}
        good = StreamResult(0, content='{"answer":42}', finish_reason="stop", ok=True)
        bad = StreamResult(0, content='{"answer":24}', finish_reason="stop", ok=True)
        self.assertTrue(validate_result(good, case, 128).ok)
        self.assertFalse(validate_result(bad, case, 128).ok)
        self.assertEqual(bad.errors, ["answer value mismatch at $.answer"])

    def test_json_types_and_values_have_distinct_errors(self):
        case = {"expected_json": {"code": "Q7M4", "units": 37, "destination": "Oslo"}}
        result = StreamResult(0, content='{"code":"Q7M4","units":"37","destination":"Oslo"}', finish_reason="stop")
        self.assertFalse(validate_result(result, case, 192).ok)
        self.assertEqual(result.errors, ["JSON type mismatch at $.units: expected integer, got string"])
        for value in ("true", "37.5"):
            result = StreamResult(0, content='{"units":' + value + '}', finish_reason="stop")
            self.assertFalse(validate_result(result, {"expected_json": {"units": 37}}, 192).ok)
            self.assertIn("JSON type mismatch", result.errors[0])
        result = StreamResult(0, content='{"units":37.0}', finish_reason="stop")
        self.assertTrue(validate_result(result, {"expected_json": {"units": 37}}, 192).ok)

    def test_strict_json_and_finish_reason(self):
        for content in ('{"answer":0,"answer":42}', '{"answer":NaN}', '{"answer":Infinity}', '```json\n{"answer":42}\n```'):
            result = StreamResult(0, content=content, finish_reason="stop")
            self.assertFalse(validate_result(result, {"expected_json": {"answer": 42}}, 128).ok)
            self.assertIn("not valid unwrapped JSON", result.errors[0])
        result = StreamResult(0, content='{"answer":42}', finish_reason="length")
        self.assertFalse(validate_result(result, {"expected_json": {"answer": 42}}, 128).ok)
        self.assertIn("did not finish normally", result.errors[0])

    def test_json_structure_and_nested_values(self):
        case = {"expected_json": {"rows": [{"number": 1}]}}
        for content, error in (
            ('{"rows":[{"number":true}]}', "JSON type mismatch at $.rows[0].number: expected integer, got boolean"),
            ('{"rows":[]}', "JSON array length mismatch at $.rows"),
            ('{"rows":[{"number":2}]}', "answer value mismatch at $.rows[0].number"),
            ('{"rows":[{"number":1}],"extra":0}', "JSON keys mismatch at $"),
        ):
            result = StreamResult(0, content=content, finish_reason="stop")
            self.assertFalse(validate_result(result, case, 128).ok)
            self.assertEqual(result.errors, [error])

    def test_partial_wave_survives_summary(self):
        self.assertEqual(summarize([{"phase": "measured", "case": "a", "concurrency": 1, "streams": []}]), [])

    def test_sse_metadata_is_not_first_token_and_missing_done_fails(self):
        events = [
            {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
            {"choices": [{"index": 0, "delta": {"content": "Hello"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}], "usage": {"prompt_tokens": 10, "completion_tokens": 1}},
        ]
        payload = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        result = run_stream("http://invalid", "m", 1, 0, timeout=5, opener=lambda *a, **k: Response(payload.encode()))
        self.assertFalse(result.ok)
        self.assertEqual(result.output_chunks, 1)
        self.assertIsNone(result.token_tps)
        self.assertIn("stream ended without [DONE]", result.errors)
        result = run_stream("http://invalid", "m", 1, 0, timeout=5, opener=lambda *a, **k: Response((payload + "data: [DONE]\n\n").encode()))
        self.assertTrue(result.ok)

    def test_local_image_hash_and_data_url(self):
        with tempfile.TemporaryDirectory() as directory:
            # Fixture bytes need not decode here: this test verifies transport,
            # while the visual fixture builder and serving canary verify pixels.
            Path(directory, "a.png").write_bytes(b"test")
            body, images = make_body({"id": "image", "prompt": "Read", "images": ["a.png"], "expected_json": {}}, "m", 128, 1, Path(directory), "measured", 1, 0)
            self.assertFalse(body["ignore_eos"])
            self.assertEqual(images[0]["bytes"], 4)
            self.assertTrue(body["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
