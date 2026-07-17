# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = REPO_ROOT / "benchmarks" / "benchmark_prefill.py"
COMPARE_PATH = REPO_ROOT / "benchmarks" / "compare_prefill.py"

SPEC = importlib.util.spec_from_file_location("benchmark_prefill", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)

COMPARE_SPEC = importlib.util.spec_from_file_location("compare_prefill", COMPARE_PATH)
assert COMPARE_SPEC is not None and COMPARE_SPEC.loader is not None
compare = importlib.util.module_from_spec(COMPARE_SPEC)
sys.modules[COMPARE_SPEC.name] = compare
COMPARE_SPEC.loader.exec_module(compare)


def snapshot(
    *,
    prefill_time_s: float = 0.0,
    prefill_requests: float = 0.0,
    computed_tokens: float = 0.0,
    cache_hit_tokens: float = 0.0,
    prompt_tokens: float = 0.0,
):
    return benchmark.MetricSnapshot(
        prefill_time_s=prefill_time_s,
        prefill_requests=prefill_requests,
        computed_tokens=computed_tokens,
        cache_hit_tokens=cache_hit_tokens,
        prompt_tokens=prompt_tokens,
    )


class PrefillPromptTests(unittest.TestCase):
    def test_concurrency_one_preserves_original_prompt_stream(self) -> None:
        pool = [11, 13, 17, 19]
        expected_rng = random.Random("dspark-prefill:4104:32:3")
        expected = [pool[expected_rng.randrange(len(pool))] for _ in range(32)]
        self.assertEqual(
            benchmark.make_prompt(pool, 32, 3, 4104),
            expected,
        )

    def test_prompt_batches_are_unique_across_shape_phase_and_concurrency(self) -> None:
        seen: set[str] = set()
        seen_prefixes: set[str] = set()
        all_hashes: list[str] = []
        pool = list(range(32))
        for concurrency in (1, 2, 4):
            for size in (16, 32):
                for trial in (0, -1, 1, 2):
                    prompts, hashes = benchmark.make_prompt_batch(
                        pool,
                        size,
                        trial,
                        4104,
                        concurrency,
                        seen,
                        seen_prefixes,
                    )
                    self.assertEqual(len(prompts), concurrency)
                    self.assertTrue(all(len(prompt) == size for prompt in prompts))
                    self.assertEqual(
                        hashes,
                        [benchmark.prompt_sha256(prompt) for prompt in prompts],
                    )
                    all_hashes.extend(hashes)
        self.assertEqual(len(all_hashes), len(set(all_hashes)))
        self.assertEqual(len(all_hashes), len(seen_prefixes))

    def test_duplicate_prompt_batch_fails_closed(self) -> None:
        seen: set[str] = set()
        benchmark.make_prompt_batch([1, 2, 3], 16, 1, 4104, 2, seen)
        with self.assertRaisesRegex(RuntimeError, "prefix-cache reuse"):
            benchmark.make_prompt_batch([1, 2, 3], 16, 1, 4104, 2, seen)

    def test_duplicate_first_cache_block_fails_closed(self) -> None:
        seen: set[str] = set()
        seen_prefixes: set[str] = set()
        prompts = [
            [1] * benchmark.PREFIX_GUARD_TOKENS + [2],
            [1] * benchmark.PREFIX_GUARD_TOKENS + [3],
        ]
        with mock.patch.object(benchmark, "make_prompt", side_effect=prompts):
            benchmark.make_prompt_batch(
                [1, 2, 3], 17, 1, 4104, 1, seen, seen_prefixes
            )
            with self.assertRaisesRegex(RuntimeError, "first cache block"):
                benchmark.make_prompt_batch(
                    [1, 2, 3], 17, 2, 4104, 1, seen, seen_prefixes
                )


class PrefillBatchTests(unittest.TestCase):
    def test_thread_batch_reaches_requested_concurrency(self) -> None:
        active = 0
        max_active = 0
        lock = threading.Lock()

        def fake_run(
            _base_url,
            _model,
            prompt,
            _timeout,
            *,
            start_barrier=None,
            start_barrier_timeout_s=30.0,
        ):
            nonlocal active, max_active
            if start_barrier is not None:
                start_barrier.wait(timeout=start_barrier_timeout_s)
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            now = time.perf_counter()
            return benchmark.CompletionTiming(
                started_s=now - 0.02,
                first_event_s=now - 0.01,
                finished_s=now,
                usage={"prompt_tokens": len(prompt), "completion_tokens": 1},
                finish_reason="length",
            )

        with mock.patch.object(benchmark, "run_completion", side_effect=fake_run):
            timings = benchmark.run_completion_batch(
                "http://unused", "model", [[index] * 8 for index in range(4)], 1
            )
        self.assertEqual(len(timings), 4)
        self.assertEqual(max_active, 4)

    def test_start_barrier_is_bounded_when_a_worker_fails_early(self) -> None:
        def fake_run(
            _base_url,
            _model,
            prompt,
            _timeout,
            *,
            start_barrier=None,
            start_barrier_timeout_s=30.0,
        ):
            if prompt[0] == 0:
                raise RuntimeError("failed before barrier")
            assert start_barrier is not None
            start_barrier.wait(timeout=start_barrier_timeout_s)
            raise AssertionError("broken barrier should not complete")

        started = time.perf_counter()
        with (
            mock.patch.object(benchmark, "run_completion", side_effect=fake_run),
            self.assertRaisesRegex(RuntimeError, "start barrier failed"),
        ):
            benchmark.run_completion_batch(
                "http://unused",
                "model",
                [[0], [1]],
                1,
                start_barrier_timeout_s=0.05,
            )
        self.assertLess(time.perf_counter() - started, 0.5)

    def test_concurrency_one_keeps_original_scalar_semantics(self) -> None:
        timing = benchmark.CompletionTiming(
            started_s=10.0,
            first_event_s=12.0,
            finished_s=12.1,
            usage={"prompt_tokens": 1024, "completion_tokens": 1},
            finish_reason="length",
        )
        result = benchmark.build_prefill_result(
            target_tokens=1024,
            concurrency=1,
            trial=1,
            prompt_hashes=["a" * 64],
            timings=[timing],
            before=snapshot(),
            after=snapshot(
                prefill_time_s=0.5,
                prefill_requests=1,
                computed_tokens=1024,
                prompt_tokens=1024,
            ),
        )
        self.assertEqual(result.prompt_sha256, "a" * 64)
        self.assertEqual(result.prompt_tokens, 1024)
        self.assertEqual(result.completion_tokens, 1)
        self.assertAlmostEqual(result.ttft_s, 2.0)
        self.assertAlmostEqual(result.elapsed_s, 2.1)
        self.assertAlmostEqual(result.client_input_tps, 512.0)
        self.assertAlmostEqual(result.aggregate_input_tps, 512.0)
        self.assertAlmostEqual(result.server_prefill_tps, 2048.0)
        self.assertAlmostEqual(result.server_mean_request_prefill_tps, 2048.0)
        self.assertTrue(result.metrics_exact)
        self.assertTrue(result.cache_isolated)
        self.assertTrue(result.measurement_valid)
        self.assertEqual(len(result.requests or []), 1)

    def test_concurrent_result_has_aggregate_and_per_request_metrics(self) -> None:
        timings = [
            benchmark.CompletionTiming(
                started_s=10.0 + request * 0.01,
                first_event_s=12.0 + request * 0.1,
                finished_s=12.1 + request * 0.1,
                usage={"prompt_tokens": 100, "completion_tokens": 1},
                finish_reason="length",
            )
            for request in range(4)
        ]
        result = benchmark.build_prefill_result(
            target_tokens=100,
            concurrency=4,
            trial=2,
            prompt_hashes=[f"{request:064x}" for request in range(4)],
            timings=timings,
            before=snapshot(),
            after=snapshot(
                prefill_time_s=0.8,
                prefill_requests=4,
                computed_tokens=400,
                prompt_tokens=400,
            ),
        )
        self.assertEqual(result.concurrency, 4)
        self.assertEqual(result.prompt_tokens, 400)
        self.assertEqual(result.completion_tokens, 4)
        self.assertAlmostEqual(result.batch_ttft_s, 2.3)
        self.assertAlmostEqual(result.batch_wall_s, 2.4)
        self.assertAlmostEqual(result.aggregate_input_tps, 400 / 2.3)
        self.assertAlmostEqual(result.server_prefill_tps, 500.0)
        self.assertAlmostEqual(result.server_mean_request_prefill_tps, 500.0)
        self.assertTrue(result.metrics_exact)
        self.assertTrue(result.cache_isolated)
        self.assertTrue(result.measurement_valid)
        self.assertEqual(len(result.requests or []), 4)
        self.assertEqual(
            len({request.prompt_sha256 for request in result.requests or []}),
            4,
        )
        self.assertTrue(
            all(request.prompt_tokens == 100 for request in result.requests or [])
        )

    def test_overlap_or_cache_hit_invalidates_server_throughput(self) -> None:
        timing = benchmark.CompletionTiming(
            started_s=1.0,
            first_event_s=2.0,
            finished_s=2.1,
            usage={"prompt_tokens": 8, "completion_tokens": 1},
            finish_reason="length",
        )
        overlap = benchmark.build_prefill_result(
            target_tokens=8,
            concurrency=1,
            trial=1,
            prompt_hashes=["a" * 64],
            timings=[timing],
            before=snapshot(),
            after=snapshot(
                prefill_time_s=1,
                prefill_requests=2,
                computed_tokens=16,
            ),
        )
        self.assertFalse(overlap.metrics_exact)
        self.assertIsNone(overlap.server_prefill_tps)

        cache_hit = benchmark.build_prefill_result(
            target_tokens=8,
            concurrency=1,
            trial=1,
            prompt_hashes=["b" * 64],
            timings=[timing],
            before=snapshot(),
            after=snapshot(
                prefill_time_s=1,
                prefill_requests=1,
                computed_tokens=4,
                cache_hit_tokens=4,
            ),
        )
        self.assertTrue(cache_hit.metrics_exact)
        self.assertFalse(cache_hit.cache_isolated)
        self.assertIsNone(cache_hit.server_prefill_tps)

    def test_usage_or_computed_length_mismatch_invalidates_trial(self) -> None:
        timing = benchmark.CompletionTiming(
            started_s=1.0,
            first_event_s=2.0,
            finished_s=2.1,
            usage={"prompt_tokens": 7, "completion_tokens": 1},
            finish_reason="length",
        )
        result = benchmark.build_prefill_result(
            target_tokens=8,
            concurrency=1,
            trial=1,
            prompt_hashes=["a" * 64],
            timings=[timing],
            before=snapshot(),
            after=snapshot(
                prefill_time_s=1,
                prefill_requests=1,
                computed_tokens=7,
                prompt_tokens=8,
            ),
        )
        self.assertFalse(result.prompt_lengths_exact)
        self.assertTrue(result.metrics_prompt_tokens_exact)
        self.assertFalse(result.server_computed_tokens_exact)
        self.assertFalse(result.measurement_valid)
        self.assertIsNone(result.server_prefill_tps)

    def test_missing_usage_or_computed_mismatch_fails_closed(self) -> None:
        missing_usage = benchmark.CompletionTiming(
            started_s=1.0,
            first_event_s=2.0,
            finished_s=2.1,
            usage={},
            finish_reason="length",
        )
        result = benchmark.build_prefill_result(
            target_tokens=8,
            concurrency=1,
            trial=1,
            prompt_hashes=["a" * 64],
            timings=[missing_usage],
            before=snapshot(),
            after=snapshot(
                prefill_time_s=1,
                prefill_requests=1,
                computed_tokens=8,
                prompt_tokens=8,
            ),
        )
        self.assertFalse(result.prompt_lengths_exact)
        self.assertFalse(result.completion_lengths_exact)
        self.assertFalse(result.measurement_valid)
        self.assertIsNone(result.server_prefill_tps)

        exact_usage = benchmark.CompletionTiming(
            started_s=1.0,
            first_event_s=2.0,
            finished_s=2.1,
            usage={"prompt_tokens": 8, "completion_tokens": 1},
            finish_reason="length",
        )
        result = benchmark.build_prefill_result(
            target_tokens=8,
            concurrency=1,
            trial=1,
            prompt_hashes=["b" * 64],
            timings=[exact_usage],
            before=snapshot(),
            after=snapshot(
                prefill_time_s=1,
                prefill_requests=1,
                computed_tokens=7,
                prompt_tokens=8,
            ),
        )
        self.assertFalse(result.server_computed_tokens_exact)
        self.assertFalse(result.measurement_valid)
        self.assertIsNone(result.server_prefill_tps)

    def test_summary_separates_concurrency_shapes(self) -> None:
        def make_result(concurrency: int, aggregate: float):
            return benchmark.PrefillResult(
                target_tokens=1024,
                trial=1,
                prompt_sha256="a" * 64,
                prompt_tokens=1024 * concurrency,
                completion_tokens=concurrency,
                ttft_s=1.0,
                elapsed_s=1.0,
                client_input_tps=aggregate,
                server_prefill_s=1.0,
                server_computed_tokens=1024 * concurrency,
                server_cache_hit_tokens=0,
                server_prefill_tps=aggregate,
                metrics_request_delta=float(concurrency),
                metrics_exact=True,
                finish_reason="length",
                concurrency=concurrency,
                aggregate_input_tps=aggregate,
                mean_ttft_s=1.0,
                p95_ttft_s=1.0,
            )

        summary = benchmark.build_summary(
            [make_result(1, 1000.0), make_result(4, 3000.0)]
        )
        self.assertEqual(
            [
                (row["concurrency"], row["median_aggregate_input_tps"])
                for row in summary
            ],
            [(1, 1000.0), (4, 3000.0)],
        )

    def test_one_invalid_trial_invalidates_entire_summary_row(self) -> None:
        valid = benchmark.PrefillResult(
            target_tokens=8,
            trial=1,
            prompt_sha256="a" * 64,
            prompt_tokens=8,
            completion_tokens=1,
            ttft_s=1.0,
            elapsed_s=1.1,
            client_input_tps=8.0,
            server_prefill_s=1.0,
            server_computed_tokens=8,
            server_cache_hit_tokens=0,
            server_prefill_tps=8.0,
            metrics_request_delta=1.0,
            metrics_exact=True,
            finish_reason="length",
            aggregate_input_tps=8.0,
            measurement_valid=True,
        )
        invalid = benchmark.PrefillResult(
            target_tokens=8,
            trial=2,
            prompt_sha256="b" * 64,
            prompt_tokens=8,
            completion_tokens=1,
            ttft_s=0.1,
            elapsed_s=0.2,
            client_input_tps=80.0,
            server_prefill_s=1.0,
            server_computed_tokens=8,
            server_cache_hit_tokens=0,
            server_prefill_tps=None,
            metrics_request_delta=2.0,
            metrics_exact=False,
            finish_reason="length",
            aggregate_input_tps=80.0,
            measurement_valid=False,
        )
        row = benchmark.build_summary([valid, invalid])[0]
        self.assertFalse(row["row_valid"])
        self.assertEqual(row["valid_trials"], 1)
        self.assertIsNone(row["median_aggregate_input_tps"])
        self.assertIsNone(row["median_ttft_s"])
        self.assertIsNone(row["median_server_prefill_tps"])


class PrefillMainTests(unittest.TestCase):
    def test_default_concurrency_one_keeps_schema_and_prompt_hash(self) -> None:
        timing = benchmark.CompletionTiming(
            started_s=1.0,
            first_event_s=2.0,
            finished_s=2.1,
            usage={"prompt_tokens": 8, "completion_tokens": 1},
            finish_reason="length",
        )
        pool = list(range(32))
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "prefill.json"
            argv = [
                "benchmark_prefill.py",
                "--sizes",
                "8",
                "--trials",
                "1",
                "--warmup-tokens",
                "0",
                "--shape-warmup-trials",
                "0",
                "--output",
                str(output),
            ]
            with (
                mock.patch.object(
                    benchmark,
                    "request_json",
                    return_value={"version": "test"},
                ),
                mock.patch.object(benchmark, "tokenize", return_value=pool),
                mock.patch.object(
                    benchmark,
                    "snapshot_metrics",
                    side_effect=[
                        snapshot(),
                        snapshot(
                            prefill_time_s=1.0,
                            prefill_requests=1.0,
                            computed_tokens=8.0,
                            prompt_tokens=8.0,
                        ),
                    ],
                ),
                mock.patch.object(
                    benchmark,
                    "run_completion_batch",
                    return_value=[timing],
                ),
                mock.patch.object(sys, "argv", argv),
                redirect_stdout(io.StringIO()),
            ):
                benchmark.main()
            report = json.loads(output.read_text())

        expected_prompt = benchmark.make_prompt(pool, 8, 1, 4104)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["concurrencies"], [1])
        self.assertEqual(
            report["results"][0]["prompt_sha256"],
            benchmark.prompt_sha256(expected_prompt),
        )

    def test_mocked_matrix_warms_and_measures_every_concurrency(self) -> None:
        state = {
            "prefill_time_s": 0.0,
            "prefill_requests": 0.0,
            "computed_tokens": 0.0,
            "cache_hit_tokens": 0.0,
            "prompt_tokens": 0.0,
            "clock": 100.0,
        }
        batches: list[tuple[int, int, tuple[str, ...]]] = []

        def fake_snapshot(_base_url):
            return snapshot(
                prefill_time_s=state["prefill_time_s"],
                prefill_requests=state["prefill_requests"],
                computed_tokens=state["computed_tokens"],
                cache_hit_tokens=state["cache_hit_tokens"],
                prompt_tokens=state["prompt_tokens"],
            )

        def fake_batch(_base_url, _model, prompts, _timeout):
            prompt_hashes = tuple(benchmark.prompt_sha256(prompt) for prompt in prompts)
            batches.append((len(prompts), len(prompts[0]), prompt_hashes))
            base = state["clock"]
            timings = []
            for request, prompt in enumerate(prompts):
                started = base + request * 0.001
                first = started + len(prompt) / 1000
                timings.append(
                    benchmark.CompletionTiming(
                        started_s=started,
                        first_event_s=first,
                        finished_s=first + 0.001,
                        usage={
                            "prompt_tokens": len(prompt),
                            "completion_tokens": 1,
                        },
                        finish_reason="length",
                    )
                )
            token_count = sum(len(prompt) for prompt in prompts)
            state["prefill_time_s"] += token_count / 1000
            state["prefill_requests"] += len(prompts)
            state["computed_tokens"] += token_count
            state["prompt_tokens"] += token_count
            state["clock"] += 1
            return timings

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "prefill.json"
            argv = [
                "benchmark_prefill.py",
                "--sizes",
                "8,16",
                "--concurrency",
                "1,2,4",
                "--trials",
                "1",
                "--warmup-tokens",
                "4",
                "--shape-warmup-trials",
                "1",
                "--output",
                str(output),
            ]
            with (
                mock.patch.object(
                    benchmark,
                    "request_json",
                    return_value={"version": "test"},
                ),
                mock.patch.object(
                    benchmark,
                    "tokenize",
                    return_value=list(range(32)),
                ),
                mock.patch.object(
                    benchmark,
                    "snapshot_metrics",
                    side_effect=fake_snapshot,
                ),
                mock.patch.object(
                    benchmark,
                    "run_completion_batch",
                    side_effect=fake_batch,
                ),
                mock.patch.object(sys, "argv", argv),
                redirect_stdout(io.StringIO()),
            ):
                benchmark.main()

            report = json.loads(output.read_text())

        # Per concurrency: one general warmup, two shape warmups, two measured.
        self.assertEqual([batch[0] for batch in batches], [1] * 5 + [2] * 5 + [4] * 5)
        all_hashes = [digest for _c, _size, hashes in batches for digest in hashes]
        self.assertEqual(len(all_hashes), len(set(all_hashes)))
        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["concurrencies"], [1, 2, 4])
        self.assertEqual(len(report["results"]), 6)
        self.assertEqual(len(report["summary"]), 6)
        self.assertTrue(all(result["metrics_exact"] for result in report["results"]))
        self.assertTrue(all(result["cache_isolated"] for result in report["results"]))
        self.assertTrue(
            all(result["measurement_valid"] for result in report["results"])
        )
        self.assertTrue(all(row["row_valid"] for row in report["summary"]))
        self.assertTrue(
            all(row["pooled_p95_ttft_s"] is not None for row in report["summary"])
        )
        self.assertEqual(
            [len(result["requests"]) for result in report["results"]],
            [1, 1, 2, 2, 4, 4],
        )


class PrefillComparisonTests(unittest.TestCase):
    def test_multi_concurrency_comparison_keeps_shapes_and_hashes(self) -> None:
        def report(label: str, multiplier: float):
            summaries = []
            results = []
            for concurrency in (1, 2, 4):
                summaries.append(
                    {
                        "target_tokens": 1024,
                        "concurrency": concurrency,
                        "median_server_prefill_tps": 1000 * multiplier,
                        "median_aggregate_input_tps": 900 * concurrency * multiplier,
                        "median_client_input_tps": 900 * concurrency * multiplier,
                        "median_ttft_s": 1 / multiplier,
                        "pooled_p95_ttft_s": 1.1 / multiplier,
                        "row_valid": True,
                    }
                )
                results.append(
                    {
                        "target_tokens": 1024,
                        "concurrency": concurrency,
                        "trial": 1,
                        "prompt_sha256": f"batch-{concurrency}",
                        "requests": [
                            {"prompt_sha256": f"prompt-{concurrency}-{request}"}
                            for request in range(concurrency)
                        ],
                    }
                )
            return {
                "label": label,
                "version": "test",
                "trials_per_size": 1,
                "seed": 4104,
                "summary": summaries,
                "results": results,
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            before_path = Path(temp_dir) / "before.json"
            after_path = Path(temp_dir) / "after.json"
            before_path.write_text(json.dumps(report("before", 1.0)))
            after_path.write_text(json.dumps(report("after", 1.1)))
            stdout = io.StringIO()
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    ["compare_prefill.py", str(before_path), str(after_path)],
                ),
                redirect_stdout(stdout),
            ):
                compare.main()

        rendered = stdout.getvalue()
        self.assertIn("| Concurrency | Input tokens |", rendered)
        self.assertIn("| 1 | 1,024 |", rendered)
        self.assertIn("| 2 | 1,024 |", rendered)
        self.assertIn("| 4 | 1,024 |", rendered)
        self.assertIn("+10.0%", rendered)
        self.assertNotIn("Before server tok/s", rendered)

    def test_multi_comparison_marks_invalid_rows(self) -> None:
        def report(label: str, valid: bool):
            return {
                "label": label,
                "version": "test",
                "trials_per_size": 1,
                "seed": 4104,
                "summary": [
                    {
                        "target_tokens": 8,
                        "concurrency": 2,
                        "row_valid": valid,
                        "median_aggregate_input_tps": 16.0,
                        "median_ttft_s": 1.0,
                        "pooled_p95_ttft_s": 1.1,
                    }
                ],
                "results": [
                    {
                        "target_tokens": 8,
                        "concurrency": 2,
                        "trial": 1,
                        "prompt_sha256": "batch",
                        "requests": [
                            {"prompt_sha256": "request-a"},
                            {"prompt_sha256": "request-b"},
                        ],
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            before_path = Path(temp_dir) / "before.json"
            after_path = Path(temp_dir) / "after.json"
            before_path.write_text(json.dumps(report("before", True)))
            after_path.write_text(json.dumps(report("after", False)))
            stdout = io.StringIO()
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    ["compare_prefill.py", str(before_path), str(after_path)],
                ),
                redirect_stdout(stdout),
            ):
                compare.main()
        rendered = stdout.getvalue()
        self.assertIn("| 2 | 8 | no | n/a | n/a | n/a |", rendered)

    def test_fingerprints_include_every_concurrent_request(self) -> None:
        report = {
            "results": [
                {
                    "target_tokens": 8,
                    "concurrency": 2,
                    "trial": 1,
                    "prompt_sha256": "batch",
                    "requests": [
                        {"prompt_sha256": "request-a"},
                        {"prompt_sha256": "request-b"},
                    ],
                }
            ]
        }
        self.assertEqual(
            compare.request_prompt_fingerprints(report),
            {(2, 8, 1): ("request-a", "request-b")},
        )


if __name__ == "__main__":
    unittest.main()
