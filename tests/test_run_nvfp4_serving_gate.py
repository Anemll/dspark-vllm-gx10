# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "benchmarks" / "run_nvfp4_serving_gate.py"
SPEC = importlib.util.spec_from_file_location("run_nvfp4_serving_gate", PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


class ServingGateValidationTests(unittest.TestCase):
    @staticmethod
    def prefill_report(*, cache_hits: int = 0):
        results = []
        summary = []
        digest_counter = 0
        for concurrency in (1, 2):
            target = 1024
            requests = []
            for request in range(concurrency):
                digest_counter += 1
                requests.append(
                    {
                        "request": request,
                        "prompt_sha256": f"{digest_counter:064x}",
                        "usage_reported": True,
                        "prompt_tokens": target,
                        "completion_tokens": 1,
                        "finish_reason": "length",
                    }
                )
            results.append(
                {
                    "concurrency": concurrency,
                    "target_tokens": target,
                    "trial": 1,
                    "measurement_valid": cache_hits == 0,
                    "metrics_exact": True,
                    "cache_isolated": cache_hits == 0,
                    "prompt_lengths_exact": True,
                    "metrics_prompt_tokens_exact": True,
                    "server_computed_tokens_exact": True,
                    "completion_lengths_exact": True,
                    "server_cache_hit_tokens": cache_hits,
                    "metrics_request_delta": float(concurrency),
                    "prompt_tokens": target * concurrency,
                    "completion_tokens": concurrency,
                    "server_computed_tokens": target * concurrency,
                    "metrics_prompt_tokens_delta": target * concurrency,
                    "ttft_s": 1.0,
                    "elapsed_s": 1.1,
                    "client_input_tps": 1024.0,
                    "server_prefill_s": 1.0,
                    "server_prefill_tps": 1024.0,
                    "batch_ttft_s": 1.0,
                    "batch_wall_s": 1.1,
                    "aggregate_input_tps": 1024.0 * concurrency,
                    "mean_ttft_s": 1.0,
                    "p95_ttft_s": 1.0,
                    "requests": requests,
                }
            )
            summary.append(
                {
                    "concurrency": concurrency,
                    "target_tokens": target,
                    "row_valid": cache_hits == 0,
                    "trials": 1,
                    "valid_trials": 1 if cache_hits == 0 else 0,
                    "exact_server_trials": 1 if cache_hits == 0 else 0,
                    "cache_isolated_trials": 1 if cache_hits == 0 else 0,
                    "exact_length_trials": 1,
                    "exact_computed_token_trials": 1,
                    "median_ttft_s": 1.0,
                    "pooled_p95_ttft_s": 1.0,
                    "median_client_input_tps": 1024.0,
                    "median_aggregate_input_tps": 1024.0 * concurrency,
                    "median_server_prefill_tps": 1024.0,
                    "median_server_mean_request_prefill_tps": 1024.0,
                }
            )
        return {
            "schema_version": 2,
            "model": "m",
            "sizes": [1024],
            "concurrencies": [1, 2],
            "trials_per_size": 1,
            "seed": 4104,
            "results": results,
            "summary": summary,
        }

    def test_decode_report(self):
        rows = []
        for concurrency in (1, 2, 4):
            for trial in (1, 2, 3):
                streams = [
                    {
                        "completion_tokens": 512,
                        "finish_reason": "length",
                        "prompt_tokens": 10,
                        "chunks": 512,
                        "ttft_s": 0.1,
                        "decode_s": 1.0,
                        "elapsed_s": 1.1,
                        "token_tps": 512.0,
                        "chunk_tps": 512.0,
                    }
                    for _ in range(concurrency)
                ]
                rows.append(
                    {
                        "concurrency": concurrency,
                        "trial": trial,
                        "wall_s": 1.0,
                        "total_tokens": concurrency * 512,
                        "aggregate_token_tps": concurrency * 512.0,
                        "mean_ttft_s": 0.1,
                        "streams": streams,
                    }
                )
        summary = gate.validate_decode_report(
            {"model": "m", "max_tokens": 512, "trials": rows},
            model="m",
            concurrencies=[1, 2, 4],
            trials=3,
            max_tokens=512,
        )
        self.assertEqual([row["concurrency"] for row in summary], [1, 2, 4])

    def test_decode_rejects_short_stream(self):
        report = {
            "model": "m",
            "max_tokens": 512,
            "trials": [
                {
                    "concurrency": 1,
                    "trial": 1,
                    "wall_s": 1.0,
                    "total_tokens": 511,
                    "aggregate_token_tps": 511.0,
                    "mean_ttft_s": 0.1,
                    "streams": [],
                }
            ],
        }
        with self.assertRaises(RuntimeError):
            gate.validate_decode_report(
                report, model="m", concurrencies=[1], trials=1, max_tokens=512
            )

    def test_prefill_report(self):
        summary = gate.validate_prefill_report(
            self.prefill_report(),
            model="m",
            sizes=[1024],
            concurrencies=[1, 2],
            trials=1,
            seed=4104,
        )
        self.assertEqual(len(summary), 2)

    def test_prefill_rejects_cache_hits(self):
        with self.assertRaises(RuntimeError):
            gate.validate_prefill_report(
                self.prefill_report(cache_hits=1),
                model="m",
                sizes=[1024],
                concurrencies=[1, 2],
                trials=1,
                seed=4104,
            )


if __name__ == "__main__":
    unittest.main()
