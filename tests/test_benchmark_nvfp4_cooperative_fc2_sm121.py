# SPDX-License-Identifier: MIT

from __future__ import annotations

import contextlib
import io
import unittest

from benchmarks import benchmark_nvfp4_cooperative_fc2_sm121 as probe


def _report(variant: str, medians: dict[tuple[str, int], float]) -> dict:
    return {
        "probe": probe.PROBE_NAME,
        "variant": variant,
        "ok": True,
        "gpu": {"capability": [12, 1], "torch": "2.11.0+cu130"},
        "checkpoint": {
            "tp_rank": 0,
            "layer_file_sha256": "c" * 64,
            "physical_validation": {
                "reference_json_sha256": "a" * 64,
                "rank0_fingerprints": {"w13.weight": "b" * 64},
            },
        },
        "settings": {
            "routing": list(probe.ROUTINGS),
            "m": list(probe.M_VALUES),
            "seed": 4104,
            "warmup": 5,
            "iters": 50,
            "repeats": 5,
        },
        "results": [
            {
                "routing": routing,
                "m": m,
                "cuda_graph": {"median_ms": medians[(routing, m)]},
            }
            for routing in probe.ROUTINGS
            for m in probe.M_VALUES
        ],
    }


class CooperativeFc2BenchmarkTest(unittest.TestCase):
    def test_parser_requires_explicit_two_run_variant(self) -> None:
        parser = probe.build_parser()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--layer-file", "/tmp/layer", "--output", "/tmp/o"])
        args = parser.parse_args(
            [
                "--layer-file",
                "/tmp/layer",
                "--output",
                "/tmp/o",
                "--variant",
                "candidate",
                "--baseline-json",
                "/tmp/base.json",
            ]
        )
        self.assertEqual(args.balanced_m4_max_ms, 0.682812)
        self.assertEqual(args.max_relative_regression, 0.02)

    def test_strict_opt_in_matches_variant_and_rejects_malformed_value(self) -> None:
        self.assertFalse(probe._strict_opt_in("baseline", {}))
        self.assertTrue(
            probe._strict_opt_in("candidate", {probe.COOPERATIVE_ENV: "1"})
        )
        with self.assertRaisesRegex(RuntimeError, "candidate run requires"):
            probe._strict_opt_in("candidate", {})
        with self.assertRaisesRegex(RuntimeError, "exactly 0 or 1"):
            probe._strict_opt_in("baseline", {probe.COOPERATIVE_ENV: "true"})

    def test_candidate_passes_absolute_and_relative_gates(self) -> None:
        baseline_values = {
            (routing, m): (0.77 if m == 4 else 0.20)
            for routing in probe.ROUTINGS
            for m in probe.M_VALUES
        }
        candidate_values = dict(baseline_values)
        candidate_values[("balanced", 4)] = 0.682812
        candidate_values[("random", 4)] = baseline_values[("random", 4)] * 1.02
        candidate_values[("hot", 4)] = 0.50
        candidate_values[("balanced", 1)] *= 1.01

        gate = probe.evaluate_candidate(
            _report("candidate", candidate_values),
            _report("baseline", baseline_values),
            balanced_m4_max_ms=0.682812,
            max_relative_regression=0.02,
        )

        self.assertTrue(gate["passed"])
        balanced_m4 = next(
            row
            for row in gate["rows"]
            if row["routing"] == "balanced" and row["m"] == 4
        )
        self.assertEqual(balanced_m4["gate"], "absolute_serving_projection")
        self.assertEqual(balanced_m4["deadline_ms"], 0.682812)

    def test_balanced_m4_near_miss_fails_even_when_it_beats_baseline(self) -> None:
        baseline_values = {
            (routing, m): (0.77 if m == 4 else 0.20)
            for routing in probe.ROUTINGS
            for m in probe.M_VALUES
        }
        candidate_values = dict(baseline_values)
        candidate_values[("balanced", 4)] = 0.683
        gate = probe.evaluate_candidate(
            _report("candidate", candidate_values),
            _report("baseline", baseline_values),
            balanced_m4_max_ms=0.682812,
            max_relative_regression=0.02,
        )
        self.assertFalse(gate["passed"])

    def test_m1_and_route_stress_regressions_fail(self) -> None:
        baseline_values = {
            (routing, m): (0.70 if m == 4 else 0.20)
            for routing in probe.ROUTINGS
            for m in probe.M_VALUES
        }
        candidate_values = dict(baseline_values)
        candidate_values[("balanced", 4)] = 0.68
        candidate_values[("random", 4)] *= 1.021
        candidate_values[("hot", 1)] *= 1.021
        gate = probe.evaluate_candidate(
            _report("candidate", candidate_values),
            _report("baseline", baseline_values),
            balanced_m4_max_ms=0.682812,
            max_relative_regression=0.02,
        )
        failed = {
            (row["routing"], row["m"])
            for row in gate["rows"]
            if not row["passed"]
        }
        self.assertEqual(failed, {("random", 4), ("hot", 1)})

    def test_fingerprint_drift_fails_before_performance_comparison(self) -> None:
        values = {
            (routing, m): 0.2
            for routing in probe.ROUTINGS
            for m in probe.M_VALUES
        }
        baseline = _report("baseline", values)
        candidate = _report("candidate", values)
        candidate["settings"]["seed"] = 999
        with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
            probe.evaluate_candidate(
                candidate,
                baseline,
                balanced_m4_max_ms=0.682812,
                max_relative_regression=0.02,
            )

    def test_result_matrix_rejects_missing_and_duplicate_rows(self) -> None:
        values = {
            (routing, m): 0.2
            for routing in probe.ROUTINGS
            for m in probe.M_VALUES
        }
        report = _report("baseline", values)
        report["results"].pop()
        with self.assertRaisesRegex(RuntimeError, "matrix mismatch"):
            probe._rows_by_key(report)

        report = _report("baseline", values)
        report["results"].append(dict(report["results"][0]))
        with self.assertRaisesRegex(RuntimeError, "duplicate result"):
            probe._rows_by_key(report)


if __name__ == "__main__":
    unittest.main()
