# SPDX-License-Identifier: MIT

import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dashboard.server import HEAD_NODE_LABEL, WORKER_NODE_LABEL, LoadSampler


def _node(weight_load: dict) -> dict:
    return {"state": "ready", "weightLoad": weight_load}


class LoadSamplerDiagnosticsTest(unittest.TestCase):
    def test_direct_load_sums_target_and_drafter_times(self) -> None:
        parsed = LoadSampler()._parse(
            "Loading weights took 81.25 seconds\n"
            "Loading drafter model\n"
            "Loading weights took 4.75 seconds\n"
            "Application startup complete\n"
        )

        self.assertEqual(parsed["state"], "ready")
        diagnostic = parsed["weightLoad"]
        self.assertEqual(diagnostic["mode"], "direct")
        self.assertEqual(diagnostic["phaseCount"], 2)
        self.assertEqual(diagnostic["elapsedSeconds"], 86.0)
        self.assertEqual(
            [phase["name"] for phase in diagnostic["phases"]],
            ["target", "drafter"],
        )
        self.assertFalse(diagnostic["timingComparable"])

    def test_synchronized_direct_markers_are_comparison_eligible(self) -> None:
        diagnostic = LoadSampler._parse_weight_load(
            "DSPARK_WEIGHT_LOAD mode=direct event=start run=a pid=7 id=0 "
            "rank=0 role=local_reader phase=Target\n"
            "DSPARK_WEIGHT_LOAD mode=direct event=complete run=a pid=7 id=0 "
            "rank=0 role=local_reader phase=Target elapsed_s=10.25\n"
            "DSPARK_WEIGHT_LOAD mode=direct event=start run=a pid=7 id=1 "
            "rank=0 role=local_reader phase=Drafter\n"
            "DSPARK_WEIGHT_LOAD mode=direct event=complete run=a pid=7 id=1 "
            "rank=0 role=local_reader phase=Drafter elapsed_s=2.50\n"
        )

        self.assertIsNotNone(diagnostic)
        assert diagnostic is not None
        self.assertEqual(diagnostic["mode"], "direct")
        self.assertEqual(diagnostic["elapsedSeconds"], 12.75)
        self.assertTrue(diagnostic["timingComparable"])
        self.assertEqual(diagnostic["timerKind"], "synchronized_ram")

    def test_roce_load_aggregates_all_phases_with_exact_bytes(self) -> None:
        parsed = LoadSampler()._parse(
            "INFO DSPARK_WEIGHT_LOAD mode=roce_tp event=start pid=41 id=0 "
            "rank=1 role=receiver phase=Target buffer_bytes=67108864 "
            "protocol=2 transport=pynccl\n"
            "INFO DSPARK_WEIGHT_LOAD mode=roce_tp event=complete pid=41 id=0 "
            "rank=1 role=receiver phase=Target tensors=120 batches=8 "
            "source_bytes=17179869184 traffic_bytes=8589934592 "
            "direct_bytes=7516192768 staged_bytes=1073741824 "
            "max_frame_bytes=67108864 max_write_bytes=536870912 "
            "elapsed_s=10.250000\n"
            "INFO DSPARK_WEIGHT_LOAD mode=roce_tp event=start pid=41 id=1 "
            "rank=1 role=receiver phase=Drafter buffer_bytes=67108864 "
            "protocol=2 transport=pynccl\n"
            "INFO DSPARK_WEIGHT_LOAD mode=roce_tp event=complete pid=41 id=1 "
            "rank=1 role=receiver phase=Drafter tensors=10 batches=2 "
            "source_bytes=2147483648 traffic_bytes=1073741824 "
            "direct_bytes=1073741824 staged_bytes=0 "
            "max_frame_bytes=33554432 max_write_bytes=134217728 "
            "elapsed_s=2.500000\n"
        )

        diagnostic = parsed["weightLoad"]
        self.assertEqual(diagnostic["state"], "complete")
        self.assertEqual(diagnostic["role"], "receiver")
        self.assertEqual(diagnostic["phaseCount"], 2)
        self.assertEqual(diagnostic["sourceBytes"], 18 * 1024**3)
        self.assertEqual(diagnostic["trafficBytes"], 9 * 1024**3)
        self.assertEqual(diagnostic["directBytes"], 8 * 1024**3)
        self.assertEqual(diagnostic["stagedBytes"], 1 * 1024**3)
        self.assertEqual(diagnostic["maxFrameBytes"], 64 * 1024**2)
        self.assertEqual(diagnostic["maxWriteBytes"], 512 * 1024**2)
        self.assertEqual(diagnostic["protocol"], 2)
        self.assertEqual(diagnostic["transport"], "pynccl")
        self.assertEqual(diagnostic["payloadRatio"], 0.5)
        self.assertEqual(diagnostic["tensors"], 130)
        self.assertEqual(diagnostic["batches"], 10)
        self.assertEqual(diagnostic["elapsedSeconds"], 12.75)
        self.assertEqual(diagnostic["bufferBytes"], 64 * 1024**2)
        self.assertAlmostEqual(
            diagnostic["throughputBytesPerSecond"], 9 * 1024**3 / 12.75
        )

    def test_roce_failure_is_exposed_without_log_message_injection(self) -> None:
        parsed = LoadSampler()._parse(
            "DSPARK_WEIGHT_LOAD mode=roce_tp event=start pid=9 id=0 "
            "rank=0 role=reader phase=Target buffer_bytes=268435456\n"
            "DSPARK_WEIGHT_LOAD mode=roce_tp event=failed pid=9 id=0 "
            "rank=0 role=reader phase=Target error_type=RuntimeError\n"
            "Engine core initialization failed\n"
        )

        self.assertEqual(parsed["state"], "failed")
        self.assertEqual(parsed["weightLoad"]["state"], "failed")
        self.assertEqual(parsed["weightLoad"]["error"], "RuntimeError")

    def test_engine_failure_without_marker_is_exposed(self) -> None:
        parsed = LoadSampler()._parse("Engine core initialization failed\n")

        self.assertEqual(parsed["state"], "failed")
        self.assertEqual(parsed["weightLoad"]["state"], "failed")
        self.assertEqual(parsed["weightLoad"]["mode"], "unknown")

    def test_newest_process_replaces_old_container_log_events(self) -> None:
        parsed = LoadSampler._parse_weight_load(
            "DSPARK_WEIGHT_LOAD mode=roce_tp event=failed pid=8 id=0 "
            "rank=0 role=reader phase=Target error_type=RuntimeError\n"
            "DSPARK_WEIGHT_LOAD mode=roce_tp event=start pid=9 id=0 "
            "rank=0 role=reader phase=Target buffer_bytes=268435456\n"
        )

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["state"], "loading")
        self.assertNotIn("error", parsed)

    def test_run_token_replaces_same_pid_and_load_ids(self) -> None:
        parsed = LoadSampler._parse_weight_load(
            "DSPARK_WEIGHT_LOAD mode=roce_tp event=complete run=old pid=9 id=0 "
            "rank=0 role=reader phase=Target source_bytes=2000 "
            "traffic_bytes=1000 elapsed_s=10.0\n"
            "DSPARK_WEIGHT_LOAD mode=roce_tp event=complete run=old pid=9 id=1 "
            "rank=0 role=reader phase=Drafter source_bytes=200 "
            "traffic_bytes=100 elapsed_s=2.0\n"
            "DSPARK_WEIGHT_LOAD mode=roce_tp event=start run=new pid=9 id=0 "
            "rank=0 role=reader phase=Target buffer_bytes=256\n"
        )

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["runId"], "new")
        self.assertEqual(parsed["state"], "loading")
        self.assertEqual(parsed["phaseCount"], 0)
        self.assertEqual(parsed["trafficBytes"], 0)

    def test_same_inherited_run_token_still_uses_newest_process(self) -> None:
        parsed = LoadSampler._parse_weight_load(
            "DSPARK_WEIGHT_LOAD mode=direct event=complete run=a pid=8 id=0 "
            "rank=0 role=local_reader phase=Target elapsed_s=10.0\n"
            "DSPARK_WEIGHT_LOAD mode=direct event=start run=a pid=9 id=0 "
            "rank=0 role=local_reader phase=Target\n"
        )

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["processId"], 9)
        self.assertEqual(parsed["state"], "loading")

    def test_drafter_only_tail_is_partial_and_not_benchmarked(self) -> None:
        sampler = LoadSampler()
        text = (
            "DSPARK_WEIGHT_LOAD mode=direct event=complete run=a pid=9 id=1 "
            "rank=0 role=local_reader phase=Draft elapsed_s=2.5"
        )
        diagnostic = sampler._parse(text)["weightLoad"]
        nodes = {
            HEAD_NODE_LABEL: _node(diagnostic),
            WORKER_NODE_LABEL: _node(dict(diagnostic, rank=1)),
        }

        summary = sampler.weight_summary(nodes, api_ready=True)

        self.assertEqual(diagnostic["state"], "partial")
        self.assertFalse(diagnostic["phaseSequenceComplete"])
        self.assertEqual(summary["state"], "partial")
        self.assertEqual(summary["observed"], {})

    def test_log_failure_does_not_reattach_cached_diagnostics(self) -> None:
        sampler = LoadSampler()
        complete = SimpleNamespace(
            returncode=0,
            stdout=(
                "DSPARK_WEIGHT_LOAD mode=direct event=complete run=a pid=9 id=0 "
                "rank=0 role=local_reader phase=Target elapsed_s=2.5"
            ),
            stderr="",
        )
        failed = SimpleNamespace(returncode=1, stdout="", stderr="denied")

        with patch("dashboard.server.subprocess.run", return_value=complete):
            first = sampler.snapshot()
        sampler._at = 0.0
        with patch("dashboard.server.subprocess.run", return_value=failed):
            second = sampler.snapshot()

        self.assertIn("weightLoad", first[HEAD_NODE_LABEL])
        self.assertEqual(second[HEAD_NODE_LABEL]["state"], "unavailable")
        self.assertNotIn("weightLoad", second[HEAD_NODE_LABEL])

    def test_cache_merge_keeps_target_when_only_drafter_remains_in_tail(self) -> None:
        target = LoadSampler._parse_weight_load(
            "DSPARK_WEIGHT_LOAD mode=roce_tp event=complete run=a pid=9 id=0 "
            "rank=1 role=receiver phase=Target tensors=100 batches=8 "
            "source_bytes=2000 traffic_bytes=1000 elapsed_s=10.0\n"
        )
        drafter = LoadSampler._parse_weight_load(
            "DSPARK_WEIGHT_LOAD mode=roce_tp event=complete run=a pid=9 id=1 "
            "rank=1 role=receiver phase=Drafter tensors=10 batches=2 "
            "source_bytes=200 traffic_bytes=100 elapsed_s=2.0\n"
        )

        self.assertIsNotNone(target)
        self.assertIsNotNone(drafter)
        assert target is not None and drafter is not None
        merged = LoadSampler._merge_weight_load(target, drafter)

        self.assertEqual(merged["phaseCount"], 2)
        self.assertEqual(merged["sourceBytes"], 2200)
        self.assertEqual(merged["trafficBytes"], 1100)
        self.assertEqual(merged["elapsedSeconds"], 12.0)

    def test_direct_cache_merge_keeps_target_and_drafter_times(self) -> None:
        target = LoadSampler._parse_weight_load(
            "Loading weights took 95.94 seconds\n"
        )
        drafter = LoadSampler._parse_weight_load(
            "DSpark draft model loaded: 96 params\n"
            "Loading weights took 26.17 seconds\n"
        )

        self.assertIsNotNone(target)
        self.assertIsNotNone(drafter)
        assert target is not None and drafter is not None
        merged = LoadSampler._merge_weight_load(target, drafter)

        self.assertEqual(merged["phaseCount"], 2)
        self.assertAlmostEqual(merged["elapsedSeconds"], 122.11)

    def test_ready_state_preserves_per_rank_diagnostics(self) -> None:
        nodes = {
            HEAD_NODE_LABEL: _node(
                {"mode": "roce_tp", "state": "complete", "rank": 0}
            ),
            WORKER_NODE_LABEL: _node(
                {"mode": "roce_tp", "state": "complete", "rank": 1}
            ),
        }

        ready = LoadSampler._mark_ready(nodes)

        self.assertEqual(ready[HEAD_NODE_LABEL]["state"], "ready")
        self.assertEqual(ready[WORKER_NODE_LABEL]["state"], "ready")
        self.assertEqual(ready[HEAD_NODE_LABEL]["weightLoad"]["rank"], 0)
        self.assertEqual(ready[WORKER_NODE_LABEL]["weightLoad"]["rank"], 1)

    def test_summary_uses_slowest_rank_and_receiver_traffic(self) -> None:
        head = {
            "mode": "roce_tp",
            "state": "complete",
            "rank": 0,
            "role": "reader",
            "phaseCount": 2,
            "elapsedSeconds": 44.0,
            "sourceBytes": 2000,
            "trafficBytes": 1000,
            "tensors": 20,
            "batches": 4,
        }
        worker = dict(head, rank=1, role="receiver", elapsedSeconds=45.0)

        summary = LoadSampler._summarize(
            {HEAD_NODE_LABEL: _node(head), WORKER_NODE_LABEL: _node(worker)}
        )

        self.assertEqual(summary["criticalElapsedSeconds"], 45.0)
        self.assertEqual(summary["trafficBytes"], 1000)
        self.assertAlmostEqual(summary["throughputBytesPerSecond"], 1000 / 45.0)
        self.assertTrue(summary["ranksAgree"])

    def test_summary_sums_each_phases_slowest_rank(self) -> None:
        head = {
            "mode": "direct",
            "state": "complete",
            "rank": 0,
            "phaseCount": 2,
            "elapsedSeconds": 12.0,
            "timingComparable": True,
            "phases": [
                {"id": 0, "state": "complete", "elapsedSeconds": 10.0},
                {"id": 1, "state": "complete", "elapsedSeconds": 2.0},
            ],
        }
        worker = {
            **head,
            "rank": 1,
            "elapsedSeconds": 13.0,
            "phases": [
                {"id": 0, "state": "complete", "elapsedSeconds": 8.0},
                {"id": 1, "state": "complete", "elapsedSeconds": 5.0},
            ],
        }

        summary = LoadSampler._summarize(
            {HEAD_NODE_LABEL: _node(head), WORKER_NODE_LABEL: _node(worker)}
        )

        self.assertEqual(summary["criticalElapsedSeconds"], 15.0)

    def test_roce_rank_agreement_checks_each_phase(self) -> None:
        def diagnostic(rank: int, first: int, second: int) -> dict:
            return {
                "mode": "roce_tp",
                "state": "complete",
                "rank": rank,
                "phaseCount": 2,
                "elapsedSeconds": 12.0,
                "sourceBytes": 240,
                "trafficBytes": 120,
                "tensors": 12,
                "batches": 4,
                "timingComparable": True,
                "phases": [
                    {
                        "id": 0,
                        "name": "Target",
                        "state": "complete",
                        "elapsedSeconds": 10.0,
                        "sourceBytes": first * 2,
                        "trafficBytes": first,
                        "tensors": 6,
                        "batches": 2,
                    },
                    {
                        "id": 1,
                        "name": "Drafter",
                        "state": "complete",
                        "elapsedSeconds": 2.0,
                        "sourceBytes": second * 2,
                        "trafficBytes": second,
                        "tensors": 6,
                        "batches": 2,
                    },
                ],
            }

        summary = LoadSampler._summarize(
            {
                HEAD_NODE_LABEL: _node(diagnostic(0, 100, 20)),
                WORKER_NODE_LABEL: _node(diagnostic(1, 20, 100)),
            }
        )

        self.assertFalse(summary["ranksAgree"])

    def test_summary_does_not_benchmark_one_rank_or_mixed_modes(self) -> None:
        direct = {
            "mode": "direct",
            "state": "complete",
            "rank": 0,
            "elapsedSeconds": 10.0,
        }
        roce = {
            "mode": "roce_tp",
            "state": "complete",
            "rank": 1,
            "elapsedSeconds": 8.0,
        }

        partial = LoadSampler._summarize({HEAD_NODE_LABEL: _node(direct)})
        mixed = LoadSampler._summarize(
            {HEAD_NODE_LABEL: _node(direct), WORKER_NODE_LABEL: _node(roce)}
        )

        self.assertEqual(partial["state"], "partial")
        self.assertEqual(mixed["mode"], "mixed")
        self.assertEqual(mixed["state"], "inconsistent")

    def test_summary_does_not_benchmark_disagreeing_ranks(self) -> None:
        sampler = LoadSampler()
        head = {
            "mode": "roce_tp",
            "state": "complete",
            "rank": 0,
            "phaseCount": 2,
            "elapsedSeconds": 10.0,
            "sourceBytes": 2000,
            "trafficBytes": 1000,
            "tensors": 20,
            "batches": 4,
            "timingComparable": True,
        }
        worker = dict(head, rank=1, trafficBytes=999)

        summary = sampler.weight_summary(
            {HEAD_NODE_LABEL: _node(head), WORKER_NODE_LABEL: _node(worker)},
            api_ready=True,
        )

        self.assertFalse(summary["ranksAgree"])
        self.assertEqual(summary["observed"], {})

    def test_observed_modes_require_ready_comparable_two_rank_samples(self) -> None:
        sampler = LoadSampler()
        direct = {
            HEAD_NODE_LABEL: _node(
                {
                    "mode": "direct",
                    "state": "complete",
                    "rank": 0,
                    "elapsedSeconds": 80.0,
                    "timingComparable": True,
                }
            ),
            WORKER_NODE_LABEL: _node(
                {
                    "mode": "direct",
                    "state": "complete",
                    "rank": 1,
                    "elapsedSeconds": 90.0,
                    "timingComparable": True,
                }
            ),
        }
        roce = {
            HEAD_NODE_LABEL: _node(
                {
                    "mode": "roce_tp",
                    "state": "complete",
                    "rank": 0,
                    "elapsedSeconds": 60.0,
                    "timingComparable": True,
                }
            ),
            WORKER_NODE_LABEL: _node(
                {
                    "mode": "roce_tp",
                    "state": "complete",
                    "rank": 1,
                    "elapsedSeconds": 60.0,
                    "timingComparable": True,
                }
            ),
        }

        self.assertEqual(sampler.weight_summary(direct)["observed"], {})
        sampler.weight_summary(direct, api_ready=True)
        comparison = sampler.weight_summary(roce, api_ready=True)

        self.assertEqual(set(comparison["observed"]), {"direct", "roce_tp"})
        self.assertEqual(comparison["directVsRoceSpeedup"], 1.5)
        self.assertEqual(comparison["directVsRoceSavedSeconds"], 30.0)


class DashboardMarkupTest(unittest.TestCase):
    def test_every_javascript_id_exists_once(self) -> None:
        html = (Path(__file__).parents[1] / "dashboard" / "index.html").read_text()
        declared = re.findall(r'\bid="([^"]+)"', html)
        referenced = set(re.findall(r'\$\("([^"]+)"\)', html))

        self.assertEqual(len(declared), len(set(declared)), "duplicate HTML id")
        self.assertEqual(
            referenced - set(declared), set(), "JavaScript references missing id"
        )

    def test_weight_diagnostic_cards_and_dynamic_labels_exist(self) -> None:
        html = (Path(__file__).parents[1] / "dashboard" / "index.html").read_text()
        expected = {
            "endpointLabel",
            "headLoadLabel",
            "workerLoadLabel",
            "weightStatus",
            "weightMode",
            "weightModeDetail",
            "weightElapsed",
            "weightElapsedDetail",
            "weightTraffic",
            "weightTrafficDetail",
            "weightCompare",
            "weightCompareDetail",
            "headComputeLabel",
            "workerComputeLabel",
        }
        declared = set(re.findall(r'\bid="([^"]+)"', html))
        self.assertEqual(expected - declared, set())


if __name__ == "__main__":
    unittest.main()
