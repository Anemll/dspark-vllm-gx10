# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIDENCE_PATH = ROOT / "overlay/vllm/v1/worker/gpu/spec_decode/dspark/confidence.py"
VARIABLE_VERIFIER_PATH = (
    ROOT / "overlay/vllm/v1/worker/gpu/spec_decode/dspark/variable_verifier.py"
)
PROBE_PATH = ROOT / "scripts/probe_dspark_confidence_head.py"
SPLIT_PROBE_PATH = ROOT / "scripts/probe_dspark_execute_sample_split.py"
DOCKERFILE_PATH = ROOT / "docker/Dockerfile.dspark-confidence-overlay"
DOCKERIGNORE_PATH = ROOT / "docker/Dockerfile.dspark-confidence-overlay.dockerignore"
OVERLAP_PATH = (
    ROOT / "overlay/vllm/v1/worker/gpu/spec_decode/dspark/overlap_trace.py"
)
PATCHER_PATH = ROOT / "scripts/patch_dspark_variable_verifier.py"
UPSTREAM_ROOT = Path(
    "/Users/anemll/SourceRelease/GITHUB/ML_playground/dspark-vllm-gx10/"
    ".build/vllm-upstream"
)
VARIABLE_SPEC = importlib.util.spec_from_file_location(
    "vllm.v1.worker.gpu.spec_decode.dspark.variable_verifier",
    VARIABLE_VERIFIER_PATH,
)
assert VARIABLE_SPEC and VARIABLE_SPEC.loader
variable_verifier = importlib.util.module_from_spec(VARIABLE_SPEC)
sys.modules[VARIABLE_SPEC.name] = variable_verifier
VARIABLE_SPEC.loader.exec_module(variable_verifier)
SPEC = importlib.util.spec_from_file_location("dspark_confidence", CONFIDENCE_PATH)
assert SPEC and SPEC.loader
confidence = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = confidence
SPEC.loader.exec_module(confidence)


class ConfidenceConfigTests(unittest.TestCase):
    def test_defaults_are_pinned_production_off(self) -> None:
        config = confidence.parse_confidence_config({})
        self.assertEqual(config.scheduler, "off")
        self.assertEqual(config.threshold, 0.0)
        self.assertFalse(config.enabled)

    def test_enabled_probability_threshold(self) -> None:
        config = confidence.parse_confidence_config(
            {
                confidence.SCHEDULER_ENV: "on",
                confidence.THRESHOLD_ENV: "0.75",
            }
        )
        self.assertTrue(config.enabled)
        self.assertEqual(config.threshold, 0.75)

    def test_invalid_mode_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be one of"):
            confidence.parse_confidence_config(
                {confidence.SCHEDULER_ENV: "true"}
            )

    def test_invalid_thresholds_fail_closed(self) -> None:
        for value in ("nan", "inf", "-0.1", "1.1", "not-a-number"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                confidence.parse_confidence_config(
                    {
                        confidence.SCHEDULER_ENV: "on",
                        confidence.THRESHOLD_ENV: value,
                    }
                )

    def test_off_requires_zero_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be 0.0"):
            confidence.parse_confidence_config(
                {
                    confidence.SCHEDULER_ENV: "off",
                    confidence.THRESHOLD_ENV: "0.5",
                }
            )


class ConfidencePrefixTests(unittest.TestCase):
    def test_probability_policy_matches_deepspec_sigmoid_domain(self) -> None:
        probabilities = torch.tensor(
            [[0.8, 0.6, 0.4, 0.9, 0.9]], dtype=torch.float32
        )
        observed, below, prefix, lengths = (
            confidence.confidence_probability_policy(
                torch.logit(probabilities),
                threshold=0.5,
            )
        )
        self.assertTrue(torch.allclose(observed, probabilities))
        self.assertEqual(below.tolist(), [[False, False, True, False, False]])
        self.assertEqual(prefix.tolist(), [[True, True, False, False, False]])
        self.assertEqual(lengths.tolist(), [2])

    def test_first_below_threshold_excludes_it_and_tail(self) -> None:
        draft_tokens = torch.tensor(
            [[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]], dtype=torch.int64
        )
        probabilities = torch.tensor(
            [[0.9, 0.8, 0.2, 0.99, 0.99], [0.4, 0.9, 0.9, 0.9, 0.9]],
            dtype=torch.float32,
        )
        masked, lengths = confidence.mask_draft_tokens_by_confidence(
            draft_tokens,
            torch.logit(probabilities),
            threshold=0.5,
        )
        self.assertEqual(lengths.tolist(), [2, 0])
        self.assertEqual(masked.tolist(), [[10, 11, -1, -1, -1], [-1] * 5])

    def test_zero_and_one_threshold_extremes(self) -> None:
        draft_tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
        logits = torch.zeros((1, 5), dtype=torch.float32)
        full, full_lengths = confidence.mask_draft_tokens_by_confidence(
            draft_tokens, logits, threshold=0.0
        )
        empty, empty_lengths = confidence.mask_draft_tokens_by_confidence(
            draft_tokens, logits, threshold=1.0
        )
        self.assertTrue(torch.equal(full, draft_tokens))
        self.assertEqual(full_lengths.tolist(), [5])
        self.assertEqual(empty.tolist(), [[-1] * 5])
        self.assertEqual(empty_lengths.tolist(), [0])

    def test_tail_trimming_rejects_holes(self) -> None:
        self.assertEqual(
            confidence.trim_invalid_draft_tail([1, 2, -1, -1]), [1, 2]
        )
        self.assertEqual(confidence.trim_invalid_draft_tail([1, 2, 3]), [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "non-contiguous"):
            confidence.trim_invalid_draft_tail([1, -1, 3, -1])
        with self.assertRaisesRegex(ValueError, "invalid negative"):
            confidence.trim_invalid_draft_tail([1, -2, -1])

    def test_physical_compaction_shrinks_target_rows_per_request(self) -> None:
        output = SimpleNamespace(
            scheduled_spec_decode_tokens={"a": [-1] * 5, "b": [-1] * 5},
            num_scheduled_tokens={"a": 6, "b": 6},
            total_num_scheduled_tokens=12,
        )
        invalid = confidence.compact_scheduler_output_for_variable_drafts(
            output,
            ["a", "b"],
            [[10, 11, -1, -1, -1], [-1, -1, -1, -1, -1]],
        )
        self.assertEqual(invalid, {"a": 3, "b": 5})
        self.assertEqual(output.scheduled_spec_decode_tokens, {"a": [10, 11]})
        self.assertEqual(output.num_scheduled_tokens, {"a": 3, "b": 1})
        self.assertEqual(output.total_num_scheduled_tokens, 4)

    def test_physical_compaction_preserves_full_prefix(self) -> None:
        output = SimpleNamespace(
            scheduled_spec_decode_tokens={"a": [-1] * 5},
            num_scheduled_tokens={"a": 6},
            total_num_scheduled_tokens=6,
        )
        invalid = confidence.compact_scheduler_output_for_variable_drafts(
            output, ["a"], [[10, 11, 12, 13, 14]]
        )
        self.assertEqual(invalid, {})
        self.assertEqual(
            output.scheduled_spec_decode_tokens["a"], [10, 11, 12, 13, 14]
        )
        self.assertEqual(output.num_scheduled_tokens["a"], 6)

    def test_physical_compaction_fails_closed_on_missing_row(self) -> None:
        output = SimpleNamespace(
            scheduled_spec_decode_tokens={"missing": [-1] * 5},
            num_scheduled_tokens={"missing": 6},
            total_num_scheduled_tokens=6,
        )
        with self.assertRaisesRegex(RuntimeError, "missing the prior proposal"):
            confidence.compact_scheduler_output_for_variable_drafts(
                output, ["other"], [[1, 2, 3, 4, 5]]
            )

    def test_telemetry_records_exposure_and_physical_rows(self) -> None:
        try:
            from prometheus_client import CollectorRegistry, generate_latest
        except ModuleNotFoundError:
            self.skipTest("prometheus_client unavailable in local test runtime")

        registry = CollectorRegistry()
        metrics = confidence.DSparkConfidenceMetrics(0.5, registry=registry)
        probabilities = torch.tensor(
            [[0.9, 0.8, 0.2, 0.9, 0.9], [0.9, 0.2, 0.9, 0.9, 0.9]],
            dtype=torch.float32,
        )
        observed = metrics.observe(torch.logit(probabilities))
        self.assertEqual(observed["exposed_per_position"], [2, 1, 0, 0, 0])
        self.assertEqual(observed["prefix_lengths"], [2, 1])
        metrics.observe_physical_target_rows([3, 2])
        metrics.observe_d2h_copy_completion(fallback_wait=False)
        metrics.observe_d2h_copy_completion(fallback_wait=True)
        exposition = generate_latest(registry).decode()
        self.assertIn("vllm:dspark_confidence_position_exposed_total", exposition)
        self.assertIn("vllm:dspark_confidence_physical_target_rows_count", exposition)
        self.assertIn(
            'vllm:dspark_confidence_d2h_copy_completion_total{result="ready"',
            exposition,
        )
        self.assertIn(
            'vllm:dspark_confidence_d2h_copy_completion_total{result="fallback_wait"',
            exposition,
        )

    def test_engine_compaction_telemetry_rejects_partial_handoff(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "incomplete DSpark"):
            confidence.observe_engine_compaction_telemetry([3], None)
        with self.assertRaisesRegex(RuntimeError, "incomplete DSpark"):
            confidence.observe_engine_compaction_telemetry(None, False)


class OverlayContractTests(unittest.TestCase):
    def test_deepseek_model_loads_exact_head_parameter(self) -> None:
        source = (
            ROOT / "overlay/vllm/models/deepseek_v4/nvidia/dspark.py"
        ).read_text()
        self.assertIn("class DSparkConfidenceHead", source)
        self.assertIn('"model.confidence_head.proj.weight"', source)
        self.assertIn('"confidence_head.",', source)
        self.assertNotIn("drop its weights", source)

    def test_speculator_uses_official_hidden_plus_markov_contract(self) -> None:
        source = (
            ROOT
            / "overlay/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py"
        ).read_text()
        self.assertIn("sample_hidden[:, i], markov_embed", source)
        self.assertIn("mask_draft_tokens_by_confidence", source)
        self.assertIn("confidence_head_loaded", source)

    def test_real_score_telemetry_is_nonblocking_and_outside_graph(self) -> None:
        source = (
            ROOT
            / "overlay/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py"
        ).read_text()
        self.assertIn("class _DSparkConfidenceTelemetry", source)
        self.assertIn("self.events[slot].query()", source)
        self.assertIn("copy_(confidence_logits, non_blocking=True)", source)
        self.assertIn("and not dummy_run", source)
        self.assertIn("self.confidence_logits[: input_batch.num_reqs]", source)

    def test_real_score_metrics_pin_probability_domain_and_positions(self) -> None:
        source = CONFIDENCE_PATH.read_text()
        self.assertIn('"vllm:dspark_confidence_probability"', source)
        self.assertIn('"vllm:dspark_confidence_below_threshold"', source)
        self.assertIn('"vllm:dspark_confidence_prefix_length"', source)
        self.assertIn('"vllm:dspark_confidence_position_exposed"', source)
        self.assertIn('"vllm:dspark_confidence_physical_target_rows"', source)
        self.assertIn('"vllm:dspark_confidence_d2h_copy_completion"', source)
        self.assertIn('(\"position\", \"threshold\")', source)
        self.assertIn("confidence_probability_policy(", source)

    def test_variable_lengths_are_transferred_only_when_enabled(self) -> None:
        source = (
            ROOT / "overlay/vllm/v1/worker/gpu/spec_decode/utils.py"
        ).read_text()
        self.assertIn("self.variable_draft_lengths", source)
        self.assertIn("trim_invalid_draft_tail", source)
        self.assertIn("compact_scheduler_output_for_variable_drafts", source)
        self.assertIn("self.scheduler_requires_draft_tokens", source)
        self.assertIn("complete_async_copy_if_needed", source)
        self.assertIn("last_physical_target_rows", source)
        self.assertIn("get_last_compaction_telemetry", source)
        confidence_source = VARIABLE_VERIFIER_PATH.read_text()
        self.assertIn(
            "scheduler_output.total_num_scheduled_tokens -= removed",
            confidence_source,
        )

    def test_probe_proves_physical_rows_and_metrics_match_prefix(self) -> None:
        source = PROBE_PATH.read_text()
        self.assertIn("compact_scheduler_output_for_variable_drafts", source)
        self.assertIn("Scheduler.make_spec_decoding_stats", source)
        self.assertIn('physical != [10, 11]', source)
        self.assertIn('output.num_scheduled_tokens != {"probe": 3}', source)
        self.assertIn('observed != {"draft": 2, "accepted": 1}', source)
        self.assertIn("truncated_length = len(truncated)", source)
        self.assertIn('"truncated_proposal_length": truncated_length', source)
        self.assertIn('"metrics_proposed_equals_truncated"', source)
        self.assertIn('"scheduled_slots_seen_by_runner": scheduled_slots', source)
        self.assertIn('"physical_verifier_shortened"', source)
        self.assertIn('"variable_length_verify_ready": True', source)
        self.assertIn("inspect.getsource(GPUModelRunner.execute_model)", source)
        self.assertIn("compact_pos >= dispatch_pos", source)
        self.assertIn('"grammar_overlap_uses_max_not_sum": True', source)
        self.assertIn('"exact_c1_cuda_graph_shapes_1_to_6": True', source)
        self.assertIn('"unstructured_scheduler_copy_wait_elided": True', source)
        self.assertIn(
            '"d2h_completion_ready_vs_fallback_telemetry": True', source
        )

    def test_split_probe_executes_real_sample_tokens_body(self) -> None:
        source = SPLIT_PROBE_PATH.read_text()
        self.assertIn("inspect.unwrap(model_runner_module.GPUModelRunner.execute_model)", source)
        self.assertIn("inspect.unwrap(model_runner_module.GPUModelRunner.sample_tokens)", source)
        self.assertIn('choices=("old-nameerror", "pass")', source)
        self.assertIn("trim=EXPECTED_TRIM, dummy_run=False", source)
        self.assertIn("trim={}, dummy_run=True", source)
        self.assertIn("self.execute_model_state.confidence_invalid_spec_tokens", source)
        self.assertIn("self.execute_model_state.dspark_overlap_trace", source)
        self.assertIn('parser.add_argument("--overlap-trace"', source)
        self.assertIn('"overlap_trace": overlap_trace', source)
        self.assertIn('"trimmed_output": trimmed', source)
        self.assertIn('"warmup_output": warmup', source)

    def test_overlap_trace_is_baked_and_opt_in(self) -> None:
        source = OVERLAP_PATH.read_text()
        dockerfile = DOCKERFILE_PATH.read_text()
        dockerignore = DOCKERIGNORE_PATH.read_text()
        self.assertIn('TRACE_ENV = "VLLM_DSPARK_OVERLAP_TRACE"', source)
        self.assertIn(
            'TRACE_JSONL_ENV = "VLLM_DSPARK_OVERLAP_TRACE_JSONL"', source
        )
        self.assertIn("end_verify_and_measure_rank_wait", source)
        self.assertIn("end_draft_and_gather", source)
        self.assertIn("overlap phase conservation drift", source)
        self.assertIn("_append_trace_jsonl(trace)", source)
        self.assertIn("dspark/overlap_trace.py", dockerfile)
        self.assertIn(
            "!overlay/vllm/v1/worker/gpu/spec_decode/dspark/overlap_trace.py",
            dockerignore,
        )

    @unittest.skipUnless(UPSTREAM_ROOT.exists(), "pinned upstream checkout unavailable")
    def test_patch_installs_only_the_four_pinned_integration_seams(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package_root = Path(tmp)
            paths = (
                "vllm/v1/worker/gpu/model_runner.py",
                "vllm/v1/outputs.py",
                "vllm/v1/core/sched/async_scheduler.py",
                "vllm/v1/worker/gpu/cudagraph_utils.py",
            )
            for relative in paths:
                destination = package_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(UPSTREAM_ROOT / relative, destination)
            subprocess.run(
                [
                    sys.executable,
                    str(PATCHER_PATH),
                    "--package-root",
                    str(package_root),
                ],
                check=True,
            )
            model_runner = (package_root / paths[0]).read_text()
            outputs = (package_root / paths[1]).read_text()
            scheduler = (package_root / paths[2]).read_text()
            cudagraph = (package_root / paths[3]).read_text()
            self.assertIn(
                "self.draft_tokens_handler.compact_scheduler_output(", model_runner
            )
            self.assertIn(
                "confidence_invalid_spec_tokens=", model_runner
            )
            self.assertIn(
                "confidence_invalid_spec_tokens: dict[str, int] | None", outputs
            )
            self.assertIn(
                "physical_invalid = (", scheduler
            )
            self.assertIn("variable_dspark = (", cudagraph)
            self.assertIn(
                "decode_query_lens = list(range(1, self.decode_query_len + 1))",
                cudagraph,
            )
            for relative in paths:
                subprocess.run(
                    [sys.executable, "-m", "py_compile", str(package_root / relative)],
                    check=True,
                )

    def test_probe_pins_exact_confidence_input_width(self) -> None:
        source = PROBE_PATH.read_text()
        self.assertIn("head.proj.input_size != 4352", source)
        self.assertIn('"input_width": inputs.shape[-1]', source)

    def test_probe_patches_parameter_modules_single_rank_accessor(self) -> None:
        source = PROBE_PATH.read_text()
        self.assertIn(
            "from vllm.model_executor import parameter as parameter_module",
            source,
        )
        self.assertIn(
            "parameter_module.get_tensor_model_parallel_rank = lambda: 0",
            source,
        )
        self.assertIn(
            "parameter_module.get_tensor_model_parallel_world_size = lambda: 1",
            source,
        )
        self.assertIn(
            "linear_module.get_tensor_model_parallel_world_size = lambda: 1",
            source,
        )

    def test_minimal_image_pins_exact_production_sources(self) -> None:
        source = DOCKERFILE_PATH.read_text()
        dockerignore = DOCKERIGNORE_PATH.read_text()
        self.assertIn(
            "COPY scripts/probe_dspark_execute_sample_split.py ", source
        )
        self.assertIn("dspark-probe-execute-sample-split", source)
        self.assertIn("!scripts/probe_dspark_execute_sample_split.py", dockerignore)
        self.assertIn(
            "efe33c32d37ed7f26d869d94626f1415906d31218ec0ee44d79bb2b815b8cf39",
            source,
        )
        self.assertIn(
            "935900f2fd98bcc1b16312f478d80fb63f32ca0aa900c61b3cd333dfaebfa81a",
            source,
        )
        self.assertIn(
            "39ebdfdc8de50d7fddc324aa011275dccd38f2dcc32c4e3268dbbf3ea915fe49",
            source,
        )
        self.assertIn(
            "58f45c58969cdd9cba707863e82fefda818002de45c621032b58b6eb364deedf",
            source,
        )
        self.assertIn(
            "1e87bf44162452c1908d3a5003685937dbdc56f5634e35e11ed7b6a5322a1c15",
            source,
        )
        self.assertIn(
            "da6343d7e7c394a1738cf72905cbecc208003ffa461ccb441268333a3eb9f884",
            source,
        )
        self.assertIn(
            "303d762141830cd8343976d5be14b34ef1666e7d1d459e089adfd4f5b8cd3ef6",
            source,
        )
        self.assertIn("sigmoid-prefix-v5-overlap-trace", source)
        self.assertIn("patch-dspark-variable-verifier", source)
        self.assertNotIn("COPY overlay/vllm/ ", source)


if __name__ == "__main__":
    unittest.main()
