# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIDENCE_PATH = ROOT / "overlay/vllm/v1/worker/gpu/spec_decode/dspark/confidence.py"
PROBE_PATH = ROOT / "scripts/probe_dspark_confidence_head.py"
DOCKERFILE_PATH = ROOT / "docker/Dockerfile.dspark-confidence-overlay"
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

    def test_variable_lengths_are_transferred_only_when_enabled(self) -> None:
        source = (
            ROOT / "overlay/vllm/v1/worker/gpu/spec_decode/utils.py"
        ).read_text()
        self.assertIn("self.variable_draft_lengths", source)
        self.assertIn("trim_invalid_draft_tail", source)

    def test_probe_exercises_real_async_padding_and_metrics(self) -> None:
        source = PROBE_PATH.read_text()
        self.assertIn("Scheduler.update_draft_token_ids_in_output", source)
        self.assertIn("Scheduler.make_spec_decoding_stats", source)
        self.assertIn('padded != [10, 11, -1, -1, -1]', source)
        self.assertIn('observed != {"draft": 2, "accepted": 1}', source)

    def test_minimal_image_pins_exact_production_sources(self) -> None:
        source = DOCKERFILE_PATH.read_text()
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
        self.assertNotIn("COPY overlay/vllm/ ", source)


if __name__ == "__main__":
    unittest.main()
