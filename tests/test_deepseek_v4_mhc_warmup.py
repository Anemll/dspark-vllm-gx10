# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "overlay/vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py"
)


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _module(name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


class DeepseekV4MhcWarmupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fake_torch = _module(
            "torch",
            nn=SimpleNamespace(Module=object),
            cuda=SimpleNamespace(),
        )
        modules = {
            "torch": fake_torch,
            "vllm": _package("vllm"),
            "vllm.logger": _module(
                "vllm.logger",
                init_logger=lambda name: SimpleNamespace(info=lambda *args: None),
            ),
            "vllm.tracing": _module(
                "vllm.tracing",
                instrument=lambda **kwargs: (lambda function: function),
            ),
            "vllm.utils": _package("vllm.utils"),
            "vllm.utils.math_utils": _module(
                "vllm.utils.math_utils",
                cdiv=lambda numerator, denominator: (
                    numerator + denominator - 1
                )
                // denominator,
            ),
        }
        cls.module_patcher = patch.dict(sys.modules, modules)
        cls.module_patcher.start()
        spec = importlib.util.spec_from_file_location(
            "_deepseek_v4_mhc_warmup_under_test", MODULE_PATH
        )
        assert spec and spec.loader
        cls.warmup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.warmup)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.module_patcher.stop()

    def test_crashing_1067_token_batch_selects_split_two(self) -> None:
        self.assertEqual(
            self.warmup._compute_mhc_pre_num_split(
                num_tokens=1067,
                hidden_size=4096,
                hc_mult=4,
                num_sms=48,
            ),
            2,
        )

    def test_every_reachable_split_has_a_warmup_representative(self) -> None:
        max_tokens = 16_384
        selected = self.warmup._select_mhc_warmup_token_sizes(
            max_tokens=max_tokens,
            cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64, 72],
            hidden_size=4096,
            hc_mult=4,
            num_sms=48,
        )
        reachable = {
            self.warmup._compute_mhc_pre_num_split(
                num_tokens=num_tokens,
                hidden_size=4096,
                hc_mult=4,
                num_sms=48,
            )
            for num_tokens in range(1, max_tokens + 1)
        }
        covered = {
            self.warmup._compute_mhc_pre_num_split(
                num_tokens=num_tokens,
                hidden_size=4096,
                hc_mult=4,
                num_sms=48,
            )
            for num_tokens in selected
        }
        self.assertEqual(covered, reachable)
        self.assertEqual(
            reachable,
            {1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 24, 48},
        )

    def test_split_two_is_materialized_at_its_first_boundary(self) -> None:
        representatives = self.warmup._select_mhc_split_representatives(
            max_tokens=16_384,
            hidden_size=4096,
            hc_mult=4,
            num_sms=48,
        )
        self.assertEqual(representatives[2], 1025)

    def test_selection_preserves_requested_shapes_and_cap(self) -> None:
        selected = self.warmup._select_mhc_warmup_token_sizes(
            max_tokens=100,
            cudagraph_capture_sizes=[7, 72, 101],
            hidden_size=4096,
            hc_mult=4,
            num_sms=48,
        )
        self.assertIn(7, selected)
        self.assertIn(72, selected)
        self.assertIn(100, selected)
        self.assertNotIn(101, selected)
        self.assertTrue(all(1 <= size <= 100 for size in selected))


if __name__ == "__main__":
    unittest.main()
