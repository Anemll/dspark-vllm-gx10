# SPDX-License-Identifier: MIT
from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "benchmarks/probe_dspark_variable_verifier_hw.py"


class HardwareProbeContractTests(unittest.TestCase):
    def test_probe_compiles_and_pins_required_assertions(self) -> None:
        source = PROBE.read_text(encoding="utf-8")
        ast.parse(source)
        for evidence in (
            '"forced_5_to_2_rows_3": True',
            '"forced_5_to_0_rows_1": True',
            '"off_or_full_rows_6": True',
            '"no_sentinel_reaches_target": True',
            '"rows3_dispatch_exact_not_6"',
            '"verify_scales_with_rows"',
        ):
            self.assertIn(evidence, source)

    def test_probe_uses_exact_dispatch_and_bounded_synthetic_kernel(self) -> None:
        source = PROBE.read_text(encoding="utf-8")
        self.assertIn("uniform_token_count=rows", source)
        self.assertIn("desc.num_tokens != rows", source)
        self.assertIn("torch.cuda.CUDAGraph()", source)
        self.assertIn("torch.bmm(routed, weights, out=output)", source)
        self.assertIn("SYNTHETIC_HIDDEN = 512", source)
        self.assertIn("TOP_K = 6", source)

    def test_scaling_gate_is_precommitted(self) -> None:
        source = PROBE.read_text(encoding="utf-8")
        self.assertIn("ordered[0] < ordered[-1] * 0.95", source)
        self.assertIn("spearman >= 0.60", source)
        self.assertIn('"--repeats", type=int, default=7', source)
        self.assertIn('"--iterations", type=int, default=200', source)


if __name__ == "__main__":
    unittest.main()
