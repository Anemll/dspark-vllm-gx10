# SPDX-License-Identifier: MIT
from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "scripts/probe_deepseek_v4_mhc_warmup.py"


class DeepseekV4MhcWarmupProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = PROBE_PATH.read_text()
        self.tree = ast.parse(self.source)

    def test_probe_calls_exact_deepgemm_hc_kernel(self) -> None:
        self.assertIn("tf32_hc_prenorm_gemm(", self.source)
        self.assertIn("_select_mhc_split_representatives(", self.source)
        self.assertIn("torch.cuda.synchronize(device)", self.source)

    def test_probe_proves_bind_and_hot_reuse(self) -> None:
        self.assertIn('"phase": "bind" if repeat == 0 else "reuse"', self.source)
        self.assertIn("if args.repeats < 2", self.source)
        self.assertIn("coverage drift", self.source)

    def test_probe_checks_finite_zero_output(self) -> None:
        self.assertIn("torch.isfinite(out).all()", self.source)
        self.assertIn("torch.count_nonzero(out)", self.source)


if __name__ == "__main__":
    unittest.main()
