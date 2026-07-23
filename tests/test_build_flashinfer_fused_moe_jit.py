from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "benchmarks" / "build_flashinfer_fused_moe_jit.py"


class FlashInferJitBuildDriverTests(unittest.TestCase):
    def test_driver_pins_full_build_and_rejects_aot(self) -> None:
        source = SCRIPT.read_text()
        ast.parse(source)
        self.assertIn("use_fast_build=False", source)
        self.assertIn("if spec.is_aot", source)
        self.assertIn('spec.name != "fused_moe_120"', source)
        self.assertIn("--expected-runner-header-sha256", source)
        self.assertIn('FLASHINFER_JIT_DEBUG") != "0"', source)
        self.assertIn('os.environ.get("MAX_JOBS", "0")', source)
        self.assertIn("SO_SHA256=", source)


if __name__ == "__main__":
    unittest.main()
