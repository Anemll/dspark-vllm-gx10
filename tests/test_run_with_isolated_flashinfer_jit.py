from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "benchmarks" / "run_with_isolated_flashinfer_jit.py"


class IsolatedFlashInferJitWrapperTests(unittest.TestCase):
    def test_wrapper_is_syntax_valid_and_fail_closed(self) -> None:
        source = SCRIPT.read_text()
        ast.parse(source)
        self.assertIn("FLASHINFER_AOT_DIR = ns.empty_aot_dir.resolve()", source)
        self.assertIn("if any(ns.empty_aot_dir.iterdir())", source)
        self.assertIn('run_name="__main__"', source)


if __name__ == "__main__":
    unittest.main()
