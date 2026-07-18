# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "benchmarks" / "run_bounded_command.py"
SPEC = importlib.util.spec_from_file_location("run_bounded_command", PATH)
assert SPEC and SPEC.loader
bounded = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bounded
SPEC.loader.exec_module(bounded)


class BoundedCommandTests(unittest.TestCase):
    def test_returns_child_status(self):
        self.assertEqual(
            bounded.run_bounded([sys.executable, "-c", "raise SystemExit(7)"], 5, 1),
            7,
        )

    def test_timeout_returns_124_quickly(self):
        started = time.monotonic()
        self.assertEqual(
            bounded.run_bounded(
                [sys.executable, "-c", "import time; time.sleep(30)"], 0.1, 0.1
            ),
            bounded.TIMEOUT_EXIT_CODE,
        )
        self.assertLess(time.monotonic() - started, 2)

    def test_rejects_invalid_limits(self):
        with self.assertRaises(ValueError):
            bounded.run_bounded([sys.executable, "-c", "pass"], 0, 1)


if __name__ == "__main__":
    unittest.main()
