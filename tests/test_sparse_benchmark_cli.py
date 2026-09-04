# SPDX-License-Identifier: MIT
"""Reject unsafe/unbounded requests before GPU packages are imported."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks/benchmark_sparse_mla.py"


class SparseBenchmarkCliTests(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *args],
                              text=True, capture_output=True, timeout=5)

    def test_finite_and_bounded_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new.json"
            for args in (("--timeout", "nan"), ("--timeout", "inf"),
                         ("--trials", "101"), ("--repetitions", "1001"),
                         ("--min-free-gib", "nan"), ("--widths", "512,512")):
                with self.subTest(args=args):
                    result = self.invoke("--output", str(output), *args)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("invalid or unbounded", result.stderr)
                    self.assertFalse(output.exists())

    def test_refuses_existing_evidence_and_dangling_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence.json"
            output.write_text("keep")
            result = self.invoke("--output", str(output))
            self.assertEqual(result.returncode, 2)
            self.assertEqual(output.read_text(), "keep")
            link = Path(directory) / "dangling.json"
            link.symlink_to(Path(directory) / "absent")
            result = self.invoke("--output", str(link))
            self.assertEqual(result.returncode, 2)
            self.assertIn("refusing to overwrite", result.stderr)

    def test_reuse_needs_coordinator_image_evidence(self):
        result = self.invoke("--output", "unused.json", "--compiled-provenance", "old.json")
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires --image-id", result.stderr)


if __name__ == "__main__":
    unittest.main()
