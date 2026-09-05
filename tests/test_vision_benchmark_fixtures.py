# SPDX-License-Identifier: MIT
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.prepare_vision_benchmark import main, render


@unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is an optional visual-fixture dependency")
class VisionFixtureTests(unittest.TestCase):
    def test_render_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory, "a.png"), Path(directory, "b.png")
            expected, question = render("chart", 512, "test", a)
            render("chart", 512, "test", b)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(expected, {"largest": "South", "total": 54})
            self.assertIn("JSON", question)

    def test_builder_manifest_and_cache_distinction(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixtures"
            with patch.object(sys, "argv", ["prepare", "--output-dir", str(output)]):
                main()
            quality = json.loads((output / "vision-correctness-v1.json").read_text())
            throughput = json.loads((output / "vision-throughput-v1.json").read_text())
            self.assertEqual(len(quality["cases"]), 5)
            self.assertEqual(len(throughput["cases"]), 15)
            self.assertTrue(quality["generator"]["pillow"])
            self.assertEqual(len(quality["generator"]["source_sha256"]), 64)
            self.assertTrue(all("expected_json" in c for c in quality["cases"]))
            self.assertTrue(all("expected_json" not in c for c in throughput["cases"]))
            for asset in quality["assets"]:
                path = output / asset["file"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), asset["sha256"])
                with Image.open(path) as image:
                    self.assertEqual(image.size, (asset["width"], asset["height"]))
            self.assertNotEqual((output / "ocr-1024-correctness.png").read_bytes(), (output / "ocr-1024-throughput.png").read_bytes())
            with patch.object(sys, "argv", ["prepare", "--output-dir", str(output)]):
                with self.assertRaises(SystemExit):
                    main()


if __name__ == "__main__":
    unittest.main()
