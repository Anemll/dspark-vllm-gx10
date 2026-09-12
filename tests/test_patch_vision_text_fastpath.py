# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "scripts/patch-vision-text-fastpath.py"

FIXTURE = """class Wrapper:
    def embed_input_ids(self, input_ids, multimodal_embeddings=None):
        inputs_embeds = self.language_model.embed_input_ids(input_ids)

        if self.image_start is not None:
            sentinel_mask = image_sentinel_mask(input_ids)
            inputs_embeds = torch.where(sentinel_mask, table[idx], inputs_embeds)

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        return merge(inputs_embeds, multimodal_embeddings)
"""


class PatchVisionTextFastPath(unittest.TestCase):
    def test_moves_text_return_before_sentinel_work(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "vl_model.py"
            target.write_text(FIXTURE)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            subprocess.run(
                [
                    sys.executable,
                    str(PATCHER),
                    "--target",
                    str(target),
                    "--expected-sha256",
                    digest,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            patched = target.read_text()
            self.assertLess(
                patched.index("multimodal_embeddings is None"),
                patched.index("image_sentinel_mask"),
            )
            self.assertEqual(patched.count("multimodal_embeddings is None"), 1)

    def test_rejects_an_unknown_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "vl_model.py"
            target.write_text(FIXTURE)
            result = subprocess.run(
                [
                    sys.executable,
                    str(PATCHER),
                    "--target",
                    str(target),
                    "--expected-sha256",
                    "0" * 64,
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to patch unknown Vision wrapper", result.stderr)


if __name__ == "__main__":
    unittest.main()
