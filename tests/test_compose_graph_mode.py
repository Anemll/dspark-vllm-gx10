# SPDX-License-Identifier: MIT
"""Preserve explicit graph settings and the upstream absent-variable contract."""
import os
from pathlib import Path
import subprocess
import unittest


class ComposeGraphModeTests(unittest.TestCase):
    def test_graph_setting_reaches_runtime_without_changing_auto_selection(self):
        compose = (Path(__file__).resolve().parents[1] / "docker-compose.yml").read_text()
        self.assertIn(
            'VLLM_USE_BREAKABLE_CUDAGRAPH: "${VLLM_USE_BREAKABLE_CUDAGRAPH:-}"',
            compose,
        )
        prefix = next(line.strip() for line in compose.splitlines()
                      if line.strip().startswith('if [ -z "$${VLLM_USE_BREAKABLE'))
        script = prefix.replace("$$", "$")
        script += ' printf "%s" "${VLLM_USE_BREAKABLE_CUDAGRAPH-ABSENT}"'
        for value, expected in ((None, "ABSENT"), ("", "ABSENT"), ("0", "0"), ("1", "1")):
            with self.subTest(value=value):
                env = dict(os.environ)
                env.pop("VLLM_USE_BREAKABLE_CUDAGRAPH", None)
                if value is not None:
                    env["VLLM_USE_BREAKABLE_CUDAGRAPH"] = value
                result = subprocess.run(["bash", "-c", script], env=env,
                                        capture_output=True, text=True, check=True)
                self.assertEqual(result.stdout, expected)


if __name__ == "__main__":
    unittest.main()
