# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class NvFp4ComposeConfigTests(unittest.TestCase):
    def test_moe_backend_is_a_reversible_compose_setting(self) -> None:
        compose = (REPO_ROOT / "docker-compose.yml").read_text()
        self.assertIn(
            "--moe-backend ${DSPARK_MOE_BACKEND:-flashinfer_b12x}",
            compose,
        )

    def test_rank_examples_pin_the_same_backend(self) -> None:
        values = []
        for relative_path in ("config/head.env.example", "config/worker.env.example"):
            lines = (REPO_ROOT / relative_path).read_text().splitlines()
            values.extend(
                line.split("=", 1)[1]
                for line in lines
                if line.startswith("DSPARK_MOE_BACKEND=")
            )
        self.assertEqual(values, ["flashinfer_b12x", "flashinfer_b12x"])


if __name__ == "__main__":
    unittest.main()
