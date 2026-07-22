# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.benchmark_nvfp4_cutlass_tactic_sweep_sm121 import (
    GEMM1_OP,
    GEMM2_OP,
    _positive_int_csv,
    build_matrix,
    cache_tactics,
    inspect_cache,
)


def _cache_key(op: str, m: int) -> str:
    return str((op, "MoERunner", ((m, 4096), (256, 1024)), ()))


class CutlassTacticSweepTests(unittest.TestCase):
    def test_positive_int_csv_deduplicates_in_order(self) -> None:
        self.assertEqual(_positive_int_csv("16,18,16"), (16, 18))

    def test_matrix_puts_service_configuration_first(self) -> None:
        matrix = build_matrix(
            (18,),
            (59,),
            (False,),
            service_pair=(16, 58),
            service_pdl=True,
        )
        self.assertEqual(matrix[0], (16, 58, True))
        self.assertEqual(matrix[1], (18, 59, False))

    def test_cache_inventory_selects_exact_m(self) -> None:
        payload = {
            "_metadata": {"gpu": "NVIDIA GB10"},
            _cache_key(GEMM1_OP, 4): ["MoERunner", 16],
            _cache_key(GEMM2_OP, 4): ["MoERunner", 58],
            _cache_key(GEMM1_OP, 8): ["MoERunner", 18],
            "not-a-python-key": ["MoERunner", 0],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "autotune_configs.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            inventory = inspect_cache(path, m=4)
        self.assertEqual(cache_tactics(inventory, GEMM1_OP), (16,))
        self.assertEqual(cache_tactics(inventory, GEMM2_OP), (58,))
        self.assertEqual(len(inventory["malformed_key_sha256"]), 1)
        self.assertEqual(inventory["total_config_entries"], 4)


if __name__ == "__main__":
    unittest.main()
