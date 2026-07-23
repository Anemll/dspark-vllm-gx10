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
    collect_tactic_inventory,
    inspect_cache,
    occupancy_valid_tactics,
    unsupported_tile_phase,
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

    def test_native_inventory_uses_combined_gemm2_ids_and_filters_occupancy(self) -> None:
        class Native:
            @staticmethod
            def get_gemm1_tactic_count() -> int:
                return 2

            @staticmethod
            def get_gemm2_tactic_count() -> int:
                return 3

            @staticmethod
            def get_tactic_occupancy(tactic: int) -> int:
                return {0: 1, 1: 0, 2: 2, 3: 0, 4: 1}[tactic]

        inventory = collect_tactic_inventory(Native())
        self.assertEqual(occupancy_valid_tactics(inventory, GEMM1_OP), (0,))
        self.assertEqual(occupancy_valid_tactics(inventory, GEMM2_OP), (2, 4))

    def test_only_exact_native_unsupported_tile_errors_are_skippable(self) -> None:
        self.assertEqual(
            unsupported_tile_phase(
                RuntimeError("Unsupported tile shape config 128128256 in Foo::gemm2(x)")
            ),
            GEMM2_OP,
        )
        self.assertIsNone(unsupported_tile_phase(RuntimeError("CUDA illegal access")))
        self.assertIsNone(
            unsupported_tile_phase(RuntimeError("Unsupported tile shape config without phase"))
        )


if __name__ == "__main__":
    unittest.main()
