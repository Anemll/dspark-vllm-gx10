from __future__ import annotations

import ast
from pathlib import Path
import unittest

from benchmarks.probe_nvfp4_sparse_group_gemm_sm121 import (
    PHYSICAL_GROUPS,
    build_sparse_group_layout,
    gate_failures,
    group_scale_row_offset,
    select_sparse_groups,
)


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "benchmarks/probe_nvfp4_sparse_group_gemm_sm121.py"


class SparseGroupGemmContractTest(unittest.TestCase):
    def test_default_c4_layout_has_256_groups_and_232_zero_problems(self) -> None:
        active = select_sparse_groups(24)
        layout = build_sparse_group_layout(
            groups=PHYSICAL_GROUPS,
            active_groups=active,
        )
        self.assertEqual(
            active,
            (
                11,
                14,
                17,
                20,
                48,
                51,
                54,
                57,
                85,
                88,
                91,
                94,
                122,
                125,
                128,
                159,
                162,
                165,
                196,
                199,
                202,
                233,
                236,
                239,
            ),
        )
        self.assertEqual(len(layout.m_indptr), 257)
        self.assertEqual(layout.output_rows, 96)
        self.assertEqual(layout.active_count, 24)
        self.assertEqual(layout.inactive_count, 232)
        self.assertEqual(layout.repeated_offset_count, 232)
        self.assertEqual(layout.leading_zero_groups, 11)
        self.assertEqual(layout.trailing_zero_groups, 16)
        self.assertEqual(layout.scale_storage_rows, 32_512)

    def test_sparse_indptr_and_scale_bases_match_flashinfer_formula(self) -> None:
        layout = build_sparse_group_layout(
            groups=8,
            active_groups=(1, 3, 6),
        )
        self.assertEqual(layout.lengths, (0, 4, 0, 4, 0, 0, 4, 0))
        self.assertEqual(layout.m_indptr, (0, 0, 4, 4, 8, 8, 8, 12, 12))
        self.assertEqual(
            layout.scale_base_rows,
            tuple(
                group_scale_row_offset(group, layout.m_indptr[group])
                for group in range(8)
            ),
        )
        for group in layout.active_groups:
            self.assertLessEqual(
                layout.scale_base_rows[group] + 128,
                layout.scale_storage_rows,
            )

    def test_sparse_layout_rejects_ambiguous_or_unsafe_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly ordered"):
            build_sparse_group_layout(groups=8, active_groups=(3, 1))
        with self.assertRaisesRegex(ValueError, "strictly ordered"):
            build_sparse_group_layout(groups=8, active_groups=(1, 1))
        with self.assertRaisesRegex(ValueError, "inactive"):
            build_sparse_group_layout(groups=3, active_groups=(0, 1, 2))
        with self.assertRaisesRegex(ValueError, "multiple of four"):
            build_sparse_group_layout(
                groups=8,
                active_groups=(1, 3),
                rows_per_active=2,
            )
        with self.assertRaisesRegex(ValueError, "coprime"):
            select_sparse_groups(3, groups=8, stride=2)

    def test_gate_is_fail_closed(self) -> None:
        common = dict(
            finite=True,
            nonzero_real_rows=True,
            padded_rows_zero=True,
            sparse_matches_independent=True,
            inactive_poison_invariant=True,
            cosine=0.99,
            normalized_rmse=0.10,
            minimum_cosine=0.97,
            maximum_nrmse=0.25,
        )
        self.assertEqual(gate_failures(**common), [])
        each_failure = (
            ("finite", False, "finite_output"),
            ("nonzero_real_rows", False, "nonzero_real_rows"),
            ("padded_rows_zero", False, "zero_padded_rows"),
            (
                "sparse_matches_independent",
                False,
                "sparse_matches_independent",
            ),
            (
                "inactive_poison_invariant",
                False,
                "inactive_poison_invariant",
            ),
            ("cosine", 0.96, "minimum_cosine"),
            ("normalized_rmse", 0.26, "maximum_normalized_rmse"),
        )
        for field, value, expected in each_failure:
            with self.subTest(field=field):
                failed = gate_failures(**{**common, field: value})
                self.assertIn(expected, failed)

    def test_probe_is_no_model_and_lazy_imports_gpu_dependencies(self) -> None:
        source = PROBE.read_text()
        module = ast.parse(source)
        top_level_imports = {
            alias.name
            for node in module.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        top_level_from_imports = {
            node.module
            for node in module.body
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        self.assertNotIn("torch", top_level_imports)
        self.assertNotIn("flashinfer", top_level_imports)
        self.assertNotIn("safetensors", top_level_imports)
        self.assertNotIn("torch", top_level_from_imports)
        self.assertNotIn("flashinfer", top_level_from_imports)
        self.assertNotIn("safetensors", top_level_from_imports)
        self.assertNotIn("safe_open", source)
        self.assertNotIn("layer_file", source)
        self.assertIn("checkpoint_loaded", source)

    def test_probe_contains_sparse_and_poison_hardware_oracles(self) -> None:
        source = PROBE.read_text()
        ast.parse(source)
        self.assertIn("group_gemm_nvfp4_nt_groupwise(", source)
        self.assertIn("m_indptr", source)
        self.assertIn("inactive_poison_bitwise_invariant", source)
        self.assertIn("sparse_vs_independent_bitwise_equal", source)
        self.assertIn("zero_length_problem_count", source)
        self.assertEqual(PHYSICAL_GROUPS, 256)


if __name__ == "__main__":
    unittest.main()
