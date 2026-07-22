from __future__ import annotations

import math
import unittest

from benchmarks import probe_nvfp4_route_major_fc2_sm121 as probe
from benchmarks.nvfp4_route_major_fc2 import build_route_major_metadata


class SparseGroupLayoutTest(unittest.TestCase):
    def test_deterministic_active_experts_are_sparse_unique_and_bounded(self) -> None:
        selected = probe.select_active_experts(24)
        self.assertEqual(len(selected), 24)
        self.assertEqual(selected, tuple(sorted(set(selected))))
        self.assertGreater(selected[0], 0)
        self.assertLess(selected[-1], probe.PHYSICAL_EXPERTS)
        self.assertGreater(
            len(set(range(selected[0], selected[-1] + 1)) - set(selected)), 0
        )

        self.assertEqual(
            probe.select_active_experts(probe.PHYSICAL_EXPERTS),
            tuple(range(probe.PHYSICAL_EXPERTS)),
        )
        with self.assertRaisesRegex(ValueError, "coprime"):
            probe.select_active_experts(3, experts=8, stride=2)

    def test_sparse_indptr_and_scale_tiles_match_flashinfer_formula(self) -> None:
        layout = probe.build_sparse_group_layout(
            experts=8,
            active_experts=(1, 6),
            rows_per_active=4,
        )
        self.assertEqual(layout["lengths"], (0, 4, 0, 0, 0, 0, 4, 0))
        self.assertEqual(layout["m_indptr"], (0, 0, 4, 4, 4, 4, 4, 8, 8))
        self.assertEqual(layout["repeated_offset_count"], 6)

        bases = layout["scale_base_rows"]
        self.assertIsInstance(bases, tuple)
        assert isinstance(bases, tuple)
        indptr = layout["m_indptr"]
        assert isinstance(indptr, tuple)
        expected = tuple(
            probe.group_scale_row_offset(group, indptr[group])
            for group in range(8)
        )
        self.assertEqual(bases, expected)
        self.assertTrue(
            all(base % probe.SCALE_ROW_ALIGNMENT == 0 for base in bases)
        )
        self.assertEqual(
            layout["scale_storage_rows"],
            bases[-1] + probe.SCALE_ROW_ALIGNMENT,
        )
        for expert in (1, 6):
            self.assertLessEqual(
                bases[expert] + probe.SCALE_ROW_ALIGNMENT,
                layout["scale_storage_rows"],
            )

    def test_consecutive_and_final_active_groups_remain_in_bounds(self) -> None:
        layout = probe.build_sparse_group_layout(
            experts=8,
            active_experts=(1, 2, 7),
            rows_per_active=4,
        )
        bases = layout["scale_base_rows"]
        assert isinstance(bases, tuple)
        self.assertGreaterEqual(bases[2] - bases[1], 4)
        self.assertEqual(
            bases[7] + probe.SCALE_ROW_ALIGNMENT,
            layout["scale_storage_rows"],
        )

    def test_dispatch_metadata_matches_feasibility_probe_contract(self) -> None:
        active = probe.select_active_experts(24)
        feasibility = probe.build_sparse_group_layout(
            experts=probe.PHYSICAL_EXPERTS,
            active_experts=active,
            rows_per_active=probe.GROUP_ROW_ALIGNMENT,
        )
        dispatch = build_route_major_metadata(
            topk_ids=active,
            num_tokens=4,
            top_k=6,
            weight_experts=probe.PHYSICAL_EXPERTS,
            intermediate_size=probe.INTERMEDIATE_SIZE,
            hidden_size=probe.HIDDEN_SIZE,
        )
        self.assertEqual(dispatch.active_experts, active)
        self.assertEqual(dispatch.m_indptr, feasibility["m_indptr"])
        self.assertEqual(
            dispatch.local_scale_row_bases,
            tuple(feasibility["scale_base_rows"][expert] for expert in active),
        )
        self.assertEqual(
            dispatch.scale_storage_rows,
            feasibility["scale_storage_rows"],
        )

    def test_layout_rejects_ambiguous_or_invalid_routes(self) -> None:
        common = {"experts": 8, "rows_per_active": 4}
        for active in ((2, 1), (1, 1), (-1,), (8,)):
            with self.subTest(active=active), self.assertRaises(ValueError):
                probe.build_sparse_group_layout(
                    active_experts=active,
                    **common,
                )
        with self.assertRaisesRegex(ValueError, "multiple of four"):
            probe.build_sparse_group_layout(
                experts=8,
                active_experts=(1,),
                rows_per_active=1,
            )


class PreparedW2ScaleAlgebraTest(unittest.TestCase):
    def test_reconstructs_checkpoint_scales_and_pins_direct_alpha(self) -> None:
        result = probe.validate_prepared_w2_scale_algebra(
            (0.25, 0.25, 0.25),
            (8.0, 12.0, 16.0),
            experts=3,
        )
        self.assertTrue(result["grouped_gemm_alpha_is_g2_direct"])
        self.assertEqual(result["a2_gscale"], 0.25)
        self.assertEqual(result["reconstructed_checkpoint_input_scale"], 4.0)
        self.assertEqual(result["reconstructed_weight_scale_2_min"], 2.0)
        self.assertEqual(result["reconstructed_weight_scale_2_max"], 4.0)

    def test_rejects_drift_nonfinite_and_nonpositive_scales(self) -> None:
        invalid = (
            ((0.25, 0.5), (8.0, 8.0), "constant"),
            ((0.0, 0.0), (8.0, 8.0), "finite positive"),
            ((0.25, 0.25), (8.0, math.inf), "finite positive"),
            ((0.25,), (8.0,), "match physical experts"),
        )
        for a2, g2, message in invalid:
            with self.subTest(a2=a2, g2=g2), self.assertRaisesRegex(
                ValueError, message
            ):
                probe.validate_prepared_w2_scale_algebra(
                    a2,
                    g2,
                    experts=2,
                )


class ProbeCommandContractTest(unittest.TestCase):
    def test_defaults_pin_the_necessary_phase2_deadline(self) -> None:
        parser = probe.build_parser()
        args = parser.parse_args(
            [
                "--layer-file",
                "/tmp/layer.safetensors",
                "--output",
                "/tmp/out.json",
            ]
        )
        self.assertEqual(args.routes, 24)
        self.assertEqual(args.max_phase2_median_ms, 0.682812)


if __name__ == "__main__":
    unittest.main()
