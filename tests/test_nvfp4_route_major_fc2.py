from __future__ import annotations

import ast
from pathlib import Path
import unittest

from benchmarks.nvfp4_route_major_fc2 import (
    align_up,
    build_route_major_metadata,
    group_scale_row_offset,
)


ROOT = Path(__file__).resolve().parents[1]


class RouteMajorFC2ContractTest(unittest.TestCase):
    def test_balanced_m4_real_shape_is_bounded(self) -> None:
        # Twenty-four distinct routes is the worst balanced M=4/top-k=6 case.
        ids = tuple(range(0, 48, 2))
        metadata = build_route_major_metadata(
            topk_ids=ids,
            num_tokens=4,
            top_k=6,
        )
        self.assertEqual(metadata.active_experts, ids)
        self.assertEqual(metadata.row_counts, (1,) * 24)
        self.assertEqual(metadata.compact_topk_ids, tuple(range(24)))
        self.assertEqual(metadata.compact_row_ids, (0,) * 24)
        self.assertEqual(metadata.padded_rows, 96)
        self.assertEqual(metadata.phase1_packed_bytes, 49_152)
        self.assertEqual(metadata.grouped_output_bytes, 786_432)
        self.assertLess(metadata.handoff_bytes, 3 * 1024 * 1024)
        self.assertEqual(len(metadata.m_indptr), 257)

    def test_repeated_routes_have_stable_rows_and_global_bases(self) -> None:
        metadata = build_route_major_metadata(
            topk_ids=(8, 3, 8, 12, 8, 3, 12, 8),
            num_tokens=2,
            top_k=4,
            weight_experts=16,
        )
        self.assertEqual(metadata.active_experts, (8, 3, 12))
        self.assertEqual(metadata.row_counts, (4, 2, 2))
        self.assertEqual(metadata.compact_topk_ids, (0, 1, 0, 2, 0, 1, 2, 0))
        self.assertEqual(metadata.compact_row_ids, (0, 0, 1, 0, 2, 1, 1, 3))
        self.assertEqual(metadata.local_a_row_bases, (4, 0, 8))
        self.assertEqual(metadata.padded_rows, 12)
        self.assertEqual(metadata.m_indptr[3], 0)
        self.assertEqual(metadata.m_indptr[4], 4)
        self.assertEqual(metadata.m_indptr[8], 4)
        self.assertEqual(metadata.m_indptr[9], 8)
        self.assertEqual(metadata.m_indptr[12], 8)
        self.assertEqual(metadata.m_indptr[13], 12)

    def test_scale_offset_matches_sm120_descriptor_algebra(self) -> None:
        self.assertEqual(group_scale_row_offset(0, 0), 0)
        self.assertEqual(group_scale_row_offset(1, 0), 0)
        self.assertEqual(group_scale_row_offset(2, 4), 256)
        self.assertEqual(group_scale_row_offset(255, 96), 32_384)
        self.assertEqual(align_up(1, 4), 4)
        self.assertEqual(align_up(4, 4), 4)

    def test_fail_closed_shapes_and_routes(self) -> None:
        common = dict(num_tokens=2, top_k=2)
        with self.assertRaisesRegex(ValueError, "exactly"):
            build_route_major_metadata(topk_ids=(1, 2, 3), **common)
        with self.assertRaisesRegex(ValueError, "outside"):
            build_route_major_metadata(topk_ids=(1, 2, 3, 256), **common)
        with self.assertRaisesRegex(ValueError, "decode-only"):
            build_route_major_metadata(
                topk_ids=tuple(range(30)), num_tokens=5, top_k=6
            )

    def test_dispatcher_source_pins_phase_boundary(self) -> None:
        source = (ROOT / "benchmarks/nvfp4_route_major_fc2.py").read_text()
        ast.parse(source)
        self.assertEqual(source.count("class Phase1Handoff:"), 1)
        self.assertEqual(source.count("class RouteMajorFC2Dispatcher:"), 1)
        self.assertIn("group_gemm_nvfp4_nt_groupwise(", source)
        self.assertIn("alpha=self.g2_alpha", source)
        self.assertIn("reduce_route_major_output(", source)
        self.assertNotIn("atomic_add", source)

        reduction = (ROOT / "benchmarks/nvfp4_route_major_reduce.py").read_text()
        ast.parse(reduction)
        self.assertIn("for route in range(top_k):", reduction)
        self.assertIn("accum += router_weight * value", reduction)
        self.assertNotIn("tl.atomic", reduction)

    def test_full_route_major_probe_uses_two_grouped_gemms(self) -> None:
        full = (ROOT / "benchmarks/nvfp4_route_major_full.py").read_text()
        ast.parse(full)
        self.assertEqual(full.count("group_gemm_nvfp4_nt_groupwise("), 2)
        self.assertEqual(full.count("nvfp4_batched_quantize("), 2)
        self.assertIn("gather_route_inputs(", full)
        self.assertIn("oai_swiglu_gather(", full)
        self.assertIn("reduce_route_major_output(", full)
        self.assertIn("if not bool(torch.equal", full)

        ops = (ROOT / "benchmarks/nvfp4_route_major_ops.py").read_text()
        ast.parse(ops)
        self.assertIn("gate = tl.minimum(gate, limit)", ops)
        self.assertIn("up = tl.maximum(tl.minimum(up, limit), -limit)", ops)
        self.assertIn("gate * tl.sigmoid(alpha * gate) * (up + beta)", ops)

        probe = (ROOT / "benchmarks/probe_nvfp4_route_major_full_sm121.py").read_text()
        ast.parse(probe)
        self.assertIn("numeric_vs_accepted", probe)
        self.assertIn("route_major_over_accepted", probe)

    def test_phase1_adapter_pins_prepared_scale_and_capacity_contracts(self) -> None:
        adapter = (ROOT / "benchmarks/nvfp4_route_major_phase1.py").read_text()
        ast.parse(adapter)
        self.assertIn("max_rows=metadata.routed_rows", adapter)
        self.assertIn("(metadata.routed_rows, h, active)", adapter)
        self.assertIn("input_scales_are_reciprocal=True", adapter)
        self.assertIn("convert_sf_to_mma_layout(", adapter)
        self.assertIn("convert_sf_from_mma_layout(", adapter)
        self.assertIn("m=2 * i", adapter)
        self.assertIn("k=h", adapter)
        self.assertIn("num_groups=e", adapter)
        self.assertIn("w13_scale_storage=w13_scale_storage", adapter)
        self.assertGreaterEqual(adapter.count("torch.zeros("), 2)
        self.assertNotIn("w13_scale.data_ptr()", adapter)

        probe = (
            ROOT / "benchmarks/probe_nvfp4_route_major_phase1_sm121.py"
        ).read_text()
        ast.parse(probe)
        self.assertIn("phase1.launch()", probe)
        self.assertIn("phase2_launch()", probe)
        self.assertIn("numeric_vs_accepted", probe)
        self.assertIn("m4_max_ms", probe)
        self.assertIn("m2_max_regression", probe)


if __name__ == "__main__":
    unittest.main()
