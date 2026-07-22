# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import unittest

from scripts import patch_flashinfer_b12x_cooperative_fc2 as cooperative_patch


ROOT = Path(__file__).resolve().parents[1]
SCRATCH_ROOT = ROOT.parent / "dspark-nvfp4-a4w4" / ".local" / "source"
ACCEPTED_DISPATCH = SCRATCH_ROOT / "mac-tune" / "moe_dispatch_base.py"
ACCEPTED_MICRO = SCRATCH_ROOT / "moe_micro_kernel_646.py"


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _class_method(tree: ast.AST, cls: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == name:
                    return child
    raise AssertionError(f"missing {cls}.{name}")


@unittest.skipUnless(
    ACCEPTED_DISPATCH.is_file() and ACCEPTED_MICRO.is_file(),
    "accepted 646 scratch sources are not present",
)
class FlashInferB12xCooperativeFc2PatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dispatch_source = ACCEPTED_DISPATCH.read_text(encoding="utf-8")
        cls.micro_source = ACCEPTED_MICRO.read_text(encoding="utf-8")
        cls.dispatch_patched = cooperative_patch.patch_dispatch_source(
            cls.dispatch_source
        )
        cls.micro_patched = cooperative_patch.patch_micro_source(cls.micro_source)

    def test_accepted_inputs_are_exactly_pinned(self) -> None:
        self.assertEqual(
            cooperative_patch._sha256(ACCEPTED_DISPATCH.read_bytes()),
            cooperative_patch.ACCEPTED_DISPATCH_SHA256,
        )
        self.assertEqual(
            cooperative_patch._sha256(ACCEPTED_MICRO.read_bytes()),
            cooperative_patch.ACCEPTED_MICRO_SHA256,
        )

    def test_results_parse_and_compile_as_python(self) -> None:
        ast.parse(self.dispatch_patched, filename="patched-moe_dispatch.py")
        ast.parse(self.micro_patched, filename="patched-moe_micro_kernel.py")
        compile(self.dispatch_patched, "patched-moe_dispatch.py", "exec")
        compile(self.micro_patched, "patched-moe_micro_kernel.py", "exec")

    def test_dispatch_gate_is_exact_m4_and_part_of_cache_key(self) -> None:
        text = self.dispatch_patched
        for condition in (
            "and m == 4",
            "and num_topk == 6",
            "and k == 4096",
            "and n == 1024",
            "and state_E == 256",
            "and weight_E == 256",
            "and mma_tiler_mn == (64, 128)",
            "and is_gated_activation(activation)",
        ):
            self.assertIn(condition, text)
        self.assertIn("cooperative_fc2,\n        activation,", text)
        self.assertIn("cooperative_fc2=cooperative_fc2", text)
        self.assertIn(
            'FLASHINFER_B12X_COOPERATIVE_FC2_M4',
            text,
        )
        self.assertIn('not in {"0", "1"}', text)

    def test_m1_scheduler_and_route_policy_are_unchanged(self) -> None:
        base_micro = ast.parse(self.micro_source)
        candidate_micro = ast.parse(self.micro_patched)
        self.assertEqual(
            ast.dump(
                _function(base_micro, "_compact_unique_get_work_tile"),
                include_attributes=False,
            ),
            ast.dump(
                _function(candidate_micro, "_compact_unique_get_work_tile"),
                include_attributes=False,
            ),
        )

        base_dispatch = ast.parse(self.dispatch_source)
        candidate_dispatch = ast.parse(self.dispatch_patched)
        base_launch = _function(base_dispatch, "launch_sm120_static_moe")
        candidate_launch = _function(candidate_dispatch, "launch_sm120_static_moe")
        self.assertEqual(
            ast.dump(base_launch, include_attributes=False),
            ast.dump(candidate_launch, include_attributes=False),
        )

    def test_cooperative_weight_and_scale_copies_match_contract(self) -> None:
        text = self.micro_patched
        self.assertIn("copy_bits = 128", text)
        self.assertIn("cpasync.CopyG2SOp", text)
        self.assertIn("cooperative_threads = self.num_mma_warps", text)
        self.assertIn("cute.nvgpu.CopyUniversalOp()", text)
        self.assertIn("directB_down", text)
        self.assertIn("directSFB_down", text)
        self.assertIn("_cpasync_copy_2d", text)
        self.assertIn("_scale_copy_2d", text)
        self.assertNotIn("Int32(directSFB_down.shape[0])", text)
        self.assertGreaterEqual(text.count("Int32(directB_down.shape[0])"), 2)
        self.assertIn("cute.arch.cp_async_commit_group()", text)
        self.assertIn("cute.arch.cp_async_wait_group(0)", text)
        self.assertIn('cute.arch.fence_proxy("async.shared", space="cta")', text)
        self.assertIn("self.epilog_sync_barrier.arrive_and_wait()", text)

    def test_dma_producer_is_disabled_only_in_opt_in_branch(self) -> None:
        text = self.micro_patched
        # Original TMA path remains present for the accepted fallback.
        self.assertIn("phase2_pipeline.producer_acquire(phase2_prod_state)", text)
        self.assertIn("tma_b_down", text)
        self.assertIn("tma_sfb_down", text)
        # The loader producer and tail are both compile-time skipped.
        self.assertGreaterEqual(
            text.count("if cutlass.const_expr(not self.cooperative_fc2):"),
            3,
        )
        self.assertIn(
            "if cutlass.const_expr(not self.cooperative_fc2):\n"
            "                phase2_pipeline.producer_tail(phase2_prod_state)",
            text,
        )

    def test_constructor_rejects_m1_and_noncanonical_tile(self) -> None:
        init = _class_method(ast.parse(self.micro_patched), "MoEMicroKernel", "__init__")
        init_text = ast.unparse(init)
        self.assertIn("if self.single_token", init_text)
        self.assertIn("self.tile_shape_mnk != (64, 128, 128)", init_text)
        self.assertIn("self.sf_vec_size != 16", init_text)

    def test_main_rejects_hash_drift_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dispatch = Path(tmp) / "moe_dispatch.py"
            micro = Path(tmp) / "moe_micro_kernel.py"
            dispatch.write_text(self.dispatch_source, encoding="utf-8")
            micro.write_text(self.micro_source + "# drift\n", encoding="utf-8")
            original_dispatch = dispatch.read_bytes()
            original_micro = micro.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "accepted microkernel"):
                cooperative_patch.patch_paths(dispatch, micro)
            self.assertEqual(dispatch.read_bytes(), original_dispatch)
            self.assertEqual(micro.read_bytes(), original_micro)

    def test_duplicate_anchor_fails_closed(self) -> None:
        duplicate = (
            self.micro_source
            + "\n"
            + cooperative_patch._MICRO_DMA_TAIL_ANCHOR
        )
        with self.assertRaisesRegex(RuntimeError, "DMA FC2 tail guard"):
            cooperative_patch.patch_micro_source(duplicate)


if __name__ == "__main__":
    unittest.main()
