# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import patch_flashinfer_b12x_all_task_head_prefetch as patcher


class FlashInferB12xAllTaskHeadPrefetchPatchTest(unittest.TestCase):
    def test_dispatch_is_default_off_and_micro_m2_to_m4_only(self) -> None:
        source = (
            patcher.DISPATCH_ENV_ANCHOR
            + patcher.DISPATCH_FACTORY_ARGS_ANCHOR
            + patcher.DISPATCH_FACTORY_GUARD_ANCHOR
            + patcher.DISPATCH_FACTORY_CACHE_ANCHOR
            + patcher.DISPATCH_FACTORY_KERNEL_ANCHOR
            + patcher.DISPATCH_POLICY_ANCHOR
            + patcher.DISPATCH_CALL_ANCHOR
        )
        patched = patcher.patch_dispatch_source(source)

        self.assertIn(
            '_ALL_TASK_HEAD_PREFETCH_ENV = '
            '"FLASHINFER_B12X_ALL_TASK_HEAD_PREFETCH"',
            patched,
        )
        self.assertIn(
            'os.environ.get(_ALL_TASK_HEAD_PREFETCH_ENV, "0")', patched
        )
        self.assertIn("_ALL_TASK_HEAD_PREFETCH and use_micro", patched)
        self.assertIn("2 <= num_tokens <= 4", patched)
        self.assertIn("not 2 <= m <= 4", patched)
        self.assertIn("all_task_head_prefetch,", patched)
        self.assertIn(
            "all_task_head_prefetch=all_task_head_prefetch", patched
        )

    def test_micro_prefetches_every_strided_task_head_before_pipelines(self) -> None:
        source = (
            patcher.MICRO_INIT_ARGS_ANCHOR
            + patcher.MICRO_INIT_ATTRS_ANCHOR
            + patcher.MICRO_PREPASS_ANCHOR
        )
        patched = patcher.patch_micro_kernel_source(source)

        self.assertIn(
            "self.all_task_head_prefetch = all_task_head_prefetch", patched
        )
        self.assertIn(
            "if cutlass.const_expr(self.all_task_head_prefetch):", patched
        )
        self.assertIn("_compact_static_get_work_tile(", patched)
        self.assertIn(
            "prefetch_work_linear_idx += prefetch_num_persistent_clusters",
            patched,
        )
        self.assertIn("cutlass.min(Int32(2), fc1_k_tile_cnt)", patched)
        self.assertIn("cutlass.min(Int32(2), output_tile_cnt)", patched)
        self.assertIn("cute.prefetch(\n                            tma_b_w13", patched)
        self.assertIn("tma_sfb_w13", patched)
        self.assertIn("tma_b_down", patched)
        self.assertIn("tma_sfb_down", patched)
        self.assertLess(
            patched.index("while prefetch_is_valid_tile:"),
            patched.index("prod_state = pipeline.make_pipeline_state("),
        )

    def test_prepass_does_not_replace_routing_or_arithmetic(self) -> None:
        patched = patcher.patch_micro_kernel_source(
            patcher.MICRO_INIT_ARGS_ANCHOR
            + patcher.MICRO_INIT_ATTRS_ANCHOR
            + patcher.MICRO_PREPASS_ANCHOR
        )
        inserted = patched[
            patched.index("if cutlass.const_expr(self.all_task_head_prefetch):") :
            patched.index("prod_state = pipeline.make_pipeline_state(")
        ]
        self.assertNotIn("cute.copy(", inserted)
        self.assertNotIn("cute.gemm(", inserted)
        self.assertNotIn("scatter_add", inserted)
        self.assertNotIn("token_map[", inserted)
        self.assertNotIn("row_counts[", inserted)

    def test_patch_rejects_anchor_drift(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "dispatcher env"):
            patcher.patch_dispatch_source("drift")
        with self.assertRaisesRegex(RuntimeError, "micro init args"):
            patcher.patch_micro_kernel_source("drift")

    def test_two_file_validation_happens_before_writes(self) -> None:
        dispatch_source = (
            patcher.DISPATCH_ENV_ANCHOR
            + patcher.DISPATCH_FACTORY_ARGS_ANCHOR
            + patcher.DISPATCH_FACTORY_GUARD_ANCHOR
            + patcher.DISPATCH_FACTORY_CACHE_ANCHOR
            + patcher.DISPATCH_FACTORY_KERNEL_ANCHOR
            + patcher.DISPATCH_POLICY_ANCHOR
            + patcher.DISPATCH_CALL_ANCHOR
        )
        micro_source = (
            patcher.MICRO_INIT_ARGS_ANCHOR
            + patcher.MICRO_INIT_ATTRS_ANCHOR
            + patcher.MICRO_PREPASS_ANCHOR
        )
        with tempfile.TemporaryDirectory() as directory:
            dispatch_path = Path(directory) / "moe_dispatch.py"
            micro_path = Path(directory) / "moe_micro_kernel.py"
            dispatch_path.write_text(dispatch_source, encoding="utf-8")
            micro_path.write_text(micro_source, encoding="utf-8")
            dispatch_before = dispatch_path.read_bytes()
            micro_before = micro_path.read_bytes()
            dispatch_sha = hashlib.sha256(dispatch_before).hexdigest()

            with (
                mock.patch.object(
                    patcher, "PINNED_DISPATCH_SHA256", dispatch_sha
                ),
                mock.patch.object(
                    patcher, "PINNED_MICRO_KERNEL_SHA256", "0" * 64
                ),
                self.assertRaisesRegex(RuntimeError, "microkernel SHA-256"),
            ):
                patcher.patch_paths(dispatch_path, micro_path)

            self.assertEqual(dispatch_path.read_bytes(), dispatch_before)
            self.assertEqual(micro_path.read_bytes(), micro_before)


if __name__ == "__main__":
    unittest.main()
