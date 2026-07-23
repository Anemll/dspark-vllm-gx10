from __future__ import annotations

import unittest

from scripts import patch_flashinfer_cutlass_workspace_cache as patcher


class FlashInferCutlassWorkspaceCachePatchTests(unittest.TestCase):
    def test_inserts_one_runner_owned_workspace(self) -> None:
        source = (
            "prefix\n"
            + patcher.MEMBER_BEFORE
            + "middle\n"
            + patcher.ALLOC_BEFORE
            + "suffix\n"
        )

        result = patcher.patch_source(source)

        self.assertNotIn(patcher.MEMBER_BEFORE, result)
        self.assertNotIn(patcher.ALLOC_BEFORE, result)
        self.assertEqual(result.count("Tensor mRuntimeWorkspace;"), 1)
        self.assertEqual(
            result.count("mRuntimeWorkspaceBytes < total_workspace_size"), 1
        )
        self.assertIn("info.workspace = mRuntimeWorkspace;", result)

    def test_rejects_source_drift_and_double_patch(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "member anchor"):
            patcher.patch_source("unrelated")
        source = patcher.MEMBER_BEFORE + patcher.ALLOC_BEFORE
        with self.assertRaisesRegex(RuntimeError, "member anchor"):
            patcher.patch_source(patcher.patch_source(source))


if __name__ == "__main__":
    unittest.main()
