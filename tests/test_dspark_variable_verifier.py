# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "overlay/vllm/v1/worker/gpu/spec_decode/dspark/variable_verifier.py"
)
PATCHER_PATH = ROOT / "scripts/patch_dspark_variable_verifier.py"
UPSTREAM_ROOT = Path(
    "/Users/anemll/SourceRelease/GITHUB/ML_playground/dspark-vllm-gx10/"
    ".build/vllm-upstream"
)
SPEC = importlib.util.spec_from_file_location("dspark_variable_verifier", MODULE_PATH)
assert SPEC and SPEC.loader
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class PhysicalCompactionTests(unittest.TestCase):
    def test_two_requests_shrink_to_distinct_physical_rows(self) -> None:
        output = SimpleNamespace(
            scheduled_spec_decode_tokens={"a": [-1] * 5, "b": [-1] * 5},
            num_scheduled_tokens={"a": 6, "b": 6},
            total_num_scheduled_tokens=12,
        )
        invalid = verifier.compact_scheduler_output_for_variable_drafts(
            output,
            ["a", "b"],
            [[10, 11, -1, -1, -1], [-1, -1, -1, -1, -1]],
        )
        self.assertEqual(invalid, {"a": 3, "b": 5})
        self.assertEqual(output.scheduled_spec_decode_tokens, {"a": [10, 11]})
        self.assertEqual(output.num_scheduled_tokens, {"a": 3, "b": 1})
        self.assertEqual(output.total_num_scheduled_tokens, 4)

    def test_full_prefix_preserves_six_target_rows(self) -> None:
        output = SimpleNamespace(
            scheduled_spec_decode_tokens={"a": [-1] * 5},
            num_scheduled_tokens={"a": 6},
            total_num_scheduled_tokens=6,
        )
        invalid = verifier.compact_scheduler_output_for_variable_drafts(
            output, ["a"], [[10, 11, 12, 13, 14]]
        )
        self.assertEqual(invalid, {})
        self.assertEqual(
            output.scheduled_spec_decode_tokens["a"], [10, 11, 12, 13, 14]
        )
        self.assertEqual(output.num_scheduled_tokens["a"], 6)
        self.assertEqual(output.total_num_scheduled_tokens, 6)

    def test_partial_scheduler_budget_caps_longer_proposal(self) -> None:
        output = SimpleNamespace(
            scheduled_spec_decode_tokens={"a": [-1] * 3},
            num_scheduled_tokens={"a": 4},
            total_num_scheduled_tokens=4,
        )
        invalid = verifier.compact_scheduler_output_for_variable_drafts(
            output, ["a"], [[10, 11, 12, 13, 14]]
        )
        self.assertEqual(invalid, {})
        self.assertEqual(output.scheduled_spec_decode_tokens["a"], [10, 11, 12])

    def test_holes_and_missing_request_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-contiguous"):
            verifier.trim_invalid_draft_tail([1, -1, 3, -1])
        output = SimpleNamespace(
            scheduled_spec_decode_tokens={"missing": [-1] * 5},
            num_scheduled_tokens={"missing": 6},
            total_num_scheduled_tokens=6,
        )
        with self.assertRaisesRegex(RuntimeError, "missing the prior proposal"):
            verifier.compact_scheduler_output_for_variable_drafts(
                output, ["other"], [[1, 2, 3, 4, 5]]
            )


@unittest.skipUnless(UPSTREAM_ROOT.exists(), "pinned upstream checkout unavailable")
class PinnedIntegrationPatchTests(unittest.TestCase):
    def test_patch_and_compile_all_four_integration_seams(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package_root = Path(tmp)
            paths = (
                "vllm/v1/worker/gpu/model_runner.py",
                "vllm/v1/outputs.py",
                "vllm/v1/core/sched/async_scheduler.py",
                "vllm/v1/worker/gpu/cudagraph_utils.py",
            )
            for relative in paths:
                destination = package_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(UPSTREAM_ROOT / relative, destination)
            subprocess.run(
                [
                    sys.executable,
                    str(PATCHER_PATH),
                    "--package-root",
                    str(package_root),
                ],
                check=True,
            )
            model_runner = (package_root / paths[0]).read_text()
            outputs = (package_root / paths[1]).read_text()
            scheduler = (package_root / paths[2]).read_text()
            cudagraph = (package_root / paths[3]).read_text()
            self.assertIn(
                "self.draft_tokens_handler.compact_scheduler_output(", model_runner
            )
            self.assertIn("confidence_invalid_spec_tokens=", model_runner)
            compact_pos = model_runner.index("compact_scheduler_output(")
            self.assertLess(
                compact_pos,
                model_runner.index("dispatch_cg_and_sync_dp(", compact_pos),
            )
            self.assertIn(
                "confidence_invalid_spec_tokens: dict[str, int] | None", outputs
            )
            self.assertIn("physical_invalid = (", scheduler)
            self.assertIn(
                "total = max(merged.get(req_id, 0), count)", scheduler
            )
            self.assertIn("variable_dspark = (", cudagraph)
            self.assertIn(
                "decode_query_lens = list(range(1, self.decode_query_len + 1))",
                cudagraph,
            )
            self.assertIn(
                "and rounded_num_reqs > 1",
                cudagraph,
            )
            for relative in paths:
                subprocess.run(
                    [sys.executable, "-m", "py_compile", str(package_root / relative)],
                    check=True,
                )

    def test_second_patch_attempt_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package_root = Path(tmp)
            paths = (
                "vllm/v1/worker/gpu/model_runner.py",
                "vllm/v1/outputs.py",
                "vllm/v1/core/sched/async_scheduler.py",
                "vllm/v1/worker/gpu/cudagraph_utils.py",
            )
            for relative in paths:
                destination = package_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(UPSTREAM_ROOT / relative, destination)
            command = [
                sys.executable,
                str(PATCHER_PATH),
                "--package-root",
                str(package_root),
            ]
            subprocess.run(command, check=True)
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("expected one anchor, found 0", second.stderr)


if __name__ == "__main__":
    unittest.main()
