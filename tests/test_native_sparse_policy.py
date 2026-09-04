# SPDX-License-Identifier: MIT
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch


path = Path(__file__).resolve().parents[1] / "overlay/vllm/utils/dsv4_sparse_policy.py"
spec = importlib.util.spec_from_file_location("sparse_policy", path)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class NativeSparsePolicyTests(unittest.TestCase):
    def test_control_preserves_legacy(self):
        with patch.dict(os.environ, {"VLLM_DSV4_NATIVE_SPARSE_WIDTHS": "0"}):
            self.assertEqual(policy.dspark_sparse_width(128, 5, sm12x=True), 256)
            self.assertEqual(policy.sparse_decode_widths(), (128, 512, 1024))

    def test_native_only_on_sm12x(self):
        with patch.dict(os.environ, {"VLLM_DSV4_NATIVE_SPARSE_WIDTHS": "1"}):
            self.assertEqual(policy.dspark_sparse_width(128, 5, sm12x=True), 192)
            self.assertEqual(policy.dspark_sparse_width(128, 5, sm12x=False), 256)
            self.assertEqual(policy.dspark_sparse_width(128, 65, sm12x=True), 256)
            self.assertEqual(policy.sparse_decode_widths(), (128, 192, 256, 512, 1024))

    def test_fail_closed(self):
        with patch.dict(os.environ, {"VLLM_DSV4_NATIVE_SPARSE_WIDTHS": "auto"}):
            with self.assertRaises(ValueError):
                policy.native_sparse_widths_enabled()
        for window, draft in ((0, 5), (128, -1)):
            with self.assertRaises(ValueError):
                policy.dspark_sparse_width(window, draft, sm12x=True)


if __name__ == "__main__":
    unittest.main()
