# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from benchmarks import probe_nvfp4_flashinfer_orphan_direct_sm121 as probe
from scripts import patch_flashinfer_orphan_direct_micro_dsv4 as patcher


class LiteralOrphanDirectProbeTests(unittest.TestCase):
    def test_pinned_identities(self) -> None:
        self.assertEqual(
            patcher.SOURCE_SHA256,
            "abfad363fae29d15c0c2af127a54b7bafe2ae667c08ff976a2caf6d0828436b2",
        )
        self.assertEqual(
            probe.PORTED_SOURCE_SHA256,
            "ce223868f247c1abb097df2e59bf0a0ac8087924e290921e11faf9fa04e6754e",
        )
        self.assertEqual(probe.DEFAULT_MAXIMUM_GRAPH_MS, 0.682812)

    def test_kernel_semantics_are_exact_w4a4_dsv4(self) -> None:
        kwargs = probe.kernel_kwargs()
        self.assertEqual(kwargs["w13_layout"], "w13")
        self.assertEqual(kwargs["swiglu_limit"], 10.0)
        self.assertEqual(kwargs["activation"], "silu")
        self.assertFalse(kwargs["w4a16_mode"])
        self.assertTrue(kwargs["fast_math"])
        self.assertNotIn("scale_format", kwargs)

    def test_bad_source_fails_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            source = Path(tempdir) / "source.py"
            output = Path(tempdir) / "output.py"
            source.write_text("not the pinned orphan\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "orphan source drifted"):
                patcher.patch(source, output)
            self.assertFalse(output.exists())

    def test_argument_contract(self) -> None:
        good = argparse.Namespace(
            warmup=1,
            iters=1,
            repeats=1,
            numeric_min_cosine=0.98,
            numeric_max_nrmse=0.25,
            maximum_graph_ms=0.682812,
        )
        probe.validate_args(good)
        good.maximum_graph_ms = 0.0
        with self.assertRaisesRegex(ValueError, "maximum-graph-ms"):
            probe.validate_args(good)


if __name__ == "__main__":
    unittest.main()
