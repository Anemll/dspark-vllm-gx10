# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from benchmarks import probe_nvfp4_direct_micro_sm121 as probe


class DirectMicroProbeTest(unittest.TestCase):
    def test_defaults_pin_real_m4_balanced_gate(self) -> None:
        args = probe.build_parser().parse_args(
            ["--layer-file", "/tmp/layer", "--output", "/tmp/result.json"]
        )
        self.assertEqual(probe.M_VALUE, 4)
        self.assertEqual(probe.ROUTING, "balanced")
        self.assertEqual(args.tp_rank, 0)
        self.assertEqual(args.numeric_min_cosine, 0.98)
        self.assertEqual(args.numeric_max_nrmse, 0.25)
        self.assertEqual(args.minimum_speedup, 1.0)

    def test_exact_dsv4_geometry(self) -> None:
        geometry = probe.direct_geometry(4, 4096, 1024, 6)
        self.assertEqual(geometry.fc1_chunks, 16)
        self.assertEqual(geometry.fc1_tasks, 384)
        self.assertEqual(geometry.fc2_tasks, 256)
        self.assertEqual(geometry.fc2_n_chunks, 4)
        self.assertEqual(geometry.intermediate_u32, 12_288)
        self.assertEqual(geometry.intermediate_bytes, 49_152)
        self.assertEqual(geometry.integration_arena_f32, 110_592)
        self.assertEqual(geometry.barrier_slots_per_array, 88)
        self.assertEqual(geometry.barrier_bytes_total, 704)
        self.assertEqual(geometry.output_bytes, 32_768)

    def test_constructor_pins_clamped_up_gate_nvfp4(self) -> None:
        kwargs = probe.direct_kernel_kwargs()
        self.assertEqual(kwargs["activation"], "silu")
        self.assertEqual(kwargs["swiglu_limit"], 10.0)
        self.assertEqual(kwargs["w13_layout"], "w13")
        self.assertEqual(kwargs["scale_format"], "e4m3_k16")
        self.assertFalse(kwargs["w4a16_mode"])
        self.assertFalse(kwargs["dynamic_down_scale"])
        self.assertFalse(kwargs["share_input_across_experts"])
        self.assertFalse(kwargs["share_expert_scales"])

    def test_prepared_scale_algebra_is_raw_modelopt(self) -> None:
        algebra = probe.prepared_scale_algebra()
        self.assertEqual(algebra["a1_gscale"], "1 / w13_input_scale")
        self.assertEqual(
            algebra["g1_alphas"],
            "w13_weight_scale_2 * w13_input_scale",
        )
        self.assertEqual(
            algebra["raw_w13_weight_scale_2"],
            "g1_alphas * a1_gscale",
        )
        self.assertIn("no B12X bake", algebra["block_scales"])
        self.assertEqual(algebra["w13_order"], "up/w3 then gate/w1")

    def test_pins_audited_b12x_sources_and_launch_guard(self) -> None:
        self.assertEqual(
            probe.B12X_MICRO_SOURCE_SHA256,
            "67847d6365b3707b54e5d68a89655666350029aa550c5f74742084f264d2d980",
        )
        self.assertEqual(
            probe.B12X_TP_MOE_SOURCE_SHA256,
            "c2ca5aca4f9efd8ac8afb52909ef18410d1afd455d7e994debcd4e0bc13e019d",
        )
        self.assertEqual(probe.DIRECT_BLOCK_DIM, 512)
        source = Path(probe.__file__).read_text(encoding="utf-8")
        self.assertIn("MoEMicroKernelBackend(**direct_kernel_kwargs())", source)
        self.assertIn("_compiled_direct_micro_accepts_block_dim", source)
        self.assertIn("else 8", source)
        self.assertIn("kernel_bench.capture_graph", source)
        self.assertIn('pair=("b12x_direct", "flashinfer_cutlass")', source)

    def test_invalid_geometry_and_args_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            probe.direct_geometry(3, 4096, 1024, 6)
        args = probe.build_parser().parse_args(
            [
                "--layer-file",
                "/tmp/layer",
                "--output",
                "/tmp/result.json",
                "--iters",
                "0",
            ]
        )
        with self.assertRaisesRegex(ValueError, "iters"):
            probe._validate_args(args)

    def test_probe_help_is_cpu_only(self) -> None:
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "benchmarks" / "probe_nvfp4_direct_micro_sm121.py"),
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--layer-file", completed.stdout)
        self.assertNotIn("--m ", completed.stdout)


if __name__ == "__main__":
    unittest.main()
