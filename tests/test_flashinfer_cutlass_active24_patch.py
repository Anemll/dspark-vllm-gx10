# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import patch_flashinfer_cutlass_active24 as active24


class FlashInferCutlassActive24PatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kernel_source = "\n// synthetic boundary\n".join(
            (
                active24.ENV_ANCHOR,
                active24.SCALING_HELPER_ANCHOR,
                active24.FINALIZE_POINTER_ANCHOR,
                active24.NORMAL_SCALING_CALL_ANCHOR,
                active24.COMPACT_KERNEL_ANCHOR,
                active24.COMPUTE_SIGNATURE_ANCHOR,
                active24.COMPUTE_LAUNCH_ANCHOR,
                active24.SETUP_GATE_ANCHOR,
                active24.SETUP_COMPUTE_CALL_ANCHOR,
                # Control-path markers intentionally not transformed.
                "expandInputRowsKernelLauncher(\n        input_activations",
                "expert_first_token_offset_, gemm2_tma_ws_input",
                "layout_info.ptr_weight[out_idx] = safe_inc_ptr(weights, expert",
            )
        )
        cls.header_source = "\n// synthetic boundary\n".join(
            (
                active24.HEADER_DISPATCH_CALL_ANCHOR,
                active24.HEADER_STATIC_DECL_ANCHOR,
            )
        )
        cls.patched_kernel = active24.patch_kernel_source(cls.kernel_source)
        cls.patched_header = active24.patch_header_source(cls.header_source)

    def test_pins_exact_flashinfer_sources_and_outputs(self) -> None:
        self.assertEqual(
            active24.PINNED_FLASHINFER_REVISION,
            "0472b9b3f2fba11b463f8526f390297d52a8aad7",
        )
        self.assertEqual(
            active24.PINNED_KERNEL_SHA256,
            "fd24f5f8234b0736f205dd2540f47dcaf90783a53c2fbbab66d0490c9494dbac",
        )
        self.assertEqual(
            active24.PINNED_HEADER_SHA256,
            "d5562b100214697950149718929fc6dd0bf6570ac79cd452d6da6c9df2ea6161",
        )
        self.assertEqual(
            active24.PATCHED_KERNEL_SHA256,
            "9a8fc3abd0d8bd3589adcf855c6f92e4534a6b914a5ab0a31bfa21339f12061b",
        )
        self.assertEqual(
            active24.PATCHED_HEADER_SHA256,
            "7be6f6f272b373157c796120e891da0b3389dee24d6348c92d909d21238a2cd8",
        )

    def test_gate_is_opt_in_and_exactly_c4_topk6_e256_ep1_nvfp4(self) -> None:
        source = self.patched_kernel
        for marker in (
            'getIntEnv("FLASHINFER_CUTLASS_ACTIVE24")',
            "value.value() == 0 || value.value() == 1",
            "getEnvCutlassActive24() && use_fp4",
            "!min_latency_mode && !use_lora && num_rows == 4",
            "expanded_num_rows == CUTLASS_ACTIVE24_GROUPS",
            "num_experts_per_node == 256",
            "parallelism_config.ep_size == 1",
            "start_expert == 0",
        ):
            self.assertIn(marker, source)
        self.assertIn(
            "requires TMA warp-specialized FC1 and FC2",
            source,
        )
        self.assertIn(
            "permuted_row_to_unpermuted_row, false, enable_pdl, stream",
            self.patched_header,
        )

    def test_compacts_only_tma_descriptors_to_fixed_graph_capacity(self) -> None:
        source = self.patched_kernel
        self.assertIn("constexpr int CUTLASS_ACTIVE24_GROUPS = 24", source)
        self.assertEqual(
            source.count("shape_info.num_groups = CUTLASS_ACTIVE24_GROUPS"), 2
        )
        self.assertIn("int const active_count = __shfl_sync", source)
        self.assertIn("assert(active_count <= CUTLASS_ACTIVE24_GROUPS)", source)
        self.assertIn("expert_first_token_offset[expert + 1]", source)
        self.assertNotIn("compact_expert_first_token_offset", source)
        self.assertIn(
            "compact_active24 ? 1 : (num_experts_per_node + threads - 1) / threads",
            source,
        )

    def test_actual_expert_selects_weights_while_compact_slot_selects_arrays(self) -> None:
        source = self.patched_kernel
        for marker in (
            "ptr_source_token_index[out_idx]",
            "ptr_router_scales[out_idx]",
            "ptr_bias[out_idx] = bias + gemm_n * expert",
            "ptr_weight[out_idx] = safe_inc_ptr(weights, expert",
            "fpX_block_scaling_factors_act[out_idx]",
            "getOffsetActivationSF(expert",
            "fpX_block_scaling_factors_weight[out_idx]",
            "getOffsetWeightSF(expert",
        ):
            self.assertIn(marker, source)

    def test_control_launch_and_original_offsets_remain_present(self) -> None:
        source = self.patched_kernel
        self.assertIn(
            ": &computeStridesTmaWarpSpecializedKernel<T, WeightType, OutputType, ScaleBiasType>",
            source,
        )
        self.assertIn(
            "expandInputRowsKernelLauncher(\n        input_activations",
            source,
        )
        self.assertIn(
            "permuted_row_to_unpermuted_row_, compact_active24, enable_pdl, stream);",
            source,
        )
        self.assertIn(
            "expert_first_token_offset_, gemm2_tma_ws_input",
            source,
        )

    def test_patch_tree_validates_both_files_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kernel = root / active24.KERNEL_RELATIVE_PATH
            header = root / active24.HEADER_RELATIVE_PATH
            kernel.parent.mkdir(parents=True)
            header.parent.mkdir(parents=True)
            kernel.write_text(self.kernel_source, encoding="utf-8")
            header.write_text("drift\n", encoding="utf-8")

            with mock.patch.object(
                active24,
                "PINNED_KERNEL_SHA256",
                active24._sha256(self.kernel_source.encode()),
            ):
                with self.assertRaisesRegex(RuntimeError, "runner header drifted"):
                    active24.patch_tree(root)
            self.assertEqual(kernel.read_text(encoding="utf-8"), self.kernel_source)
            self.assertEqual(header.read_text(encoding="utf-8"), "drift\n")

    def test_patch_tree_publishes_deterministic_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kernel = root / active24.KERNEL_RELATIVE_PATH
            header = root / active24.HEADER_RELATIVE_PATH
            kernel.parent.mkdir(parents=True, exist_ok=True)
            header.parent.mkdir(parents=True, exist_ok=True)
            kernel.write_text(self.kernel_source, encoding="utf-8")
            header.write_text(self.header_source, encoding="utf-8")
            kernel_before = active24._sha256(kernel.read_bytes())
            header_before = active24._sha256(header.read_bytes())
            kernel_after = active24._sha256(self.patched_kernel.encode())
            header_after = active24._sha256(self.patched_header.encode())

            with mock.patch.multiple(
                active24,
                PINNED_KERNEL_SHA256=kernel_before,
                PINNED_HEADER_SHA256=header_before,
                PATCHED_KERNEL_SHA256=kernel_after,
                PATCHED_HEADER_SHA256=header_after,
            ):
                result = active24.patch_tree(root)
            self.assertEqual(
                active24._sha256(kernel.read_bytes()), kernel_after
            )
            self.assertEqual(
                active24._sha256(header.read_bytes()), header_after
            )
            self.assertEqual(result["environment"], "FLASHINFER_CUTLASS_ACTIVE24")

    def test_anchor_drift_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "active24 environment contract"):
            active24.patch_kernel_source("drift")
        with self.assertRaisesRegex(RuntimeError, "public dispatch control"):
            active24.patch_header_source("drift")


if __name__ == "__main__":
    unittest.main()
