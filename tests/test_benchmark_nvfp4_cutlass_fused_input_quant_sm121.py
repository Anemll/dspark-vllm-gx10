# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from __future__ import annotations

import inspect
import unittest

from benchmarks import benchmark_nvfp4_cutlass_fused_input_quant_sm121 as bench


def _trace(label: str) -> dict[str, object]:
    fused = label == bench.FUSED_PATH
    return {
        "label": label,
        "input_dtype": "torch.bfloat16" if fused else "torch.uint8",
        "input_shape": [4, 4096] if fused else [4, 2048],
        "input_sf_present": not fused,
        "w1_data_ptr": 101,
        "w2_data_ptr": 202,
        "quant_scale_count": 6,
        "tp_size": 2,
        "tp_rank": 0,
        "ep_size": 1,
        "ep_rank": 0,
        "use_deepseek_fp8_block_scale": False,
        "use_mxfp8_act_scaling": False,
        "use_w4_group_scaling": False,
    }


class FusedInputQuantContractTests(unittest.TestCase):
    def test_gate_is_pinned_to_m1_and_m4(self) -> None:
        self.assertEqual(bench._require_exact_m((1, 4)), (1, 4))
        for values in ((4, 1), (1,), (1, 2, 4), (1, 4, 4)):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "pinned"):
                    bench._require_exact_m(values)

    def test_good_trace_proves_bf16_input_sf_none(self) -> None:
        proof = bench.validate_path_contract(
            (_trace(bench.CURRENT_PATH), _trace(bench.FUSED_PATH)),
            m=4,
            hidden_size=4096,
            tp_rank=0,
            expected_w1_data_ptr=101,
            expected_w2_data_ptr=202,
        )
        self.assertTrue(proof["passed"])
        self.assertTrue(proof["candidate_internal_expand_quantization"])
        self.assertFalse(proof["python_fallback_observed"])

    def test_trace_rejects_input_scale_on_fused_path(self) -> None:
        fused = _trace(bench.FUSED_PATH)
        fused["input_sf_present"] = True
        with self.assertRaisesRegex(RuntimeError, "pre-quantized scales"):
            bench.validate_path_contract(
                (_trace(bench.CURRENT_PATH), fused),
                m=4,
                hidden_size=4096,
                tp_rank=0,
                expected_w1_data_ptr=101,
                expected_w2_data_ptr=202,
            )

    def test_trace_rejects_wrong_fused_dtype(self) -> None:
        fused = _trace(bench.FUSED_PATH)
        fused["input_dtype"] = "torch.uint8"
        with self.assertRaisesRegex(RuntimeError, "BF16"):
            bench.validate_path_contract(
                (_trace(bench.CURRENT_PATH), fused),
                m=4,
                hidden_size=4096,
                tp_rank=0,
                expected_w1_data_ptr=101,
                expected_w2_data_ptr=202,
            )

    def test_trace_rejects_fallback_or_extra_call(self) -> None:
        current = _trace(bench.CURRENT_PATH)
        fused = _trace(bench.FUSED_PATH)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            bench.validate_path_contract(
                (current, fused, dict(fused)),
                m=4,
                hidden_size=4096,
                tp_rank=0,
                expected_w1_data_ptr=101,
                expected_w2_data_ptr=202,
            )

    def test_trace_rejects_weight_storage_copy(self) -> None:
        fused = _trace(bench.FUSED_PATH)
        fused["w1_data_ptr"] = 303
        with self.assertRaisesRegex(RuntimeError, "w1_data_ptr"):
            bench.validate_path_contract(
                (_trace(bench.CURRENT_PATH), fused),
                m=4,
                hidden_size=4096,
                tp_rank=0,
                expected_w1_data_ptr=101,
                expected_w2_data_ptr=202,
            )

    def test_source_pins_direct_bf16_apply_without_input_scale(self) -> None:
        source = inspect.getsource(bench._make_launches)
        self.assertIn("hidden_states=x", source)
        self.assertIn("input_sf=None", source)
        self.assertIn("moe_kernel_quantize_input", source)
        self.assertIn('"external_quant_only"', source)

    def test_report_uses_milliseconds_not_token_rate(self) -> None:
        source = inspect.getsource(bench)
        self.assertIn('"timing_ms"', source)
        self.assertNotIn("tokens_per_second", source)
        self.assertNotIn("tok/s", source)


if __name__ == "__main__":
    unittest.main()
