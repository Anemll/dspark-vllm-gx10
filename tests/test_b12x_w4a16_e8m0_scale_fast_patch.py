# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from __future__ import annotations

import hashlib
import pathlib
import sys
import tempfile
import unittest
from itertools import product
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import probe_nvfp4_modelopt_tc_e8m0_scale_fast_sm121 as probe  # noqa: E402
from scripts import patch_b12x_w4a16_e8m0_scale_fast as patcher  # noqa: E402
from scripts import patch_b12x_w4a16_modelopt_tc_decode as policy_patcher  # noqa: E402


UPSTREAM_KERNEL = pathlib.Path(
    "/Users/anemll/SourceRelease/GITHUB/ML_playground/dspark-vllm-gx10/"
    ".build/b12x-upstream/b12x/moe/fused/w4a16/kernel.py"
)


def _pack_bytes(values: tuple[int, int, int, int]) -> int:
    return sum(value << (8 * index) for index, value in enumerate(values))


def _generic_finite_reference(packed: int) -> tuple[int, int]:
    values = tuple((packed >> (8 * index)) & 0xFF for index in range(4))
    if any(value > 247 for value in values):
        raise ValueError("finite fast path requires every E8M0 byte <=247")
    bf16 = tuple((value + 7) << 7 for value in values)
    return bf16[0] | (bf16[2] << 16), bf16[1] | (bf16[3] << 16)


def _finite_packed_oracle(packed: int) -> tuple[int, int]:
    values = tuple((packed >> (8 * index)) & 0xFF for index in range(4))
    if any(value > 247 for value in values):
        raise ValueError("finite fast path requires every E8M0 byte <=247")
    # PTX prmt selectors 0x4240 and 0x4341 widen byte pairs [0,2] and
    # [1,3] into independent 16-bit lanes.  The packed add cannot carry
    # between lanes because 247 + 7 = 254.
    pair02 = values[0] | (values[2] << 16)
    pair13 = values[1] | (values[3] << 16)
    return (
        ((pair02 + 0x00070007) << 7) & 0xFFFFFFFF,
        ((pair13 + 0x00070007) << 7) & 0xFFFFFFFF,
    )


class B12xE8m0ScaleFastPatchTests(unittest.TestCase):
    @staticmethod
    def _fixture() -> str:
        return "\n".join(anchor for anchor, _replacement, _label in patcher._REPLACEMENTS)

    def test_fast_converter_matches_generic_finite_branch_for_every_byte_lane(self) -> None:
        backgrounds = (
            (0, 0, 0, 0),
            (119, 120, 122, 123),
            (247, 0, 247, 0),
        )
        for lane in range(4):
            for background in backgrounds:
                for value in range(248):
                    values = list(background)
                    values[lane] = value
                    packed = _pack_bytes(tuple(values))
                    self.assertEqual(
                        _finite_packed_oracle(packed),
                        _generic_finite_reference(packed),
                    )

    def test_checkpoint_range_cartesian_and_boundaries_are_bit_exact(self) -> None:
        for values in product(range(119, 124), repeat=4):
            packed = _pack_bytes(values)
            self.assertEqual(
                _finite_packed_oracle(packed),
                _generic_finite_reference(packed),
            )
        for values in ((0, 0, 0, 0), (247, 247, 247, 247), (0, 247, 0, 247)):
            packed = _pack_bytes(values)
            self.assertEqual(
                _finite_packed_oracle(packed),
                _generic_finite_reference(packed),
            )
        with self.assertRaisesRegex(ValueError, "<=247"):
            _finite_packed_oracle(_pack_bytes((248, 119, 120, 121)))

    def test_patch_is_strictly_opt_in_and_decode_only(self) -> None:
        patched = patcher.patch_source(self._fixture())
        specialization = next(
            replacement
            for _anchor, replacement, label in patcher._REPLACEMENTS
            if label == "decode-only finite-fast specialization"
        )
        self.assertIn('weight_layout == "modelopt"', specialization)
        self.assertIn("self.scale_format_e8m0_k32", specialization)
        self.assertIn("not self.is_fp16", specialization)
        self.assertIn("self.direct_topk_routes", specialization)
        self.assertNotIn("and self.fused_topk_sum", specialization)
        self.assertIn("_e8m0_finite_fast_enabled()", specialization)
        self.assertIn("if cutlass.const_expr(self.e8m0_finite_fast):", patched)
        self.assertEqual(patched.count("self.e8m0_finite_fast,"), 1)
        self.assertNotIn("_load_b_registers_only", patched)
        self.assertNotIn("reuse_scale", patched)

    def test_inline_ptx_is_the_exact_six_instruction_finite_transform(self) -> None:
        intrinsic = next(
            replacement
            for _anchor, replacement, label in patcher._REPLACEMENTS
            if label == "finite-E8M0 environment and intrinsic"
        )
        expected = (
            "prmt.b32 q0, $2, 0, 0x4240;",
            "prmt.b32 q1, $2, 0, 0x4341;",
            "add.u32 q0, q0, 0x00070007;",
            "add.u32 q1, q1, 0x00070007;",
            "shl.b32 $0, q0, 7;",
            "shl.b32 $1, q1, 7;",
        )
        for instruction in expected:
            self.assertEqual(intrinsic.count(instruction), 1)
        self.assertNotIn("setp.", intrinsic)
        self.assertNotIn("selp.", intrinsic)

    def test_full_pinned_source_patch_has_exact_result_hash(self) -> None:
        source = UPSTREAM_KERNEL.read_text()
        policy = policy_patcher.patch_source(source)
        self.assertEqual(
            hashlib.sha256(policy.encode()).hexdigest(),
            patcher.PINNED_SOURCE_SHA256,
        )
        patched = patcher.patch_source(policy)
        self.assertEqual(
            hashlib.sha256(patched.encode()).hexdigest(),
            patcher.PATCHED_SOURCE_SHA256,
        )
        compile(patched, str(UPSTREAM_KERNEL), "exec", dont_inherit=True)

    def test_patch_is_fail_closed_and_deterministic(self) -> None:
        fixture = self._fixture()
        patched = patcher.patch_source(fixture)
        with self.assertRaisesRegex(RuntimeError, "PTX imports"):
            patcher.patch_source(patched)

        source = fixture.encode()
        expected = patched.encode()
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / "kernel.py"
            target.write_bytes(source)
            with (
                mock.patch.object(
                    patcher, "PINNED_SOURCE_SHA256", hashlib.sha256(source).hexdigest()
                ),
                mock.patch.object(
                    patcher,
                    "PATCHED_SOURCE_SHA256",
                    hashlib.sha256(expected).hexdigest(),
                ),
                mock.patch.object(sys, "argv", ["patcher", "--target", str(target)]),
            ):
                self.assertEqual(patcher.main(), 0)
            self.assertEqual(target.read_bytes(), expected)

    def test_finite_scale_report_contract_rejects_special_values(self) -> None:
        def report(maximum: int) -> dict[str, object]:
            row = {
                "passed": True,
                "exact_exponent_reconstruction": True,
                "e8m0_minimum_byte": 119,
                "e8m0_maximum_byte": maximum,
            }
            return {
                "backend_proof": {
                    probe.base.CANDIDATE: {
                        "conversion": {
                            "w13_scale_collapse": dict(row),
                            "w2_scale_collapse": dict(row),
                        }
                    }
                }
            }

        self.assertTrue(probe._finite_scale_contract(report(123))["passed"])
        self.assertTrue(probe._finite_scale_contract(report(247))["passed"])
        self.assertFalse(probe._finite_scale_contract(report(248))["passed"])

    def test_probe_ignores_direct_diagnostics_but_requires_w4a4_numeric(self) -> None:
        report = {
            "native_modelopt_tc_path_gate": {"passed": True},
            "memory_gate": {"passed": True},
            "results": [
                {
                    "m": 4,
                    "numeric_passed": {
                        f"{probe.base.CANDIDATE}_vs_w4a4": True,
                        f"{probe.base.CANDIDATE}_vs_direct": False,
                    },
                    "cuda_graph_status": {
                        probe.base.CANDIDATE: {"passed": True}
                    },
                    "activity": {probe.base.CANDIDATE: {"passed": True}},
                }
            ],
            "backend_proof": {
                probe.base.CANDIDATE: {
                    "conversion": {
                        "shared_fp4_payload_with_w4a4": {
                            "w13_same_data_ptr": True,
                            "w2_same_data_ptr": True,
                        }
                    }
                }
            },
            "failures": [
                {"kind": "numeric", "comparison": "candidate_vs_direct"},
                {"kind": "performance"},
            ],
        }
        self.assertEqual(probe._probe_failures(report), [])
        report["results"][0]["numeric_passed"][
            f"{probe.base.CANDIDATE}_vs_w4a4"
        ] = False
        self.assertEqual(
            probe._probe_failures(report),
            [{"kind": "candidate_numeric", "m": 4}],
        )

    def test_probe_pins_source_requests_fast_path_and_cc_geometry(self) -> None:
        self.assertEqual(probe.EXPECTED_KERNEL_SHA256, patcher.PATCHED_SOURCE_SHA256)
        args = probe.build_parser().parse_args(
            [
                "--layer-file",
                "/model/model-layer-00000.safetensors",
                "--output",
                "/tmp/e8m0-fast.json",
                "--finite-e8m0-fast",
            ]
        )
        self.assertTrue(args.finite_e8m0_fast)
        self.assertEqual(probe.WINNING_TILE, "c")
        self.assertEqual(probe.WINNING_GEOMETRY, (128, 64))


if __name__ == "__main__":
    unittest.main()
