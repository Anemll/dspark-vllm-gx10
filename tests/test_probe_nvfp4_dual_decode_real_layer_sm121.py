# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from __future__ import annotations

import pathlib
import sys
import unittest
from types import SimpleNamespace


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import probe_nvfp4_dual_decode_real_layer_sm121 as probe  # noqa: E402


class _Storage:
    def __init__(self, pointer: int, size: int) -> None:
        self._pointer = pointer
        self._size = size

    def data_ptr(self) -> int:
        return self._pointer

    def nbytes(self) -> int:
        return self._size


class _Tensor:
    def __init__(self, pointer: int, storage_pointer: int, size: int) -> None:
        self._pointer = pointer
        self._storage = _Storage(storage_pointer, size)

    def data_ptr(self) -> int:
        return self._pointer

    def untyped_storage(self) -> _Storage:
        return self._storage


class DualDecodeRealLayerProbeTests(unittest.TestCase):
    def test_parser_is_pinned_to_the_serving_cutover_matrix(self) -> None:
        args = probe.build_parser().parse_args(
            ["--layer-file", "layer.safetensors", "--output", "result.json"]
        )
        self.assertEqual(args.m, (1, 2, 4, 8))
        self.assertEqual(args.routing, "balanced")
        self.assertEqual(args.numeric_min_cosine, 0.98)
        self.assertEqual(args.numeric_max_nrmse, 0.25)
        self.assertEqual(probe.require_exact_m(args.m), (1, 2, 4, 8))
        with self.assertRaisesRegex(ValueError, "pinned"):
            probe.require_exact_m((1, 4))

    def test_environment_requires_exact_dual_and_modelopt_tc_contract(self) -> None:
        self.assertEqual(
            probe.require_environment(dict(probe.EXPECTED_ENVIRONMENT)),
            probe.EXPECTED_ENVIRONMENT,
        )
        wrong = dict(probe.EXPECTED_ENVIRONMENT)
        wrong["B12X_W4A16_SMALL_M_DIRECT"] = "1"
        with self.assertRaisesRegex(RuntimeError, "environment drifted"):
            probe.require_environment(wrong)

    def test_sidecar_bytes_are_exact_for_dsv4_tp2_rank(self) -> None:
        shape = SimpleNamespace(
            num_experts=256,
            hidden_size=4096,
            intermediate_size_per_rank=1024,
        )
        observed = probe.expected_sidecar_bytes(shape)
        self.assertEqual(observed["w13_e8m0_k32"], 67_108_864)
        self.assertEqual(observed["w2_e8m0_k32"], 33_554_432)
        self.assertEqual(observed["w13_global_fp32"], 1_024)
        self.assertEqual(observed["w2_global_fp32"], 1_024)
        self.assertEqual(observed["total"], 100_665_344)

    def test_storage_identity_requires_data_storage_and_size(self) -> None:
        source = _Tensor(100, 80, 4096)
        same = _Tensor(100, 80, 4096)
        view_drift = _Tensor(104, 80, 4096)
        copy = _Tensor(200, 180, 4096)
        self.assertTrue(all(probe.tensor_storage_identity(source, same).values()))
        self.assertFalse(
            probe.tensor_storage_identity(source, view_drift)["same_data_ptr"]
        )
        self.assertFalse(
            probe.tensor_storage_identity(source, copy)["same_storage_ptr"]
        )

    def test_branch_contract_is_cutlass_m1_and_w4a16_m2_m4_m8(self) -> None:
        rows = [
            {"m": 1, "candidate_branch": "flashinfer_cutlass"},
            {"m": 2, "candidate_branch": "w4a16"},
            {"m": 4, "candidate_branch": "w4a16"},
            {"m": 8, "candidate_branch": "w4a16"},
        ]
        self.assertTrue(probe.evaluate_branch_contract(rows)["passed"])
        rows[-1]["candidate_branch"] = "flashinfer_cutlass"
        self.assertFalse(probe.evaluate_branch_contract(rows)["passed"])
        self.assertFalse(probe.evaluate_branch_contract(rows[:-1])["passed"])

    def test_runner_calls_actual_dual_apply_and_does_not_patch_implementation(self) -> None:
        source = pathlib.Path(probe.__file__).read_text()
        self.assertIn("NvFp4CutlassW4A16DualExperts", source)
        self.assertIn("dual.initialize_prepared_w4a16_decode(layer)", source)
        self.assertIn("dual.apply(", source)
        self.assertIn("hidden_states=x", source)
        self.assertIn("a1q_scale=None", source)
        self.assertIn("install_compile_trace", source)
        self.assertIn("evaluate_modelopt_tc_contract", source)
        self.assertNotIn("from scripts import patch_", source)
        self.assertNotIn("apply_patch", source)


if __name__ == "__main__":
    unittest.main()
