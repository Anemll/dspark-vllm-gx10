# SPDX-License-Identifier: MIT

import unittest

from benchmarks import benchmark_prepared_vs_abliterated_b12x_w4a16_sm121 as probe


class _FakeTensor:
    def __init__(self, shape, operations=()):
        self.shape = tuple(shape)
        self.operations = tuple(operations)

    def narrow(self, dim, start, length):
        shape = list(self.shape)
        shape[dim] = length
        return _FakeTensor(shape, self.operations + (("narrow", dim, start, length),))

    def contiguous(self):
        return _FakeTensor(self.shape, self.operations + (("contiguous",),))


class NativeTpSliceTests(unittest.TestCase):
    def test_w1_slices_output_rows(self):
        value = _FakeTensor((2048, 2048))
        sliced = probe.tp_slice_native_expert(
            value, family="w1.weight", tp_rank=1
        )
        self.assertEqual(sliced.shape, (1024, 2048))
        self.assertIn(("narrow", 0, 1024, 1024), sliced.operations)

    def test_w2_slices_packed_input_columns(self):
        value = _FakeTensor((4096, 1024))
        sliced = probe.tp_slice_native_expert(
            value, family="w2.weight", tp_rank=1
        )
        self.assertEqual(sliced.shape, (4096, 512))
        self.assertIn(("narrow", -1, 512, 512), sliced.operations)

    def test_w2_scale_slices_k32_columns(self):
        value = _FakeTensor((4096, 64))
        sliced = probe.tp_slice_native_expert(
            value, family="w2.scale", tp_rank=0
        )
        self.assertEqual(sliced.shape, (4096, 32))
        self.assertIn(("narrow", -1, 0, 32), sliced.operations)

    def test_unknown_family_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            probe.tp_slice_native_expert(
                _FakeTensor((2, 2)), family="bad", tp_rank=0
            )


class DecisionTests(unittest.TestCase):
    def _rows(self, converted_delta=0.0):
        return {
            m: {
                "converted_graph_ms": 1.0 + converted_delta,
                "abliterated_graph_ms": 1.0,
                "graph_passed": True,
                "activity_passed": True,
            }
            for m in probe.REQUIRED_M
        }

    def test_parity_accepts_three_percent_boundary(self):
        decision = probe.evaluate_parity(
            self._rows(0.03), maximum_absolute_delta=0.03
        )
        self.assertTrue(decision["passed"])

    def test_parity_rejects_slow_or_fast_drift(self):
        self.assertFalse(probe.evaluate_parity(self._rows(0.04))["passed"])
        self.assertFalse(probe.evaluate_parity(self._rows(-0.04))["passed"])

    def test_decision_requires_graph_and_activity(self):
        rows = self._rows()
        rows[24]["graph_passed"] = False
        self.assertFalse(probe.evaluate_parity(rows)["passed"])
        rows = self._rows()
        rows[48]["activity_passed"] = False
        self.assertFalse(probe.evaluate_parity(rows)["passed"])

    def test_missing_required_shape_rejected(self):
        rows = self._rows()
        del rows[48]
        with self.assertRaisesRegex(ValueError, "missing"):
            probe.evaluate_parity(rows)


if __name__ == "__main__":
    unittest.main()
