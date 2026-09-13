# SPDX-License-Identifier: MIT
import unittest

import numpy as np

from benchmarks.audit_dsv4_fp4_transcode import (
    decode_e4m3fn,
    decode_e8m0,
    metrics,
    unpack_e2m1,
)


class Dsv4Fp4TranscodeAuditTests(unittest.TestCase):
    def test_e8m0_decode(self):
        actual = decode_e8m0(np.asarray([0, 126, 127, 128], dtype=np.uint8))
        np.testing.assert_array_equal(actual, [0.0, 0.5, 1.0, 2.0])

    def test_e4m3fn_decode(self):
        actual = decode_e4m3fn(
            np.asarray([0x00, 0x38, 0x40, 0x7E, 0xB8, 0x7F], dtype=np.uint8)
        )
        np.testing.assert_array_equal(actual[:5], [0.0, 1.0, 2.0, 448.0, -1.0])
        self.assertTrue(np.isnan(actual[-1]))

    def test_unpack_low_nibble_then_high_nibble(self):
        actual = unpack_e2m1(np.asarray([[0x21, 0xFE]], dtype=np.uint8))
        np.testing.assert_array_equal(actual, [[0.5, 1.0, -4.0, -6.0]])

    def test_metrics(self):
        exact = metrics(np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0]))
        self.assertEqual(exact['relative_l2'], 0.0)
        self.assertEqual(exact['exact_equal_fraction'], 1.0)
        changed = metrics(np.asarray([1.0, 2.0]), np.asarray([1.0, 3.0]))
        self.assertGreater(changed['relative_l2'], 0.0)
        self.assertEqual(changed['exact_equal_fraction'], 0.5)


if __name__ == '__main__':
    unittest.main()
