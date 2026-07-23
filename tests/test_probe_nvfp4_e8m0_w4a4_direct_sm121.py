from __future__ import annotations

import unittest

from benchmarks import probe_nvfp4_e8m0_w4a4_direct_sm121 as probe


class E8m0W4a4DirectProbeTest(unittest.TestCase):
    def test_scale_algebra_keeps_w4a4(self) -> None:
        algebra = probe.candidate_scale_algebra()
        self.assertEqual(algebra["arithmetic"], "W4A4; w4a16_mode=False")
        self.assertIn("1 / a_gscale", algebra["candidate_alpha"])

    def test_fixed_shape(self) -> None:
        self.assertEqual(probe.M_VALUE, 4)


if __name__ == "__main__":
    unittest.main()
