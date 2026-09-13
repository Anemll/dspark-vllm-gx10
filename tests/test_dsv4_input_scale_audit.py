# SPDX-License-Identifier: MIT
import json
from pathlib import Path
import struct
import tempfile
import unittest

from benchmarks.audit_dsv4_hybrid import Checkpoint
from benchmarks.audit_dsv4_input_scales import compare_input_scales


def make_checkpoint(root, values):
    root = Path(root)
    payload = b''.join(struct.pack('<f', value) for value in values)
    header = {}
    weight_map = {}
    for expert, _ in enumerate(values):
        name = f'layers.0.ffn.experts.{expert}.w1.input_scale'
        header[name] = {
            'dtype': 'F32', 'shape': [],
            'data_offsets': [expert*4, (expert+1)*4],
        }
        weight_map[name] = 'model.safetensors'
    encoded = json.dumps(header).encode()
    (root/'model.safetensors').write_bytes(
        struct.pack('<Q', len(encoded))+encoded+payload
    )
    (root/'config.json').write_text('{}')
    (root/'model.safetensors.index.json').write_text(json.dumps({
        'weight_map': weight_map,
    }))
    return Checkpoint(root)


class InputScaleAuditTests(unittest.TestCase):
    def test_complete_match_and_mismatch(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            report = compare_input_scales(
                make_checkpoint(a, [0.5, 0.25]), make_checkpoint(b, [0.5, 0.25])
            )
            self.assertTrue(report['complete_match'])
            self.assertEqual(report['common_tensor_count'], 2)

        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            report = compare_input_scales(
                make_checkpoint(a, [0.5, 0.25]), make_checkpoint(b, [0.5, 0.125])
            )
            self.assertFalse(report['complete_match'])
            self.assertEqual(report['mismatch_count'], 1)


if __name__ == '__main__':
    unittest.main()
