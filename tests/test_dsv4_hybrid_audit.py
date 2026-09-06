# SPDX-License-Identifier: MIT
import ast
from enum import Enum
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

from benchmarks.audit_dsv4_hybrid import Checkpoint, sample_ranges, vision_part


class HybridAuditTests(unittest.TestCase):
    def test_sample_ranges_are_bounded_and_do_not_double_count(self):
        for length in (0, 1, 4000, 4096, 8192, 10000, 1000000000):
            ranges = sample_ranges(length)
            self.assertLessEqual(sum(b-a for a,b in ranges), 3*4096)
            for i,(start,end) in enumerate(ranges):
                self.assertTrue(0 <= start <= end <= length)
                if i: self.assertGreater(start,ranges[i-1][1])
            if length <= 8192:
                self.assertEqual(sum(b-a for a,b in ranges),length)
        with self.assertRaises(ValueError): sample_ranges(-1)

    def make_checkpoint(self, root, payload=b'abcdefgh', offsets=None):
        root=Path(root)
        header={'norm.weight':{'dtype':'BF16','shape':[4],
                               'data_offsets':offsets or [0,len(payload)]}}
        encoded=json.dumps(header).encode()
        shard='model.safetensors'
        (root/shard).write_bytes(struct.pack('<Q',len(encoded))+encoded+payload)
        (root/'config.json').write_text('{}')
        (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'norm.weight':shard}}))
        return Checkpoint(root)

    def test_sample_reads_only_declared_tensor(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint=self.make_checkpoint(root)
            result=checkpoint.sample('norm.weight')
            self.assertTrue(result['complete_tensor'])
            self.assertEqual(checkpoint.payload_read,8)
            self.assertEqual(result['shape'],[4])
            self.assertFalse(checkpoint.identity()['whole_checkpoint_checksum_verified'])

    def test_truncated_offsets_and_oversized_header_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint=self.make_checkpoint(root,offsets=[0,99])
            with self.assertRaisesRegex(ValueError,'outside shard'):
                checkpoint.sample('norm.weight')
            (Path(root)/'model.safetensors').write_bytes(struct.pack('<Q',32*1024**2))
            with self.assertRaisesRegex(ValueError,'size limit'):
                Checkpoint(root).sample('norm.weight')

    def test_cumulative_payload_budget_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint=self.make_checkpoint(root)
            checkpoint.payload_read=16*1024**2
            with self.assertRaisesRegex(ValueError,'budget'):
                checkpoint.sample('norm.weight')
            for path in ('../escape.safetensors','/tmp/escape.safetensors','model.bin'):
                with self.assertRaises(ValueError): checkpoint.shard_path(path)

    def test_vision_parts_include_language_and_draft_router_biases(self):
        for name in ('vision.weight','aligner.weight','image_start','layers.0.ffn.gate.bias',
                     'layers.42.ffn.gate.bias_vl','mtp.2.ffn.gate.bias_vl'):
            self.assertTrue(vision_part(name),name)
        for name in ('layers.3.ffn.experts.0.w1.weight','mtp.0.ffn.gate.weight','head.weight'):
            self.assertFalse(vision_part(name),name)

    def test_pinned_real_nvfp4_selector_rejects_missing_b12x_clamp(self):
        path=Path(__file__).resolve().parents[1]/'.build/vllm-upstream/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py'
        if not path.exists(): self.skipTest('optional pristine vLLM source absent')
        tree=ast.parse(path.read_text())
        names={'NvFp4MoeBackend','map_nvfp4_backend','select_nvfp4_moe_backend'}
        nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
        self.assertEqual(len(nodes),3)
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
        def forbidden(*_): raise AssertionError('GPU/backend construction must not be reached')
        namespace={'Enum':Enum,'mk':SimpleNamespace(FusedMoEActivationFormat=SimpleNamespace(Standard='standard',BatchedExperts='batched')),
                   'backend_to_kernel_cls':forbidden}
        exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
        config=SimpleNamespace(swiglu_limit=10.0,moe_backend='flashinfer_b12x',
                               moe_parallel_config=SimpleNamespace(use_batched_activation_format=False))
        with self.assertRaisesRegex(ValueError,'does not apply the SwiGLU clamp'):
            namespace['select_nvfp4_moe_backend'](config,None,None)

    def test_cutlass_alternative_declares_sm121_nvfp4_support(self):
        path=Path(__file__).resolve().parents[1]/'.build/vllm-upstream/vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py'
        if not path.exists(): self.skipTest('optional pristine vLLM source absent')
        tree=ast.parse(path.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FlashInferExperts')
        names={'_supports_current_device','_supports_quant_scheme'}
        nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
        self.assertEqual(len(nodes),2)
        for node in nodes: node.decorator_list=[]
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
        namespace={'current_platform':SimpleNamespace(is_cuda=lambda:True,
                   is_device_capability=lambda n:n==121,is_device_capability_family=lambda n:n==120,
                   has_device_capability=lambda n:121>=n),'has_flashinfer_cutlass_fused_moe':lambda:True}
        for n in ast.walk(module):
            if isinstance(n,ast.Name) and n.id.startswith('k'):
                namespace.setdefault(n.id,object())
        exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
        self.assertTrue(namespace['_supports_current_device']())
        self.assertTrue(namespace['_supports_quant_scheme'](namespace['kNvfp4Static'],namespace['kNvfp4Dynamic']))
        # This is a source-declared alternative, not proof the installed
        # compiled kernel runs correctly on our model or hardware.


if __name__ == '__main__':
    unittest.main()
