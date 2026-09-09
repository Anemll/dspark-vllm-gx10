# SPDX-License-Identifier: MIT
import ast
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from benchmarks.benchmark_moe_tc_decode import TC_ENV, make_run_kwargs, make_scratch_plan, parity_gate, tc_mode, timing_decision, validate_dispatch


class MoETCContractTests(unittest.TestCase):
    def test_actual_serving_apply_with_fakes_matches_complete_benchmark_call(self):
        root = Path(__file__).resolve().parents[1]
        source = root/'overlay/vllm/model_executor/layers/fused_moe/experts/b12x_mxfp4_moe.py'
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='B12xExperts')
        apply = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='apply')
        module = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),apply],type_ignores=[])
        x = SimpleNamespace(shape=(6,4096),device='cuda:0',dtype='bf16')
        ids = SimpleNamespace(shape=(6,6))
        scores,out,unit,w1,s1,w2,s2,workspace,scratch,plan = [object() for _ in range(10)]
        prepared = SimpleNamespace(w4a16=SimpleNamespace(num_experts=256,intermediate_size=1024))
        spec = SimpleNamespace(num_experts=256,hidden_size=4096,I_tp=1024,top_k=6)
        weights = SimpleNamespace(w13_weight=w1,w13_blockscale_swizzled=s1,w2_weight=w2,w2_blockscale_swizzled=s2,source_format='fp4_e8m0_k32',w13_layout='w31')
        expected_plan = make_scratch_plan(SimpleNamespace(_plan_b12x_moe_fp4_scratch=lambda **kw:kw),spec,x,weights)
        # Serving spells out this default while the factory relies on the
        # adapter's identical False default.
        expected_plan['apply_router_weight_on_input'] = False
        observed = []
        def plan_spy(**kw):
            self.assertEqual(kw,expected_plan)
            return plan
        def scratch_spy(actual,actual_plan):
            self.assertIs(actual,workspace)
            self.assertIs(actual_plan,plan)
            return scratch
        namespace = {'_b12x_activation_name':lambda x:x,'_normalize_b12x_moe_topk_ids':lambda x:x,
                     '_normalize_b12x_moe_topk_weights':lambda x:x,'_plan_b12x_moe_fp4_scratch':plan_spy,
                     '_workspace2_as_b12x_scratch':scratch_spy,'_run_b12x_moe_fp4':lambda **kw:observed.append(kw)}
        exec(compile(ast.fix_missing_locations(module),str(source),'exec'),namespace)
        fake_self = SimpleNamespace(_get_or_prepare_fp4_moe_weights=lambda **_:prepared,w1_scale=s1,w2_scale=s2,
            _unit_expert_scale=lambda *_:unit,_source_format=lambda:'fp4_e8m0_k32',_w13_layout=lambda:'w31',quant_config=SimpleNamespace(gemm1_clamp_limit=10.0))
        namespace['apply'](fake_self,out,x,w1,w2,scores,ids,'silu',256,None,None,None,None,workspace,None,False)
        self.assertEqual(len(observed),1)
        expected = make_run_kwargs(x,weights,prepared,unit,ids,scores,out,plan,scratch)
        self.assertEqual(observed[0],expected)

    def test_run_binding_matches_serving_keyword_contract(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root/'overlay/vllm/model_executor/layers/fused_moe/experts/b12x_mxfp4_moe.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_run_b12x_moe_fp4']
        self.assertEqual(len(calls),1)
        tensors = [object() for _ in range(11)]
        x,unit,ids,scores,out,plan,scratch,w1,s1,w2,s2 = tensors
        weights = SimpleNamespace(w13_weight=w1,w13_blockscale_swizzled=s1,w2_weight=w2,w2_blockscale_swizzled=s2,source_format='fp4_e8m0_k32',w13_layout='w31')
        prepared = SimpleNamespace(w4a16=object())
        kwargs = make_run_kwargs(x,weights,prepared,unit,ids,scores,out,plan,scratch)
        self.assertEqual(set(kwargs),{k.arg for k in calls[0].keywords})
        for keyword in calls[0].keywords:
            if isinstance(keyword.value,ast.Constant):
                self.assertEqual(kwargs[keyword.arg],keyword.value.value,keyword.arg)
        self.assertIs(kwargs['input_scales_are_reciprocal'],True)
        for key,value in [('a',x),('a1_gscale',unit),('a2_gscale',unit),('w1_alphas',unit),('w2_alphas',unit),('topk_ids',ids),('topk_weights',scores),('output',out),('plan',plan),('scratch',scratch),('prepared_w4a16',prepared.w4a16)]:
            self.assertIs(kwargs[key],value,key)
        self.assertEqual(kwargs['activation'],'silu')
        self.assertEqual(kwargs['source_format'],'fp4_e8m0_k32')
        self.assertEqual(kwargs['w13_layout'],'w31')
        self.assertEqual(kwargs['swiglu_limit'],10.0)

    def test_scratch_plan_uses_resolved_tensor_device_and_dtype(self):
        adapter = SimpleNamespace(_plan_b12x_moe_fp4_scratch=lambda **kwargs: kwargs)
        spec = SimpleNamespace(num_experts=256,hidden_size=4096,I_tp=1024,top_k=6)
        weights = SimpleNamespace(source_format='fp4_e8m0_k32',w13_layout='w31')
        for device in ('cuda:0','cuda:1'):
            x = SimpleNamespace(shape=(6,4096),device=device,dtype='bf16')
            plan = make_scratch_plan(adapter,spec,x,weights)
            self.assertEqual(plan['device'],device)
            self.assertEqual(plan['dtype'],x.dtype)
            self.assertEqual(plan['tokens'],6)
            self.assertEqual(plan['n'],1024)

    def test_env_restored_after_failure(self):
        with patch.dict(os.environ, {TC_ENV:'original'}):
            with self.assertRaises(ValueError):
                with tc_mode(True):
                    self.assertEqual(os.environ[TC_ENV],'1')
                    raise ValueError('stop')
            self.assertEqual(os.environ[TC_ENV],'original')
        with patch.dict(os.environ, clear=True):
            with tc_mode(False):
                self.assertEqual(os.environ[TC_ENV],'0')
            self.assertNotIn(TC_ENV,os.environ)

    def test_dispatch_requires_observed_actual_path(self):
        for tokens in (1,6,8,12):
            for enabled in (False,True):
                good = {'token_count':tokens,'weight_layout':'packed','tc_decode_fused_sum':enabled and tokens<=8}
                validate_dispatch([good],tokens,enabled)
                with self.assertRaises(RuntimeError):
                    validate_dispatch([good|{'tc_decode_fused_sum':not good['tc_decode_fused_sum']}],tokens,enabled)
                with self.assertRaises(RuntimeError):
                    validate_dispatch([good|{'weight_layout':'modelopt'}],tokens,enabled)
        with self.assertRaises(RuntimeError):
            validate_dispatch([],6,True)

    def test_nonfinite_or_inaccurate_oracle_output_rejected(self):
        good = {'cos':0.9975,'rmse':0.2,'max_abs':1.0,'mean_abs':0.1}
        parity_gate(good)
        for bad in (good|{'cos':0.9974},good|{'rmse':math.nan},good|{'max_abs':math.inf}):
            with self.assertRaises(RuntimeError):
                parity_gate(bad)

    def test_speed_gate_distinguishes_m6_gain_and_other_shape_regression(self):
        def rows(tokens,cold,warm):
            return [{'tokens':tokens,'cache':cache,'variant':variant,'median_us':value} for cache,candidate in [('cold_l2',cold),('warm_l2',warm)] for variant,value in [('control',100),('tc_decode',candidate)]]
        self.assertTrue(timing_decision(rows(6,88,101),6)['worthwhile_component_gate'])
        self.assertFalse(timing_decision(rows(6,95,101),6)['worthwhile_component_gate'])
        self.assertFalse(timing_decision(rows(6,80,104),6)['worthwhile_component_gate'])
        self.assertTrue(timing_decision(rows(12,102,101),12)['worthwhile_component_gate'])
        with self.assertRaises(RuntimeError):
            timing_decision(rows(6,math.nan,80),6)

    def test_offline_plan_never_imports_cuda_and_preserves_output(self):
        runner = Path(__file__).resolve().parents[1]/'benchmarks/benchmark_moe_tc_decode.py'
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'plan.json'
            command = [sys.executable,str(runner),'--model-dir','/not-a-model','--plan-only','--output',str(out)]
            first = subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(first.returncode,0,first.stderr)
            original = out.read_bytes()
            second = subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(second.returncode,0)
            self.assertEqual(out.read_bytes(),original)


if __name__ == '__main__':
    unittest.main()
