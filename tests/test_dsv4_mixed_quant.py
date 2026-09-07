# SPDX-License-Identifier: MIT
"""CPU contract tests; these do not validate compiled kernels or model quality."""
import ast
import builtins
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "overlay/vllm/models/deepseek_v4/mixed_expert_quant.py"
SPEC = importlib.util.spec_from_file_location("mixed_expert_quant_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
expert_quant_algo = MODULE.expert_quant_algo


def config():
    return SimpleNamespace(num_hidden_layers=43, quantization_config={
        "moe_quant_algo": "NVFP4", "ignore": ["mtp.*"],
        "quantized_layers": {f"layers.{i}.ffn.experts": {
            "quant_algo": "NVFP4", "group_size": 16} for i in range(43)},
    })


class MixedQuantTests(unittest.TestCase):
    def test_real_dispatch_reproduces_upstream_draft_bug_and_fixes_it(self):
        cfg = config()
        class Expert:
            moe_config = object()
        class Base:
            def get_quant_method(self, layer, prefix):
                return "linear-fp8"
        def fake_import(name, *args, **kwargs):
            if name == "mixed_expert_quant":
                return MODULE
            if name == "vllm.model_executor.layers.quantization.modelopt":
                return SimpleNamespace(ModelOptNvFp4FusedMoE=lambda **kw: "nvfp4")
            return builtins.__import__(name, *args, **kwargs)
        def dispatch(path):
            tree = ast.parse(path.read_text())
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4FP8Config")
            keep = {"get_quant_method", "is_mxfp4_quant", "_moe_quant_algo_for_prefix"}
            cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in keep]
            namespace = {
                "Fp8Config": Base, "RoutedExperts": Expert,
                "is_layer_skipped": lambda **kw: False,
                "Mxfp4MoEMethod": lambda *a: "mxfp4",
                "get_current_vllm_config": lambda: SimpleNamespace(model_config=SimpleNamespace(hf_config=cfg)),
                "__builtins__": {**vars(builtins), "__import__": fake_import},
            }
            exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
            obj = namespace["DeepseekV4FP8Config"]()
            obj.expert_dtype, obj.moe_quant_algo = "fp4", "NVFP4"
            obj.ignored_layers, obj.packed_modules_mapping = [], {}
            obj._get_nvfp4_config = lambda: object()
            return obj
        patched = dispatch(ROOT / "overlay/vllm/models/deepseek_v4/quant_config.py")
        for i in range(46):
            prefix = f"model.layers.{i}.ffn.experts"
            self.assertEqual(patched.get_quant_method(Expert(), prefix), "nvfp4" if i < 43 else "mxfp4")
            self.assertEqual(patched.is_mxfp4_quant(prefix, Expert()), i >= 43)
        self.assertEqual(patched.get_quant_method(object(), "attn.wq"), "linear-fp8")
        original = ROOT / ".build/vllm-upstream/vllm/models/deepseek_v4/quant_config.py"
        if original.exists():
            upstream = dispatch(original)
            self.assertEqual(upstream.get_quant_method(Expert(), "model.layers.43.ffn.experts"), "nvfp4")
            self.assertFalse(upstream.is_mxfp4_quant("model.layers.43.ffn.experts", Expert()))

    def test_all_target_and_three_draft_layers(self):
        cfg = config()
        for wrapper in ("", "model.", "language_model.model."):
            for suffix in ("", ".routed_experts"):
                for i in range(46):
                    prefix = f"{wrapper}layers.{i}.ffn.experts{suffix}"
                    self.assertEqual(expert_quant_algo(cfg, prefix, "NVFP4"),
                                     "NVFP4" if i < 43 else "MXFP4")

    def test_original_and_uniform_checkpoints_keep_dispatch(self):
        cfg = config()
        self.assertEqual(expert_quant_algo(cfg, "irrelevant", ""), "")
        del cfg.quantization_config["quantized_layers"]
        self.assertEqual(expert_quant_algo(cfg, "model.layers.43.ffn.experts", "NVFP4"), "NVFP4")

    def test_missing_target_or_unknown_layer_fails_closed(self):
        cfg = config()
        del cfg.quantization_config["quantized_layers"]["layers.3.ffn.experts"]
        for prefix in ("model.layers.3.ffn.experts", "model.layers.46.ffn.experts", "unknown"):
            with self.assertRaises(ValueError): expert_quant_algo(cfg, prefix, "NVFP4")

    def test_draft_requires_explicit_exclusion(self):
        cfg = config()
        cfg.quantization_config["ignore"] = []
        with self.assertRaisesRegex(ValueError, "Missing explicit"):
            expert_quant_algo(cfg, "model.layers.43.ffn.experts", "NVFP4")
        cfg.quantization_config["ignore"] = ["mtp.*"]
        cfg.quantization_config["quantized_layers"]["mtp.0.ffn.experts"] = {"quant_algo": "NVFP4", "group_size": 16}
        with self.assertRaisesRegex(ValueError, "Missing explicit"):
            expert_quant_algo(cfg, "model.layers.43.ffn.experts", "NVFP4")

    def test_wrong_group_and_malformed_manifest_fail_closed(self):
        for entry in ({"quant_algo": "NVFP4", "group_size": 32},
                      {"quant_algo": "INT4", "group_size": 16}, None):
            cfg = config()
            cfg.quantization_config["quantized_layers"]["layers.0.ffn.experts"] = entry
            with self.assertRaises(ValueError):
                expert_quant_algo(cfg, "model.layers.0.ffn.experts", "NVFP4")
        cfg.quantization_config["quantized_layers"] = []
        with self.assertRaisesRegex(ValueError, "mapping"):
            expert_quant_algo(cfg, "model.layers.0.ffn.experts", "NVFP4")

    def test_dispatch_and_weight_layout_use_the_same_resolver(self):
        source = ROOT / "overlay/vllm/models/deepseek_v4/quant_config.py"
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4FP8Config")
        for name in ("get_quant_method", "is_mxfp4_quant"):
            method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
            calls = [n for n in ast.walk(method) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == "_moe_quant_algo_for_prefix"]
            self.assertEqual(len(calls), 1)
            self.assertEqual(ast.unparse(calls[0].args[0]), "prefix")


if __name__ == "__main__":
    unittest.main()
