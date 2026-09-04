# SPDX-License-Identifier: MIT
"""CPU execution-order fakes for the opt-in graph boundary; not GPU proof."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "overlay/vllm/models/deepseek_v4/attention.py"


def methods():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4Attention")
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in ("_prepare_and_attn", "_sparse_indexer_and_attn")]
    for node in selected:
        node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    scope = {
        "get_forward_context": lambda: SimpleNamespace(attn_metadata={}),
        "execute_in_parallel": lambda main, aux, *a, **k: (main(), [f() for f in aux]),
        "maybe_execute_in_parallel": lambda main, aux, *a, **k: (main(), aux()),
    }
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), scope)
    return scope


class Value:
    def view(self, *args):
        return self


class GraphBoundaryTests(unittest.TestCase):
    def test_actual_constructor_guard_and_platform_import(self):
        tree = ast.parse(SOURCE.read_text())
        self.assertTrue(any(
            isinstance(node, ast.ImportFrom) and node.module == "vllm.platforms"
            and any(alias.name == "current_platform" for alias in node.names)
            for node in tree.body
        ), "the opt-in constructor must import its platform helper")
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "DeepseekV4Attention")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                    and n.name == "__init__")
        prefix = []
        for node in init.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "tp_size"
                for target in node.targets
            ):
                break
            prefix.append(node)
        init.body = prefix
        harness = ast.ClassDef(name="GuardHarness", bases=[ast.Name(id="Base", ctx=ast.Load())],
                               keywords=[], body=[init], decorator_list=[])

        class Base:
            def _prepare_and_attn(self):
                pass

        for enabled, v2, sm12x, dtype, allowed in (
            (True, True, True, "nvfp4_ds_mla", True),
            (True, False, True, "nvfp4_ds_mla", False),
            (True, True, False, "nvfp4_ds_mla", False),
            (True, True, True, "fp8", False),
            (True, True, True, None, False),
            (False, False, False, None, True),
        ):
            scope = {
                "Base": Base,
                "narrow_attention_graph_enabled": lambda: enabled,
                "current_platform": SimpleNamespace(is_device_capability_family=lambda family: sm12x and family == 120),
                "eager_break_during_capture": lambda fn: fn,
            }
            module = ast.Module(body=[ast.ImportFrom(module="__future__", level=0,
                                names=[ast.alias(name="annotations")]), harness], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), scope)
            config = SimpleNamespace(model_config=SimpleNamespace(hf_config=None),
                                     quant_config=None, use_v2_model_runner=v2,
                                     cache_config=None if dtype is None else SimpleNamespace(cache_dtype=dtype))
            with self.subTest(enabled=enabled, v2=v2, sm12x=sm12x, dtype=dtype):
                if allowed:
                    self.assertEqual(scope["GuardHarness"](config, "test")._narrow_attention_graph, enabled)
                else:
                    with self.assertRaises(ValueError):
                        scope["GuardHarness"](config, "test")

    def run_case(self, narrow, branch, quant_tuple):
        events = []
        scope = methods()
        q, iq, scale, weights, out = (Value() for _ in range(5))
        owner = SimpleNamespace(_narrow_attention_graph=narrow,
                                aux_stream_list=[0, 1, 2], ln_events=[0, 1, 2],
                                n_local_heads=32, head_dim=512, rotary_emb=None,
                                indexer_rotary_emb=None)
        owner.wq_b = lambda value: (events.append("q-project"), q)[1]
        owner._fused_qnorm_rope_kv_insert = lambda *args: (events.append("kv-insert"), q)[1]
        owner.forward_mqa = lambda *args: events.append("attention")
        owner.compressor = (lambda *args: events.append("compress")) if branch != "swa" else None

        class Indexer:
            def indexer_op(self, *args):
                events.append("score")
                self_case.assertIs(args[1][0] if quant_tuple else args[1], iq)
                self_case.assertIs(args[3], weights)

            def __call__(self, *args, prepare_only=False):
                events.append("index-prepare")
                packed = (iq, scale) if quant_tuple else iq
                if prepare_only:
                    return packed, weights
                self.indexer_op(None, packed, None, weights)
                return Value()

        self_case = self
        owner.indexer = Indexer() if branch == "c4" else None
        owner._sparse_indexer_and_attn = lambda *args: scope["_sparse_indexer_and_attn"](owner, *args)
        scope["_prepare_and_attn"](owner, *[Value() for _ in range(7)], out)
        self.assertEqual(events.count("q-project"), 1)
        self.assertEqual(events.count("kv-insert"), 1)
        self.assertEqual(events.count("attention"), 1)
        self.assertEqual(events[-1], "attention")
        self.assertEqual(events.count("score"), int(branch == "c4"))
        if branch == "c4":
            self.assertLess(events.index("index-prepare"), events.index("score"))
            if narrow:
                self.assertLess(events.index("compress"), events.index("score"))
            else:
                self.assertLess(events.index("score"), events.index("compress"))

    def test_control_and_narrow_all_layer_types(self):
        for narrow in (False, True):
            for branch in ("swa", "c4", "c128"):
                for quant_tuple in (False, True):
                    with self.subTest(narrow=narrow, branch=branch, quant_tuple=quant_tuple):
                        self.run_case(narrow, branch, quant_tuple)


if __name__ == "__main__":
    unittest.main()
