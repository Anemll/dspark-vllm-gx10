# SPDX-License-Identifier: MIT
"""Run the actual V2 loader with CPU doubles; not GPU or checkpoint evidence."""
import ast
import builtins
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
RELATIVE = Path("vllm/v1/worker/gpu/spec_decode/dspark/utils.py")
PATCHED = ROOT / "overlay" / RELATIVE
UPSTREAM = ROOT / ".build/vllm-upstream" / RELATIVE


@dataclass
class Kernel:
    moe_backend: str = "flashinfer_cutlass"
    linear_backend: str = "unchanged"


@dataclass
class Attention:
    backend: str | None = "target_attention"
    use_non_causal: bool = False


@dataclass
class Config:
    speculative_config: object
    kernel_config: Kernel
    attention_config: Attention
    cache_config: object
    compilation_config: object


def make_config(override="flashinfer_b12x", target="flashinfer_cutlass"):
    return Config(
        SimpleNamespace(moe_backend=override, attention_backend=None,
                        draft_model_config=object()),
        Kernel(target), Attention(), object(), object(),
    )


class LoaderHarness:
    def __init__(self, path=PATCHED, share=True, pp=1, fail=False):
        self.loaded = []
        self.tags = []
        self.target = SimpleNamespace(model=SimpleNamespace(embed_tokens=object()),
                                      lm_head=object())
        self.draft = SimpleNamespace(model=SimpleNamespace(embed_tokens=object()),
                                     lm_head=object())

        @contextmanager
        def set_model_tag(tag):
            self.tags.append(tag)
            try:
                yield
            finally:
                self.tags.pop()

        def get_model(**kwargs):
            self.loaded.append(kwargs)
            if self.tags != ["dspark_head"]:
                raise AssertionError("DSpark model tag was lost")
            if fail:
                raise RuntimeError("simulated loader failure")
            return self.draft

        def fake_import(name, *args, **kwargs):
            if name == "vllm.compilation.backends":
                return SimpleNamespace(set_model_tag=set_model_tag)
            return builtins.__import__(name, *args, **kwargs)

        tree = ast.parse(path.read_text())
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == "load_dspark_model"]
        if len(funcs) != 1:
            raise AssertionError("Expected one actual DSpark loader")
        future = ast.parse("from __future__ import annotations").body
        namespace = {
            "replace": replace, "get_model": get_model,
            "get_pp_group": lambda: SimpleNamespace(world_size=pp),
            "_should_share": lambda *args: share,
            "__builtins__": {**vars(builtins), "__import__": fake_import},
        }
        exec(compile(ast.Module(body=[*future, *funcs], type_ignores=[]),
                     str(path), "exec"), namespace)
        self.run = namespace["load_dspark_model"]


class DSparkDraftBackendTests(unittest.TestCase):
    def test_pinned_v2_loader_ignores_override(self):
        if not UPSTREAM.exists():
            self.skipTest("optional pristine upstream source absent")
        harness = LoaderHarness(UPSTREAM)
        cfg = make_config()
        harness.run(harness.target, cfg)
        loaded = harness.loaded[0]["vllm_config"]
        self.assertEqual(loaded.kernel_config.moe_backend, "flashinfer_cutlass")
        self.assertNotEqual(loaded.kernel_config.moe_backend,
                            cfg.speculative_config.moe_backend)

    def test_override_reaches_loader_without_mutating_target(self):
        harness = LoaderHarness()
        cfg = make_config()
        result = harness.run(harness.target, cfg)
        loaded = harness.loaded[0]["vllm_config"]
        self.assertIs(result, harness.draft)
        self.assertEqual(loaded.kernel_config.moe_backend, "flashinfer_b12x")
        self.assertEqual(cfg.kernel_config.moe_backend, "flashinfer_cutlass")
        self.assertIsNot(loaded.kernel_config, cfg.kernel_config)
        self.assertEqual(loaded.kernel_config.linear_backend, "unchanged")
        self.assertIs(loaded.cache_config, cfg.cache_config)
        self.assertIs(loaded.compilation_config, cfg.compilation_config)
        self.assertIs(harness.loaded[0]["model_config"],
                      cfg.speculative_config.draft_model_config)
        self.assertEqual(harness.tags, [])

    def test_no_override_retains_existing_text_kernel_config(self):
        for target in ("flashinfer_b12x", "flashinfer_cutlass", "auto"):
            with self.subTest(target=target):
                harness = LoaderHarness()
                cfg = make_config(None, target)
                harness.run(harness.target, cfg)
                loaded = harness.loaded[0]["vllm_config"]
                self.assertIs(loaded.kernel_config, cfg.kernel_config)
                self.assertEqual(loaded.kernel_config.moe_backend, target)

    def test_noncausal_attention_and_explicit_attention_override_survive(self):
        for backend in (None, "draft_attention"):
            harness = LoaderHarness()
            cfg = make_config()
            cfg.speculative_config.attention_backend = backend
            harness.run(harness.target, cfg)
            loaded = harness.loaded[0]["vllm_config"]
            self.assertTrue(loaded.attention_config.use_non_causal)
            self.assertEqual(loaded.attention_config.backend, backend)
            self.assertFalse(cfg.attention_config.use_non_causal)
            self.assertEqual(cfg.attention_config.backend, "target_attention")

    def test_multimodal_embedding_sharing_and_own_weights_are_preserved(self):
        for share in (True, False):
            harness = LoaderHarness(share=share)
            old_embed, old_head = harness.draft.model.embed_tokens, harness.draft.lm_head
            wrapper = SimpleNamespace(get_language_model=lambda: harness.target,
                                      lm_head=harness.target.lm_head)
            result = harness.run(wrapper, make_config())
            self.assertIs(result.model.embed_tokens,
                          harness.target.model.embed_tokens if share else old_embed)
            self.assertIs(result.lm_head, harness.target.lm_head if share else old_head)

    def test_failed_load_does_not_mutate_target_or_leak_model_tag(self):
        harness = LoaderHarness(fail=True)
        cfg = make_config()
        with self.assertRaisesRegex(RuntimeError, "simulated loader failure"):
            harness.run(harness.target, cfg)
        self.assertEqual(cfg.kernel_config.moe_backend, "flashinfer_cutlass")
        self.assertEqual(cfg.attention_config.backend, "target_attention")
        self.assertEqual(harness.tags, [])

    def test_pipeline_parallel_guard_is_preserved(self):
        harness = LoaderHarness(pp=2)
        with self.assertRaisesRegex(NotImplementedError, "pipeline parallelism"):
            harness.run(harness.target, make_config())

    def test_component_image_includes_the_v2_loader_fix(self):
        source = (ROOT / "docker/Dockerfile.nvfp4-cutlass").read_text()
        self.assertIn(f"COPY overlay/{RELATIVE} "
                      f"/usr/local/lib/python3.12/dist-packages/{RELATIVE}", source)


if __name__ == "__main__":
    unittest.main()
