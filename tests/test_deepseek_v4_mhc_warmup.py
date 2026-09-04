# SPDX-License-Identifier: MIT
"""Exercise mHC startup dispatch and shape coverage without a GPU runtime."""

import contextlib
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "overlay/vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py"
)
CAPTURE_SIZES = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72]


class Tensor:
    def __init__(self, *shape, device=None, dtype="bf16"):
        self.shape = shape
        self.device = device or SimpleNamespace(type="cuda", index=0)
        self.dtype = dtype
        self.data = self

    def __getitem__(self, key):
        assert isinstance(key, slice) and key.start is None
        return Tensor(key.stop, *self.shape[1:], device=self.device, dtype=self.dtype)


class Node(SimpleNamespace):
    def modules(self):
        yield self
        for child in getattr(self, "children", []):
            yield from child.modules()


class DeepseekV4DecoderLayer(Node):
    pass


class DeepseekV4Model(Node):
    pass


class DSparkDeepseekV4Model(Node):
    pass


def layer(*, native=True, device_type="cuda"):
    device = SimpleNamespace(type=device_type, index=0)
    value = DeepseekV4DecoderLayer(hidden_size=4096, hc_mult=4)
    for name, shape in (("fn", (24, 16384)), ("scale", (3,)), ("base", (24,))):
        setattr(value, "hc_attn_" + name, Tensor(*shape, device=device))
        setattr(value, "hc_ffn_" + name, Tensor(*shape, device=device))
    if native:
        value.rms_norm_eps = 1e-6
        value.hc_eps = 1e-6
        value.hc_post_alpha = 2.0
        value.hc_sinkhorn_iters = 20
        value.attn_norm = SimpleNamespace(weight=Tensor(4096), variance_epsilon=1e-6)
        # Deliberately different to catch accidentally forwarding attention norm.
        value.ffn_norm = SimpleNamespace(weight=Tensor(4096), variance_epsilon=2e-6)
    return value


def model(decoder, cls=DeepseekV4Model):
    return cls(
        children=[decoder],
        config=SimpleNamespace(model_type="deepseek_v4", hidden_size=4096),
        hc_mult=4,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_head_fn=Tensor(4, 16384, device=decoder.hc_attn_fn.device),
        hc_head_scale=Tensor(1),
        hc_head_base=Tensor(4),
    )


class WarmupTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.allocations = []
        self.logs = []
        self.synchronizations = []
        self.deep_gemm_supported = True

        def zeros(*shape, **kwargs):
            self.allocations.append((shape, kwargs))
            return Tensor(*shape, **kwargs)

        def pre(*args, **kwargs):
            self.calls.append(("pre", args, kwargs))
            t, hc, hidden = args[0].shape
            return Tensor(t, hc, 1), Tensor(t, hc, hc), Tensor(t, hidden)

        def fused(*args, **kwargs):
            self.calls.append(("fused", args, kwargs))
            t, hc, hidden = args[1].shape
            return (
                Tensor(t, hc, hidden),
                Tensor(t, hc, 1),
                Tensor(t, hc, hc),
                Tensor(t, hidden),
            )

        def post(*args, **kwargs):
            self.calls.append(("post", args, kwargs))
            return Tensor(*args[1].shape)

        def head(*args, **kwargs):
            self.calls.append(("head", args, kwargs))
            return Tensor(args[0].shape[0], args[0].shape[2])

        def module(name, **attrs):
            result = ModuleType(name)
            result.__dict__.update(attrs)
            return result

        fake_torch = module(
            "torch",
            zeros=zeros,
            bfloat16="bf16",
            Tensor=Tensor,
            nn=SimpleNamespace(Module=Node),
            inference_mode=contextlib.nullcontext,
            accelerator=SimpleNamespace(
                synchronize=lambda: self.synchronizations.append(True)
            ),
            cuda=SimpleNamespace(
                get_device_properties=lambda device: SimpleNamespace(
                    multi_processor_count=48
                )
            ),
        )
        self.tilelang = module(
            "vllm.model_executor.kernels.mhc.tilelang",
            mhc_pre_tilelang=pre,
            mhc_fused_post_pre_tilelang=fused,
            mhc_post_tilelang=post,
            hc_head_fused_kernel_tilelang=head,
        )
        modules = {
            "torch": fake_torch,
            "vllm.logger": module(
                "vllm.logger",
                init_logger=lambda name: SimpleNamespace(
                    info=lambda *args: self.logs.append(args)
                ),
            ),
            "vllm.tracing": module(
                "vllm.tracing", instrument=lambda **kwargs: lambda fn: fn
            ),
            "vllm.utils.math_utils": module(
                "vllm.utils.math_utils", cdiv=lambda a, b: (a + b - 1) // b
            ),
            "vllm.utils.deep_gemm": module(
                "vllm.utils.deep_gemm",
                is_deep_gemm_supported=lambda: self.deep_gemm_supported,
            ),
            self.tilelang.__name__: self.tilelang,
        }
        self.module_patch = patch.dict(sys.modules, modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        spec = importlib.util.spec_from_file_location("mhc_warmup_under_test", SOURCE)
        self.warmup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.warmup)

    def test_detects_current_target_and_dspark_decoder_without_old_methods(self):
        for cls in (DeepseekV4Model, DSparkDeepseekV4Model):
            decoder = layer()
            with self.subTest(model=cls.__name__):
                instance = model(decoder, cls)
                self.assertIs(self.warmup._find_first_mhc_layer(instance), decoder)
                self.assertIs(self.warmup._find_deepseek_v4_model(instance), instance)
                self.assertFalse(hasattr(decoder, "hc_pre"))
                self.assertTrue(self.warmup._uses_functional_mhc(decoder))

    def test_native_calls_match_standalone_and_fused_norm_contracts(self):
        decoder = layer()
        self.warmup._warmup_layer_mhc(decoder, [14, 294])
        self.assertEqual(
            [call[0] for call in self.calls], ["pre", "fused", "fused", "post"] * 2
        )
        for offset, size in ((0, 14), (4, 294)):
            pre, ffn, attn, post = self.calls[offset : offset + 4]
            self.assertEqual(pre[1][0].shape, (size, 4, 4096))
            self.assertEqual(pre[1][4:], (1e-6, 1e-6, 1e-6, 2.0, 20))
            self.assertIs(pre[2]["norm_weight"], decoder.attn_norm.weight)
            self.assertEqual(pre[2]["norm_eps"], 1e-6)
            for call, norm, fn in (
                (ffn, decoder.ffn_norm, decoder.hc_ffn_fn),
                (attn, decoder.attn_norm, decoder.hc_attn_fn),
            ):
                self.assertEqual(call[1][0].shape, (size, 4096))
                self.assertEqual(call[1][1].shape, (size, 4, 4096))
                self.assertIs(call[1][4], fn)
                self.assertEqual(call[1][7:], (1e-6, 1e-6, 1e-6, 2.0, 20))
                self.assertEqual(
                    call[2],
                    {
                        "n_splits": 1,
                        "tile_n": 1,
                        "norm_weight": norm.weight,
                        "norm_eps": norm.variance_epsilon,
                    },
                )
            self.assertEqual(post[1][0].shape, (size, 4096))
        self.assertEqual(len(self.allocations), 1)
        self.assertEqual(self.allocations[0][0], (294, 4, 4096))

    def test_legacy_non_norm_and_custom_head_dispatch_are_preserved(self):
        decoder = layer(native=False)
        legacy_calls = []

        def legacy_pre(residual, fn, scale, base):
            legacy_calls.append(("pre", residual.shape, fn))
            t, hc, hidden = residual.shape
            return Tensor(t, hidden), Tensor(t, hc, 1), Tensor(t, hc, hc)

        decoder.hc_pre = legacy_pre
        decoder.hc_post = lambda *args: legacy_calls.append(("post", args[0].shape))
        instance = model(decoder)
        instance.hc_head_op = lambda *args: legacy_calls.append(("head", args[0].shape))
        self.warmup.deepseek_v4_mhc_warmup(instance, max_tokens=2)
        self.assertEqual(
            [call[0] for call in legacy_calls],
            ["pre", "post", "pre", "post"] * 2 + ["head", "head"],
        )
        self.assertEqual(self.calls, [])
        del instance.hc_head_op
        self.warmup._warmup_hc_head(instance, [1])
        self.assertEqual(self.calls, [])

    def test_gb10_split_representatives_cover_294_and_all_supported_prefill(self):
        base = self.warmup._select_mhc_warmup_token_sizes(
            max_tokens=2048, cudagraph_capture_sizes=CAPTURE_SIZES
        )
        sizes = self.warmup._select_mhc_warmup_token_sizes(
            max_tokens=2048,
            cudagraph_capture_sizes=CAPTURE_SIZES,
            hidden_size=4096,
            hc_mult=4,
            num_sms=48,
        )
        self.assertEqual(sorted(set(sizes) - set(base)), [129, 257, 321, 513, 577, 1025])
        # These are the distinct hardware reduction contracts across 1..2048.
        expected = {48, 24, 16, 12, 9, 8, 6, 5, 4, 3, 2, 1}
        self.assertEqual({max(48 // ((t + 63) // 64), 1) for t in sizes}, expected)
        self.assertIn(257, sizes)  # 257 and the 294-token request both use split 9.
        self.assertIn(1, sizes)  # fused small FMA split 8, tile_n 2
        self.assertIn(8, sizes)  # fused small FMA split 4, tile_n 3
        self.assertEqual(sizes, sorted(set(sizes)))

    def test_target_and_draft_use_full_residual_width_on_each_tp_rank(self):
        for cls in (DeepseekV4Model, DSparkDeepseekV4Model):
            with self.subTest(model=cls.__name__):
                self.calls.clear()
                self.allocations.clear()
                self.warmup.deepseek_v4_mhc_warmup(
                    model(layer(), cls),
                    max_tokens=2048,
                    cudagraph_capture_sizes=CAPTURE_SIZES,
                )
                pre_shapes = [
                    args[0].shape for name, args, kw in self.calls if name == "pre"
                ]
                head_shapes = [
                    args[0].shape for name, args, kw in self.calls if name == "head"
                ]
                self.assertEqual(pre_shapes, head_shapes)
                self.assertEqual(len(pre_shapes), 23)
                self.assertTrue(all(shape[1:] == (4, 4096) for shape in pre_shapes))
                self.assertEqual(len(self.allocations), 2)
                self.assertTrue(
                    all(shape == (2048, 4, 4096) for shape, kw in self.allocations)
                )

    def test_unsupported_models_and_cpu_do_no_work(self):
        for instance in (
            model(layer(device_type="cpu")),
            Node(config=SimpleNamespace(model_type="qwen")),
            Node(children=[Node()]),
        ):
            self.warmup.deepseek_v4_mhc_warmup(instance, max_tokens=2048)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.allocations, [])
        self.assertEqual(self.synchronizations, [])

    def test_bounds_and_no_deepgemm_keep_existing_shape_budget(self):
        self.assertEqual(
            self.warmup._select_mhc_warmup_token_sizes(
                max_tokens=0, cudagraph_capture_sizes=[1]
            ),
            [],
        )
        small = self.warmup._select_mhc_warmup_token_sizes(
            max_tokens=7,
            cudagraph_capture_sizes=[0, 6, 8, 64],
            hidden_size=4096,
            hc_mult=4,
            num_sms=48,
        )
        self.assertEqual(small, [1, 2, 4, 6, 7])
        self.deep_gemm_supported = False
        self.warmup.deepseek_v4_mhc_warmup(
            model(layer()), max_tokens=2048, cudagraph_capture_sizes=CAPTURE_SIZES
        )
        self.assertEqual(
            len([name for name, args, kw in self.calls if name == "pre"]), 17
        )
        large = self.warmup._select_mhc_warmup_token_sizes(
            max_tokens=32768,
            cudagraph_capture_sizes=[32768],
            hidden_size=4096,
            hc_mult=4,
            num_sms=48,
        )
        self.assertLessEqual(max(large), 16384)


if __name__ == "__main__":
    unittest.main()
