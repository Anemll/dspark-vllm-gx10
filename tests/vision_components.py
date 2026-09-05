# SPDX-License-Identifier: MIT
"""Real pinned-vLLM component imports and CPU semantics; no GPU allocations."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import ExitStack

import torch
from PIL import Image
from vllm.models.deepseek_v4.common.mm_preprocess import (
    IMAGE, IMAGE_PAD, IMAGE_SENTINEL_BASE_ID, IMAGE_START, IMAGE_END,
    DeepseekV4VLImageProcessor, build_image_block_pad_free, align_image_placeholders,
)
from vllm.models.deepseek_v4.common.vision import (
    get_vision_cos_sin, apply_rotary, DeepseekV4ViT, DeepseekV4Aligner,
)
from vllm.models.deepseek_v4.nvidia.vl_model import DeepseekV4ForConditionalGeneration
from vllm.models.deepseek_v4.nvidia.vision_router import DeepseekV4VisionRouter
from vllm.multimodal.processing.processor import PlaceholderFeaturesInfo
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config
from vllm.transformers_utils.model_arch_config_convertor import DeepseekV4ModelArchConfigConvertor
from vllm.v1.worker.gpu.attn_utils import compute_mm_prefix_ranges
from vllm.tokenizers.deepseek_v4_encoding import encode_messages, flatten_content_blocks


class VisionComponents(unittest.TestCase):
    def test_real_vision_constructor_and_tiny_forward(self):
        from vllm.config import VllmConfig, DeviceConfig, CompilationConfig, set_current_vllm_config
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        cfg = DeepseekV4Config(vision_n_layers=1, vision_dim=32, vision_n_heads=4,
                              vision_inter_dim=32, hidden_size=32)
        # Supply single-rank CPU infrastructure, but run actual constructors,
        # actual vLLM parallel linears, rotary, attention and aligner code.
        with ExitStack() as stack:
            stack.enter_context(set_current_vllm_config(VllmConfig(
                device_config=DeviceConfig(device="cpu"),
                compilation_config=CompilationConfig(mode=0, custom_ops=["none"]),
            )))
            stack.enter_context(patch("vllm.distributed.parallel_state._TP",
                                      SimpleNamespace(rank_in_group=0, world_size=1)))
            for name in ("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
                         "vllm.models.deepseek_v4.common.vision.get_tensor_model_parallel_world_size"):
                stack.enter_context(patch(name, return_value=1))
            stack.enter_context(patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", return_value=0))
            stack.enter_context(patch("vllm.model_executor.layers.attention.mm_encoder_attention.get_vit_attn_backend",
                                      return_value=AttentionBackendEnum.TORCH_SDPA))
            tower, aligner = DeepseekV4ViT(cfg), DeepseekV4Aligner(cfg)
            for param in list(tower.parameters()) + list(aligner.parameters()):
                torch.nn.init.normal_(param, std=0.02)
            output = aligner(tower(torch.randn(16, 3, 14, 14), 4, 4), 4, 4)
            self.assertEqual(output.shape, (4, 32))
            self.assertTrue(torch.isfinite(output).all())

    def test_real_image_transform_budget_and_patch_layout(self):
        processor = DeepseekV4VLImageProcessor(DeepseekV4Config(vision_n_layers=32))
        for width, height in ((1024, 701), (450, 308), (64, 4096), (2000, 100), (1, 1)):
            patches, vh, vw, lh, lw = processor(Image.new("RGB", (width, height), "red"))
            types, perm = build_image_block_pad_free(lh, lw)
            self.assertEqual(patches.shape, (vh*vw, 3, 14, 14))
            self.assertEqual(patches.dtype, torch.bfloat16)
            self.assertLessEqual(len(types) + 3, 384)
            self.assertEqual(len(perm), lh*lw)

    def test_architecture_routing_and_no_draft_recursion(self):
        for vision, arch, expected in ((0, "DeepseekV4ForCausalLM", "DeepseekV4ForCausalLM"),
                (32, "DeepseekV4ForCausalLM", "DeepseekV4ForConditionalGeneration"),
                (32, "DSparkDraftModel", "DSparkDraftModel")):
            cfg = DeepseekV4Config(vision_n_layers=vision, architectures=[arch])
            converter = DeepseekV4ModelArchConfigConvertor(cfg, cfg)
            self.assertEqual(converter.get_architectures(), [expected])
            self.assertEqual(converter.is_mm_prefix_lm(), expected.endswith("ConditionalGeneration"))
        cfg = DeepseekV4Config(vision_n_layers=32, architectures=["DeepseekV4ForCausalLM"])
        cfg._dsv4_vl_inner = True
        self.assertEqual(DeepseekV4ModelArchConfigConvertor(cfg, cfg).get_architectures(),
                         ["DeepseekV4ForCausalLM"])

    def test_multiple_image_alignment_all_offsets(self):
        for start in range(4):
            for h, w in ((2, 3), (3, 5), (8, 12), (14, 21)):
                types, perm = build_image_block_pad_free(h, w)
                block = (types + IMAGE_SENTINEL_BASE_ID).tolist()
                items = [PlaceholderFeaturesInfo("image", 0, start, block,
                          types == IMAGE), PlaceholderFeaturesInfo("image", 1,
                          start + len(block) + 5, block, types == IMAGE)]
                prompt = [7]*start + block + [8]*5 + block + [9]*7
                original = copy.deepcopy(prompt)
                result, out = align_image_placeholders(prompt, {"image": items})
                self.assertEqual(prompt, original)
                for p in out["image"]:
                    pad = 3 - p.start_idx % 4
                    self.assertEqual((p.start_idx + pad) % 4, 3)
                    self.assertEqual(p.tokens[pad], IMAGE_SENTINEL_BASE_ID + IMAGE_START)
                    self.assertEqual(p.tokens[-1], IMAGE_SENTINEL_BASE_ID + IMAGE_END)
                    self.assertEqual(int(p.is_embed.sum()), h*w)
                    self.assertEqual(result[p.start_idx:p.start_idx+p.length], p.tokens)
                self.assertEqual(sorted(perm.tolist()), list(range(h*w)))

    def test_v2_whole_span_includes_sentinels_not_alignment_pad(self):
        for offset in range(4):
            pos = SimpleNamespace(offset=offset, length=384,
                                  extract_embeds_range=lambda: [(10, 20), (22, 30)])
            features = {"a": [SimpleNamespace(modality="image", mm_position=pos)]}
            ranges = compute_mm_prefix_ranges(["a"], features, 128,
                         clamps_in_kernel=True, span_leading_pad_modulus=4)
            self.assertEqual(ranges, {0: [(3, offset+383)]})
            self.assertEqual(compute_mm_prefix_ranges(["a"], features, 128),
                             {0: [(10, 20), (22, 30)]})

    def test_text_rendering_unchanged_for_plain_strings(self):
        self.assertEqual(flatten_content_blocks("unchanged"), "unchanged")
        self.assertEqual(flatten_content_blocks([{"type":"text", "text":"a"},
              {"type":"image_url", "image_url":{"url":"unused"}},
              {"type":"text", "text":"b"}]), "a<｜deepseek_image｜>b")
        prompt = encode_messages([{"role":"user", "content":"hello"}], thinking_mode="chat")
        self.assertIn("hello", prompt)

    def test_rotary_preserves_norm(self):
        cos, sin = get_vision_cos_sin(2, 3, 4, 10000.)
        x = torch.randn(6, 2, 8)
        actual = apply_rotary(x, cos, sin)
        torch.testing.assert_close(actual.square().sum(-1), x.square().sum(-1), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(VisionComponents)
    )
    raise SystemExit(not result.wasSuccessful())
