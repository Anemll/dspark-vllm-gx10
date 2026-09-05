# SPDX-License-Identifier: MIT
"""Real registry/config/processor integration, without model-weight loading."""
import json
import sys

from PIL import Image
from vllm.engine.arg_utils import EngineArgs
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.models.deepseek_v4.common.mm_preprocess import IMAGE_SENTINEL_BASE_ID


def main():
    model_path = sys.argv[1]
    config = EngineArgs(
        model=model_path, tokenizer_mode="auto", max_model_len=4096,
        max_num_batched_tokens=2048, max_num_seqs=2,
        tensor_parallel_size=2, nnodes=2,
        distributed_executor_backend="mp", moe_backend="flashinfer_b12x",
        kv_cache_dtype="nvfp4_ds_mla", block_size=256,
        limit_mm_per_prompt={"image": 2},
        speculative_config={"method": "dspark", "num_speculative_tokens": 3,
                            "draft_sample_method": "probabilistic"},
    ).create_engine_config()
    model = config.model_config
    assert model.architectures == ["DeepseekV4ForConditionalGeneration"]
    assert model.is_multimodal_model and model.is_mm_prefix_lm
    assert model.tokenizer_mode == "deepseek_v4"
    assert config.scheduler_config.disable_chunked_mm_input
    assert config.speculative_config.use_dspark()
    assert config.speculative_config.draft_model_config.architectures == ["DSparkDraftModel"]
    cache = MULTIMODAL_REGISTRY.processor_cache_from_config(config)
    processor = MULTIMODAL_REGISTRY.create_processor(model, cache=cache)
    images = [Image.new("RGB", (336, 252), "red"), Image.new("RGB", (252, 336), "blue")]
    outputs = []
    for prefix in ("a", "different prefix", "a"):
        inputs = ProcessorInputs(
            prompt=prefix + "<｜deepseek_image｜> between <｜deepseek_image｜>",
            mm_data_items=processor.info.parse_mm_data({"image": images}),
        )
        result = processor.apply(inputs, TimingContext())
        tokens = result["prompt_token_ids"]
        placeholders = result["mm_placeholders"]["image"]
        assert len(placeholders) == 2
        for item in placeholders:
            pad = 3 - item.offset % 4
            assert tokens[item.offset + pad] == IMAGE_SENTINEL_BASE_ID
            assert tokens[item.offset + item.length - 1] == IMAGE_SENTINEL_BASE_ID + 4
            assert item.length <= 384
        outputs.append(tokens)
    assert outputs[0] == outputs[2]
    print(json.dumps({"processor_requests": len(outputs), "images_per_request": 2,
                      "draft": config.speculative_config.draft_model_config.architectures,
                      "target": model.architectures}), flush=True)


if __name__ == "__main__":
    main()
