# DeepSeek V4 Flash vision on SM120/SM121 (experimental)

This branch backports upstream vLLM's multimodal wrapper to the pinned Spark
runtime. It is under validation: component tests are not an end-to-end model
or performance pass. Do not replace a known-good text deployment on that basis.

## Implementation

- Checkpoints with `vision_n_layers > 0` select the multimodal architecture;
  text-only checkpoints retain the original model and router.
- A TP-sharded vision encoder and aligner produce image embeddings. Original
  sentinel IDs reach the image-specific `bias_vl` router, including DSpark.
- Image blocks are compressor-aligned and scheduled atomically, including
  encoder-cache hits. Image attention sees the complete image block.
- Weight loading streams the language model iterator; it never materializes
  the checkpoint into a list. This is essential on the Sparks' shared memory.
- A separately named, prebuilt FlashInfer module adds primary topk512 prefill
  for H32/H64 and compressed page64/page2 caches. It does not replace compiled
  text kernels. The binary is architecture-checked and SHA256-verified; there
  is no serving-time build fallback.

## Candidate configuration

Build `docker/Dockerfile.vision` once from an exact validated text image ID,
with an immutable source-revision tag and `VISION_CUDA_ARCH_LIST=12.1a` for
GB10. Transfer the identical image over the dedicated fabric. Keep both
rollback images and role-specific configurations intact.

Use the `DeepSeek-V4-Flash-Vision-Exp` checkpoint and `method=dspark`, not
`mtp`. Its three draft layers require a separately verified draft setup; do
not copy the 0731 checkpoint's speculative configuration blindly. Keep the V2
runner and at least 387 scheduled tokens available for one image block.
The configuration validator disables partial image chunking. Text chunks
remain enabled. No repository default model or live environment is changed.

## Required acceptance evidence

First run the CPU components, GPU routing/visibility canary and prebuilt
image-prefill numerical canary. Then verify real configuration/processor
loading before loading the full model on both ranks.

Text non-regression must compare the same 0731 weights, DSpark settings,
prompts, seeds, cache budget and scheduler against the existing control.
Vision requires separate image-content, multiple-image, cached-prefix,
streaming, tool-call and both-rank stability checks. Repeatable regressions
over 3% reject a candidate. A different checkpoint's speed is not a matched
text A/B result.

## Sources

- [Upstream vLLM vision support, PR 54566](https://github.com/vllm-project/vllm/pull/54566)
- [Upstream deployment recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp)
- [Anemll SGLang two-Spark vision implementation](https://github.com/Anemll/SGLang-DSv4F-vision-2xSparks)
- [FlashInfer topk512 proposal, superseded by PR 4802](https://github.com/flashinfer-ai/flashinfer/pull/4850)

The SGLang recipe uses B12X/SM120 on GB10; it is not evidence of a separate
SM121-only kernel that can be copied unchanged into vLLM.
