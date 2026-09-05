# DeepSeek V4 Flash vision on SM120/SM121 (experimental)

This branch backports upstream vLLM's multimodal wrapper to the pinned Spark
runtime. CPU, real processor/configuration and bounded CUDA component gates
pass. The first full TP2 test with text-only 0731 weights passed streaming and
tool calls but failed the text-speed gate. The candidate is not accepted; the
validated text image was restored. Full Vision-Exp serving remains untested.

## Text regression result (2026-09-05)

Candidate `f4596d54e122` was tested against control `8c97ddf50b07` using identical
0731 weights, DSpark with five speculative tokens, TP2, scheduler, cache budget,
benchmark client and prompts. Each measured trial requested 512 output tokens.
Only the serving image changed. All 14 measured requests completed correctly.

| Concurrent requests | Recorded control tok/s | Candidate tok/s | Change |
|---|---:|---:|---:|
| 1 | 50.27 | 42.58 | -15.3% |
| 2 | 68.85 | 69.10 | +0.4% |
| 4 | 98.67 | 100.71 | +2.1% |

Values are median aggregate output throughput across two trials, including TTFT
and client overhead. Differences below 3% are noise, not claimed improvements.
A bounded single-request confirmation returned 46.09 tok/s: 8.3% below the
recorded control. After rollback, two fresh control trials returned 48.54 tok/s;
the candidate confirmation was still 5.0% slower than that recovery check.
The limited samples fail the non-regression gate but do not establish its cause.

Both images captured full and piecewise target graphs and full DSpark graphs;
missing full-graph capture is not an explanation for this result. Candidate
prefill, prefix-cache and full vision-model tests were deferred after rejection.
No new image was built or deployed to fix this result during the test phase.
Do not replace a known-good text deployment with this candidate.

The subsequent [source and trace diagnosis](vision-regression-diagnosis.md)
found stable client inter-chunk cadence but different generated outputs and
chunk counts, including within control repeats. It adds text-dispatch tests
and a source-only decode callback reduction; neither establishes a root cause,
a measured speedup or acceptance of a new candidate.

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

The numerical canary requires `BASELINE_SPARSE_LIBRARY` and
`BASELINE_SPARSE_SHA256` pointing to the already-built control library. It
loads that verified binary directly, with no JIT fallback. The existing
single-cache case must match bit-for-bit. Dense-reference checks use the
pinned FlashInfer suite's `atol=0.05, rtol=0.05` for its quantized prefill.
The canary fails if peak test allocations exceed 128 MiB.

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
