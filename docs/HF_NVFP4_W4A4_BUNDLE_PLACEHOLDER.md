---
pipeline_tag: text-generation
base_model:
  - nvidia/DeepSeek-V4-Flash-NVFP4
  - deepseek-ai/DeepSeek-V4-Flash-DSpark
license: mit
library_name: vllm
tags:
  - deepseek-v4
  - nvfp4
  - w4a4
  - dspark
  - dgx-spark
  - tensor-parallel
---

# DeepSeek-V4-Flash NVFP4 TP2 W4A4 v1

This is an initial packaging card. A complete model card, runtime repository
reference, launch commands, benchmark methodology, and limitations will be
added in a later revision.

The repository is a **single-download bundle** for two NVIDIA DGX Spark / GB10
nodes:

- the repository root contains the offline-prepared, tensor-parallel-2 NVIDIA
  NVFP4 target checkpoint used with W4A4 FlashInfer CUTLASS MoE kernels;
- `dspark/` contains the original three-stage DeepSeek-V4-Flash-DSpark draft
  (`mtp.0`, `mtp.1`, and `mtp.2`) as shards 46, 47, and 48, with a filtered
  draft-only safetensors index.

The target and draft remain two explicit model paths at runtime: use the
repository root as the target and `dspark/` as the speculative draft. The
prepared target requires the matching custom vLLM loader contract recorded in
`dspark-nvfp4-tp2-repack.json`; it is not a drop-in Transformers checkpoint.

The prepared W4A4 target load path and target-only prefill benchmarks have
been validated. The bundled W4A4-target + DSpark-draft combination should be
treated as experimental until its final end-to-end serving validation is
published.

