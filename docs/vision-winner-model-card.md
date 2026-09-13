---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
pipeline_tag: image-text-to-text
library_name: vllm
tags:
  - deepseek-v4
  - multimodal
  - vision-language
  - dspark
  - dgx-spark
  - gb10
  - vllm
---

# DeepSeek V4 Flash Vision-Exp — DSpark on 2x GB10

This is a serving package for the complete official
[DeepSeek-V4-Flash-Vision-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp)
checkpoint on two NVIDIA GB10 systems with vLLM, B12X and DSpark speculative
decoding. The source checkpoint is pinned to revision
`6821d6ad3681a4b137b066b76094fa82ebd0a380`.

## Checkpoint and vision provenance

All deployed language and multimodal tensors are distributed together in
**DeepSeek AI's official Vision-Exp checkpoint**. This includes the language
model, vision tower, vision aligner, image sentinels and routing data,
tokenizer, and trained DSpark draft layers. No model merge, projector
transplant, fine-tuning, or new weight conversion was applied by this package.

That describes the files' immediate source, not an unsupported claim about
who first designed or pretrained every visual component. DeepSeek's published
model card says it incorporated visual modules and performed continued
training, but neither the card nor its reference `vision.py` identifies a GLM,
MoonViT, or other third-party donor for the tower or aligner. Record the
upstream origin as **not disclosed by the publisher** unless a primary source
or a tensor-identity audit establishes it. Do not add a GLM credit based only
on architectural resemblance.

This package does **not** contain weights from
[NVIDIA/DeepSeek-V4-Flash-0731-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4-Flash-0731-NVFP4).
That text-only NVIDIA conversion was evaluated as a possible 0731 backbone but
is not part of this accepted model. Do not describe this upload as NVIDIA
0731, W4A4, or an NVIDIA NVFP4 model.

The running checkpoint uses DeepSeek's original mixed layout: its config
declares FP8 E4M3 dynamic quantization with 128x128 blocks for the general
quantized weights, while the routed experts use source FP4/MXFP4 E2M1 values
with E8M0 group-32 scales. The serving KV cache is separately configured as
FP8. The rejected msuiche experiment was the candidate with NVIDIA-style
NVFP4/W4A4 routed experts.

## Credits

| Component | Credit and source | Role in this package |
|---|---|---|
| Published checkpoint: model, tokenizer, vision tower/aligner and DSpark draft | [DeepSeek AI — DeepSeek-V4-Flash-Vision-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp) | Complete, unmodified source checkpoint; earlier donor lineage of visual modules is not disclosed |
| Multimodal model support | [vLLM PR #54566, by Isotr0py](https://github.com/vllm-project/vllm/pull/54566) | Upstream DeepSeek V4 Flash Vision-Exp integration |
| GB10 vLLM runtime | [local-inference-lab/vLLM](https://github.com/local-inference-lab/vllm) | GB10/B12X serving fork, pinned in the runtime lock |
| GB10 kernels | [local-inference-lab/B12X](https://github.com/local-inference-lab/b12x) | Blackwell SM12x inference kernels |
| CUDA platform | [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) and NVIDIA CUDA/CUTLASS DSL | GB10 hardware and GPU software platform |
| Base runtime image | [eugr/spark-vllm-b12x](https://hub.docker.com/r/eugr/spark-vllm-b12x) | Reproducible B12X runtime base |
| Integration and validation | [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10) | Two-node deployment, dashboard, tests and accepted profile |

NVIDIA is credited for the hardware and CUDA software used by this deployment,
and for the separately evaluated 0731 NVFP4 reference. NVIDIA is **not** the
source of the model or vision weights in this upload.

The Baseten GLM-5.2 Vision work and community DeepSeek 0731 projector grafts
are useful research precedents, but they are separate models and are not a
declared source for this checkpoint.

## Accepted serving profile

The machine-readable profile is
[`config/vision-winner.lock.json`](../config/vision-winner.lock.json). Its main
settings are:

- tensor parallelism: 2 across two NVIDIA GB10 systems;
- DSpark probabilistic speculative decoding with 5 draft tokens;
- B12X target MoE, linear and attention backends;
- FP8 KV cache, fixed at 10 GiB per rank;
- 350,000-token served context limit;
- 4 active sequences and 2,048 batched tokens;
- full-and-piecewise CUDA graph mode;
- no explicit per-prompt image-count limit.

Omitting the image-count override restores this runtime's default allowance of
999 images, but the context window, image tokenization cost, memory, and
request-size limits remain the practical bounds. It does not promise that 999
full-resolution images are usable in one request.

## Validation scope

The accepted profile passed a three-image request, streaming generation, a
parsed tool call, dashboard prefill/decode phase reporting, and a clean fatal
error scan on both tensor-parallel ranks. These are deployment canaries, not a
general multimodal-quality evaluation. Do not infer benchmark, safety, or
accuracy claims beyond published result artifacts.

## License and notices

The official DeepSeek checkpoint declares the MIT license. Preserve its
license, model card, attribution, tokenizer files, configuration, and any
third-party notices when redistributing model files. Runtime components retain
their own licenses. Review the upstream repositories and applicable terms
before publishing a combined image or package.

## Suggested upload description

> Two-node GB10 serving package for the unmodified official DeepSeek AI
> DeepSeek-V4-Flash-Vision-Exp checkpoint, using vLLM, B12X and DSpark. All
> deployed tensors are sourced from DeepSeek's official release; the publisher
> does not disclose an earlier donor lineage for its visual modules. NVIDIA
> provides the DGX Spark/GB10 and CUDA platform; no NVIDIA 0731 NVFP4 model
> weights are included.
