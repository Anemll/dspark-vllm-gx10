# Credits and provenance

This project is a downstream integration, not an independent implementation of
the underlying model, serving engine, speculative decoder, or GPU kernels. The
distinctions below are intentional: **direct code ancestry** is separated from
**recipe and validation lineage**.

## Direct code and runtime dependencies

### vLLM contributors — Apache-2.0

The files under `overlay/vllm/` are based on vLLM and retain vLLM's
Apache-2.0 SPDX and copyright notices.

- Upstream: https://github.com/vllm-project/vllm
- Pinned base: `752a3a504485790a2e8491cacbb35c137339ad34` (`v0.25.1`)
- License: Apache-2.0

### FlashInfer contributors — Apache-2.0

The runtime uses FlashInfer's native SM120/SM121 DeepSeek V4 sparse-MLA
kernel and API. FlashInfer source is not copied into this repository; the
reproducible image build fetches the pinned revision.

- Upstream: https://github.com/flashinfer-ai/flashinfer
- Pinned revision: `0472b9b3f2fba11b463f8526f390297d52a8aad7`
- License: Apache-2.0

### Luke Alonso / b12x — Apache-2.0

The native MXFP4 MoE path calls the b12x SM120/SM121 kernels. The tested
container source was compared against public b12x history and matches commit
`7dc6fb8fcc6446ea093537d1657df81985fa5f43` for every tracked source file.
That revision declares version `0.15.3` and license `Apache-2.0` in
`pyproject.toml`; it was never published as a PyPI `0.15.3` release, so this
project pins the Git commit rather than claiming a PyPI dependency.

- Upstream: https://github.com/lukealonso/b12x
- Pinned revision: `7dc6fb8fcc6446ea093537d1657df81985fa5f43`
- Author/maintainer: Luke Alonso (`lukealonso`)
- License declared by the pinned package metadata: Apache-2.0

The vLLM modular-MoE integration was also informed by `voipmonitor`'s public
Apache-2.0 vLLM work in
[vllm-project/vllm#39634](https://github.com/vllm-project/vllm/pull/39634).
The backend in this repository is substantially extended for the tested
DeepSeek V4 path, but that earlier integration deserves explicit credit.

## Model and serving lineage

### Keys / drowzeys

The optional abliterated model was created and published by Keys / drowzeys:

- Model: https://huggingface.co/drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored
- Model notes: https://github.com/drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored-1M-57toks
- Model card license at the time of this audit: MIT

No model weights are included or relicensed here.

Keys / drowzeys also authored the Apache-2.0
[DSpark concurrency patch](https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash)
used by the earlier serving stack. This vLLM 0.25 port does **not** copy that
patch's three overlay files; it relies on the pinned vLLM 0.25 DSpark code. The
patch remains an important correctness and validation reference.

### Earlier DSpark integration and two-node recipes

- Rafael Caricio — early DSpark vLLM integration:
  https://github.com/rafaelcaricio/vllm/pull/1 and
  https://github.com/rafaelcaricio/spark_vllm_docker/pull/1 (Apache-2.0 repos)
- MiaAI-Lab — two-node DGX Spark packaging:
  https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark (MIT)
- TonyD2Wild — the directly preceding two-node abliterated-model recipe:
  https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-Abliterated-Uncensored-2x-DGX-Spark (MIT)
- Fraser Price — DeepSeek V4 Flash DSpark model/runtime research:
  https://huggingface.co/fraserprice/DeepSeek-V4-Flash-DSpark and
  https://github.com/fraserprice/dspark-vllm

The Fraser Price GitHub repository did not declare a license when audited, so
it is credited as a research/reference source only; no code is copied from it.

### roady001

`roady001` identified the scheduler guard that fixed cold-resume garbling in
the earlier stack. That contribution should remain credited in descriptions of
the earlier deployment. This repository does not redistribute it as a separate
patch; the current build starts from the pinned upstream vLLM source.

## Work introduced in this repository

The new downstream work is the vLLM 0.25 NVFP4/DSpark bridge, the expanded
b12x modular-MoE adapter, two-node deployment and update tooling, the live
dashboard, and the benchmark packaging. Repo-local work is MIT licensed;
vLLM-derived overlay code remains Apache-2.0.
