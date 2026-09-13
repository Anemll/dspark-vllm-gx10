# NVIDIA 0731 NVFP4 vision-transplant result

Date: 2026-09-13. Result: **rejected for vision correctness**. This was an
experimental Anemll hybrid, not an NVIDIA or DeepSeek release.

## Exact candidate

- Branch: `nvidia-nvfp4-0731-vision-transplant`
- Assembly/runtime commit after the startup fix:
  `00bd896b04e5aeff293b86794434f1f08ffe4580`
- Runtime image (identical on both ranks):
  `sha256:990f5efe129e407da0d48af75005040b0e745970818f8d564591b07767bb1709`
- Target: `nvidia/DeepSeek-V4-Flash-0731-NVFP4` revision
  `f1caa71142bd0be02f728c79f75042ac1e461579`
- Vision donor: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` revision
  `6821d6ad3681a4b137b066b76094fa82ebd0a380`
- Target expert path: ModelOpt NVFP4, native B12X NVFP4 MoE (W4A4 routed
  experts). DSpark draft: original NVIDIA `mtp.*` tensors on B12X
  MXFP4/MXFP8, K5. KV cache: FP8. Vision tensors: donor precision.
- Hybrid payload: the 48 NVIDIA shards hard-linked unchanged plus one donor
  shard containing 316 vision-specific tensors (932,836,352 payload bytes).

Both nodes produced the same hybrid hashes:

| Artifact | SHA-256 |
| --- | --- |
| Hybrid config | `a8b6dcd89cabac993b57af92e0d840597d2c441be6ebedd17c265a83ab055bab` |
| Hybrid index | `25b5295fafa1cb69bf4653fe06f2131ff240a8d62972aaf8810d788c5cb8c9de` |
| Vision donor shard | `d656e981e9afdcd97f88af432728e9630136b3fe8493d96cab0c82669fb00850e` |
| Merged tokenizer | `e4837c71e00fcfbe089d2d91ebdcb7974aa5531847e3cbe6704668dc1e928b9c` |

## Startup finding and fix

The first launch loaded all target, vision, and draft weights but failed during
multimodal profiling because the NVIDIA tokenizer did not contain
`<｜deepseek_image｜>`. The two source tokenizers have an identical BPE model
and identical tokenizer configuration, but eight reserved-token labels differ.

The assembler was changed to pin both tokenizer files and transplant only the
required label at token ID 129264 (`<｜image2｜>` to
`<｜deepseek_image｜>`). It deliberately preserves NVIDIA's other seven labels,
including its System/table tokens. All 113 repository tests passed (two optional
Pillow fixture tests skipped). The corrected candidate then loaded, completed
multimodal profiling and warmup, captured CUDA graphs, and served normally.

## Matched text gate

The control and hybrid used the same runtime, target/draft weights, DSpark K5,
FP8 KV cache, 350K context, scheduler limits, prompt, output length, and two
C1 trials. Values below are best aggregate throughput from a 512-token run.

| Candidate | C1 aggregate tok/s | C1 output tok/s | TTFT | Cumulative DSpark acceptance |
| --- | ---: | ---: | ---: | ---: |
| NVIDIA 0731 text control | 45.249 | 46.286 | 0.253 s | 685/1695 = 40.41% |
| NVIDIA 0731 + vision transplant | 44.752 | 45.706 | 0.238 s | 706/1780 = 39.66% |

The best aggregate difference was -1.10%, below the predeclared 3% regression
threshold. Text speed therefore passed this small canary. It is not a full
C1-C4 benchmark because the mandatory vision gate failed next.

## Vision correctness gate

The first known-answer OCR fixture expected:

```json
{"code":"Q7M4","units":37,"destination":"Oslo"}
```

The hybrid returned:

```json
{"code":"81f069256ab3b2bfd807","units":1,"destination":"Benchmark"}
```

All three fields were wrong. The values were copied from the benchmark's text
prefix instead of read from the image. Server metrics proved that the request
was processed as a 406-token multimodal request; the response was not a client
transport failure. The benchmark intentionally stopped after this first
mandatory failure, so no full visual or C1-C4 sweep was run.

## Conclusion

The transplant is mechanically loadable and preserves text speed, but it does
not provide usable vision. The September 6 compatibility audit already showed
that the NVIDIA 0731 target is an exact quantization of the 0731 text model for
sampled shared tensors, while every sampled shared tensor differs from
Vision-Exp—including embeddings, normalization, attention/FFN parameters, and
the language head. The live failure confirms that matching dimensions and
copying the 316 modality-specific tensors do not preserve the trained
vision/language alignment.

The viable next path is to quantize the complete trained Vision-Exp checkpoint
to NVIDIA-style NVFP4 while preserving its language, draft, and vision weights
together. Copying progressively more BF16 language tensors into the NVIDIA
checkpoint would no longer be the requested 0731 W4A4 target and would create
an unbounded, uncalibrated mixed model.

The rejected checkpoint, image, raw benchmark JSON, metrics, and both-rank logs
were retained for diagnosis. The accepted official Vision-Exp service was
restored after the gate failure.
