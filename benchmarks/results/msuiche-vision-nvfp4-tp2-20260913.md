# Vision-Exp NVFP4 on two Sparks: vision passes, DSpark rejected

Date: 2026-09-13. Branch: `nvidia-nvfp4-0731-vision-transplant`.

## Outcome

The pinned community conversion
`msuiche/DeepSeek-V4-Flash-Vision-Exp-NVFP4@a807708335b28da39b629c38377b3d169827b360`
has working vision on two Sparks. It passed the complete bounded visual canary
both without speculation and with DSpark K5. It is nevertheless rejected for
the current deployment because DSpark acceptance collapsed and matched C1 text
decode regressed by 64.7%.

This is **not** NVIDIA's official 0731 checkpoint. Its language weights come
from the jointly trained DeepSeek Vision-Exp release. Only the routed experts
are NVFP4/W4A4. Attention, shared experts, head, MTP, embeddings, vision tower,
aligner, sentinels, and vision-routing biases remain in their source formats;
the KV cache is FP8.

## Integrity and runtime gates

- All 42 shards (176,483,421,216 bytes) were staged on local NVMe on both
  ranks and every LFS SHA-256 passed.
- All 6,585 non-expert tensors (20,641,634,040 bytes) match official
  Vision-Exp byte-for-byte.
- Four sampled early/middle/final routed experts cover 100,663,296 decoded
  logical values; their source MXFP4 to candidate NVFP4 transcodes are exact.
- All 33,024 input-scale tensors exist, but 32,421 differ from NVIDIA's base
  export and 32,878 differ from NVIDIA 0731. These are calibration artifacts,
  not consequences of exact weight relayout.
- Startup selected `DeepseekV4ForConditionalGeneration`, detected ModelOpt
  NVFP4, and used the native B12X NVFP4 MoE backend on both ranks.

## Functional results

| Phase | Vision suite | Same-prompt image swap | Streaming | Tool call |
| --- | ---: | ---: | ---: | ---: |
| Target-only | 10/10 | 4/4 | pass | pass |
| DSpark K5 | 10/10 | 4/4 | pass | pass |

The five visual categories were OCR, chart, document, spatial layout, and
two-image comparison. The explicit image-dependence control used the identical
prompt for two warehouse cards and returned the exact distinct answers A/17
and B/29 on first-touch and repeat requests.

## Matched C1 rejection

Both measurements used the same 512-token `upstream-explanation` request,
seed, runtime image, FP8 KV cache, scheduler limits, and DSpark K5 policy.

| Measurement | Accepted Vision-Exp | NVFP4 candidate | Change |
| --- | ---: | ---: | ---: |
| Aggregate decode | 40.86 tok/s | 14.41 tok/s | -64.7% |
| Server decode | 41.71 tok/s | 14.48 tok/s | -65.3% |
| DSpark acceptance | 32.73% | 0.318% | -32.41 pp |
| Server prefill | 136.31 tok/s | 143.85 tok/s | +5.5% |

Candidate trials were 14.47 and 14.34 aggregate tok/s. They needed 501 and
505 draft steps for 512 output tokens, accepting only 10 and 6 draft tokens.
The accepted trials needed 195 and 193 draft steps and accepted 317 and 318
tokens. Comparable prefill, healthy clocks, and zero throttle flags exclude
the prompt path, NCCL/RoCE, and frequency degradation as explanations for this
specific regression.

The proven proximate cause is draft/target disagreement. Because the decoded
expert weights and every preserved MTP tensor match their Vision-Exp sources,
the leading hypothesis is an activation-precision/calibration mismatch between
the NVFP4 target path and the preserved MXFP4 MTP path. That hypothesis still
needs target-only text timing and target/draft logit-overlap instrumentation;
this canary alone does not prove which scale or layer causes the disagreement.

## Decision and next experiment

Do not start projector training: vision already works. Do not promote this
checkpoint with DSpark. The next bounded tranche is:

1. Measure candidate target-only C1 and text quality to separate target speed
   and quality from speculative overhead.
2. Capture target/draft top-k overlap and per-stage acceptance on fixed text,
   then compare accepted MXFP4 with candidate NVFP4 hidden/logit drift.
3. Recalibrate routed-expert activation input scales on pinned mixed text and
   image samples, with DSpark acceptance as an explicit validation metric.
4. If scale-only recalibration cannot restore agreement, test an MTP-aware
   conversion of the three draft layers before considering projector/remap SFT.

## Recovery

The rejected candidate was stopped head-first, then worker. The accepted
Vision-Exp worker was started before the head. The API and dashboard are healthy,
both ranks use image ID
`sha256:8deadf84e3960f1df47843f7a04a4fea8e1467f2688e144cd065dd7e9331725d`,
the original model mount is restored, and OCR, streaming, parsed tool calls,
and C1 recovery checks pass. Raw immutable evidence is retained under
`.local/results/msuiche-q0-20260913T175512Z`.

The machine-readable companion is
[msuiche-vision-nvfp4-tp2-20260913.json](msuiche-vision-nvfp4-tp2-20260913.json).
