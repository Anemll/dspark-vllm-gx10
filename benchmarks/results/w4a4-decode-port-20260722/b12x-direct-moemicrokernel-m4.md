# Audited direct B12X MoE microkernel: M=4 negative result

This gate tests whether B12X's evolved low-level `MoEMicroKernelBackend` can
recover the remaining target-only W4A4 decode gap.  It is the maintained
descendant of the orphaned FlashInfer `MoEDirectMicroKernel` source.  The
probe invokes it directly, bypassing the incompatible public B12X wrapper
ABI.  It uses real prepared DeepSeek V4 Flash
NVFP4 layer-0 rank-0 weights, the TP=2 rank-local shape, balanced `M=4` routing, the
checkpoint's E4M3/K16 scale contract, `w13` ordering, and the serving
SwiGLU clamp of 10.  The paired reference is the current FlashInfer CUTLASS
W4A4 path over the same inputs and weights.

## Result

| Execution | Direct B12X | FlashInfer CUTLASS | Direct/CUTLASS speed |
|---|---:|---:|---:|
| CUDA graph median | 1.186592 ms | **0.781968 ms** | 0.659003x |
| Eager median | 1.194176 ms | **0.787952 ms** | 0.659829x |

The direct kernel is numerically valid: cosine similarity is `0.99959594`
and normalized RMSE is `0.0284303`.  It nevertheless takes 51.7% more graph
latency and 51.6% more eager latency than CUTLASS.  The performance gate is
therefore rejected.

## Decision

The dormant direct-kernel source is not a hidden decode win for this W4A4
checkpoint.  Direct compilation closes the wrapper/dispatch ambiguity, but
the kernel remains about 34% lower in throughput than the current CUTLASS
reference.  Do not wire it into serving.  The remaining investigation should
focus on the materially different packed K32/E8M0 schedule or on a new packed
K16/E4M3 W4A4 layout, not on activating this direct kernel unchanged.

Raw evidence:

- `b12x-base-direct-final-m4.json`
- `b12x-base-direct-final-m4.log`
