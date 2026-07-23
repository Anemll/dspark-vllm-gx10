# SM121 low-latency and split-stream decode gates

These are target-only, rank-0, real prepared layer-0 / captured C4-route gates.
They investigate decode-only replacements for the current FlashInfer CUTLASS
NVFP4 route.  The serving baseline is unchanged.

## Current reference

- FlashInfer CUTLASS M=4 CUDA graph: **0.779912 ms**.
- Canonical TP=2 service: FP8/B12X **76.8534 tok/s**, W4A4/CUTLASS
  **73.6563 tok/s** at C4 (W4A4 is 4.2% slower).

## Restored low-latency path

FlashInfer commit `20435b4` removed the SM120 low-latency descriptor builder
and replaced it with an unconditional throw.  The source-pinned patch restores
that builder against FlashInfer 0.6.15.  The corrected incremental build is:

- patched header SHA-256:
  `db3bfc93e0a969a412882853666d7b7af92e123b2c1b47afa5b4a988878fa019`
- fused-MoE shared object SHA-256:
  `6f1f6d7de9c1a20e18ff2ade4cbbf1cd93ffeafae44d091fd32c43f60c9a4eed`

The low-latency-specific autotuner selected GEMM1 tactic 17 and GEMM2 tactic
37.  The raw call measured **0.730400 ms**, but the output was invalid: one
token column was unwritten and five selected output values were non-finite.
An earlier tactic pair reached **0.717504 ms** with the same incomplete-output
failure.  These numbers are not performance wins because they do incomplete
work.  The path is rejected and must not be integrated.

## Split-stream alternatives

| Route | Median | Numeric result | Decision |
|---|---:|---|---|
| 4 independent M=1 streams | 0.814240 ms | cosine 1.0, NRMSE 0.0 | reject, 4.4% slower |
| 2 independent M=2 streams | 0.794528 ms | cosine 0.9999909, NRMSE 0.0042707 | reject, 1.9% slower |

Both alternatives execute complete output and confirm that GB10 cannot hide
the duplicated route/workspace overhead sufficiently to beat monolithic M=4.

## Conclusion

No tested dispatch/layout substitution closes the C4 serving gap.  The matched
profiler localizes roughly 41--42 microseconds per routed layer to the W4A4
expert path, dominated by GEMM1 and GEMM2.  Closing the remaining gap requires
a new correct decode-specialized fused expert kernel/epilogue, not a backend
selector, route compaction, tactic change, or stream split.
