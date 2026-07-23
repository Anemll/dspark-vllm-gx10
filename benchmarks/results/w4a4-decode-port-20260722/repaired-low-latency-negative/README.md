# Repaired SM121 low-latency CUTLASS result

This gate revisits FlashInfer's removed FP4 low-latency MoE path using real
prepared DeepSeek-V4 layer-0 weights and captured production routes. The path
is now numerically correct, but it is not a decode optimization for this
model.

## Repairs

The historical implementation had three independent defects for the
DeepSeek-V4 asymmetric W13/W2 shapes:

1. It retained the normal `M * top_k` expanded-row count after constructing an
   expert-major `M * active_experts` activation map.
2. GEMM2 wrote to the zero-sized low-latency `fc2_result_` workspace instead of
   the public final-output buffer.
3. W13 and W2 advanced by one obsolete shared weight stride instead of their
   distinct `gemm_n * gemm_k` strides.

The patch in `scripts/patch_flashinfer_cutlass_min_latency_sm121.py` repairs
all three contracts and fails closed if its pinned source anchors drift.

## Real-route results

All timings use tactic pair GEMM1=17/GEMM2=37. The comparison control is the
already-banked standard FlashInfer CUTLASS result on the exact same captured
route.

| Route | Active experts | Repaired low latency | Standard CUTLASS | Delta |
|---|---:|---:|---:|---:|
| sample 131 | 24 | 0.811808 ms | 0.792800 ms | **+2.40% slower** |
| sample 13 | 15 | 0.526656 ms | 0.510528 ms | **+3.16% slower** |

Correctness is strong:

- sample 131: cosine `0.9999998212`, NRMSE `0.0005785764`;
- sample 13: cosine `0.9999997616`, NRMSE `0.0005845016`;
- both outputs are finite and active.

These low-latency timings are still optimistic because they exclude the
required router-score reduction. The implementation computes `M` rows for
every active expert (60 rows for sample 13) rather than only the 24 actual
routed rows. It therefore cannot displace the standard routed-row CUTLASS
path for DeepSeek-V4 decode.

## Decision

**Rejected.** Do not wire this low-latency selector into serving. The
remaining W4A4 decode gap is in the small-M two-GEMM execution path, not in a
missing selector for this historical kernel.

