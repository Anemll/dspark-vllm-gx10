# Native B12X W4A16 TP=2 serving result

This result closes the native-layout NVFP4 serving experiment at revision
`c55daaf5241a2daabfd89cdfca05b2d044b7adbc`.

## Configuration

- Target checkpoint: `DeepSeek-V4-Flash-NVFP4-TP2-CUTLASS-Prepared-v1`
- Target experts: losslessly collapsed E8M0/K32 scales and in-place
  native-packed FP4 weights
- Routed-expert backend: B12X W4A16 for every phase
- Tensor parallelism: two GB10 nodes
- Speculation: off for the accepted measurements
- KV cache: explicit 10 GiB per rank
- Served model: `deepseek-v4-flash-nvfp4-native-b12x-nodraft`

The target loaded 43/43 routed layers with
`duplicate_weight_bytes=0 weight_layout=packed`. Rank 0 reported 67.22 seconds
for checkpoint weights, 95.226 seconds for model loading, 82.14 GiB model
residency, and an 836,346-token KV cache.

## Target-only decode

The baseline is the already-banked, same-prompt, no-draft CUTLASS W4A4 result
in `../decode-w4a4-target-only-c1-c4.json`. Both arms use the canonical prompt,
temperature zero, 512 output tokens, and three trials.

| Concurrency | CUTLASS W4A4 best (median) | Native B12X W4A16 best (median) | Best delta | Median delta |
|---:|---:|---:|---:|---:|
| 1 | 27.03 (26.92) tok/s | **27.37 (27.34) tok/s** | **+1.27%** | **+1.57%** |
| 4 | 73.37 (72.55) tok/s | **78.01 (77.40) tok/s** | **+6.33%** | **+6.68%** |

The first trial of each arm includes cold/warmup effects. The two warm native
C4 trials were 78.01 and 77.40 tok/s.

Raw result: `../decode-native-b12x-target-only-c1-c4.json`.

## Prefill

Native W4A16 is faster than the preceding FP8/B12X production target at every
canonical size, but it does not retain the larger CUTLASS W4A4 prefill win.

| Input tokens | Previous FP8/B12X | CUTLASS W4A4 | Native B12X W4A16 | Native vs FP8 | Native vs W4A4 |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 2,033.0 | **2,242.5** | 2,044.6 tok/s | +0.57% | -8.82% |
| 2,048 | 2,252.0 | **2,473.2** | 2,283.8 tok/s | +1.41% | -7.66% |
| 4,096 | 2,320.7 | **2,659.3** | 2,387.0 tok/s | +2.86% | -10.24% |
| 8,192 | 2,184.2 | **2,593.5** | 2,322.3 tok/s | +6.32% | -10.45% |
| 16,384 | 2,203.8 | **2,501.7** | 2,269.5 tok/s | +2.98% | -9.28% |
| 32,768 | 2,176.1 | **2,477.3** | 2,210.7 tok/s | +1.59% | -10.76% |

Values are median server-computed prefill tokens/s over two measured trials.
Raw result: `../prefill-native-b12x-target-only.json`.

## Quality and DSpark isolation

Target-only native W4A16 passed deterministic short and longer-output probes:
exact `OK`, `Hello!`, arithmetic `4`, and a correct Python `add` function.
Six concurrent target-only requests were also correct.

DSpark is **not accepted** for this split checkpoint. Native W4A16 + the
abliterated draft corrupted selected prompts at both MTP=5 and MTP=1.
Repeating MTP=1 with the target restored to CUTLASS W4A4 produced the same
prompt-dependent corruption. This exonerates the native packing and identifies
the NVFP4-target/abliterated-draft speculative pairing (or its acceptance
integration) as a separate quality defect. The failed MTP=5 smoke response is
archived under `dspark-invalid/`.

## Decision

Native-packed B12X W4A16 is the accepted **decode-optimized target-only mode**:
it closes the C4 decode regression while preserving one FP4 payload and adding
only scale sidecars.

It is not a universal replacement for CUTLASS W4A4 prefill. Native packing is
an actual nibble/tile permutation performed in place; CUTLASS cannot consume
that payload through a stride-only view. A single simultaneously optimal
prefill/decode payload therefore requires a new CUTLASS reader for the native
packing (or another kernel that is fast in both phases), not merely another
scale tensor.

