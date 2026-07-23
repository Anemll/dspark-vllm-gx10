# W4A4 target-only decode: B12X occupancy A/B

## Decision

Select `DSPARK_B12X_MICRO_MAX_ACTIVE_CLUSTERS=40` for the experimental
FlashInfer B12X W4A4 decode path. It is the best tested full-serving occupancy
point, but it does **not** close the complete FP8/B12X decode gap.

All API comparisons use the same 35-token canonical prompt
(`sha256:652af3aabacfd4360432d28e0c237e9e445f938d032a604d3a4f7f42a2a7ed38`),
temperature 0, 512 output tokens per request, TP=2, target only, and MTP off.
The MAC36/MAC40 rows are fully warmed measurements; one-time lazy-compilation
trials were excluded before the decision set.

| Backend / setting | C1 median | C4 trials | C4 median | Delta vs W4A4 CUTLASS | Delta vs FP8 |
|---|---:|---:|---:|---:|---:|
| FP8, native B12X | 27.373 tok/s | 65.386 cold, 76.853, 77.486 | 76.853 tok/s | +4.99% | control |
| W4A4, CUTLASS | — | 73.298, 72.841, 73.204 | 73.204 tok/s | control | -4.75% |
| W4A4, B12X uncapped | — | 73.193, 72.641, 72.135 | 72.641 tok/s | -0.77% | -5.48% |
| W4A4, B12X MAC36 | — | 73.061, 73.370, 73.473 | 73.370 tok/s | +0.23% | -4.53% |
| **W4A4, B12X MAC40** | **27.008 tok/s** | **72.972, 74.076, 73.855** | **73.855 tok/s** | **+0.89%** | **-3.90%** |

MAC40 improves the uncapped B12X median by 1.67%. Its C1 result is 1.33%
below the matched FP8 control, so the occupancy cap does not trade away a
material single-stream win.

## Real-route component gate

The route artifact contains all 2,752 `(decode step, routed layer)` C4
samples, each shaped `[4, 6]`. Both TP ranks captured byte-identical route IDs.
The full-distribution gate replays each route through one prepared physical
layer and alternates B12X/CUTLASS graph order.

| B12X setting | B12X mean | CUTLASS mean | Mean CUTLASS/B12X | B12X-faster samples |
|---|---:|---:|---:|---:|
| default | 543.011 us | 551.280 us | 1.0164x | 2,190 / 2,752 |
| MAC44 | 545.148 us | 550.685 us | 1.0118x | 1,911 / 2,752 |
| **MAC40** | **541.021 us** | **552.628 us** | **1.0225x** | **2,309 / 2,752** |
| MAC36 | 542.964 us | 551.023 us | 1.0163x | 2,067 / 2,752 |

The earlier median-route sample overstated the average layer advantage:
sample 22 was 1.0364x on TP rank 0 and 1.0414x on TP rank 1. The complete
distribution is the correct projection and explains why the full-service gain
is only about one percent.

## What the result means

- Reserving eight GB10 SMs reduces TP arrival/collective contention enough to
  turn B12X from a small service regression into a repeatable ~1% win over the
  current W4A4 CUTLASS path.
- The remaining ~3.9% C4 gap to FP8 cannot be recovered by occupancy tuning.
- Route imbalance, TP-rank weight slices, output copying, route compaction,
  CUTLASS tactics, W4A16 fallback, and the existing direct/resident kernels
  have all been ruled out by earlier gates.
- Closing the residual gap requires a materially better SM121 W4A4 expert
  schedule or a fused whole-layer path whose gain survives the full CUDA graph;
  more wrapper/MAC tuning is below the worthwhile threshold.

## Immutable evidence

- Selected adapter source SHA-256:
  `5cc787c52510e614be63b62d2e49f9b8e0c6fc4494a0e8578e7d72250e3f9058`.
- The prepared loader pins that exact runtime source digest before bypassing
  ordinary B12X post-load transforms.
- HEAD diagnostic image:
  `sha256:45cc3a5f9bc6b2ed8ce39d242971ae0c258162a788076474f8ad2d5703e5c2b8`.
- WORKER diagnostic image:
  `sha256:65db356397d0101f0e4fe1331fb985ac4a8b2b3e2271fa1ac79c922661b61d0e`.
- The two diagnostic image config IDs differ because the minimal child images
  were assembled independently; both consume the same base image and the
  exact source hashes above. A promoted image must be built once and transferred
  unchanged to both ranks.

Raw JSON files in this directory are the source of every number above.
