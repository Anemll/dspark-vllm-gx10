# DSpark Mode-A concurrency and draft/verify timing

## Result

The approved TP=2 run completed all 15 cases on the five handoff prompts at
concurrency 1, 4, and 8. Mean aggregate decode throughput across prompts rose
from **69.28 tok/s** at C=1 to **147.57 tok/s** at C=4 (**2.14x**) and
**212.42 tok/s** at C=8 (**3.09x**). Peak observed throughput was
**293.77 tok/s** for `tool_agentic` at C=8.

This proves useful server-level Mode-A throughput scaling, but not a causal
draft/verify overlap win versus no-draft: this cost-bounded outage intentionally
used one DSpark-ON model load and did not run the secondary no-draft arm.
Per-stream throughput fell from a five-prompt mean of 69.42 tok/s at C=1 to
38.50 at C=4 and 28.53 at C=8.

The required `long_code_html` quality gate passed for all 13 generated streams
(C=1/4/8). Their repeated-four-gram maxima were 1-3 and unique-four-gram ratios
were 0.981-1.000. The gate was deliberately scoped to this prompt; some other
prompt families triggered the diagnostic repetition heuristic, so this result
must not be described as a universal quality evaluation.

## Minimal report

All rows are DSpark-ON, MTP=5, confidence OFF, probabilistic draft sampling,
temperature 0, and 512 requested output tokens. `serial` is the matching
prompt's C=1 aggregate decode rate. Phase timings are the mean of TP ranks 0
and 1. `p_ready` is not defined in this confidence-OFF run.

| Prompt | Context tokens | C | Serial tok/s | Aggregate tok/s | Speedup | Tau | p_full | Draft ms | Verify ms | Verify slowdown | Commit ms | TP diagnostic ms | Quality gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| short_code | 19 | 1 | 65.47 | 65.47 | 1.00x | 4.328 | 0.521 | 7.953 | 53.822 | 0.0% | 2.635 | 0.0170 | pass |
| short_code | 19 | 4 | 65.47 | 145.11 | 2.22x | 4.233 | 0.468 | 11.045 | 95.601 | 77.6% | 3.517 | 0.0248 | pass |
| short_code | 19 | 8 | 65.47 | 200.67 | 3.07x | 4.119 | 0.431 | 12.766 | 125.437 | 133.1% | 3.805 | 0.0250 | pass |
| long_code_html | 69 | 1 | 71.87 | 71.87 | 1.00x | 4.830 | 0.566 | 8.034 | 55.194 | 0.0% | 2.634 | 0.0184 | pass |
| long_code_html | 69 | 4 | 71.87 | 159.88 | 2.23x | 4.559 | 0.510 | 11.201 | 93.812 | 70.0% | 3.516 | 0.0261 | pass |
| long_code_html | 69 | 8 | 71.87 | 236.27 | 3.29x | 4.498 | 0.502 | 13.432 | 123.546 | 123.8% | 3.903 | 0.0271 | pass |
| tool_agentic | 40 | 1 | 75.83 | 75.83 | 1.00x | 5.150 | 0.700 | 7.998 | 55.589 | 0.0% | 2.631 | 0.0205 | pass |
| tool_agentic | 40 | 4 | 75.83 | 184.75 | 2.44x | 5.221 | 0.728 | 11.054 | 93.115 | 67.5% | 3.555 | 0.0289 | pass |
| tool_agentic | 40 | 8 | 75.83 | 293.77 | 3.87x | 5.099 | 0.680 | 12.954 | 117.573 | 111.5% | 3.942 | 0.0261 | pass |
| long_context_retrieval | 46 | 1 | 87.27 | 87.27 | 1.00x | 5.516 | 0.882 | 7.834 | 51.440 | 0.0% | 2.624 | 0.0181 | pass |
| long_context_retrieval | 46 | 4 | 87.27 | 152.63 | 1.75x | 4.835 | 0.684 | 10.830 | 94.840 | 84.4% | 3.411 | 0.0247 | pass |
| long_context_retrieval | 46 | 8 | 87.27 | 191.43 | 2.19x | 4.461 | 0.553 | 12.672 | 123.259 | 139.6% | 3.633 | 0.0270 | pass |
| json_structured | 41 | 1 | 45.96 | 45.96 | 1.00x | 3.078 | 0.175 | 7.837 | 55.350 | 0.0% | 2.622 | 0.0190 | pass |
| json_structured | 41 | 4 | 45.96 | 95.48 | 2.08x | 2.846 | 0.119 | 10.874 | 99.769 | 80.3% | 3.557 | 0.0252 | pass |
| json_structured | 41 | 8 | 45.96 | 139.97 | 3.05x | 2.905 | 0.127 | 12.905 | 136.688 | 147.0% | 3.875 | 0.0258 | pass |

Block-weighted both-rank means were:

| C | Draft ms | Verify ms | Commit ms | TP diagnostic ms | Other overhead ms | Total ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.924 | 54.428 | 2.629 | 0.0186 | 1.094 | 66.093 |
| 4 | 10.986 | 95.937 | 3.515 | 0.0258 | 1.621 | 112.083 |
| 8 | 12.925 | 126.503 | 3.824 | 0.0262 | 2.055 | 145.333 |

`TP diagnostic ms` is the measured rank-arrival slack plus a tiny diagnostic
all-reduce. It is not a claim about all NCCL time internal to normal model
kernels.

## Occupancy and overlap decision

Nsight sampled the full sweep window on both ranks. The percentages below are
profile-wide averages, followed by the mean among non-zero samples and the
peak. They are not segmented by concurrency.

| Rank | SM active avg / nonzero avg / peak | Tensor active avg / nonzero avg / peak | 1 Hz active SM mean / peak | HBM activity |
|---|---|---|---|---|
| HEAD / rank 0 | 52.61% / 86.03% / 100% | 3.74% / 6.44% / 81% | 94.64% / 96% | unavailable |
| WORKER / rank 1 | 48.61% / 86.42% / 100% | 3.46% / 6.46% / 80% | 94.33% / 96% | unavailable |

The GB10 `nvidia-smi dmon` memory-utilization field returned zero for every
sample, including intervals with 94% SM activity, and the actual Nsight metric
table contained SM and Tensor counters but no DRAM counter. Therefore no HBM
number is fabricated from this capture.

- **Mode A (cross-request server concurrency): viable.** Aggregate throughput
  scales to 2.14x at C=4 and 3.09x at C=8, with the required HTML quality gate
  intact. Scaling is substantially sublinear and per-stream latency/throughput
  worsens.
- **Mode D (same-GPU draft/verify overlap): contention-limited.** At C=1 the
  verifier is about 82% of measured block time. Verify time rises 70-84% for
  the C=4 cases and 112-147% for C=8 while active SM samples are already high.
  This does not support treating draft and verify as independent resources.
- **Mode E (rank-aware overlap): no useful rank slack found.** The diagnostic
  TP wait is only about 0.019 ms at C=1 and 0.026 ms at C=8, and both ranks'
  phase totals track closely.

## Provenance and evidence

- Code revision: `0175f8c0189b4d266ac22c9cbf331c14b27f3679`
- Candidate image: `sha256:a883e1208a45afab026ecdde9bddea34445a942a99cf8840ed21183ffcd41752`
- Raw benchmark JSON: `dspark-overlap-mode-a-0175f8c.json`, SHA-256
  `30d7d5dce58bf27f52a7925e987344d0b108690c530807108ff81854dfbfd7f8`
- Lossless engine trace: 1,953 unique sequences, every row containing rank 0
  and rank 1. Final Prometheus block/phase counts are exactly 1,953 for every
  rank and phase. The benchmark's case deltas account for 1,950 rows; the three
  preceding readiness/smoke blocks account for the remainder. Trace SHA-256:
  `2fa9d7eae329a90adbd8e80aa9390d428d24c5b90be8b429fcd9c66af1a6be16`.
- Nsight profile SHAs: HEAD
  `d05e3c1b0c3195f8e24fdbfca4e67b79cf4f71f4a645cbd4cab1c168b98795b6`,
  WORKER
  `fe0f840637573d061cf20c7bf229c911b47a80073a975456f48dc3bbe61d2dcd`.
- Local 29-file evidence manifest:
  `.local/overlap-phase1/SHA256SUMS`, SHA-256
  `abf6a5445d37ac855e2ea63bb0052b238dc72fe55bd53c56f30c001c9aab43c4`.
- Node evidence manifest SHAs: HEAD
  `1c4761eb510491301b6acc416148da56377797b8b3cb8de57731c513ef5a47d3`,
  WORKER
  `9b195391e061c88761c7b1676259410e37fa8b6b5811cb90484a477c69ad9a8d`.

The candidate loaded in 199.416 seconds and exposed 10.87 GiB for KV cache.
The outage from first HEAD stop to restored production readiness was exactly
1,500 seconds (25 minutes), within the 40-minute ceiling. Production was
restored WORKER first then HEAD on the pinned image
`sha256:3430d6614a8e2925f34d059af6caf05aff42387326db4d05639a60f10f2654d8`;
both ranks were running with `OOMKilled=false`, HEAD health was HTTP 200, and
the deterministic smoke returned exactly `OK` before watchdog disarm and lock
release.
