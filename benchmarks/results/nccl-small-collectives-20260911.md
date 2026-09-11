# GB10 TP=2 small-collective recovery

On 2026-09-11, the same two-node image, 0731 checkpoint, DSpark K=5
configuration and benchmark request that had previously reached about 99
aggregate output tok/s at C4 regressed to about 54 tok/s. GPU clocks, target
MoE, sparse MLA, DSpark work and draft acceptance did not explain the change.
A matched PyTorch trace localized the regression to NCCL all-reduce kernels.

The GB10 NCCL auto-selection created 64 channels for two ranks and used the LL
protocol. An isolated all-reduce gate showed that the serving workload's small
collectives performed better with the Simple protocol and two channels, one
per direct RoCE rail. Both settings remain overridable through Compose.

## Matched full-model A/B

The primary table uses the fixed `upstream-explanation` content case, seed
5205, two measured trials, 512 output tokens and identical server/model
settings. The cluster was reloaded between arms after the unrelated M5M link
was disconnected. The rendered Compose configurations are identical except
for `NCCL_PROTO`, `NCCL_MIN_NCHANNELS` and `NCCL_MAX_NCHANNELS`. Values are
median aggregate output tok/s.

| Concurrency | Auto NCCL | Simple + 2 channels | Gain vs auto |
|---:|---:|---:|---:|
| C1 | 31.35 | 40.01 | 27.6% |
| C4 | 54.31 | 71.27 | 31.2% |

The completed Simple + 2-channel concurrency curve was C1 39.99, C2 52.27,
C3 62.21 and C4 71.27 tok/s. All 20 measured requests completed with the
expected 512 output tokens and no API, CUDA or NCCL failure. Median C4 TTFT
improved from 0.53 seconds to 0.41 seconds. Draft acceptance stayed within
normal run variance (40.2% to 41.1% at C4), so acceptance does not explain the
throughput gain.

The same reload A/B also rejected the concern that two channels improve decode
by sacrificing prefill. Exact server-side prefill throughput improved at both
tested prompt lengths:

| Input tokens | Auto NCCL | Simple + 2 channels | Gain vs auto |
|---:|---:|---:|---:|
| 1,024 | 257.8 tok/s | 566.4 tok/s | 119.7% |
| 4,096 | 398.8 tok/s | 549.8 tok/s | 37.9% |

The complete accepted-setting prefill sweep reached 545-559 tok/s from 2K
through 32K, with exact metrics for every measured request. This is still far
below the historical Sep 4 prefill control, so the NCCL override is a measured
recovery rather than a full root-cause repair.

The exact legacy Sep 4 request was also repeated with seed 4104. It reached C1
41.1, C2 52.7 and C4 71.8 tok/s, versus the saved Sep 4 control medians of C1
50.3, C2 68.8 and C4 98.7 tok/s. The channel fix therefore recovers a large
part, but not all, of the historical performance. It must not be described as
restoring the 99 tok/s result.

## Rejected explanations and knobs

- Disconnecting the unrelated M5M QSFP link changed neither C1 nor C4 beyond
  run noise. NCCL was already restricted to the two Spark-to-Spark `f0` HCAs.
- Both GPUs sustained normal clocks and utilization without thermal or power
  throttling.
- Both direct links reported 200 Gb/s, Gen5 x4 PCIe, no packet drops and no
  active congestion-notification burst during a measured transfer.
- Forcing one HCA, alternate GID selection, extra QPs, relaxed ordering,
  merged NICs, cuMem, GDR level and NCCL thread counts did not beat the accepted
  two-channel gate.

These results are a runtime recovery measurement, not an NVIDIA SpeedBench
claim. Raw traces, request bodies, server metrics and node logs remain in the
local ignored experiment archive.
