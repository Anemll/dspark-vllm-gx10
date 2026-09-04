# IFA native-width serving integration

Evidence: 2026-09-04, TP=2 GB10/SM121, DeepSeek-V4-Flash-0731 with DSpark five.
This is a follow-up to the [initial experiments](ifa26-results.md), not a
reproduction of NVIDIA SpeedBench. **Native widths did not produce a useful
serving-speed gain and remain disabled by default.**

## What was actually integrated

Source `8c97ddf50b07601e37e6dd74b91dc78c059a3eab` packages the exact compiled
FlashInfer sparse-MLA 192/256-width backport in an immutable image, alongside
runtime source and binary verification. It changes the SM12x DSpark metadata
policy so 133 valid slots can use width 192 instead of the old padded decode
width 512. It also clamps negative padded indexer lengths, derived from
[vLLM #51538](https://github.com/vllm-project/vllm/pull/51538).
This is the DeepSeek sparse-MLA path, **not generic XQA attention**.

Both ranks ran image ID
`sha256:dd79b04562de82ec4195f14f20b67693fa82b25fa63bf757900576faa373bcaf`.
The verified sparse library SHA256 is
`eaf38ba3550392457f263019496582265aaddfdce7d0b3937cb36affe5219e0b`.
The base remains vLLM 0.25.2 development / FlashInfer 0.6.15 / B12X 0.15.3;
this is a reviewable backport, not an upgrade to those packages' latest releases.

## Matched-capacity control

Earlier automatic memory profiling produced different KV capacities, so its
timings were not reused for a causal A/B. Both new arms explicitly reserve
10 GiB KV per rank and report **1,649,158 effective cache tokens**. Control and
native use the same image, weights, DSpark-five probabilistic sampling, TP=2,
12-sequence limit, 2048 configured/2000 effective batched-token limit, fixed
requests, seed 4104, and graph configuration. Only
`VLLM_DSV4_NATIVE_SPARSE_WIDTHS=0/1` differs. Target uses V2 breakable PIECEWISE
graphs; draft uses FULL graphs. This must not be described as an MRV1 test.

The control passed streaming/tool calls, 14 decode requests, ten exact 8K coding
requests, exact prefill through 65,536, and 8/8 known-answer prefix cases with
six measured 768-token cache hits. Short and long prefill used distinct per-shape
warmups and zero measured prefix reuse. Control 65K TTFT median was 31.062 seconds.
The fixed 10 GiB budget is an experimental capacity setting, not a memory-saving
optimization claim.

## Native-width canary result

Real metadata-kernel tests passed seven context boundaries, padded requests,
and widths 192/256/512. Native TP=2 startup, streaming, and automatic tool calls
passed. A separate 128-token warmup preceded two measured 512-token trials at
each concurrency. All 14 measured requests completed without API errors.

| Decode concurrency | Control aggregate tok/s | Native aggregate tok/s | Change |
| --- | ---: | ---: | ---: |
| 1 | 50.2724 | 46.6251 | −7.26% |
| 2 | 68.8486 | 69.3305 | +0.70% |
| 4 | 98.6732 | 96.4495 | −2.26% |

Both native C1 trials were below their matched control trials. This fails the
speed acceptance gate; it is not proof that kernel time alone caused the
difference. The longer coding/prefill/prefix runs and native profiling were
not performed after that decision. Therefore native-width long-context
correctness is **not established**, and no speedup or production recommendation
is made. The validated flag-0 control is the immediate rollback configuration.

The bounded five-iteration **control** profile recorded 208.323 ms in B12X W4A16
fused MoE kernels, 4.783 ms in width-128 primary sparse attention, and 0.216 ms in
width-512 draft sparse attention on rank 0. Instrumented timings are not serving
TPS, and overlapping GPU durations cannot simply be added to wall time.
They indicate a very small end-to-end ceiling for changing draft sparse width
alone, and larger opportunities in MoE and repeated host launch work.

Raw immutable JSON, both-rank logs, image provenance, control traces, and the
time ledger are retained under ignored
`.local/results/ifa26-native-20260904/`. Original models, role environment files,
production images, and the working 4af/a399 rollback images are preserved.

## Separate next candidate

An opt-in V2 attention graph boundary derived from
[vLLM #51430](https://github.com/vllm-project/vllm/pull/51430) and
[#52401](https://github.com/vllm-project/vllm/pull/52401) captures preparation
while retaining eager sparse scoring and attention. It preserves the original
control's auxiliary-stream overlap, avoids capturing the short-context scoring
skip, and does not introduce the persistent scratch pool reverted by
[#52836](https://github.com/vllm-project/vllm/pull/52836).
`DSPARK_NARROW_ATTN_GRAPH` defaults to 0 and fails closed outside V2/SM12x/NVFP4.
Its synthetic GPU lifetime canary is not a model-quality or model-speed test.
Live acceptance requires separate matched measurements and correctness gates;
do not combine it with the rejected native-width flag.

The isolated GPU canary passed 120 changed-input/metadata replays across five
capture sizes (1, 2, 8, 24, 72), all three layer types, nested streams, shared
graph pools, allocator pressure, and alternating replay order. It completed in
12.22 seconds. Synthetic single-token replay medians were 37.49 → 24.52 µs
(SWA), 71.64 → 26.07 µs (C128), and 162.59 → 47.18 µs (C4). These are structural
launch-overhead measurements with synthetic preparation, not real-model TPS.
The tested attention-source SHA256 is
`94b5e497f89a0027146ab962aa6e00541c8a3204beaddc317c8626386ce01718`.
