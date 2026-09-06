# Mia recipe comparison and benchmark audit

Reviewed 2026-09-06 at MiaAI-Lab revision
`957890ac5e26ce149646719298169f80603a3e1f`. No downloaded patch or launcher
was executed. This is a source/arithmetic audit, not a reproduction of Mia's
hardware results.

## What the 80 tok/s claim supports

The [published August 14 matrix](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/blob/957890ac5e26ce149646719298169f80603a3e1f/results/bench-screenshot-matrix-20260814.json)
contains one C1 request in each prompt-size cell, not repeated C1 trials.
The fastest cell has 273 actual prompt tokens, 128 output tokens, 0.270 s
TTFT, 1.537 s client decode time and 1.807 s whole-wave time.

- Its **82.648 tok/s** is exactly `(128 - 1) / decode_seconds`. There is no
  off-by-one inflation in that saved result.
- Including the whole-wave wall time gives **70.837 output tok/s**. This is
  a different metric, not a correction to a fraudulent measurement.
- Other C1 cells report 64.57, 65.89, 61.99 and 74.93 decode tok/s for the
  2K, 8K, 32K and 128K prompt labels, respectively.
- The data is for text-only 0731, not evidence of Vision-Exp performance.

The current replica scripts request numbered lowercase English words with
repetitive context, temperature 0.6 and no fixed seed. They use `N / decode_s`,
unlike the saved matrix's `N-1`; that difference is only 0.79% at 128 tokens.
The saved matrix does not include the complete request bodies, seeds or output
hashes needed to prove it used exactly those current scripts.
Its first-event timestamp is also a streaming-chunk boundary: without raw
events, we cannot check whether speculative tokens were already grouped in
that first chunk. Arithmetic consistency alone does not prove token-level
timing accuracy.

Our [content benchmark](../benchmarks/results/content-0731-20260905.md) uses
512 output tokens, temperature 0, seed 5205 and three measured trials.
Its C1 explanation median is **47.61 server decode tok/s**; Python and
TypeScript content reach 70.04 and 72.51. Server decode counters and client
first-token timing are also distinct measurements. These results cannot be
turned into a speedup ratio against Mia's short, differently shaped request.

Published aggregate decode also excludes prefill. For example, the short C6
cell is 162.11 aggregate decode tok/s versus 154.16 output tok/s over the
whole wave. The 128K C6 cell is 1.869 versus 1.574. Report the denominator,
not just the larger number. The [derived audit JSON](../benchmarks/results/mia-matrix-audit-20260906.json)
preserves the source hash, selected raw fields and recomputed values.

Conclusion: the headline is workload-specific and too weak to establish a
general 80 tok/s rate. The available evidence does **not** justify calling it
fabricated. Our >80 target remains repeated C1 text explanation decoding;
neither concurrency aggregate nor a favorable coding maximum satisfies it.
Vision has separate correctness/stability gates and may decode more slowly.

## Code differences worth testing

| Difference | Our action | Evidence / limit |
|---|---|---|
| DSpark block length versus three draft stages | Integrate a narrow config fix for `DSparkDraftModel` | Actual pinned guard rejected Vision-Exp's block 5; MTP guard remains intact |
| Regular rather than breakable CUDA graphs | Prioritize an isolated runtime experiment | Mia reports 74.6 to 95.9, but no matching raw graph A/B artifact was found in this revision |
| Shared-memory IPC spin window, 1 s to 2 ms | Profile CPU contention before changing | Independent variable; no measured benefit on our stack yet |
| Larger batch/KV/context settings | Do not copy into an A/B control | Changes memory, prefill and scheduling simultaneously |
| Vision routing and weight-name hotfixes | Retain our architecture-specific implementation | Our router owns `bias_vl`; copying Mia's different bias remapping would be incorrect |
| Image-prefix visibility / primary topk512 | Retain our separate SM121 prefill module | Equivalent attention changes were not found in the compared Mia overlay; equivalence is unproven |

The DSpark fix exempts only `method=dspark` with architecture
`DSparkDraftModel` from the MTP module-reuse divisibility rule. It does not
change defaults, existing block-3/block-6 support, other DSpark models or the
0731 block-5 path. It is not evidence that a prior block-3 run was corrupt.

Seven config-guard tests pass. Two bounded, real-runtime CPU checks process
three two-image requests each, with block 5 and no CUDA initialization.
Breakable=1 selects compilation mode 0; breakable=0 selects mode 3. Both
retain `FULL_AND_PIECEWISE`. These checks establish config/processor support,
**not** successful GPU graph capture, image inference or improved throughput.

There is an additional prerequisite: pinned NVIDIA target/draft classes lack
`@support_torch_compile`. The V2 manager's non-breakable piecewise path calls
the model directly, while FULL capture/replay is independent of that flag.
Thus the printed graph mode may conceal eager piecewise execution, and Mia's
historical speedup may not transfer to our V2 runner. Trace actual dispatch
before spending a service outage on this hypothesis.

A subsequent recording-fake probe executes the pristine enum, descriptor
construction, dispatch and replay methods: C1/2/4/12 all select identical
FULL descriptors and the same replay operation with either flag. The
piecewise call differs. This falsifies the proposed direct decode-path
mechanism, not every possible indirect timing effect; no live graph-switch
experiment is justified by the saved headline alone.

If that gate supports a speed trial, keep the accepted 0731 image, checkpoint,
DSpark5, scheduler and workload fixed; change only graph mode in a bounded TP2 window.
Do not combine that result with a checkpoint switch, block-length change or
IPC patch. The new Vision-Exp block-5 setup needs its own correctness run.
