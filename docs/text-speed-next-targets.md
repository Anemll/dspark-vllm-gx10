# Text speed: evidence for the next component gate

The >80 tok/s C1 text explanation target is not met. Reuse the existing
47.61 median server-decode baseline; coding maxima and concurrent aggregate
throughput do not satisfy that goal. This note adds no live profiling run.

The saved accepted-control trace was reanalyzed through CPU launch correlation
into four target decode annotations, each with one six-token DSpark
verification batch. The prefill annotation and unmatched draft/scheduling
kernels were excluded. This reproduces the saved context counts (1883,
1883, 1883, 1882). It is instrumented per-rank evidence, not unprofiled TPS.

[Machine-readable kernel-family totals](../benchmarks/results/control-decode-kernel-audit-20260906.json)
identify the original trace by hash. The B12X W4A16 MoE kernel alone sums to
27.71 ms per target context. DeepGEMM projection kernels and small WMMA GEMMs
are the next substantial groups. FlashInfer sparse MLA is much smaller.
Sums overlap across GPU streams, so they cannot be added or subtracted as
critical-path speed predictions.

A subsequent CPU audit executed the pristine pinned B12X selector using the
saved trace's GB10 properties (48 SMs, 101376-byte opt-in shared memory).
At representative M=1/6/8/12/16/24, TP2 FC1 N/K=2048/4096 and FC2=4096/1024,
the E8M0 selector chooses K64/N128/128 threads with three blocks per SM.
The proposed K128/N64/128-thread alternative permits two; K128/N128/256 permits
one. These are resource/legal-shape results, not measured speed rankings.

The installed B12X planning source matches the pinned reference byte-for-byte
(`49cd151aa80f4fdfa603eafe21b792b51a6483fa6f39452892ed2240fd79da34`). Its
executable layout selector returns `packed` for E8M0 sources too; an older
nearby docstring describing native ModelOpt serving is stale. The current
profile explicitly disables `B12X_W4A16_TC_DECODE`. This fused-sum small-M
path therefore remains an additional component candidate, not an established
no-op or an accepted speedup. Verify the actual prepared layout and exact-M
preplanned launch in a numerical microbenchmark before trying it in serving.
Do not change this flag during an image-correctness decision.

The [Mia graph comparison](mia-comparison.md) now includes a real-source
dispatch probe: the breakable flag does not select a different V2 FULL
decode replay. A blind graph-flag restart is not the next speed gate.

Next: a bounded real-shape MoE/projection component test, fixed weights and
inputs, separate cold-cache and warm-cache timings, captured execution and
numerical checks. The existing failed concurrent MoE attempt never created
its CUDA context; do not rerun it beside the loaded server. Use an explicitly
bounded exclusive resource window after vision correctness is established.
Any candidate must predict a meaningful end-to-end gain before a full matched
text run. Do not equate a microkernel percentage with serving improvement.
