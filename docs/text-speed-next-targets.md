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
