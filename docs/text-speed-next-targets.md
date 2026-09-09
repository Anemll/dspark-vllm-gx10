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

## Real-weight TC component harness

`benchmarks/benchmark_moe_tc_decode.py` compares the original and fused-sum
paths in a disposable, exclusive-GPU process using the installed serving
adapter. It loads one real 0731 layer, not the shape-only B12X profile. The
actual checkpoint confirms K4096, TP2 intermediate width 1024, 256 experts and
topk6, with packed E8M0K32 weights and `w31` source order.

Independent captured scratch/output buffers, observed dispatch selection,
two seeded inputs and post-timing parity checks prevent a silent fallback or
stale-output speed claim. Cold-L2 and warm-L2 timings use four alternating
repeats of 20 CUDA events. The first M6 gate requires at least 10% cold-cache
gain and at most 3% warm-cache regression; only then do M1/8/12 run. Model
loading, compilation and routing generation are excluded from GPU timing.
Inputs are seeded synthetic activations passed through the real model router,
not captured end-to-end hidden states. A pass is component evidence only.

The first isolated attempt loaded and prepared the real layer in 19.48 seconds,
then stopped before any timed kernel because its scratch plan used unresolved
`cuda` while the allocated tensor used `cuda:0`. This is a harness failure,
not evidence against TC decode. Its raw result contains zero timing rows.
The harness now derives device and dtype from the activation tensor; a unit
regression test covers resolved indices, and the pristine B12X scratch
validator reproduces the old error and accepts the fixed contract with CPU
fakes. The serving implementation already uses the resolved device and is
unchanged.

A fresh corrected run, after verified restoration, passed that device guard
but stopped on another harness mismatch: it passed the deprecated
`input_scales_are_reciprocal=False`, while serving passes `True`. B12X rejected
it before timing. Both attempts have zero timing rows and remain failures;
neither is evidence that the TC kernel is faster or slower. The scale values
were unit tensors, so this fix changes the call contract, not model weights.
The harness now uses a single binding-keyword factory. An offline test executes
the actual overlaid `B12xExperts.apply` method with fakes and compares its
complete call—including tensor identities, device/dtype, scratch and all
flags—to that factory. No serving code or node profile was changed.

All 51 local tests pass, including eight new harness-contract tests. These
offline checks do not validate GPU compilation, numerical parity or speed.

On 2026-09-09, the corrected harness completed its first real-weight GPU gate
at M6. Both the control and TC-decode outputs passed the upstream-style,
independent FP32 W4A16 oracle check (worst candidate cosine 0.99998462 versus
a 0.9975 minimum). The candidate was nevertheless slower: 1051.38 µs versus
1014.95 µs cold-L2 and 1005.17 µs versus 971.88 µs warm-L2. It fails the
component speed gate, was not enabled in serving, and the native TP2 profile
was restored. See [the measured gate](../benchmarks/results/moe-tc-oracle-gate-20260909.md).

[Sanitized failure and recovery evidence](../benchmarks/results/moe-tc-component-failures-20260906.json)
retains both runner hashes, zero timing rows, exact errors, source and model
fingerprints, and hashes of the raw logs and memory recordings. The attempts
ran for 19.76 and 23.28 seconds, but their two avoidable service interruptions
required 9m37s and 9m14s from first stop through verified recovery. Total elapsed
time from starting this work window to final cleanup was 31m28s; source checks
and documentation continued afterward. Those costs must not be described as
only 43 seconds of testing.

Both recoveries used the unchanged original text image on both ranks, with
worker-before-head startup. Health, model/version, dashboard, streaming usage
and tool-call checks passed. No new fatal/route-pack-JIT signatures or OOM
events were found in the recorded recovery windows. Loading produced transient
swap writes; final swap-out rate and full-memory-pressure avg10 were zero on
both nodes. Component samplers continued briefly after process exit; missing
process/cgroup errors in those post-exit samples are counted separately, not
treated as an OOM or hidden. Final cleanup verified the original text service,
stopped all experiment recorders and released both locks at 17:11:46 UTC.

No serving optimization was accepted. The >80 tok/s goal remains open. Under
the bounded A/B workflow, another GPU attempt requires a fresh decision window
after this complete restoration; do not continue retrying inside this one.

Never run this diagnostic beside the loaded API. Prepare and verify rollback,
reserve an explicit outage window and stop both serving ranks first. Use an
immutable existing image, read-only weights and script, independent cache,
hard external timeout and continuous host/cgroup memory observation. A failed
gate requires restoration before a new decision. No serving flag should be
enabled based on the first failed attempt or the offline tests.
