# IFA26: measured baseline and optimization decisions

Evidence date: 2026-09-04. This report summarizes the saved local experiment
artifacts, not NVIDIA's SpeedBench results. The two-node DSv4 deployment was
restored and its baseline, long-prefill, and bounded prefix-contract checks
passed. The native sparse-width component experiment does **not** justify a
serving deployment for the predeclared minimum 5% coding-throughput gain.
No causal end-to-end optimization gain has been established.

The corrected mHC startup-warmup canary passed its isolated component gates.
Its first instrumentation failure is preserved. The TP2 candidate has passed
startup, streaming/tool smoke, decode, coding, multi-turn, and exact prefill
through 65,536 tokens, and the known-answer prefix/cache contract. It is accepted
only as a startup-warmup fix for the tested 0731/DSpark-five workload, and both
candidate containers are left running. Release defaults are not promoted.
Automatic KV capacity differs from baseline, so the serving timings below are
observations, not a matched-capacity performance A/B.

## Deployment and rollback identity

The measured baseline is the previously validated prefix-cache deployment,
not an untouched `v0.1.1` deployment:

| Item | Measured configuration |
| --- | --- |
| Nodes | Two GB10 / SM121 GPUs, TP=2, PP=1 |
| Source | `a399cc0f8d26b97165ef3544c24095b90d1e0f6a` |
| Image | `dspark-vllm-gx10:dev-a399cc0f8d26` on both ranks |
| Exact image ID | `sha256:ce940a0674df2c43bc9b2363a2153c1f8c618fbed195b24bf792bcd7758083b0` |
| Checkpoint / served ID | DeepSeek-V4-Flash-0731 / `deepseek-v4-flash-0731-dspark` |
| Runtime | vLLM `0.25.2.dev0+g752a3a504.d20260714`, Torch `2.11.0+cu130` |
| FlashInfer / MoE | Python `0.6.15`, B12X `0.15.3`; installed cubin/cache packages report `0.6.13` |
| Speculation | DSpark, five draft tokens, probabilistic draft sampling, confidence scheduler off |
| Scheduling | 12 maximum sequences; 2048 configured batched tokens, 2000 effective after draft reservation |
| KV / memory | `nvfp4_ds_mla`, block size 256, GPU utilization 0.78; approximately 13.50/13.66 GiB available KV memory |
| Graphs | Existing breakable graph configuration; `FULL_AND_PIECEWISE`, maximum capture size 72 |
| Other settings | Prefix caching, async scheduling, chunked prefill enabled; JIT monitor in warning mode |

The rendered command, rather than example-profile defaults, defines this
baseline. In particular, an old 10-GiB KV-memory environment value is not
consumed by the restored command. Installed package version strings alone do
not establish dependency compatibility; runtime/source manifests accompany the
experiment. Model config, tokenizer, index, and shard filename/size manifests
matched between nodes; the latter is not a full checkpoint payload checksum.

Both ranks loaded the same image. Readiness took 5 minutes 30 seconds from
head launch; API, model listing, dashboard, streaming `STREAM_OK`, and automatic
tool-call parsing were checked. Both model variants were retained. Route-pack
warmup ran before serving, but other cold compilations remained observable.

The `v0.1.1` image and checkout remain cached. Their original model location is
empty or absent, so they are **not a startup-verified rollback** for this run.
The verified working rollback is the above a399 image with the existing 0731
checkpoint and preserved role-specific configuration. This distinction must
remain explicit before any later serving interruption.

## Serving baseline

The explanation workload uses 512 output tokens, seed 4104, two trials per
concurrency, and the repeated short prompt. All 14 requests passed stream,
usage, and completion checks. The coding proxy uses unique, server-tokenized
8192-token prompts, 512 output tokens, seed 4104, and two trials at C1/C4;
all ten requests passed. It is a synthetic workload, not SpeedBench-Coding.

| Workload | Concurrency | Median aggregate output tok/s | Trial range tok/s | Median request TTFT, seconds |
| --- | ---: | ---: | ---: | ---: |
| Explanation | 1 | 45.67 | 45.62–45.73 | 0.217 |
| Explanation | 2 | 70.03 | 70.01–70.05 | 0.292 |
| Explanation | 4 | 90.74 | 79.90–101.59 | 2.780 |
| Coding 8192 | 1 | 38.26 | 36.47–40.06 | 4.617 |
| Coding 8192 | 4 | 56.47 | 50.01–62.94 | 11.302 |

Aggregate throughput is validated output tokens divided by complete trial
wall time. It is not single-request decode speed. TTFT includes queueing and
prefill. SSE chunks can contain multiple speculative tokens, so first-to-last
chunk throughput and TPOT remain explicitly labelled proxies.

Cold behavior matters here. Every request in the first explanation C4 trial
waited about 5.144 seconds for first output; the second trial's first-output
times were about 0.151–0.416 seconds. The ledger records a cold draft-input
Triton compilation during decode. The available evidence does not attribute
all timing variance to one compilation. Retain both trials; do not select the
best as the baseline or use these small samples to claim a 0–3% improvement.
No p95/p99 values are supported by these sample counts.

## Exact-token prefill and long-context boundaries

Short prefill used three measured trials at each size. Long prefill used two.
Both used seed 4104, a 1024-token initial warmup, and one distinct warmup for
every measured shape, including 65,536. Every measured request generated one
token, finished with `length`, matched its exact input usage, and had isolated
server counter deltas. All 26 measured requests reported zero cache hits and
computed-token counts equal to their prompt lengths.

| Input tokens | Measured trials | Median TTFT, seconds | Median server prefill tok/s |
| ---: | ---: | ---: | ---: |
| 1024 | 3 | 0.620 | 1667.40 |
| 2048 | 3 | 1.020 | 2026.80 |
| 4096 | 3 | 1.938 | 2126.36 |
| 8192 | 3 | 3.782 | 2176.26 |
| 16384 | 3 | 7.581 | 2169.13 |
| 32768 | 3 | 15.867 | 2071.30 |
| 33966 | 2 | 16.113 | 2115.93 |
| 36549 | 2 | 17.878 | 2057.00 |
| 40720 | 2 | 19.769 | 2069.72 |
| 65536 | 2 | 34.258 | 1917.92 |

These are synthetic token-ID timing and stability checks, not long-context
retrieval or answer-quality evaluations. They do not validate the configured
350,000-token maximum. Larger shapes also retain meaningful timing variation;
use the raw matched trials for a future comparison.

## Prefix reuse: semantic result and cache evidence

The earlier four-request free-form coding replay returned coherent answers,
but identical requests produced different wording and its usage did not expose
cache counts. That result was not treated as proof of corruption or as proof
of correct cache reuse.

The subsequent deterministic contract passed **8/8 requests** in 5.48 seconds.
It used seed 4106, C1, a 64-token output budget, and exact 1023/1025-token
initial families. It required parsed JSON values, natural `stop` completion,
and server metrics bracketing each request. Fixed assistant history was
neutral; it did not repeat the answer. All six replay/branch requests had
**768 observed cache-hit tokens**; both initial requests had zero.

| Initial family | Request | Actual prompt tokens | Computed tokens | Cached tokens | TTFT, seconds |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1023 | Initial | 1023 | 1023 | 0 | 0.603 |
| 1023 | Exact repeat | 1023 | 255 | 768 | 0.320 |
| 1023 | Growing / divergent | 1051 / 1051 | 283 / 283 | 768 / 768 | 0.336 / 0.332 |
| 1025 | Initial | 1025 | 1025 | 0 | 0.638 |
| 1025 | Exact repeat | 1025 | 257 | 768 | 0.296 |
| 1025 | Growing / divergent | 1053 / 1053 | 285 / 285 | 768 / 768 | 0.330 / 0.325 |

Every answer retained `anchor=PINE263`. Initial/repeat answers returned
`current=COPPER417`; growing and divergent branches returned `INDIGO592` and
`SAFFRON846`, respectively. This establishes the tested semantic contract
across a 1024-token boundary, with measured reuse. It does not establish
general numerical equivalence, all sliding-window alignments, or long-context
cache correctness. The observed 255/257 recomputed tokens also illustrate why
recomputation should not be described as invariably exactly 128 tokens.

## Native sparse widths: component result and stop decision

The isolated FlashInfer patch adds native 192/256 widths to the existing pinned
stack; it was compiled once and the exact library reused for graph timing.
No checkpoint was loaded by the component diagnostic, and the live serving
FlashInfer installation was not replaced. The final graph report used 32 query
heads, 133 active entries, 584-byte packed FP8-footer caches with BF16 RoPE,
unit-scale inputs, and widths 512/256/192. All arms attended to the same
entries; only padding differed.

Six timing trials each replayed 30 captured public calls. Eager, changed-query,
restored-query, and post-timing output checks passed against the dequantized
reference with `atol=0.05, rtol=0.05`. Worst recorded absolute error across the
final report was 0.05078125; this passes the combined absolute-plus-relative
criterion and must not be restated as a strict 0.05 maximum-error bound.

| Query tokens | Width 512, microseconds | Width 256, microseconds | Width 192, microseconds | Time saved by 256 vs 512 |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 18.154 | 14.174 | 14.441 | 3.980 microseconds |
| 20 | 24.764 | 24.443 | 25.172 | 0.321 microseconds |
| 65 | 61.277 | 47.844 | 47.884 | 13.433 microseconds |

These are median component times for captured calls on reused synthetic
buffers. The 65-token case crosses the pinned decode/prefill dispatch boundary.
Neither its larger savings nor the approximately 22% component-time reduction
at five tokens establishes a proportional model improvement. Width 192 is
also not consistently faster than 256. The earlier eager test covered
5/10/20/60/64/65 tokens; it included host enqueue overhead and is not directly
comparable to graph timings.

Decision: retain the patch, build/library provenance, reference checks, and
diagnostic tooling, but **do not perform a full serving FlashInfer backport for
this experiment**. The component evidence is insufficient to predict the
predeclared 5% minimum improvement in C4 coding aggregate throughput. Broader
C4/C128 cache, metadata, and model correctness gates remain necessary before
any future serving use. No end-to-end native-width A/B result exists.

## mHC startup canary: component passed

Cold fused-normalization compilation observed during restoration motivates a
separate startup-warmup candidate. The proposed gate is coverage of the missed
294-token shape, finite results, parameter preservation, and no new mHC compile
after warmup. This targets first-request latency, not a claimed steady-state
throughput gain.

The initial isolated report has `status=failed`: instrumentation inspected an
unused base kernel cache rather than the active TileLang backend instances.
The ledger records finite warmup output in 77.28 seconds; total failed-canary
time was 83.63 seconds. It stopped before the 294-token and parameter-preservation
decision probes. Its exact error was
`RuntimeError: cannot correlate warmup with isolated adapter libraries`.

The fresh rerun with corrected instrumentation passed. Warmup took 78.357
seconds; total canary time was 83.969 seconds. Peak GPU allocation was
440,101,376 bytes (419.7 MiB). All 16 observed TileLang libraries had verified
provenance; synthetic parameters were unchanged and outputs were finite.
Two post-warmup 294-token probes took 0.276 and 0.295 milliseconds, with no
observed compiler process or cache changes.

The report records `no_new_compilation_observed=true` and
`zero_jit_proven=false`. This establishes the bounded canary's observation,
not a universal zero-JIT guarantee or full model numerical equivalence.

## mHC TP2 candidate: completed checks and comparison limits

The candidate was built once from source
`4af757b42af4070371c4ac1cd09085db570660d9`. Both ranks used the exact ARM64
image `sha256:88759d53395ee9890f9d68c984160109e8d29d9df7c8aa1d84ccb597e64d7a56`.
The runtime change is mHC startup warmup; the existing checkpoint, DSpark-five
configuration, dependency packages, and rendered per-role serving settings were
retained. Compose differed only in image. The runtime manifests agree on the
new warmup source hash and installed packages on both ranks; they are source
inspection evidence, not proof by themselves that a kernel ran.

The old head and worker stopped completely before the new worker started.
Head launch was 21:12:29 UTC; readiness was 21:18:23, **5 minutes 54 seconds**
later, versus the baseline's 5 minutes 30 seconds. The measured interruption
from old-head stop to readiness was 6 minutes 55 seconds. Both ranks warmed
23 mHC token sizes before readiness: 50.89 seconds on the head and 50.79 on
the worker, finishing at 21:17:50 before JIT monitoring began at 21:18:19.
These are one startup each, not an isolated estimate of warmup overhead.

API health, version, model listing, and dashboard checks passed. Streaming
returned exactly `STREAM_OK`, with valid usage, `stop`, and `[DONE]`; the tool
smoke returned `get_weather({"city":"Paris"})` and `tool_calls` completion.
No mHC compilation appeared during that initial smoke: the previously observed
first-smoke mHC cold work moved before readiness. This is not zero-JIT serving.
One `W4A16FusedMoeKernel` compilation per rank remained in the initial smoke;
the subsequent head log also records `_prepare_dflash_inputs_kernel`,
`_compute_global_topk_indices_and_lens_kernel`, and another W4A16 shape.
The DFlash-named helper is used by the current draft path; it does not imply a
switch from DSpark to a DFlash model.

### Capacity and request matching

Automatic profiling reported 13.02/12.10 GiB available KV memory and a GPU KV
cache size of 1,995,392 tokens, versus 13.50/13.66 GiB and 2,226,543 tokens for
baseline: **10.38% fewer reported cache tokens**. Profiling occurred before the
new warmup executed. This is an unresolved host/profiling-state mismatch, not a
measured memory cost of mHC warmup. Identical configured utilization does not
establish identical effective capacity. The tested contexts and concurrency
are below capacity, but that does not remove the comparison confound.

Matching was checked on saved data, not inferred from labels:

- All 14 decode and ten coding requests matched their baseline's complete
  JSON request body and SHA-256, indexed by concurrency/trial/request. Client
  source hashes matched. Coding case bodies, case-bundle hash, and tokenization
  evidence also matched; every coding request reported exactly 8192 input tokens.
- All four multi-turn bodies and hashes, including fixed assistant history,
  matched baseline; their complete saved case bundles also matched.
- Every prefill prompt token-ID array and SHA-256 matched, including all
  warmups: 25 short requests and 13 long requests. All 26 measured requests
  had exact isolated metrics, zero cache hits, full prompt computation,
  one output token, and `length` completion.
- The eight known-answer prefix cases and their complete case-bundle hash
  matched baseline. All semantic, stream, and per-request metric gates passed.

Decode/coding timeouts were shorter for the candidate, but no request failed
or was censored. Cache-state labels alone do not prove equal cold state.
Small samples, cold compilation, and the capacity mismatch preclude causal
steady-throughput claims or certification of the 5% serving-improvement gate.

### Observed serving results

All 14 candidate decode requests and all ten coding requests passed stream,
usage, `[DONE]`, and 512-output-token checks. Both runs retained two trials per
concurrency. No p95/p99 or generated-code quality score is justified.

| Workload | C | Baseline median tok/s | Candidate median tok/s | Candidate trial range tok/s | Baseline / candidate median TTFT, seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Explanation | 1 | 45.67 | 48.11 | 47.93–48.30 | 0.217 / 0.234 |
| Explanation | 2 | 70.03 | 72.25 | 71.50–73.00 | 0.292 / 0.257 |
| Explanation | 4 | 90.74 | 101.96 | 100.51–103.41 | 2.780 / 0.210 |
| Coding 8192 | 1 | 38.26 | 38.79 | 38.02–39.56 | 4.617 / 4.645 |
| Coding 8192 | 4 | 56.47 | 59.44 | 55.72–63.15 | 11.302 / 11.114 |

The candidate explanation C4 range includes the baseline's 101.59 tok/s second
trial. Coding C4 second trials were 62.94 and 63.15 tok/s. These illustrate the
cold-trial influence; neither is a replacement for the full reported baseline.
Do not attribute the higher candidate medians to mHC warmup.

The candidate fixed-history conversation completed all four 256-token requests.
Initial/repeat/growing/divergent TTFT was 4.353/0.409/0.434/0.394 seconds,
versus 4.243/5.140/0.425/0.427 in baseline. This single conversation mixes cold
and reused-prefix behavior. Cache counts are absent from its usage, and repeated
answers differ in wording; neither a cache-speedup claim nor cache corruption
follows from that free-form test.

The separate known-answer candidate contract passed **8/8 requests** in 5.67
seconds. Initial families again had zero cache hits; all six repeat/branch
requests had 768 observed cache-hit tokens. All retained `anchor=PINE263` and
returned the correct final `current` values with natural `stop` and `[DONE]`.
Computed-token counts were identical to baseline: 255/257 for exact repeats
and 283/285 for the growing/divergent branches. This confirms the bounded
semantic/cache contract, not arbitrary long-context correctness.

### Candidate exact prefill

Both short and long reports have `status=complete`. The same distinct shape
warmups and matched measured prompts were used as in baseline; all 26 measured
requests passed the exact metrics and stream checks described above.

| Input tokens | Trials | Baseline / candidate median TTFT, seconds | Candidate median server prefill tok/s |
| ---: | ---: | ---: | ---: |
| 1024 | 3 | 0.620 / 0.523 | 1990.55 |
| 2048 | 3 | 1.020 / 1.030 | 2005.60 |
| 4096 | 3 | 1.938 / 1.946 | 2118.26 |
| 8192 | 3 | 3.782 / 3.793 | 2167.79 |
| 16384 | 3 | 7.581 / 7.552 | 2175.93 |
| 32768 | 3 | 15.867 / 15.117 | 2173.22 |
| 33966 | 2 | 16.113 / 15.849 | 2149.01 |
| 36549 | 2 | 17.878 / 17.011 | 2152.96 |
| 40720 | 2 | 19.769 / 19.113 | 2134.51 |
| 65536 | 2 | 34.258 / 31.687 | 2072.44 |

The 65,536-token measured TTFTs were 31.667 and 31.707 seconds after a distinct
warmup. This establishes the tested length/timing stability, not retrieval
quality or 350K-context coverage. The same capacity and causal-attribution
limitations apply to these timing differences.

### Final disposition

Both final rank logs show no post-monitor TileLang compilation and no observed
engine, CUDA, or NCCL error. Each rank still recorded two W4A16, two draft-input,
and two top-k inference compilation warnings. This closes the observed mHC
coverage gap without claiming all cold compilation has been removed.

The final snapshots confirm both candidate containers running the same image,
with the a399 rollback image retained and rollback environment/Compose hashes
unchanged. Available host memory was about 12.1/14.5 GB. Interval samples showed
zero swap-out, though existing swap occupancy remained; these samples do not
prove a swap-free run. Both GPUs were 61 C with no active thermal slowdown;
cumulative thermal counters did not increase across the long sweep. No active
build/transfer process remained. Disk use was 97% on each host at that snapshot,
before removal of the temporary 19-GB transfer archive. Both loaded images and
the evidence were retained; future build headroom still needs deliberate
management.

Decision: **accept the narrow startup-warmup fix for the tested
DeepSeek-V4-Flash-0731 / DSpark-five workload and leave both candidate containers
running**. This is not a release/default promotion, a 350K-context certification,
a universal zero-JIT guarantee, or an accepted causal throughput improvement.
Keep the verified a399 rollback and both model variants intact.

## Evidence and follow-up

The persistent, Git-ignored local archive is
`.local/results/ifa26-20260904/`. Raw reports can contain deployment details
and are not copied into this public document. The following names identify the
source artifacts used for the tables and decisions:

- `WORKLOG.md`, `vllm-restore/RESTORE_REPORT.md`, rendered compose reports,
  and `runtime-head.json` / `runtime-worker.json`: restored identity and settings.
- `decode-baseline-live.json`, `coding-baseline.json`,
  `multi-turn-baseline.json`: serving trials and their interpretation limits.
- `prefill-baseline.json`, `prefill-long-baseline.json`,
  `prefix-contract-baseline.json`: exact-token and semantic/cache evidence.
- `sparse-native-provenance.json`, `sparse-native-eager.json`,
  `sparse-native-graph-final.json`: isolated component build and final measurements.
- `mhc-canary-initial.json`, `mhc-canary-passed.json`: preserved instrumentation
  failure and successful fresh component rerun.
- `candidate-runtime-head.json`, `candidate-runtime-worker.json`, candidate
  compose reports, `candidate-start-head.log`, `candidate-start-worker.log`,
  and `candidate-smoke.json`: candidate identity, startup, and smoke evidence.
- `candidate-benchmark-head-sofar.log`, `decode-candidate.json`,
  `coding-candidate.json`, `multi-turn-candidate.json`, `prefill-candidate.json`,
  and `prefill-long-candidate.json`: completed candidate trials and remaining
  observed cold compilations.
- `prefix-contract-candidate.json`, `candidate-final-head.log`,
  `candidate-final-worker.log`, `candidate-final-head-state.txt`,
  `candidate-final-worker-state.txt`, and `pre-long-*-resources.txt`: final
  semantic/cache, both-rank, rollback, and resource checks.
- `CANDIDATE_REPORT.md`: private operational handoff and cleanup/disposition.

Preserve the accepted candidate and verified a399 rollback. This change is
judged on bounded pre-readiness compilation coverage, correctness, stability,
and startup cost; resolve capacity/cold-state matching
before any causal throughput experiment. Treat vLLM 0.28 migration and optional
Qwen/XQA support as
separate work requiring dependency/overlay review and their own matched tests.
NVIDIA's announced ratios remain external context, never locally measured
targets or results. See the [implementation plan](ifa26-plan.md).
