# IFA 2026: applicability and implementation plan

Status: first implementation and bounded experiment completed; source review and measured
results dated 2026-09-04. Branch `IFA26` started at `081fda9`, matching `main`
when created. Measurement clients, runtime inspection, and regression tests
have been implemented; restored a399/0731 TP2 baseline and bounded prefix tests
passed. The native-width component gate did not justify a serving deployment.
The corrected mHC startup canary and TP2 startup/smoke, decode, coding,
multi-turn, exact-prefill checks through 65,536 tokens, and eight known-answer
prefix/cache checks passed. The narrow startup-warmup fix is accepted for the
tested 0731/DSpark-five workload; both candidate containers remain running,
with release defaults unchanged. See [measured results and evidence
limits](ifa26-results.md); the sections below retain the broader migration plan.

## Recommendation

Retain the accepted mHC startup-warmup candidate and validated a399 rollback.
After an initial instrumentation failure, its fresh isolated canary passed:
finite outputs, unchanged parameters,
verified compiled libraries, and no observed new compilation on two 294-token
probes. The same candidate image now ran on both TP ranks; warmup took
50.89/50.79 seconds before readiness and the initial smoke showed no new mHC
compile. W4A16, draft-input, and top-k cold compilations remain. Readiness took
5 minutes 54 seconds from head launch. Final both-rank logs show no post-monitor
TileLang compilation or observed engine/CUDA/NCCL error, and the eight-case
semantic/cache contract passed with measured reuse. Acceptance is limited to
this tested workload, not a universal zero-JIT guarantee or release promotion.

All benchmark replay bodies/hashes matched baseline, including exact prompt
IDs through 65,536 tokens. However, automatic KV profiling reported 1,995,392
cache tokens versus baseline 2,226,543, a 10.38% capacity reduction measured
before mHC warmup. Same configured settings do not make this a matched-capacity
A/B. Do not attribute observed higher throughput or lower prefill times to the
warmup. Judge the startup change on compilation coverage and correctness;
resolve effective capacity and cold-state controls before causal TPS claims.

Native sparse widths have now been measured on SM121. Captured width-256 calls
saved about 3.98 microseconds at five query tokens and 0.32 microseconds at
twenty, with larger savings beyond the 64-token dispatch boundary. These
component results do not establish the predeclared minimum 5% serving-throughput
gain. Keep the patch and diagnostic evidence, but defer a full serving
FlashInfer backport for this experiment. No end-to-end native-width improvement
was measured.

Evaluate vLLM 0.28 as a separate compatibility migration. Add Qwen/XQA as a
separate optional profile after the DeepSeek deployment passes its gates.
The active experiment's working rollback is the validated a399 prefix-cache
image with the existing DeepSeek-V4-Flash-0731 checkpoint. The original
`v0.1.1` image remains cached, but its original model location is empty or
absent, so it is not a startup-verified rollback for this run.

## Assessment of the supplied information

| Claim | Assessment and consequence |
| --- | --- |
| NVIDIA announced 1.2x RTX PRO 6000 and up to 1.4x two-Spark gains | Confirmed in the September 3 announcement. It attributes vLLM improvements to XQA and backend work. It does not isolate each kernel's contribution. |
| Per-model ratios, SpeedBench-Coding 8K/AIPerf, COMPUTEX-to-September-1 baseline | Supplied context; the retrieved announcement text did not expose the chart or complete benchmark recipe. Preserve this distinction until the chart, exact models, workload, concurrency, output lengths, and revisions are archived. These are not a usable local baseline. |
| A throughput ratio predicts C1 decode or TTFT | Unsupported. Measure aggregate output throughput, per-request latency, and prefill independently. Two GPUs alone do not establish the benchmark's concurrency or aggregation method. |
| PR #49718 adds SM12x XQA | Confirmed; merged August 11. It uses the dedicated API with speculative query lengths, masks, and graph padding, requires FlashInfer 0.6.16.post1, and explicitly excludes SM12x NVFP4 KV enablement. |
| FlashInfer is the default for all Blackwell GPUs | Too broad. Both the inspected v0.28 source and current main prioritize generic FlashInfer on SM10x causal attention; generic SM12x still lists FlashAttention first. DeepSeek V4 has its own selector. Inspect the actual selected backend. |
| These features require nightlies | Outdated for several features: v0.28.0 was released August 26 and lists XQA, DFlash2, DSpark adaptive verification, and DeepSeek V4 improvements. This does not mean every feature works on GB10. |
| FULL graphs with XQA require caution | Supported by reported correctness failures. Scope the gate to the exact backend, KV representation, masks, and revision. Do not assume PIECEWISE alone proves correctness or that generic XQA failures apply identically to DSv4 sparse MLA. |
| LM Studio/Ollama expose this through vLLM wrappers | The announcement mentions both applications, but does not establish that they wrap vLLM. Omit that implementation claim from our docs. |

Sources: [NVIDIA announcement](https://blogs.nvidia.com/blog/local-ai-ifa-next-gen-agents-nv-pair-rtx-spark/),
[XQA PR #49718](https://github.com/vllm-project/vllm/pull/49718),
[v0.28 release](https://github.com/vllm-project/vllm/releases/tag/v0.28.0),
[v0.28 backend priorities](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/platforms/cuda.py),
[current backend priorities](https://github.com/vllm-project/vllm/blob/main/vllm/platforms/cuda.py),
[XQA graph issue #49010](https://github.com/vllm-project/vllm/issues/49010).

## What this repository already has

- Exact vLLM source `752a3a504485790a2e8491cacbb35c137339ad34`, reporting
  `0.25.2.dev0+g752a3a504.d20260714`; FlashInfer source
  `0472b9b3f2fba11b463f8526f390297d52a8aad7` / 0.6.15; B12X source
  `7dc6fb8fcc6446ea093537d1657df81985fa5f43` / 0.15.3.
- DeepSeek V4 SM121 sparse MLA, the 584-byte `nvfp4_ds_mla` cache contract,
  256-to-64-token SWA page views, C128 page preservation, and TP2 head support.
- B12X MXFP4 MoE, caller-owned scratch, capture allocation checks, and route-pack
  startup warmup. This is not a missing backend to add again.
- DSpark speculation, async scheduling, chunked prefill, and prefix caching.
  The example profiles use five draft tokens, twelve sequences, and an 8192
  token scheduling budget; the Compose fallbacks differ, so rendered settings
  must be recorded rather than inferred from defaults.
- The newly merged prefix-cache fix reserves a recomputed draft sliding window
  through `KVCacheManager` and `Scheduler`. Block alignment can enlarge that
  recompute; tests should check the effective window rather than assume exactly
  128 recomputed tokens.
- Streaming decode and exact-length prefill benchmarks, plus a dashboard with
  mean latency and speculative acceptance metrics.

The production checkpoint is described in the README as FP8 on disk. B12X's
MXFP4 execution format and the NVFP4 DS-MLA cache are separate choices; do not
rename the checkpoint or substitute a quantization during a performance A/B.

The runtime path is selected by upstream
`vllm/models/deepseek_v4/nvidia/model.py::_select_dsv4_attn_cls`, which chooses
`DeepseekV4FlashInferSM120Attention` on SM12x. Our
[`flashinfer_sparse.py`](../overlay/vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py)
calls `flashinfer_trtllm_batch_decode_sparse_mla_dsv4`. Generic
`vllm/v1/attention/backends/flashinfer.py` XQA is not a replacement for this
packed sparse-cache interface. Qwen full-attention layers are the more direct
target for generic XQA; their GDN layers require a different kernel path.

All three local reference HEADs match `upstream.lock`. vLLM and B12X are clean;
FlashInfer reports existing changes in its `3rdparty/cccl` and `3rdparty/cutlass`
submodules. Leave those alone and prepare a clean candidate reference before
using any compiled results. This review used the tracked Python dispatch code.

## Proposed work, in order

### 1. Measurement, regression tests, and capability reporting

Implemented for the first experiment: dependency-free stream/prefill clients,
exact 8K coding and fixed-history replay, a known-answer prefix contract,
runtime inspection, and CPU regression coverage. Raw reports preserve request
and source hashes, usage, SSE events, validation failures, and sample-aware
summaries. The [results](ifa26-results.md) distinguish measured coverage from
remaining gaps; dashboard changes and broader GPU gates remain separate work.

| Proposed change | Files | Acceptance criterion |
| --- | --- | --- |
| Record actual runtime and kernel capabilities | New `scripts/inspect-runtime.py`; `docs/implementation.md` | JSON reports package/source versions, architecture, selected target/draft backends, KV formats, graph modes, sparse dispatch shapes, effective speculation, and unsupported settings. Checks do not load the checkpoint. |
| Add reproducible coding and multi-turn workloads | Extend `benchmarks/benchmark_dsv4_api.py`; add `benchmarks/benchmark_agent.py` and versioned fixtures | Exact prompt hashes, tokenizer/model revision, request count, output budget, seed, concurrency, cache state, and tool schemas are saved. Include unique 8K coding prompts and repeated-prefix agent turns. |
| Make latency and throughput interpretation explicit | Existing benchmark clients and dashboard | Aggregate output tokens/wall time, per-request throughput, TTFT/TPOT distributions, completion/errors, acceptance, and computed/cached tokens are distinct. Preserve raw trials and report medians, not only the best trial. |
| Add regression coverage | New CPU/GPU test groups; `.github/workflows/ci.yml` | CPU tests cover contracts and metadata; opt-in GPU tests cover actual numerical behavior. Syntax and license checks remain. |

The updated streaming client preserves chunk counts and legacy throughput
fields with explicit proxy labels. It records content/reasoning/tool deltas,
rejects missing usage or invalid output, and distinguishes visible-content
TTFT from first output. Exact-token prefill excludes empty metadata from TTFT
and retains partial-failure evidence. True token timing still requires server
instrumentation. Small two-trial smoke runs cannot establish reliable p95
latency; collect enough requests for distribution reports.

The existing random-token prefill harness deliberately avoids prefix hits and
is useful for raw prefill timing. It does not test coding quality or multi-turn
cache correctness. Reuse it alongside the new workload.

Optional AIPerf/SpeedBench integration belongs in a pinned benchmark-only
environment. Label a local 8K coding test as a proxy until the published
workload and execution settings can actually be reproduced.

### 2. Migrate to an explicitly pinned v0.28 compatibility candidate

The repo contains 14 overlaid Python files totaling 16,590 lines. Most contain
small downstream changes; two scheduler/cache files exist primarily for the
recent prefix fix. Copying these complete old files onto v0.28 would overwrite
new upstream implementations.

Create an overlay inventory with each local delta, regression test, upstream
equivalent, and retain/rebase/remove decision. Reconstruct overlay files from
the new pinned source, then reapply only necessary deltas. Preserve the cache
contract, TP2 geometry, B12X scratch and warmup, and prefix-window fix until an
equivalent replacement passes tests. Review new KV packing APIs in particular.

Change `upstream.lock`, `scripts/build-image.sh`, `docker/Dockerfile.runtime`,
credits, and compatibility docs together. v0.28's inspected dependency files
specify FlashInfer Python/cubin **0.6.16.post3**, Torch **2.13.0**, CUTLASS DSL
**4.6.2**, and TVM FFI **0.1.11**; its Docker defaults include CUDA **13.0.3**.
Resolve full source commits and matching ARM64 artifacts before building.
Audit the complete transitive set and B12X compatibility: our Dockerfile
currently overrides several of these packages and disables FlashInfer version
checks. A version-check bypass is not evidence of compatibility.

Sources: [v0.28 CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.28.0/requirements/cuda.txt)
and [Docker versions](https://github.com/vllm-project/vllm/blob/v0.28.0/docker/versions.json).

Treat this as a whole-stack compatibility comparison. Keep scheduler settings,
draft length, weights, cache format, and supported graph policy explicit.
Do not attribute any total migration gain to XQA or one PR. Both ranks must use
the same once-built image. Retain the pinned `v0.1.1` image, but use the
startup-verified a399/0731 deployment as this experiment's working rollback;
the original `v0.1.1` model path is no longer available. A future migration
must reverify the complete image/model/config rollback tuple before interruption.

### 3. Prioritize the DeepSeek V4 improvements that touch our path

**Correctness prerequisite:** audit and retain the relevant changes from
[vLLM #51538](https://github.com/vllm-project/vllm/pull/51538): sparse-width
metadata, target/draft workspace ownership, graph padding, draft-KV writes, and
padded indexer lengths. Its published hardware validation used a different
model revision and GPU topology, so our checkpoint/TP2 path still needs tests.
The FlashInfer CUTLASS MoE activation fix in that PR does not automatically
mean our B12X MoE path has that defect.

**First performance experiment, component gate completed:** use
[FlashInfer #4380](https://github.com/flashinfer-ai/flashinfer/pull/4380)'s native
192/256 DSV4 sparse dispatch. On the same compatible image, compare the
existing forced-512 behavior with native 256. Then compare 192 only after the
metadata producer represents all active window and draft entries correctly.
Our pinned FlashInfer's dispatch table contains 128/512/1024, so changing only
the wrapper's supported-width list would be invalid.

The isolated backport passed the tested SM121 reference and graph-replay
checks, but the component savings did not justify a serving rollout for the
minimum 5% C4 coding-throughput target. See [the stop decision](ifa26-results.md#native-sparse-widths-component-result-and-stop-decision).
Retain this section's untested cache/model gates for any future revival; do
not treat the component pass as a TP2 model-validation result.

Test 32 query heads, the real 584-byte packed caches, SWA-only/C4/C128 layers,
active-length sentinels, and caller-owned output/workspace. Include token counts
around 64/65, the current kernel's decode/prefill dispatch boundary, and graph
padded batches. A reduction in padded width does not imply a proportional
model speedup; use a representative microbenchmark plus end-to-end profiling.

**Next experiments:** examine
[sparse top-k metadata #52084](https://github.com/vllm-project/vllm/pull/52084)
and [narrower eager graph regions #51430](https://github.com/vllm-project/vllm/pull/51430).
The former changes metadata work relevant to prefill; the latter captures more
attention input preparation. Verify both reach the SM121 path and preserve
stream dependencies. Use separate controlled comparisons where practical.
Do not transfer published measurements from other GPUs to GB10.

Retain B12X route-pack warmup and add coverage for the already documented
indexer, W4A16, and draft-preparation cold shapes using upstream warmup support
where compatible. Serving remains in JIT warning mode. A strict cold-JIT run
is a separately scoped diagnostic after ordinary correctness succeeds.

### 4. Make speculative settings truthful; defer unsupported adaptation

The Compose file exports `VLLM_DSPARK_CONFIDENCE_*`, but no consumers were found
in the pinned upstream or overlay. Upstream's pinned DSv4 loader explicitly
skips `confidence_head.*` weights. Exported environment variables therefore
do not prove that adaptive verification is active in a reproducible build.

Use the new runtime report to distinguish requested settings from effective
features. On a compatible runtime, use upstream's
`enable_adaptive_verification` configuration and validate checkpoint weights.

However, v0.28's indexer grants variable-length `AttentionCGSupport.ALWAYS`
only on SM10x with DeepGEMM; SM121 returns `UNIFORM_BATCH`. Upstream documents
adaptive verification as requiring FULL variable-length support and rejecting
unsupported configurations. Thus adaptive DSpark on our GB10 path is a
separate porting project, not a profile toggle. Do not force the capability
flag to bypass the gate. Establish a correct variable-length indexer and graph
path before considering it.

Sources: [adaptive verification design and limitations](https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification)
and [v0.28 indexer capability checks](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/attention/backends/mla/indexer.py).

Fixed DSpark lengths can be evaluated separately after the stack is stable;
retain five as the initial example-profile control. DFlash2 needs a compatible
drafter and loader and is not interchangeable with the current DSpark weights.

### 5. Optional Qwen/XQA profile

Add an explicit model profile and documentation instead of applying DeepSeek
tokenizer, parser, cache layout, and speculation flags to Qwen. Parameterize
the launch configuration only after profile-specific validation exists.

Start with a pinned Qwen checkpoint and a supported higher-precision cache,
verify the full-attention layers actually select XQA, and independently report
the GDN backend. Establish eager/reference numerical agreement and PIECEWISE
graph coverage before testing FULL graphs or speculation. Keep the DeepSeek
profile's graph behavior separate.

Generic NVFP4 KV is a later experiment: it needs a compatible writer, scale
layout, reader, masking, and graph contract. PR #49718 does not supply that
contract. [PR #49818](https://github.com/vllm-project/vllm/pull/49818) was closed
in favor of the broader [#46329](https://github.com/vllm-project/vllm/pull/46329)
after a V-scale writer gap was identified. Check current fixes and tensor-level
tests before adopting a combination. A passing short text completion cannot
detect every scale-layout error.

PAIR routing, Ray migration, alternate cluster fabrics, new model downloads,
and new quantized checkpoints are separate projects. They are unnecessary to
evaluate the current deployment's sparse kernels.

## Validation and experiment contract

Start with static/import tests and small tensor tests, then representative
SM121 microbenchmarks, a small canary, and finally one bounded TP2 decision
run. GPU work requires an idle reserved system; a small test can still contend
with a live server. Obtain the local read-only advisor review before a risky
deployment decision or live restart, following the existing operating guide.

Required correctness coverage:

- Sparse cache shape/stride and pack/unpack tests; compare attention against
  a reference computed from the same dequantized cache, with dtype-specific
  tolerances fixed before testing. Separately assess quantization quality.
- Prefix hits, misses, exact repeats, growing turns, and divergent suffixes,
  crossing both sliding-window and 256-token block boundaries. Check draft
  context coverage, acceptance, output quality, and recomputed-token counts.
- Uniform and mixed-length batches, padded/null entries, request arrival,
  cancellation, batch drain to zero, and subsequent restart of traffic. Include
  consecutive graph replays to expose stale buffers and target/draft aliasing.
- Streaming content, usage, finish reasons, tool-call JSON/schema validity,
  tool responses followed by another turn, and both reasoning settings.
- Long-context retrieval/quality at 8K/32K/65K as well as raw prefill timing.
  Validate longer advertised contexts before claiming that the 350K profile
  is covered. Random-token timing alone is not a quality evaluation.

Use C1/C2/C4 decode with 512 output tokens; exact 1K through 32K prefill; the
33,966/36,549/40,720 boundaries; and a 65,536-token warmup plus two unique
measured trials. Add repeated-prefix and unique 8K coding workloads. Extend
concurrency to 8/12 only after the lower-load gates pass within memory limits.

Proposed decision defaults, to record before a future hardware run:

- Primary metric: median aggregate output throughput for the fixed 8K coding
  workload at C4; minimum worthwhile improvement 5%. Also require no repeatable
  regression over 3% in C1/C2/C4 decode, TTFT, or prefill without an explicit
  accepted tradeoff. Resolve borderline differences with alternating A/B runs.
- Reuse a baseline only when model/tokenizer, image/runtime, GPU topology,
  scheduler, draft settings, cache state, prompt hashes, and measurement method
  match. Existing best-of-two historical reports are context, not a new median
  baseline; adding the prefix fix or changing workload semantics can invalidate
  a direct comparison.
- Planning cap: 180 minutes total for one full experiment, including a
  90-minute build cap and a 60-minute serving interruption cap. Reserve the
  final 15 minutes of interruption for recovery; abort the candidate by minute
  45. Revise these estimates from recorded load/build times before scheduling,
  not silently during a run. A compatibility run need not earn a 5% speedup to
  establish support; it must pass the regression and correctness gates.
- Abort immediately for wrong output, either-rank failure, CUDA/NCCL errors,
  route-pack JIT during inference, exhausted memory, swap growth, or lost
  rollback readiness. For a matched request, investigate/abort at 1.25x its
  accepted baseline duration, subject to the absolute experiment cap. Monitor
  model loading separately using both ranks' progress.
- Preserve current image IDs, logs, health, rendered role-specific settings,
  memory/swap/disk/thermal evidence and the rollback image before interruption.
  Build once, transfer the image over the dedicated fabric, verify identical
  image IDs, stop head then worker, and start worker before head. Keep unique
  candidate tags and all raw evidence under `.local/results/<run-id>/`.
- End with either an explicitly accepted candidate or verified production
  recovery, including both ranks, API, dashboard, and a streaming completion.
  Never publish or replace a production default based on microbenchmarks alone.

## First implementation deliverable

The first deliverable is implemented and exercised: capability reporting,
stream/prefill validation, reproducible 8K coding and fixed-history replay,
and a known-answer prefix contract with measured cache hits. The baseline,
26 exact-prefill trials through 65,536 tokens, and eight prefix-contract
requests passed. Their scope and variance are recorded in [the results](ifa26-results.md).

The mHC candidate has also completed smoke, 14 decode, ten coding, four
fixed-history, 26 exact-prefill measured requests, and eight known-answer
prefix/cache cases. Final both-rank health/resource review passed; both
candidate containers are retained for the narrow startup-warmup fix, with
the verified a399 rollback intact and no release/default promotion. The
observed serving timings are not a causal throughput result because capacity
and cold state were not matched. Maximum tested input length remains 65,536.
The v0.28 overlay inventory and dependency compatibility check remain planned work.
No migration, Qwen/XQA enablement, or NVIDIA benchmark ratio has been validated
by the current component experiment.
