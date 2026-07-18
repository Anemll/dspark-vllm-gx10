# Prepared NVFP4 target with native DSpark draft

## Decision

Phase A0 passed. The pinned vLLM runtime supports a prepared NVIDIA NVFP4
target and a separate native three-stage DSpark draft checkpoint. No hybrid
checkpoint build or payload copy is required.

The candidate contract is:

| Component | Source | Runtime path | Quantization |
|---|---|---|---|
| Target | `DeepSeek-V4-Flash-NVFP4-TP2-CUTLASS-Prepared-v1` | `/models/dsv4-abliterated` | ModelOpt NVFP4 W4A4, prepared FlashInfer CUTLASS |
| Draft | `DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored` | `/models/dspark-draft` | Native MXFP4, three MTP stages |

This is an operational cross-lineage experiment: the official NVIDIA target
and abliterated DSpark draft are not a quantization-only A/B. Acceptance must
therefore be measured rather than assumed.

## A0 contract evidence

The dependency-free verifier proves all of the following:

- the prepared target manifest is bound to the target config, index, tensor
  count, 43 routed layers, and integrity flags;
- vLLM constructs a distinct draft model config from the explicit draft path;
- `DeepSeekV4DSpark.load_weights` selects only `mtp.*` weights;
- target and draft architecture, width, vocabulary, token IDs, tokenizer
  contract, and DSpark metadata are compatible;
- the native draft has three exact stages with tensor counts 1,568 / 1,565 /
  1,572;
- stage shards 46 / 47 / 48 have pinned SHA-256 identities, dtype×shape byte
  counts, gapless payload offsets, and header/index parity;
- the pinned non-EPLB TP=2 loader assigns each rank a contiguous half of the
  routed experts;
- the exact Prometheus acceptance counters and formulas are pinned.

Key immutable evidence:

| Artifact | SHA-256 |
|---|---|
| A0 split contract | `2db1156483824ab2623bc17c9ad3f51c423261c6b1a18b8f3ebbc0a6f82f1e20` |
| Pinned-upstream audit | `8009edf1dd4b324d1e60db7ad27461367e3e6656555ff075815fa5b1f5291ea8` |
| HEAD rendered Compose | `23037e385b4693666c0c6ea2e0e06c0cc49a3f8ff11a5c99133a761a3e43a4f6` |
| WORKER rendered Compose | `8c0a91748f7eb066b506754d6adeec3e784c040fe73e3eb4f24000c57c21942f` |

Both live renders prove distinct read-only target and draft mounts, explicit
`/models/dspark-draft`, five speculative positions, FlashInfer CUTLASS, and
an explicit 30 GiB KV allocation. Production stayed live and unchanged.

The immutable A0 inputs, hardware probe outputs, rendered Compose files, and
raw logs are archived under
`benchmarks/results/evidence/nvfp4-dspark-a0-20260718-7b877ea/` with a
`MANIFEST.sha256` integrity list.

## Memory projection

The relevant per-rank envelope is the measured post-stop 121 GiB of available
memory, not the earlier 10.8 GiB KV allocation chosen by the old runtime
configuration.

| Resident class | Per-rank budget |
|---|---:|
| Prepared target | 78.110 GiB |
| Native draft parameters | 5.336 GiB |
| Draft loader/runtime budget including 15% overhead | 6.136 GiB |
| Graph/workspace reserve | 4.000 GiB |
| System safety reserve | 2.000 GiB |
| Projected remaining KV capacity | **30.754 GiB** |
| Candidate explicit KV allocation | **30.000 GiB** (`32212254720` bytes) |

Any materially smaller observed KV allocation is a configuration failure to
diagnose, not evidence that the hardware lacks capacity. Phase C must observe
at least 30 GiB of KV, zero swap growth, and `OOMKilled=false` on both ranks.

## Phase B evidence

The archived real-checkpoint evidence remains the authoritative real-layer
CUTLASS evidence. It covers rank 0 and M=1/2/4/6/12/64/128/512/2048. It is a
routed-MoE microkernel result, not end-to-end DSpark serving.

A production-live WORKER diagnostic separately covered synthetic E=8, TP=2
rank 1 at M=4/8/12/16/20/24/32/64. Eager and CUDA-graph paths completed with
no failures; correctness rows M=4/24/64 were finite, nonzero, and graph/eager
identical. The worker JSON SHA-256 is
`4528748bd74fa8838963d1fa98cf18e8edf1d75bafc9d30d9fa08f0ef8bc2d76`.
This closes a rank-1 synthetic kernel gap only; it does not predict acceptance
or end-to-end decode throughput.

## Production DSpark baseline

The matched baseline uses five speculative positions, 512 generated tokens,
two trials, and exact counter deltas. Its JSON SHA-256 is
`8674acf55e12a3c7cd765f91bf0ee57655d8134cd73688bcd7c703b5a4a9eaef`.

A read-only inspection of both live production ranks verified the effective
speculation configuration, rather than inferring it from Compose defaults:

| Setting | HEAD | WORKER |
|---|---|---|
| `num_speculative_tokens` | 5 | 5 |
| `VLLM_DSPARK_CONFIDENCE_SCHEDULER` | `off` | `off` |
| `VLLM_DSPARK_CONFIDENCE_THRESHOLD` | `0.0` | `0.0` |
| `draft_sample_method` | `probabilistic` | `probabilistic` |

The live-config evidence SHA-256 is
`006f8c8919f472aa079cc67679093a4032591070b857dba0317eb0b31d72e4aa`.
The measured 45.31 tok/s / 38.74% / 2.94 result is therefore an
**underperforming-configuration baseline**: the confidence scheduler is off,
and mean accepted length is materially below the expected value near five.
It remains the correct Phase C control because Phase C is a target
quantization/layout comparison, not a speculation-policy experiment.

| Concurrency | Median output tok/s | Median acceptance | Mean accepted length | Median per-position acceptance |
|---:|---:|---:|---:|---|
| 1 | 45.311 | 38.743% | 2.937 | 77.71%, 49.71%, 31.43%, 22.29%, 12.57% |
| 2 | 74.685 | 45.999% | 3.300 | 81.32%, 61.67%, 41.23%, 27.88%, 17.90% |
| 4 | 95.110 | 41.860% | 3.093 | 77.69%, 55.38%, 36.57%, 24.30%, 15.35% |

Concurrency 1 is the Phase C decision workload. The serving validator now
requires, per candidate trial:

- aggregate acceptance at least 30%;
- mean accepted length at least 2.5;
- final-position acceptance at least 6%;
- monotonic five-position acceptance;
- at least 80% retention of baseline aggregate acceptance and accepted excess
  length;
- at least 50% retention at every speculative position;
- median output throughput at least the production median (1.0×).

These are promotion floors, not performance claims.

The bounded decode decision command is:

```bash
python3 benchmarks/run_nvfp4_serving_gate.py \
  --base-url http://192.168.1.68:8888 \
  --model deepseek-v4-flash-nvfp4-dspark \
  --output-dir /path/to/banked-phase-c \
  --label nvfp4-dspark-7b877ea \
  --concurrency 1 --trials 2 --max-tokens 512 \
  --require-spec-metrics --expected-spec-positions 5 \
  --baseline-decode-json benchmarks/results/dspark-production-acceptance-baseline.json \
  --skip-prefill
```

## Phase C result — rejected before readiness

The authorized run `20260718T210716Z-7b877ea-dspark-phase-c` stopped at the
native-draft post-load gate. The prepared W4A4 target path itself passed
completely on rank 1: 43 layers, 344 reads, 344 copies, 43 zero-transform
post-load rows, and 73.786 seconds total. The separate DSpark draft then loaded
96 parameters in 28.65 seconds and selected
`FLASHINFER_CUTLASS_MXFP4_MXFP8`, but the pinned native-MXFP4 converter raised:

```text
Unsupported mxfp4_backend for Mxfp4MoEMethod:
Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8.
Expected TRTLLM, Triton, AITER, or XPU backend.
```

The worker exited with code 1 and `OOMKilled=false`. The candidate never became
ready, so no graph canary or C1 performance trial ran. Production was restored
WORKER first then HEAD; health, dashboard, models, and deterministic `OK` chat
all passed. Outage to banked chat evidence was 1,112.73 seconds, below the
1,800-second ceiling. The immutable result is
`benchmarks/results/nvfp4-dspark-phase-c-failure.json`.

Both-rank candidate, production, Compose, rollback, and hash evidence for the
failed attempt is archived byte-for-byte under
`benchmarks/results/evidence/nvfp4-dspark-phase-c-failure-20260718t210716-7b877ea/`.

This disproves the earlier assumption that one global backend selection can
serve both quantization families. The prepared target requires NVFP4 CUTLASS,
while the native DeepSeek-V4 MXFP4 draft's ordinary converter supports TRTLLM,
Triton, AITER, or XPU rather than the globally forced CUTLASS variant.
Production confirms the same draft works with its own B12X selection. This is
therefore a backend-scoping failure, not evidence that a new draft kernel is
required. Before another outage, leave draft dispatch on its supported
per-quantization path, scope CUTLASS only to the prepared target, and execute
the draft post-load converter in the free in-image probe.

The executed outage contract was:

Hard global outage ceiling: **1,800 seconds from the first HEAD stop**, with
unconditional rollback. The candidate decision deadline is 1,440 seconds so
at least six minutes remain for restore.

Expected schedule:

| Phase | Expected | Abort bound |
|---|---:|---:|
| HEAD-first stop, WORKER stop, quiescence | 1–2 min | 2 min |
| WORKER-first candidate start and readiness | 10–12 min | 12 min from HEAD start |
| Bank load, memory, health, and smoke evidence | 1 min | immediate after readiness |
| Draft graph-path canary and C1 decode, two trials | 3–4 min | decision deadline |
| WORKER-first production restore and verification | 5–6 min | global ceiling |
| Expected total outage | **19–24 min** | **30 min hard ceiling** |

The gate order is cost-ordered:

1. Reconcile the pinned production digest, candidate digest, role configs,
   target/draft paths, free memory, rollback image, and global watchdog once.
   Both candidate ranks must explicitly render the verified production values:
   five draft tokens, confidence scheduler `off`, threshold `0.0`, and
   `draft_sample_method=probabilistic`.
2. Stop HEAD, then WORKER; require stable quiescence.
3. Start candidate WORKER, then HEAD.
4. Prove prepared target loading: 43 layers, 344 reads, 344 copies, zero
   transformations, CUTLASS only, no fallback.
5. Prove native draft loading: all three stages, exact shard identities and
   counts, native MXFP4 dispatch, both ranks, no NVIDIA one-stage fallback.
6. Require the explicit 30 GiB KV allocation, memory floors, zero swap growth,
   and `OOMKilled=false` on both ranks.
7. Bank readiness, complete load evidence, per-layer timings, model list,
   health, metrics snapshot, and deterministic generation before performance
   work.
8. Run a bounded eager-versus-configured-graph draft canary. Reject near-zero
   acceptance, illegal memory access, or output/acceptance drift.
9. Run only concurrency 1, two 512-token trials, with exact five-position
   Prometheus deltas. Compare against the banked production JSON using the
   thresholds above and bank immediately. Do not change any speculation
   setting inside Phase C.
10. Skip optional prefill unless more than six minutes of rollback reserve
    remains. Do not reload target-only, change drafts, tune kernels, or retry.
11. Restore production WORKER first, then HEAD; verify health, dashboard,
    models, and deterministic chat.

Phase C consumed its one authorized attempt and is closed as a rejected run.
Its authorization does not permit a retry; the backend-scoping correction
needs in-image draft-conversion evidence and fresh outage approval.

## C2 — matched confidence-enabled recheck

C2 is a separate configuration experiment after Phase C. It must not replace,
reinterpret, or mutate the Phase C baseline.

The controlled axes are:

- five speculative tokens on both production and NVFP4;
- `draft_sample_method=probabilistic` on both;
- confidence scheduler enabled on both;
- one identical, predeclared confidence threshold on both;
- identical C1 prompt, seed, 512-token length, two trials, and exact
  Prometheus counter deltas.

The exact enabled scheduler token and threshold semantics still need a pinned
runtime proof before C2 is authorized; `on` and a threshold value must not be
guessed from the environment-variable names. Once pinned, add them explicitly
to both role files and render-proof them before either restart.

Use two separately bounded restart windows under the same outage discipline:

1. **C2-P:** production image with confidence enabled; bank output tok/s,
   aggregate acceptance, mean accepted length, and all five positions; then
   restore the original confidence-off production configuration.
2. **C2-N:** the same NVFP4 candidate and exact confidence settings; bank the
   same evidence; then restore original production.

This permits three clean comparisons: production on versus production off,
NVFP4 on versus NVFP4 off, and NVFP4 on versus production on. Per-position
metrics preserve the distinction between five-token depth and scheduler
effects; any later three-token run must receive its own immutable label rather
than being pooled with these rows.

## C3 — anemll DSpark optimizations

C3 is reserved for anemll's additional DeepSeek V4 Flash DSpark optimization
work. Its configuration, artifact, and acceptance gates are intentionally
undefined until the advisor channel supplies the design. It is not authorized
by A0, Phase C, or the C2 plan.
