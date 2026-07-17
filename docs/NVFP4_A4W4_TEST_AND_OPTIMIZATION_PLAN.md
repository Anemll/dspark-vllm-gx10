# DeepSeek V4 Flash NVFP4 W4A4 test and optimization plan

## Purpose

This plan answers two separate questions without conflating their evidence:

1. Does the Blackwell SM121 NVFP4 routed-MoE path execute DeepSeek V4 Flash
   correctly, and is W4A4 materially faster than an activation-matched W4A16
   path on the same packed weights?
2. Does that kernel-level result survive model loading, TP=2 communication,
   three-stage DSpark speculation, sparse MLA, scheduling, and the OpenAI API?

The first question is answered by a single-GPU kernel harness. The second
requires the two-node GX10 deployment and its operational lock. A synthetic or
single-layer result is never an end-to-end serving result.

## Current facts, hypotheses, and unknowns

### Established facts

- The validated hardware class is GB10/SM121 with two nodes and TP=2. A full
  DeepSeek V4 Flash checkpoint does not fit as a TP=1 serving model on one
  node. A single routed-expert layer does fit and is the safe scope of the
  single-head kernel harness.
- The runtime is pinned by `upstream.lock`:

  | Component | Pinned revision/version |
  |---|---|
  | vLLM | `752a3a504485790a2e8491cacbb35c137339ad34` |
  | FlashInfer | `0472b9b3f2fba11b463f8526f390297d52a8aad7` |
  | B12X | `7dc6fb8fcc6446ea093537d1657df81985fa5f43` / `0.15.3` |
  | CUDA | `13.0.2` |
  | PyTorch | `2.11.0+cu130` |

- NVIDIA's pinned DeepSeek V4 Flash artifact declares ModelOpt NVFP4 routed
  experts with group size 16. Its target geometry is K=4,096, 256 routed
  experts, top-k=6, and expert intermediate size 2,048, or 1,024 per TP=2
  rank.
- The NVIDIA artifact has 43 target layers and only one draft stage:
  133,660 non-`mtp.*` tensors plus 1,575 `mtp.0.*` tensors in 46 shards. Its
  Hub revision is
  `e3cd60e7de98e9867116860d522499a728de1cf9`.
- One `mtp.0` stage is not a runnable three-stage DSpark checkpoint. It must
  not be described or tested as one.
- The native DSpark source has three draft stages: 1,568 `mtp.0.*`, 1,565
  `mtp.1.*`, and 1,572 `mtp.2.*` tensors in shards 46-48 of 48.
- The hybrid builder selects NVIDIA's 133,660 target tensors and all 4,705
  native-MXFP4 DSpark tensors. The expected output is 138,365 tensors in 48
  shards with `175,535,844,088` indexed tensor-payload bytes.
- The hybrid config preserves NVIDIA's complete `quantization_config` and
  grafts only DSpark's `compress_ratios` and four `dspark_*` fields.
- The overlay dispatches target decoder layers below `num_hidden_layers` to
  `ModelOptNvFp4FusedMoE` and synthetic DSpark layers at or above that boundary
  to native `Mxfp4MoEMethod`.
- DeepSeek V4's clamped SwiGLU contract is mapped to FlashInfer B12X
  `swigluoai_uninterleave(alpha=1, beta=0, limit=10)`. The FC1 physical layout
  is `[up/w3, gate/w1]`, named `w13`/`up_gate` by B12X.
- The Compose runtime explicitly selects `--moe-backend flashinfer_b12x`,
  chunked prefill, NVFP4 DS-MLA KV cache, TP=2, and DSpark speculation. The
  W4A4 backend is explicit rather than an AUTO-selection assumption.
- The kernel harness compares the same resident ModelOpt NVFP4 weights and the
  same routes:
  - W4A4: FlashInfer B12X, FP4 weights and FP4 activations.
  - W4A16: B12X native ModelOpt path, the same FP4 weights and BF16
    activations.
- The existing abliterated deployment is a separate checkpoint lineage and is
  not proof that NVIDIA ModelOpt NVFP4 target experts are in use. It must not
  be relabeled as the W4A4 control without independently passing the NVIDIA
  config/index contract.
- `roce_tp` changes startup weight transport only. It does not make steady
  inference kernels faster. Rank 0 reads payloads; rank 1 still needs model
  config, tokenizer, index, and other construction metadata locally.

### Hypotheses to test

| ID | Hypothesis | Evidence that would support it |
|---|---|---|
| H1 | W4A4's largest win is at prefill M, not tiny decode M. | Repeatable graph-replay speedup across M=128-8,192, not only M=1-24. |
| H2 | Small-M decode gains are limited by routing, launch, and activation-quantization overhead. | Kernel speedup rises with M; profiles show fixed overhead dominates small M. |
| H3 | Kernel gains will be diluted end to end. | Layer speedup is larger than API prefill speedup because attention, router, shared experts, KV work, scheduler, and TP collectives are unchanged. |
| H4 | The hybrid DSpark draft can preserve most of the target-only gain. | Same-target hybrid improves total accepted-token throughput without a quality or acceptance collapse. |
| H5 | A weak large-M result indicates route packing, activation quantization, tactic selection, or memory traffic is hiding tensor-core gains. | Profiles identify one of those costs and a one-variable change improves the matched matrix. |
| H6 | Head-only weights plus `roce_tp` can avoid a second full SSD copy. | Both ranks become ready, rank 1 opens no payload shards, checksums/config agree, and startup finishes without memory pressure or transport failure. |

### Unknowns that require hardware evidence

- The actual SM121 tactic returned for every M and route distribution.
- CUDA graph capture/replay success for both paths at all agreed shapes.
- The numerical envelope between W4A4 and W4A16 on real checkpoint weights.
- Whether the official native-MXFP4 DSpark draft remains well aligned with the
  NVIDIA-quantized target after grafting.
- The end-to-end prefill gain after sparse MLA, TP communication, and chunked
  scheduling.
- The best `MAX_NUM_BATCHED_TOKENS`, graph-capture size, route tactic
  thresholds, and DSpark draft length for this checkpoint.
- Whether `roce_tp` can load the hybrid without rank-0 packing becoming the
  startup bottleneck or causing transient memory exhaustion.

## Test variants and valid comparisons

| Variant | Target experts | Draft | Serving mode | Purpose |
|---|---|---|---|---|
| K | Same NVIDIA packed NVFP4 weights | None | Single-layer harness | Strict W4A4-versus-W4A16 kernel comparison. |
| T | NVIDIA NVFP4 | Disabled | TP=2 target-only | Runtime/correctness baseline for the new target. The bundled one-stage `mtp.0` is unused. |
| H | NVIDIA NVFP4 | Native-MXFP4 `mtp.0-2` | TP=2 DSpark | Primary deployable hybrid. |
| P | Existing validated DSpark checkpoint | Existing draft | TP=2 DSpark | Operational and serving reference, not a same-weight quantization A/B. |
| O | H plus exactly one changed knob | Same as H | TP=2 DSpark | Optimization candidate. |

Hybrid H is a functional A4W4 integration test artifact: its target comes from
the official NVIDIA NVFP4 checkpoint and its draft comes from the official
native-MXFP4 DSpark checkpoint. It is not the production abliterated lineage
and must not be promoted under that identity. A production-lineage A4W4
artifact requires a separate closed-form native MXFP4-to-NVFP4 expert
conversion, activation-scale calibration, exact provenance, and the complete
quality gates in this plan.

Use these comparisons deliberately:

- K/W4A4 versus K/W4A16 isolates activation precision and kernel path while
  holding weights, routes, activation function, shapes, and seed constant.
- T versus H isolates the net benefit and overhead of the grafted DSpark draft
  on the same NVFP4 target.
- H versus O attributes a result to one optimization knob.
- P versus H is useful operationally and for user-visible throughput, but it
  is not a strict W4A4 attribution if model lineage or non-expert tensors
  differ.

Target-only T requires an invocation that omits `--speculative-config`. The
current default Compose command enables DSpark; do not point that command at
the NVIDIA-only artifact and call the result target-only. Use a reviewed
diagnostic command or a dedicated reversible configuration switch.

## Phase 0: immutable inputs and local validation

All commands in this section run on the staging/development host. They neither
contact nor reserve the GX10 nodes.

Set generic paths once:

```bash
REPO=<repo>
NVIDIA_CKPT=<nvidia-nvfp4-checkpoint>
DSPARK_CKPT=<native-mxfp4-dspark-checkpoint>
HYBRID_CKPT=<hybrid-checkpoint-view>
ARTIFACTS=<persistent-results-root>/<run-id>
mkdir -p "$ARTIFACTS/checkpoint"
cd "$REPO"
```

### 0.1 Source and overlay gates

```bash
test "$(git rev-parse HEAD)" = <candidate-commit>
git status --short
python3 -m compileall -q dashboard benchmarks overlay scripts tests
find scripts dashboard -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
git diff --check

python3 -S -m unittest discover -s tests \
  -p 'test_deepseek_v4_mixed_quant_dispatch.py' -v
python3 -S -m unittest discover -s tests \
  -p 'test_nvfp4_b12x_clamp.py' -v
python3 -S -m unittest discover -s tests \
  -p 'test_benchmark_nvfp4_a4w4_sm121.py' -v
python3 -S -m unittest discover -s tests \
  -p 'test_build_hybrid_nvfp4_dspark_checkpoint.py' -v
```

Gate: all focused tests pass; overlay files retain their Apache-2.0 markers;
the assembled image contains byte-identical overlay replacements; the
publishable private-data scan is clean; and `upstream.lock` matches the
candidate image.

### 0.2 Validate the two source checkpoints

```bash
python3 scripts/build_hybrid_nvfp4_dspark_checkpoint.py \
  --nvidia-dir "$NVIDIA_CKPT" \
  --dspark-dir "$DSPARK_CKPT" \
  --validate-only \
  | tee "$ARTIFACTS/checkpoint/source-validation.json"
```

This must fail closed on a wrong revision, tensor count, stage placement,
config, quantization map, missing shard, or index/header mismatch. Supplying
the NVIDIA one-stage artifact as `--dspark-dir` must also fail.

Gate: the JSON reports the pinned NVIDIA revision, three draft stages, the
exact source counts, and expected merged count/size. Do not override a
contract failure to continue testing.

### 0.3 Build and review the hybrid view

```bash
python3 scripts/build_hybrid_nvfp4_dspark_checkpoint.py \
  --nvidia-dir "$NVIDIA_CKPT" \
  --dspark-dir "$DSPARK_CKPT" \
  --output "$HYBRID_CKPT" \
  | tee "$ARTIFACTS/checkpoint/hybrid-build.json"

test -f "$HYBRID_CKPT/checkpoint.provenance.json"
test -f "$HYBRID_CKPT/model.safetensors.index.json"
test "$(find "$HYBRID_CKPT" -maxdepth 1 -name 'model-*.safetensors' | wc -l | tr -d ' ')" = 48
```

The default output is an absolute-symlink view. It is valid locally, but any
transfer must dereference it with `rsync -aL` or an equivalent copy. Plain
`rsync -a` would send broken source-local symlinks. Use `--hash-shards` only
when a full payload hash is needed; it intentionally reads the complete
selected model.

Review and archive:

- source config/index SHA-256 values;
- observed NVIDIA revision and etags when available;
- generated config/index SHA-256 values;
- every source-to-destination shard mapping;
- target quantization `NVFP4 W4A4` and draft quantization `native MXFP4`;
- target layers 0-42 and draft stages 0-2.

Gate: the hybrid is metadata-correct, provenance-complete, and its NVIDIA
`quantization_config` is semantically unchanged.

### 0.4 Host-side harness preflight

```bash
python3 benchmarks/benchmark_nvfp4_a4w4_sm121.py \
  --dry-run \
  --model-path "$NVIDIA_CKPT" \
  --m 1,2,4,6,12,24,48,72,128,256,512,1024,2048,4096,8192 \
  --correctness-m 1,24,128,2048 \
  > "$ARTIFACTS/checkpoint/kernel-plan.json"
```

Gate: K=4,096, I/rank=1,024, E=256, top-k=6, NVFP4 group size 16, and the
expected upstream pins appear in the plan.

## Phase 1: single-head SM121 kernel validation

This phase uses one GX10 GPU but not the full model and not TP/NCCL. It still
requires an exclusive GX10 window because the serving container must be
stopped and checkpoint reads/GPU work must not overlap another test.

### 1.1 Preconditions

- Obtain the explicit GX10 lock described below.
- Capture the current image ID, container state, health, memory, temperature,
  and free disk before interruption.
- Confirm the digest-pinned rollback image and role-specific configuration are
  intact.
- Stop the serving head and worker through the clean stop barrier. A stopped
  head with a live old worker is not a safe test state.
- Run from the exact candidate image built once from the candidate commit.
- Mount the NVIDIA checkpoint read-only and a persistent results directory
  read-write.

### 1.2 Synthetic compile/API smoke

```bash
python3 benchmarks/benchmark_nvfp4_a4w4_sm121.py \
  --synthetic --synthetic-experts 8 \
  --m 1,8,128 \
  --correctness-m 1,8,128 \
  --warmup 2 --iters 5 --repeats 2 \
  --require-graphs \
  --output <results-mount>/kernel/smoke.json
```

Gate: SM121 is detected; both backends import; eager launches work; all graphs
capture and replay; no allocation-during-capture, unsupported activation,
layout, CUDA, or JIT error occurs. Synthetic timing is not publishable
performance evidence.

### 1.3 Real-weight correctness and performance matrix

Run the complete matrix first with `--tp-rank 0`, then repeat with
`--tp-rank 1` on the same head GPU. This validates both TP slices without
copying full weights to the worker.

```bash
for TP_RANK in 0 1; do
  python3 benchmarks/benchmark_nvfp4_a4w4_sm121.py \
    --model-path <container-nvidia-checkpoint> \
    --layer-idx 0 --tp-size 2 --tp-rank "$TP_RANK" \
    --backend both \
    --m 1,2,4,6,12,24,48,72,128,256,512,1024,2048,4096,8192 \
    --correctness-m 1,24,128,2048 \
    --routing balanced \
    --seed 4104 \
    --warmup 5 --iters 20 --repeats 5 \
    --require-graphs \
    --output <results-mount>/kernel/rank${TP_RANK}-balanced.json
done
```

The agreed matrix is:

| Phase | M | Routed rows (`M * 6`) | Expected W4A4 family |
|---|---|---:|---|
| Decode | 1, 2, 4, 6 | 6-36 | micro |
| Decode | 12, 24, 48, 72 | 72-432 | static |
| Prefill | 128, 256, 512, 1,024, 2,048, 4,096, 8,192 | 768-49,152 | dynamic |

Record eager and graph median/p95, repeat-median range, selected tactic,
tokens/s, routed rows/s, effective TFLOP/s, peak memory, and graph-vs-eager
metrics. `speedup_w4a4_over_w4a16` is W4A16 median divided by W4A4 median;
values above one favor W4A4.

Correctness gates are initially the harness defaults:

- cosine similarity at least 0.98;
- normalized RMSE at most 0.25;
- no graph/eager corruption or non-finite result;
- no unexpected `w31` layout or unclamped-SiLU path;
- all requested graphs captured when `--require-graphs` is used.

These are integration tripwires, not a substitute for model-quality tests.
Do not weaken them until at least one known-good hardware report is archived.

### 1.4 Route-distribution sensitivity

Only after balanced passes, repeat both TP slices with `--routing random`.
Use `--routing hot` as a skew stress case, not the headline result. Keep all
other arguments and the seed unchanged. If a result is close to noise, run an
alternating order such as A/B/B/A and compare repeat medians.

Optional cache sensitivity can use `--l2-flush-mib <size>` with the flush
outside the event interval. Keep the resident/default run primary because one
real expert layer already exceeds L2.

### 1.5 Kernel decision gate

Use these provisional classifications, then revise only from archived data:

- **Broken:** any correctness, layout, graph, or tactic-proof failure.
- **No material prefill win:** less than 1.05x at most prefill M values.
- **Promising:** at least 1.10x at four or more of the seven prefill M values,
  no repeatable prefill regression below 0.97x, and stable p95.
- **Strong:** at least 1.20x over most M=512-8,192 points with no numerical or
  graph regression.
- **Decode-only:** gains exist at M<128 but not M>=128. This does not confirm
  the main prefill hypothesis and triggers profiling before TP=2 promotion.

The thresholds guide engineering effort; API promotion still depends on the
end-to-end gates.

## Phase 2: checkpoint staging to the head

This is a bulk-I/O operation and requires a fresh explicit lock if it was not
included in the active window.

1. Reconfirm the hybrid source validation and free space.
2. Copy exactly one runnable hybrid payload to the head SSD, dereferencing the
   staging-host symlinks:

   ```bash
   rsync -aL --info=progress2 \
     "$HYBRID_CKPT/" \
     <head-fabric-destination>/<hybrid-name>/
   ```

3. Do not copy the full checkpoint to the worker. For `roce_tp`, provide only
   the metadata required for construction: config, index, tokenizer,
   generation/quantization metadata, and provenance. Verify from logs that
   rank 1 never opens checkpoint payload files.
4. On the head, verify metadata hashes and selected shard sizes against
   `checkpoint.provenance.json`. If full payload hashes were generated, verify
   them before model load.
5. Record transfer start/end, bytes, route/interface, exit status, destination
   free space, and any residual copy process or storage activity.

Gate: the transfer completed over the dedicated fabric route, the head has a
fully materialized 48-shard checkpoint, worker metadata matches, no payload
copy exists on the worker, and no transfer process remains before releasing
or changing the lock scope.

## Phase 3: TP=2 and RoCE integration

Do not begin this phase until the RoCE task releases its current window and
the new test owner receives an explicit ACK for a fresh GX10 window.

### 3.1 Candidate and rollback preparation

- Build the candidate image once. Transfer the exact image archive over the
  fabric and verify identical image IDs on both nodes.
- Keep the digest-pinned production image preloaded on both nodes.
- Keep production and candidate checkouts separate and immutable.
- Preserve each node's role-specific environment; never copy the head env
  over the worker env.
- Render Compose configuration on both nodes and verify image, rank, master,
  model mount, cache mount, `--moe-backend flashinfer_b12x`, MTP length,
  scheduler limits, and loader mode.
- Use `JIT_MONITOR_MODE=warn` for normal validation. Reserve `error` for a
  specifically scheduled cold zero-JIT diagnostic after warning-mode success.

### 3.2 Startup order and loader modes

The required sequence is:

1. Stop old head, then old worker; verify no old service container remains.
2. Start candidate worker/rank 1.
3. Wait until rank 1 reaches the distributed rendezvous.
4. Start candidate head/rank 0.
5. Monitor both logs continuously until both ranks complete model loading.
6. Require `/health`, `/version`, and `/v1/models` success; dashboard health
   alone is insufficient.

For the head-only checkpoint experiment, set on both ranks:

```dotenv
DSPARK_WEIGHT_LOAD_FORMAT=roce_tp
DSPARK_ROCE_LOAD_BUFFER_MB=256
```

Required loader evidence includes:

- rank 0 is the sole payload reader;
- rank 1 does not open payload shards;
- target and drafter phases both complete on both ranks;
- `source_bytes`, `traffic_bytes`, tensor count, batch count, and synchronized
  elapsed time are recorded;
- no head OOM, swap storm, peer loss, NCCL timeout, short write, or incomplete
  phase occurs.

Do not claim a RoCE startup speedup unless a matched `direct_timed` run exists
with the same checkpoint, image, cache policy, and settings. A direct run
requires local payloads on both ranks and is therefore optional; never create
a worker payload copy silently just to obtain it. A cold-direct versus
warm-RoCE comparison is invalid.

If `roce_tp` cannot finish without rank-0 packing memory pressure, stop and
preserve logs. Do not fall back to ordinary loading or copy weights to the
worker without a separately authorized scope change.

### 3.3 Target-only T smoke

Before enabling the grafted draft, run the NVIDIA target in a command that
omits speculative decoding:

- one non-streaming completion;
- one streaming completion with usage and finish reason;
- one 1K prefill and one 512-token decode;
- both-rank log inspection for quant dispatch, W4A4 backend selection, JIT,
  CUDA, NCCL, and allocator errors.

Gate: API readiness, sane output, target layers use ModelOpt NVFP4, no draft
layer is constructed, and no one-stage MTP claim appears in the result label.

### 3.4 Hybrid H smoke

Enable DSpark with the agreed MTP length and keep every other setting fixed.
Confirm logs show target layers 0-42 on ModelOpt NVFP4 and synthetic draft
layers 43-45 on native MXFP4. Run:

- streaming chat with usage and finish reason;
- tool-call parser smoke;
- reasoning/non-reasoning template smoke as configured;
- 1K and 8K prefill;
- concurrency-1 decode;
- acceptance metrics and both-rank logs.

Gate: all three draft stages load, output is sane, tool calls and streaming are
well formed, DSpark acceptance is nonzero and stable, and no mixed-quant
dispatch or clamp error occurs.

## Phase 4: end-to-end API quality and performance

Run on an otherwise idle server. Freeze image ID, checkpoint hashes, loader
mode, prompts, seed, MTP length, scheduler limits, max context, cache policy,
and thermal policy. Change one performance variable per candidate.

### 4.1 Functional and quality suite

The minimum fixed suite is:

1. Deterministic factual, code, math, instruction-following, and multilingual
   canaries with `temperature=0`.
2. Long-form streaming: first token, continuous chunks, usage, token counts,
   finish reason, and valid UTF-8/JSON.
3. Tool-call cases: no-tool, one tool, multiple arguments, escaped strings,
   and invalid-tool refusal.
4. Reasoning template cases with the deployment's reasoning setting both
   enabled and disabled where supported.
5. Long-context retrieval at 32K and 65K, plus the deployment's operational
   boundary if higher.
6. Repetition/garble sentinels, non-finite logits if observable, engine health,
   and both-rank errors.
7. T versus H comparison of output quality, accepted-token rate, rejected
   draft work, TTFT, inter-token latency, and total throughput.

Archive request bodies or stable hashes, response text, token IDs where
available, status, latency, finish reason, usage, and a pass/fail reason. If a
scored evaluation suite is available, freeze its exact version and require no
regression beyond its predeclared statistical tolerance. Do not use exact text
identity as the sole quantization-quality metric.

Immediate quality failures are malformed streaming/tool JSON, persistent
garble or repetition, loss of required content, invalid finish reasons,
engine death, or a reproducible material score regression.

### 4.2 Decode matrix

```bash
python3 benchmarks/benchmark_dsv4_api.py \
  --base-url <head-api-url> \
  --model <served-model-name> \
  --concurrency 1,2,4 \
  --trials 2 \
  --max-tokens 512 \
  --output "$ARTIFACTS/api/decode-<variant>.json"
```

Compare per-stream token/s, aggregate token/s, mean TTFT, chunks/s, finish
reason, and variance. Run at least one alternating H/O/H/O sequence before
accepting a change smaller than 5%.

### 4.3 Exact-length prefill matrix

```bash
python3 benchmarks/benchmark_prefill.py \
  --base-url <head-api-url> \
  --model <served-model-name> \
  --sizes 1024,2048,4096,8192,16384,32768 \
  --trials 3 \
  --shape-warmup-trials 1 \
  --seed 4104 \
  --label <variant-and-commit> \
  --output "$ARTIFACTS/api/prefill-<variant>.json"

python3 benchmarks/benchmark_prefill.py \
  --base-url <head-api-url> \
  --model <served-model-name> \
  --sizes 33966,36549,40720,65536 \
  --trials 2 \
  --shape-warmup-trials 1 \
  --seed 4104 \
  --label long-<variant-and-commit> \
  --output "$ARTIFACTS/api/prefill-long-<variant>.json"
```

Require `metrics_exact=true` for server-side prefill comparisons, zero
unexpected prefix-cache hits, matching prompt hashes, and one excluded warmup
per shape. Compare median server prefill tokens/s, client input tokens/s,
TTFT, p95 where available, computed tokens, and logs.

Treat a single difference below roughly 3% as noise. A useful W4A4 promotion
target is a repeatable gain above 5% at multiple prefill lengths with no
repeatable regression above 3% elsewhere. A much larger kernel gain with a
small API gain is a profiling signal, not evidence that the kernel result is
wrong.

### 4.4 Stability, memory, and JIT

- Repeat 65K prefill with unique prompts after one excluded warmup.
- Run the decode matrix immediately after long prefill.
- Record peak/resident GPU memory, host memory/swap, temperature, power, and
  NVMe temperature.
- Inspect both logs for inference-time route-pack, MLA indexer, dflash, or
  fused-MoE compilation; allocator growth; CUDA/NCCL errors; and rank skew.
- After warning-mode success, schedule one cold `JIT_MONITOR_MODE=error` run
  across known boundary shapes. A failure identifies a missing warmup; it does
  not justify using strict mode in production.

## Expected bottlenecks and ordered optimization levers

Optimize in this order. Re-run the smallest test that can falsify the current
hypothesis before moving to the next layer.

1. **Contract and dispatch correctness.** Verify target/draft layer boundary,
   `w13` layout, ModelOpt scales, group size, clamped SwiGLU, and explicit
   backend selection. A wrong contract can look fast while producing invalid
   output.
2. **W4A4 graph and tactic selection.** Use the matched kernel matrix to tune
   micro/static/dynamic thresholds only if the returned tactic or latency has
   a discontinuity. Keep eager and graph results separate.
3. **Activation quantization and scale generation.** Profile large M when
   W4A4 is unexpectedly close to W4A16. Look for quantize/scale kernels,
   extra layout transforms, or intermediate traffic outside the fused path.
4. **Route packing and imbalance.** The isolated harness holds routes fixed;
   the API does not. Measure pack time, expert occupancy, hot-expert skew, and
   route capacity. Retain the existing startup prewarm across aligned and
   unaligned capacities.
5. **DSpark native-MXFP4 draft cost.** H uses W4A16 draft activations. Measure
   T versus H, acceptance, draft time, and rejected work. Tune MTP length and
   DSpark scheduler only after target correctness. Treat NVFP4-quantizing the
   draft as a separate checkpoint/quality project, not a flag flip.
6. **Chunked-prefill scheduling.** Test `MAX_NUM_BATCHED_TOKENS` one value at a
   time around the current 8,192 setting. Keep prompt size, max sequences, MTP,
   graph settings, and seed fixed. Watch TTFT, throughput, memory, and graph
   recapture.
7. **Sparse MLA/indexer and KV path.** If routed-MoE kernels are fast but API
   prefill is flat, profile index construction, attention, cache writes, and
   compressed-MLA options. These are separate from W4A4 expert compute.
8. **TP=2 collectives and rank imbalance.** Compare rank timings, NCCL time,
   expert ownership, and synchronization. A faster local kernel can expose
   all-reduce or the slower rank as the new critical path.
9. **CUDA graph coverage and warmup.** Tune capture sizes and existing target
   capture/defer switches only from logged graph/JIT evidence. Do not expand
   graph pools until memory headroom is measured.
10. **RoCE startup transport.** After serving passes, tune loader buffer size
    and packing/copy count using 128/256/512 MiB trials. This improves startup,
    not tokens/s. Gate every trial on peak head memory and complete both-rank
    phase records.
11. **Kernel-source changes.** If profiling proves a FlashInfer/B12X issue,
    implement it as a durable overlay-compatible patch or pinned fork and
    update build provenance. Never leave the only fix in a disposable
    `.build/*-upstream` checkout.

### Prioritized backlog

| Priority | Work item | First experiment | Promotion condition |
|---|---|---|---|
| P0 | Archive real SM121 balanced K matrix for both TP slices. | Phase 1.3 unchanged defaults. | Correctness/graphs pass and tactic proof is present. |
| P0 | Boot target-only T through `roce_tp`. | Minimal API smoke, no speculation. | Both ranks ready; rank 1 opens no payloads; output sane. |
| P0 | Boot hybrid H with mixed quant dispatch. | Same settings as T plus three-stage DSpark. | Three stages load; acceptance and quality smoke pass. |
| P1 | Confirm large-M prefill advantage. | Balanced then random M=128-8,192. | Repeatable material gain with stable p95. |
| P1 | Measure T versus H speculation economics. | Fixed prompts, MTP length, and scheduler. | H improves accepted/output throughput without quality loss. |
| P1 | Profile weak W4A4 prefill shapes. | Nsight trace only for representative M=128,1,024,8,192. | One dominant cost is identified before code changes. |
| P1 | Tune W4A4 tactic boundaries. | One boundary change per K matrix. | Better median/p95 without a route-pattern regression. |
| P1 | Eliminate missing graph/JIT shapes. | Cold warning run, then targeted strict run. | No inference-time compile for declared supported matrix. |
| P2 | Tune chunk size and graph capture. | 4K/8K/16K batching, one knob at a time. | API prefill improves with memory and decode neutral. |
| P2 | Tune DSpark MTP/scheduler. | T/H and multiple fixed MTP lengths. | Net throughput/latency improves and acceptance stays healthy. |
| P2 | Reduce route-pack/quantization traffic. | Profile-backed fusion/layout experiment. | Kernel/API gain survives random and hot routing. |
| P2 | Tune RoCE loader memory/packing. | 128/256/512 MiB matched warm-cache starts. | Complete startup with lower critical time and safe memory. |
| P3 | Quantize the three-stage draft to NVFP4. | Separate conversion plus quality calibration. | Exact checkpoint contract and full quality suite pass. |
| P3 | Produce an abliterated-lineage NVFP4 hybrid. | Closed-form native MXFP4-to-NVFP4 expert conversion plus activation-scale calibration, never metadata relabeling. | Same-lineage provenance and quality plus all K/H gates pass. |

## Decision tree

```text
Source checkpoint contracts pass?
├─ No  -> stop; repair provenance/config/index/shards.
└─ Yes -> hybrid count, stage placement, and config graft pass?
   ├─ No  -> stop; do not load a partially grafted checkpoint.
   └─ Yes -> synthetic SM121 smoke and graph capture pass?
      ├─ No  -> fix imports/API/workspace/graph integration.
      └─ Yes -> real-weight W4A4 numerical gates pass?
         ├─ No  -> inspect w13 layout, scales, clamp, and TP slicing.
         └─ Yes -> W4A4 has a material M>=128 gain?
            ├─ No  -> profile tactic, quantization, packing, and memory traffic.
            └─ Yes -> target-only T reaches TP=2 API readiness?
               ├─ No  -> isolate loader/runtime/TP integration before DSpark.
               └─ Yes -> hybrid H loads all three native-MXFP4 stages?
                  ├─ No  -> inspect graft and mixed target/draft dispatch.
                  └─ Yes -> quality, streaming, tools, and long context pass?
                     ├─ No  -> reject/rollback; preserve evidence.
                     └─ Yes -> H beats or usefully complements T end to end?
                        ├─ No  -> tune/disable speculation; keep target evidence.
                        └─ Yes -> repeated prefill/decode/stability gates pass?
                           ├─ No  -> identify full-stack bottleneck one variable at a time.
                           └─ Yes -> promote exact commit/image/checkpoint tuple.
```

## Promotion and rollback gates

### Promotion requires all of the following

- Exact source commit, pinned dependency revisions, image ID, and checkpoint
  config/index hashes are archived.
- The hybrid checkpoint passes the strict builder contract and provenance
  review.
- Both TP slices pass real-weight K correctness and required graph capture.
- Target-only T and hybrid H both reach API readiness on both ranks.
- Streaming, tool calls, reasoning configuration, deterministic canaries, and
  long-context cases pass.
- Baseline/candidate prompts, hashes, seeds, MTP length, scheduler, cache
  policy, and thermal conditions match.
- Prefill and decode results are repeatable. No unexplained regression above
  roughly 3% remains; any accepted tradeoff is written into `decision.json`.
- DSpark acceptance and net throughput are healthy; the draft does not merely
  add rejected work.
- No engine/CUDA/NCCL error, rank loss, memory growth, swap storm, thermal
  throttle, or unsupported inference-time JIT occurs.
- The exact same candidate image ID ran on both ranks.
- The digest-pinned production rollback remains preloaded and unmodified.
- The cluster is explicitly left on either the accepted candidate or the
  verified production image.

### Immediate rollback triggers

- Either TP rank exits, freezes, or loses its peer.
- A loader phase is incomplete, rank 1 reads payload shards in `roce_tp`, or
  head memory pressure threatens the host.
- No first token after the known prefill interval.
- Malformed streaming/tool output, garble, repetition, or quality failure.
- Real-weight numerical or CUDA graph gate failure.
- Repeatable throughput regression above 3% without an approved tradeoff.
- Unexpected allocator growth, swap storm, thermal throttling, storage error,
  or new inference-time route-pack/kernel compilation.

Rollback sequence: capture both-rank logs; stop candidate head then worker;
verify both candidate containers are gone; start the digest-pinned production
worker first; wait for rendezvous; start the production head; require health,
version, model listing, dashboard, both-rank load completion, and one streaming
request. Preserve the failed image, checkpoint, logs, and result bundle until
root cause is recorded.

## Result artifact layout and schema

Keep raw artifacts in an ignored persistent directory. Published summaries
must redact absolute source paths, hostnames, addresses, credentials, and API
endpoints; note that the kernel harness records its resolved model path.

```text
<artifacts>/<run-id>/
├── manifest.json
├── checkpoint/
│   ├── source-validation.json
│   ├── hybrid-build.json
│   ├── checkpoint.provenance.json
│   └── kernel-plan.json
├── kernel/
│   ├── smoke.json
│   ├── rank0-balanced.json
│   ├── rank1-balanced.json
│   ├── rank0-random.json
│   └── rank1-random.json
├── cluster/
│   ├── compose-head.txt
│   ├── compose-worker.txt
│   ├── image-head.json
│   ├── image-worker.json
│   ├── startup-head.log
│   ├── startup-worker.log
│   ├── readiness.json
│   ├── metrics-before.txt
│   └── metrics-after.txt
├── api/
│   ├── quality.jsonl
│   ├── streaming.json
│   ├── tool-calls.json
│   ├── decode-T.json
│   ├── decode-H.json
│   ├── prefill-T.json
│   ├── prefill-H.json
│   └── prefill-long-H.json
├── system/
│   ├── memory-head.txt
│   ├── memory-worker.txt
│   ├── gpu-head.txt
│   ├── gpu-worker.txt
│   └── storage.txt
├── comparison/
│   ├── kernel-summary.json
│   ├── api-summary.json
│   └── summary.md
├── decision.json
└── rollback/
    ├── head.log
    ├── worker.log
    └── readiness.json
```

`manifest.json` must contain at least:

```json
{
  "schema_version": 1,
  "run_id": "<utc>-<short-commit>-<variant>",
  "variant": "K|T|H|P|O",
  "started_at_utc": "<iso-8601>",
  "source_commit": "<full-commit>",
  "image": {"name": "<immutable-tag>", "id": "<image-id>"},
  "upstream": {
    "vllm": "<commit>",
    "flashinfer": "<commit>",
    "b12x": "<commit>"
  },
  "checkpoint": {
    "config_sha256": "<sha256>",
    "index_sha256": "<sha256>",
    "provenance_sha256": "<sha256-or-null>",
    "target_quantization": "NVFP4 W4A4",
    "draft_quantization": "none|native MXFP4"
  },
  "runtime": {
    "tp": 2,
    "mtp_tokens": "<integer-or-null>",
    "loader": "roce_tp|direct_timed|auto",
    "cache_policy": "<declared-warm-or-cold>",
    "seed": 4104
  },
  "lock": {
    "owner": "<task-id>",
    "scope": "<copy|kernel|tp2|combined>",
    "ack_at_utc": "<iso-8601>",
    "released_at_utc": "<iso-8601-or-null>"
  }
}
```

Each quality JSONL row should contain `case_id`, `request_sha256`, variant,
parameters, status, TTFT, elapsed time, usage, finish reason, response text or
its protected artifact reference, automatic checks, human score if any, and a
pass/fail reason. `decision.json` should name the accepted/rejected tuple,
numeric gates, exceptions, rollback state, reviewer, and UTC decision time.

## GX10 mutual-exclusion protocol

The GX10 nodes and their fabric are one exclusive resource shared with the
RoCE loader task. The following operations are mutually exclusive:

- any staging-host-to-head or head-to-worker bulk checkpoint/image copy;
- container stop/start/restart or image load;
- single-head GPU/kernel tests;
- checkpoint loading;
- TP=2 startup, API, performance, or failure-recovery tests;
- read-only remote preflight when another task has reserved a no-contact
  window.

Local source work, metadata validation, report analysis, and storage work that
does not touch either GX10 or its fabric may continue without this lock.

Protocol for every GX10 window:

1. Send the peer task the requested scope, expected operations, nodes/routes,
   and current activity (`none`, `copy`, or `test`).
2. Wait for an explicit ACK. Silence, an old ACK, or an observed idle node is
   not permission.
3. Do only the acknowledged operations. A change from copy to kernel test, or
   from single-head to TP=2, requires a fresh ACK unless the original window
   explicitly included the combined sequence.
4. While holding the lock, provide concise progress and failure updates. The
   peer task performs no GX10 copy, build/transfer, restart, load, or test.
5. On failure, stop residual processes, preserve logs, restore or deliberately
   leave the declared safe state, and report remaining I/O/container activity.
6. Release the lock explicitly with completion/abort status and residual
   process/I/O state. Do not assume release merely because a command ended.
7. Before any later GX10 operation, request and receive a fresh ACK.

The head-only checkpoint policy remains in force: one payload copy on the head
SSD, metadata only on the worker, and `roce_tp` for TP=2 loading. A full worker
copy is a separate scope expansion requiring explicit authorization.

## Definition of done

The experiment is complete only when checkpoint provenance, both TP-slice
kernel reports, TP=2 target-only and hybrid evidence, API quality/performance,
both-rank logs, system telemetry, the lock record, and the final promotion or
rollback decision are archived under one run ID. A downloaded checkpoint, a
passing dry run, or a fast isolated kernel alone is not completion.
