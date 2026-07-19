# DSpark confidence-head benchmark plan

> Executed 2026-07-19. Result: keep the confidence scheduler OFF. The matched
> no-draft denominator was 27.86 median output tok/s, the OFF control was
> 48.77 tok/s, and thresholds 0.30/0.40/0.50 reached 46.98/45.07/42.25 tok/s.
> See `benchmarks/results/decode-confidence-head-sweep.md` and its five raw
> JSON artifacts.

## Status: integration gates passed; outage approval pending

The confidence-head integration is complete at source revision
`7f64c326c9566afb9bb345a55680195cabd4c895`. The final immutable candidate is
`sha256:c78bd4cbbbe554bb18b348c8b23839da792fcab23eec2b043ed96ce64fc4daac`
on both GX10 nodes and remains inert. Production was not stopped, restarted,
removed, or exec'd during the no-outage gate.

The candidate bakes probe SHA
`c94741c6e64515a2b56597797b24526ad205458194925024d6be5907825ec234`,
exactly matching the prior bind-mounted physical micro-gate. Worker CPU/CUDA
and HEAD CPU/CUDA all passed against the real draft checkpoint:

- `mtp.2.confidence_head.proj.weight` is BF16 `[1, 4352]` on disk and loads
  into an FP32 runtime parameter;
- the forward input is exactly hidden 4096 + Markov 256 = 4352;
- the forced policy example truncates a five-token proposal to a contiguous
  two-token prefix;
- scheduler padding is `[10, 11, -1, -1, -1]`, with three invalid slots;
- speculative metrics report proposed draft tokens = 2, not the padded five;
- both CUDA probes used one NVIDIA GB10, capability 12.1, with Torch
  `2.11.0+cu130`.

CPU JSON SHA is
`70cc750ce43c6c1e647d306b72102decc314bdfe08b501c4483c4ec80a2e63d5`
and CUDA JSON SHA is
`2b6121962367150e357cc10725bbb16160ff9a5cca136012ebb2a22777e8e43b`
on each node. Node evidence is under
`/home/anemll/nvfp4-artifacts/20260719-7f64c32-confidence-b1`.

The benchmark outage described below is still **not authorized** until the
user explicitly approves its final ceiling and arm sequence.

## Historical Gate 0: blocked on the pinned production image

Do **not** run the proposed restart sweep on the current production image.
The two Compose variables exist, but the pinned vLLM build does not consume
them and does not load or execute the confidence head.

Exact live evidence from image
`sha256:3430d6614a8e2925f34d059af6caf05aff42387326db4d05639a60f10f2654d8`
and vLLM `0.25.2.dev0+g752a3a504.d20260714`:

- the live environment contains
  `VLLM_DSPARK_CONFIDENCE_SCHEDULER=off` and
  `VLLM_DSPARK_CONFIDENCE_THRESHOLD=0.0`;
- the installed vLLM tree contains zero exact references to either variable;
- `DSparkSpeculator._sample_sequential` iterates over all configured draft
  positions and has no confidence input or early-stop branch;
- `DSparkDeepseekV4ForCausalLM._remap_dspark_name` explicitly says the
  confidence head is not wired and returns `None` for every
  `confidence_head.*` weight.

The immutable audit is
`benchmarks/results/dspark-confidence-head-pinned-audit.json`. Flipping the
variables on this image would only change inert process environment and would
not measure a confidence scheduler.

This corrects the assumed env contract:

| Item | Pinned-image reality |
|---|---|
| Scheduler parser | absent |
| Accepted scheduler values | none |
| Threshold parser/range | absent / undefined |
| Confidence weights | intentionally skipped |
| Runtime behavior | fixed five-position proposal |

## Prerequisite before requesting an outage

An immutable candidate image must first wire the existing V4 DSpark
confidence head into proposal generation and add a fail-closed startup proof.
That integration is now the active no-outage work item; this document still
does not authorize a service restart.
The free, production-live probe for such an image must prove:

1. the `mtp.2.confidence_head.proj.weight` tensor is loaded rather than
   skipped;
2. a synthetic forward emits one finite FP32 logit per request and draft
   position;
3. the proposal path applies `sigmoid(logit)` and truncates at the first
   probability below the configured threshold;
4. threshold `0.0` produces the full five-token proposal, while threshold
   `1.0` produces no speculative proposal for finite logits;
5. the exact scheduler mode and threshold are emitted in startup logs and a
   machine-readable probe artifact;
6. variable proposal lengths survive the installed async-scheduler padding
   path, and invalid tail slots are excluded from speculative metrics.

DeepSeek's [reference DSpark model](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/aa22cb07426656189b2573b8e77a9b7333b8ae0f/inference/model.py)
returns an unbounded scalar logit from the confidence head. The official
[DeepSpec evaluator](https://github.com/deepseek-ai/DeepSpec/blob/main/deepspec/eval/dspark/draft_ops.py)
applies sigmoid before comparing against a probability threshold, so the
intended threshold domain is `[0.0, 1.0]`; it cannot be inferred from the
current vLLM env placeholders.

## Approved-shape experiment after the prerequisite passes

### Fixed target and settings

- current abliterated production checkpoint;
- current production image lineage plus only the reviewed confidence wiring;
- B12X target/draft kernels unchanged;
- TP=2, native DSpark, five speculative positions;
- `draft_sample_method=probabilistic`;
- approximately 10 GiB current KV allocation;
- 512 generated tokens, temperature zero, `ignore_eos=true`;
- no W4A4, model-layout, kernel, KV, context, or sampling change.

### Required denominator and arms

Run a fresh five-token OFF control on the candidate image first. Then run the
same model with speculation completely disabled. This **no-draft** result is
the denominator for every DSpark claim; DSpark throughput must be reported as
speedup versus this result, not as a standalone token rate. Then run these
probability thresholds in ascending order:

| Arm | Speculation | Scheduler | Threshold | Contract |
|---|---|---|---:|---|
| NoDraft | disabled | n/a | n/a | Mode A denominator |
| Control | five-token DSpark | off | 0.0 | fresh production control |
| T30 | five-token DSpark | on | 0.30 | permissive low-region point |
| T40 | five-token DSpark | on | 0.40 | DS4 reference optimum |
| T50 | five-token DSpark | on | 0.50 | upper low-region point |

The low `0.30--0.50` range comes from the DS4 evidence; high thresholds were
associated with a quality cliff and are excluded. The free probe uses the real
weight but synthetic hidden states, so its score distribution is **not** a
calibrated runtime histogram and must not select thresholds. If a later
production-live, no-restart probe can collect real hidden-state confidence
percentiles without changing execution, prefer those percentiles. Otherwise
freeze `0.30/0.40/0.50`; do not adapt thresholds inside the outage.

Confidence only controls how much speculative work is proposed. It is not a
quality gate. The unchanged target rejection sampler remains the quality
authority, and every arm in this batch is tagged **Mode A Strict**.

### Workload and metrics

The canonical command for every arm is:

```bash
python3 benchmarks/benchmark_dsv4_api.py \
  --base-url http://192.168.1.68:8888 \
  --model deepseek-v4-flash-dspark-abliterated \
  --concurrency 1 \
  --max-tokens 512 \
  --trials 2 \
  --require-spec-metrics \
  --expected-spec-positions 5 \
  --output ARM.json
```

The no-draft denominator uses the same command, prompt, model, 512-token
limit, temperature-zero request, and two trials, but replaces the last two
metric options with:

```bash
  --require-no-spec-metrics
```

That mode accepts either absent speculative counters or a complete unchanged
counter family and fails if any draft/acceptance counter moves. Every stream
also records a SHA-256 of the generated text. All confidence-arm output hashes
must match the no-draft Mode-A reference for the corresponding deterministic
request.

`--require-spec-metrics` is mandatory. It closes the earlier acceptance gap:
the run fails if the four Prometheus counter families or any of positions
0..4 are absent. Each trial records:

- aggregate and per-stream output tokens/s;
- TTFT;
- mean proposed draft length (`draft_tokens / drafts`);
- aggregate acceptance (`accepted / draft_tokens`);
- effective accepted length (`1 + accepted / drafts`);
- acceptance count and rate at each of five positions.

Run C1 first and bank it. Run C2 and C4 only when at least eight minutes remain
before the decision deadline; they are secondary throughput/load evidence and
must never displace C1 or rollback reserve.

### Promotion decision

No single acceptance-rate increase is sufficient. First report each DSpark
arm's median C1 speedup versus NoDraft. Select a threshold only if both trials
are valid, its median C1 output tokens/s exceeds the fresh OFF control by at
least 5%, its output hash matches the Mode-A no-draft reference, TTFT does not
regress by more than 10%, and per-position acceptance remains monotonic.
Report the entire Pareto table even if no threshold passes. A no-promotion
result is valid.

## Outage contract

- Explicit user approval is required after the confidence implementation and
  free probe are green.
- Acquire a fresh cluster lock.
- Capture the exact live image digest, both role envs, five-token speculation,
  model name, health, version, and current metric counters once.
- Proposed global ceiling: **45 minutes** from the first HEAD stop, subject to
  explicit user approval after the free probe passes.
- Decision deadline: **38 minutes**, preserving seven minutes for rollback.
- Stop order for every transition: HEAD first, then WORKER.
- Start and restore order: WORKER first, then HEAD.
- Bank each arm's JSON, before/after metrics, output hashes, rendered env,
  readiness time, and both-rank logs before moving to the next threshold.
- No retry of a failed arm. Any missing counter, wrong env, failed readiness,
  OOM, swap growth, or deadline breach jumps directly to rollback.
- Unconditionally restore the exact pinned production env
  (`scheduler=off`, `threshold=0.0`) and pinned image.
- Release only after both containers are running and not OOM-killed, HEAD
  health/version/models/dashboard pass, deterministic chat returns `OK`, and
  no benchmark/SSH/transfer process remains.

Recent production readiness is about 313--328 seconds per restart. A rigorous
comparison uses the candidate image for the OFF control, so candidate OFF,
NoDraft, three ON arms, and final production restoration mean six starts. That
projects 31--33 minutes of readiness plus roughly 5--7 minutes of C1
measurement and evidence banking. The honest hard ceiling is therefore 45
minutes, with a 38-minute decision deadline and seven-minute rollback reserve.
C2/C4 are optional and run only if the decision deadline still has at least
eight minutes of margin.
