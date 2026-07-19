# DSpark confidence-head benchmark plan

## Decision: Gate 0 is blocked on the pinned production image

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
That integration is outside this plan and is not authorized by this document.
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
   machine-readable metric.

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

### Arms

Run the OFF control first in the same window, then these probability
thresholds in ascending order:

| Arm | Scheduler | Threshold | Purpose |
|---|---|---:|---|
| Control | off | 0.0 | fresh same-window production baseline |
| T25 | on | 0.25 | permissive pruning |
| T50 | on | 0.50 | midpoint |
| T75 | on | 0.75 | aggressive pruning |

The three points span the sigmoid probability range without pretending that
the unobserved logit distribution is calibrated. If the free probe can emit a
confidence histogram, replace these fixed points with its 25th/50th/75th
percentiles before freezing the outage batch.

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

No single acceptance-rate increase is sufficient. Select a threshold only if
both trials are valid and its median C1 output tokens/s exceeds the fresh OFF
control by at least 5%, without TTFT regression above 10% or any non-monotonic
per-position acceptance. Report the entire Pareto table even if no threshold
passes. A no-promotion result is valid.

## Outage contract

- Explicit user approval is required after the confidence implementation and
  free probe are green.
- Acquire a fresh cluster lock.
- Capture the exact live image digest, both role envs, five-token speculation,
  model name, health, version, and current metric counters once.
- Global ceiling: **30 minutes** from the first HEAD stop.
- Decision deadline: **24 minutes**, preserving six minutes for rollback.
- Stop order for every transition: HEAD first, then WORKER.
- Start and restore order: WORKER first, then HEAD.
- Bank each arm's JSON, before/after metrics, rendered env, readiness time, and
  both-rank logs before moving to the next threshold.
- No retry of a failed arm. Any missing counter, wrong env, failed readiness,
  OOM, swap growth, or deadline breach jumps directly to rollback.
- Unconditionally restore the exact pinned production env
  (`scheduler=off`, `threshold=0.0`) and pinned image.
- Release only after both containers are running and not OOM-killed, HEAD
  health/version/models/dashboard pass, deterministic chat returns `OK`, and
  no benchmark/SSH/transfer process remains.

Recent production readiness is about 313--328 seconds per restart. The OFF
control needs no initial restart; three ON arms plus final restoration project
to about 22 minutes of restart time and 3--5 minutes of C1 measurement and
banking, fitting the 30-minute ceiling narrowly. C2/C4 are therefore optional.
