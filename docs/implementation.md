# Spark-specific integration

FlashInfer includes the native SM120/SM121 DeepSeek V4 sparse MLA kernel, but
the pinned vLLM wrapper does not fully describe Spark's NVFP4 layout. The
overlay provides the missing integration:

- routes `nvfp4` to the `nvfp4_ds_mla` packed `uint8` cache format;
- uses the 584-byte token layout required by the native kernel;
- preserves valid compressed C128 pages while normalizing SWA cache tensors;
- supports TP=2 query-head geometry;
- pads 256-wide DSpark draft indices to the native 512-wide dispatch using
  `-1` sentinels without changing active lengths;
- carries the b12x MoE integration used by the validated runtime.

The b12x adapter registers `flashinfer_b12x` with vLLM's modular MXFP4 oracle,
prepares the checkpoint's native MXFP4 tensors into b12x's W4A16 runtime
format, plans caller-owned scratch, and rejects allocations during CUDA graph
capture. The small-M selector override is opt-in/configurable through the
`VLLM_B12X_W4A16_*` environment variables.

For the ModelOpt NVFP4/W4A4 B12X micro kernel, TP=2 C4 serving uses
`DSPARK_B12X_MICRO_MAX_ACTIVE_CLUSTERS=40` by default. This reserves eight of
GB10's 48 SMs for overlapping runner/collective work. On the canonical
target-only 4-client decode test it improved the warmed median from 72.64 to
73.86 tok/s without changing C1 materially. Set the value identically on both
TP ranks; removing it restores FlashInfer's uncapped schedule. The complete
route-distribution and API evidence is under
`benchmarks/results/w4a4-decode-port-20260722/full-serving-mac/`.

## Optional ModelOpt NVFP4 W4A4 target

The NVIDIA DeepSeek V4 Flash NVFP4 checkpoint changes the routed target
experts from native MXFP4/W4A16 execution to ModelOpt NVFP4/W4A4. It does not
provide the three native draft stages required by this deployment. The
optional integration therefore has three explicit contracts:

- `deepseek_v4/quant_config.py` sends decoder layers below
  `num_hidden_layers` to `ModelOptNvFp4FusedMoE`, while the synthetic DSpark
  layers at and above that boundary remain on `Mxfp4MoEMethod`;
- the SM121 FlashInfer B12X adapter maps DeepSeek's clamped SiLU to
  `swigluoai_uninterleave(alpha=1, beta=0, limit=10)` and preserves the
  physical `[up/w3, gate/w1]` FC1 layout;
- the NVFP4 oracle accepts `flashinfer_b12x` when selected explicitly, but
  keeps it out of AUTO until the real SM121 correctness/performance matrix is
  archived.

`scripts/build_hybrid_nvfp4_dspark_checkpoint.py` constructs a strict
metadata/symlink view from NVIDIA target shards and native-MXFP4 DSpark draft
shards without rewriting tensor payloads. The resulting hybrid is a functional
integration artifact, not the existing abliterated production lineage. See
`docs/NVFP4_DSPARK_HYBRID_CHECKPOINT.md` for its exact counts and provenance
rules.

`benchmarks/benchmark_nvfp4_a4w4_sm121.py` compares W4A4 against an
activation-matched B12X W4A16 path using the same packed ModelOpt weights,
routes, TP slice, shapes, clamp, and seed. It separates decode M values from
prefill M values and records eager/graph latency, tactic proof, numerical
metrics, and provenance. Full promotion still requires the lock-gated TP=2
and API gates in `docs/NVFP4_A4W4_TEST_AND_OPTIMIZATION_PLAN.md`.

## B12X route-pack startup warmup

B12X route packing uses Triton kernels specialized by both a power-of-two
route capacity and the divisibility of the live route count. Previously, a new
long-prefill shape could compile `_pack_topk_routes_prefix_kernel` after
request execution began. If compilation occurred during CUDA graph capture,
it could terminate the engine.

Each TP rank now prewarms route packing during model loading, before CUDA graph
capture. It covers every power-of-two capacity through
`MAX_NUM_BATCHED_TOKENS`; for capacities greater than two, it warms both the
aligned capacity and capacity minus one. Those calls cover Triton's aligned
and generic scalar specializations. The completed warmup is cached per CUDA
device, expert count, top-k, and maximum capacity. With the default
`MAX_NUM_BATCHED_TOKENS=8192`, startup increases by approximately 10--11
seconds per rank and adds no request-path warmup.

The expected startup log is:

```text
Prewarmed B12X route-pack capacities (...) on cuda:0 (experts=256, topk=6)
```

## JIT monitor

Compose enables vLLM's JIT monitor in warning mode by default. Set these values
in both node environment files when more detail is needed:

```bash
JIT_MONITOR_MODE=warn
JIT_MONITOR_VERBOSE=1
```

`warn` reports unexpected inference-time compilation while allowing the server
to continue. `error` is intended only for cold-start validation: it terminates
the engine on any previously unwarmed kernel, including kernels unrelated to
B12X.

A cold TP=2 run with `JIT_MONITOR_MODE=error` completed 33,966-, 36,549-, and
40,720-token prefill requests without route-pack compilation. A subsequent
65,536-token test exposed separate first-use specializations in vLLM's indexer
(`_build_prefill_chunk_metadata_kernel`) and the CuTeDSL
`W4A16FusedMoeKernel`. The first decode matrix also compiled a dflash input
preparation shape. None was a route-pack kernel, so the result validates this
fix but is not yet a full-stack zero-JIT guarantee for every cold request. Use
`warn` for normal serving until those independent warmup gaps are addressed.

The new backend was informed by `voipmonitor`'s earlier Apache-2.0 b12x vLLM
integration in vllm-project/vllm pull request 39634, then expanded for the
current vLLM modular-MoE APIs and the tested DeepSeek V4 TP=2 path. Full
provenance is in `CREDITS.md`.

All upstream revisions are pinned in `upstream.lock` so a build cannot silently
move to incompatible wrapper or kernel behavior. In particular, b12x is pinned
to Git commit `7dc6fb8f`; its package reports `0.15.3`, but that version was not
published to PyPI.
