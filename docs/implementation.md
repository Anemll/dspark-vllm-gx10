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
- carries the b12x NVFP4 MoE integration used by the validated runtime.

The b12x adapter registers `flashinfer_b12x` with vLLM's modular MXFP4 oracle,
prepares the checkpoint's native MXFP4 tensors into b12x's W4A16 runtime
format, plans caller-owned scratch, and rejects allocations during CUDA graph
capture. The small-M selector override is opt-in/configurable through the
`VLLM_B12X_W4A16_*` environment variables.

The new backend was informed by `voipmonitor`'s earlier Apache-2.0 b12x vLLM
integration in vllm-project/vllm pull request 39634, then expanded for the
current vLLM modular-MoE APIs and the tested DeepSeek V4 TP=2 path. Full
provenance is in `CREDITS.md`.

All upstream revisions are pinned in `upstream.lock` so a build cannot silently
move to incompatible wrapper or kernel behavior. In particular, b12x is pinned
to Git commit `7dc6fb8f`; its package reports `0.15.3`, but that version was not
published to PyPI.
