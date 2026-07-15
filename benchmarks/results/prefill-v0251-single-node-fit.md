# Single-node prefill reference

**Result: not runnable with the same checkpoint.**

The test changed only the deployment topology from TP=2 to TP=1. The model,
vLLM candidate, KV-cache type, context limit, batching limits, speculative
configuration, and memory-utilization setting were unchanged.

The checkpoint contains 155.43 GiB of FP8 Safetensors weights, while one GX10
reports approximately 121 GiB of usable physical unified memory. During model
loading, available memory fell to 369 MiB before the NVIDIA driver reported
`NV_ERR_NO_MEMORY`. The API never became ready, so no 1K–32K throughput samples
are valid. The engine was terminated after the OOM errors to recover the host.

The original TP=2 vLLM 0.25 candidate was then restored. Both ranks are active,
the API is healthy, and a live completion succeeded. See
[`prefill-v0251-single-node-fit.json`](prefill-v0251-single-node-fit.json) for
the machine-readable record.
