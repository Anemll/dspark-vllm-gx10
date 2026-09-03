# vLLM patches

These patches are applied to the exact `VLLM_COMMIT` from `upstream.lock`
before the repository overlay is copied into the vLLM source tree.

`0001-dspark-graph-replay-safety.patch` is an unmodified, path-limited
backport of vLLM commit
[`dfecbb52ce1801d12e99c1d8bfb6e37a5530cd20`](https://github.com/vllm-project/vllm/commit/dfecbb52ce1801d12e99c1d8bfb6e37a5530cd20),
merged through [vLLM PR #51538](https://github.com/vllm-project/vllm/pull/51538).
It applies cleanly to the pinned vLLM commit and retains the upstream authorship
and DCO trailers in the patch header.
