# ModelOpt-native B12X tensor-core W4A16: negative gate

This one-layer gate tested B12X's existing tensor-core W4A16 decode kernel
directly on the prepared DeepSeek V4 Flash NVFP4 layer-0, TP-rank-0 tensors.
The benchmark-only policy patch admits the checkpoint's native ModelOpt nibble
layout without constructing a packed-weight duplicate.  The compile proof
requires `weight_layout=modelopt`, BF16 activations, direct top-k routes, fused
FC2 top-k summation, and a non-zero FC2 output.  Both `M=1` and `M=4` satisfied
that physical contract on NVIDIA GB10 (SM121).

## Result

| M | ModelOpt TC W4A16 | FlashInfer CUTLASS W4A4 | W4A16/W4A4 throughput |
|---:|---:|---:|---:|
| 1 | 0.341096 ms | **0.197552 ms** | 0.5792x |
| 4 | 1.305224 ms | **0.769408 ms** | 0.5895x |

The W4A16 candidate adds 72.7% latency at `M=1` and 69.6% at `M=4`.  Its
outputs are finite and non-zero, CUDA-graph capture passed, and the
cross-quantization numeric comparison passed the benchmark contract:

- `M=1`: cosine `0.9869652`, normalized RMSE `0.1627378`
- `M=4`: cosine `0.9875339`, normalized RMSE `0.1579103`

The negative result is therefore a performance rejection, not a layout,
dispatch, graph-capture, or correctness failure.

## Decision

Do not route ModelOpt NVFP4 decode through this B12X W4A16 tensor-core path.
The fused in-register ModelOpt loader removes the otherwise-fatal inverse
materialization cost, but the kernel itself is still about 41% lower in
throughput than FlashInfer CUTLASS W4A4.  It cannot close the target-only C4
service gap.

Raw evidence:

- `modelopt-tc-negative/modelopt-tc-rank0-balanced.json`
- `modelopt-tc-negative/modelopt-tc-rank0-balanced.log`

Implementation/proof commits:

- `2cb8fe4` (`Enable ModelOpt weights in B12X TC decode benchmark`)
