# Native B12X all-phase integration gate

This gate validates the serving-side `NvFp4NativeB12xExperts` integration on
one real prepared NVIDIA NVFP4 layer (TP rank 0, GB10/SM121). The integration:

- losslessly collapses the prepared E4M3/K16 scale expansion to E8M0/K32;
- repacks the existing FP4 payload in place to native B12X layout;
- retains exactly one FP4 payload (`duplicate_weight_bytes=0`);
- selects B12X W4A16 for every routed-expert batch size.

The authoritative immutable image was built at source revision
`f10fe9cfc561b5b49e956b0c0806b8b6098c8619`:

`sha256:a0beefab75524ab9a386f541a4015cdab15f2245e034b7b76df60da2371a4a2f`

## Results

| Routed rows | Native B12X W4A16 | CUTLASS W4A4 | Native throughput gain |
|---:|---:|---:|---:|
| 1 | 0.199232 ms | 0.221984 ms | +11.42% |
| 24 | 4.168240 ms | 4.539288 ms | +8.90% |
| 512 | 7.909712 ms | 8.593496 ms | +8.64% |

All output-activity, CUDA-graph-versus-eager, and cross-quant numeric gates
passed. The prepared backend reported native `packed` layout,
`fp4_e8m0_k32` source format, and identical data/storage pointers between the
source and prepared FP4 tensors.

The first immutable integration image stopped before kernel launch because its
scratch wrapper passed a newer optional layout keyword into the pinned
production `TPMoEScratchCaps` ABI. Revision `f10fe9c` fixes the native-packed
branch to use the exact production ABI and adds an AST regression test that
forbids the unsupported keyword.

## Evidence

- `real-layer.json` SHA-256:
  `0bedc02b8d107a2b9d794a06b6357ca3b6d02e7fc3ab9e6dd80f04981de9e9db`
- `real-layer.log` SHA-256:
  `7c2778c80b0aff96a4865ae56ea97202292c2fb76e6a7356fab0bf22e8b9c802`
- `cpu-dispatch.log` SHA-256:
  `26666472e7ebcd63440d635ba5e661b745e0a993580d273b751194e017efc0d4`

This clears the one-layer integration gate for a bounded TP=2 model-load and
serving A/B. It does not itself prove full-checkpoint loading or API-level
throughput.
