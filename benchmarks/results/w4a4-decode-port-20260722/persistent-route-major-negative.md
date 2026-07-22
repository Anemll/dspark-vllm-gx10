# Persistent route-major producer/consumer overlap: negative gate

This real-layer experiment removed the global route-compute boundary between
W4A4 FC1 and FC2.  FC1 publishes ready tiles through a release/acquire queue,
while FC2 consumes them through its retained two-stage TMA/MMA pipeline.  It
used prepared layer-0, TP-rank-0 weights, balanced `M=4` routing, 24 active
experts, and the serving SwiGLU-OAI clamp of 10.  The matched reference is the
accepted fused FlashInfer CUTLASS path over the same inputs, routes, and
weights.

## Result

| Path | CUDA-graph median |
|---|---:|
| Accepted fused CUTLASS | **0.770224 ms** |
| Persistent route-major | 0.886564 ms |

The producer/consumer path is **15.1% slower** (`0.86877x` reference
throughput) and misses the `0.682812 ms` performance screen.

The overlap itself is proven rather than inferred: both readiness banks report
`overlap_observed=true`, all 192 FC2 tasks were published and consumed, and no
global route-compute boundary remained.  Correctness also passed:

- eager cosine `0.99993944`, normalized RMSE `0.0109997`;
- graph cosine `0.99994069`, normalized RMSE `0.0108838`;
- the double-buffer comparison passed;
- outputs were finite and non-zero.

The probe returned non-zero solely because of the performance gate.

## Decision

Do not integrate this persistent route-major schedule.  Genuine FC1/FC2
overlap is possible on GB10, but its queue, readiness, and finer-grained
scheduling costs exceed the overlap benefit by a wide margin.  Together with
the earlier route-major variants, this closes the route-order and
producer/consumer-overlap family for the current kernel structure.

Raw evidence:

- `persistent-route-major-negative/persistent-route-m4-v3.json`
- `persistent-route-major-negative/persistent-route-m4-v3.log`
