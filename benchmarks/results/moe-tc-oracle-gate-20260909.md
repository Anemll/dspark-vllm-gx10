# B12X TC-decode real-weight oracle gate — 2026-09-09

This is a component measurement, not an end-to-end serving TPS result.

## Scope

- Hardware: one SM121 GB10, with the TP=2 API stopped for exclusive GPU use.
- Model shard: native `DeepSeek-V4-Flash-0731`, TP=2 rank 0, routed MoE layer 3.
- Shape: six tokens, 256 local experts, top-k six, hidden size 4096 and TP-local
  intermediate size 1024. This is the DSpark M6 decode shape.
- Runtime: audited B12X 0.15.3. The runner verified the pinned B12X integration
  and W4A16 kernel source fingerprints before loading a single layer.
- Both paths used the same fixed activations and router results. The candidate
  engaged `tc_decode_fused_sum=true` on packed E8M0-K32 W4A16 weights.

## Correctness

The first control-versus-candidate identity gate was intentionally replaced:
the candidate atomically accumulates expert outputs while the control performs
a separate ordered top-k sum, so BF16 identity is not an appropriate contract.
Instead each path was checked against an independent FP32 W4A16 oracle using
the immutable, pre-repack source weights. This is the same criterion used by
the upstream TC-decode coverage: mean per-token cosine must be at least 0.9975.

| Seed | Control cosine | TC-decode cosine |
|---:|---:|---:|
| 4104 | 0.99998856 | 0.99998462 |
| 4105 | 0.99998903 | 0.99998510 |

Both paths passed. The worst candidate cosine was 0.99998462.

## Timing and decision

| Cache state | Control median | TC-decode median | Candidate/control |
|---|---:|---:|---:|
| Cold L2 | 1014.95 µs | 1051.38 µs | 1.0359× |
| Warm L2 | 971.88 µs | 1005.17 µs | 1.0343× |

The fused-sum variant is 3.6% slower cold and 3.4% slower warm at the relevant
M6 shape. It therefore fails the component speed gate and must remain disabled;
it was not deployed to the text-serving profile. The native TP=2 image was then
restored using the retained worker-first recovery profile.
