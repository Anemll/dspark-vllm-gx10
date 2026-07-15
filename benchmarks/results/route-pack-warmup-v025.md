# vLLM 0.25 route-pack warmup validation

Configuration: two-node TP=2, DSpark draft length 5, b12x MXFP4 MoE, and the
same DeepSeek V4 Flash checkpoint described in the repository README. The
runtime reported `0.25.2.dev0+g752a3a504.d20260714`.

## Strict JIT validation

A cold launch with `JIT_MONITOR_MODE=error` completed unique 33,966-, 36,549-,
and 40,720-token prefill requests without compiling
`_pack_topk_routes_prefix_kernel` during inference. This crosses the route-pack
capacity and scalar-specialization boundaries that reproduced the original
failure.

The first cold 65,536-token request reached unrelated, previously unwarmed
specializations in vLLM's MLA indexer and the CuTeDSL fused-MoE kernel. Strict
mode intentionally stopped at the first such compilation. Normal serving uses
warning mode, where these shapes compile once and the engine remains healthy.

## Steady 65K prefill

After one excluded 65,536-token shape warmup, two unique measured requests
produced:

| Trial | Server prefill | Client input | TTFT |
|---:|---:|---:|---:|
| 1 | 1,967.1 tok/s | 1,963.0 tok/s | 33.386 s |
| 2 | 2,000.4 tok/s | 1,996.0 tok/s | 32.833 s |
| Median | 1,983.8 tok/s | 1,979.5 tok/s | 33.109 s |

## Decode regression check

Best aggregate throughput from two 512-token trials at each concurrency:

| Concurrent streams | Aggregate output throughput |
|---:|---:|
| 1 | 44.9 tok/s |
| 2 | 71.4 tok/s |
| 4 | 104.5 tok/s |

No route-pack JIT warning appeared on either TP rank during the prefill or
decode runs. The warmup adds approximately 10--11 seconds per rank during model
loading and no request-path work.
