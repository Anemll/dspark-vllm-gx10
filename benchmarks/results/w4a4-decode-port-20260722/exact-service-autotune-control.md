# Exact service FlashInfer autotune control

The W4A4 serving configuration already enables FlashInfer autotuning.  The
production log identifies the exact SM121 cache as:

```text
/cache/huggingface/vllm-cache/flashinfer_autotune_cache/0.6.15/121a/
31931eb956f6ba4d6883239cd070b73e95547a853348645ef2d654002844916b/
autotune_configs.json
```

The CUTLASS-only control loaded all 60 entries from that exact cache, then
recorded explicit cache hits for `trtllm::fused_moe::gemm1` and
`trtllm::fused_moe::gemm2`.  For the observed `M=4` shape, the service cache
selects GEMM1 tactic `16` and GEMM2 tactic `58`.

## Result

The benchmark used the prepared layer-0, TP-rank-0 tensors and balanced `M=4`
routing:

| Execution | Median | p95 |
|---|---:|---:|
| CUDA graph | **0.777232 ms** | 0.886736 ms |
| Eager | 0.782608 ms | 0.834754 ms |

The graph result is finite and non-zero.  Graph-vs-eager cosine is
`0.9999988` with normalized RMSE `0.0016246`, and the graph-capture gate
passed.

## Decision

The current service is not accidentally running an untuned default kernel:
the exact cache, fused-MoE cache hits, and real-layer latency are proven.  The
`0.777232 ms` result lies inside the established approximately
`0.77--0.80 ms` CUTLASS range and remains far above the `0.682812 ms` screen.
There is no evidence that re-running autotune will recover the target gap.

Raw evidence:

- `service-autotune-control/w4a4-cache31931-cutlass-rank0-m4.json`
- `service-autotune-control/w4a4-cache31931-cutlass-rank0-m4.log`
- wrapper: `benchmarks/run_nvfp4_prepared_with_autotune_cache.py`

The cache path and full hash are preserved in the log.  A copied cache file is
not required to reproduce this control as long as that exact mounted path
remains available.
