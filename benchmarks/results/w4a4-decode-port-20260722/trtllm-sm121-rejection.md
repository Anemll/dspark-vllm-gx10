# Latent FlashInfer TRTLLM NVFP4 backend: SM121 rejection

The installed image contains the routed TRTLLM NVFP4 symbol, and vLLM contains
`TrtLlmNvFp4ExpertsModular` with the ModelOpt NVFP4 quantization contract.
This is a real latent backend, not a missing Python implementation.  It is not
normally selectable on NVIDIA GB10: vLLM restricts it to the SM100 family,
while GB10 reports capability `(12, 1)`.

The bounded comparator in
`benchmarks/benchmark_nvfp4_prepared_trtllm_sm121.py` proved that device-family
policy is the sole high-level vLLM rejection, then overrode only that policy to
exercise the actual routed symbol on one real prepared TP=2 layer.  The
DeepSeek SwiGLU-OAI clamp contract remained enabled.  The comparator did not
change the serving selector or load a full model.

## Hardware result

After replacing a missing reverse-scale helper with an algebraic setup-only
inverse and fetching the missing TRTLLM headers, the comparator reached the
physical GEMM launch.  The launcher selected this kernel:

```text
bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512u2_s5_
et128x8_m128x8x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_
biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_sm100f
```

It failed in `trtllm_batched_gemm_runner.cu:305` with `Error occurred when
running GEMM` for the real `numBatches=256`, `M=1`, `N=2048`, `K=4096`
shape.  No valid output or timing was produced.  The `_sm100f` kernel selected
on an SM121 GB10 confirms that bypassing vLLM's family gate does not make the
shipped kernel hardware-compatible.

## Decision

Do not wire or expose the TRTLLM NVFP4 backend on GB10.  This is a physical
launch rejection, not merely conservative Python dispatch.  Supporting SM121
requires NVIDIA to ship or generate a compatible TRTLLM kernel; a vLLM
selector patch alone cannot recover the path.

Raw evidence is under `trtllm-sm121-negative/`:

- `trtllm-vs-cutlass-rank0-v4.log` is the decisive physical launch rejection.
- `trtllm-vs-cutlass-rank0.log` records the initially missing reverse-scale
  helper.
- `trtllm-vs-cutlass-rank0-v2.log` records the first algebraic inverse shape
  correction.
- `trtllm-vs-cutlass-rank0-v3.log` records the dependency-fetch DNS failure
  before the successful fetch in v4.

There is no JSON because the fail-closed comparator stopped at the native
launch before an eligible result existed.
