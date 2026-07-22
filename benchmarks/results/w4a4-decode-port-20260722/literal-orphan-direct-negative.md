# Literal FlashInfer `MoEDirectMicroKernel`: definitive negative gate

This is the direct hardware test of the previously unwired FlashInfer source,
not a descendant or proxy.  The audit pins the original
`moe_direct_micro_kernel.py` at SHA-256
`abfad363fae29d15c0c2af127a54b7bafe2ae667c08ff976a2caf6d0828436b2`.
The SM121/DeepSeek-compatible port is pinned at
`ce223868f247c1abb097df2e59bf0a0ac8087924e290921e11faf9fa04e6754e`
and changes only the runtime ABI/cache key, W13 order, and SwiGLU clamp
plumbing required to preserve the serving contract.  Descendant-only dual-dot,
L2-prefetch, and E8M0/K32 markers are explicitly absent.

The probe used real prepared DeepSeek V4 Flash layer-0, TP-rank-0 W4A4
weights, balanced `M=4` routing, the checkpoint's E4M3/K16 scales, and the
serving SwiGLU-OAI limit of 10.  Its matched reference is the current
FlashInfer CUTLASS path over identical inputs, routes, and weights.

## Result

| Execution | Literal orphan | FlashInfer CUTLASS | Literal/CUTLASS throughput |
|---|---:|---:|---:|
| CUDA graph | 1.177312 ms | **0.779912 ms** | 0.66245x |
| Eager | 1.184592 ms | **0.787600 ms** | 0.66487x |

The literal kernel adds **51.0% graph latency** and 50.4% eager latency.  It is
approximately 33.8% lower in graph throughput than CUTLASS.

Correctness is valid:

- literal and CUTLASS outputs are finite and fully non-zero;
- literal eager and graph outputs are bit-exact;
- literal-vs-CUTLASS cosine is `0.99959594`;
- normalized RMSE is `0.0284303`.

The probe exited non-zero solely because the performance gate failed; there
was no ABI, launch, graph, activity, or numeric failure.

## Decision

Do not wire the orphan into vLLM serving.  This direct test closes the earlier
evidence gap: the literal source itself runs correctly on GB10, but it is much
slower than the already-selected CUTLASS W4A4 kernel and cannot improve C4
decode.

Raw evidence:

- `literal-orphan-direct-negative/flashinfer-literal-orphan-direct-m4.json`
- `literal-orphan-direct-negative/flashinfer-literal-orphan-direct-m4.log`
