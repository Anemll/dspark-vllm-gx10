# Packed-canonical inverse before W4A4: negative gate

This bounded layer-0, TP-rank-0 probe tested whether W4A16 packed-canonical
bytes could be converted back to the ModelOpt byte order immediately before the
existing FlashInfer CUTLASS W4A4 MoE. It used the real prepared checkpoint,
balanced M=4 routing (24 active experts), identical activations/routes, and
matched eager/CUDA-graph correctness checks on an NVIDIA GB10 (SM121).

## Result

The conversion is correct but far too expensive for decode:

| Path | Median |
|---|---:|
| Existing FlashInfer CUTLASS W4A4 | 0.782480 ms |
| Inverse scatter only (W13 + W2) | 2.249824 ms |
| Inverse scatter + unchanged W4A4 | 3.062472 ms |

The combined path is **3.9138x slower** than the unchanged W4A4 reference
(0.2555x its throughput). This rejects scratch materialization as a decode
optimization.

## Correctness and scope

- W13 and W2 inverse layouts are bit-exact.
- Raw E4M3 K/16 scale storage is reused unchanged.
- Eager and CUDA-graph outputs match the reference exactly: max absolute error
  0, normalized RMSE 0, cosine 1.000000119, finite and non-zero.
- Scratch is 150,994,944 bytes for the 24 active experts, 9.375% of the full
  layer's 1,610,612,736 weight bytes. No full-model copy was constructed.

## Decision

Do not integrate this path. The viable follow-up is the already-present B12X
ModelOpt tensor-core B-loader, which performs the nibble permutation while
loading native ModelOpt bytes into MMA registers and requires no global scratch.
The next gate is to lift its packed-only TC-decode policy checks and benchmark
that zero-copy path at M=4.

Raw evidence is under
`packed-canonical-inverse-negative/`; `WORKER-SHA256SUMS` is the immutable
worker-side artifact manifest copied with the result.
