# FlashInfer CUTLASS active24 real-route A/B

This gate tested an opt-in SM121 specialization that preserves DeepSeek's
authoritative 257 routing offsets but reduces the grouped-GEMM descriptor
count from 256 to a fixed 24 slots.  Both arms used the same patched release
module; `FLASHINFER_CUTLASS_ACTIVE24=0` selected the untouched control path and
`=1` selected the compact path.  Each row is a captured canonical C4 target
route, real layer-0 NVIDIA NVFP4 weights, TP rank 0/2, five repeats of 1,000
CUDA-graph launches, and 50 warmups.

| sample | active experts | max rows/expert | control graph | active24 graph | active24 delta |
|---:|---:|---:|---:|---:|---:|
| 131 | 24 | 1 | 792.800 us | 792.640 us | +0.02% |
| 0 | 16 | 3 | 546.016 us | 545.792 us | +0.04% |
| 13 | 15 | 2 | 510.528 us | 511.168 us | -0.13% |
| 2178 | 7 | 4 | 259.296 us | 262.368 us | -1.17% |
| 242 | 20 | 4 | 663.808 us | 670.592 us | -1.01% |

All five arm pairs produced byte-identical eager output SHA-256 values, all
CUDA graphs captured, and every numeric/activity gate passed.  The
specialization is therefore correct but **not a performance win**.  The
existing CUTLASS grouped kernel already makes zero-M descriptors effectively
free; real time follows the number and multiplicity of active expert rows.
This path must not be promoted into serving.

## Immutable inputs and builds

- source revision: `92d053d86e1a45112e5f6e10d0ee50e43b7fbb6f`
- pinned FlashInfer revision: `0472b9b3f2fba11b463f8526f390297d52a8aad7`
- control kernel/header SHA-256: `fd24f5f8...` / `d5562b10...`
- active24 kernel/header SHA-256: `9a8fc3ab...` / `7be6f6f2...`
- control release SO SHA-256: `dbdfcdffbab51a5a5a6fe923592607b6356afa0950763ec5a60208c2cd324d1e`
- active24 release SO SHA-256: `6ae932156bd6dad5ee6bb7306a92be0fdf0acb20d9a55540b3ed01a9a136710e`
- captured layer-sample routes SHA-256:
  `2b687fcc275984a37b3c5777bfa73b7f6e3a0de7e38abac38f0390d918449a88`
- candidate image:
  `sha256:105cb6b85510c27ad3e772bed971efa30a6847058e1b7263b45b2141626e6726`

Raw JSON, per-run logs, both JIT build logs, and `SHA256SUMS` are preserved in
this directory.
