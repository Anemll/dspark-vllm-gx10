# Prepared NVFP4 dual-decode real-layer gate

This gate used prepared DeepSeek-V4 layer 0, TP rank 0, balanced routing, and
CUDA graphs on GB10 (SM121).  It instantiated the serving
`NvFp4CutlassW4A16DualExperts` class: CUTLASS W4A4 remains authoritative for
M=1 and prefill, while uniform decode M=2..8 uses the single-copy B12X W4A16
path.

| M | Selected branch | Dual median | W4A4 reference | Speedup |
|---:|---|---:|---:|---:|
| 1 | FlashInfer CUTLASS W4A4 | 0.215408 ms | 0.214648 ms | 0.9965x |
| 2 | B12X W4A16 | 0.376632 ms | 0.408648 ms | **1.0850x** |
| 4 | B12X W4A16 | 0.736976 ms | 0.782208 ms | **1.0614x** |
| 8 | B12X W4A16 | 1.467168 ms | 1.531776 ms | **1.0440x** |

The gate proved all of the following:

- the FP4 weight payload has identical data and storage pointers in both paths;
- retained E8M0/K32 scale storage is exactly 100,665,344 bytes per layer/rank;
- duplicate FP4 weight bytes are zero;
- frozen planning compiled exact `modelopt/e8m0_k32` TC-decode launches;
- all four CUDA graphs captured with finite, non-zero output;
- M=2/4/8 W4A16 versus W4A4 cosine was 0.98706--0.98768 and NRMSE was
  0.15682--0.16098, within the established real-layer gate.

The accepted kernel is the vector shared-load variant.  Two subsequent
address/packing experiments were rejected: SWAR packing reached 0.737944 ms
at M=4 and signed-i32 stage addressing reached 0.738992 ms, versus 0.736176 ms
for the accepted vector microbenchmark.

Raw evidence is `dual-real-layer/dual-real-layer-v4.json`; its SHA-256 is
`7c7e179cdaf3272673b0beda359614a1e841849d4f0a2c9e389809c0c597f1a5`.
