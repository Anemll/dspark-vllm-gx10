# DeepGEMM W4A8 one-layer gate (SM121)

Decision: **rejected**. Do not convert the full prepared checkpoint for this
backend.

> Scope correction: this result rejects only the experimental
> `DeepGemmFP4Experts` W4A8 backend. It does **not** test or reject converting
> the prepared checkpoint back to the abliterated target's exact native-MXFP4
> layout and running the production `FlashInferB12xExperts` path. Banked
> production evidence proves the abliterated target uses
> `--moe-backend flashinfer_b12x`; DeepGEMM serves the DSpark draft.

This gate reverses the prepared checkpoint's exact NVFP4 scale expansion to
native MXFP4 E8M0/K32, restores DeepGEMM's `[gate, up]` W13 order without
changing the packed E2M1 payload, and compares DeepGEMM W4A8 against the
accepted prepared-NVFP4 FlashInfer CUTLASS W4A4 path. Both arms use identical
balanced routes and activations from real prepared layer 0, TP rank 0.

| M | DeepGEMM MXFP4 W4A8 | CUTLASS NVFP4 W4A4 | DeepGEMM delta | Cosine | NRMSE |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.232592 ms | 0.218800 ms | 6.3% slower | 0.985882 | 0.168150 |
| 4 | 0.921632 ms | 0.783480 ms | 17.6% slower | 0.986538 | 0.163852 |
| 24 | 5.507656 ms | 4.536512 ms | 21.4% slower | 0.986153 | 0.166188 |
| 48 | 9.745064 ms | 8.036736 ms | 21.3% slower | 0.986092 | 0.166595 |

All eager/graph and numerical gates passed. The promotion rule required
DeepGEMM to be at least 3% faster at both M=24 and M=48; observed speedups were
0.8237x and 0.8247x. The failure is therefore kernel performance, not the
lossless conversion or activation semantics.

## Provenance

- Benchmark commit: `7a669ee`
- Image: `sha256:da0ef9bf64df7a6e49e152e0c4f8f8bb82d0752851f975074f182ec635743795`
- Image revision: `8f29b903ea0b80bccee000a96ca149c358669e2d`
- GPU: NVIDIA GB10, SM121
- Layer: `model-layer-00000.safetensors`, TP rank 0
- Routing: balanced, seed 4104 + M
- Timing: CUDA graph, 3 warmups, 20 iterations, 3 repeats, both execution orders
- JSON SHA-256: `fb4e6d33b56cde4113f12021c0525230dd4961f3b12eb1d8b42e0c788383ca5c`
- Log SHA-256: `87e8825da3743480c8f3c06086aaa5fafeba038235cfd39ea9d1e5c7f4206d4a`

Node artifact root:
`/home/anemll/nvfp4-artifacts/20260723-5182f34-deepgemm-w4a8`
