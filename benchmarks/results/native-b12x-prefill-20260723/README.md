# Native B12X W4A16 prefill-shape gate (2026-07-23)

Decision: **passed**. Use one native-packed B12X payload for both prefill and
decode; do not build a dual-layout runtime.

This real-layer SM121 gate compared native-packed B12X W4A16 with the current
CUTLASS W4A4 path using identical NVIDIA layer-0 weights, balanced routes,
inputs, CUDA graphs, and TP-rank-0 shapes.

| M | Native B12X W4A16 | CUTLASS W4A4 | Native latency delta | Native throughput gain |
|---:|---:|---:|---:|---:|
| 512 | 7.880344 ms | 8.642176 ms | -8.82% | +9.67% |
| 1,024 | 8.363048 ms | 9.233904 ms | -9.43% | +10.41% |
| 2,048 | 9.542912 ms | 10.272320 ms | -7.10% | +7.64% |

All activity, graph/eager, and cross-quant numeric gates passed.  Cosine was
0.98709--0.98714 and normalized RMSE was 0.16041--0.16076.

Combined with the already-banked decode-shape evidence:

- native B12X is approximately 8.5--9.0% faster than CUTLASS at M=24/48;
- converted NVIDIA native B12X is within 0.33% of the abliterated native B12X
  path at M=24/48; and
- native B12X is now 7.6--10.4% faster than CUTLASS at M=512--2048.

Therefore the winning integration is simpler than the proposed dual view:
losslessly collapse E4M3/K16 scales to E8M0/K32 once at load, pack the existing
FP4 payload once into B12X native layout, and dispatch B12X W4A16 for both
prefill and decode.  No second full weight payload, per-layer phase repack, or
DeepGEMM path is justified.

## Provenance

- Harness revision: `929a64b6bfd799ddcf6e415b95737027b56a1507`
- Image: `sha256:da0ef9bf64df7a6e49e152e0c4f8f8bb82d0752851f975074f182ec635743795`
- GPU: NVIDIA GB10, SM121
- Layer: `model-layer-00000.safetensors`, TP rank 0
- Routing: balanced, seed `4104 + M`
- Timing: CUDA graph, 3 warmups, 20 iterations, 3 repeats, both orders
- JSON SHA-256:
  `6a02ec75b4e19533b9c9c9b2224995728f1bb713d36051c3c82543ebd346357d`
- Log SHA-256:
  `682608bf0794c4c46995dc02caec60f86e7f8680b02c4053ce434f2b41af4ab2`

Node artifact root:
`/home/anemll/nvfp4-artifacts/20260723-929a64b-native-prefill`

