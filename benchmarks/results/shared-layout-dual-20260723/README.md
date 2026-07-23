# Shared-payload dual-dispatch gate (2026-07-23)

This gate tested whether the prepared NVIDIA FP4 payload can remain in its
CUTLASS `[up, gate]` layout while serving both:

- FlashInfer CUTLASS W4A4 for prefill; and
- B12X W4A16 with a losslessly derived E8M0/K32 scale view for decode.

The run used one real layer, TP rank 0, balanced routes, CUDA graphs, and the
ABI-coherent `dev-8f29b90-static-shared` image on an SM121 GB10.  The native
B12X arm had a private packed-layout oracle copy.  The shared B12X and CUTLASS
arms provably aliased the same original FP4 payload.

| M | Shared/modelopt B12X | Native-packed B12X | CUTLASS W4A4 | Shared vs native | Shared vs CUTLASS |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.284896 ms | 0.199904 ms | 0.218304 ms | +42.52% | -23.37% |
| 4 | 1.149136 ms | 0.707696 ms | 0.786576 ms | +62.38% | -31.55% |
| 24 | 7.384704 ms | 4.161632 ms | 4.548800 ms | +77.45% | -38.40% |
| 48 | 13.159632 ms | 7.386256 ms | 8.050624 ms | +78.16% | -38.82% |

At the decisive M=24/48 shapes, shared/modelopt B12X and native-packed B12X
were bit-exact (cosine 1.0, NRMSE 0, max-absolute error 0).  CUTLASS
cross-quant numerics also passed (cosine approximately 0.9871 and NRMSE
approximately 0.1605).  Thus the large-M result is a pure layout/kernel
performance difference, not a semantic mismatch.

The shared-payload design **fails**.  Native-packed B12X remains about
8.5--9.0% faster than CUTLASS, but B12X's modelopt/shared-layout kernel is
about 38.4--38.8% slower than CUTLASS at M=24/48.  Generating only a second
scale view cannot preserve decode parity.  The weight payload's native B12X
packing is performance-critical.

M=1/4 shared/modelopt output did not match the native/CUTLASS references in
this image, independently confirming that its small-M direct-micro scale ABI
is not a promotable fallback.

Artifacts:

- `shared-layout-v3.json` — SHA-256
  `8ce2e1edfe40639de935ce02899e3e593b0deddb6ea0f3e20ba2c25a6ee3bbc2`
- `shared-layout-v3.log` — SHA-256
  `2ef28a2510496d9d978d659b6c40a62730cc882b31f7e05291c9f6c2089ea57a`
- harness revision:
  `1171fa3b220671e19f932edc9f68d66c3ab27c05`

