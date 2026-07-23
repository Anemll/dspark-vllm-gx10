# Static token-shared W4A4 packing at DSpark verifier shapes

This bounded HEAD-only component gate tested whether deduplicating FP4
activation packing helps the W4A4 target when DSpark verifies five draft
tokens.  The two tested target-row shapes correspond to:

- `M=24`: concurrency 4 × (5 drafts + 1 target), or 144 routed rows at top-k 6;
- `M=48`: concurrency 8 × (5 drafts + 1 target), or 288 routed rows at top-k 6.

Both arms used the same immutable candidate image, prepared layer-0 rank-0
weights, seed 4104, forced FlashInfer B12X static scheduling, MAC48, CUDA
graphs, 3 warmups, 20 iterations, and 3 repeats.  The control disabled the
new path.  The candidate quantized only the first expert route for each token,
then copied its packed FP4 bytes and E4M3 scale bytes to the remaining five
expert-major destinations behind a resident-grid barrier.

## Result

| Target rows | Existing B12X | Token-shared B12X | Candidate delta | CUTLASS control |
|---:|---:|---:|---:|---:|
| 24 | **4.758200 ms** | 4.773496 ms | **0.32% slower** | 4.536504 ms |
| 48 | **8.435592 ms** | 8.468656 ms | **0.39% slower** | 8.039328 ms |

The candidate misses the predeclared `>=3%` improvement gate at both shapes.
It also fails correctness at `M=48`: eager candidate-vs-CUTLASS cosine is
`0.880434` with normalized RMSE `0.478685`, and CUDA graph vs eager is
nondeterministic (`0.926820` cosine).  `M=24` remains numerically valid but
still regresses latency.

## Decision

Do not integrate or run a full TP=2 serving test.  Activation-pack deduplication
does not address the dominant static-kernel cost at DSpark C4/C8.  The bytes
saved by avoiding five redundant quantizations are offset by route resolution,
fan-out copies, and an additional grid barrier.  A gather/scatter variant that
still materializes expert-major activation rows has the same lower bound.

The next useful optimization must remove or overlap expert GEMM/weight-stream
work, not add another activation-layout handoff.  The accepted W4A4+DSpark
serving default therefore remains FlashInfer CUTLASS; the B12X MAC40 micro
optimization remains useful only for the C1 verifier shape.

## Immutable evidence

- source revision: `8f29b903ea0b80bccee000a96ca149c358669e2d`
- candidate image:
  `sha256:da0ef9bf64df7a6e49e152e0c4f8f8bb82d0752851f975074f182ec635743795`
- accepted shared-input parent:
  `sha256:d38a1c534dc93a03846a80e0a8e5dd4e2e5844ff5c332e6143a5ab8c98cc5464`
- control JSON SHA-256:
  `f63bbcdd3a9bd89c4ce6d6f1111ed50c06ae5e5310eddd40b6bd4afdaa74b523`
- candidate JSON SHA-256:
  `d45fffd9ed46e97ecdd49fc362b3bd5c04ade0fd5db40fed7712d643bf70b8d7`

The raw JSON, logs, and immutable build log are preserved in this directory.
