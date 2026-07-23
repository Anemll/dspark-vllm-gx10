# FlashInfer B12X resident W4A4 — real C4 route gate

This gate replays captured target-only TP=2 route sample 131
(`M=4`, `top_k=6`, 24 active experts) against one real prepared checkpoint
layer on an NVIDIA GB10 (SM121). It compares FlashInfer's resident
`MoEMicroKernel` W4A4 path with the current FlashInfer CUTLASS W4A4 path.

The default resident kernel is a small positive result, but not large enough
to close the serving gap:

| Variant | B12X graph | CUTLASS graph | B12X speedup |
|---|---:|---:|---:|
| default 64x128 | 0.776392 ms | 0.785832 ms | 1.0122x |
| static schedule | 0.773288 ms | 0.784064 ms | 1.0139x |
| 128x64 | 0.797800 ms | 0.786848 ms | 0.9863x |
| 64x64 | 0.795432 ms | 0.795640 ms | 1.0003x |
| 12 MAC clusters | slower | — | 0.8080x |

Default numerical agreement against CUTLASS is finite and active:
cosine `0.986538`, normalized RMSE `0.163559`. The gate uses the same
acceptance limits as the earlier real-layer bridge.

At 43 routed layers, the best observed saving is about 10.8 microseconds per
layer. The canonical C4 service gap requires roughly 52.4 microseconds per
layer, so this kernel cannot close C4 by itself. It remains a useful candidate
for an M=1-only hybrid path, where the existing balanced gate showed a larger
win.
