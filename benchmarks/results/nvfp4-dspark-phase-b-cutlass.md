# Phase B: archived NVFP4 CUTLASS evidence

This report extracts the immutable real-checkpoint SM121 evidence from `.local/results/20260717T163428Z-fddb151/ledger.md`. The source ledger is 15,059 bytes with SHA-256 `72b7961338893847af1c850fab2a1ed6ada4f6261d3cbbf24fd2578c5f82bede`.

Scope: official NVIDIA DeepSeek V4 Flash NVFP4, W4A4, FlashInfer CUTLASS, real layer 0, TP=2 rank 0. These are routed-MoE microkernel measurements, not end-to-end serving or DSpark/MTP results.

## CUTLASS graph gate

| M | Eager median (us) | CUDA graph median (us) | Correctness |
|---:|---:|---:|---|
| 1 | 229.664 | 224.320 | cosine 1.0, NRMSE 0.0, finite/nonzero, pass |

Recorded artifacts: log SHA-256 `9054735a57b2c1bc620d03b1b7f90618185f10dcad61a2e5a9fae50f166684f2`; JSON SHA-256 `c6d6b7f0bcfdceb01ba840de3ceed8c386de5a7a832c9b1770eb18fc3131d430`.

## CUTLASS eager frontier

| M | Median (us) |
|---:|---:|
| 2 | 414.976 |
| 4 | 786.304 |
| 6 | 1,162.976 |
| 12 | 2,289.728 |
| 64 | 8,348.864 |
| 128 | 8,180.224 |
| 512 | 8,597.408 |
| 2,048 | 10,222.016 |

All archived frontier rows passed with zero swap growth. Recorded artifacts: log SHA-256 `d67b786575bcf11efe4ef8914cd3364524d1efdc003f5836bc955b0c8d95d`; JSON SHA-256 `55e6ba570082b692e4f7428e233acba54dbb8edef77c4220ed3e993a03b943a4`.

## CUTLASS-first B12X bridge

The ratio is throughput-oriented: `CUTLASS latency / B12X latency`; values above 1 mean B12X is faster.

| M | CUTLASS median (us) | B12X median (us) | B12X/CUTLASS throughput | Cosine | NRMSE |
|---:|---:|---:|---:|---:|---:|
| 1 | 226.624 | 217.120 | 1.043773x | 0.9877584 | 0.1560521 |
| 4 | 795.616 | 803.072 | 0.990716x | 0.9871223 | 0.1604985 |
| 128 | 8,417.472 | 9,186.336 | 0.916304x | 0.9865133 | 0.1641534 |
| 2,048 | 10,077.376 | 9,904.320 | 1.017473x | 0.9865086 | 0.1642079 |

All bridge correctness gates passed; the workspace contract was correctly non-applicable below M=8192. B12X is shape-dependent and 8.37% slower at M=128, so the archived decision was to retain CUTLASS. Recorded artifacts: log SHA-256 `bf4c2bd85390d7db3cc54b25b9341fa458f1cfbf80c26123727440b3bd4c9de8`; JSON SHA-256 `1e386ad0d9649f85ea29db616f38129a5796e496c53991da6b9514f4f6169878`.

## Evidence gaps

- **M=24 is absent.** It was not measured in this archived gate chain and is not inferred.
- The local archive contains only `ledger.md`; the raw graph/frontier/bridge files are not present locally, so their recorded hashes were not re-verified here.
- These measurements do not establish TP=2 API throughput, decode speed, speculative acceptance, or DSpark performance.
- No DSpark/MTP draft model was active.
- The M=1 graph and bridge numbers came from separate runs.
