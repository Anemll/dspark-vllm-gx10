# W4A4 decode optimization: revision 646be4d

This record is for the target-only, no-MTP decode path. It uses the prepared
DeepSeek V4 Flash NVFP4 W4A4 checkpoint, TP=2, temperature 0, the canonical
prompt, and a 512-token cap. The optimization is intentionally scoped to
decode; prefill and DSpark/MTP code are unchanged.

## Promoted change

Revision `646be4dba058d1010fd53543d1e65ac4f0bc0061` keeps the existing compact
expert scheduler for multi-token decode and enables the direct unique-route
micro path only for `M=1`. This removes the compaction/row-map overhead for a
single token while preserving expert weight reuse when concurrent requests
route multiple tokens to the same expert. The revision is pushed on branch
`dspark-nvfp4-a4w4`.

## API result

The promoted image was run after one clean startup and CUDA-graph warmup. The
first request of each run is not used as a kernel comparison; all rows below
are from the same canonical prompt and no-speculation contract.

| Concurrency | Best aggregate | Other warm aggregates | TTFT |
|---:|---:|---:|---:|
| 1 | **27.7 tok/s** | 27.0, 27.5, 26.9, 26.9 | 0.17–0.18 s |
| 4 | **72.9 tok/s** | 71.7, 71.5 | 0.32–0.38 s |

Two additional C4 rows (43.5 and 53.9 tok/s) had queueing TTFT of 9.80 s and
4.95 s and are retained in the raw JSON but excluded from the warm throughput
summary. Raw artifacts:

- `decode-target-only-w4a4-b12x-646be4d-service-c1.json`
- `decode-target-only-w4a4-b12x-646be4d-service-c4.json`

For context, the preceding accepted W4A4 target-only run was 27.6 tok/s at C1
and 72.7 tok/s at C4; the promoted change therefore preserves C4 while
removing the C1 regression. The FP8/B12X control was 27.4 tok/s at C1 and
77.5 tok/s at C4, so the remaining C4 gap is in the target W4A4 kernel, not
DSpark acceptance.

## Kernel gates and rejected directions

All microkernel gates used the real prepared layer-0 weights and required
finite/nonzero output, graph/eager parity, cosine at least 0.98, and normalized
RMSE at most 0.25.

| Variant | M=4 graph median | Result |
|---|---:|---|
| Promoted FlashInfer W4A4 | ~0.766 ms | reference |
| Upstream B12X 0.23 W4A4 | 1.282 ms | reject, 67.5% slower |
| Upstream B12X 0.23 W4A8/NVFP4 | 1.301 ms | reject, 69.8% slower |
| FC1 MMA disabled | 0.769 ms | reject; MMA is not the exposed bottleneck |
| FC2 MMA disabled | 0.766 ms | reject; same conclusion |
| Input quantization disabled | ~0.758 ms | reject; only a small upper bound |
| Scatter/atomics disabled | ~0.769 ms | reject |
| Grid barrier disabled | ~0.767 ms | reject |
| Pipeline depth 1 | 0.771 ms | reject; slower |
| Pipeline depth 4 request | compiled back to depth 2 | no additional stage available |
| Dynamic scheduler | 0.887 ms | reject |
| Static scheduler forced at M=4 | 0.768 ms | reject |
| O(1) compact work mapping for M<=4 | 0.759 ms balanced; 0.203 ms hot-route | reject; <1% |

The hard upper-bound tests show that the remaining M=4 deficit is dominated by
TMA/shared-memory movement and launch scheduling, rather than FP4 MMA,
activation quantization, route scatter, or the resident barrier. A larger
rewrite would need a fused/interleaved weight-and-scale transfer or a new
W4A4 TMA layout; no such rewrite is promoted by this campaign.

## Startup/load note

The prepared loader completed all 43 layers with 344 reads and 344 copies in
62.588 s; total target model load was 68.379 s. The roughly 3.5-minute API
startup is dominated by one-time TileLang/DeepGEMM compilation and CUDA-graph
capture, not checkpoint I/O. Subsequent starts can reuse the on-disk caches.
