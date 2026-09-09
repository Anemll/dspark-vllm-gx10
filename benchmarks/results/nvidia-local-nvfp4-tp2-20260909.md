# NVIDIA 0731 NVFP4 local-NVMe TP2 decision

The NVIDIA `DeepSeek-V4-Flash-0731-NVFP4` checkpoint was staged to local NVMe
on both GB10 nodes before this run. The target index and all 48 referenced
target shards matched on both nodes. The DSpark loader then selected and
loaded all three indexed MTP shards (`model-00046` through `model-00048`).

Candidate source revision: `d547047e61627c2e3e86cdbc0e0d5a4052eee758`.
The target used `FLASHINFER_CUTLASS` W4A4 NVFP4 MoE, the draft used
`flashinfer_b12x`, and the cache used `nvfp4_ds_mla`. This is a text-only
candidate; it is not evidence for Vision-Exp correctness.

## Matched C1 explanation gate

The canonical fixed-prompt benchmark used seed 5205, thinking disabled, one
excluded 128-token warm-up, and one 512-token measured request per independent
run. Server counters isolated exactly one request and its 512 output tokens in
each measured wave.

| Run | Server decode tok/s | Aggregate tok/s | TTFT s | DSpark acceptance |
|---|---:|---:|---:|---:|
| local-NVFP4 run 1 | 29.491 | 29.157 | 0.232 | 40.71% |
| local-NVFP4 run 2 | 28.294 | 27.986 | 0.234 | 37.85% |
| accepted 0731 native baseline median | 47.610 | — | 0.177 | 41.9% |

The independent candidate runs are approximately 39–41% below the accepted
native server-decode baseline. Draft acceptance is comparable to native, so
the regression is consistent with the required W4A4 target backend rather
than missing DSpark draft weights. The candidate was rejected and the cached
native image was restored with worker-before-head startup. Its health, version,
model-list, and dashboard gates passed after recovery.

Raw per-run reports remain under the ignored local result directory
`d547047-20260909`; both-rank candidate logs and container metadata were
preserved on the Spark nodes before replacement.
