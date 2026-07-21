---
pipeline_tag: text-generation
base_model:
  - nvidia/DeepSeek-V4-Flash-NVFP4
  - deepseek-ai/DeepSeek-V4-Flash-DSpark
license: mit
library_name: vllm
tags:
  - deepseek-v4
  - nvfp4
  - w4a4
  - dspark
  - dgx-spark
  - tensor-parallel
---

# DeepSeek-V4-Flash NVFP4 TP2 W4A4 v1

This is a single-download, two-node GB10 serving bundle. It combines an
offline-prepared NVIDIA NVFP4 W4A4 target with the native three-stage DSpark
speculative draft.

This is **not** a drop-in Transformers checkpoint. The target has a custom
rank-contiguous TP=2 physical layout and requires the matching loader in
[Anemll/dspark-vllm-gx10, branch `dspark-nvfp4-a4w4`](https://github.com/Anemll/dspark-vllm-gx10/tree/dspark-nvfp4-a4w4).

## Bundle layout

| Path | Contents | Runtime role |
|---|---|---|
| repository root | Prepared NVIDIA NVFP4 target | W4A4 FlashInfer CUTLASS target |
| `dspark/` | Native `mtp.0`, `mtp.1`, and `mtp.2` shards plus a draft-only index | Three-stage DSpark draft |

The root target contains 87 Safetensors files, 3,483 tensors, and
168,266,881,608 payload bytes. The draft contains 4,705 tensors and
10,862,838,300 payload bytes. `bundle-manifest.json` records the complete
layout and provenance.

The target and draft remain two explicit model paths at runtime: mount the
repository root as the target and `dspark/` as the speculative model.

## Requirements

- two NVIDIA DGX Spark or equivalent GB10/SM121 nodes;
- TP=2 with a dedicated node-to-node fabric;
- enough local storage for the complete bundle on each node;
- the custom vLLM runtime branch linked above;
- an ARM64 runtime image built from that branch.

The checkpoint is not supported as a one-node TP=1 model. The prepared target
format itself is pinned to TP=2.

## Download

Run on both nodes, choosing a local SSD path with sufficient free space:

```bash
hf download anemll/DeepSeek-V4-Flash-NVFP4-TP2-W4A4-v1 \
  --local-dir /srv/dspark/models/DeepSeek-V4-Flash-NVFP4-TP2-W4A4-v1
```

## Runtime setup

Clone the matching runtime branch on both nodes:

```bash
git clone --branch dspark-nvfp4-a4w4 \
  https://github.com/Anemll/dspark-vllm-gx10.git
cd dspark-vllm-gx10
```

Use an image built from this branch. The legacy `0.1.1` image does not contain
the prepared bulk direct reader unless it has been rebuilt from the branch:

```bash
FINAL_IMAGE=dspark-vllm-gx10:nvfp4-a4w4-v1 ./scripts/build-image.sh
```

Set the following values in both `config/head.env` and `config/worker.env`, in
addition to their normal rank, fabric, NCCL, cache, and rendezvous fields:

```bash
DSPARK_VLLM_IMAGE=dspark-vllm-gx10:nvfp4-a4w4-v1
DSPARK_MODEL_HOST=/srv/dspark/models/DeepSeek-V4-Flash-NVFP4-TP2-W4A4-v1
DSPARK_DRAFT_MODEL_HOST=/srv/dspark/models/DeepSeek-V4-Flash-NVFP4-TP2-W4A4-v1/dspark
SERVED_MODEL_NAME=deepseek-v4-flash-nvfp4-dspark

DSPARK_WEIGHT_LOAD_FORMAT=auto
DSPARK_MOE_BACKEND=auto
DSPARK_SPECULATION_MODE=dspark
MTP_NUM_TOKENS=5
VLLM_DSPARK_CONFIDENCE_SCHEDULER=off
VLLM_DSPARK_CONFIDENCE_THRESHOLD=0.0

VLLM_DSV4_NVFP4_CUTLASS_PREPARED_LOAD=1
VLLM_DSV4_NVFP4_CUTLASS_PREPARED_MANIFEST_SHA256=REPLACE_WITH_MANIFEST_SHA256
VLLM_DSV4_NVFP4_CUTLASS_PREPARED_DIRECT_READ=1
```

Use the first field of `dspark-nvfp4-tp2-repack.json.sha256` as the manifest
value. Keep the global MoE backend at `auto`: the prepared target is scoped to
FlashInfer CUTLASS by its loader, while the native DSpark draft requires its
own backend selection. Direct reads are the prepared-loader default; `0`
selects the slower diagnostic mmap path.

Start rank 1 first, then rank 0:

```bash
# WORKER / rank 1
./scripts/start-node.sh config/worker.env

# HEAD / rank 0, after rank 1 enters the rendezvous
./scripts/start-node.sh config/head.env
```

When passwordless worker SSH is configured in the head environment, the head
can launch both ranks:

```bash
./scripts/start-cluster.sh config/head.env config/worker.env
```

## Startup verification

Follow the container log on either rank:

```bash
cid="$(sudo docker ps -q --filter label=com.docker.compose.service=vllm-dspark)"
sudo docker logs -f "$cid"
```

A valid prepared bulk load reports:

- `NVFP4_PREPARED event=enabled ... io_mode=preadv`;
- 43 `event=layer_load` records;
- `event=complete layers=43 reads=344 copies=344 ... io_mode=preadv`;
- 43 post-load records with `transforms=0` and backend
  `FLASHINFER_CUTLASS`.

After startup:

```bash
curl -fsS http://HEAD_HOST:8888/health
curl -fsS http://HEAD_HOST:8888/v1/models
```

## Validated load performance

The bulk direct reader was validated end to end on two GB10 nodes:

| Measurement | Previous prepared path | Bulk direct reader | Improvement |
|---|---:|---:|---:|
| Slower-rank target load | 558.19 s | **65.23 s** | **8.56x** |
| Complete head model load | 595.90 s | **108.54 s** | **5.49x** |
| Head start to API server | n/a | 231.64 s | n/a |

The run reached HTTP health 200, exposed the expected model, kept both ranks
out of OOM state, and returned a coherent smoke response. Raw evidence is in
[`nvfp4-prepared-direct-read-full-3689b1c.json`](https://github.com/Anemll/dspark-vllm-gx10/blob/dspark-nvfp4-a4w4/benchmarks/results/nvfp4-prepared-direct-read-full-3689b1c.json).

## Prefill comparison

These are warmed, server-side prefill rates on the same two-node TP=2 cluster.
Prefill is target-only and does not benefit from speculative decoding.

| Input tokens | FP8/B12X production | NVFP4 W4A4 | Gain | Production TTFT | W4A4 TTFT |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 2,033.0 tok/s | 2,242.5 tok/s | +10.3% | 0.512 s | 0.463 s |
| 2,048 | 2,252.0 tok/s | 2,473.2 tok/s | +9.8% | 0.920 s | 0.835 s |
| 4,096 | 2,320.7 tok/s | 2,659.3 tok/s | +14.6% | 1.776 s | 1.552 s |
| 8,192 | 2,184.2 tok/s | 2,593.5 tok/s | +18.7% | 3.765 s | 3.173 s |
| 16,384 | 2,203.8 tok/s | 2,501.7 tok/s | +13.5% | 7.455 s | 6.573 s |
| 32,768 | 2,176.1 tok/s | 2,477.3 tok/s | +13.8% | 15.119 s | 13.264 s |

The reports are same-size aggregates, not paired identical-prompt trials. See
the [comparison and raw-artifact links](https://github.com/Anemll/dspark-vllm-gx10/blob/dspark-nvfp4-a4w4/benchmarks/results/prefill-v0251-vs-nvfp4-a4w4.md).

## Decode and agentic comparisons

All rows use MTP=5, confidence scheduling off, probabilistic draft sampling,
no draft/verify overlap optimization, temperature zero, and 512 requested
tokens per stream.

| Workload | C | FP8/B12X + DSpark | W4A4 + DSpark | Comparison | W4A4 mean accepted length |
|---|---:|---:|---:|---:|---:|
| Canonical chat/tool | 4 | **105.48 tok/s** | 96.02 tok/s | **-9.0%** | 3.127 |
| `tool_agentic` | 8 | not archived cleanly | **360.68 tok/s** | no matched control | 5.270 |

The canonical control is a three-trial median; its W4A4 row is one
post-promotion run. The agentic row is a clean W4A4 production measurement;
the older timing-instrumented concurrency study is deliberately not used as a
control. The agentic prompt SHA-256 is
`6173a7ae0ea3c64b364d0c405be28808efb8486c68a7011e966d31ce222c1736`,
and all eight W4A4 streams passed the automated no-collapse diagnostic.

The report is intentionally mixed rather than marketing a single decode
number: W4A4 lost about 9% in the exploratory canonical comparison, while the
high-acceptance agentic path reached 360.68 aggregate tok/s without a clean
legacy control. See the
[full serving comparison and raw JSON links](https://github.com/Anemll/dspark-vllm-gx10/blob/dspark-nvfp4-a4w4/benchmarks/results/w4a4-dspark-serving-comparison.md).

## Limitations

- This artifact is tied to the custom prepared-loader contract and TP=2.
- It is not supported by stock Transformers or stock vLLM.
- The target and speculative draft have separate physical layouts and must be
  mounted at the two paths shown above.
- Decode performance depends strongly on prompt content and draft acceptance;
  report prompt, concurrency, acceptance, output length, and timing convention
  with every result.
- The agentic no-collapse diagnostic is not a comprehensive quality
  evaluation.
- The target originates from NVIDIA's NVFP4 checkpoint and the draft from the
  native DeepSeek V4 Flash DSpark lineage; review the upstream licenses and
  terms as well as the files included here.

## Reproducibility

- Runtime source:
  [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10/tree/dspark-nvfp4-a4w4)
- Prepared format manifest: `dspark-nvfp4-tp2-repack.json`
- Bundle manifest: `bundle-manifest.json`
- Benchmark summary:
  [`w4a4-dspark-serving-comparison.md`](https://github.com/Anemll/dspark-vllm-gx10/blob/dspark-nvfp4-a4w4/benchmarks/results/w4a4-dspark-serving-comparison.md)
