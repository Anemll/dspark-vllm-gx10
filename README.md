# DSpark vLLM for two DGX Spark / ASUS GX10 nodes

This is a tested two-node GB10 port of the DeepSeek V4 Flash DSpark/NVFP4
serving path to vLLM 0.25.1. It bridges vLLM's DeepSeek V4 runtime to
FlashInfer's native SM120/SM121 sparse-MLA kernel, adds a b12x native-MXFP4 MoE
backend, and packages reproducible deployment, a live dashboard, version
switching, and benchmark evidence.

## Validated configuration

- 2 × NVIDIA DGX Spark or ASUS Ascent GX10 (GB10, SM121, ARM64)
- dedicated high-speed fabric between nodes
- tensor parallelism: TP=2
- DeepSeek V4 Flash DSpark model using NVFP4 DS MLA KV cache
- vLLM source tag `v0.25.1`; runtime reports
  `0.25.2.dev0+g752a3a504.d20260714`
- FlashInfer pinned to `0472b9b3f2fba11b463f8526f390297d52a8aad7`
- b12x pinned to `7dc6fb8fcc6446ea093537d1657df81985fa5f43`

## What this port changes

- adds `nvfp4_ds_mla` as a first-class DeepSeek V4 KV-cache format throughout
  vLLM configuration, quantization, and cache-size accounting;
- uses the tested 584-byte packed sparse-MLA token envelope for both MLA and
  sliding-window cache groups;
- adapts vLLM's FlashInfer SM120/SM121 wrapper to split oversized 256-token SWA
  pages into zero-copy 64-token views while preserving compressed C128 pages;
- supports TP=2's 32 query heads and pads unsupported sparse-index widths to
  FlashInfer's native 128/512/1024 dispatch widths with invalid-slot sentinels;
- adds a modular b12x MXFP4 MoE backend with native weight preparation,
  caller-owned scratch, CUDA-graph-safe execution, GB10 small-M tuning, and
  startup route-pack specialization warmup;
- adds an opt-in ModelOpt NVFP4 W4A4 routed-expert path for DeepSeek V4,
  including mixed dispatch that keeps three-stage DSpark draft experts on
  their native MXFP4 path and preserves DeepSeek's clamped SwiGLU contract;
- adds a strict NVIDIA-target/DSpark-draft hybrid-checkpoint builder and a
  same-weight SM121 W4A4-versus-W4A16 decode/prefill kernel harness;
- adds two-node Compose/start/update tooling plus the separate real-time
  dashboard and controlled performance harness;
- optionally makes TP rank 0 the sole checkpoint payload reader and streams
  only TP rank 1's required raw weight writes into RAM over NCCL/RoCE.

The exact file-level implementation is described in
[docs/implementation.md](docs/implementation.md).

The opt-in, reversible startup weight path is documented in
[docs/ROCE_RAM_WEIGHT_LOADING.md](docs/ROCE_RAM_WEIGHT_LOADING.md).

The NVFP4 evidence audit, checkpoint contract, SM121 harness, and complete
test/optimization plan are documented in
[docs/NVFP4_A4W4_EXTERNAL_EVIDENCE.md](docs/NVFP4_A4W4_EXTERNAL_EVIDENCE.md),
[docs/NVFP4_DSPARK_HYBRID_CHECKPOINT.md](docs/NVFP4_DSPARK_HYBRID_CHECKPOINT.md),
[benchmarks/NVFP4_A4W4_SM121.md](benchmarks/NVFP4_A4W4_SM121.md), and
[docs/NVFP4_A4W4_TEST_AND_OPTIMIZATION_PLAN.md](docs/NVFP4_A4W4_TEST_AND_OPTIMIZATION_PLAN.md).
The hybrid is an integration artifact using NVIDIA's target and the native
DSpark draft; it is not the production abliterated checkpoint lineage.

Model weights are **not** included. The validated one-download prepared target
and DSpark draft bundle is
[anemll/DeepSeek-V4-Flash-NVFP4-TP2-W4A4-v1](https://huggingface.co/anemll/DeepSeek-V4-Flash-NVFP4-TP2-W4A4-v1).
Set `DSPARK_MODEL_HOST` to a model directory you are licensed to use.

## Install

Clone this repository on both nodes:

```bash
git clone https://github.com/anemll/dspark-vllm-gx10.git
cd dspark-vllm-gx10
./scripts/install.sh --role worker
./scripts/install.sh --role head
```

Edit `config/worker.env` and `config/head.env`. Replace every `CHANGEME` value,
verify the model/cache paths, and use the dedicated fabric addresses—not Wi-Fi
or the general LAN.

Start rank 1 first, then rank 0:

```bash
# Worker
./scripts/start-node.sh config/worker.env

# Wait until rank 1 is listening for the rendezvous, then on the head:
./scripts/start-node.sh config/head.env
```

The head API is available at `http://HEAD_HOST:8888`. A successful startup
returns HTTP 200 from `/health` and the runtime string from `/version`.

The head can also launch both ranks when `WORKER_SSH` and `WORKER_REPO_DIR`
are set in `config/head.env`:

```bash
./scripts/start-cluster.sh config/head.env config/worker.env
```

### Prepared W4A4 + DSpark profile

The one-download
[`DeepSeek-V4-Flash-NVFP4-TP2-W4A4-v1`](https://huggingface.co/anemll/DeepSeek-V4-Flash-NVFP4-TP2-W4A4-v1)
bundle uses two explicit runtime paths. Its repository root is the prepared
W4A4 target and its `dspark/` subdirectory is the native three-stage
speculative draft. Set the following values identically in the head and worker
environment files, apart from the normal rank and address fields:

```bash
DSPARK_MODEL_HOST=/srv/dspark/models/DeepSeek-V4-Flash-NVFP4-TP2-W4A4-v1
DSPARK_DRAFT_MODEL_HOST=/srv/dspark/models/DeepSeek-V4-Flash-NVFP4-TP2-W4A4-v1/dspark
SERVED_MODEL_NAME=deepseek-v4-flash-nvfp4-dspark

DSPARK_MOE_BACKEND=auto
DSPARK_SPECULATION_MODE=dspark
MTP_NUM_TOKENS=5

VLLM_DSV4_NVFP4_CUTLASS_PREPARED_LOAD=1
VLLM_DSV4_NVFP4_CUTLASS_PREPARED_MANIFEST_SHA256=REPLACE_WITH_MANIFEST_SHA256
VLLM_DSV4_NVFP4_CUTLASS_PREPARED_DIRECT_READ=1
```

Use the first field of
`dspark-nvfp4-tp2-repack.json.sha256` for the manifest value. Keep the global
MoE backend at `auto`: the prepared target is scoped to FlashInfer CUTLASS by
its loader, while the native DSpark draft requires its own backend selection.
Bulk direct reads are the prepared-loader default; set
`VLLM_DSV4_NVFP4_CUTLASS_PREPARED_DIRECT_READ=0` only to diagnose the older
mmap path.

This profile requires an image built from a revision that includes the
prepared loader and bulk direct reader. Do not combine it with the legacy
`0.1.1` image unless that image has been rebuilt and pinned to such a revision.

For a reload, stop the existing head/rank 0 container first and then the
worker/rank 1 container. Start in the opposite order: worker first, then head.
Follow either rank's startup log with:

```bash
cid="$(sudo docker ps -q --filter label=com.docker.compose.service=vllm-dspark)"
sudo docker logs -f "$cid"
```

A prepared direct-read launch must report `NVFP4_PREPARED event=enabled` with
`io_mode=preadv`, followed by 43 `event=layer_load` records and one
`event=complete` record with `reads=344`, `copies=344`. The validated TP=2 run
loaded the target in 65.23 seconds on the slower rank, completed the head model
load in 108.54 seconds, and made the API ready about four minutes after the
head container started. See
[the archived result](benchmarks/results/nvfp4-prepared-direct-read-full-3689b1c.json).

## Dashboard

The dashboard is a dependency-free Python service. It displays decode/prefill
throughput, active non-zero averages, token totals, DSpark acceptance, request
latency, load state for both TP ranks, vLLM version, temperature, power, GPU
utilization, and optional NVMe temperature.

See [docs/dashboard.md](docs/dashboard.md) for installation and configuration.

On the head, the recommended persistent setup is:

```bash
./scripts/install-dashboard-service.sh
# Edit dashboard/dashboard.env after the first run, then run the installer again.
./scripts/install-dashboard-service.sh
```

Open `http://HEAD_HOST:11001`. Install the service before starting vLLM if the
dashboard should show startup/load state. The systemd installer configures the
restricted container-log helper used to discover the active Compose container;
worker progress additionally requires `DASHBOARD_WORKER_SSH` and its identity
file. `STALE` is expected while the vLLM metrics endpoint is unavailable during
startup, but the two model-load cards should remain available and report
log-derived startup state.

## Update

Pull the repository and prepare a new image tag on each node:

```bash
git pull --ff-only
./scripts/update.sh 0.1.1 config/worker.env
./scripts/update.sh 0.1.1 config/head.env
```

Restart rank 1 first and rank 0 second. The updater preserves a timestamped
backup of the previous environment file.

## Build from source

The prebuilt ARM64 image is published as:

```text
ghcr.io/anemll/dspark-vllm-gx10:0.1.1
```

To reproduce it locally, run `./scripts/build-image.sh`. The script checks out
the exact vLLM commit in `upstream.lock`, applies `overlay/`, builds the vLLM
ARM64 image, and installs pinned FlashInfer and b12x Git revisions. The b12x
pin is intentionally a Git commit: the tested `0.15.3` source was not released
on PyPI.

`docker/Dockerfile.promote-tested` is a maintainer-only release step that adds
OCI source/revision labels and bundled license notices to an image that has
already passed the two-node validation. It does not replace the reproducible
source build above and does not change runtime code.

## Performance

### Prepared W4A4 + DSpark

The current model artifact is
[anemll/DeepSeek-V4-Flash-NVFP4-TP2-W4A4-v1](https://huggingface.co/anemll/DeepSeek-V4-Flash-NVFP4-TP2-W4A4-v1).
It combines an offline-prepared NVIDIA NVFP4 W4A4 target at the repository root
with the native three-stage DSpark draft under `dspark/`. Measurements below
use two GX10/GB10 nodes with TP=2. Prefill is target-only; decode uses MTP=5,
probabilistic draft sampling, and confidence scheduling off.

Warmed server-side prefill improved at all six tested sizes:

| Input tokens | FP8/B12X production | NVFP4 W4A4 | Gain | Production TTFT | W4A4 TTFT |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 2,033.0 tok/s | 2,242.5 tok/s | +10.3% | 0.512 s | 0.463 s |
| 2,048 | 2,252.0 tok/s | 2,473.2 tok/s | +9.8% | 0.920 s | 0.835 s |
| 4,096 | 2,320.7 tok/s | 2,659.3 tok/s | +14.6% | 1.776 s | 1.552 s |
| 8,192 | 2,184.2 tok/s | 2,593.5 tok/s | +18.7% | 3.765 s | 3.173 s |
| 16,384 | 2,203.8 tok/s | 2,501.7 tok/s | +13.5% | 7.455 s | 6.573 s |
| 32,768 | 2,176.1 tok/s | 2,477.3 tok/s | +13.8% | 15.119 s | 13.264 s |

To isolate target-model decode from speculative acceptance, we first disabled
DSpark completely and required the speculative counters to remain inactive.
All targets used the same canonical prompt, 512 output tokens, temperature 0,
and TP=2. Values are best aggregate throughput, with the median in parentheses.
The standard W4A4 arm used three C=1 trials and five fully warmed C=4 trials;
the FP8 control used three trials.  The experimental dual arm is included to
show the rejected W4A16 optimization:

| Target-only concurrency | FP8/B12X, no draft | W4A4/CUTLASS, no draft | Experimental W4A16 dual | W4A4/CUTLASS vs FP8 | W4A16 vs CUTLASS |
|---:|---:|---:|---:|---:|---:|
| 1 | **27.40 (27.37) tok/s** | 27.19 (27.12) tok/s | 27.25 (27.21) tok/s | -0.8% / -0.9% | +0.2% / +0.4% |
| 4 | **77.49 (76.85) tok/s** | 74.11 (73.66) tok/s | 70.04 (69.83) tok/s | -4.4% / -4.2% | **-5.5% / -5.2%** |

Standard FlashInfer CUTLASS is the accepted W4A4 decode path.  The valid
W4A16 service branch was 5.2% slower at C=4 by median and remains default-off;
its balanced-route layer win did not survive the correlated routes of the
canonical service workload.  The remaining W4A4-vs-FP8 deficit is already
present with the target alone and is not primarily a DSpark acceptance
difference.  Prefill still favors W4A4, as shown above.  The exact optimization
gates and raw artifacts are documented in
[decode-w4a4-kernel-optimization-646be4d.md](benchmarks/results/decode-w4a4-kernel-optimization-646be4d.md)
and the [valid dual-decode service gate](benchmarks/results/w4a4-decode-port-20260722/service-dual-dispatch/README.md).

Decode with speculation remains prompt- and acceptance-dependent. A same-prompt
canonical check with **MTP=5 on both sides**, probabilistic draft sampling, and
confidence scheduling off put W4A4 within about 3--4% of the preceding
FP8/B12X v0.25 candidate:

| Canonical concurrency | FP8/B12X + DSpark (MTP=5) | W4A4 + DSpark (MTP=5) | W4A4 delta |
|---:|---:|---:|---:|
| 1 | **48.49 tok/s** | 47.13 tok/s | -2.8% |
| 4 | **103.48 tok/s** | 99.44 tok/s | -3.9% |

The high-acceptance `tool_agentic` prompt benefits from longer drafts. The
following grid uses the exact prompt SHA-256
`6173a7ae0ea3c64b364d0c405be28808efb8486c68a7011e966d31ce222c1736`,
512 output tokens per stream, confidence OFF, no draft/verify overlap, one
shape warm-up, and two measured trials per cell. Values are best aggregate
throughput:

| MTP draft tokens | C=1 | C=2 | C=4 | C=8 |
|---:|---:|---:|---:|---:|
| 1 | 39.7 tok/s | 66.5 tok/s | 96.0 tok/s | 146.7 tok/s |
| 2 | 53.2 tok/s | 88.2 tok/s | 119.8 tok/s | 175.3 tok/s |
| 3 | 62.5 tok/s | 94.6 tok/s | 148.3 tok/s | 224.7 tok/s |
| 4 | 69.1 tok/s | **135.6 tok/s** | **157.9 tok/s** | 234.8 tok/s |
| 5 | **76.4 tok/s** | 111.9 tok/s | 156.6 tok/s | **244.2 tok/s** |

MTP=5 remains the general-purpose default: it wins C=1 and C=8 and is
effectively tied at C=4. MTP=4 is the specialized choice for C=2 and is the
best single C=4 trial. Do not compare the canonical and agentic tables as if
they used the same prompt.

The bulk direct reader loaded the prepared target on the slower rank in
**65.23 seconds** and completed the full head model load in **108.54 seconds**.
The earlier 558.19/595.90-second measurements came from an intermediate
non-direct prototype, not the release model, so they are not presented as a
release speedup. The complete methodology, caveats, and raw-artifact links are
in the
[W4A4 serving comparison](benchmarks/results/w4a4-dspark-serving-comparison.md),
[agentic MTP grid](benchmarks/results/decode-w4a4-agentic-mtp-grid.md),
[prefill comparison](benchmarks/results/prefill-v0251-vs-nvfp4-a4w4.md), and
[direct-load result](benchmarks/results/nvfp4-prepared-direct-read-full-3689b1c.json).

### Legacy FP8/B12X production reference

Benchmark model:

- [drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored](https://huggingface.co/drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored)
- DeepSeek V4 Flash, 284B MoE / approximately 13B active parameters
- mixed native checkpoint: FP8 block-quantized non-expert linear tensors plus
  native MXFP4 routed experts, executed as W4A16 by the current b12x path; 48
  Safetensors shards totaling 166,886,535,336 bytes (155.43 GiB) on each node
- `nvfp4_ds_mla` KV cache; NVFP4 describes the DS-MLA cache format, not the
  checkpoint's routed-expert weight/activation format
- TP=2 across two GX10 nodes; server maximum context 350,000 tokens

Single-node reference: the unchanged checkpoint is **not runnable on one
GX10**. Its 155.43 GiB of weight files exceed the node's approximately 121 GiB
of usable unified memory before KV cache and runtime allocations. A controlled
TP=1 launch reached NVIDIA `NV_ERR_NO_MEMORY` before the API became ready, so
there are no valid single-node throughput samples. The full
[fit-check record](benchmarks/results/prefill-v0251-single-node-fit.md) is kept
with the benchmark results.

Best aggregate output throughput from the controlled 512-token workload:

| Concurrency | Previous runtime | vLLM 0.25 candidate | Gain |
|---:|---:|---:|---:|
| 1 | 40.7 tok/s | 48.5 tok/s | 19.1% |
| 2 | 59.1 tok/s | 70.4 tok/s | 19.2% |
| 4 | 91.4 tok/s | 103.5 tok/s | 13.2% |

Raw results and the dependency-free client are in `benchmarks/`.
The post-fix
[route-pack warmup validation](benchmarks/results/route-pack-warmup-v025.md)
includes strict-JIT boundary coverage, a 65K prefill check, and decode
regression results.

Warmed, server-side prefill results on the same two-node TP=2 deployment:

| Input tokens | vLLM 0.21.1 | vLLM 0.25 candidate | Gain |
|---:|---:|---:|---:|
| 1,024 | 1,778.7 tok/s | 2,033.0 tok/s | 14.3% |
| 2,048 | 1,990.5 tok/s | 2,252.0 tok/s | 13.1% |
| 4,096 | 2,083.1 tok/s | 2,320.7 tok/s | 11.4% |
| 8,192 | 2,049.8 tok/s | 2,184.2 tok/s | 6.6% |
| 16,384 | 2,052.6 tok/s | 2,203.8 tok/s | 7.4% |
| 32,768 | 1,901.1 tok/s | 2,176.1 tok/s | 14.5% |

The [comparison](benchmarks/results/prefill-v0211-vs-v0251.md) and
[raw reports](benchmarks/results/) contain TTFT and per-trial details.

Prefill is measured at exact 1K, 2K, 4K, 8K, 16K, and 32K input lengths.
The harness records client TTFT plus vLLM's server-side prefill duration and
computed-token count. It uses reproducible token-ID prompts and checks both
full-prompt and first-16-token prefix hashes. The server's cache-hit counter is
the authoritative guard against prefix-cache reuse.
The default remains concurrency 1; pass `--concurrency 1,2,4` for synchronized
multi-request rows with aggregate input throughput, mean/p95 TTFT, and nested
per-request metrics. An initial warm-up plus one excluded pass at every tested
`(length, concurrency)` shape prevents first-shape compilation from
contaminating the three-trial medians:

```bash
# Run against the previous runtime, then switch the two-node server version.
python3 benchmarks/benchmark_prefill.py --label before --concurrency 1,2,4 \
  --output benchmarks/results/prefill-before.json

# Run the identical matrix against the candidate runtime.
python3 benchmarks/benchmark_prefill.py --label after --concurrency 1,2,4 \
  --output benchmarks/results/prefill-after.json

python3 benchmarks/compare_prefill.py \
  benchmarks/results/prefill-before.json \
  benchmarks/results/prefill-after.json
```

Run these tests on an otherwise idle server. A shape is reported only when all
of its trials have the expected request count, exact per-request usage and
Prometheus/computed-token totals, and zero cache hits; otherwise its summary is
marked invalid while raw per-request diagnostics are retained. At concurrency
2/4, aggregate input tokens divided by batch TTFT is the scaling metric. The
server duration-derived rate is a mean request-service rate because its
denominator sums overlapping per-request durations.

## Important operational note

Start the worker before the head. Starting both simultaneously can leave the
distributed initialization waiting on TCPStore/NCCL. The provided scripts do
not store passwords or SSH credentials.

## Licensing and attribution

Repo-local dashboard, deployment, documentation, and benchmark work is MIT
licensed under [LICENSE](LICENSE). The vLLM-derived files under `overlay/`
remain Apache-2.0; the complete Apache text is included at
[LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).

See [CREDITS.md](CREDITS.md) for exact dependency revisions and explicit credit
to vLLM, FlashInfer, Luke Alonso/b12x, voipmonitor, Keys/drowzeys, Rafael
Caricio, MiaAI-Lab, TonyD2Wild, Fraser Price, and roady001. Model weights are
not included or relicensed.
