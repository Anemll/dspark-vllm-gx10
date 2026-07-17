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

Model weights are **not** included. Set `DSPARK_MODEL_HOST` to a model directory
you are licensed to use.

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

## Dashboard

The dashboard is a dependency-free Python service. It displays decode/prefill
throughput, active non-zero averages, token totals, DSpark acceptance, request
latency, load state for both TP ranks, vLLM version, temperature, power, GPU
utilization, and optional NVMe temperature.

See [docs/dashboard.md](docs/dashboard.md) for installation and configuration.

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
computed-token count. It uses reproducible, unique token-ID prompts so the
before and after runs receive identical input without prefix-cache reuse.
An initial warm-up plus one excluded pass at every tested input length prevents
first-shape compilation from contaminating the three-trial medians:

```bash
# Run against the previous runtime, then switch the two-node server version.
python3 benchmarks/benchmark_prefill.py --label before \
  --output benchmarks/results/prefill-before.json

# Run the identical matrix against the candidate runtime.
python3 benchmarks/benchmark_prefill.py --label after \
  --output benchmarks/results/prefill-after.json

python3 benchmarks/compare_prefill.py \
  benchmarks/results/prefill-before.json \
  benchmarks/results/prefill-after.json
```

Run these tests on an otherwise idle server. The harness detects overlapping
requests and excludes contaminated server-side trials from its median.

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
