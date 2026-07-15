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
  caller-owned scratch, CUDA-graph-safe execution, and GB10 small-M tuning;
- adds two-node Compose/start/update tooling plus the separate real-time
  dashboard and controlled performance harness.

The exact file-level implementation is described in
[docs/implementation.md](docs/implementation.md).

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
ghcr.io/anemll/dspark-vllm-gx10:0.1.0
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

Best aggregate output throughput from the controlled 512-token workload:

| Concurrency | Previous runtime | vLLM 0.25 candidate | Gain |
|---:|---:|---:|---:|
| 1 | 40.7 tok/s | 48.5 tok/s | 19.1% |
| 2 | 59.1 tok/s | 70.4 tok/s | 19.2% |
| 4 | 91.4 tok/s | 103.5 tok/s | 13.2% |

Raw results and the dependency-free client are in `benchmarks/`.

Prefill can be measured at exact 1K, 2K, 4K, 8K, 16K, and 32K input lengths.
The harness records client TTFT plus vLLM's server-side prefill duration and
computed-token count. It uses reproducible, unique token-ID prompts so the
before and after runs receive identical input without prefix-cache reuse:

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
