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

## Connect Pi or Factory Droid with vision

Both clients use the OpenAI-compatible endpoint on the head node. Replace
`HEAD_HOST` below with a LAN address or hostname that is reachable from the
machine running the client. Keep the served model name synchronized with
`SERVED_MODEL_NAME` in `config/head.env`.

### Pi

Merge this provider into `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "dspark": {
      "baseUrl": "http://HEAD_HOST:8888/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "maxTokensField": "max_tokens"
      },
      "models": [
        {
          "id": "deepseek-v4-flash-vision-exp-dspark",
          "name": "DeepSeek V4 Flash Vision · 2× Spark",
          "reasoning": false,
          "input": ["text", "image"],
          "contextWindow": 350000,
          "maxTokens": 32768,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        }
      ]
    }
  }
}
```

Open `/model` with no arguments, wait for the list to refresh, and reselect the
model after changing this file. The `input` list must include both `text` and
`image`, and `images.blockImages` must not be enabled in global or project
settings. Pi can return an image through its Read tool; `@/path/to/image.png`
is also supported.

For optional client-side performance visibility, see the
[model-neutral live decode TPS and TTFT extension](https://gist.github.com/Anemll/f95a14877862f289e19b12586850eded).
It reports final client-observed decode TPS when the provider supplies token
usage and labels its input/TTFT value as an estimate rather than server prefill
throughput.

### Factory Droid 0.218.1

Add this entry under `customModels` in `~/.factory/settings.json`:

```json
{
  "customModels": [
    {
      "model": "gemini-3-flash-preview",
      "displayName": "DeepSeek V4 Flash Vision · 2× Spark",
      "baseUrl": "http://HEAD_HOST:8888/v1",
      "provider": "generic-chat-completion-api",
      "noImageSupport": false,
      "maxContextLimit": 350000,
      "maxOutputTokens": 32768,
      "extraArgs": {
        "model": "deepseek-v4-flash-vision-exp-dspark"
      }
    }
  ]
}
```

The top-level Gemini model value is an intentional local capability alias for
Droid 0.218.1. It makes Droid preserve image content for an otherwise unknown
custom model; it does not contact Google. `baseUrl` still points to this local
vLLM deployment, and `extraArgs.model` is the real model name sent to vLLM.
Keep `provider` set to `generic-chat-completion-api`; a Claude-family alias
triggers Droid's Anthropic-provider validation and is not a valid substitute.

Fully quit and restart Droid after editing the file, start a new session, and
select the custom model. Read-tool images, paste, and drag-and-drop are then
forwarded to DeepSeek V4 Flash Vision. If a request says that it contains too
many images, start a clean turn and add the image only once rather than both
attaching it and reading the same file.

The `350000` value is the server's total context window, while `32768` is only
the maximum generated output. Input plus requested output must still fit in
the context window.

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

### Experimental B12X 1.3 Vision runtime

The native B12X sparse-MLA experiment uses a matched September vLLM/B12X
runtime, rather than installing B12X 1.3 into the July vLLM image. B12X 1.3
removed the legacy integration APIs used by the released overlay, so mixing
those generations is intentionally unsupported. The immutable base digest,
vLLM commit, package versions, and wrapper hashes are recorded in
`config/b12x-vision.lock.json`.

Build the small, hash-guarded Vision candidate on top of that exact base:

```bash
BASE_IMAGE=$(python3 -c 'import json; print(json.load(open("config/b12x-vision.lock.json"))["base_image"])')
WRAPPER_SHA=$(python3 -c 'import json; print(json.load(open("config/b12x-vision.lock.json"))["vision_wrapper_sha256"])')
SOURCE_REVISION=$(git rev-parse HEAD)

docker build --network none \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg SOURCE_REVISION="$SOURCE_REVISION" \
  --build-arg EXPECTED_VL_MODEL_SHA256="$WRAPPER_SHA" \
  --file docker/Dockerfile.b12x-vision \
  --tag "dspark-vllm-gx10:b12x-vision-${SOURCE_REVISION:0:12}" .
```

For native sparse MLA, set `TARGET_ATTENTION_BACKEND=B12X`,
`DSPARK_ATTENTION_BACKEND=B12X`, `TARGET_MOE_BACKEND=b12x`,
`TARGET_LINEAR_BACKEND=b12x`, and `KV_CACHE_DTYPE=fp8` in both role-specific
environment files. Keep the two ranks on the same image ID and configuration.
The default Compose values remain the released July configuration.

The API can preserve older client model names without loading another model.
Keep the preferred response name first and list compatibility names separated
by spaces:

```bash
SERVED_MODEL_NAME=deepseek-v4-flash-vision-exp-dspark
SERVED_MODEL_ALIASES=deepseek-v4-flash-0731-dspark
```

Every listed name is advertised by `/v1/models` and resolves to the same
checkpoint. Prometheus metrics and response bodies use `SERVED_MODEL_NAME`.

### Accepted Vision-Exp serving profile

The accepted two-Spark profile is recorded in
[`config/vision-winner.lock.json`](config/vision-winner.lock.json). It serves
the complete, jointly trained official Vision-Exp checkpoint with DSpark K5;
it is not the rejected NVIDIA-0731/vision tensor transplant or the rejected
community Vision-Exp NVFP4 conversion. Use the
[`docs/vision-winner-model-card.md`](docs/vision-winner-model-card.md) template
when publishing the model or serving package; it records the exact sources,
credits, weight provenance, and validation limits without attributing the
DeepSeek vision weights to NVIDIA.

Keep the role-specific network and filesystem values from each node, and use
these identical serving values on both ranks:

```bash
SERVED_MODEL_NAME=deepseek-v4-flash-vision-exp-dspark
SERVED_MODEL_ALIASES=deepseek-v4-flash-0731-dspark
MAX_MODEL_LEN=350000
MAX_NUM_SEQS=4
MAX_NUM_BATCHED_TOKENS=2048
GPU_MEMORY_UTILIZATION=0.78
KV_CACHE_MEMORY_BYTES=10737418240
KV_CACHE_DTYPE=fp8
MTP_NUM_TOKENS=5
TARGET_MOE_BACKEND=b12x
TARGET_LINEAR_BACKEND=b12x
TARGET_ATTENTION_BACKEND=B12X
DSPARK_ATTENTION_BACKEND=B12X
COMPILATION_CONFIG={"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}
```

The accepted profile intentionally omits `--limit-mm-per-prompt`. In this
runtime that restores vLLM's default allowance of 999 images per prompt; the
350,000-token context window remains the natural request bound. Do not add an
empty or zero image limit: zero disables image inputs.

## Performance

Benchmark model:

- [drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored](https://huggingface.co/drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored)
- DeepSeek V4 Flash, 284B MoE / approximately 13B active parameters
- FP8 weights (E4M3, UE8M0 scales, 128x128 blocks): 48 Safetensors
  shards totaling 166,886,535,336 bytes (155.43 GiB) on each node
- `nvfp4_ds_mla` KV cache; NVFP4 describes the DS-MLA cache format, not the
  checkpoint's FP8 weight format
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
The [content and visual benchmark suite](docs/benchmark-suite.md) adds eight
text workloads with separate decode/TTFT/DSpark metrics and an independently
gated visual protocol. It does not replace the historical results below or
claim that a text-only checkpoint supports images.
The [text and vision benchmark report](benchmarks/results/content-0731-20260905.md)
includes all eight September 5 text workloads at C1/C2/C4, plus a separately
measured September 6 Vision-Exp addendum: 15/15 synthetic correctness passes
and partial 512-pixel C1/C2 timings. Unmeasured image sizes and C3/C4 vision
results are explicitly marked; the different checkpoints are not a matched A/B.
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
