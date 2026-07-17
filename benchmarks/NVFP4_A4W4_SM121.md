# DeepSeek V4 NVFP4 W4A4 kernel harness

`benchmark_nvfp4_a4w4_sm121.py` isolates one routed-expert layer at the real
NVIDIA DeepSeek V4 Flash geometry used on the two GX10 nodes:

| Quantity | Full model | One TP=2 rank |
|---|---:|---:|
| Hidden size | 4,096 | 4,096 |
| Routed experts | 256 | 256 |
| Experts per token | 6 | 6 |
| Expert intermediate size | 2,048 | 1,024 |

The harness measures decode and prefill shapes with CUDA events, reports
median and p95 latency, attempts CUDA graph capture/replay for every fixed
shape, and emits a JSON provenance record. It does not contact another node or
exercise NCCL/RoCE.

## What is compared

Both paths consume the exact same resident ModelOpt NVFP4 packed weights and
the same routes:

- Hidden states are generated with per-token RMS 1.0 by default, matching the
  scale expected after RMSNorm. The previous `1/sqrt(hidden_size)` distribution
  would have had RMS about 0.0156 and would not meaningfully exercise A4
  activation scaling or the clamp. Use `--input-rms` only as an explicit
  sensitivity experiment, and keep it identical for both paths. Every row
  records the mean/min/max per-token RMS after BF16 conversion and fails if
  either extreme differs from the requested value by more than 1%.

- The shared FC1 tensor is physically `[up/w3, gate/w1]`. B12X names that
  layout `w13`/`up_gate`; `w31` would mean `[gate, up]` and is intentionally
  rejected by the harness contract.

- **W4A4:** pinned FlashInfer `B12xMoEWrapper` with
  `quant_mode="nvfp4"`. BF16 hidden states are quantized to FP4 inside the
  fused routed-MoE kernel.
- **W4A16:** pinned B12X native-ModelOpt W4A16 kernel. It keeps activations in
  BF16 and dequantizes the same FP4 weights inline.

The comparator intentionally uses B12X's direct W4A16 entry point. The pinned
FlashInfer W4A16 wrapper exposes only ordinary SiLU/ReLU2, while DeepSeek V4
uses the clamp below. B12X W4A16 supports it, giving an activation-matched A/B:

```text
W4A4:  swigluoai_uninterleave(alpha=1, beta=0, limit=10)
W4A16: silu(swiglu_limit=10)
```

These are the same operation: clamp the gate above 10, clamp the up branch to
[-10, 10], then compute `gate * sigmoid(gate) * up`.

## Safe host-side preflight

This imports no CUDA packages and reads only `config.json` plus the checkpoint
index:

```bash
python3 benchmarks/benchmark_nvfp4_a4w4_sm121.py \
  --dry-run \
  --model-path /path/to/DeepSeek-V4-Flash-NVFP4 \
  --m 1,2,4,6,12,24,48,72,128,256,512,1024,2048,4096,8192 \
  --correctness-m 1,24,128,2048
```

Preflight rejects a checkpoint that does not declare DeepSeek V4, FP4 routed
experts, NVFP4 MoE quantization, and group size 16. It also checks representative
first/last expert keys and records SHA-256 hashes of the config and index.

## GPU smoke test

Run only inside the candidate image containing the pinned CUDA 13, vLLM,
FlashInfer, and B12X revisions. A small synthetic test verifies compilation and
graph capture without reading model shards:

```bash
python3 benchmarks/benchmark_nvfp4_a4w4_sm121.py \
  --synthetic --synthetic-experts 8 \
  --m 1,8,128 \
  --correctness-m 1,8,128 \
  --warmup 2 --iters 5 --repeats 2 \
  --output /results/nvfp4-a4w4-smoke.json
```

The synthetic mode preserves K=4,096, I/rank=1,024, and top-k=6. Reducing E is
only for a fast compile/API smoke test; it is not a performance result.

## Full checkpoint run

The default matrix covers all three W4A4 tactics and representative server
prefill chunks:

```bash
python3 benchmarks/benchmark_nvfp4_a4w4_sm121.py \
  --model-path /models/DeepSeek-V4-Flash-NVFP4 \
  --layer-idx 0 --tp-size 2 --tp-rank 0 \
  --routing balanced \
  --output /results/nvfp4-a4w4-layer0-rank0.json
```

For a disposable candidate image whose normal entrypoint is `vllm serve`, a
typical isolated launch is:

```bash
sudo docker run --rm --gpus all --ipc=host \
  --entrypoint python3 \
  -v /path/to/DeepSeek-V4-Flash-NVFP4:/models/DeepSeek-V4-Flash-NVFP4:ro \
  -v "$PWD":/workspace:ro \
  -v "$PWD/.local/results":/results \
  CANDIDATE_IMAGE \
  /workspace/benchmarks/benchmark_nvfp4_a4w4_sm121.py \
  --model-path /models/DeepSeek-V4-Flash-NVFP4 \
  --output /results/nvfp4-a4w4-layer0-rank0.json
```

This allocates roughly one TP rank's expert weights plus both backends'
workspaces. It must run only in an exclusive GX10 test window with the serving
container stopped; it is not a client-side benchmark.

## Tactics and M values

With top-k=6, the pinned W4A4 dispatcher selects:

| M | Routed rows | Expected W4A4 tactic | Phase |
|---:|---:|---|---|
| 1, 2, 4, 6 | 6, 12, 24, 36 | micro | decode |
| 12, 24, 48, 72 | 72, 144, 288, 432 | static | decode |
| 128 and above | 768+ | dynamic | prefill |

The JSON records the selector's returned path as backend proof. W4A16 reports
its fused native path; its internal small-M/direct choice remains an internal
kernel decision.

For the exact TP=2, E=256, K=4,096, I/rank=1,024, top-k=6, Mmax=8,192 contract, the
harness also walks the W4A4 wrapper's static workspace, dynamic workspace, and
output tensor, deduplicates views by underlying storage, and fails if their
combined unique storage exceeds 635,144,040 bytes. This is the one-arena
ceiling; the 43-layer sharing regression is separately enforced by the overlay
unit test and must still be confirmed by full-model allocator telemetry.

Use `--routing random` after the balanced baseline to sample realistic route
imbalance, and `--routing hot` only as a skew stress case. Keep route mode and
seed identical for comparisons. `--l2-flush-mib N` touches a buffer before
each launch while keeping the flush outside the CUDA-event interval; the
default is zero because one real expert layer's packed weights already far
exceed L2.

## Reading results

For each M and mode, the report includes:

- eager and graph-replay median, p95, min, max, and per-repeat medians;
- model tokens/s, routed rows/s, and effective local-rank TFLOP/s;
- the selected tactic and normalized quant mode;
- graph-vs-eager numerical metrics where correctness was requested;
- W4A4-vs-W4A16 max/mean absolute error, RMSE, normalized RMSE, cosine, and
  relative-error percentiles;
- `speedup_w4a4_over_w4a16`, calculated as W4A16 median divided by W4A4
  median, so values above 1 mean W4A4 is faster.

The default differential gate is provisional: cosine at least 0.98 and
normalized RMSE at most 0.25. It catches broken layouts, scales, and activation
contracts; it is not a model-quality acceptance threshold. Do not loosen it
until a known-good hardware run is archived. End-to-end logits, prompts,
streaming, tool calls, and long-context quality remain mandatory integration
gates.

Prefill is the main W4A4 hypothesis. Compare M=128 through 8,192, paying
particular attention to graph replay and p95. A speedup confined to tiny M is
not evidence of a prefill win. Conversely, a small or absent gain can indicate
that route packing, activation quantization, or memory traffic dominates the
FP4 tensor-core advantage.

## Provenance and limitations

Every report prints and stores:

- expected commits and package versions from `upstream.lock`;
- actual package versions and module paths;
- GPU name, CUDA version, and compute capability;
- callable names/signatures for both measured kernels;
- sampled SHA-256 fingerprints and data pointers proving both modes received
  the same source weight tensors;
- checkpoint config/index hashes, layer, and TP slice.

This is intentionally not an end-to-end vLLM result. It excludes the router,
shared expert, attention/indexer, KV cache, scheduler, TP all-reduce, HTTP, and
RoCE weight loading. Repeat rank 1 after rank 0; only matched rank results may
be used to predict TP=2 behavior. Full API prefill and decode A/B tests remain
the promotion evidence.
