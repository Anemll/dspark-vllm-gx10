# NVFP4 W4A4 prefill comparison archive

## Goal

Measure whether the prepared NVIDIA NVFP4 W4A4 target improves prefill on
the two-node GX10 TP=2 deployment relative to the current v0.25.1 production
result.

This objective intentionally isolates the base target: DSpark/MTP/speculation
was disabled and RoCE remained paused. It is not a speculative-decode result.

## Method

The candidate image was built from
`7b877eaae2a8e2b5800e84b585d7f14fb90f5294` and has immutable ID
`sha256:222c3295b804664f19442a953143fef45a7fdc3ed278ae5e82eab546f7519b99`.
The prepared checkpoint and load path are archived separately in
`docs/NVFP4_PREPARED_LOAD_ARCHIVE.md`.

The canonical prefill harness ran concurrency 1, two trials per size, seed
4104, one output token, exact prompt-length and server-token checks, and zero
prefix-cache hits at 1K, 2K, 4K, 8K, 16K, and 32K input tokens.

The comparison baseline is `prefill-v0251-candidate.json`, which used three
trials and seed 4106. Prompt fingerprints therefore differ. The report is an
explicitly caveated same-size aggregate comparison, not a paired-prompt A/B.

## Result

NVFP4 W4A4 improved server prefill throughput at every tested size and
reduced median time to first token at every size.

| Input tokens | v0.25.1 production tok/s | NVFP4 W4A4 tok/s | Gain | Production TTFT | NVFP4 TTFT |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 2,033.0 | 2,242.5 | **+10.3%** | 0.512 s | 0.463 s |
| 2,048 | 2,252.0 | 2,473.2 | **+9.8%** | 0.920 s | 0.835 s |
| 4,096 | 2,320.7 | 2,659.3 | **+14.6%** | 1.776 s | 1.552 s |
| 8,192 | 2,184.2 | 2,593.5 | **+18.7%** | 3.765 s | 3.173 s |
| 16,384 | 2,203.8 | 2,501.7 | **+13.5%** | 7.455 s | 6.573 s |
| 32,768 | 2,176.1 | 2,477.3 | **+13.8%** | 15.119 s | 13.264 s |

The gain range is **+9.8% to +18.7%** across all six sizes.

## Reproduction and evidence

- Candidate raw JSON:
  `benchmarks/results/prefill-v0251-nvfp4-a4w4-candidate.json`
- Existing production baseline:
  `benchmarks/results/prefill-v0251-candidate.json`
- Rendered comparison:
  `benchmarks/results/prefill-v0251-vs-nvfp4-a4w4.md`
- Immutable evidence directory:
  `benchmarks/results/evidence/nvfp4-prefill-20260718t193724-7b877ea/`
- Both-rank logs and exact container inspections:
  `head/candidate-head.{log,inspect.json}` and
  `worker/candidate-worker.{log,inspect.json}`
- First assembled-model generation:
  `sample-generation.json`, exact content `NVIDIA ready`
- Integrity list: `MANIFEST.sha256` in the evidence directory.

The comparison table is generated with:

```bash
python3 benchmarks/compare_prefill.py \
  benchmarks/results/prefill-v0251-candidate.json \
  benchmarks/results/prefill-v0251-nvfp4-a4w4-candidate.json \
  --allow-unmatched-prompts \
  --output benchmarks/results/prefill-v0251-vs-nvfp4-a4w4.md
```

The final three-line candidate-scope note in the committed Markdown is
editorial metadata; it does not alter the generated table or either raw JSON.
