# Prepared W4A4 + DSpark benchmark summary

## Scope

These measurements cover the prepared NVIDIA NVFP4 W4A4 target on the same
two-node GB10 TP=2 cluster as the preceding v0.25.1 FP8/B12X production path.
The W4A4 target uses FlashInfer CUTLASS, the native three-stage DSpark draft
uses backend auto-selection, MTP=5, confidence is off, draft sampling is
probabilistic, and no draft/verify overlap optimization is enabled.

The three tables answer different questions and must not be collapsed into one
portable "tokens/s" number:

- prefill is target-only and independent of speculative decoding;
- canonical decode uses a difficult 35-token chat/tool prompt;
- agentic decode uses the exact 40-token `tool_agentic` prompt with SHA-256
  `6173a7ae0ea3c64b364d0c405be28808efb8486c68a7011e966d31ce222c1736`.

## Prefill

Both sides use the same vLLM runtime and two GX10 nodes. Results are same-size
aggregates rather than paired identical-prompt trials; the raw reports retain
their own trial counts and prompt fingerprints.

| Input tokens | FP8/B12X production tok/s | NVFP4 W4A4 tok/s | Gain | Production TTFT | W4A4 TTFT |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 2,033.0 | 2,242.5 | +10.3% | 0.512 s | 0.463 s |
| 2,048 | 2,252.0 | 2,473.2 | +9.8% | 0.920 s | 0.835 s |
| 4,096 | 2,320.7 | 2,659.3 | +14.6% | 1.776 s | 1.552 s |
| 8,192 | 2,184.2 | 2,593.5 | +18.7% | 3.765 s | 3.173 s |
| 16,384 | 2,203.8 | 2,501.7 | +13.5% | 7.455 s | 6.573 s |
| 32,768 | 2,176.1 | 2,477.3 | +13.8% | 15.119 s | 13.264 s |

Source: [`prefill-v0251-vs-nvfp4-a4w4.md`](prefill-v0251-vs-nvfp4-a4w4.md).

## Canonical decode

This is the 35-token canonical chat/tool prompt at concurrency 4, temperature
zero, and 512 requested output tokens. The production row is the median of
three trials; the W4A4 row is one post-promotion production run, so this is an
exploratory comparison rather than a variance-bounded A/B.

| Target path | Trials | Aggregate tok/s | Mean stream tok/s | TTFT | Mean accepted length |
|---|---:|---:|---:|---:|---:|
| FP8/B12X + DSpark | 3 | **105.48** | **28.66** | 0.383 s | 3.182 |
| NVFP4 W4A4 + DSpark | 1 | 96.02 | 25.34 | 0.384 s | 3.127 |
| W4A4 delta |  | **-9.0%** | **-11.6%** | +0.4% | -1.7% |

The harder prompt's modest draft acceptance does not hide verifier cost; W4A4
is slower in this single-run comparison. Raw sources:
[`decode-v0251-production-mtp5-matched.json`](decode-v0251-production-mtp5-matched.json)
and
[`decode-w4a4-dspark-production-c4.json`](decode-w4a4-dspark-production-c4.json).

## Agentic decode

This is the clean W4A4 production path on the exact `tool_agentic` prompt at
concurrency 8, temperature zero, and 512 tokens per stream. Confidence is off
and no draft/verify overlap optimization is enabled.

| Aggregate decode tok/s | Mean stream tok/s | Mean TTFT | Mean accepted length | Aggregate acceptance | Full-draft acceptance |
|---:|---:|---:|---:|---:|---:|
| **360.68** | **46.64** | 0.439 s | 5.270 | 85.39% | 74.33% |

All eight W4A4 streams passed the automated no-collapse diagnostic. Higher
agentic throughput is associated with strong draft acceptance. A clean,
uninstrumented legacy agentic control using this exact prompt was not archived,
so no cross-target speedup is claimed from this row. Raw source:
[`decode-w4a4-dspark-agentic-c8.json`](decode-w4a4-dspark-agentic-c8.json).

## Prepared-load result

The bulk direct reader reduced the slower rank's prepared target load from
558.19 to 65.23 seconds (8.56x) and complete head model load from 595.90 to
108.54 seconds (5.49x). The full TP=2 startup proved 43 layers, 344 reads, 344
copies, zero transforms, `io_mode=preadv`, HTTP health 200, and a coherent
smoke response. See
[`nvfp4-prepared-direct-read-full-3689b1c.json`](nvfp4-prepared-direct-read-full-3689b1c.json).

## Interpretation

W4A4 consistently improves prefill in this dataset. Decode remains prompt- and
acceptance-dependent: it lost about 9% in the exploratory canonical comparison,
while the high-acceptance agentic path reached 360.68 aggregate tok/s without a
clean legacy control. Report prompt identity, concurrency, acceptance, token
limit, confidence state, overlap state, and timing convention with every decode
number.
