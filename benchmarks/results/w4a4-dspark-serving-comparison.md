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

This is the same 35-token canonical chat/tool prompt at temperature zero and
512 requested output tokens. Values are the best aggregate result from each
report, matching the repository release-table convention.

| Concurrency | FP8/B12X + DSpark | NVFP4 W4A4 + DSpark | W4A4 delta |
|---:|---:|---:|---:|
| 1 | **48.49 tok/s** | 47.13 tok/s | -2.8% |
| 4 | **103.48 tok/s** | 99.44 tok/s | -3.9% |

This matched prompt removes the earlier mismatched 105.48-versus-96.02 row.
The remaining difference is small enough that it should be treated as a
prompt-specific modest regression rather than a universal decode result. Raw
sources: [`v0251-candidate.json`](v0251-candidate.json),
[`decode-w4a4-canonical-c1.json`](decode-w4a4-canonical-c1.json), and
[`decode-w4a4-canonical-c4.json`](decode-w4a4-canonical-c4.json).

### Target-backend retry

A later same-prompt retry compared the prepared W4A4 target under both CUTLASS
and B12X while keeping the DSpark draft on its native `DEEPGEMM_MXFP4`
backend. Values are best aggregate throughput with the median in parentheses;
C=4 uses a separate warmed three-trial recheck:

| C | FP8/B12X + DSpark | W4A4/CUTLASS + DSpark | W4A4/B12X + DSpark |
|---:|---:|---:|---:|
| 1 | **47.94 (47.80)** | 47.67 (46.95) | **49.12 (47.53)** |
| 4 | **103.98 (101.07)** | 101.12 (98.08) | 91.97 (90.40) |

B12X's C1 best trial coincided with higher acceptance. At C4, comparable
acceptance did not prevent a 9.1% best-throughput regression versus CUTLASS.
The split-backend proof, raw JSON, and hashes are archived in
[`w4a4-decode-port-20260722/dspark-target-backend-ab`](w4a4-decode-port-20260722/dspark-target-backend-ab/README.md).

## Agentic decode

This is the W4A4 path on the exact `tool_agentic` prompt at temperature zero
and 512 tokens per stream. Confidence is off and no draft/verify overlap
optimization is enabled. The prompt SHA-256 is
`6173a7ae0ea3c64b364d0c405be28808efb8486c68a7011e966d31ce222c1736`.
Each arm used one short shape warm-up followed by two measured trials per
concurrency. Values are best aggregate throughput.

| MTP draft tokens | C=1 | C=2 | C=4 | C=8 |
|---:|---:|---:|---:|---:|
| 1 | 39.7 | 66.5 | 96.0 | 146.7 |
| 2 | 53.2 | 88.2 | 119.8 | 175.3 |
| 3 | 62.5 | 94.6 | 148.3 | 224.7 |
| 4 | 69.1 | **135.6** | **157.9** | 234.8 |
| 5 | **76.4** | 111.9 | 156.6 | **244.2** |

MTP=5 wins C=1 and C=8 and remains the recommended default. MTP=4 wins C=2;
at C=4 it wins the best trial by 0.8%, while MTP=5 has the higher two-trial
mean. The exact methodology, accepted-length table, MTP=1 context caveat, and
raw JSON hashes are in
[`decode-w4a4-agentic-mtp-grid.md`](decode-w4a4-agentic-mtp-grid.md).

## Prepared-load result

The bulk direct reader loaded the prepared target on the slower rank in 65.23
seconds and completed the full head model load in 108.54 seconds. The earlier
558.19/595.90-second measurements were an intermediate non-direct prototype,
not the release model, and are retained only as prototype history. The full
TP=2 startup proved 43 layers, 344 reads, 344 copies, zero transforms,
`io_mode=preadv`, HTTP health 200, and a coherent smoke response. See
[`nvfp4-prepared-direct-read-full-3689b1c.json`](nvfp4-prepared-direct-read-full-3689b1c.json).

## Interpretation

W4A4 consistently improves prefill in this dataset. Decode remains prompt- and
acceptance-dependent: the latest CUTLASS retry is within 0.6--2.8% of FP8 by
best throughput, while forcing B12X regresses C4 materially. The agentic prompt
scales to 244.2 aggregate tok/s at C=8 with MTP=5. Report prompt identity,
concurrency, target backend, acceptance, token limit, confidence state, overlap
state, and timing convention with every decode number.
