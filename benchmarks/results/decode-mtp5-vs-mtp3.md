# DeepSeek V4 Flash DSpark: MTP=5 versus MTP=3

Date: 2026-07-19

Runtime: `0.25.2.dev0+g752a3a504.d20260714`

Image: `sha256:3430d6614a8e2925f34d059af6caf05aff42387326db4d05639a60f10f2654d8`

Model: `deepseek-v4-flash-dspark-abliterated`

## Controlled variable

The same two-node production image, checkpoint, TP=2 topology, probabilistic draft
sampling, confidence-scheduler state (`off`), and confidence threshold (`0.0`) were
used for both sides. The only serving change was `num_speculative_tokens`: five
versus three. Each side was loaded into a fresh service process. The worker was
started before the head, and exact MTP=5 production was restored after the test.

The installed DSpark implementation does **not** use the learned confidence head.
Its loader explicitly drops `confidence_head.*` weights with the source comment
"The confidence head is not wired into inference yet; drop its weights."

## Canonical chat/tool prompt, 512 output tokens

Values are medians of three trials. Aggregate throughput measures all concurrent
streams; per-stream throughput is the median of each trial's mean stream decode
rate. Speculative acceptance is captured from the server's Prometheus counters.

| Concurrency | MTP=5 aggregate tok/s | MTP=3 aggregate tok/s | MTP=3 delta | MTP=5 per-stream tok/s | MTP=3 per-stream tok/s | MTP=5 mean accepted length | MTP=3 mean accepted length |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 49.03 | 48.60 | -0.9% | 50.14 | 49.64 | 3.140 | 2.834 |
| 2 | 73.89 | 77.46 | +4.8% | 37.90 | 40.28 | 3.122 | 2.811 |
| 4 | 105.48 | 107.65 | +2.1% | 28.66 | 28.55 | 3.173 | 2.730 |

For this hard tool-use prompt, MTP=3 and MTP=5 are effectively tied. MTP=3 saves
work at C=2/4, while MTP=5 accepts slightly longer runs. The differences are much
smaller than run-to-run prompt and scheduling effects.

## Exact-token context sweep, 512 output tokens

The context workload uses deterministic token-ID prompts at the same depths as the
earlier repository context/decode matrix. Values are medians of three trials. The
decode rate uses the public MiaAI-Lab convention:
`(completion_tokens - 1) / (last_token_time - first_token_time)`.

| Context | MTP=5 TTFT | MTP=3 TTFT | MTP=5 decode tok/s | MTP=3 decode tok/s | MTP=3 delta | MTP=5 mean accepted length | MTP=3 mean accepted length |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K | 0.570 s | 0.557 s | 96.61 | 72.08 | -25.4% | 5.818 | 3.908 |
| 2K | 0.989 s | 1.010 s | 92.95 | 72.54 | -22.0% | 5.775 | 3.916 |
| 4K | 1.787 s | 1.804 s | 90.46 | 71.83 | -20.6% | 5.648 | 3.793 |
| 8K | 3.733 s | 3.778 s | 93.38 | 71.81 | -23.1% | 5.400 | 3.757 |
| 16K | 7.471 s | 7.613 s | 87.53 | 67.78 | -22.6% | 5.495 | 3.821 |
| 32K | 14.935 s | 15.064 s | 91.35 | 66.13 | -27.6% | 5.400 | 3.698 |

MTP length has almost no effect on TTFT. On predictable continuation text, however,
all five positions are accepted often enough that MTP=5 is 20.6-27.6% faster in
decode. Its median mean accepted length remains 5.4-5.8 tokens, versus 3.7-3.9 for
MTP=3.

## Interpretation versus MiaAI-Lab

MiaAI-Lab reports 66.6 tok/s at C=1 using the same released image but MTP=3, a
different agent/file-writing prompt set, three-trial medians, and a post-first-token
decode window. We measure approximately the same 66-73 tok/s range with MTP=3 on
predictable exact-token continuation prompts, but only 49.64 tok/s on the canonical
tool-use prompt. This demonstrates that the higher published number is attainable
without a changed image and is dominated by prompt acceptance and measurement
scope. No published evidence shows an image or kernel improvement in that repo.

The production default should remain MTP=5: it is neutral on the difficult tool
prompt and materially faster when draft acceptance is high. A single decode number
without its prompt, acceptance counters, concurrency, and timing window is not a
portable performance claim.

## Raw evidence

- `decode-v0251-production-mtp5-matched.json`
- `decode-v0251-production-mtp3.json`
- `decode-context-v0251-production-mtp5-matched.json`
- `decode-context-v0251-production-mtp3.json`
- `decode-context-v0251-production-mtp5.json` (initial two-trial exploratory control)
