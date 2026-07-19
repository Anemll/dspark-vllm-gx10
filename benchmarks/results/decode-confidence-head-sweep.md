# DSpark confidence-head C1 sweep

Date: 2026-07-19  
Candidate source: `7f64c326c9566afb9bb345a55680195cabd4c895`  
Candidate image: `sha256:c78bd4cbbbe554bb18b348c8b23839da792fcab23eec2b043ed96ce64fc4daac`

## Scope

This is a matched two-node DGX Spark TP=2 decode test of the current
abliterated production model. The target remains on the production B12X path;
this is **not** the W4A4 prepared target. Every arm used the canonical API
benchmark with concurrency 1, 512 generated tokens, temperature 0, and two
trials. Speculative arms used five configured draft positions and required all
four Prometheus speculative-counter families. The no-draft arm required those
counters to remain absent or unchanged.

## Corrected validity result

**The confidence scheduler was never validly exercised by this sweep.** Every
enabled arm retained all five logical draft positions, so thresholds 0.30-0.50
measured confidence-scoring and synchronization overhead on the fixed-five
path, not the value of confidence-controlled early stopping. The result is
inconclusive about confidence scheduling and must not be presented as evidence
that confidence does not help.

The threshold domain was re-audited against DeepSpec after the run. DeepSpec
compares `confidence_logits.sigmoid()` with the configured threshold, exactly
matching this candidate's probability-domain policy. Therefore this was **not**
a raw-logit-versus-sigmoid scale error. The remaining logical-truncation
ambiguity is calibration (real probabilities all at least 0.50 for this prompt)
versus a live wiring/execution problem; the sweep did not persist real scores,
so it cannot distinguish them.

Authoritative reference:
[`deepspec/eval/dspark/draft_ops.py`](https://github.com/deepseek-ai/DeepSpec/blob/main/deepspec/eval/dspark/draft_ops.py).

Only two performance claims survive the validity correction:

1. Confidence-OFF DSpark delivered about **1.74x** the matched no-draft rate
   (1.75x by median per-stream decode rate; 1.73x by aggregate request rate).
2. Enabling confidence scoring adds real overhead when it removes no draft
   positions.

| Arm | Median output tok/s | Best aggregate tok/s | Mean TTFT | Mean proposed | Acceptance | Effective accepted | Delta vs OFF |
|---|---:|---:|---:|---:|---:|---:|---:|
| No draft | 27.86 | 27.65 | 0.178 s | 0 | n/a | n/a | -42.9% |
| Confidence OFF | **48.77** | **47.95** | 0.225 s | 5.000 | 41.57% | 3.078 | baseline |
| Threshold 0.30 | 46.98 | 46.24 | 0.227 s | 5.000 | 41.31% | 3.065 | -3.7% |
| Threshold 0.40 | 45.07 | 44.25 | 0.230 s | 5.000 | 39.54% | 2.977 | -7.6% |
| Threshold 0.50 | 42.25 | 42.60 | 0.233 s | 5.000 | 36.13% | 2.806 | -13.4% |

Mean per-position acceptance rates remained monotonic within every arm:

| Arm | p0 | p1 | p2 | p3 | p4 |
|---|---:|---:|---:|---:|---:|
| Confidence OFF | 0.777 | 0.524 | 0.380 | 0.241 | 0.157 |
| Threshold 0.30 | 0.756 | 0.571 | 0.363 | 0.226 | 0.149 |
| Threshold 0.40 | 0.803 | 0.548 | 0.333 | 0.188 | 0.104 |
| Threshold 0.50 | 0.763 | 0.488 | 0.274 | 0.186 | 0.096 |

The real requests reported a mean proposed length of exactly 5.000 at every
tested threshold. Thus 0.30-0.50 did not produce a logical proposal-length
reduction, while scheduler overhead and/or the altered proposal path reduced
throughput. The threshold promotion gate is invalid because its prerequisite
-- observed truncation -- never occurred.

There is a second, independent execution confound. With async scheduling, a
short confidence prefix is transferred to the scheduler and then padded back
to the configured five slots with `-1`; the current runner derives target work
from that padded list length. Statistics subtract the invalid slots, so a
reported logical proposal length can be shorter than the physical verifier
shape. Consequently, even a future arm that reports truncation cannot be
called a valid confidence performance test until physical target work is also
proven to shrink. The current CUDA-graph/rows-5-optimized path is a fixed-five
kernel path, not a demonstrated variable-length verifier.

The exact-output-hash gate also did not pass: the no-draft, 0.40, and 0.50
arms were not even internally hash-stable across their two temperature-zero
trials, and no confidence arm matched the no-draft hashes. This makes the
cross-arm quality comparison inconclusive and rules out exact hashes as the
promotion contract for this runtime. There is no confidence speed decision to
make from these arms because no early stopping occurred.

## Required next evidence

No threshold or content-matrix sweep should be run from this result. A later
candidate must first expose the real per-position sigmoid-probability
distribution, below-threshold counts, and logical prefix-length histogram. It
must then prove that a forced logical `5 -> 2` truncation also reduces the
physical target verifier from five draft slots to two. Only after both gates
pass is a threshold sweep meaningful.

The next candidate now carries passive, confidence-enabled-only Prometheus
telemetry for that first gate:

- `vllm:dspark_confidence_probability` histogram, labeled by position and
  threshold;
- `vllm:dspark_confidence_below_threshold_total`, labeled identically;
- `vllm:dspark_confidence_prefix_length` histogram;
- `vllm:dspark_confidence_telemetry_dropped_batches_total`, which makes an
  incomplete asynchronous sample visible rather than silently biasing it.

## Reproducibility notes

The confidence candidate inherits production's FlashInfer package combination
(`flashinfer-python` 0.6.15 with `flashinfer-cubin` 0.6.13). Its first start
exposed a candidate Compose drift that cleared the production-required version
check bypass. The measured arms used the production-parity setting
`FLASHINFER_DISABLE_VERSION_CHECK=1`; the temporary Compose edit was restored
after the sweep.

Production was restored on the exact pinned image
`sha256:3430d6614a8e2925f34d059af6caf05aff42387326db4d05639a60f10f2654d8`.
Final state was WORKER CID `ff133f4271a2`, HEAD CID `10a8d25d64a1`, both
running and not OOM-killed; HEAD health and dashboard returned 200 and the
final chat smoke was coherent (`OK.`).

## Raw artifacts

- `decode-confidence-head-no-draft.json` — SHA-256 `98c03d419fc9ec3da2516c61878ce00660e6325f2a80429b8d38373f5f5f045d`
- `decode-confidence-head-off.json` — SHA-256 `8186ec04e1864de53bf05521dd083c52fc7e1500ec61cb90d70f6ef2ca5b492a`
- `decode-confidence-head-threshold-030.json` — SHA-256 `802279b1b0c68c8dab4c308ffdac9941f80d2e6305604e0892aaba57a5ccd9e9`
- `decode-confidence-head-threshold-040.json` — SHA-256 `cfff44142394d864e92d82e8f6acdf46c50115fe73ea281b2bfce9383516c54b`
- `decode-confidence-head-threshold-050.json` — SHA-256 `3f61f84a81a56119702fbc42a8666b660d2db350b33bfe0351962faa6f2e87ba`

Node-side logs and rendered environment evidence are preserved under
`/home/anemll/nvfp4-artifacts/20260719-7f64c32-confidence-sweep2` on both
nodes.
