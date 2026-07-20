# Agentic-only confidence threshold 0.90

## Result

Confidence threshold `0.90` produced real physical verifier compaction but was
substantially slower than both confidence OFF and threshold `0.40` on the
`tool_agentic` workload.

All arms used the same immutable image/revision, MTP=5, probabilistic draft
sampling, temperature 0, a 512-token output limit, and the same 40-token prompt
(SHA-256
`6173a7ae0ea3c64b364d0c405be28808efb8486c68a7011e966d31ce222c1736`).
Only the confidence threshold changed.

| C | Confidence OFF tok/s | Threshold 0.40 tok/s | Threshold 0.90 tok/s | 0.90 vs OFF | 0.90 vs 0.40 |
|---:|---:|---:|---:|---:|---:|
| 1 | 75.829 | 75.198 | 65.069 | -14.19% | -13.47% |
| 4 | 184.755 | 168.135 | 104.594 | -43.39% | -37.79% |
| 8 | 293.772 | 178.447 | 164.874 | -43.88% | -7.61% |

| C | Tau OFF | Tau 0.40 | Tau 0.90 | p_full 0.90 | Draft blocks OFF | Draft blocks 0.90 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.150 | 4.961 | 3.772 | 0.309 | 100 | 136 |
| 4 | 5.221 | 5.042 | 3.667 | 0.278 | 393 | 558 |
| 8 | 5.099 | 5.102 | 3.734 | 0.292 | 805 | 1,096 |

Threshold `0.90` reduced the settled physical target width to **3.821 rows**
out of 6. Of 1,803 physical observations after subtracting the excluded
warm-up, 1,253 (69.50%) were narrower than six rows. Exact physical widths
were: 212 at one row, 348 at two, 274 at three, 236 at four, 183 at five, and
550 at six.

That compaction did not translate into speed. The benchmark required 1,790
draft blocks across C=1/4/8 versus 1,298 with confidence OFF, an increase of
37.9%. The shorter proposals therefore created many more sequential
draft/verify cycles. Small verification shapes plus confidence/compaction
overhead were more expensive than keeping the high-quality agentic drafts.
The generated-output quality gate passed for every measured case.

## Telemetry caveat

After warm-up subtraction, the HTTP scrape contained 1,814 logical-prefix and
1,803 physical-row observations. Physical statistics are reported from the
1,803 complete physical observations rather than falsely joining the two
streams one-for-one. This 11-observation skew is small enough that it cannot
change the performance conclusion.

## Provenance

- Code revision: `0175f8c0189b4d266ac22c9cbf331c14b27f3679`
- Image: `sha256:a883e1208a45afab026ecdde9bddea34445a942a99cf8840ed21183ffcd41752`
- Confidence-OFF source: `dspark-overlap-mode-a-0175f8c.json`
- Threshold-0.40 source: `dspark-agentic-confidence-040-0175f8c.json`
- Threshold-0.90 raw JSON: `dspark-agentic-confidence-090-0175f8c.json`,
  SHA-256
  `4aaf78b6e7e5705a3417fa49d8394a0806bd5abc2ecd10370b6845e6dd6c3bfe`
- Settled post-run metrics SHA-256:
  `1cb60464f29bcbc5fab684dd670332dc88d3db3469e5fa9b0fc6a085b71ba641`

Production was restored worker-first/head-second on the pinned image
`sha256:3430d6614a8e2925f34d059af6caf05aff42387326db4d05639a60f10f2654d8`.
Both ranks were running with `OOMKilled=false`; HEAD health, model list, and
version endpoints returned HTTP 200, and the deterministic smoke returned
exactly `OK` before watchdog disarm and lock release.
