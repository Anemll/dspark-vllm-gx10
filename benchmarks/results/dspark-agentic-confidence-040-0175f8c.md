# Agentic-only confidence threshold 0.40

## Result

Confidence threshold `0.40` did not improve the `tool_agentic` workload. It
was effectively neutral at concurrency 1, 9.0% slower at concurrency 4, and
39.3% slower at concurrency 8 relative to the exact confidence-OFF run from
the immediately preceding Mode-A sweep.

Both arms used the same immutable image and code revision, DSpark MTP=5,
probabilistic draft sampling, temperature 0, a 512-token output limit, and the
same 40-token prompt (SHA-256
`6173a7ae0ea3c64b364d0c405be28808efb8486c68a7011e966d31ce222c1736`).
The only intended change was confidence scheduler ON at threshold `0.40`.

| Concurrency | Confidence OFF tok/s | Threshold 0.40 tok/s | Delta | Tau OFF | Tau 0.40 | p_full OFF | p_full 0.40 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 75.829 | 75.198 | -0.83% | 5.150 | 4.961 | 0.700 | 0.641 |
| 4 | 184.755 | 168.135 | -9.00% | 5.221 | 5.042 | 0.728 | 0.653 |
| 8 | 293.772 | 178.447 | -39.26% | 5.099 | 5.102 | 0.680 | 0.685 |

The physical-row telemetry explains the result. Across 1,330 recorded target
verification blocks, mean physical width was **5.941 rows** out of 6. Only
36 blocks (2.71%) used fewer than 6 rows: 5 used one row, 6 used three, 11
used four, and 14 used five. The remaining 1,294 blocks ran the full six-row
verifier. In other words, threshold `0.40` removed only about 0.99% of target
rows while retaining confidence scoring and compaction overhead.

At C=8 the measured per-block verifier time rose from 117.57 ms OFF to
235.32 ms at threshold 0.40, while draft time rose from 12.95 to 21.82 ms.
This is contention/scheduling behavior, not a useful confidence-truncation
win. The generated-output quality gate passed for all three cases, but this is
the previously documented Mode-B runtime and output hashes are not expected
to be stable across arms.

## Telemetry caveat

The settled HTTP scrape reported 1,355 logical-prefix observations but 1,330
physical-row observations after subtracting the pre-run snapshot. Therefore
the physical-row distribution is reported directly from its 1,330 complete
observations; it is not falsely joined one-for-one to every logical-prefix
sample. D2H completion telemetry recorded 327 ready events and 4 fallback
waits.

## Provenance

- Code revision: `0175f8c0189b4d266ac22c9cbf331c14b27f3679`
- Image: `sha256:a883e1208a45afab026ecdde9bddea34445a942a99cf8840ed21183ffcd41752`
- Confidence-OFF source: `dspark-overlap-mode-a-0175f8c.json`, SHA-256
  `30d7d5dce58bf27f52a7925e987344d0b108690c530807108ff81854dfbfd7f8`
- Threshold-0.40 raw JSON: `dspark-agentic-confidence-040-0175f8c.json`,
  SHA-256
  `0d4ff526c66563659f195e8b2f25408b4fe0276e8759d90186192d94485f745d`
- Post-run settled metrics SHA-256:
  `fb0ad9e461f6aef9a717d2d52aa04bb7fc9e78eba72782a4c9679e8fe289e982`

Production was restored worker-first/head-second on the pinned image
`sha256:3430d6614a8e2925f34d059af6caf05aff42387326db4d05639a60f10f2654d8`.
Both ranks were running with `OOMKilled=false`; HEAD health, model list, and
version endpoints returned HTTP 200, and the deterministic smoke returned
exactly `OK` before watchdog disarm and lock release.
