# Prepared W4A4 + DSpark agentic MTP grid

## Scope

This is a two-node GX10/GB10 TP=2 decode sweep of the prepared NVIDIA NVFP4
W4A4 target plus the bundled three-stage DSpark draft. Every measured request
uses the exact 40-token `tool_agentic` prompt with SHA-256
`6173a7ae0ea3c64b364d0c405be28808efb8486c68a7011e966d31ce222c1736`,
temperature zero, 512 requested output tokens, confidence scheduling off, and
no draft/verify overlap optimization.

The immutable image is
`sha256:60524a195418707601903e56937e1ef6c33e317daafd86d69fce489e9bb156cb`
from source revision `d6a2e5c7a0702a3b03f29295e18263a27bffd726`. The prepared
target used the direct `preadv` reader. Each MTP arm received one short
C=1/2/4/8 shape warm-up followed by two 512-token trials per concurrency.
The primary metric is aggregate completion tokens divided by trial wall time;
per-stream timing is not used because a streamed response can occasionally
arrive in a small number of large chunks.

## Best aggregate throughput

Values are the better of two trials, matching the repository's release-table
convention.

| MTP draft tokens | C=1 | C=2 | C=4 | C=8 |
|---:|---:|---:|---:|---:|
| 1 | 39.7 tok/s | 66.5 tok/s | 96.0 tok/s | 146.7 tok/s |
| 2 | 53.2 tok/s | 88.2 tok/s | 119.8 tok/s | 175.3 tok/s |
| 3 | 62.5 tok/s | 94.6 tok/s | 148.3 tok/s | 224.7 tok/s |
| 4 | 69.1 tok/s | **135.6 tok/s** | **157.9 tok/s** | 234.8 tok/s |
| 5 | **76.4 tok/s** | 111.9 tok/s | 156.6 tok/s | **244.2 tok/s** |

## Trial means and accepted length

| MTP | C=1 mean tok/s / tau | C=2 mean tok/s / tau | C=4 mean tok/s / tau | C=8 mean tok/s / tau |
|---:|---:|---:|---:|---:|
| 1 | 38.7 / 1.964 | 57.9 / 1.967 | 88.0 / 1.962 | 137.2 / 1.961 |
| 2 | 53.1 / 2.850 | 86.2 / 2.893 | 113.2 / 2.854 | 173.5 / 2.857 |
| 3 | 62.1 / 3.687 | 93.4 / 3.671 | 145.3 / 3.726 | 220.0 / 3.705 |
| 4 | 68.9 / 4.461 | **135.4 / 4.631** | 147.0 / 4.438 | 231.1 / 4.512 |
| 5 | **70.0 / 4.996** | 103.9 / 5.198 | **155.5 / 5.040** | **236.9 / 5.028** |

`tau` is the mean physical accepted length including the target token. MTP=4
has the best C=2 result. MTP=4 and MTP=5 are effectively tied at C=4: MTP=4
wins the best-trial comparison by 0.8%, while MTP=5 has the higher two-trial
mean. MTP=5 wins C=1 and C=8 and remains the general-purpose default.

## Caveat for MTP=1

MTP=2 through MTP=5 used the same 65,536-token benchmark context. The complete
MTP=1 report was captured at the production 350,000-token context. A matched
65,536-token MTP=1 rerun reached 40.6 tok/s at C=1 and 66.8 tok/s at C=2, then
the server stalled during the second C=4 trial; that incomplete run was banked
as failure evidence rather than published as a complete report. Reducing the
context only changes the reserved KV capacity for this short prompt, and the
partial matched results remain far below MTP=4/5, so repeating MTP=1 cannot
change the selection decision.

## Raw artifacts

- [`decode-w4a4-agentic-mtp1-350k-grid.json`](decode-w4a4-agentic-mtp1-350k-grid.json) — SHA-256 `6724088258618d06c4e68c6a01f18bf4a8fb47e2a1e5348297fa578c4b9abe36`
- [`decode-w4a4-agentic-mtp2-64k-grid.json`](decode-w4a4-agentic-mtp2-64k-grid.json) — SHA-256 `c3b6b17717bf82b01a47765a9fb6299c1eeec83d7689bfcda1410a95dca302e4`
- [`decode-w4a4-agentic-mtp3-64k-grid.json`](decode-w4a4-agentic-mtp3-64k-grid.json) — SHA-256 `2fee620c5b192ace8fd7f2f0ea7250c8bc78e20229b5761e98fbc0b3d13a6880`
- [`decode-w4a4-agentic-mtp4-64k-grid.json`](decode-w4a4-agentic-mtp4-64k-grid.json) — SHA-256 `e7ffe791c556bd74a09f759009fa0e5388a279620a72425cd60d893e9905b730`
- [`decode-w4a4-agentic-mtp5-64k-grid.json`](decode-w4a4-agentic-mtp5-64k-grid.json) — SHA-256 `664ae32c6419041090e0b64710fbf4df59eb3a6ec6921f7e478f9b2a0d23c9ca`
