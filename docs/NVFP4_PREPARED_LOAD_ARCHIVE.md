# Prepared NVFP4 W4A4 load path archive

## Goal

Make the NVIDIA DeepSeek V4 Flash NVFP4 checkpoint practical on two GX10
nodes by paying CUTLASS scale preparation once, offline, and loading a
rank-contiguous prepared layout without outage-time tensor transforms.

## Implementation

- `80ab0f2274b60d324930d036ac9929ce255a2b28` added the resumable offline
  TP=2 repacker and physical-layout verifier.
- `7b877eaae2a8e2b5800e84b585d7f14fb90f5294` completed the prepared loader
  contract and produced the immutable candidate image
  `sha256:222c3295b804664f19442a953143fef45a7fdc3ed278ae5e82eab546f7519b99`.

The prepared checkpoint contains 101 physical files and 168,281,460,149
bytes. Its 87 payload files hold 168,266,881,608 bytes and 3,483 tensors.
The manifest SHA-256 is
`972ba797456da80e586324a5a8c29af42bac86510ceff983e674de41d31e6f26`.
Layer 0 independently crossed the serialization boundary with all eight
family fingerprints matching and contiguous rank slices.

## TP=2 result

The successful run was `prepared-prefill-20260718t1921`, serving
`deepseek-v4-flash-nvfp4-prepared` with DSpark/MTP disabled.

| Evidence | HEAD / rank 0 | WORKER / rank 1 |
|---|---:|---:|
| Routed layers | 43 | 43 |
| Reads and copies per layer | 8 / 8 | 8 / 8 |
| Total reads and copies | 344 / 344 | 344 / 344 |
| Zero-transform post-load rows | 43 | 43 |
| Prepared backend | `FLASHINFER_CUTLASS` | `FLASHINFER_CUTLASS` |
| Prepared-loader elapsed | **514.633559 s** | 73.378866 s |
| Sum of per-layer timings | 475.234259 s | 53.463941 s |
| Complete model load | 518.376028 s | 77.280961 s |
| Model memory | 78.11 GiB | 78.11 GiB |

The operational load time is the slower rank: about **514.6 seconds** for
the prepared loader and 518.4 seconds for complete model loading. The large
rank asymmetry is preserved as measured evidence; this archive does not
reinterpret it as a storage or kernel claim.

The model reached serving readiness and returned the deterministic response
`NVIDIA ready`. The same session then completed the archived prefill suite.

## Reproduction and evidence

- Machine-readable summary:
  `benchmarks/results/nvfp4-prepared-load-7b877ea.json`
- Immutable evidence directory:
  `benchmarks/results/evidence/nvfp4-prepared-load-20260718t1921-7b877ea/`
- Both-rank banked logs:
  `head/banked/candidate-head.log` and
  `worker/banked/candidate-worker.log`
- Container identity snapshots:
  `head/banked/candidate-head-inspect.json` and
  `worker/banked/candidate-worker-inspect.json`
- Repacker boundary proofs:
  `layer0-physical-probe.json`, `repack-final-verify.json`, and
  `deployment-manifest.json`
- Integrity list: `MANIFEST.sha256` in the evidence directory.

The raw logs, final candidate logs, production pre-stop evidence, rollback
evidence, and exact container inspections were copied unchanged from the two
nodes. No checkpoint payload is stored in git.
