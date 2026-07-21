# Target-only decode: FP8/B12X vs NVFP4 W4A4

This comparison isolates the target model from DSpark. Speculation was disabled
for both arms and `benchmark_dsv4_api.py` required the speculative counters to
remain inactive. Both arms used TP=2, the canonical prompt, temperature 0,
512 output tokens, concurrency 1 and 4, and three trials.

Canonical prompt SHA-256:
`652af3aabacfd4360432d28e0c237e9e445f938d032a604d3a4f7f42a2a7ed38`.

| Concurrency | FP8/B12X best | W4A4 best | W4A4 delta | FP8/B12X median | W4A4 median | W4A4 median delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | **27.57 tok/s** | 27.03 tok/s | -2.0% | **27.45 tok/s** | 26.92 tok/s | -1.9% |
| 4 | **78.74 tok/s** | 73.37 tok/s | -6.8% | **77.21 tok/s** | 72.55 tok/s | -6.0% |

Raw trials:

- FP8/B12X C1: 27.28, 27.45, 27.57 tok/s.
- W4A4 C1: 26.77, 26.92, 27.03 tok/s.
- FP8/B12X C4: 65.03, 78.74, 77.21 tok/s.
- W4A4 C4: 62.64, 73.37, 72.55 tok/s.

The first C4 trial in both arms had an approximately four-second TTFT warmup
outlier, so both best and median are reported. The direction is unchanged.

## Interpretation

The C1 deficit is small, but it widens at C4. Because DSpark was physically off
and its counters were absent for every trial, this is target-kernel behavior,
not a draft-acceptance effect. The leading optimization hypothesis is small-M
routed-MoE dispatch: four simultaneous sequences scatter routed rows over 256
experts, leaving many tiny expert groups where the current FlashInfer CUTLASS
NVFP4 schedule scales less efficiently than the FP8/B12X path. This must be
confirmed with per-expert row histograms and kernel-time attribution before
changing backend selection.

## Evidence

- [`decode-fp8-b12x-target-only-c1-c4.json`](decode-fp8-b12x-target-only-c1-c4.json), SHA-256 `354fda11a996d535e066fed20ed3cfd635a348e1603a9563d5678c9d441b245d`
- [`decode-w4a4-target-only-c1-c4.json`](decode-w4a4-target-only-c1-c4.json), SHA-256 `25355abdc91d9e51b6523c673234001ef607d9547ce71256a6aaac0765ffd631`

The legacy FP8 image requires its released compatibility setting
`FLASHINFER_DISABLE_VERSION_CHECK=1` because that immutable image contains
FlashInfer Python 0.6.15 with 0.6.13 cubin/cache packages. The W4A4 image uses a
coherent 0.6.15 package set and leaves the bypass disabled.
