# Target-only decode: FP8/B12X vs NVFP4 W4A4 backends

This comparison isolates the target model from DSpark. Speculation was disabled
for both arms and `benchmark_dsv4_api.py` required the speculative counters to
remain inactive. Both arms used TP=2, the canonical prompt, temperature 0,
512 output tokens, concurrency 1 and 4, and three trials.

Canonical prompt SHA-256:
`652af3aabacfd4360432d28e0c237e9e445f938d032a604d3a4f7f42a2a7ed38`.

| C | FP8/B12X best (median) | W4A4/CUTLASS best (median) | W4A4/B12X best (median) | W4A4/B12X vs FP8 best / median | W4A4/B12X vs CUTLASS best / median |
|---:|---:|---:|---:|---:|---:|
| 1 | **27.57 (27.45) tok/s** | 27.03 (26.92) tok/s | 27.14 (27.05) tok/s | -1.6% / -1.5% | +0.4% / +0.5% |
| 4 | **78.74 (77.21) tok/s** | 73.37 (72.55) tok/s | 71.64 (71.44) tok/s | -9.0% / -7.5% | -2.4% / -1.5% |

Raw trials:

- FP8/B12X C1: 27.28, 27.45, 27.57 tok/s.
- W4A4/CUTLASS C1: 26.77, 26.92, 27.03 tok/s.
- W4A4/B12X C1: 17.98, 27.14, 27.05 tok/s.
- FP8/B12X C4: 65.03, 78.74, 77.21 tok/s.
- W4A4/CUTLASS C4: 62.64, 73.37, 72.55 tok/s.
- W4A4/B12X C4: 48.14, 71.64, 71.44 tok/s.

The first trial in each arm includes cold-path effects; the new W4A4/B12X arm
had 9.81 s C1 and 10.94 s C4 first-trial TTFT. Both best and median are therefore
reported. All arms used the unchanged canonical benchmark prompt (35 input
tokens) and produced exactly 512 tokens per stream.

## Interpretation

The prepared B12X implementation does not turn its real-layer microkernel gain
into an end-to-end decode gain. It is tied with CUTLASS at C1 and 1.5--2.4%
slower at C4, while both W4A4 backends remain behind FP8/B12X. Because DSpark
was physically off and its counters were absent for every trial, this is target
execution behavior, not a draft-acceptance effect. The next attribution must
measure routed-MoE time in the complete model; the isolated layer result is not
a substitute for API throughput.

## Evidence

- [`decode-fp8-b12x-target-only-c1-c4.json`](decode-fp8-b12x-target-only-c1-c4.json), SHA-256 `354fda11a996d535e066fed20ed3cfd635a348e1603a9563d5678c9d441b245d`
- [`decode-w4a4-target-only-c1-c4.json`](decode-w4a4-target-only-c1-c4.json), SHA-256 `25355abdc91d9e51b6523c673234001ef607d9547ce71256a6aaac0765ffd631`
- [`decode-w4a4-b12x-target-only-c1-c4.json`](decode-w4a4-b12x-target-only-c1-c4.json), SHA-256 `f3e66ed2485724424c4de7ccfebc795e652872914437efc1f9efed53912158e3`

The legacy FP8 image requires its released compatibility setting
`FLASHINFER_DISABLE_VERSION_CHECK=1` because that immutable image contains
FlashInfer Python 0.6.15 with 0.6.13 cubin/cache packages. The W4A4 image uses a
coherent 0.6.15 package set and leaves the bypass disabled.
