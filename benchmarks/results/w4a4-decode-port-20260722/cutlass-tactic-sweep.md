# FlashInfer CUTLASS tactic and PDL sweep: no material win

This bounded real-layer sweep tested the only nearby tactics present in the
SM121 service autotune caches.  It used prepared layer-0, TP-rank-0 weights,
balanced `M=4` routing, CUDA graphs, five repeats, and 500 samples per point.
All eight points produced finite, non-zero outputs and passed their eager and
graph numeric contracts.

## Results

| GEMM1 | GEMM2 | PDL | Graph median |
|---:|---:|:---:|---:|
| 16 | 58 | true | 0.766912 ms |
| 16 | 59 | true | **0.763104 ms** |
| 18 | 58 | true | 0.767904 ms |
| 18 | 59 | true | 0.765184 ms |
| 16 | 58 | false | 0.768064 ms |
| 16 | 59 | false | 0.765200 ms |
| 18 | 58 | false | 0.768960 ms |
| 18 | 59 | false | 0.768128 ms |

The exact service choice is GEMM1 `16`, GEMM2 `58`, PDL enabled.  The best
point changes only GEMM2 to `59` and improves median latency by `0.003808 ms`,
or **0.499%** (`1.00499x`).  That is below the predeclared 3% materiality gate
and far too small to close the service-level C4 gap.

## Decision

Keep the service cache unchanged.  PDL enabled is consistently equal or
better for these tactic pairs, and the alternative `16/59` result is within
sub-percent tuning noise.  The remaining optimization must change the kernel
schedule or eliminate work; tactic selection alone is exhausted.

Raw evidence:

- `cutlass-tactic-sweep/rank0-m4-balanced.json`
- `cutlass-tactic-sweep/rank0-m4-balanced.log`
