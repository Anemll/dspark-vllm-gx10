# Final SM121 W4A4 decode gates

These gates close the remaining bounded experiments after selecting the
FlashInfer B12X resident W4A4 microkernel with
`DSPARK_B12X_MICRO_MAX_ACTIVE_CLUSTERS=40`.

All real-layer rows use prepared DeepSeek V4 layer 0, TP rank 0, the captured
C4 route sample 131, CUDA graphs, and the same numerical acceptance envelope.

| Candidate | Result | Decision |
|---|---:|---|
| Accepted B12X MAC40 baseline | 0.776392 ms | default |
| Collision-safe C2--C4 activation-pack sharing | 0.771368 ms, +0.65% | correct but too small for a service reload |
| All-task weight-head prefetch | 0.778032 ms, -0.21% | reject |
| E8M0/K32 direct-kernel scale sidecar | 1.151088 ms vs 1.199936 ms E4M3 direct, +4.24% | useful attribution only; not representable in the resident kernel without also changing activation quantization |
| Sparse 256-group grouped GEMM (24 active, 232 zero-length) | 0.031040 ms | primitive passes, but the previously measured full route-major handoff is neutral/slower |

The direct E8M0 experiment proves that scale traffic has a measurable cost,
but the resident FlashInfer tensor-core operator couples the activation and
weight scale format. Its exact mixed contract—NVFP4 E4M3/K16 activations with
E8M0/K32 weights—would require a new split-SFA/SFB MMA abstraction rather than
a safe overlay. Homogeneous MXFP4 would change the activation quantization
contract and is not an eligible drop-in W4A4 optimization.

The all-task prefetch is distinct from the earlier rolling prefetch: it walks
every future compact task and hints the first two FC1 and FC2 weight/scale
tiles before wave zero. It compiled, ran with active finite output, and passed
the numerical gate, but did not beat the accepted kernel.

The sparse grouped-GEMM gate proves that repeated `m_indptr` offsets correctly
represent zero-length experts. This does not reopen the route-major design:
the real-layer grouped handoff, multiple ordering variants, and a persistent
producer/consumer implementation have already been measured and rejected.

Therefore the target-only deployment result remains:

```text
DSPARK_MOE_BACKEND=flashinfer_b12x
DSPARK_B12X_MICRO_MAX_ACTIVE_CLUSTERS=40
```

For W4A4+DSpark serving, keep the global backend on `auto` so the MXFP4 draft
layers use their compatible backend; apply the B12X selection only to the
prepared W4A4 target.

Prefill and M>micro-cutover stay on their existing selectors. Experimental
W4A16 dual decode, E8M0 sidecars, activation-pack fan-out, and prefetch
variants remain default-off.

## Evidence

- `all-task-head-sample131.json`
- `c4-token-shared-sample131.json`
- `e8m0-w4a4-direct-m4-mac40.json`
- `sparse-group-gemm.json`
- `../weight-prefetch/decode-w4a4-b12x-prefetch1-c4.json`
- `../route-major-negative.md`
- `../persistent-route-major-negative.md`
- `../service-dual-dispatch/README.md`
