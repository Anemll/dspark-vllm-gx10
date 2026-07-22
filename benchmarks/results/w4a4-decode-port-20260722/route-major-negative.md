# Route-major W4A4 phase-1 variants: negative gate

These bounded real-layer probes tested whether preserving route-major
activation order between W4A4 FC1 and FC2 could recover the remaining `M=4`
decode gap.  Every variant used the prepared layer-0, TP-rank-0 tensors and a
matched FlashInfer CUTLASS fused-MoE reference.  Timing was paired and
alternated in both execution orders to reduce order bias.

## Results

| Variant | Accepted fused reference | Route-major candidate | Candidate latency delta |
|---|---:|---:|---:|
| Base phase-1 | **0.799592 ms** | 0.803688 ms | +0.51% |
| Phase overlap | **0.795280 ms** | 0.803416 ms | +1.02% |
| Bounded combined | **0.797504 ms** | 0.807552 ms | +1.26% |
| Combined W13 | **0.801192 ms** | 0.815968 ms | +1.84% |
| Max active clusters = 48 | **0.800760 ms** | 0.810352 ms | +1.20% |

All five candidates are slower than their paired reference and all miss the
`0.682812 ms` layer screen.  The phase-overlap experiment also failed its
numeric gate (cosine approximately `0.953`, normalized RMSE approximately
`0.307`), so its timing is not eligible even as a performance result.

An additional `N=256` schedule did not compile: CuTe rejected the generated
scale-factor view as weakly incongruent in `moe_phase1_kernel.py`.  It produced
no timing and is retained only as compile-failure evidence.

## Decision

Do not integrate these route-major variants.  Reordering alone does not remove
enough work from the dominant fused CUTLASS kernel, and the measured overhead
is already larger than any saving.  A viable follow-up would need a structural
kernel change that fuses routing/TMA setup into the main tensor-core schedule,
not another handoff or order-only adapter.

Raw evidence is under `route-major-negative/`:

- `phase1-m4.json` and `.log`
- `phase1-m4-overlap.json` and `.log`
- `phase1-m4-combined-bounded.json` and `.log`
- `phase1-m4-combined-w13.json` and `.log`
- `phase1-m4-mac48.json` and `.log`
- `phase1-m4-n256.log` (compile rejection; no JSON)

Relevant implementation commits: `dfc22c2`, `138c4d8`, `ae61dee`, and
`a234118`.
