# Exhaustive native CUTLASS W4A4 tactic screen

This gate used the real prepared layer-0 rank-0 weights, the production TP=2
shape (`M=4`, `K=4096`, `I/rank=1024`, 256 experts, top-k 6), balanced
collision-free routes, CUDA graphs, and PDL enabled.

- Launchable native pairs measured: 480
- Native unsupported profiles skipped: 14
- Accepted service pair: GEMM1 tactic 16, GEMM2 tactic 58
- Accepted service median: 0.768480 ms
- Best pair: GEMM1 tactic 16, GEMM2 tactic 58
- Best median: 0.768480 ms
- Speedup over service: 1.000x

Decision: the current FlashInfer service pair is exactly fastest in the native
tactic space. The remaining C4 service gap cannot be recovered by selecting a
different existing CUTLASS profile; the next gate targets repeated-expert route
structure instead.

Evidence:

- `rank0-m4-all-pdl.json` SHA-256
  `c9564440c59c94af6dc9318a537b55c7910a661ee786efcd127f7fa01125b9b6`
- `rank0-m4-all-pdl.log` SHA-256
  `02f6b7fa08ff6bfa0fb49c32f5fe9c3333c6da65f5540fa0c01ad1b9db784823`
