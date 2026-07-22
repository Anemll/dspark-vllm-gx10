# W4A16 dual-decode service gate: negative result

This is the first valid TP=2 service A/B of the optional prepared-NVFP4
W4A16 decode branch.  Both arms used the same image, prepared checkpoint,
canonical prompt, 512 output tokens, temperature 0, TP=2, no DSpark draft, and
inactive speculative counters.  Revision
`1d514a2cbf70c121669253af16730413b285a4ab` and image
`sha256:f26ff9d4905df31365f254fb0d35c29f816a9ac363067b3c46820f1fe4cde91b`
were used throughout.

The dispatch proof is explicit: each rank logged exactly 43
`NVFP4_DUAL_DECODE event=selected` lines during FULL CUDA-graph capture, one
for every target MoE layer.  Both ranks loaded the prepared checkpoint through
43 layer reads with eight `preadv` calls and eight copies per layer, and neither
rank OOMed.

## Results

Values are best aggregate throughput with median in parentheses.  The CUTLASS
C=4 row uses the five-trial steady run; the three-trial run separately reached
75.38 tok/s once and had a 73.64 tok/s median.

| Target-only concurrency | Standard CUTLASS W4A4 | CUTLASS/W4A16 dual | Dual delta |
|---:|---:|---:|---:|
| 1 | 27.19 (27.12) tok/s | 27.25 (27.21) tok/s | +0.2% / +0.4% |
| 4 | **74.11 (73.66) tok/s** | 70.04 (69.83) tok/s | **-5.5% / -5.2%** |

The C=1 result is expected to tie because the policy leaves M=1 on CUTLASS.
At C=4, the valid W4A16 branch is materially slower.  Its single-layer win on
collision-free balanced routes did not survive the canonical service workload,
which launches four identical temperature-zero prompts and benefits from
CUTLASS expert reuse.  The dual class also retains about 4.03 GiB of E8M0/K32
scale sidecars per rank and uses a BF16-input contract; the standard CUTLASS
class retains neither overhead.

Compared with the matched FP8/B12X target-only reference (77.49 best, 76.85
median), standard W4A4 CUTLASS is 4.4% slower by steady-run best and 4.2% slower
by median.  The experimental W4A16 branch widens rather than closes that gap.

## Decision

Keep `VLLM_NVFP4_W4A16_DUAL_DECODE=0` as the default.  Do not promote the
W4A16 sidecar path for serving.  The current FlashInfer CUTLASS W4A4 backend is
the accepted decode implementation as well as the faster prefill path.

Raw evidence:

- `decode-dual-dispatch-c1-c4.json`: valid W4A16 arm.
- `decode-cutlass-control-c1-c4.json`: same-image standard CUTLASS control.
- `decode-cutlass-control-c4-steady5.json`: five additional warmed C=4 trials.
- `head.log`, `worker.log`: both-rank load and branch-selection proof.
- `head-inspect.json`, `worker-inspect.json`: exact candidate identity.
