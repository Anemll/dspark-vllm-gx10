# W4A4 decode: route normalization and cold-weight attribution

This gate separates checkpoint routing behavior from intrinsic SM121 W4A4
kernel behavior.  Every row uses the same prepared W4A4 layer-0 weights and
the accepted B12X MAC40 kernel.  The only changed input is either the captured
route tensor or the L2 state.

## Route-normalized result

The FP8 production checkpoint activates 15.068 experts per layer on average.
The NVIDIA W4A4 checkpoint activates 15.576 (+0.508, +3.37%) and therefore has
less expert reuse.

| Captured C4 route set | B12X mean | CUTLASS mean |
|---|---:|---:|
| FP8/B12X checkpoint routes | 503.592 us | 512.866 us |
| W4A4 checkpoint routes | 518.881 us | 527.767 us |

The route-distribution penalty is **15.289 us/layer**, or about **0.657 ms**
over 43 routed layers.  It is checkpoint behavior, not a W4A4 kernel cost.

## Cold-weight result

The first 256 identical FP8 route samples were replayed both normally and with
a 64 MiB L2 eviction immediately before every timed launch.

| L2 state | B12X MAC40 mean | CUTLASS mean |
|---|---:|---:|
| Hot | 381.216 us | 388.210 us |
| Evicted | 426.989 us | 440.336 us |
| Penalty | **+45.774 us** | **+52.126 us** |

This reproduces the approximately 54 us/layer difference between the hot
one-layer replay and the rotating 43-layer serving profile.  The residual
intrinsic C4 opportunity is therefore the layer-to-layer expert-weight stream,
not NCCL, routing compaction, or activation quantization.

## Rejected kernel changes

Two source-pinned variants were stopped at the route-replay gate:

| Variant | B12X mean | Delta vs accepted 503.592 us |
|---|---:|---:|
| One A/B pipeline stage | 533.430 us | **+5.92% slower** |
| Fused gate/up A+SFA load, v2 | 505.671 us | +0.41% slower |
| Fused gate/up with early next-stage wait, v3 | 505.304 us | +0.34% slower |

One stage reduces shared memory but loses too much prefetch depth.  Fusing
gate/up activation-side loads is effectively neutral and does not address the
cold weight stream.  None of these variants should be deployed.

The accepted deployable configuration remains
`DSPARK_B12X_MICRO_MAX_ACTIVE_CLUSTERS=40`.
