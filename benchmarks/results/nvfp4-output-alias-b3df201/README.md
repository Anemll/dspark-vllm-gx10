# NVFP4 B12X decode-overhead gates (`b3df201`)

Hardware: one GX10/GB10 rank, SM121, TP-rank-0 real layer-0 weights. All
timings are CUDA-graph medians; they are layer-kernel timings, not model API
tokens/second.

## Direct-output alias

The direct-output implementation removes the vLLM adapter copy and the
downstream no-op finalizer copy. It is numerically within the standard
cross-launch FP4 envelope and passes the explicit output-pointer identity
contract. Independent B12X launches are not bit deterministic; the raw first
run's `ok=false` is solely the superseded exact-zero-difference assertion.

| M | Legacy two-copy | Direct output | Saved/layer | 43-layer projection |
|---:|---:|---:|---:|---:|
| 1 | 0.203824 ms | 0.202704 ms | 1.120 us | 0.048 ms/step |
| 4 | 0.777784 ms | 0.774352 ms | 3.432 us | 0.148 ms/step |

This is correct and retained, but too small to justify a full serving A/B by
itself.

## Same-weight W4A4 versus W4A16

| M | W4A4 | W4A16 | W4A4 speedup |
|---:|---:|---:|---:|
| 1 | 0.197824 ms | 0.287904 ms | 1.455x |
| 4 | 0.772064 ms | 1.185840 ms | 1.536x |

The W4A4 expert kernel is not the source of the API regression. The missing
performance is integration/dispatch overhead outside the same-weight expert
compute.

## B12X wrapper-capacity A/B

The serving adapter used `max_num_batched_tokens=8192` for every decode call;
the earlier microbenchmarks used a capacity of four. The capacity alone has a
measurable recurring cost at the same actual M:

| M | Capacity 8192 | Capacity 4 | Saved/layer | 43-layer projection |
|---:|---:|---:|---:|---:|
| 1 | 0.210864 ms | 0.200624 ms | 10.240 us | 0.440 ms/step |
| 4 | 0.802880 ms | 0.775216 ms | 27.664 us | 1.190 ms/step |

The accepted implementation therefore uses a model-shared decode wrapper
sized to the CUDA-graph capture frontier and preserves the original 8192-token
wrapper for larger/prefill M. This is decode-only selection; prefill retains
its existing capacity and path.

Raw evidence is in this directory:

- `result.json`, `run.log`: direct-output and B12X-vs-CUTLASS gate.
- `w4a4-vs-w4a16.json`, `.log`: activation-path comparison.
- `b12x-max-4.json`, `.log` and `b12x-max-8192.json`, `.log`: capacity A/B.
