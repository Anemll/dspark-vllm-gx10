# Superseded dispatch run

`decode-dual-c1-c4.json` did **not** execute the W4A16 branch.  The pinned V2
runner computed `uniform_tok_count` but dropped it when constructing both the
runtime and FULL CUDA-graph forward descriptors.  Both ranks therefore logged
zero `NVFP4_DUAL_DECODE event=selected` events and silently used the CUTLASS
fallback.

The artifact is retained only as forensic evidence and must not be cited as a
W4A16 service result.  Revision `1d514a2cbf70c121669253af16730413b285a4ab`
fixes the descriptor propagation, excludes active prefill requests, preserves
PIECEWISE semantics, and logs one selection event per target layer.  The valid
results are under `../service-dual-dispatch/`.
