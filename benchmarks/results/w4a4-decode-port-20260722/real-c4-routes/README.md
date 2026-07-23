# Real target-only C4 route structure

This diagnostic used the prepared NVFP4 target-only service with four
identical canonical requests, temperature zero, top-k 6, and no DSpark draft.
The capture hook was installed in the active vLLM V1 runner and recorded 64
steady decode steps for all 43 routed-MoE layers on both TP ranks.  The two
rank arrays are byte-identical, proving that the target routes agree across
TP.

## Result

The logical shape is `[64 steps, 4 tokens, 43 layers, 6 routes]`.  Of the 24
routes in each layer/step:

- mean unique experts: **15.5763** (median 15, p90 20);
- mean repeated routes: **8.4237**, or **35.10%** of all routes;
- maximum expert-row multiplicity: 4;
- mean per-layer/step maximum multiplicity: **3.5109**;
- layer/step maximum multiplicity was 4 in 1,829/2,752 cases (66.46%);
- active-expert occurrences comprised 29,039 singletons, 7,540 pairs, 3,219
  triples, and 3,068 quadruples.

This confirms that the canonical C4 workload is not the collision-free
balanced route pattern used by earlier component screens.  Any decode
specialization must preserve CUTLASS's benefit from repeated experts and
should target the common 2--4 rows per active expert.  The capture run's API
throughput is intentionally not a performance result: the diagnostic CUDA
graphs contain route-copy operations.

## Evidence

- `analysis.json` SHA-256
  `cb732fa978083dfdf2dbbe321bd83a753b24e2f5af34c36edce19def3ea964f7`
- rank-0 and rank-1 arrays both SHA-256
  `7322bccdd515e0f0c6c35eef1331a04c53d20cf45edbc2b2e91e1d67f9d1dcd6`
- rank-0 layer-sample view `[64, 43, 4, 6]` SHA-256
  `2b687fcc275984a37b3c5777bfa73b7f6e3a0de7e38abac38f0390d918449a88`;
  this is a byte-preserving axis transpose used by the isolated real-layer
  kernel replay harness
- diagnostic API JSON SHA-256
  `6dbf72c73a503752933ce0b56f4faa8b4f5c577dff12841fb43af5e09173ba68`
- capture revision `e22aa1cb8dfe04f198b2f8e3714a6446825f2f50`
- image `sha256:105cb6b85510c27ad3e772bed971efa30a6847058e1b7263b45b2141626e6726`
