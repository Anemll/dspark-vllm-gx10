# ModelOpt E8M0/K32 tensor-core tactic screen

Real prepared layer 0, TP rank 0, balanced routing, CUDA graph, M=1 and M=4.
The packed FP4 payload is shared with the W4A4 source; only the exact
E4M3/K16 scale pairs are collapsed to E8M0/K32.

| FC1/FC2 tactic | M=1 (ms) | M=4 (ms) | Result |
|---|---:|---:|---|
| B/B (K64/N128) | 0.329176 | 1.288096 | reject |
| B/C | 0.292688 | 1.098712 | reject |
| C/B | 0.228160 | 0.847320 | reject |
| C/C (K128/N64) | **0.196312** | **0.745904** | best |
| A/A (K128/N128) | 0.200424 | 0.746136 | tied with C/C |

The one-stage canonical-to-packed shared-memory transform was also rejected:
C/C M=4 was 1.055704 ms. A corrected cooperative global loader reached
0.757512 ms and did not beat the existing C/C loader.

The promotion threshold is 0.682812 ms at M=4. Geometry and cooperative
loading are therefore closed; the next experiment targets redundant generic
E8M0 special-value conversion instructions in the retained ModelOpt C/C path.

Raw JSON/log evidence is in `modelopt-tc-stage-pack/` and
`modelopt-tc-cooperative/`.
