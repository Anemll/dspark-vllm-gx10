# Exact-path B12X W4A16 comparison

This one-layer gate compares two different DeepSeek V4 Flash checkpoints after
putting both into the same native MXFP4 representation and executing the exact
production `B12xExperts` W4A16 path:

- **Converted:** NVIDIA prepared NVFP4 layer 0, losslessly collapsed from
  E4M3/K16 scale pairs to E8M0/K32 and restored to W13 `[w1, w3]` order.
- **Control:** abliterated production layer 0, read directly as native packed
  FP4 + E8M0/K32 and TP-sliced exactly as serving does.

The arms use identical BF16 activations, routes, B12X preparation, scratch
planning, CUDA graphs, and paired-order timing. The checkpoints contain
different trained weight values, so cross-checkpoint byte/output equality is
neither expected nor used. Each arm independently passed finite/non-zero
output and graph-versus-eager identity.

## Result

| Routed tokens (M) | Converted NVIDIA | Abliterated native | Converted delta |
|---:|---:|---:|---:|
| 1 | 0.196560 ms | 0.196256 ms | +0.15% |
| 4 | 0.706928 ms | 0.713480 ms | -0.92% |
| 24 | 4.164192 ms | 4.173048 ms | -0.21% |
| 48 | 7.405936 ms | 7.430728 ms | -0.33% |

The pre-registered decision shapes M=24 and M=48 pass the ±3% parity gate.
They also remain within 0.34%, the tighter diagnostic expectation for
identical representation and kernel dispatch.

Relative to the banked W4A4 CUTLASS controls from the MAC-sweep/DeepGEMM
control lineage, the converted exact B12X path is:

- **+11.3%** at M=1 (`0.218800 / 0.196560`)
- **+10.8%** at M=4 (`0.783480 / 0.706928`)
- **+8.94%** at M=24 (`4.536504 / 4.164192`)
- **+8.55%** at M=48 (`8.039328 / 7.405936`)

The pre-registered component gate requires at least **+3%** at both DSpark
verifier shapes. It therefore passes with **+5.94 percentage points** of
margin at M=24 and **+5.55 points** at M=48. No additional hardware run is
needed for this decision.

## Decision

The earlier decode gap is not caused by the NVIDIA FP4 payload or trained
weights. Once the prepared checkpoint is restored to native MXFP4 scale/layout
and dispatched through production B12X W4A16, its expert-kernel latency matches
the abliterated checkpoint and clears the project's component-speed gate over
W4A4 CUTLASS. Proceed with serving integration that retains the existing W4A4
CUTLASS prefill path and selects native E8M0/K32 B12X W4A16 for
decode/verifier-sized batches.

## Provenance

- Harness revision: `5f19b4402369ea9a97b41eb97db5135a707582f3`
- Runtime image: `sha256:3430d6614a8e2925f34d059af6caf05aff42387326db4d05639a60f10f2654d8`
- GPU: NVIDIA GB10, SM121
- Prepared source:
  `/home/anemll/models/DeepSeek-V4-Flash-NVFP4-TP2-W4A4-v1/model-layer-00000.safetensors`
- Abliterated source:
  `/srv/dspark/models/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored/model-00002-of-00048.safetensors`
- Raw JSON SHA-256:
  `49d32d46896e6f6d285c9218be88ed2771a9784b4487ea0e9a53b112503340e3`
- Raw log SHA-256:
  `55d07765630ed25493a5dbcb55a4699fa22b822161021fec1a9124dd0e824f08`
- Load time: 14.827 s
- Peak CUDA allocation: 3.875 GiB

The inert dev image used for the first launch contained a Python/package ABI
mismatch (`w4a16_weight_layout` was passed to an older `TPMoEScratchCaps`) and
failed before timing. The accepted run therefore used the immutable production
image, whose B12X helper signature is the exact serving reference.
