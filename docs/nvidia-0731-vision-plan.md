# Plan: NVIDIA 0731 NVFP4 + original DSpark + official Vision-Exp modules

Date: 2026-09-13. Status: executed and rejected at the first vision-correctness
gate. The accepted Vision-Exp deployment was restored. See the
[execution report](../benchmarks/results/nvidia-0731-vision-transplant-20260913.md).

## Objective and checkpoint identity

Evaluate the requested experimental combination on two GB10 nodes:

| Component | Source | Required behavior |
| --- | --- | --- |
| Language target | `nvidia/DeepSeek-V4-Flash-0731-NVFP4`, revision `f1caa71142bd0be02f728c79f75042ac1e461579` | Preserve target tensors, NVFP4 calibration/scales, and language configuration; prove W4A4 routed-expert execution |
| DSpark draft | Original `mtp.*` tensors retained in that NVIDIA checkpoint | Preserve MXFP4 draft format; begin with DSpark K5 |
| Vision modules | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`, pinned to the available donor revision `6821d6ad3681a4b137b066b76094fa82ebd0a380` after verification | Add image encoder, aligner, sentinel vectors, and required modality-specific routing parameters |

The user-selected TB36 source directory is named
`DeepSeek-V4-Flash-0731-NVFP4`. Its config was rechecked during planning:
`moe_quant_algo=NVFP4`, expert group size 16, and explicit `mtp.*` exclusion.
The global `quant_method=fp8` field does not describe all expert layers.

Use a distinct candidate name, for example
`anemll-dsv4-0731-nvfp4-vision-hybrid-exp`. Publish the three source identities
and per-component precision with any result. This combination would be an
Anemll experimental hybrid, not an official NVIDIA vision release.

## Evidence to reuse

- [September 6 compatibility audit](../benchmarks/results/nvidia-vision-hybrid-preflight-20260906.json):
  checked structural dimensions match, but all 18 sampled shared tensors differ
  between 0731 and Vision-Exp. Equal shapes do not prove visual alignment.
- The audit identified 316 additional tensors: 263 encoder/aligner tensors,
  four sentinel vectors, 46 target/draft visual-router biases, and three
  hash-layer correction biases. Copying only the encoder would be incomplete.
- The language RMSNorm epsilon differs (`1e-6` for NVIDIA/0731 versus `1e-20`
  for Vision-Exp). The reported next-token prediction layer count also differs.
  Resolve these fields explicitly against tensor names and the actual loader.
- [September 9 NVIDIA text-only TP2 test](../benchmarks/results/nvidia-local-nvfp4-tp2-20260909.md):
  NVIDIA W4A4 target plus original DSpark loaded from local NVMe and generated
  text. C1 was 28.29–29.49 tok/s with 37.85–40.71% acceptance. This is an older
  runtime/network result, not a current matched speed baseline or proof that
  W4A4 itself caused the slowdown.
- The latest Arena run measured official Vision-Exp with B12X MXFP4/MXFP8
  experts. It provides a reference for the current deployment, not a benchmark
  of this proposed hybrid. Its text prompts must not be compared directly with
  different content-suite prompts.

## 1. Audit and assemble an immutable candidate

1. Inspect existing NVIDIA local-NVMe copies on both nodes before copying any
   large data. Verify their index, tensor coverage, sizes, source revision, and
   available integrity evidence against TB36. Record fresh checksums for new
   or changed artifacts; config/index hashes alone do not prove shard equality.
2. Build a tensor manifest recording source revision, source tensor name, dtype,
   shape, destination, and whether each entry is shared or modality-specific.
   Reject unknown names, conflicting mappings, and missing required tensors.
3. Reuse NVIDIA target and draft shards read-only. Extract the small donor
   tensor set into a separate shard and construct a separate hybrid index and
   config. Do not edit either source checkpoint or reserialize all 175.55 GB.
   Use references that resolve inside the candidate container.
4. Preserve NVIDIA shared language weights, target normalization, expert scales,
   and draft weights. Map visual biases to their intended image path. Any bias
   required on text tokens is an explicit behavior change and must pass parity
   testing; never silently overwrite shared parameters to make loading succeed.
5. Audit token IDs, image sentinels, prompt encoding, image preprocessing,
   aligner dimensions, vision-specific normalization, and draft image handling.
   Retain text prompt compatibility and add only the needed multimodal behavior.
   Derive draft layer/shard selection from the real `mtp.*` tensor manifest.

Deliverables: reproducible assembly script with dry-run validation, provenance
manifest, candidate config/index, and focused assembly/dispatch tests.

## 2. Verify the current runtime can execute the hybrid

The September 10 B12X image uses a newer vLLM tree than `upstream.lock` and the
July overlay. Start from the current digest-pinned image and
`config/b12x-vision.lock.json`; inspect its actual runtime sources. Do not copy
the entire old overlay or the historical CUTLASS binary into this newer stack.

- Check independent NVFP4 target / MXFP4 draft dispatch, draft backend override,
  and the MTP-only shard-loading optimization in the active runner.
- Preserve the required SwiGLU clamp of 10. Verify NVFP4 W4A4 activation
  quantization and scale handling with the installed compiled kernel.
- Start from the previously working target CUTLASS / draft B12X split if the
  current runtime supports it. Any newer target backend is a separate measured
  choice, not an implicit part of enabling vision.
- Keep the existing text-only vision-wrapper fast path and verify that image
  prefill still injects embeddings and modality routing correctly.
- Run CPU contract tests, then a bounded SM121 kernel correctness canary. If
  patches are needed, build once, pin the result, and transfer that identical
  image to the second node over the dedicated fabric.

Deliverables: exact runtime identity, tested minimal patches if required,
compiled-kernel evidence, and matching candidate image IDs on both ranks.

## 3. Establish NVIDIA-only control A, then load hybrid B

Acquire the experiment lock, verify an idle service, capture both-rank logs and
configs, and preserve the current accepted Vision-Exp image/config for recovery.
Keep the release rollback artifacts as well. Consult the configured read-only
advisor before disruptive work per repository guidance; record unavailability
if that channel is absent.

Use local NVMe for inference. If restaging is necessary, measure the approved
wired storage route before committing to a transfer; inter-Spark bulk copies
must use the dedicated fabric. Never serve directly from the NAS for this run.

Hold runtime image, network, FP8 KV format and cache budget, context (350K),
max active sequences (4), batched-token budget (2048), DSpark K5, prompts,
token limits, sampling, and seeds fixed between A and B. Verify the requested
context fits after vision memory is reserved; do not silently lower it.

- **A:** NVIDIA 0731 NVFP4 + its original DSpark, text only, on the selected
  current runtime. Run a small correctness/performance control. The historical
  NVIDIA result cannot replace this control because runtime and fabric changed.
- **B:** The same target/draft with the added vision modules. Stop head then
  worker; start worker then head. Require complete weight loading, exact tensor
  coverage, both-rank backend evidence, and successful text/image requests.

## 4. Prove image understanding and DSpark behavior

Before a full speed sweep, run known-answer OCR, chart reading, spatial/color
questions, and two-image comparison using the existing vision fixtures.
Include paired requests with the same text and different images whose correct
answers differ. Confirm image payloads reach the server and change the answer.

Use a bounded same-hybrid DSpark-off control for diagnosis, followed by K5.
Check image-request draft acceptance and target verification, output sanity,
streaming, usage, and tool calls. Speculation must remain correctly conditioned
on the image. If that cannot be established, DSpark-plus-vision is still an open
gate even if image inference without speculation works.

Compare fixture correctness with the accepted official Vision-Exp baseline.
Loading successfully or describing one image is insufficient to establish
compatibility. Failure here rejects the transplant for deployment.

## 5. Measure text overhead, then separate vision performance

- Repeat the canonical content suite for A and B: identical content categories,
  fixture hashes, seeds, thinking settings, output lengths, and C1/C2/C3/C4.
  Report decode-only and aggregate speed separately, with TTFT, prefill,
  acceptance, draft time, and target-verification time.
- Measure image prefill and decode separately across the existing image cases,
  resolutions, and tested concurrency. Slower vision decoding is acceptable
  under the user's stated preference; image correctness remains mandatory.
- Check exact-length prefill and long-context boundaries before promotion,
  including 65K and 100K. Watch both ranks for CUDA/NCCL errors and preemptions.
- Treat differences below 3% as noise. If needed, alternate short A/B runs to
  resolve a near-threshold result. A repeatable text slowdown above 3% from
  adding vision fails the text-preservation gate.
- Compare the hybrid with the existing accepted deployment as a separate
  practical decision. W4A4 does not establish a speed improvement by itself.
  Reuse existing measurements only when their relevant fingerprints match.

Only after these gates pass, repeat Spark Arena V2. For its C10 cells,
temporarily allow ten active sequences, verify actual running/waiting counts,
and restore four afterward. Disclose capacity differences with public controls.

## Budgets, aborts, and completion

For the first execution tranche, budget 60 minutes for preparation and a
maximum 45-minute service interruption for A/B canaries including 15 minutes
reserved for recovery. Validate load-time projections before stopping the
service. If preparation cannot fit, report the measured bottleneck and revised
plan before starting an outage. Full content and Arena sweeps are subsequent
stages with estimates based on measured canaries, not part of this initial cap.

Abort a candidate on missing/misrouted tensors, image correctness failure,
non-finite results, either-rank death, CUDA/NCCL errors, or rollback/storage
readiness loss. Preserve logs and restore the accepted deployment before
further diagnostics. Initial storage floor: 20 GiB free per node after all
planned writes, with source checkpoints, rollback images, and failure evidence
retained. Send progress at least once per minute during loads.

Success requires identified NVIDIA W4A4 target execution, original draft
weights with correct dispatch, usable vision with DSpark enabled, and measured
text-speed preservation. Record each gate honestly, update the README/dashboard
with the exact experimental model identity, and keep benchmark provenance
distinct from official releases. Arena upload remains a separate authenticated
submission step.

If visual alignment fails, the alternative is to quantize the complete trained
Vision-Exp model to NVFP4 while preserving its language/vision relationship.
That is a separate candidate and must not silently replace the requested
NVIDIA-0731 transplant.

Sources: [NVIDIA model card](https://huggingface.co/nvidia/DeepSeek-V4-Flash-0731-NVFP4),
[official Vision-Exp model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp),
and the repository evidence linked above.
