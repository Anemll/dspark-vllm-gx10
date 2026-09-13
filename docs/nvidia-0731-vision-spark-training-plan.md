# Plan: Train the Vision Aligner on Two DGX Sparks

Date: 2026-09-13. Status: conditional proposal, revised after candidate review. Branch:
`nvidia-nvfp4-0731-vision-transplant`.

This plan answers one question raised by the
[remap and fine-tuning plan](nvidia-0731-vision-remap-finetune-plan.md): can
Phase 5 projector-only SFT run on the two GB10 Sparks instead of external
GPUs? A pilot is plausible under the design below, but actual memory, gradients
and throughput are not yet validated. A BF16 copy of the 304B language model
does not fit, but a BF16 copy is not required. The corrections in the
[source review](nvidia-0731-vision-spark-training-review.md) apply before
implementing this harness; estimates below are not measured training results.

## Q0 result: do not schedule projector training yet

Q0 in the [main plan](nvidia-0731-vision-remap-finetune-plan.md) is complete.
The [TP=2 result](../benchmarks/results/msuiche-vision-nvfp4-tp2-20260913.md)
proved working target-only and DSpark vision, including paired image swaps, but
rejected the current DSpark combination: matched C1 fell 64.7% as acceptance
collapsed from 32.73% to 0.318%. This is not a projector failure, so do not book
a training window or prepare the 2K corpus yet. First measure target-only text,
instrument target/draft drift, and recalibrate the routed-expert activation
scales. Remapped-aligner training remains justified only if exact NVIDIA 0731
is required or the no-training scale/MTP path fails.

## Scope

Trainable on the Sparks: the remapped donor aligner only, about 54.5M
parameters, warm-started from the closed-form map. The four sentinel vectors
become trainable only in explicitly named sentinel ablations. Keep `bias_vl`
fixed for donor/zero routing ablations: hard top-k yields no ordinary answer-loss
gradient for these biases. Training them requires a separately validated
estimator, as explained in the source review. Everything else is frozen and
never receives a gradient or optimizer state: the Vision-Exp tower, all 43
language layers, embeddings, head, and the DSpark draft.

## Why it might fit, subject to the resource gates

The frozen language model does not need a resident full BF16 copy, weight
gradients or optimizer state. Temporary decoded weights still need memory.
Only the aligner is trained. Backward through a frozen
linear layer is one more matrix product with the same weights, so the weights
can stay in their stored 4-bit and 8-bit formats and be dequantized on the
fly in both directions. Serving demonstrates that compressed weights can fit,
not that the training harness and its additional buffers fit alongside them.

Facts below describe the **original DeepSeek storage baseline**, not the NVIDIA
export. It may be used for the frozen recipient only after the equivalence
gate below passes. Direct NVIDIA loading instead requires its group-16 NVFP4
decoder/global scales and approximately 175.551 GB checkpoint accounting.

| Item | Value |
| --- | ---: |
| Language layers | 43 |
| Hidden size | 4096 |
| Hyper-connection streams (`hc_mult`) | 4 |
| Routed experts / active per token / shared | 256 / 6 / 1 |
| Expert intermediate size | 2048 |
| Expert storage | FP4 `e2m1` packed pairs, block 32, E8M0 scales |
| Non-expert linears | FP8 `e4m3`, block 128, E8M0 scales |
| Checkpoint size on disk | about 167 GB |
| Aligner parameters (9216 to 4096 to 4096) | 54.5M |
| Image tokens per image (`vision_max_n_token`) | 384 |
| Sparse attention selection (`index_topk`) | 512 |
| Sliding window | 128 |
| Hash-routed layers | first 3 |

## Conditional equivalence: original 0731 versus NVIDIA 0731

Extend the audit with decoded-value comparisons of original 0731 and the exact
NVIDIA 0731 snapshot. For each covered expert projection compare E2M1 values
times E8M0 group-32 scales with E2M1 values times E4M3 group-16/global scales;
record packed nibble identity separately. Begin with one expert's w1/w2/w3,
expand across layers/scales, and do not generalize a sample pass to every weight.
Verify passthrough tensors and configuration semantics too. Full equivalence
requires complete value/scale/passthrough coverage or a validated exact-conversion
check covering the entire relevant payload. The claim from a different community
transcode does not certify NVIDIA's export.

**If that gate passes**, training the frozen original 0731 recipient uses the
same logical weight values as NVIDIA 0731 despite different storage. The
weight-format approximation gap disappears; the intended training-to-serving
quantization gap is activation precision/calibration, not changed weights.
That permits the original compressed decoder as an implementation choice,
with the audit result pinned in the experiment manifest.

This conclusion does not make Vision-Exp's continued-trained language weights
equal to 0731. It also does not guarantee numerical runtime equivalence: keep
epsilon, tokenizer, masks, KV precision and other settings matched, and validate
loader fusion/rescaling and kernel accumulation. Separate tests compare
Vision-Exp with msuiche; their results cannot establish original/NVIDIA 0731
equivalence. If the weight gate fails, load the NVIDIA values directly and
retain a weight-difference term in the training/deployment attribution.

### Short-context and activation assumptions

Two consequences drive the design:

- **Sequences under 512 tokens make sparse attention exact and dense.** With
  384 image tokens plus a short prompt and answer, every compressed KV entry
  is inside the top-512 selection. The lightning indexer, Hadamard rotation,
  and the TileLang sparse-attention kernel are unnecessary; masked dense
  attention in plain PyTorch computes the same result.
- **Activation memory is small.** Per-layer checkpointing at micro-batch 4 by
  512 tokens stores four residual streams of 4096 per token per layer, about
  6 GB for the whole model. Weights dominate, and weights are fixed.

## Memory budget per Spark

Tensor-parallel 2 in the DeepSeek reference layout: experts split 128 per
rank, column/row linears sharded, embedding sharded by vocabulary.

| Component | GiB (estimate) |
| --- | ---: |
| Sharded language weights, stored precision | 84 |
| Vision tower and aligner, BF16, replicated | 1 |
| Aligner fp32 master, AdamW moments, BF16 grads | 1 |
| Checkpointed activations, micro-batch 4 by 512 | 6 |
| In-layer recompute peak (MLA q/k/v, expert activations) | 3 |
| Dequant scratch, one expert shard at a time | 2 |
| CUDA context, allocator slack | 6 |
| **Total** | **103** |

The Sparks have 128 GB unified memory shared with the host. The serving runs
in `vision-loading-memory.md` bottomed out near 14 GiB available with a
79 GiB model resident, so 103 GiB leaves less headroom than serving. The
first gate below measures this rather than assuming it. If it does not fit,
micro-batch 2 halves the activation and recompute lines.

## Harness design

Base the harness on DeepSeek's `inference/model.py` from the Vision-Exp
repository, not on the vLLM overlay. It is a readable reference with plain
`torch.distributed` tensor parallelism, `ColumnParallelLinear`,
`RowParallelLinear`, sharded experts, the `Compressor`, hyper-connections,
hash gating with `tid2eid`, `bias_vl` gating, and `merge_image_embeddings`.
Run it at model-parallel 2 with a bounded streaming loader/resharder. Its
unmodified convert.py retains whole rank dictionaries and is unsuitable for
the claimed streaming memory budget. Required changes (plus the source review):

1. **Dequantize-on-the-fly autograd.** Replace `linear()` with a custom
   `torch.autograd.Function`. Forward unpacks FP4 pairs or FP8 blocks to BF16
   for one weight at a time, multiplies, and discards the BF16 copy. Backward
   repeats the unpack and computes only the input gradient. Nothing but the
   input activation is saved. This is exact because the weights are frozen.
2. **Replace TileLang kernels with torch ops.** `fp4_gemm`, `fp8_gemm`,
   `act_quant`, `fp4_act_quant`, `sparse_attn`, and `hc_split_sinkhorn` are
   all TileLang JIT with no torch fallback. None is needed: use BF16
   activations, masked SDPA for the sliding window and the compressed KV, and a
   twenty-iteration Sinkhorn in fp32. The requirement on `tilelang` and
   `fast_hadamard_transform` disappears.
3. **Remove `@torch.inference_mode()`** from `Transformer.forward` and
   `encode_image`, and remove every in-place quantization.
4. **Differentiable collectives.** Route the tensor-parallel `all_reduce`
   calls through `torch.distributed.nn.functional` so gradients cross ranks.
5. **Per-`Block` activation checkpointing** with `use_reentrant=False`.
6. **Image mechanics from the overlay.** Sentinel substitution before the
   hash lookup, `bias_vl` at image positions, and the image-span visibility
   mask must match `overlay/vllm/models/deepseek_v4/nvidia/` exactly. The
   overlay's CPU tests are the reference for these.
7. **Trainable set.** `Aligner.w1`, `Aligner.w2`, and their biases in fp32
   master weights. Sentinel vectors and `bias_vl` stay frozen unless an
   ablation arm enables them by explicit allowlist.
8. **Loss.** Teacher-forced cross-entropy on answer tokens through the real
   head, plus the auxiliary terms from the fine-tuning plan. Prompt and image
   positions are masked out of the loss.
9. **Streaming weight load** into preallocated tensors per rank, following
   the repository rule that no loader materializes the checkpoint as a list.

The training-time numerics are W4A16 for experts and W8A16 for the relevant
FP8 linears. The served NVIDIA target is W4A4 and the original expert serving
path is W4A8. After the equivalence gate passes, attribute this intended gap
to activation precision and input scales rather than weight remapping. Keep
kernel/loader differences separately controlled. A forward activation
fake-quantization mode with an explicit straight-through surrogate is a possible
follow-up; its buffers, operations and numerical behavior must be measured.
It is neither the exact derivative of quantization nor memory-free.

## Throughput expectation

Analytic floor per micro-batch of 4 by 512 tokens on one Spark:

| Term | Estimate |
| --- | ---: |
| Weight bytes streamed per pass | 84 GB |
| Passes per step (forward, recompute, backward) | 3 |
| Weight streaming at 273 GB/s, with unpack | about 1.5 s |
| Active parameters per token | about 12B |
| Dense FLOPs per step at 4 by 512 tokens | about 200 TFLOP |
| Compute at 50 to 60 TFLOPS sustained BF16 | about 3.5 s |
| Expected step time, good implementation | 4 to 6 s |
| Expected per-sample time | 1 to 2 s |

A naive per-expert Python loop adds several seconds of launch overhead per
step and could push this to 3 to 4 s per sample. The served vLLM prefill
rate of about 550 tok/s is not a lower bound here, because it carries the
indexer, KV writes, and small-batch kernel inefficiencies that training does
not.

What the range implies:

| Run | Samples | At 2 s/sample | At 4 s/sample |
| --- | ---: | ---: | ---: |
| Feasibility and equivalence gates | 50 | minutes | minutes |
| Arm C vs arm D, 250 steps, batch 16 | 4,000 | 2.2 h per arm | 4.4 h per arm |
| Winning arm to 1,000 steps, batch 32 | 32,000 | 18 h | 36 h |
| Baseten-scale, 2 epochs of 66K | 132,000 | 3 days | 6 days |

At the assumed rates, a pilot would take a night per arm and the larger run
would occupy the service for days. These are conditional scheduling scenarios;
neither local training feasibility nor either sample rate is established.

## Gates

Each gate is a hard stop. Nothing later runs until the earlier gate passes.

- **G0, equivalence.** Load Vision-Exp weights into the harness. On the
  repository's OCR, chart, and two-image fixtures, greedy answers must match
  the accepted Vision-Exp service exactly and per-position top-1 logits must
  agree on at least 99% of positions. Repeat text-only against the 0731
  service. This validates the torch ports of attention, compression,
  hyper-connections, hash routing, and sentinels before any gradient is
  trusted.
- **G1, memory.** Load 0731 weights, run forward and backward on micro-batch
  4 by 512 with the repository's memory sampler active. Peak device memory
  under 110 GiB on both ranks, zero swap-out, zero OOM events.
- **G2, step time.** Ten timed steps. Under 4 s per sample proceeds as
  planned. Between 4 and 10 s per sample runs the 250-step A/B locally and
  moves the 1,000-step run elsewhere. Over 10 s per sample stops and profiles
  before any scheduled window.
- **G3, gradient hygiene.** Gradients reach only the allowlisted parameters,
  are finite, and a single-sample overfit run drives the answer loss toward
  zero within 50 steps. A loss that does not fall on one sample means the
  harness, not the data, is broken.
- **G4, pilot result.** The A/B criteria from the fine-tuning plan: arm D
  beats arm C at matched budget and beats remap-only arm B on held-out
  accuracy.

## Windows and operations

Every training window displaces the served model, because both Sparks hold
the checkpoint. Rules:

1. Capture only the small reference fixtures needed for equivalence initially.
   The teacher corpus follows Q0 and the remap image-dependence gate. Cache
   tower outputs, not final aligner outputs, when training the complete aligner.
2. Stop head then worker. Run the window. Restore the captured accepted
   Vision-Exp service, worker then head, and verify both ranks and vision/text
   before handing the cluster back. Keep immutable v0.1.1 as emergency rollback.
3. Checkpoint the aligner and optimizer every 25 steps. The state is under
   1 GB, so any window can end early and resume.
4. Log per-step loss, grad norm, step time, and both ranks' memory to the
   run directory, following the existing `.local/results` layout.
5. Never run training and serving concurrently on the same rank.

Illustrative sequence, not a booked outage: replace these lengths with G2
measurements and the main plan's bounded-window/recovery contract before launch.

| Window | Work | Length |
| --- | --- | ---: |
| 1 | G0 to G3 | 3 to 4 h |
| 2 | Arm C, 250 steps | one night |
| 3 | Arm D, 250 steps | one night |
| 4 | Winner to 1,000 steps, resumable | 1 to 2 nights |

## Considered and not recommended

- **Zeroth-order tuning against the live vLLM server.** A rank-256 residual
  could be optimized from answer log-probabilities with no training stack and
  no downtime, and it would train against the true W4A4 target. It cannot
  reach the 54.5M-parameter aligner and needs tens of thousands of forward
  pairs. Keep it as a last resort if the harness cannot be built.
- **Pipeline parallel instead of tensor parallel.** Simpler collectives, but
  the reference implementation already ships tensor parallel and its
  per-layer all-reduces are about 16 MB each at this batch size, well under a
  second per pass on the dedicated fabric.
- **Truncated-model training.** Running only the first few layers cannot use
  the answer loss and adds nothing beyond the cached-aligner-output pilot
  already in the fine-tuning plan.

## Risks

1. **Porting effort is the long pole.** G0 is the deliverable that proves the
   port. Budget engineering time for it before booking cluster windows.
2. **Unified-memory pressure.** The 103 GiB estimate is close to the
   documented serving floor. G1 exists for this; micro-batch 2 is the fallback.
3. **Expert loop overhead.** If G2 is slow, sort tokens by expert and use
   grouped matmul where the platform supports it before touching anything
   else.
4. **Numerics gap to W4A4 deployment.** Covered by Stage 4 attribution and
   optional fake quantization.
5. **Fabric and NCCL.** The harness uses the same NCCL path as serving. Both
   ranks must remain free of NCCL, allocator, and CUDA errors across a whole
   window, the same rule as the serving gates.

## Deliverables

1. `training/` harness derived from the DeepSeek reference with the nine
   changes above, plus CPU tests for the dequantize-on-the-fly function and
   the image mechanics.
2. G0 equivalence report against the accepted Vision-Exp service.
3. G1 and G2 measurement JSON for both ranks.
4. Resumable aligner checkpoints with provenance manifest.
5. Pilot A/B result feeding the fine-tuning plan's Phase 5 gate.
