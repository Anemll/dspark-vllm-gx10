# Plan: validate W4A4 vision, then conditionally remap NVIDIA 0731

Date: 2026-09-13. Status: Q0 executed; vision passed, DSpark performance rejected.
Branch: nvidia-nvfp4-0731-vision-transplant.

The [research report](nvidia-0731-vision-remap-research.md) explains the evidence
and alternatives. This plan specifies executable gates; proposed experiments
and budgets below are not completed measurements.

## Objective and tracks

The immediate delivery goal is useful W4A4 vision without text-speed regression.
Test the ready-made full Vision-Exp conversion before building a training stack.
The NVIDIA-0731 remap experiment remains available when that exact language
checkpoint is required or when it offers a measured advantage.

| Track | Language checkpoint | Work | Priority and gate |
| --- | --- | --- | --- |
| Q: ready-made full Vision-Exp NVFP4 | msuiche Vision-Exp conversion and its preserved draft | Integrity/value audit, vision wrapper, canary; recalibration only if indicated | Executed: vision pass, DSpark performance reject; scale/MTP diagnosis next |
| R: remap | NVIDIA 0731, original draft | Drift audit, image-space diagnostics, mapping, optional aligner SFT | Conditional after Q; corpus waits for useful image dependence |
| P: existing 0731 projector | Original 0731 control, then NVIDIA 0731 | Integrate the FlyCockpit tower/projector and test transfer | Secondary zero-training comparison; integration is required |

Track Q does not satisfy an exact NVIDIA-0731 checkpoint requirement. Label
every result by language checkpoint; do not call msuiche an NVIDIA-published
0731 model. Parallel work means independent preparation; only one experiment
owns the two-Spark service at a time. The research report records the public
sources and remaining uncertainty for each candidate.

Q0 is complete. The [TP=2 result](../benchmarks/results/msuiche-vision-nvfp4-tp2-20260913.md)
passed target-only and DSpark visual correctness and image-dependence gates, but
failed matched C1 by 64.7% because DSpark acceptance fell to 0.318%. Track Q is
therefore not promotable with its current preserved MTP path. Scale/MTP
compatibility diagnostics now precede any projector training.

## Evidence and identity contract

Use the existing rejected-transplant logs and valid matched text canary until
a relevant fingerprint changes. The earlier embedding cosine/diagonal-fit
numbers have no located reproducible artifact and are excluded from decisions.

Pin in each experiment manifest:

- source repository, revision, config/index/tokenizer hashes and weight integrity
  evidence for original 0731, NVIDIA 0731, Vision-Exp and msuiche;
- tower, aligner, sentinel, bias and draft tensor names and source identities;
- for Track R only, merged tokenizer label at **ID 129264**: NVIDIA's image2
  label becomes the official deepseek_image label; preserve the other seven differing labels;
- for Track R only, the corrected transplant tokenizer SHA-256
  e4837c71e00fcfbe089d2d91ebdcb7974aa5531847e3cbe6704668dc1e928b9c;
- for Track Q, the unchanged official Vision-Exp tokenizer and template;
  tokenizer.json SHA-256 c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b
  and tokenizer_config.json SHA-256 6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547;
- runtime image digest on both ranks, source revisions, adapter identity,
  precision per component, context, scheduler limits and DSpark settings;
- prompt/template, image hashes, preprocessing, cache mode, seeds, metrics,
  parsers, dataset splits, evaluator commit and all planned comparisons.

The baseline result and exact source revisions are recorded in the
[transplant report](../benchmarks/results/nvidia-0731-vision-transplant-20260913.md).
The [new header audit](../benchmarks/results/vision-remap-metadata-20260913.json)
supports memory/aligner accounting only. Re-score the saved failed answer
offline; do not reload the old failed service merely to recreate that evidence.

## First execution tranche — Q0 ready-made W4A4 vision

The first canary candidate is
[msuiche/DeepSeek-V4-Flash-Vision-Exp-NVFP4](https://huggingface.co/msuiche/DeepSeek-V4-Flash-Vision-Exp-NVFP4)
at a807708335b28da39b629c38377b3d169827b360. Its published receipt reports
42 shards / 176.483 GB. Our
[public metadata/tokenizer preflight](../benchmarks/results/msuiche-vision-nvfp4-preflight-20260913.json)
was followed by the completed local weight and GPU gates reported in the
[Q0 result](../benchmarks/results/msuiche-vision-nvfp4-tp2-20260913.md).

### Q0.1 — stage and resolve the runtime contract

- Download to a new candidate directory on the designated TB36 model archive.
  Check mounted-volume identity, free space, existing partial/cache files and
  projected transfer duration first. Reserve payload plus staging overhead
  and the existing 20 GiB free-space floor; never overwrite accepted weights.
  Pin revision and verify every shard's published integrity metadata. Bound
  the transfer using measured progress and record its time separately.
- Both tokenizer files are present and hash-identical to pinned official
  Vision-Exp. Verify local copies against the hashes above; do not substitute
  the NVIDIA transplant tokenizer. Pin image processing and template files too.
- Keep the full vision artifact. The published config still declares
  DeepseekV4ForCausalLM, but this branch's DeepseekV4 architecture convertor
  already routes configs with vision_n_layers > 0 to
  DeepseekV4ForConditionalGeneration in memory. Verify that route in source,
  component tests and both-rank startup logs; do not mutate the checkpoint
  config, strip vision tensors or install unrelated steering.
- Stage immutable node-local weights before the test, using the approved bulk
  transfer path. Serving/training must not silently load shards over the archive
  network. Keep accepted Vision-Exp and release rollback artifacts available.

### Q0.2 — extend the hybrid audit, with value-level checks

Add an optional msuiche checkpoint to benchmarks/audit_dsv4_hybrid.py without
breaking its existing three-model mode. Keep metadata-only, sampled-value and
full-streaming equality modes distinct and budget each separately.

1. Compare msuiche's complete non-target-expert inventory and streaming tensor
   hashes against accepted Vision-Exp: vision/aligner, four sentinels, all
   routing biases, embeddings/head, attention/shared experts, norms and draft.
   Shard-level hashes cannot compare checkpoints with different sharding.
2. Check all target expert layouts and scale names against NVIDIA's supported
   schema, including group size 16, global/input scales and excluded MTP.
   A shape/dtype match is not numerical equality or loader validation.
3. Decode source MXFP4 GS32 and candidate NVFP4 GS16 in validated CPU routines.
   Compare effective scale products and logical values for all three projections
   of one fixed expert first. Then sample deterministic blocks across layers,
   experts and exponent extremes; record coverage and all mismatches. Test
   original 0731 versus NVIDIA 0731 separately from Vision-Exp versus msuiche.
   Check packed nibble identity separately from floating-point value equality.
4. One matching expert authorizes broader checking, not a full-checkpoint
   exactness claim. For that claim require all relevant blocks to be verified
   directly or by validated exact-conversion checks over every scale and packed
   payload. Record lossy/saturated blocks, unexpected tensor changes and NaNs.
5. Audit input-scale provenance separately. The msuiche card names NVIDIA's
   non-0731 DeepSeek-V4-Flash-NVFP4; do not relabel it 0731 calibration without
   checking source revision and values. Verify fused w1/w3/global scales and
   any load-time requantization against the actual loader/kernel path.

### Q0.3 — bounded TP=2 functional and text canary

Use the existing candidate-load contract and accepted Vision-Exp rollback.
Reuse a baseline only if runtime, prompts, precision/KV, scheduler and speculation
fingerprints match. If a new target-only control is necessary, include its
load/test cost in the same recorded budget before starting.

1. Target-only first: known-answer OCR with misleading text-prefix control,
   paired image swaps/blank control, and the canonical C1 text gate. Require
   correct fixture values, image-dependent answers and no new errors. Verify
   actual vision-wrapper and W4A4 expert-kernel selection in both-rank logs.
2. Only after target-only success, repeat these checks with the candidate's
   preserved official Vision-Exp draft using DSpark K5. Keep the same draft
   policy for the control; record acceptance and normal streaming/tool behavior.
3. Require no repeatable C1 decode regression above 3% under matched settings.
   This is a canary, not the full C1–C4 or powered vision acceptance matrix.
   Restore the accepted service after the window unless separately promoted.
4. If it passes, the no-training route is viable. Continue the existing broad
   quality, text-speed, stability and long-context gates before promotion.
   A hybrid must then justify its extra complexity against Q0 with measured
   quality/performance benefit, unless exact 0731 is the explicit requirement.
5. If it fails, preserve evidence and restore. Check image plumbing, weight
   identity and loader/kernel behavior before attributing the failure to
   activations. A separate diagnostic may compare matched W4A16 and W4A4
   execution and measure clipping by expert. If that isolates activation error,
   recalibrate image-bearing input scales as the next single-variable change.
   No aligner SFT or 2K corpus is required for that calibration.

## Conditional Track R, Step 1 — numerical audits and evidence artifacts

Deliver one reproducible audit/fitting tool with a JSON report. Record its
revision, seed, tensor/block samples, hashes, dtype/scale decoding, elapsed
time, peak memory and read volume. Preserve failures as separate artifacts.

### Embedding audit

Verify original-versus-NVIDIA input embeddings over every vocabulary row.
Regenerate the previous embedding comparison, including identity and diagonal
fit controls, with exact train/validation/held-out row IDs.

For fitting, use at least 32,768 shared, non-special vocabulary anchors.
Reserve disjoint validation and final held-out anchors. Stratify numbers,
punctuation, code-like pieces and scripts using a pinned classifier. Token
frequency strata require an actual pinned frequency source; token ID order
must not be described as measured frequency.

### Layer drift

Compare original 0731 with Vision-Exp across all 43 target layers:

- attention projections and compression/indexer components;
- shared FFN and deterministic routed-expert samples;
- routing gates, norms and Hyper-Connection parameters;
- draft layers separately.

Start with four fixed experts per layer and scale-aligned blocks spanning each
matrix. Record sampling coverage and expand only if informative. Decode FP8
with its scales and MXFP4 E2M1 nibbles with E8M0 scales before calculating
relative L2/cosine. Report absolute error for near-zero tensors. Validate the
decoder against the pinned reference; a packed-byte distance is not a weight
distance. No full checkpoint dequantization is needed for the initial audit.

Compare layer drift with regenerated embedding drift, but do not invent a
universal drift cutoff that proves vision cannot transfer. Large distributed
changes lower confidence in the map and increase Track Q's priority.

### Text-subspace coverage of actual image vectors

Use 32 diagnostic images spanning documents, charts, photos, simple geometry,
and blank/image-swap controls. Cache donor tower and aligner outputs without
creating the 2K teacher corpus. Obtain them offline or in a bounded window on
the accepted reference; any server instrumentation follows the service gate.

Fit PCA on centered donor fit embeddings only. Report image and held-out-text
residual energy at fixed ranks and at 90/95/99% text-variance ranks, vector-norm
quantiles, mean shifts, and the singular spectrum. Use uncentered/raw norms as
an additional diagnostic.

A full-rank vocabulary span cannot diagnose out-of-distribution directions.
Compare effective principal subspaces instead. If image residual energy at the
99% text-variance rank exceeds twice the held-out-text residual, flag weak
anchor support; do not escalate to a general affine fit solely to lower text
error. Prefer the constrained rotation/identity behavior in unsupported
directions and let the canary decide.

## Step 2 — fit and screen the map

Candidates: identity, centered norm correction, orthogonal Procrustes, and a
regularized affine map. Select regularization on validation anchors only.
Evaluate the selected map once on held-out anchors.

Predeclared offline gate:

- at least **5% relative held-out MSE reduction** versus identity;
- paired bootstrap 95% lower bound on MSE reduction above zero;
- no predeclared token stratum worsens by more than 1% relative MSE;
- no non-finite values, transform condition number above 2, or image/sentinel
  p95 norm change outside 0.8–1.2 times the original norm.

These are conservative engineering thresholds, not predicted effects. If no
map passes, stop text-anchor mapping and retain identity only for the routing
diagnostic. Freeze thresholds before reading fitting results.

Fold the selected map into aligner.w2 and its bias, applying the same map to
the four learned sentinels. Compare folded and explicit outputs using
scale-aware numerical tolerances; BF16 accumulation need not be bit-identical.

Add CPU and TP component checks for the image-only hook and bias mode.
Verify text-only requests bypass both, and that hashes include every adapter,
sentinel and routing change. Search the pinned full upstream tree and overlay
before editing runtime code.

## Step 3 — target-only routing and mapping canary

Run this exact matrix before fine-tuning:

| Arm | Image/sentinel map | Image-specific bias_vl |
| --- | --- | --- |
| A | Identity | Donor |
| B | Identity | Zero |
| C | Best passing map | Donor |
| D | Best passing map | Zero |

If no map passed Step 2, run only A/B. Bias toggles affect image positions
only; ordinary text corrections, safe early hash-layer handling and visibility
masks remain fixed. Record which target/draft namespaces change. Start with
DSpark disabled, then test the winning target configuration with the existing
NVIDIA draft at K5.

Run the five existing synthetic fixtures plus 40 diagnostic image pairs
(same prompt, different correct answers). Include OCR, charts, color/spatial,
counting and paired-image cases. Include blank controls and text prefixes
containing plausible wrong values. Log output content, format errors, real
image counts, finish reason and cache state. Clear encoder/prefix caches
between variants or prove their keys include the adapter/bias identity.

The four-arm matrix chooses a candidate; confirm it on 20 additional pairs
that were not used for selection. Gate for investment in training:

- both images answered correctly on at least 10/20 confirmation pairs;
- successes span at least three content categories;
- no more than one blank control invents the requested structured values;
- no non-finite output, image-path error, or text-path change.

This is an initial dependence gate, not deployable vision quality. The five
known-answer fixtures must all pass before eventual promotion; their failure
here is recorded while the diagnostic matrix completes within its budget.
Abort immediately for runtime, integrity, or stability failures.

If all arms fail the confirmation gate, stop this donor-remap training path.
If identity plus zero bias works but the map does not help, continue from that
evidence and make no claim that remapping succeeded.

## Conditional Track Q repair — recalibration or fresh conversion

Use this only if Q0's integrity/layout gate requires a fresh artifact, or a
matched diagnostic identifies activation-scale mismatch. Prefer input-scale-only
recalibration when weights are verified exact; keep those weights unchanged.

1. Pin NVIDIA Model Optimizer's V4 calibration/export revision. Audit the
   current cast_mxfp4_to_nvfp4 path, scale layout, fused w1/w3 scale convention,
   and runtime compatibility against the installed image.
2. Inventory source and temporary storage. Start with a small set of expert
   blocks spanning layers and exponent ranges. Record exact-cast coverage,
   fallback blocks, dequantized error and kernel parity before any full export.
3. Prepare multimodal calibration. The existing V4 script sends input IDs;
   extend/verify the vision input path, masks, routing and actual forward
   coverage. Collect representative text and image activations, including
   w2/intermediate activations and rarely routed experts.
4. Keep calibration inputs disjoint from quality tests. No answer labels or
   SFT corpus are required. Do not reuse NVIDIA text-only activation scales as
   validated Vision-Exp scales.
5. Calibrate then export into a fresh candidate directory, preserving all
   untargeted language, vision and draft tensors and tokenizer/config semantics.
6. Report exact/lossy weight-cast block counts, saturation and any activation
   scale fallback. Test official Vision-Exp versus its converted copy first,
   then publish a separately labeled comparison with NVIDIA 0731.

The cast changes expert weight storage; W4 activation behavior still changes.
A lossless weight cast alone cannot establish quality or text-speed parity.
Use the same quality and runtime gates as Track R. A full export starts only
after calibration capability, storage and component timings predict completion
within the recorded budget.

## Parallel Track P — test the existing 0731 projector

Inventory FlyCockpit's pinned DeepEncoderV2 tower, architecture
details, projector, tiling, preprocessing, insertion ID 129279 and routing.
Keep repeated image IDs through the early hash-routed layers and preserve the
adapter's learned separator row. Do not add the official sentinel block or
donor image-routing biases. Its 257/769/1281 token layouts require independent
shape and memory gates; they are not the official tower's 384-token budget.
The reference does not use our official donor's image boundary. First reproduce
its own original-0731 image controls, then change only to NVIDIA 0731 weights.
This isolates quantization compatibility. It is a zero-training comparison;
runtime integration and quality verification are real work.

## Step 4 — teacher corpus after the dependence gate

Create 2,000 train and 250 validation examples only after Step 3 passes.
Use licensed training splits, synthetic samples and user-authorized images,
with source/hash-based splits and perceptual duplicate grouping. Exclude every
final acceptance image and external test/dev image from training/calibration.

Suggested mix: 30% OCR/documents/UI, 20% general VQA, 15% charts/tables,
15% counting/spatial, 10% multi-image and 10% hallucination/negative controls.

For each example store image/region identity, exact prompt, verified ground
truth where available, official Vision-Exp answer, normal/truncated finish,
usage and runtime identity. Optional top-k log probabilities are diagnostic
unless the provider exposes compatible teacher-forced probabilities; a few
sampled probabilities do not define full-distribution KL.

Teacher answers supplement objective labels. Record disagreements instead of
overwriting ground truth. The old 200-image holdout becomes a development
smoke set with no promotion claim.

Cache **tower outputs** for whole-aligner training. Caching only final aligner
outputs prevents gradients from reaching earlier aligner weights. Optional
cached patch/box alignment is a cheap preliminary experiment; it must improve
held-out grounding before receiving a larger budget.

## Step 5 — matched projector SFT, local feasibility first

The donor aligner is 9216 -> 4096 -> GELU -> 4096 with biases:
54,534,144 parameters. Freeze the same tower and 0731 recipient in every arm.

| Arm | Aligner initialization | Trainable set | Question |
| --- | --- | --- | --- |
| Q0 reference | Published msuiche full Vision-Exp NVFP4 | None | Can a hybrid improve on working no-training W4A4 vision? |
| T0 | Random | Full donor-architecture aligner | Random baseline |
| T1 | Donor, no map | Same full aligner | Value of transferred weights |
| T2 | Donor with selected map folded into w2 | Same full aligner | Incremental value of remapping |
| T3, optional | Frozen donor plus rank-256 residual | Approximately 2.1M residual | Restricted-capacity ablation |

Keep sentinel values, routing mode and preprocessing identical for T0/T1/T2.
Q0 has a different language backbone and is an external reference candidate,
not a matched-initialization SFT arm. Before Q0 passes its gates, call it a
candidate rather than an accepted baseline. Hybrids must meet the same quality
criteria and demonstrate an advantage over accepted Q0 unless exact 0731 is
required; do not infer that advantage from different prompts or image budgets.
Train sentinels only in separately named follow-ups. Keep routing biases fixed
in the main arms: bias_vl affects hard top-k selection only and receives no
ordinary answer-loss gradient. Any routing-bias training needs an explicitly
designed and validated estimator. T3 has different capacity and cannot isolate
initialization.

Use ground-truth answer cross-entropy, with optional validated teacher
distillation and patch/geometry regularization. Teacher loss may never replace
objective validation. Use the same examples, loss terms, trainable count,
optimizer schedule and effective batch size for the main three arms.

First implement and validate the compressed frozen-weight path described in
the proposed [Spark training design](nvidia-0731-vision-spark-training-plan.md),
subject to the corrections and gates in the
[source review](nvidia-0731-vision-spark-training-review.md). A local pilot is
plausible, not yet demonstrated. Use the NVIDIA block-16 NVFP4 scale contract,
not the official donor's block-32 MXFP4 format. Test component gradients and
TP equivalence before a bounded integrated memory/timing probe. External
hardware is a fallback if the validated local stack cannot meet the budget.
Merely disabling weight gradients does not remove the need to backpropagate
through the recipient. Do not retain dequantized full-model weights for backward.

Initial experiment: 250 optimizer updates per arm, validation every 25.
Cap sequence/output length and effective batch size before comparing arms.
Continue an improving candidate to 1,000 updates only within its predeclared
budget. Report examples/tokens/updates and accelerator-hours to a fixed
validation target. Confirm promising initialization effects across three seeds
before claiming a reliable convergence advantage.

## Memory and compute accounting

The [header audit](../benchmarks/results/vision-remap-metadata-20260913.json)
and NVIDIA card distinguish approximately 304B logical / 13B active parameters
from stored bytes:

| Item | Footprint |
| --- | ---: |
| Original 0731 mixed-precision shards | 166.887 GB |
| Official Vision-Exp mixed-precision shards | 167.819 GB |
| NVIDIA 0731 NVFP4 shards | 175.551 GB |
| Approximate 304B weights in BF16 | 608 GB / 566.2 GiB |
| Target routed experts alone in BF16 | 554.05 GB / 516 GiB |
| Full donor aligner in BF16 | 109.1 MB |
| Generous combined two-Spark memory upper bound | 256 GiB before system use |

Resident full-BF16 SFT is excluded on two Sparks even when the draft is omitted.
That bound does not exclude compressed frozen weights with per-operation
dequantization for input gradients. Investigate that local path first, using
the training review's correctness and resource gates. The NVIDIA checkpoint
is 175.551 GB before the transplanted components; 167 GB describes the original
DeepSeek formats. Shard bytes divided by two are not exact resident memory.

Account for promoted/replicated tensors, checkpoint boundaries, temporary BF16
weights, head/loss buffers, communication, optimizer state and host memory.
No local fit or seconds-per-sample claim is established yet. External hardware
remains the fallback for failed capacity, correctness or timing gates. PTQ
calibration and cached-feature fitting have separate memory behavior.

## Quality evaluation and statistical decision

1. **Smoke:** five repository fixtures, contract/error cases and development
   images. Mandatory exact values/types at promotion, normal completion and
   recorded formatting failures.
2. **Selection:** validation set only, including paired image swaps and blank
   controls. Use it for adapter/routing/hyperparameter selection.
3. **Final:** sealed, source-group-disjoint, pinned external subsets and official
   metric implementations. Evaluate once after candidate selection.

Core reading endpoints: TextVQA accuracy, DocVQA ANLS and ChartQA relaxed
accuracy. Start with at least 1,000 independent image groups per endpoint,
subject to dataset availability. Report OCRBench separately. Supplement with
MMMU-Pro, MathVista, a specifically named/pinned multi-image benchmark, POPE,
and local grounded UI/spatial tests. Do not call custom subsets full official
benchmark scores. Availability/licensing and evaluator pins are a preparation
deliverable, not assumed complete.

Promotion criteria:

- all existing known-answer fixtures pass;
- per core reading endpoint, paired score difference
  delta = candidate - 0.90 * official_reference has a lower confidence bound
  above zero;
- use image-group resampling and one-sided 98.33% bounds for the three core
  endpoints (Bonferroni family error at most 5%); publish raw scores and deltas;
- compute the required sample size before final testing from development
  variance, aiming for 80% power at a predeclared five-point advantage above
  the allowed non-inferiority boundary; increase the fixed final sample before
  opening it if 1,000 groups is insufficient;
- require a one-sided 95% lower bound of at least 95% on correctness of both
  members of objective image-swap pairs, initially 200 independent pairs;
- require the upper one-sided 95% bound on hallucination-rate increase to be
  at most two percentage points, using a powered fixed set of negative images.

A 200-image smoke set with 60 OCR images cannot establish those claims.
Repeated questions from one image are not independent samples. Inconclusive
intervals mean no promotion; do not repeatedly extend a final test until it
passes. The 90%-of-reference margin is an engineering target, not equivalence.

Score objective labels separately from teacher agreement. Pin answer
normalization, ANLS/relaxed accuracy implementations, JSON/type handling,
timeouts and exclusions. Preserve raw outputs and score parser failures as
failures under the fixed policy.

## Deployment and speed gates

First pass target-only quality, then verify DSpark K5 preserves it with the
unchanged matching draft. Check image conditioning through target verification
and record draft acceptance. Deterministic checks and distribution-aware
comparisons should account for legitimate numerical variation; no image
quality failure may be excused as speculation.

For Track R, compare NVIDIA text-only and the selected NVIDIA hybrid with
matched canonical content prompts, C1/C2/C3/C4, seeds, runtime, FP8 KV,
scheduler limits, context and output limits. A repeated text slowdown above 3%
fails the existing requirement. For Track Q, first compare native versus
converted official Vision-Exp, then report the NVIDIA comparison separately.

Run streaming, tool calls, short/long prefill and 65K/100K boundary checks.
Report per-request decode and aggregate throughput, TTFT, server prefill timing
where available, DSpark acceptance, memory, NCCL and thermal state. Vision
timing is separate by resolution, image count and concurrency; mark warmed
image-cache runs. Slower vision is acceptable if quality/stability pass.

Full content/vision sweeps and Spark Arena follow only after these gates.
Arena C10 temporarily uses ten active sequences with observed running counts;
restore the normal four-sequence setting afterward.

## Budgets and service contract

The following are initial caps, not elapsed-time promises. Before starting a
stage, project duration from measured I/O/step/load rates and shorten the scope
if it cannot fit. Record any revised budget before launching the larger job.

| Stage | Initial wall-clock cap | Service outage |
| --- | ---: | ---: |
| Metadata, numeric samples and map fitting | 30 min | 0 |
| Q0 download, complete integrity and node-local staging | Set from transfer/I/O probe before launch; exclude from service window | 0 |
| Q0 target-only + K5 canary, including recovery | 75 min; revise before launch if measured loads cannot fit | At most the recorded window |
| Tower/aligner diagnostic capture | 10 min | 0 if offline |
| Q/P code and integration feasibility review | 45 min each | 0 |
| Remap/routing candidate window including recovery | 75 min | At most 75 min |
| Q representative component test | 15 min | 0 on separate hardware |
| Training component checks, without full checkpoint | 30 min initial probe | 0 |
| Integrated local training feasibility, after component gates | 75 min including recovery; revise before launch if measured loads cannot fit | At most the recorded window |
| Initial matched 250-update SFT arms | Set from measured optimizer-update time, validation and recovery | Scheduled local windows, or 0 on separate hardware |
| Teacher generation / full export / final evaluation | Set from short probes before launch | Scheduled separately if needed |

For the remap window, reserve at least 25 minutes for restoring the accepted
service, increasing that reserve if current load measurements demand it.
If four variants require repeated full model loads and cannot fit, finish the
safe adapter-switch design before scheduling the window. Do not silently
multiply the outage.

Preserve accepted Vision-Exp and immutable release rollback artifacts. Acquire
the experiment lock, capture logs/config/image IDs, and consult the standing
read-only advisor before service changes. Build once, transfer the identical
image over the approved fabric, stop head then worker, start worker then head.

Abort on either-rank failure, non-finite results, CUDA/NCCL errors, missed
progress deadlines, swap storm, lost rollback readiness, or less than 20 GiB
free storage after planned writes. Recover and verify both ranks, API,
streaming and one known-answer image request. Record total elapsed time,
restoration time and final running model.

## Deliverables and current next step

- Pinned Q0 snapshot, official tokenizer hashes, value/passthrough audit and
  target-only/K5 canary report; conditional scale-only recalibration.
- Reproducible audit/map script and immutable numeric JSON if Track R proceeds.
- Image-subspace activation artifact and report.
- Affine-fold and image-only routing controls with TP/text-path tests.
- Completed 2 x 2 canary with untouched confirmation pairs.
- Q calibration/export package and P compatibility inventory.
- Conditional teacher corpus, full-aligner SFT configs and training evidence.
- Powered quality reports, text/vision speed evidence and explicit model card.

The next implementation tranche is Q0 staging, audit and the bounded canary.
Keep the P integration inventory independent. Remap Steps 1–3 and offline
training component checks become conditional follow-ups; the 2K corpus and
full-model-backed aligner training still require the dependence gate. The
frozen language weights never become trainable in this plan.
