# Vision adaptation for DeepSeek V4 Flash 0731 NVFP4: research and decision

Date: 2026-09-13. Revised after Q0 execution. Implementation, budgets, and acceptance
criteria are in the [execution plan](nvidia-0731-vision-remap-finetune-plan.md).

## Decision

The already published **msuiche/DeepSeek-V4-Flash-Vision-Exp-NVFP4** candidate
does preserve real vision without training, but it is rejected for the current
DSpark deployment. The bounded TP=2 test passed 10/10 visual canaries and 4/4
same-prompt image swaps both target-only and with DSpark K5, while matched C1
fell from 40.86 to 14.41 tok/s because draft acceptance collapsed from 32.73%
to 0.318%. The complete result is [recorded here](../benchmarks/results/msuiche-vision-nvfp4-tp2-20260913.md).[^18]

Keep remapping NVIDIA 0731 as a conditional track, particularly if that exact
language checkpoint is required. Its drift, image-vector and routing gates
still precede corpus construction and matched random/donor/remapped-aligner
training. A passing Vision-Exp conversion does not become NVIDIA 0731; record
that distinction rather than treating the two targets as interchangeable.

FlyCockpit plus NVIDIA 0731 remains an independent control. Projector/remap
training is not the next action because Q0 proved that the Vision-Exp relationship
survives the weight transcode. First isolate target-only text behavior and the
NVFP4-target/MXFP4-MTP mismatch, then recalibrate activation scales with DSpark
acceptance as an explicit validation metric.

The Q0 audit verified all 42 shard hashes on both ranks, byte equality for all
6,585 non-expert tensors, and exact decoded values for 100,663,296 logical values
across four sampled experts. It also found that 32,421 of 33,024 activation
input-scale tensors differ from NVIDIA's base export and 32,878 differ from its
0731 export. Exact weight relayout therefore does not imply calibration identity.

## What the failed transplant establishes

The [September 13 test](../benchmarks/results/nvidia-0731-vision-transplant-20260913.md)
loaded the hybrid, preserved matched C1 text speed within the 3% gate, and
processed a request accounted for as multimodal. Its first OCR answer copied
values from the text prefix and got all three requested image fields wrong.

That establishes a correctness failure on one image. It does not establish
complete visual blindness: the run stopped before image-swap controls or
target-only versus DSpark comparisons. Request accounting also does not prove
every vector, attention mask, or expert-routing choice is semantically correct.

DeepSeek says Vision-Exp adds visual modules and undergoes continued training.
The previous local audit found differences in all sampled shared language
tensors, as well as normalization and prediction-layout differences.[^3][^11]
There are at least three competing explanations:

1. Input representation drift that a map or adapted aligner could compensate.
2. Image-routing biases selecting donor-trained experts whose behavior differs
   in the 0731 recipient.
3. Visual interpretation learned throughout the continued-trained attention,
   FFN, and expert layers, requiring more than an input bridge.

Embedding cosine alone cannot distinguish these explanations. Small average
weight drift can still produce large functional changes, particularly around
sparse expert selection. Conversely, a large drift does not mathematically
prove that a trained projector cannot compensate. Layer drift is a diagnostic;
the image-conditioned canary remains the functional decision.

## Evidence status and reproducibility

The former six-number embedding-cosine table and diagonal-fit result were
reported in the conversation, but a search of the repository and local result
files did not locate their reproducible artifacts. They are withdrawn as
decision evidence until a versioned audit regenerates them. No map choice or
training budget may rely on those numbers.

The committed
[September 6 audit](../benchmarks/results/nvidia-vision-hybrid-preflight-20260906.json)
supports sampled tensor-byte comparisons, not full vocabulary equality or a
layer-wise numeric drift estimate. The new
[header audit](../benchmarks/results/vision-remap-metadata-20260913.json)
records source/config/index identities, payload formats, storage sizes, aligner
shapes, and memory arithmetic. It reads metadata, not weight values.

The first implementation deliverable is the candidate integrity/transcode audit
described below. Embedding statistics and layer drift remain required before
mapping. Every numerical report needs the script revision, sampling seed,
exact tensor/block selections, scale interpretation and data hashes. A proposed
artifact name in a plan is not an existing measurement.

## Missing diagnostics before mapping

### Numeric drift across the language network

Compare original 0731 against Vision-Exp for every shared target layer,
separating attention, shared FFN, routed experts, routers, norms, and
Hyper-Connection parameters. Report draft layers separately. Use relative
Frobenius L2 and cosine on corresponding logical values; describe the
denominator and explicitly handle near-zero tensors.

These checkpoints are mixed precision. Safetensors headers reveal where bytes
are stored but do not supply dequantized values. FP8 comparisons must apply
the associated block scales; MXFP4 expert comparisons must unpack E2M1 values
and apply E8M0 scales. Raw I8/U8 differences are not numeric weight drift.
Validate the decoder against the pinned reference before trusting these scores.

Start with deterministic scale-aligned blocks from every layer and a fixed
expert sample, then expand only informative families. Report sampling coverage
and uncertainty; do not call sampled block drift an exact whole-tensor norm.
A full scan of multiple large checkpoints is a separately budgeted I/O job.

Where affordable, compare activations on fixed text inputs and image-plus-text
inputs. Weight distances alone cannot locate the origin of visual capability.

### Does the text-derived map cover image vectors?

Collect donor tower/aligner activations for a small diagnostic set of images,
with no generated labels needed. Fit PCA/SVD on centered donor vocabulary
embeddings using the fit split only. For retained principal components U_k,
measure:

    residual_fraction(k) =
        sum ||(z_image - mu_text)(I - U_k U_k^T)||^2
        / sum ||z_image - mu_text||^2

Report the curve for fixed ranks, and ranks retaining 90%, 95%, and 99% of
text variance. Compare it with held-out text-vector residuals, norm quantiles,
mean shift, and conditioning in weakly supported directions.

A vocabulary matrix may have full rank in 4096 dimensions; its complete
mathematical span is then uninformative. The concern is low-density directions
and poor conditioning in the principal subspace, not necessarily an absent
literal span. Fit directions that lack text support conservatively and assess
them on actual image activations.

A predeclared mapping gate requires at least 5% lower held-out embedding MSE
than identity and a positive paired 95% confidence bound on the improvement.
Reject destructive norm changes or poorly conditioned transforms. This is an
engineering screen, not a theorem about image quality. Full numerical criteria
and the fixed fitting split are specified in the execution plan.

## Baseten and the available precedents

The remembered "Besten" source is almost certainly Harry Partridge's Baseten
article, "GLM 5.2 with vision." It describes training a 49.5M projector between
a frozen MoonViT encoder and a frozen language model, using 66K image/QA
samples and later projector training for reasoning.[^1][^2] This supports the
feasibility of learning an image bridge; it does not demonstrate that our
text-anchor remap is a good initialization.

The Baseten model card explicitly lists a 744B total / 40B active text
backbone. The same page's automated Safetensors summary displays 381B.
Use the architectural statement with attribution. The two displays are
inconsistent and have not been reconciled by a logical parameter recount.
Packed 4-bit tensors can store two logical weights per byte; automatic counts
depend on the quantization parser. Hugging Face's parser contains explicit
packing corrections.[^2][^15] Do not substitute 381B as the architecture size.

Relevant DeepSeek work includes:

- WebBrain: MoonViT plus a routing-aware projector, using a copied
  **MJPansa/DeepSeek-V4-Flash-0731-NVFP4** backbone at
  64d64cd89bc63a66aa46506da89d7821f7491c62, not NVIDIA's published checkpoint.[^5]
  MJPansa reports 500K calibration tokens and a two-Spark TP=2/FP8-KV loading
  and generation check; this is not a speed certification.[^19]
- Fakoli's independent first-load report: image conditioning succeeded while
  dense OCR and GUI-grounding gates failed.[^6]
- Jarrelscy's continuation: warm-started WebBrain training, residual capacity,
  OCR/UI emphasis, and patch/box alignment; documented training and serving
  limitations remain.[^7]
- FlyCockpit: an approximately 865 MB DeepEncoderV2 tower plus 40 MB adapter,
  spliced using repeated ID 129279 and served through a two-Spark TP=2/RoCE
  plugin. It has no official four-sentinel block or donor bias_vl routing, but
  its adapter does include a learned separator row. Preserve that row, global
  view ordering and the trained 257/769/1281-token tiling layout.[^8][^21]
- msuiche: an existing routed-expert NVFP4 transcode of full official
  Vision-Exp. The author reports preservation of excluded components and zero
  lossy tensors, with text-only B200 validation. This is a candidate, not a
  demonstrated Spark vision result.[^18]

FlyCockpit is a promising zero-training comparison against NVIDIA 0731.
Embedding equality is only one compatibility condition: reproduce its image
preprocessing, tower, token-129279 insertion, tiling, and serving conventions.
Its published stack cannot be dropped into the official donor's token-129264
path. All image positions retain the same in-vocabulary ID through the early
hash-routed layers. First reproduce that contract, then change only the
recipient quantization. Its card's shorthand "FP8 0731" is not an all-FP8
weight claim: original 0731 includes MXFP4 experts. Matching embeddings supports
testing transfer, but does not establish W4A4 activation compatibility.

Model stitching supports testing learned bridges between frozen networks.[^10]
VILA warns that projector-only tuning has capability limitations.[^4]
Patch-alignment work and BASIC motivate auxiliary grounded supervision if a
working image path survives the initial gates.[^13][^14] These sources motivate
experiments; they do not predict a DeepSeek quality score.

## Remapping and the corrected training comparison

Fit an orthogonal map, with a regularized affine alternative, using shared
vocabulary anchors. For row-vector outputs:

    y = z T + c
    z = h W2^T + b2
    W2_new = T^T W2
    b2_new = b2 T + c

The pinned runtime aligner ends in a linear layer with bias and no following
normalization. Its output is not partitioned across tensor-parallel ranks, so
folding on the output dimension is compatible with its input-sharded second
linear. Verify the unfused/folded paths numerically under TP=1 and TP=2; BF16
rounding means bit-for-bit equality is not guaranteed.

The local donor has:

| Tensor | Shape | Stored precision |
| --- | --- | --- |
| aligner.w1.weight | 4096 x 9216 | BF16 |
| aligner.w1.bias | 4096 | BF16 |
| aligner.w2.weight | 4096 x 4096 | BF16 |
| aligner.w2.bias | 4096 | BF16 |

That is **54,534,144 trainable parameters**, approximately 109.1 MB in BF16,
confirmed by the header audit. Freeze the vision tower and recipient language
weights, but train this complete aligner in the main SFT experiment. The
language-model backward pass dominates cost; freezing most of the aligner
offers limited savings while restricting the bridge's capacity.

Use three equal-architecture arms:

- random initialization of this exact aligner;
- the donor aligner without remapping;
- the donor aligner with the selected map folded into its last layer.

Only the third versus second comparison isolates the incremental value of
remapping. Third versus first compares transferred initialization with random
initialization. Keep the same frozen tower, parameter count, sentinel state,
routing mode, examples, optimizer, and update budget. A frozen-aligner plus
rank-256 residual is optional; report its smaller capacity as a separate arm.

If sentinels are mapped in the selected canary, use the same sentinel values
in all three main SFT arms. Sentinel-training ablations follow separately.

## Conditional remap track: routing in its first functional test

Before training, run the 2 x 2 matrix: identity versus selected remap, crossed
with donor bias_vl versus zero bias_vl. Evaluate the same ordered image pairs
and objective prompts target-only. Cache keys must include adapter and bias
identities; invalidate encoder and prefix caches between variants.

Toggle only the image-specific bias branch. Keep ordinary text gate corrections,
the safe image-ID handling in early hash layers, and visibility semantics
fixed. Record target and draft bias namespaces separately; test the draft
after the target-only matrix.

Changing image bytes must change the answer in the direction required by the
ground truth. Include a prefix containing plausible wrong values to catch the
observed text-copying failure. If all four arms fail this dependence gate,
stop donor-remap training and direct effort to the two alternatives.

## First action: audit the ready-made full Vision-Exp NVFP4 candidate

Pin msuiche at a807708335b28da39b629c38377b3d169827b360. Its receipt reports
42 shards / 176,483,421,216 bytes and three expert spot checks. The full Hub
inventory includes tokenizer.json and tokenizer_config.json; the first file
listing page omitted them. Both downloaded tokenizer hashes match official
Vision-Exp at 6821d6ad3681a4b137b066b76094fa82ebd0a380. See the
[preflight artifact](../benchmarks/results/msuiche-vision-nvfp4-preflight-20260913.json).
Use those pinned official files; no tokenizer-label transplant is required.

The config declares group-16 NVFP4 for all 43 main expert layers and excludes
MTP, but still names DeepseekV4ForCausalLM. Explicitly select the installed
vision wrapper and preserve full vision tensors. Do not run strip_vision.py
for this test. Loader acceptance is a gate, not implied by compatible keys.[^20]

The card names **nvidia/DeepSeek-V4-Flash-NVFP4**, not the 0731 repository,
as the source of borrowed input scales. Their exact source revision and values
need an audit. Published validation does not establish image quality; neither
"no one has tested this" nor runtime exclusivity has been verified.[^18]

### Value-equivalence gate and what it can establish

Extend the hybrid audit with two independent comparisons:

- original 0731 versus NVIDIA 0731 NVFP4;
- accepted official Vision-Exp versus msuiche Vision-Exp NVFP4.

Compare passthrough tensor inventories, shapes, dtypes and full streaming hashes
where identity is claimed. Include attention, shared experts, router/norms,
head/embeddings, all three draft layers, tower/aligner, sentinels and image
biases. Check expert layout/scale names against the supported NVIDIA loader,
but compare expert *values* to their own source, never across training lineages.

Start with all w1/w2/w3 values in one deterministic expert, then scale-aligned
samples across layers and exponent extremes. Compare decoded E2M1 values times
effective scales: E8M0 GS32 on the source versus E4M3 GS16 times global scale
on NVFP4. Splitting each group alone is insufficient if the effective scales
are not exactly representable. Record mismatches, coverage, zero/nonfinite
handling and decoder validation. Do not compare packed signed/unsigned bytes
as floating-point distances or infer whole-checkpoint equality from one expert.

If equivalent values and unchanged non-expert/config semantics are established,
there is no *weight* approximation gap in the covered tensors. Activation
quantization/input scales become the principal intended difference. Kernel
accumulation, loader rescaling/fusion, KV precision and image plumbing remain
possible sources of functional differences; an OCR failure alone cannot prove
that activation scales caused it. The msuiche claim does not prove anything
about a separate NVIDIA 0731 export without the first comparison.

After metadata/value gates, run the bounded OCR, image-swap and matched C1
canary target-only, then with DSpark K5. Passing establishes initial W4A4 vision
viability without training; the broader quality and C1–C4 gates still precede
promotion. If a matched diagnostic isolates activation error, recalibrate
image-bearing expert inputs before changing weights or training an aligner.

## Conditional repair: calibration or a fresh Vision-Exp conversion

NVIDIA already provides DeepSeek V4 expert-only calibration and export tools.
The export has a cast_mxfp4_to_nvfp4 option that retains representable E2M1
weight blocks while rewriting scales. Activation input scales still require
calibration, and out-of-range weight blocks can fall back to a lossy path.[^16]

This materially reduces the need to build a converter from scratch. However,
it does not make the vision conversion a completed or automatically correct
operation. The current calibration script feeds token IDs. It must exercise
the official image path to measure visual expert activation ranges, including
rarely visited experts, and record fallback coverage.[^17]

Use a fresh export only if the published candidate's integrity/layout fails;
use scale-only recalibration first if activation mismatch is isolated.
Use the whole Vision-Exp source, preserving its language weights, vision tower,
aligner, tokenizer, norms, image biases, and native DSpark. Convert only selected
target experts; keep the vision and draft precision unchanged initially.
Capture both text and image activation distributions. Verify weight-cast
coverage, scale saturation, output quality, and text performance.

This track needs no supervised image answers for calibration. It still needs
representative inputs, correct multimodal calibration plumbing, storage,
runtime support, and independent quality tests.

## Training memory and where work belongs

Advertised 0731 size is approximately 304B logical parameters, with 13B active
per token.[^9] Active size does not describe resident weights. The on-disk
sizes below were rechecked from local shard metadata:

| Checkpoint | Shard bytes | Decimal GB |
| --- | ---: | ---: |
| Original 0731 | 166,886,535,336 | 166.887 |
| Official Vision-Exp | 167,819,404,368 | 167.819 |
| NVIDIA 0731 NVFP4 | 175,550,788,904 | 175.551 |

Original/Vision-Exp routed experts use packed MXFP4 with E8M0 scales, alongside
FP8 and BF16/F32 tensors. NVIDIA uses NVFP4 for target experts and retains the
source draft formats. None of these file sizes is the size of an all-BF16 model.

At 304B x 2 bytes, BF16 weights alone require approximately **608 GB / 566.2
GiB**, before activations, communication buffers, and trainable-state storage.
Even just the target's routed expert matrices require **554.05 GB / 516 GiB**.
That already exceeds a generous 256 GiB upper bound for two 128 GB Sparks.

This BF16 bound does not exclude a compressed frozen-weight training harness
that dequantizes one matrix at a time for forward and input gradients. The
new [Spark proposal](nvidia-0731-vision-spark-training-plan.md) makes a local
pilot plausible; investigate it before defaulting to external GPUs. The
[source review](nvidia-0731-vision-spark-training-review.md) identifies required
corrections to NVIDIA scale decoding, attention, gradient flow, loading and
memory accounting. Its CPU checks are not a full-model feasibility result.
Use external hardware if the corrected local stack fails its resource/time
gates. Small cached-feature alignment remains suitable for local experiments.

Cache tower outputs, not only final aligner outputs, when training the whole
aligner. Final aligner outputs suffice only for mapping/residual diagnostics.

## Data and quality decisions

Generate a small diagnostic image activation set first. Build the 2K supervised
pilot only after a canary establishes useful image dependence. Teacher answers
from official Vision-Exp supplement verified labels and should be collected
with source/image hashes, prompts, sampling parameters, and model identities.

Keep the 200-image pilot as a development smoke test. Sixty independent OCR
items have a worst-case approximate 95% sampling margin of 12.7 percentage
points. They cannot establish a narrow non-inferiority claim.

Use disjoint, pinned external subsets for promotion, initially at least 1,000
independent image groups per core reading category, with paired confidence
intervals and a power check from the development set. A nominal sample count
alone does not guarantee power. Reserve final tests until candidate selection
is complete; an inconclusive interval remains inconclusive.

The execution plan defines the metric-specific thresholds, parsing rules,
image-swap controls, DSpark checks, and text C1-C4 regression requirements.

## Recommended next tranche

1. Run a bounded target-only C1 and text-quality lane on the already verified
   msuiche bytes; Q0 intentionally used target-only as the vision-isolation gate.
2. Instrument fixed-prompt target/draft top-k overlap, hidden/logit drift, and
   per-stage DSpark acceptance for accepted MXFP4 versus candidate NVFP4.
3. Recalibrate routed-expert activation scales on pinned mixed text and image
   samples. Require restored DSpark acceptance as well as quality.
4. If scale-only work fails, test an MTP-aware conversion of the three draft
   layers. Keep FlyCockpit/NVIDIA 0731 as an independent integration control.
5. Pursue remap/aligner training only if exact 0731 remains required or these
   no-training paths fail. Do not build the 2K corpus first.

These diagnostics preserve the remap hypothesis as a testable option while
giving the delivery alternatives their appropriate priority.

## Sources

[^1]: Harry Partridge, Baseten, [GLM 5.2 with vision](https://www.baseten.co/blog/glm-52-with-vision/).
[^2]: Baseten, [GLM-5.2-Vision-NVFP4 model card](https://huggingface.co/baseten/GLM-5.2-Vision-NVFP4).
[^3]: Anemll, [NVIDIA 0731 transplant result](../benchmarks/results/nvidia-0731-vision-transplant-20260913.md) and [compatibility audit](../benchmarks/results/nvidia-vision-hybrid-preflight-20260906.json).
[^4]: NVIDIA, [Visual Language Models on NVIDIA Hardware with VILA](https://developer.nvidia.com/blog/visual-language-models-on-nvidia-hardware-with-vila/).
[^5]: WebBrain, [DeepSeek-V4-Flash-0731-Vision-NVFP4](https://huggingface.co/webbrain-one/DeepSeek-V4-Flash-0731-Vision-NVFP4).
[^6]: Fakoli, [first GPU load and vision gate](https://github.com/fakoli/anvil-serving/blob/main/docs/findings/2026-08-07-deepseek-0731-vision-nvfp4-sglang-first-load.md).
[^7]: Jarrelscy, [DeepSeek V4 Flash 0731 vision training notes](https://huggingface.co/jarrelscy/deepseek-v4-flash-0731-vision).
[^8]: FlyCockpit, [vision stack](https://huggingface.co/FlyCockpit/DeepSeek-V4-Flash-0731-vision) and [capability report](https://github.com/FlyCockpit/DeepSeek-V4-Vision-2x-DGX-Sparks/blob/main/docs/CAPABILITIES.md).
[^9]: NVIDIA, [DeepSeek-V4-Flash-0731-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4-Flash-0731-NVFP4).
[^10]: Bansal, Nakkiran, and Barak, [Revisiting Model Stitching to Compare Neural Representations](https://arxiv.org/abs/2106.07682).
[^11]: DeepSeek AI, [DeepSeek-V4-Flash-Vision-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp).
[^12]: vLLM, [Vision-Exp implementation design issue #54561](https://github.com/vllm-project/vllm/issues/54561), for the sentinel, visibility, and routing contract underlying the canary.
[^13]: Jiang et al., [Analyzing Fine-Grained Alignment and Enhancing Vision Understanding in Multimodal Language Models](https://arxiv.org/abs/2505.17316).
[^14]: Tang et al., [BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models](https://arxiv.org/abs/2508.06895).
[^15]: Hugging Face, [Safetensors parameter-count implementation](https://github.com/huggingface/huggingface.js/blob/main/packages/hub/src/lib/parse-safetensors-metadata.ts).
[^16]: NVIDIA Model Optimizer, [DeepSeek V4 conversion documentation](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/deepseek/README.md) and [NVFP4 exporter](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/deepseek/deepseek_v4/quantize_to_nvfp4.py).
[^17]: NVIDIA Model Optimizer, [DeepSeek V4 calibration implementation](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/deepseek/deepseek_v4/ptq.py).
[^18]: msuiche, [Vision-Exp NVFP4 model card](https://huggingface.co/msuiche/DeepSeek-V4-Flash-Vision-Exp-NVFP4/blob/a807708335b28da39b629c38377b3d169827b360/README.md).
[^19]: MJPansa, [0731 NVFP4 conversion and validation](https://huggingface.co/MJPansa/DeepSeek-V4-Flash-0731-NVFP4).
[^20]: msuiche, [pinned configuration](https://huggingface.co/msuiche/DeepSeek-V4-Flash-Vision-Exp-NVFP4/blob/a807708335b28da39b629c38377b3d169827b360/config.json) and [conversion receipt](https://huggingface.co/msuiche/DeepSeek-V4-Flash-Vision-Exp-NVFP4/blob/a807708335b28da39b629c38377b3d169827b360/_DONE.json).
[^21]: FlyCockpit, [vision wrapper and separator contract](https://github.com/FlyCockpit/DeepSeek-V4-Vision-2x-DGX-Sparks/blob/main/plugin/src/dsv4_vision_vllm/model.py).

Web sources rechecked September 13, 2026 where noted in this revision.
Pin mutable source revisions before implementation.
