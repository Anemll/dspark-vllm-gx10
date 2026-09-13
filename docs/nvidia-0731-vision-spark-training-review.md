# Review: compressed-weight aligner training on two Sparks

Date: 2026-09-13. Status: source review and small CPU checks, not a completed
training harness or a measured Spark feasibility result.

Reviewed: [proposed local training plan](nvidia-0731-vision-spark-training-plan.md).
This addendum records the original source review. The proposal was subsequently
amended to prioritize the ready-made msuiche candidate and conditional weight
equivalence; remaining implementation corrections below still apply to the
[main experiment plan](nvidia-0731-vision-remap-finetune-plan.md).

## Decision

Investigate the local pilot. The earlier recommendation to default to external
GPUs was too strong: a resident BF16 model is unnecessary if frozen packed
weights can be decoded for each forward/input-gradient operation and promptly
released. Training all 54,534,144 donor-aligner parameters remains in scope;
language weights, embeddings, head, tower and draft remain frozen.

However, neither the 103 GiB budget nor the 2–4 seconds/sample estimate has
been demonstrated. Several essential training paths are absent from the
inference reference. Correct these before scheduling a long outage.

## Required corrections

### 1. Load the NVIDIA target, not the donor's quantization format

The proposal's block-32 FP4/E8M0 description and approximately 167 GB footprint
describe original DeepSeek storage. The NVIDIA checkpoint audited here is
175,550,788,904 bytes and uses **block-16 NVFP4** for target experts. Its config
and index hashes match the saved preflight metadata. The index includes
weight_scale, weight_scale_2 and input_scale for target expert matrices.

Implement separate decoders: donor/original MXFP4 with E8M0 block scales,
NVIDIA NVFP4 with E4M3 block scales and FP32 global scales, and the relevant
FP8/BF16/F32 tensors. Validate nibble order, scale direction and shard boundaries
against the actual checkpoint/runtime. DeepSeek's reference linear and
converter do not acquire NVIDIA support merely by changing the directory.
See the [header audit](../benchmarks/results/vision-remap-metadata-20260913.json)
and [contract evidence](../benchmarks/results/vision-spark-training-contract-review-20260913.json).

### 2. The dense short-context path is valid with the complete attention contract

At total sequence length 512, ratio-4 compression produces at most 128 entries;
top-k 512 therefore does not select a subset. The CPU check agrees with the
reference index sets at tested lengths through 2048. This is a set-selection
check, not full attention parity. Assert start_pos=0, total token length and
valid lengths; a short query appended to a long cache does not qualify.

Remove only the indexer and its private compressor/Hadamard path. Preserve the
main compressor, ratio-4 overlap pooling, ratio-128 behavior, sliding window,
completed-group causality, bidirectional visibility within image spans, RoPE,
and the learned per-head attention sink. Window and compressed entries share
one softmax; separate attention calls added together are not equivalent.

The sink contributes to the softmax denominator, not the value numerator.
A zero-value dummy key with the appropriate per-head additive logit can express
it. Our tiny float64 test matches that identity within 1.2e-16; omitting the
sink changes the output by up to 0.269 on this synthetic sample. Retain exact
Hyper-Connection/Sinkhorn normalization order and epsilons too.

Dense mathematical equivalence does not promise bitwise parity with TileLang's
BF16 accumulation/casts. The reference also quantizes activations and cached
KV components. W4A16/W8A16 removes those operations and is a changed training
forward, not the exact W4A4 serving computation. A declared straight-through
estimator may restore quantized forward behavior; its derivative is a surrogate
and its operations/buffers are not memory-free. [Reference model][model],
[attention and Hyper-Connection kernels][kernel].

### 3. The head and TP gradient contracts need explicit implementation

ParallelHead defaults to the last sequence position. Its raw distributed
all_gather writes into new tensors without an autograd gather. Removing
inference_mode and replacing all_reduce is not enough to obtain answer-token
loss gradients. Implement shifted teacher-forced answer loss using a tested
vocabulary-parallel cross-entropy, or a differentiable gather with the correct
loss normalization. Only requested answer positions need full logits.

Define each replicated/sharded tensor boundary and its backward operation.
A column-parallel linear needs input-gradient reduction even though its
forward has no all_reduce. A conventional row-parallel mapping reduces in
forward but passes the replicated upstream gradient in backward. Blanket
replacement with an operation that sums in both directions can mis-scale or
misroute gradients under the chosen loss convention. Test TP=2 against a tiny
unsharded model, including replicated aligner gradients. [Reference head][head];
[NVIDIA's TP forward/backward mappings][tp].

For a frozen affine matrix, dX needs the packed weight/scales, not saved X.
Save references to the compressed data and its layout, never the temporary
full BF16 matrix. Ordinary autograd through a dequantized F.linear can retain
that matrix for backward and invalidate the memory budget. Any activation
quantization surrogate may need additional saved state, accounted separately.

### 4. Checkpointed blocks must be functional; loading must really stream

The inference implementation mutates compressor state, KV buffers and RoPE
views. Removing in-place quantization alone does not make checkpoint
recomputation safe. Build a stateless full-sequence training path, or pass
state explicitly with proven lifetimes. Test checkpointing on/off, consecutive
samples, image embedding insertion, and future-token leakage.

The shipped convert.py accumulates per-rank state dictionaries before writing
them. It is not a bounded streaming resharder. Replace that preparation step
with an index-driven loader/streaming converter; avoid simultaneous original,
converted and preallocated copies. Inventory reference dtype promotions,
including wo_a conversion and the FP32 head. Omit draft execution and optionally
its resident allocation during training, recording the omission; its stored
weights remain unchanged for serving. [Reference converter][convert].

### 5. Routing-bias toggles are valid; ordinary bias training is not

bias_vl changes hard top-k indices but is excluded from gathered route weights.
With the pinned Gate implementation, tiny hash and non-hash tests produce
nonzero input gradients but **bias_vl.grad is None**. Marking it trainable
does not create an answer-loss gradient. Keep donor-versus-zero as fixed
canary ablations. Bias training requires a separately specified surrogate or
other estimator, not just an optimizer allowlist. Sentinel-vector training is
a separate differentiable arm. [Reference Gate][gate].

## Memory and time: hypotheses to measure

For B=4, L=512, four 4096-wide streams and 43 saved block inputs:

| Arithmetic item | Size |
| --- | ---: |
| BF16 block-input boundaries | 2.69 GiB |
| FP32 block-input boundaries | 5.38 GiB |
| Full B x L x 129280 FP32 logits | 0.99 GiB |
| Half of NVIDIA's on-disk shard bytes, not resident size | 81.75 GiB |

Thus 6 GiB for boundaries is plausible, but not a total activation bound.
Include dequantization/cast scratch, selected expert activations, FP32 pooling
and routing, gradients through frozen layers, loss buffers, communication,
replicated tensors, allocator reservations and host usage. All sample tokens
count toward 512: image tokens, sentinels, template, prompt and answer. Two
full 384-token images need a separate larger-length gate; they cannot be used
unchanged in a 512-token training fixture.

Keep 103 GiB only as an unvalidated planning estimate after correcting the
checkpoint identity. A device-allocation threshold alone is insufficient on
unified memory: sample host MemAvailable, cgroup use, swap activity, CUDA
allocated/reserved high-water marks and both ranks. Do not double-count shared
host/device accounting. Set a measured host safety reserve before launch.

The proposal's 84 GB x three reads / 273 GB/s is 0.92 seconds of ideal packed
reads. It excludes expanded BF16 writes/reads, temporary lifetimes, expert
launches, recomputation details, communication and host synchronization.
It does not establish 1.5-second unpacking or 2–4 seconds/sample training.
The schedule arithmetic is correct *if* those sample rates hold; week-long
feasibility and overnight completion are not established.

## Revised gates and operating order

1. **Offline G0a:** tiny decoded linear input-gradient checks, finite differences
   away from hard-routing boundaries, and explicit tests of surrogate gradients;
   attention/mask/sink parity; full-answer head loss; TP=1/2 gradient equivalence;
   checkpoint on/off and batch-accumulation equivalence. No full checkpoint or
   service interruption is required for initial component tests.
2. **Integrated G0b:** bounded reference forward checks using the correct
   checkpoint, tokenizer ID 129264, preprocessing and quantization mode. Capture
   actual teacher-forced reference logits on identical prefixes; generated API
   answers alone cannot supply per-position logit parity. Report numerical
   errors and top-1 disagreement margins as well as fixture answers. Forward
   agreement cannot substitute for G0a's backward checks.
3. **G1:** start B=1, then 2, then 4 at the maximum admitted token shape. Measure
   both ranks through load, forward, backward and optimizer update. Reject
   sustained swap, OOM, unbounded growth or inadequate host reserve. The
   proposed 110 GiB device ceiling is not the sole acceptance condition.
4. **G2:** warm up and time ten complete optimizer updates with the actual
   accumulation/effective batch. Report distributions, samples/s, tokens/s,
   packing and answer lengths. Budget validation, checkpoint saves, load and
   recovery separately. Estimate each arm from this measurement before booking
   a window; do not automatically proceed at 4–10 seconds/sample.
5. **G3:** finite gradients only for the allowlist, unchanged frozen-weight
   checks, and a single-example learning sanity test. Failure to reduce loss
   triggers diagnosis of implementation, optimizer, capacity and objective;
   it does not by itself prove a harness bug. Do not require unattainable
   zero loss from an arbitrary frozen recipient after exactly 50 updates.
6. **G4:** use the main plan's T0 random / T1 donor / T2 remapped donor arms,
   the same full aligner architecture and matched budget. The submitted C/D/B
   and E/F labels are stale. Fine-tuning follows the Step 3 image-dependence
   gate; initial G0 diagnostics need only a small reference set, not the 2K corpus.

Cache tower outputs, not final aligner outputs, for full-aligner training.
Local training replaces serving during its scheduled window. Restore the
captured **accepted Vision-Exp deployment** and verify vision/text afterwards;
keep immutable v0.1.1 as emergency rollback, not an automatic replacement for
the accepted vision service. Preserve the main plan's outage limits and
rollback reserve until measured evidence justifies an explicitly revised window.
The long-optimization workflow motivates these component-first gates; a new
multi-hour outage is not implicitly authorized by this review.

External GPUs remain the fallback if a correct local stack cannot fit or meet
the time budget. Zeroth-order optimization is not mathematically restricted
to tiny adapters, but its sample cost here is unmeasured; truncated-model
auxiliary losses likewise do not establish full-model answer quality.

## Evidence and limits

[CPU results and NVIDIA metadata checks](../benchmarks/results/vision-spark-training-contract-review-20260913.json)
record the test shapes, reference revision, excerpt hash, compatibility
adjustment and measured outputs. Local diagnostic scripts are retained under
.local/results/vision-training-review-20260913/. The reference excerpts cover
model.py lines 308–317 and 589–639; the float32 Gate path uses F.linear and
the old CPU PyTorch test uses int64 hash indices. These are limited contract
checks, not the production quantized kernels, a validated backward port, a
full-model quality test, or a hardware benchmark. No training window was run
and no serving configuration changed for this review.

[model]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/blob/6821d6ad3681a4b137b066b76094fa82ebd0a380/inference/model.py
[kernel]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/blob/6821d6ad3681a4b137b066b76094fa82ebd0a380/inference/kernel.py
[head]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/blob/6821d6ad3681a4b137b066b76094fa82ebd0a380/inference/model.py#L769
[gate]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/blob/6821d6ad3681a4b137b066b76094fa82ebd0a380/inference/model.py#L589
[convert]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/blob/6821d6ad3681a4b137b066b76094fa82ebd0a380/inference/convert.py
[tp]: https://raw.githubusercontent.com/NVIDIA/Megatron-LM/main/megatron/core/tensor_parallel/mappings.py
