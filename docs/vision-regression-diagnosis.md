# Vision text regression: source and saved-trace diagnosis

Status: **candidate still rejected; no new GPU test or image deployment.**
This diagnosis does not replace the [failed matched text gate](vision.md).

## What the saved traces establish

All eight C1 requests below used the identical saved request body: the original
explanation prompt, temperature 0, seed 4104 and 512 output tokens. They used
the same benchmark client SHA. Every request passed framing, usage and length
checks. Each visible output has a different SHA256, including repeats within
the accepted control. Identical input therefore did not produce identical
generated content in these observations.

| Arm | Trial | Aggregate tok/s | Meaningful output chunks | Mean inter-chunk ms |
|---|---:|---:|---:|---:|
| Recorded control | 1 | 48.30 | 164 | 63.81 |
| Recorded control | 2 | 52.24 | 151 | 63.97 |
| Vision candidate | 1 | 43.92 | 182 | 63.31 |
| Vision candidate | 2 | 41.24 | 192 | 63.81 |
| Candidate confirmation | 1 | 45.19 | 174 | 64.48 |
| Candidate confirmation | 2 | 46.99 | 168 | 63.98 |
| Restored control | 1 | 48.03 | 166 | 63.54 |
| Restored control | 2 | 49.05 | 162 | 63.44 |

The candidate needed more output chunks, while mean inter-chunk time stayed
close to 64 ms. This is evidence to investigate changed generation/acceptance
before attributing the result to slower kernels. It is **not** a measurement of
engine-step duration or proof of equal GPU performance: speculative tokens can
be grouped in SSE chunks, and server counters were not isolated per trial in
these historical files. Log-window acceptance ratios must not be substituted
for missing per-trial counters.

At pinned vLLM revision `752a3a504485790a2e8491cacbb35c137339ad34`,
`v1/worker/gpu/sample/gumbel.py::gumbel_block_argmax` adds Gumbel noise only for
nonzero temperature. `spec_decode/rejection_sampler_utils.py::_rejection_kernel`
uses target-argmax equality at temperature 0. Thus the setting named
`draft_sample_method=probabilistic` is not, by itself, an explanation for
different greedy outputs. Numerical, shape-dependent or state-dependent
variation remains unisolated; no particular cause is established.

[Machine-readable observations](../benchmarks/results/vision-text-drift-20260905.json)
include original artifact hashes, actual request hashes and visible-output
hashes. Raw request bodies/SSE remain in the previously retained local evidence.

## Text dispatch isolation

The complete pinned vLLM source and the effective accepted `8c97ddf50b07`
overlay were compared, not merely the partial vision overlay. Findings:

- All four top-level sparse-SWA function ASTs are unchanged from the control.
- The NVIDIA model differs from pinned source only in the MoE constructor's
  vision-router installation branch; all its forward methods are unchanged.
- Executing the real metadata-build methods with recording CPU fakes produced
  identical metadata, allocation calls and GPU-dispatch arguments in 16
  cold/warm cases: target decode, DSpark decode, text prefill, mixed batches and
  an empty batch. This does not test CUDA numerics, timings or full inference.
- Vision-only callbacks remain inactive on the text-only checkpoint. The
  image-prefill module is separately named; no compiled text library was
  changed during this diagnosis.

`tests/test_vision_text_dispatch.py` keeps executable coverage for these
dispatch contracts. A small source change now skips the image-visibility
callback entirely for decode-only batches, including Vision-Exp requests.
Mixed image prefill still receives its 512-wide image metadata. This saves an
unnecessary Python callback in vision serving; **it has no measured speedup**
and cannot explain the earlier 0731 regression, where that callback was already
disabled. No new image containing this change has been built or tested.

## Next diagnostic and unchanged acceptance gates

Before another full-model decision run, use a bounded fixed-input target/draft
component replay to measure execution separately from accepted-token counts.
Check repeated outputs and the first divergent position, use fixed buffers and
shapes, and distinguish eager/captured behavior. A subsequent candidate run
must retain the original sampling settings and collect isolated per-wave draft,
accepted-token and request-decode counters (the content client already saves
them). Do not switch sampling modes or select a favorable prompt to erase the
failed gate.

The prior head-side standalone CUDA canary could not even create its CUDA
context beside the live server; its saved log reports out of memory. A small
tensor budget does not bound context overhead. Do not repeat that concurrent
launch without a separately approved resource/rollback contract.

Acceptance still requires matched text performance, long-context/protocol
reliability and actual Vision-Exp image-content, multiple-image, cached-prefix,
streaming and tool checks on both TP ranks. No result here proves any missing
full-model vision gate. The accepted text control remains the serving image.
