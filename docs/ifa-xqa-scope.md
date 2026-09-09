# IFA XQA scope for DeepSeek V4 Flash on two Sparks

## Finding

Do not directly backport upstream vLLM PR 49718's generic XQA integration into
this runtime as a DSv4 text-speed optimization.

The current Spark runtime is vLLM 0.25.2 with FlashInfer 0.6.15. Its generic
FlashInfer backend contains the earlier SM90-only XQA path and explicitly
rejects speculative XQA decode. The upstream work adds a dedicated SM12x path,
including speculative-decode metadata/graph fixes, and requires FlashInfer
0.6.16.post1 or later.

However, DeepSeek V4 Flash 0731 does not use that generic backend. Its model
selector chooses `DeepseekV4FlashInferSM120Attention` in the repository-owned
`flashinfer_sparse.py` implementation. That is a custom sparse-MLA path for
SM120/SM121, with its own KV layout, index handling and decode dispatch.
Changing only `vllm/v1/attention/backends/flashinfer.py` would therefore not
put the 0731 target model on XQA.

## Evidence

- Upstream PR 49718's XQA commits modify generic FlashInfer utilities/backend
  code and raise the FlashInfer dependency to 0.6.16.post1+.
- The 0731 model selector explicitly routes SM12x to the dedicated sparse-MLA
  class, bypassing the generic backend.
- The matched C1 decode trace assigns a substantial target-context total to
  B12X W4A16 MoE; its sparse-MLA group is smaller. See
  [text-speed-next-targets.md](text-speed-next-targets.md).
- The local vision implementation already supplies the missing SM121
  dual-cache, primary-topk512 image-prefill envelope as a separately named
  binary. It deliberately does not replace text decode kernels.

## Decision

A partial XQA cherry-pick is rejected: it carries compatibility risk while not
covering the active DSv4 attention path. A full vLLM/FlashInfer runtime rebase
is a separate candidate, not a kernel toggle. It must preserve the DSv4 sparse
overlay and Vision-Exp integration, build once, run on both TP ranks, and pass
matched text and vision gates before any claim of IFA improvement.

The next near-term text-speed candidates should instead address the observed
critical path: B12X W4A16 MoE tile/occupancy variants and DSpark target/draft
scheduling, each through a bounded real-weight component gate before a TP2
serving A/B. The fused-sum MoE candidate has already been rejected because it
was 3.4–3.6% slower at the relevant M6 shape.
