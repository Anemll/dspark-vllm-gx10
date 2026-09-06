# NVIDIA 0731 NVFP4 + Vision-Exp: compatibility test

Status (2026-09-06): **CPU preflight completed; hybrid GPU inference has not
run.** No vision-quality or speed result is claimed for this hybrid. The
original 0731 text service remained running on both Sparks.

The requested experiment combines NVIDIA's 0731-NVFP4 text backbone and its
DSpark weights with the multimodal-only parameters from Vision-Exp. This is
a new hybrid model. It is not the previously tested Vision-Exp checkpoint,
and is not equivalent to quantizing the complete trained Vision-Exp model.

## Checkpoint already available

The user-selected TB36 share already contains all 75 files for NVIDIA revision
`f1caa71142bd0be02f728c79f75042ac1e461579`. A pinned Hugging Face download dry
run reported **zero missing files**. All 48 weight shards are present, totaling
175,550,788,904 bytes (175.55 GB). No duplicate weight download was started.
Configuration/index hashes and cached shard revisions match the pinned model.
This is presence/provenance checking, not a fresh whole-checkpoint checksum.

The share is mounted over wired Ethernet on the source workstation. The
Sparks' similarly named local mount-point directories are not mounted NAS
volumes. A TB36-to-Spark staging or serving route still needs to be established
without filling the internal disks or silently violating the fabric policy.

## Actual CPU tests

[Machine-readable evidence](../benchmarks/results/nvidia-vision-hybrid-preflight-20260906.json)
records model identities, bounded tensor-sample hashes and the backend gate.

| Check | Result |
|---|---|
| Shared structural configuration | No differences in checked model dimensions, attention layout and DSpark block/target-layer fields |
| Vision-only tensor additions | 316: 263 tower/aligner, 4 image sentinel vectors, 46 target/draft visual-router biases, 3 hash-layer correction biases |
| Unrecognized additional tensor names | None |
| Shared text versus Vision-Exp tensor samples | All 18 samples differ, including sampled norms, gates and embeddings |
| Shared text versus NVIDIA tensor samples | 15/18 byte samples match; three packed-expert samples differ |
| Actual pinned NVFP4 backend selector | Rejects `flashinfer_b12x` with required `swiglu_limit=10.0` |
| Alternative backend source predicates | `flashinfer_cutlass` declares SM121 NVFP4 W4A4 support and forwards the clamp; installed GPU kernel not tested |

The repository suite passes all 58 tests, including seven new CPU audit and
backend-source tests. These are not hybrid inference or throughput tests.

The sample audit read only 155,936 tensor-payload bytes per checkpoint and
took 6.62 seconds. It uses safetensors headers and bounded CPU reads, not
PyTorch/CUDA. Small tensors are read completely; large tensors are sampled
at beginning/middle/end. Byte samples do not establish whole-model equality
or dequantized numerical equivalence. The three different packed-expert
samples must not be interpreted as measured accuracy loss.

The vision backbone is not simply the original language backbone plus an
encoder: its sampled shared weights differ, and its `rms_norm_eps` is
`1e-20`, versus `1e-6` in both 0731 checkpoints. Its
`num_nextn_predict_layers` field is 3 rather than 1; both use DSpark block
size 5. These fields are recorded as explicit hybrid-design choices, not
silently copied over the NVIDIA configuration. Shape compatibility alone
does not establish learned image/text alignment.

## Launch gates still open

1. **Clamp-preserving W4A4 backend.** A dependency-free test executes the
   actual pinned backend-selector function and reproduces rejection before
   backend construction. The NVFP4 adapter is different from our working
   MXFP4 W4A16 adapter. Removing the clamp to bypass this guard is not an
   acceptable fix. A second CPU test executes the pinned FlashInfer CUTLASS
   device/quantization predicates for simulated SM121 and NVFP4 W4A4; both
   pass. Its implementation forwards the SILU clamp to FlashInfer. Validate
   this existing alternative before considering a new kernel implementation;
   source-declared support does not prove the installed kernel works.
2. **Separate target/draft quantization.** NVIDIA main routed experts use
   NVFP4; retained DSpark expert tensors still use MXFP4 `.scale` entries.
   Target and draft loader dispatch must be verified independently. This
   remains a source-identified integration risk, not a reproduced GPU failure.
   Target FlashInfer CUTLASS plus draft B12X is a candidate configuration,
   contingent on independent quantization dispatch and GPU validation.
3. **Complete multimodal parameters and explicit configuration.** Include
   the routing biases and sentinel vectors, not just `vision.*`/`aligner.*`.
   Keep source identities separate and reject missing/duplicate weights.
4. **Storage access and bounded canary.** Once loading is supported, verify
   rollback and reserve an exclusive outage before one image-content canary.
   Correctness must pass before DSpark acceptance or throughput comparisons.

Reusing only selected Vision-Exp tensors is the user-requested experiment,
not an endorsed assumption that it will preserve vision quality. Applying
NVIDIA's quantization approach to the complete Vision-Exp backbone would be
a different experiment and must not be substituted silently.

NVIDIA describes its released checkpoint as text-only and says DSpark heads
were preserved but speculative decoding was not exercised in release
validation. DeepSeek describes Vision-Exp as continued-trained for visual
understanding. Sources: [NVIDIA model card](https://huggingface.co/nvidia/DeepSeek-V4-Flash-0731-NVFP4),
[Vision-Exp model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp).

The bounded A/B skill kept this preflight within a zero-outage window while
compatibility gates remain open. The required advisor design question is recorded locally; no
advisor tool or fallback script was available, and no advisor approval is
claimed. No serving image, model weights or node environment was modified.
