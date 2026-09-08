# NVIDIA 0731 NVFP4 + Vision-Exp: compatibility test

Status (2026-09-08): **The CUTLASS ABI repair passed the real SM121 GPU
retest; a separate V2 draft-backend routing fix now passes CPU tests.
Hybrid model inference has not run.** Read-only TB36 checkpoint access is now
verified on both Sparks. The corrected candidate is built once and its exact
image ID and code/module hashes verified on both nodes; full-model tests remain
open. No hybrid vision-quality,
DSpark acceptance or decode-speed result is claimed. The original 0731 text
service stayed running throughout this preparation.

## V2 DSpark mixed-backend correction

The pinned runtime exposes `speculative_config.moe_backend`, and the legacy
`LLMBaseProposer` honors it. Our V2 runner uses a different entry point:
`DSparkSpeculator.load_draft_model` calls `load_dspark_model` directly.
That loader originally copied only the draft attention settings, silently
inheriting the target's MoE backend even when a draft override was requested.
An executable CPU regression reproduces this with the pristine pinned loader:
target `flashinfer_cutlass` plus draft `flashinfer_b12x` still passed
`flashinfer_cutlass` to draft model construction.

The [V2 loader overlay](../overlay/vllm/v1/worker/gpu/spec_decode/dspark/utils.py)
now copies the kernel configuration only when an explicit draft override is
present. It preserves the target configuration, non-causal draft attention,
embedding/head sharing, and the no-override original text path. This runs at
model load, not per decode token. Eight
[CPU tests](../tests/test_dspark_draft_backend.py) cover the original failure,
correct override delivery, target isolation, unchanged defaults, attention
configuration, multimodal sharing, loader failure cleanup, and the existing
pipeline-parallel guard, plus inclusion in the component image Dockerfile.

The intended candidate settings are:

```json
{
  "moe_backend": "flashinfer_cutlass",
  "speculative_config": {
    "method": "dspark",
    "num_speculative_tokens": 5,
    "draft_sample_method": "probabilistic",
    "moe_backend": "flashinfer_b12x"
  }
}
```

These are configuration fields, not a standalone launch command. Use a new
immutable candidate image containing this overlay. The previously tested
`7a30f22955f8...` image does **not** contain this later loader correction;
do not relabel it or describe the correction as GPU-validated. The passed
CUTLASS kernel result remains valid for that unchanged binary, but actual
mixed-format weight loading, both-rank backend selection, DSpark acceptance,
text speed and combined vision correctness still require integration tests.

The corrected candidate has now been built once from source
`8cc38c3745935b422323b733a834f67e45a32d2c`, producing image
`sha256:2a1a3f6b7d4e744790aecd79649e1e8e39d8054ebb2cb132f6135078b9a6bcf4`
(`linux/arm64`). A no-GPU, read-only container verified the exact new loader
hash and unchanged repaired CUTLASS module. The exact image has now been copied
over the dedicated fabric and verified on the worker without rebuilding there.
It has not been publicly published or run as the two-rank server. Both nodes
can now read the pinned NVIDIA config and index, with hashes matching the
saved preflight; 48 shards totaling 175,550,788,904 bytes are visible from the
head. These are presence and metadata checks, not fresh whole-shard checksums.
Preparation initially stopped before a full image export because its temporary
storage could breach the declared free-space floor. A temporary localhost-only
registry, reached through an authenticated fabric SSH tunnel, completed the
copy without a full archive. The worker reused its existing base layers and
finished with the same image ID, source revision, loader hash and CUTLASS hash.
Both transfer listeners/processes were stopped; 9.16 GiB of duplicate registry
data was removed after verification, retaining all model/image/log artifacts.
No serving restart or benchmark occurred. Next is actual mixed-backend
checkpoint loading and inference under the bounded test/rollback procedure.

## Successful GPU retest

[Retest evidence](../benchmarks/results/nvidia-cutlass-retest-20260907.json)
records the unchanged canary on candidate source
`b2d902b8ea8e69b46e41c34e99d6f302e6e4a53a`, image
`sha256:7a30f22955f8a5e3018ef05e7c8b4c9eadbfb039d30edc7e36f0ca2360c55977`.

| Shape: tokens / hidden / intermediate / experts / top-k | Relative L2 | Cosine | Result |
|---|---:|---:|---|
| 4 / 128 / 128 / 8 / 2 | 1.869% | 0.999825 | Pass |
| 6 / 4096 / 1024 / 8 / 6 | 2.038% | 0.999794 | Pass |

Both use NVFP4 **W4A4**, BF16 output, seed 5205 and **clamp=10**.
The unchanged gates are relative L2 below 8%, cosine above 0.995, finite
output and exercised clipping. The larger shape uses target per-TP matrix
dimensions but a deliberately reduced expert count: this is not a full TP2
model test. Peak CUDA allocation was 321,036,288 bytes. The 0.878-second
component duration includes setup and is **not decode throughput**.

The isolated image replaces only `fused_moe_120.so` using the official
[FlashInfer 0.6.15 CUDA 13 ARM artifact](https://flashinfer.ai/whl/cu130/flashinfer-jit-cache/).
The [artifact lock](../config/flashinfer-cutlass.lock.json) pins the whole
wheel and selected module hashes, installed Python core hash, version and
architecture. The [extractor](../scripts/extract-cutlass-module.py) rejects
mismatches before writing anything. The image retains all other cache
binaries and adds the CPU-tested mixed-format dispatcher below.
Distribution metadata still says cache `0.6.13+cu130`: this is a **single
component repair**, not a complete cache upgrade. Its exact provenance is
stored at `/opt/dspark-cutlass/manifest.json` inside the image.

The [isolated Dockerfile](../docker/Dockerfile.nvfp4-cutlass) uses a BuildKit
named context `cache_artifacts` containing the pinned wheel. Build from a
clean exact source checkout, verify the local base tag resolves to image
`sha256:d768bb8b27788bb046a10b54c7df7de7d0d0e3f26b1ee0c7aab04b2c69b9cd1b`,
and pass that local tag as `BASE_IMAGE`, with the full `SOURCE_REVISION`.
BuildKit treats a bare `sha256:...` in `FROM` as a repository name, not a
local image ID. No runtime image has been published or production default
changed. Future two-rank tests must transfer this exact built image rather
than rebuilding independently.

All **69** repository unit tests pass. Pre/post streaming, idle metrics,
health, model list, version, dashboard and both-rank logs were checked.
The original containers, images and start times stayed unchanged; no outage
or baseline rerun was needed. Completed test/download containers were removed
after evidence capture, both experiment locks released, and the wheel,
candidate, checkpoints and raw logs retained. The bounded-test skill limited
the GPU test to a small memory-capped component check while storage remains
blocked; this does not establish candidate text-speed preservation.

## GPU result and mixed-format loader fix

[GPU canary evidence](../benchmarks/results/nvidia-cutlass-canary-20260907.json)
records the earlier actual SM121 attempt with NVFP4 W4A4, BF16 output and clamp=10.
It failed after 1.17 seconds in component code, before any numerical row:
FlashInfer's Python initializer supplied eight arguments, while the inherited
compiled CUTLASS module accepted seven. The image contains
`flashinfer-python==0.6.15` but `flashinfer-jit-cache==0.6.13+cu130`.
The installed Python source hash matches our pinned FlashInfer source exactly;
source-declared SM121 support did not establish binary compatibility.

The new prefix-aware quantization resolver honors NVIDIA's explicit
`quantized_layers` manifest and `mtp.*` exclusion. Seven new CPU tests check
all 43 target layers, all three draft layers, wrapper prefixes, format/layout
agreement, unchanged original dispatch and fail-closed malformed manifests.
They also execute the actual upstream dispatcher to reproduce its selection
of NVFP4 for an unchanged MXFP4 draft layer. All **65** repository tests pass.
This loader fix is **not deployed or GPU-checkpoint-validated**.

That ABI gate is now resolved by the pinned-artifact GPU retest above.
TB36 still needs authenticated read-only mounts on both Sparks. No credentials
were read or requested in chat. The earlier failed GPU evidence remains
unchanged and preserved.

Before that attempt, the existing text service returned HTTP health=200 but
timed out on a tiny completion and logged TCPStore/NCCL broken-pipe warnings
on both ranks. Restarting the **same** containers in worker-first startup order
recovered it. Endpoints, streaming and tool calls passed afterward; no model,
image or node environment file was changed. This preexisting stall is not a
failure of the NVIDIA hybrid, which has not been loaded.

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
| Alternative backend source predicates | `flashinfer_cutlass` declares SM121 NVFP4 W4A4 support and forwards the clamp; repaired kernel subsequently passed the bounded GPU retest above |

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

1. **Clamp-preserving W4A4 backend: component gate passed.** A dependency-free test executes the
   actual pinned backend-selector function and reproduces rejection before
   backend construction. The NVFP4 adapter is different from our working
   MXFP4 W4A16 adapter. Removing the clamp to bypass this guard is not an
   acceptable fix. A second CPU test executes the pinned FlashInfer CUTLASS
   device/quantization predicates for simulated SM121 and NVFP4 W4A4; both
   pass. Its implementation forwards the SILU clamp to FlashInfer. The
   repaired existing backend now also passes the real-GPU component test.
   Full expert-count, vLLM adapter, checkpoint and TP2 integration remain open.
2. **Separate target/draft quantization.** NVIDIA main routed experts use
   NVFP4; retained DSpark expert tensors still use MXFP4 `.scale` entries.
   CPU dispatch is now fixed and tested; actual mixed-format checkpoint
   loading remains an integration gate, not a reproduced GPU failure.
   Target FlashInfer CUTLASS plus draft B12X is a candidate configuration,
   contingent on independent quantization dispatch and GPU validation. The V2
   loader also needs the explicit backend-override correction above; the
   legacy proposer's existing support does not cover this runner.
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
