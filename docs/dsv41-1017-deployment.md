# DeepSeek V4.1 Flash 1017 native-vision candidate

Status: **NOT cluster validated**. This profile prepares a reproducible vLLM
candidate; downloading weights or passing the renderer tests does not establish
vision correctness, speed, or memory fit on this cluster.

## Model and runtime provenance

The language model, Engram tables and native vision components come from
[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).
The chosen EXL3 pack is
[HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark](https://huggingface.co/HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark).
This is EXL3 quantization, not an NVIDIA NVFP4 checkpoint or a Vision-Exp
transplant. Preserve the native vision tower, aligner, image sentinels and
vision routing biases. No additional vision donor is required.

The reviewed runtime is HeNryous `v24`, pinned to the public linux/arm64 manifest
`sha256:a796b0515cf6245f95cd97e2f5d12d000e7d99ad475d53d4dc21a0bc9a68cc81`.
Its image config ID is
`sha256:709f3991d0d3966608cfa1e59cd03bd8d9796467b1c307ecfcbc01adb9e7291e`.
These were checked through the registry on 2026-10-26 without downloading image
layers. Registry labels inherit the vLLM 0.31.0 base image and do not describe
all later patches; use the image digest to identify the runnable artifact.

The command settings come from two pinned sources:

- [HeNryous v24 example at 38207172](https://github.com/HeNryous/deepseek-v41-1017-dgx-spark/blob/382071728ce69700b28e7a5ea197ab0a48b2232c/serve/serve_example.sh).
- [sudoingX TP2 launcher at cc59c36](https://github.com/sudoingX/dgx-spark-deepseek-v41/blob/cc59c3609e8e671418f39eb92aa16b0e4f770b77/scripts/launch/dsv41_tp2.sh).

Credit DeepSeek for the model, HeNryous for this 1017 quantization and runtime
patches, sudoingX for the vLLM EXL3 integration, and the vLLM, exllamav3 and
TensorFold projects for their runtime/kernel work. The adapted launch settings
retain Apache-2.0 attribution in the renderer; the license text is in
`LICENSES/Apache-2.0.txt`. Preserve the upstream image's licenses and notices.

This candidate is independent of the existing DSv4 `docker-compose.yml`,
`upstream.lock`, and `overlay/vllm/models/deepseek_v4` changes. Applying that
older overlay to the V4.1 runtime has not been reviewed and is not part of this
profile.

## Archive and staging gates

Use `config/dsv41-1017.lock.json` and its linked file manifests to pin downloads.
The EXL3 manifest is `config/dsv41-1017-exl3.files.json`; the separate official
Engram manifest is `config/dsv41-1017-engram.files.json`. Both are required for
one model directory on **each** Spark, with roughly 402 GB of model files.
An old 0909 checkpoint is not a source for the 1017 Engram tables: their shard
names and sizes can match while their bytes differ.

Complete and hash-verify the old EXL3 and Engram archive before reclaiming its
local SSD space. Capture the old image ID, launch arguments, per-rank environment,
and logs with the archive so it can be restored. Keep production images,
checkouts, environment files and failure evidence. If the old model is removed
from local SSD, rollback includes copying the archived weights back and loading
them; it cannot meet an image-only restart estimate.

Stage the new model on local NVMe. Use the dedicated fabric for large transfers
between Sparks; verify the route before copying, then verify the destination
against the same pinned manifests. Network storage is an archive/source, not
the intended Engram serving path. Save the complete download/verification report
before a launch window.

The staging tool needs `huggingface_hub` and `hf-xet` in the selected Python
environment. Use Xet for the approximately 100 GB Engram files; HTTP-only
downloads are not the intended path for files of that size. Put Xet's cache on
the archive volume as well. This example uses a generic mounted volume; substitute
the actual mount and keep receipts/logs in the ignored working directory:

```bash
archive_root=/Volumes/MODEL_ARCHIVE
candidate_dir="$archive_root/Models/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark"
export HF_HUB_DISABLE_XET=0
export HF_XET_CACHE="$archive_root/.hf-xet"
mkdir -p .local/results/1017-staging

python3 scripts/stage-hf-checkpoint.py download \
  --manifest config/dsv41-1017-exl3.files.json \
  --destination "$candidate_dir" --require-mount "$archive_root" \
  --receipt .local/results/1017-staging/exl3.receipt.json --workers 2 \
  >.local/results/1017-staging/exl3.log 2>&1 &
candidate_download_pid=$!
python3 scripts/stage-hf-checkpoint.py download \
  --manifest config/dsv41-1017-engram.files.json \
  --destination "$candidate_dir" --require-mount "$archive_root" \
  --receipt .local/results/1017-staging/engram.receipt.json --workers 1 \
  >.local/results/1017-staging/engram.log 2>&1 &
engram_download_pid=$!
wait "$candidate_download_pid"
wait "$engram_download_pid"
```

New or changed files receive a full hash check before entering the receipt.
Re-running `download` with the same receipt reuses existing hash proofs only
when repository, pinned revision, destination, expected hash, size and file
modification time still match. Original verification timestamps are retained.
Use `--verified-receipt <saved-receipt.json>` to recover proofs from an earlier
attempt; the option is repeatable. This avoids re-reading large verified shards
on every resume. These local receipts must be trusted: this is a resume shortcut,
not protection against modifications that deliberately preserve file metadata.

A successful receipt for one source does not cover the other source. To audit an
existing destination without downloading, run the same command with `verify`
instead of `download` and use a new receipt filename; `verify` always rehashes
every file. Keep the old-model archive
job in a separate destination so it can run alongside this staging job.

For ordinary EXL3 shards, `download --transport http` runs with Python's standard
library alone. It resumes partial files with HTTP byte ranges, rejects an ignored
or mismatched range, retries interrupted transfers within its deadline, and
performs the same final hash check. Very large Engram shards still require the
Xet path described above; the HTTP transport is not a replacement for that path.
When the archive is a NAS, run staging and hashing on the storage host when
possible. Check the actual connection used by an existing SMB mount: changing
the Mac's default route does not prove an established share moved to Ethernet.

For the existing 0909 deployment, generate separate manifests at the audited
revisions, resume the old EXL3 archive, and verify the Engram files already in
the official-model archive. These jobs can run in parallel with the candidate
downloads above. They preserve the existing source files and partial downloads:

```bash
python3 scripts/stage-hf-checkpoint.py plan \
  --repo Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw \
  --revision 64ba41b6c916a587db06eae2e19b7845f7be6e6b \
  --manifest .local/results/1017-staging/old-exl3.files.json
python3 scripts/stage-hf-checkpoint.py plan \
  --repo deepseek-ai/DeepSeek-V4.1-Flash \
  --revision dba1be0a40aa45a94ad051997016db3960a90277 \
  --file model-00047-of-00048.safetensors \
  --file model-00048-of-00048.safetensors \
  --file config.json --file model.safetensors.index.json \
  --manifest .local/results/1017-staging/old-engram.files.json
python3 scripts/stage-hf-checkpoint.py download \
  --manifest .local/results/1017-staging/old-exl3.files.json \
  --destination "$archive_root/Models/DeepSeek-V4.1-Flash-EXL3-2.9bpw" \
  --require-mount "$archive_root" --workers 1 \
  --receipt .local/results/1017-staging/old-exl3.receipt.json
python3 scripts/stage-hf-checkpoint.py verify \
  --manifest .local/results/1017-staging/old-engram.files.json \
  --destination "$archive_root/Models/DeepSeek-V4.1-Flash" \
  --require-mount "$archive_root" --workers 1 \
  --receipt .local/results/1017-staging/old-engram.receipt.json
```

Before local source removal, compare the Spark source hashes with these same
manifests as well. A verified download proves that archive matches its pinned
upstream snapshot; it does not by itself prove that a locally modified source
matches. Preserve any differing local tokenizer, configuration or other files
separately as part of the rollback artifact.

## Review one launch per rank

Copy `config/dsv41-1017.env.example` to two private, ignored environment files,
one for each node. Use explicit absolute paths; the parser does not expand
shell variables. Fill each node's own fabric IP, netdev, HCA list and RoCE v2 GID
index. Verify the GID maps to that node's fabric address. CPU pinning defaults
are from the GB10 recipe and need checking against each node's CPU topology.

The renderer accepts only the reviewed digest and rejects unknown settings,
text-only settings, zero image allowance, obsolete parser overrides and invalid
limits. It prints commands or JSON; it does not contact or change either node.

```bash
python3 scripts/render-dsv41-1017.py .local/deploy/1017-head.env --format json
python3 scripts/render-dsv41-1017.py .local/deploy/1017-worker.env --format json
python3 scripts/render-dsv41-1017.py .local/deploy/1017-worker.env --stage jit --format shell
python3 scripts/render-dsv41-1017.py .local/deploy/1017-worker.env --format shell
python3 scripts/render-dsv41-1017.py .local/deploy/1017-head.env --format shell
```

Review and save the rendered output with the experiment artifacts. The JSON has
`preflight_argv` and `docker_argv`; the shell form runs the same checks before
the container command. It never removes an existing container, enables automatic
restart, pulls an image, or changes a host's network settings. Execute the saved
commands on the intended Spark only after completing the cluster preparation.

The upstream launcher uses a DSpark Markov-token asset. Obtain
[`data/dspark/markov_top_tokens.json` at cc59c36](https://github.com/sudoingX/dgx-spark-deepseek-v41/blob/cc59c3609e8e671418f39eb92aa16b0e4f770b77/data/dspark/markov_top_tokens.json)
and set `MARKOV_FILE` on each node. Its SHA256 is
`0b6e71e638abd5abce2db433965bb5ff523cbbe92ed7ae54324a50ccaeffa19a`; the rendered
serve command checks it. The model mount and this asset are read-only.

Create the dedicated cache directory on each node, preload the reviewed image
once and transfer the exact image to its peer, then compare image IDs. Prebuild
the plugin's JIT kernels using the rendered `--stage jit` command on **both**
nodes while inference is stopped. The compiler can exhaust memory if invoked
beside an already loaded model. The same cache mount and
`EXL3_SMOE_FLAGS=-DSMOE_RUNFLAGS=1` are used by JIT and serve.

The environment file pins the registry manifest; the renderer runs its pinned
arm64 image config ID on both ranks. `docker save`/`docker load` can preserve
that exact ID without retaining a registry `RepoDigest`, so a digest-name
lookup on the receiving rank is not required. Verify the ID after transfer;
do not substitute a mutable tag or pull a different image on the worker.

Before replacing an existing service, capture evidence and use the complete
head-then-worker stop barrier. For the candidate and for rollback, start the
**worker first**, wait for its distributed initialization, then start the head.
Keep separate per-rank environment files and confirm that both ranks loaded the
same runtime, weights and configuration.

## Initial settings and client contract

| Setting | Initial candidate |
|---|---|
| API / distributed master port | 8888 / 25000 |
| Tensor parallelism | 2 ranks, worker headless |
| Model context limit | 240,000 tokens per request, input plus output |
| Maximum active sequences | 4 |
| Batch budget / long-prefill chunk threshold | 4,096 / 512 tokens |
| KV allocation | 3.5 GiB per rank, upstream FP8 layout |
| KV offload | upstream 256 MiB CPU tier and filesystem tier |
| Images | native vision, positive request limit (initially 4) |
| Speculation | DSpark K5; set `SPECULATIVE_TOKENS=0` for target-only |
| Tool and reasoning parsers | `deepseek_v41` |
| Client maximum output | 32,768 tokens |

The context ceiling is per request, but KV memory is shared across active
requests. Four slots do **not** guarantee four simultaneous full 240K contexts;
measure actual capacity and preemption after loading. The 32,768 output value is
a client setting, not a server-wide output cap in this renderer. Input plus
requested output must fit the context limit.

The initial context, sequence count, aliases and port deliberately differ from
the source recipe's 1M/8-slot example. The v24 kernel settings, four NCCL channels,
`NCCL_PROTO=^LL128`, cache layout, offload and adaptive K5 policy are preserved.
This is a new deployment candidate, so its measurements cannot be attributed to
one kernel optimization. The supplied DSpark round-cost table came from the
upstream author's hardware; measure it locally before claiming tuned performance.
Filesystem KV offload also needs a bounded storage/retention policy before long
term operation. The renderer does not delete cache files.

For a separately budgeted DSpark depth-calibration diagnostic, optionally set
`DEPTH_POLICY_CACHE_FILE=dspark-depth-policy.txt`. Create that file inside the
node's existing `CACHE_DIR` before launch; the renderer adds only the native
`VLLM_DSPARK_DEPTH_FILE` hook and checks that the file exists. It does not change
the default confidence policy or enable this hook by default. The pinned runtime
lets rank 0 read a policy such as `fixed:3` or `conf:<six measured round costs>`
and broadcasts it to both ranks every 64 verification steps. Preserve the
working configuration, use atomic file replacement inside the mounted cache
directory, warm at least 512 output tokens after each policy change, and confirm
the applied policy in the logs before measuring. A shorter fixed depth changes
how many drafts may be accepted, not which target-verified tokens are valid.
Do not claim a locally tuned policy from the borrowed cost table alone.

Both API names serve this same native-vision checkpoint:

- Pi: `DeepSeek-v4.1-Flash-EXL3`, `input: ["text", "image"]`.
- Droid: `gemini-3-flash-preview`, provider `generic-chat-completion-api`,
  `noImageSupport: false`. The Gemini name is a local capability alias.

Preserve existing clients' IDs and defaults. Do not add an override to an obsolete
0731 or Vision-Exp model. The accepted client settings use context 240,000 and
maximum output 32,768. This preparation does not edit any client's configuration
or the dashboard. Confirm both aliases by actual generation as well as `/v1/models`.

The v24 parser reads thinking options from `chat_template_kwargs`. First test
`{"thinking": false}` and `{"reasoning_effort": "high"}`. Do not map a client's
unsupported `medium` value blindly into the server template. The vLLM recipe's
V4.1 encoding/parser supplies prompting; this profile does not transplant the
TensorFold Jinja workaround.

The renderer intentionally leaves `--tokenizer-mode` and `--kv-cache-dtype`
at the exact pinned launcher's defaults. The pinned model config declares
`DeepseekV41ForCausalLM`, which
[vLLM's model config maps from `auto` to `deepseek_v41`](https://github.com/vllm-project/vllm/blob/6e517b15c1833cf72a7f557ee32524d98682e617/vllm/config/model.py#L742).
On SM12x the
[NVIDIA model selects the SM120 FlashInfer class](https://github.com/vllm-project/vllm/blob/6e517b15c1833cf72a7f557ee32524d98682e617/vllm/models/deepseek_v41/nvidia/model.py#L232),
whose packed layout
[resolves `auto` to `fp8_ds_mla`](https://github.com/vllm-project/vllm/blob/6e517b15c1833cf72a7f557ee32524d98682e617/vllm/models/deepseek_v41/attention.py#L132).
Check the startup logs for both resolved values before accepting the deployment;
a different backend or tokenizer is a failed configuration gate.

## Acceptance window

First capture a recoverable baseline and reserve enough time for rollback,
including model copying when necessary. Use a target-only canary first, then
the same fixtures with DSpark K5. A completed model load and a healthy HTTP socket
alone are insufficient.

1. Check both ranks, API health, runtime identity, streaming first token and
   completion, final token usage, and both API aliases.
2. Run known-answer OCR and image-swap fixtures. Test multi-image input,
   image followed by text, and text followed by image to catch encoder-cache
   recovery failures. Verify image content actually changes the answer.
3. Test tool calls, reasoning off/high, and the Pi/Droid image-history path.
4. Run the repository's matched C1/C2/C4 text prompts and prefill ladder. Keep
   output lengths, seeds, temperature and cached/cold treatment fixed. Record
   DSpark acceptance separately from decode throughput.
5. Check long-context reliability, request waiting/preemption, memory, disk
   growth and both-rank CUDA/NCCL errors. Stop on incorrect vision, a dead rank,
   a stall or an unexplained regression, and restore the accepted deployment.

Record limitations explicitly: v24 has upstream vision support, but its encoder
cache behavior, current-client interoperability and sustained stability remain
local test gates. Do not report this branch as deployed or accepted until those
results exist. Expand context or concurrency only in a subsequent measured test.

Static checks:

```bash
python3 -m unittest discover -s tests -p test_dsv41_1017_profile.py
node tests/test_dashboard_agent_snippets.mjs
git diff --check
```
