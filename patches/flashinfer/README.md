# FlashInfer SM12x DSV4 native sparse widths

`sm12x-dsv4-native-widths.patch` is the complete, unmodified diff of official
[FlashInfer PR #4380](https://github.com/flashinfer-ai/flashinfer/pull/4380),
backported to the FlashInfer revision in this repository's `upstream.lock`.
It enables native 192/256-entry DSV4 sparse-MLA dispatch without a vLLM or
FlashInfer dependency-version migration. The vLLM caller must separately stop
padding the selected path to 512 before serving can benefit.

## Provenance

| Item | Value |
| --- | --- |
| Target source | `0472b9b3f2fba11b463f8526f390297d52a8aad7` (FlashInfer 0.6.15) |
| Official merged commit | `24d7dfb2639083c5a4d418881099421fc800b7bb` |
| Merged commit parent used for exact diff | `6d4f3095ee429227fb115f4e6fae9e4e5c5c7696` |
| Original PR base | `b1d95851675b8799d623df4d5a7d6eac3254b3ff` |
| Final PR head | `cbd7705ee3776cd020f528319a0935b40c92f7f6` |
| Patch SHA-256 | `e982c904e4ab015a10d1d0bfb0d1321f8d6d52b27db4ea750b8061dec3f9191d` |
| Inventory | 7 files; 299 insertions, 60 deletions |

The patch is exactly `git diff --binary 24d7dfb^ 24d7dfb`. Apply it to a clean,
detached checkout of the target commit with `git apply --check` followed by
`git apply`; do not copy entire files from the newer commit. Six affected files
are identical between the target and the merge parent. The Python module has
an intervening autotuner API migration at the merge parent, outside the patch
hunks. Applying the patch preserves the old target's compatible autotuner API.

## Scope and compatibility

- Adds DSV4 standalone decode instantiations for 192/256 entries at
  8, 16, 32, 64, and 128 heads.
- Adds BF16-QK single-cache prefill dispatch for the same widths.
- Retains the complete upstream H8 prefill change: padded 16-head tiles with
  guarded global Q, sink, output, and LSE accesses, including dual-cache H8.
- Adds a Python error for unsupported decode shapes before they reach the
  prefill-only C++ assertion.
- Includes upstream correctness tests and the benchmark HND layout fix.

No public Python signature, TVM-FFI signature, cache format, build configuration,
dependency requirement, CUDA requirement, or vendored submodule pin changes.
The packed cache remains DSV4's 584-byte FP8-footer format with BF16 RoPE;
this is not generic NVFP4 KV support. The sparse-MLA JIT module must be rebuilt
because it contains new compiled instantiations. Existing compiled caches
are not evidence that this patch was built or loaded.

The upstream H8 safety change is retained so this remains a coherent reviewed
patch. It does not change the tile dimensions or valid-head paths for H32,
the relevant 64-head model configuration at TP=2. Still validate both decode
and prefill: the 64/65-token dispatch boundary changes kernel families, and
native-256 prefill uses BF16 QK while the padded-512 path uses FP8 QK.

## Runtime file mapping

FlashInfer's pinned `pyproject.toml` packages `csrc/**`, `include/**`, and
vendored headers beneath `flashinfer/data`. Its `build_backend.py` copies
actual files for wheels and creates symlinks for editable installs.

| Patched source | Installed package-relative path |
| --- | --- |
| `flashinfer/mla/_sparse_mla_sm120.py` | `mla/_sparse_mla_sm120.py` |
| `csrc/sparse_mla_sm120_decode_dsv4.cu` | `data/csrc/sparse_mla_sm120_decode_dsv4.cu` |
| `csrc/sparse_mla_sm120_jit_binding.cu` | `data/csrc/sparse_mla_sm120_jit_binding.cu` |
| `csrc/sparse_mla_sm120_prefill.cu` | `data/csrc/sparse_mla_sm120_prefill.cu` |
| `include/flashinfer/attention/sparse_mla_sm120/prefill_kernel.cuh` | `data/include/flashinfer/attention/sparse_mla_sm120/prefill_kernel.cuh` |

The remaining two changed files are the upstream test and benchmark.

## Diagnostic image and component test

The implemented route is `docker/Dockerfile.flashinfer-width-diagnostic`.
It requires `TESTED_IMAGE` and `SOURCE_REVISION`; use the exact accepted image
ID or digest and the repository commit containing these artifacts. It creates
a new image layer, never mutates the running container, and performs no pip,
package-manager, dependency, submodule, or model operation. The diagnostic
image is not a serving candidate.

The Dockerfile copies this authoritative patch, provenance, manifest, license,
`scripts/apply-flashinfer-width-patch.py`, the diagnostic launcher, and
`benchmarks/benchmark_sparse_mla.py`. The helper verifies the patch SHA-256,
all five old runtime hashes, every exact hunk context, and all five expected
new hashes before writing any runtime file. It rejects traversing paths,
symlinks, duplicates, mismatched versions, and repeat application. It applies
only the five runtime files; the complete original patch, tests, and benchmark
remain available for audit. `runtime-manifest.json` supplies the installed
path mapping and old/new SHA-256 values.

Build once using an immutable commit-derived diagnostic tag. The coordinator
should inspect the accepted base identity and choose actual values:

```bash
docker build --network none --pull=false \
  --build-arg TESTED_IMAGE="$TESTED_IMAGE" \
  --build-arg SOURCE_REVISION="$SOURCE_REVISION" \
  --file docker/Dockerfile.flashinfer-width-diagnostic \
  --tag "$DIAGNOSTIC_IMAGE" .
```

The diagnostic launcher is the image entry point. Supply `--run-dir` as a new
directory beneath a persistent output mount, followed by harness arguments
after `--`. Run only under the coordinator's GPU reservation, with a hard
outer timeout for the entire process. The launcher sets a unique JIT workspace
and a new empty AOT directory before requesting the sparse module. This matters
because `JitSpec.build_and_load()` prefers an old AOT `.so` even when the JIT
workspace is new. `FLASHINFER_AOT_DIR` is overridden as a Python module
attribute; this pinned version has no corresponding supported shell override.

The launcher defaults to GB10's `FLASHINFER_CUDA_ARCH_LIST=12.1a`, sets
`FLASHINFER_JIT_DEBUG=0`, and limits NVCC to one thread with two build jobs.
Debug must be explicit: `FLASHINFER_JIT_VERBOSE=1` otherwise also enables
`-O0` and device debugging in the pinned source. It prebuilds the one sparse
module, records its `.so` and `build.ninja` hashes, runtime source hashes,
compiler/version/GPU information, compile elapsed time, AOT/JIT paths, and
image/source build provenance, then invokes the bounded component harness
in the same process. No checkpoint is loaded. `--compile-only` provides a
separate feasibility gate if the coordinator needs it.

For the first component run, compare widths `512,256,192` with identical
active indices/lengths, heads, query/cache tensors, seed, stream, and timing
mode. Include 64 and 65 tokens, variable active lengths, sinks, and output
reuse. Check output against a dense reference for every width and compare
widths directly. A 192-wide arm is valid only when all active entries fit.
The public-API event envelope includes host enqueue overhead and internal
runner/LSE allocation; it is not a pure standalone kernel measurement.
Add upstream public-API and dual-cache tests before considering serving.

Only after the component gate wins, assemble one immutable serving image with
the same five runtime changes and a reviewed vLLM dispatch change. Preserve
patch/source/image provenance and run the identical image on both ranks.
Full TP=2 correctness and long-context gates remain required. The local
disposable patched source at `.build/ifa26-flashinfer` is for analysis and
testing only; the patch and manifest are the durable sources of truth.

Do not run `pip wheel` or `pip install` against the pinned checkout for the
initial component experiment, even with `--no-deps`: its wheel preparation
unconditionally calls `_install_cuda_tile_compile_deps`, which attempts to
install newer CUDA compiler wheels. `BUILD_NVEP=0` disables the separate EP
build but does not disable that cuTile dependency-install hook. A dependency
audit is required before using the upstream packaging hooks for a release.

## Validation performed

- The complete patch applies cleanly to the exact target with `git apply --check`.
- The resulting source passes `git diff --check`.
- The three changed Python files pass `python3 -m py_compile`.
- The stdlib helper applies all five installed-layout files to exact pinned
  fixtures and produces the manifest's expected hashes. Seven tests pass,
  including complete preflight before writes, old/new hash failures,
  traversal/symlink refusal, repeat-apply refusal, and no fuzzy hunk offsets.
- Both helper and launcher pass Python syntax checks and support CPU-only help.
- Python and CUDA decode dispatch additions were compared against the
  unchanged signatures and existing 64-entry tile implementation.
- All seven original copyright/SPDX headers remain intact; the relevant
  BSD-3-Clause notice is retained alongside this patch.

These are source checks only. No CUDA compilation, GPU correctness result,
component gain, or serving improvement is claimed by this backport artifact.
