# NVIDIA NVFP4 + DSpark hybrid checkpoint

`scripts/build_hybrid_nvfp4_dspark_checkpoint.py` creates a metadata-correct
checkpoint view with NVIDIA's W4A4 NVFP4 target model and the existing
native-MXFP4 DSpark drafter. It does not quantize, convert, or rewrite tensor
payloads.

The accepted layout is deliberately strict:

| Source | Selected tensors | Required count | Selected shards |
|---|---|---:|---:|
| NVIDIA DeepSeek V4 Flash NVFP4 | every non-`mtp.*` tensor | 133,660 | 1–45 of 46 |
| DSpark native MXFP4 | `mtp.0.*` | 1,568 | 46 of 48 |
| DSpark native MXFP4 | `mtp.1.*` | 1,565 | 47 of 48 |
| DSpark native MXFP4 | `mtp.2.*` | 1,572 | 48 of 48 |

The merged index contains 138,365 tensors and 175,535,844,088 tensor-payload
bytes. NVIDIA target shard references are renamed from
`model-00001-of-00046` … `model-00045-of-00046` to the corresponding
`of-00048` names. The NVIDIA one-stage `mtp.0` shard is excluded; the three
DSpark shards retain their existing `of-00048` names.

## Read-only validation

Validation reads the two configs and indexes, stats NVIDIA target shards, and
reads only the JSON headers of the four source MTP shards. It requires the
pinned Hugging Face revision when download metadata is present; a manually
copied source without that metadata must match the pinned config and index
SHA-256 identities instead. It does not read bulk tensor payloads.

```bash
python3 scripts/build_hybrid_nvfp4_dspark_checkpoint.py \
  --nvidia-dir /path/to/DeepSeek-V4-Flash-NVFP4 \
  --dspark-dir /path/to/DeepSeek-V4-Flash-DSpark \
  --validate-only
```

An NVIDIA-only one-stage checkpoint supplied as `--dspark-dir` is explicitly
rejected. Validation also requires the exact target-layer quantization map,
all three DSpark stages, their exact tensor counts, and index-to-shard-header
agreement.

## Build the local symlink view

```bash
python3 scripts/build_hybrid_nvfp4_dspark_checkpoint.py \
  --nvidia-dir /path/to/DeepSeek-V4-Flash-NVFP4 \
  --dspark-dir /path/to/DeepSeek-V4-Flash-DSpark \
  --output /path/to/DeepSeek-V4-Flash-NVFP4-DSpark
```

The default view contains absolute symlinks for the 48 selected weight shards.
Dereference them when transferring the view to another machine:

```bash
rsync -aL --info=progress2 \
  /path/to/DeepSeek-V4-Flash-NVFP4-DSpark/ \
  destination:/path/to/DeepSeek-V4-Flash-NVFP4-DSpark/
```

Plain `rsync -a` is not sufficient: it would transfer source-local symlinks
which are broken on the destination. `--materialize manifest` is a lightweight
metadata review artifact with no shard files and is not runnable. Hardlink and
copy modes are also available for same-filesystem or fully independent views.

The builder refuses a nonempty output directory. `--force` is required to
replace one, and replacement is staged before the old directory is removed.
Source and output paths may not overlap.

## Config and provenance rules

`config.json` starts as an exact semantic copy of NVIDIA's config. Its complete
`quantization_config` is preserved. Only these fields are grafted from DSpark:

- `compress_ratios`
- `dspark_block_size`
- `dspark_noise_token_id`
- `dspark_target_layer_ids`
- `dspark_markov_rank`

NVIDIA supplies the tokenizer, generation config, and standalone Hugging Face
quantization config. `checkpoint.provenance.json` records source paths,
source config/index SHA-256 values, observed Hub revision and etags when
available, generated-metadata checksums, every source-to-destination shard
mapping, and source file sizes. Use `--hash-shards` only when full payload
SHA-256 values are required; it intentionally reads the complete selected
checkpoint and is therefore slow.

Run the dependency-free focused tests with:

```bash
python3 -m unittest discover -s tests \
  -p 'test_build_hybrid_nvfp4_dspark_checkpoint.py' -v
```
