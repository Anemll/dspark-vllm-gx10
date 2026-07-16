# RoCE RAM weight loading

`roce_tp` is an opt-in startup loader for the validated two-node DSpark
topology. TP rank 0 is the only process that opens checkpoint payload files.
TP rank 1 runs the model's normal weight-loader functions against storage-free
tensor metadata; the requested source slices and transformations are evaluated
on rank 0, packed, and sent over the existing TP NCCL/RoCE communicator.

This feature only changes how weights reach rank 1 RAM. It is not shared
storage, NFS, checkpoint replication, KV transfer, or runtime weight syncing.
Rank 1 still needs local access to the model configuration, tokenizer, and any
other non-weight metadata required to construct the model.

## Traffic and memory contract

- Rank 0 reads each checkpoint tensor once and loads its normal TP-local write.
- Rank 1's existing loader logic determines its exact expert ownership, TP
  slice, packed dtype view, padding, and fused-parameter destination.
- Only source bytes used by rank 1 writes are transferred. TP-sharded tensors
  therefore send the rank-1 shard; replicated parameters are sent in full.
- Transfers are packed into approximately 256 MiB NCCL messages by default to
  avoid per-expert send overhead. `DSPARK_ROCE_LOAD_BUFFER_MB` controls the
  bound. One checkpoint tensor's writes can form an oversized batch when they
  exceed that value.
- Rank 1 writes directly into final model parameters. B12X and other
  `process_weights_after_loading` transformations still execute locally after
  the raw checkpoint writes have arrived.

The resulting traffic is close to rank 1's raw resident weight size, not the
full checkpoint size. It can be slightly larger or smaller than exactly half
because some parameters are replicated, experts can be rank-owned, and loader
post-processing can change the final representation.

## Build and enable

Build a distinct image from the feature branch so the validated `0.1.1` image
remains available:

```bash
git switch wip/roce-ram-loader
FINAL_IMAGE=ghcr.io/anemll/dspark-vllm-gx10:roce-ram-loader \
  ./scripts/build-image.sh
```

Set the following values in both `config/head.env` and `config/worker.env`:

```dotenv
DSPARK_VLLM_IMAGE=ghcr.io/anemll/dspark-vllm-gx10:roce-ram-loader
DSPARK_WEIGHT_LOAD_FORMAT=roce_tp
DSPARK_ROCE_LOAD_BUFFER_MB=256
```

Start the worker first through the existing cluster launcher. Successful
startup logs include `rank 0 is the sole checkpoint reader`, `rank 1 will not
open checkpoint payload files`, the transferred GiB count, and the number of
packed batches.

## Guardrails

The loader fails early unless all of these are true:

- tensor parallel size is exactly 2;
- pipeline parallel size is exactly 1;
- model parameters are on CUDA and the TP NCCL communicator is initialized;
- the underlying checkpoint format is `auto`, `hf`, `safetensors`, or `pt`.

The loader deliberately does not use vLLM's RL weight-transfer engine. That
engine is initialized only after startup model loading and broadcasts
checkpoint-format tensors rather than rank 1's exact TP writes.

## Rollback

Set `DSPARK_WEIGHT_LOAD_FORMAT=auto` on both nodes and restart to restore the
original behavior while keeping the feature image. For a complete image
rollback, also restore:

```dotenv
DSPARK_VLLM_IMAGE=ghcr.io/anemll/dspark-vllm-gx10:0.1.1
```

No checkpoint files or formats are modified, so rollback requires no data
conversion or cleanup.
