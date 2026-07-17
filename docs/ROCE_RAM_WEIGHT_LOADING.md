# RoCE RAM weight loading

`roce_tp` is an opt-in startup loader for the validated two-node DSpark
topology. TP rank 0 is the only process that opens checkpoint payload files.
TP rank 1 runs the model's normal weight-loader functions against storage-free
tensor metadata; the requested source slices and transformations are evaluated
on rank 0 and sent over the existing TP NCCL/RoCE communicator. Matching,
contiguous rank-1 writes are received directly into their final CUDA parameter
storage rather than through a second full-size packed allocation.

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
- Rank 0 uses one reusable 64 MiB CUDA staging window by default. Every logical
  source view is split into element-aligned frames at or below that hard cap;
  a non-contiguous TP view is sliced before copying, so it cannot silently
  materialize an oversized contiguous transport buffer.
- Matching contiguous writes use PyNccl/NCCL receive directly into final rank-1
  parameter storage. Dtype-converting, non-contiguous, and small broadcast
  writes use one lazy fixed-size receive window and then copy locally.
- `DSPARK_ROCE_LOAD_BUFFER_MB` controls the hard transport-frame and scratch
  bound. Shape-changing writes larger than one frame fail explicitly instead
  of falling back to an unbounded allocation.
- The bound applies to transport scratch. A model-specific recipe such as a
  padding `cat` or dtype conversion can still materialize its one complete
  logical write on rank 0 before that write is chunked. Writes are processed
  and released one at a time; `max_write_bytes` makes this distinct from the
  fixed-frame bound in diagnostics.
- B12X and other
  `process_weights_after_loading` transformations still execute locally after
  the raw checkpoint writes have arrived.

The resulting traffic is close to rank 1's raw resident weight size, not the
full checkpoint size. It can be slightly larger or smaller than exactly half
because some parameters are replicated, experts can be rank-owned, and loader
post-processing can change the final representation.

## Build and enable

Build an immutable local candidate once on the head, based on the digest-pinned
production image, so the validated `0.1.1` image remains preloaded for rollback:

```bash
DEV_SHA='<full candidate commit>'
git checkout --detach "$DEV_SHA"
test "$(git rev-parse HEAD)" = "$DEV_SHA"
test -z "$(git status --porcelain)"
SHORT_SHA="$(git rev-parse --short=12 HEAD)"
DEV_IMAGE="dspark-vllm-gx10:dev-$SHORT_SHA"
PROD_REF='ghcr.io/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8'
sudo docker build --network none \
  --build-arg BASE_IMAGE="$PROD_REF" \
  --build-arg SOURCE_REVISION="$DEV_SHA" \
  --tag "$DEV_IMAGE" \
  --file - . <<'DOCKERFILE'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG SOURCE_REVISION
LABEL org.opencontainers.image.revision="${SOURCE_REVISION}"
COPY overlay/vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
DOCKERFILE
```

Transfer that exact image over the dedicated fabric and verify the Docker image
ID is identical on both nodes before launch. Do not build separately on the
worker or publish an untested development image to GHCR. Save/compress the
candidate on the head, copy its archive only over the fabric interface, load it
on the worker, and compare `docker image inspect --format '{{.Id}}'` on both
nodes. Keep the digest-pinned production image and its clean checkout intact.
The loader also negotiates an explicit protocol version, frame size, and
PyNccl-versus-ProcessGroup transport before the first checkpoint tensor. A
mixed image or asymmetric NCCL configuration therefore fails on the control
plane before any weight payload is sent.

Set the following values in both `config/head.env` and `config/worker.env`:

```dotenv
DSPARK_VLLM_IMAGE=dspark-vllm-gx10:dev-<12-char-commit>
DSPARK_WEIGHT_LOAD_FORMAT=roce_tp
DSPARK_ROCE_LOAD_BUFFER_MB=64
```

Start the worker first through the existing cluster launcher. Successful
startup logs include `rank 0 is the sole checkpoint reader`, `rank 1 will not
open checkpoint payload files`, the transferred GiB count, and the number of
bounded frames. Each model phase also emits `DSPARK_WEIGHT_LOAD` start,
complete, or failed records. A completion record is written only after the
current CUDA stream is synchronized, so its elapsed time covers weights being
resident in RAM rather than only the enqueue interval.

The completion counters are exact within the loader's application boundary:

- `source_bytes` is the logical size of checkpoint tensors consumed on rank 0,
  not a physical-disk byte counter;
- `traffic_bytes` is the tensor payload handed to NCCL for rank 1, excluding
  NCCL/RoCE/Ethernet overhead;
- `tensors` and `batches` count logical source tensors and bounded NCCL frames;
- `direct_bytes` and `staged_bytes` separate direct-to-parameter traffic from
  traffic that required the fixed rank-1 receive window;
- `max_frame_bytes` proves the configured hard transport bound, while
  `max_write_bytes` records the largest logical recipe payload before
  chunking (which can be smaller than a broadcast destination).

The dashboard sums the slower rank for each target/drafter phase as the
cluster-critical elapsed time and labels payload/time as an effective
application rate rather than wire throughput.

## Matched direct/RoCE timing

For the direct baseline, set `DSPARK_WEIGHT_LOAD_FORMAT=direct_timed` on both
nodes. This selects the unchanged default checkpoint loader with the same outer
timer and final CUDA-stream synchronization as `roce_tp`. Run `direct_timed`
and `roce_tp` with the exact same model, image digest, vLLM settings, node
roles, and declared cache policy. Start the worker first in both cases. The
dashboard only compares samples that completed on both ranks and reached API
readiness; ordinary `auto` timings remain visible but are not comparison
eligible.

Cache state can dominate a startup benchmark. Either declare a warm-cache
comparison and warm both nodes consistently, or declare a cold-cache run and
evict the relevant model pages on both nodes before each mode. Do not present a
direct cold run against a RoCE warm run as a loader speedup. Rank-1 payload is
expected to be near its resident weight size, not necessarily exactly half of
the checkpoint.

## Guardrails

The loader fails early unless all of these are true:

- tensor parallel size is exactly 2;
- pipeline parallel size is exactly 1;
- model parameters are on CUDA and the TP NCCL communicator is initialized;
- the underlying checkpoint format is `auto`, `hf`, `safetensors`, or `pt`.

Use an external startup timeout and a complete head-then-worker stop barrier
during testing. The startup control protocol uses blocking point-to-point
operations; a process, storage, or transport failure can otherwise wait for the
distributed runtime's own timeout before the peer is torn down.

The loader deliberately does not use vLLM's RL weight-transfer engine. That
engine is initialized only after startup model loading and broadcasts
checkpoint-format tensors rather than rank 1's exact TP writes.

## Rollback

Set `DSPARK_WEIGHT_LOAD_FORMAT=auto` on both nodes and restart to restore the
original behavior while keeping the feature image. For a complete image
rollback, also restore:

```dotenv
DSPARK_VLLM_IMAGE=dspark-vllm-gx10:prod-0.1.1
```

This local tag should already point at the verified digest-pinned `0.1.1`
image on both nodes. No checkpoint files or formats are modified, so rollback
requires no data conversion or cleanup. Stop the candidate head and worker
completely, then start the production worker before the production head.
