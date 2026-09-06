# First real image: correct OCR values, incomplete acceptance

On September 6, candidate source `d3195176f9c9` loaded
`DeepSeek-V4-Flash-Vision-Exp` and its five-token DSpark draft on both GB10
ranks. Both used image `sha256:d768bb8b27788bb046a10b54c7df7de7d0d0e3f26b1ee0c7aab04b2c69b9cd1b`.
No rebuild or dependency change was needed for this run. This is now a real
image-inference result, not only a processor or isolated kernel test.

The first 1024-pixel shipment image produced:

```json
{"code": "Q7M4", "units": "37", "destination": "Oslo"}
```

All three values match the image. The test expected numeric `37`, but its v1
prompt requested only the key names, without specifying value types. The
response is valid JSON; the failure is the string-versus-integer contract,
not a demonstrated wrong OCR value or image-kernel error. The original failed
artifact is unchanged. It is not relabelled as a passed correctness suite.

## What ran

- Both ranks loaded the full model and 99 draft parameters, reporting
  79.41 GiB each. Both completed initialization and graph capture.
- Health/version/models/dashboard returned HTTP 200; streaming and a parsed
  `get_weather` tool call passed.
- One first-touch OCR warmup completed normally with usage and stream-end
  markers. Server counters accounted for exactly that one request.
- The suite stopped on the JSON-type mismatch. OCR repeats, chart, invoice,
  spatial, multiple-image and vision-throughput tests did **not** run.

This request had 387 prompt tokens and 23 output tokens, with 8.529 s TTFT
and 8.853 s total time. These are diagnostic first-touch timings, not a
published vision-speed result or a matched text comparison. Image visibility
and image sliding-window kernels cold-JITed on both ranks; encoder time was
not isolated. No inference route-pack JIT or engine/CUDA/NCCL fatal signature
was found in the captured logs.

The progress-aware startup completed despite transient loading pressure.
Recorded minimum available memory was 13.68/16.16 GiB and minimum free swap
6.85/8.70 GiB on head/worker; host OOM kills were zero. Worker peak interval
full PSI reached 78.41%, but it was not a sustained no-progress stall. These
short-run observations are not long-context or extended-stability acceptance.

## Test-specification correction

`vision-correctness-v2` makes JSON value types explicit without including the
image answers. The invoice prompt also now correctly requests integer
**values**, rather than describing JSON object keys as integers. All 24 image
hashes are identical to the previous fixtures using Pillow 12.3.0. Throughput
prompts and text-baseline prompts are unchanged.

The validator reports syntax, type/structure and value failures separately;
it does not coerce quoted numbers into passing answers. Duplicate keys,
non-JSON constants and booleans in numeric fields are rejected. A correctness
response must finish normally, not at its token cap. Forty-one local unit
tests pass, including fixture generation. The corrected five-case request
manifest has been checked offline, not rerun live in this phase.

The next decision can reuse the preloaded candidate with the v2 specification
and a fresh bounded window. Require all five cases and repeats to pass before
vision timing. No serving-code patch is justified by this JSON-type result.
Vision may decode more slowly: correctness/stability are its gates. The
separate >80 TPS C1 text goal and matched text non-regression remain unmet.

## Recovery and evidence

The original `8c97ddf50b07` text control was restored, worker first after a
complete two-rank stop barrier. Both image IDs and original role-profile
hashes match. Health, streaming, tool calls and both-rank logs were checked.
Verified text recovery took at most 19m34s from the first stop; cleanup was
complete at 20m12s. This includes a 3m13s delay between the failed gate and
head stop while context was recovered. It is a text-service interruption
window, not uninterrupted HTTP downtime: the candidate API served requests
in between. The unchanged text benchmark was not repeated.

All temporary recorders stopped and owned locks were released. The accepted
text service remains running; candidate images, checkpoints and failure logs
were preserved. No second candidate startup occurred in this decision phase.

The [sanitized result](../benchmarks/results/vision-first-image-20260906.json)
contains exact provenance, the original response, diagnostic timings, memory
summaries and hashes of raw request/SSE, rank logs, identities and recovery
evidence. The full private evidence remains in the persistent experiment
directory.
