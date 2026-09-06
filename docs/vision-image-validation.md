# Real-image correctness: five synthetic categories pass

Latest follow-up: the v3 run passed all 12 remaining chart, document, spatial
and two-image requests. Together with three reused OCR passes on the identical
image/configuration, all five synthetic categories pass first-touch plus two
repeats. One chart answer used the explicitly permitted whole-response JSON
fence; the other 11 new answers were unwrapped JSON. Exact values and types,
normal finishes and exclusive server counters passed in every request.
The earlier v1/v2 failures below remain unchanged. This is bounded synthetic
image-content evidence, not broad visual quality, text non-regression or the
>80 TPS text goal. Separate visual timing is partial, as reported below.

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

## Follow-up: v2 OCR repeats and chart presentation

The same source, image, model and server profiles were reused. The v2 prompt
specified JSON value types. All three OCR requests (first-touch warmup and two
measured repeats) returned the exact expected object, with numeric units.
All had normal streaming finishes and exact exclusive server counters.

The first chart response was:

````text
```json
{
  "largest": "South",
  "total": 54
}
```
````

Both values are correct. The Markdown fence failed the predeclared unwrapped
JSON requirement, so the benchmark stopped and the original text service was
restored. It is not an image-reading error, a CUDA failure or a passed suite.
Chart repeats, invoice, spatial, multiple-image and throughput cases did not
run. No repeated text baseline or second candidate startup was performed.

OCR first-touch TTFT was 8.747 s; the two repeats were approximately 0.431 s
each. All emitted only 23 tokens. The chart emitted 22 tokens. Its single
server-decode rate above 80 TPS is **not** the requested repeated 512-token
text result and is not a vision-throughput claim.

The next v3 specification keeps every prompt, expected value and image hash
unchanged. It explicitly permits a single whole-response JSON fence for the
image-content gate and records strict presentation compliance separately.
Raw responses remain intact; wrong values/types, duplicate keys, surrounding
prose and multiple answers still fail. Strict JSON is the client default for
fixtures that do not opt in. Forty-three local tests pass. Offline rescoring
diagnoses the v2 wrapper; it does not create new live v3 measurements.

The [v2 result and diagnostic evidence](../benchmarks/results/vision-ocr-chart-20260906.json)
retain original per-request pass/fail states, exact source and image identities,
small-output timing caveats, recovery evidence and raw-artifact hashes.

## Follow-up: v3 content pass and partial visual timing

The unchanged `d768bb8b2778` image and role profiles passed all 12 remaining
image requests in 21.307 seconds: chart, invoice, spatial layout and two-image
comparison, each with first-touch plus two repeats. All values/types matched;
all responses finished normally with exact exclusive server counters. The
first chart answer had one permitted JSON fence; the other 11 were unwrapped
JSON. Three unchanged OCR passes from v2 were reused, not rerun or relabelled.
This establishes 15/15 content passes across five synthetic categories, not
broad visual quality, long-context stability or text-speed non-regression.

The separate timing run requested 128 output tokens, temperature zero,
thinking off, seed 5205, three measured trials at C1/C2, with excluded warmup.
It was interrupted at 147.975 seconds after observed throughput projected
approximately 465 seconds for the full matrix, beyond its 300-second cap.
There were 38 completed passing waves (56 requests) and one interrupted,
unscored wave. No request-content or exact-counter gate failed in a completed
wave. This is not a complete 15-case visual benchmark.

The following are measured medians for **512-pixel** images only:

| Content | C1 server decode tok/s | C1 total output tok/s | C1 TTFT (s) | C2 total output tok/s |
|---|---:|---:|---:|---:|
| OCR description | 49.38 | 43.44 | 0.374 | 53.80 |
| Chart description | 53.17 | 46.56 | 0.361 | 54.57 |
| Document description | 49.27 | 44.03 | 0.355 | 54.13 |
| Spatial description | 51.11 | 44.62 | 0.360 | 55.21 |
| Two-image comparison | 59.66 | 50.24 | 0.420 | incomplete |

Each displayed timing cell has three measured trials. The two-image C2 case
has only one scored measured trial (63.46 aggregate tok/s), excluded from the
three-trial comparison. All ten 1024/2048-pixel timing cases remain unrun.
Raw partial waves remain in the evidence rather than being averaged as zeros
or silently discarded. These vision descriptions are not the 512-token text
explanation workload. The >80 text target remains unmet.

Exact images repeat after first touch with caches enabled. Warmed TTFT is not
cold encoder latency, and encoder GPU time is unavailable. Image visibility,
image sliding-window and W4A16 MoE shapes cold-JITed in warning mode; these
first-touch delays are preserved separately. There was no inference route-pack
JIT or fatal engine/CUDA/NCCL signature in the captured candidate logs.

The original text control was restored and verified within 20m40s of the
first stop; recorder cleanup and lock release finished at 21m04s. This is the
text-service interruption window, not continuous HTTP downtime. Both original
image IDs and role-profile hashes match, all four HTTP endpoints returned 200,
and streaming/tool calls passed. No text baseline was repeated. Images, models
and logs were retained. The cluster is left on the original text control.

The [v3 result](../benchmarks/results/vision-correctness-v3-20260906.json)
contains per-request correctness, partial timing medians/ranges, excluded
warmups, exact image/source/configuration identities, both-rank memory and JIT
summaries, recovery evidence and raw-artifact hashes. No interval thermal trace
was collected for timing; the 50/51 C readings are post-recovery snapshots only.
The full timing matrix needed about eight minutes at observed pace, so the
five-minute timing estimate was too optimistic. Future timing decisions must
budget from these measured rates and reuse completed cells.
