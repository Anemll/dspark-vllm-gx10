# Streaming and agent benchmark evidence

The clients use only the Python standard library. Run these on an otherwise
idle server and preserve results under the ignored `.local/results/` tree.
Keep model weights, scheduler settings, speculation, graph mode, concurrency,
output limits, and seeds matched between control and candidate. Record the
actual runtime inspection report and both-rank logs alongside these files.

The original explanation workload remains available:

```sh
python3 benchmarks/benchmark_dsv4_api.py \
  --base-url http://spark-head.local:8888 --concurrency 1,2,4 \
  --trials 2 --max-tokens 512 --timeout 900 \
  --cache-state repeated-prompt --output .local/results/control-decode.json
```

The new workload is a synthetic coding proxy, **not NVIDIA SpeedBench** and
not a coding quality benchmark. Every initial coding request is verified as
8192 tokens by the server's `/tokenize` endpoint, including the chat template.
The inference usage must match that count. Changing `--target-tokens` records
and verifies the requested length; the default mode name does not override it.
If exact sizing fails, the run fails before inference instead of reporting an
approximate prompt as exact. Tokenizer preparation has its own deadline.

```sh
python3 benchmarks/benchmark_agent.py \
  --base-url http://spark-head.local:8888 --mode tool-smoke \
  --concurrency 1 --trials 1 --max-tokens 256 --timeout 120 \
  --output .local/results/control-tools.json

python3 benchmarks/benchmark_agent.py \
  --base-url http://spark-head.local:8888 --mode coding-8k \
  --concurrency 1,2,4 --trials 2 --max-tokens 512 --seed 4104 \
  --cache-state unique-prefix-intended \
  --output .local/results/control-coding.json

python3 benchmarks/benchmark_agent.py \
  --base-url http://spark-head.local:8888 --mode multi-turn \
  --concurrency 1 --trials 2 --max-tokens 512 --seed 4104 \
  --output .local/results/control-agent.json
```

The multi-turn fixture runs four requests per conversation in order: initial,
exact repeat, a growing fixed-history turn, and a divergent suffix. It replays
the fixture's assistant history instead of feeding generated answers into the
next prompt; this keeps A/B inputs fixed. The initial prompt is 8192 tokens;
later turns are longer and their exact counts are saved. The report separates
phase latency, observed cached/computed prompt tokens where available, and
whether the exact-repeat visible answer matches the initial answer. A text
presence check or repeat comparison is not proof of numerical correctness.

The tool smoke requires one parsed `read_file` call with exactly
`{"path":"src/cache.py"}` and `finish_reason=tool_calls`. No tool is executed.
It uses automatic tool selection to exercise the model/parser together.
Reasoning-only output fails text validation; increase the output budget when
the model needs more tokens to reach a visible answer or tool call.

Reuse the saved exact messages, tool schemas, sampling settings, token hashes,
and concurrency/trial cases for a candidate:

```sh
python3 benchmarks/benchmark_agent.py \
  --base-url http://spark-head.local:8888 --model deepseek-v4-flash-dspark-abliterated \
  --max-tokens 512 --replay-report .local/results/control-coding.json \
  --cache-state candidate-after-restart \
  --output .local/results/candidate-coding.json
```

Replay uses saved request bodies and case scheduling, ignoring new workload,
concurrency, trial, seed, and target-length defaults. The supplied model and
output budget must match. Every token hash is reverified; a changed tokenizer
or template prevents a supposedly matched comparison. Model/tokenizer
revisions remain `unknown` unless supplied with `--model-revision` and
`--tokenizer-revision`; `/version` and `/v1/models` are archived as evidence,
but those endpoints do not necessarily expose checkpoint revisions.

Reports retain the legacy throughput fields while adding raw SSE payloads,
arrival times, content, reasoning, tools, usage, validation errors, exact
request hashes, and client source hashes. SSE chunks may contain multiple
tokens. `token_tps` and `tpot_proxy_s` use first-to-last output-chunk timing
and are explicitly proxies; true per-token timing requires server evidence.
One-chunk streams have no meaningful first-to-last decode rate and report
null. `ttft_s` begins with the first nonempty content/reasoning/tool delta;
`content_ttft_s` measures visible prose separately. This broadens the original
content-only TTFT behavior when reasoning or tools appear.

Aggregate throughput counts only successfully validated completion tokens,
divided by whole-trial wall time. Failed requests remain in wall time and raw
results. Summary medians include all completed trials, with request/error
counts. p95 appears only with at least 20 observations and p99 only with at
least 100; small smoke runs cannot establish reliable tail latency.

Cache labels express intent, not measured state, and neither client flushes
the server cache. Initial coding prompts vary early by deterministic case ID;
the same seed on an already warm server can still hit the previous run's
prefix cache. Compare the usage counters or bracket requests with the existing
prefill harness's server metrics. Missing cache usage is null, not zero.
No speculation acceptance score is inferred from SSE chunks.

Each completed request checkpoints its partial report atomically. Errors stop
subsequent trials; already-running peer requests complete within their request
budgets. Streaming responses have both an idle timeout and a whole-stream
socket watchdog. The watchdog bounds body streaming; OS DNS resolution and
connection establishment still depend on the platform network stack. Failed
or interrupted runs retain their status and completed request evidence.

Run the offline contract tests:

```sh
python3 -m unittest discover -s tests -p 'test_benchmark*.py' -v
```

## Known-answer prefix-cache contract

`benchmark_prefix_contract.py` sends eight sequential requests across exact
1023/1025-token prompt families. It checks fixed JSON state values after a
repeat, growing turn, and divergent branch. Each request is bracketed by server
metrics: cold inputs must be uncached, and replay/branches must have measured
cache hits. Correct free-form prose or identical text alone is not the gate.
Use a new seed for a new cold run; retain the same seed after a candidate
restart for a matched comparison. A cold wrong answer is not labelled cache
corruption, and missing/contaminated metrics are inconclusive.

```sh
python3 benchmarks/benchmark_prefix_contract.py \
  --base-url http://spark-head.local:8888 --model MODEL_ID \
  --seed 4106 --timeout 30 --budget-seconds 240 \
  --output .local/results/control-prefix-contract.json
```

## Model-free GPU diagnostics

`summarize_cuda_trace.py TRACE.json.gz` summarizes an existing per-rank Torch
trace without contacting or importing the GPU runtime. It assigns kernels to
`execute_context_*` forwards by CPU launch correlation, **not** overlapping CPU
and GPU timestamps: asynchronous execution may lag by several forwards. It
retains unmatched/ambiguous kernels explicitly. Kernel-duration sums include
stream overlap and are not request wall time or uninstrumented throughput.

The experimental narrow-graph lifetime canary is blocked by the installed
constructor on the current pinned runner. Its earlier synthetic replays did
not exercise the runner's metadata-free PIECEWISE capture. Do not use those
replays to waive the required real-runner cache-write checks; see the
[integration report](../docs/ifa26-native-integration.md#graph-integration-blocked).

`benchmark_sparse_mla.py` compares identical active entries with different
sentinel padding. Eager timings include host enqueue gaps; graph mode captures
a batch of calls and reports per-call device time. These hot synthetic buffers
do not measure model throughput or prove full-model graph correctness. Results
record checks after changed-input and timed replays, atomic partial evidence,
fresh autotuner storage, tactics, source and binary hashes. Output must be new.
Reusing `--compiled-provenance` requires the coordinator to verify the same
immutable image and supply `--image-id`; preserve Docker inspect evidence next
to the report. The client cannot introspect the host image ID. See
[`patches/flashinfer/README.md`](../patches/flashinfer/README.md) for isolated
assembly and hard outer timeouts.

`benchmark_mhc_warmup.py` is a bounded, model-free canary for the mHC warmup
overlay. It requires a compatible SM121 runtime, explicit baseline/candidate
source files, and a new output directory. It checks synthetic target/draft
dispatch, finite outputs, unchanged parameters, isolated TileLang cache roots,
generated-library provenance and the previously missed 294-token norm path.
It does not establish model quality, TP=2 correctness, or zero JIT from timing
alone. Inspect the phase-specific cache/compiler evidence before deployment.

The exact-token prefill client also rejects missing/inconsistent usage,
metadata-only streams, missing finish reasons, and missing `[DONE]`. Its TTFT
starts at the first nonempty completion text, including a whitespace token.
Successful result fields remain compatible with the existing comparison tool.
Warmups and measured requests now save raw stream evidence, prompt IDs, and
before/after counters incrementally; a failed later request preserves earlier
results. Server throughput is withheld if request/prompt counter deltas do not
isolate one matching request, or counters decrease. Run its dedicated tests
with `python3 -m unittest discover -s tests -p 'test_prefill_benchmark.py' -v`.
