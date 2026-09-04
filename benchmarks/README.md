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

The exact-token prefill client also rejects missing/inconsistent usage,
metadata-only streams, missing finish reasons, and missing `[DONE]`. Its TTFT
starts at the first nonempty completion text, including a whitespace token.
Successful result fields remain compatible with the existing comparison tool.
Warmups and measured requests now save raw stream evidence, prompt IDs, and
before/after counters incrementally; a failed later request preserves earlier
results. Server throughput is withheld if request/prompt counter deltas do not
isolate one matching request, or counters decrease. Run its dedicated tests
with `python3 -m unittest discover -s tests -p 'test_prefill_benchmark.py' -v`.
