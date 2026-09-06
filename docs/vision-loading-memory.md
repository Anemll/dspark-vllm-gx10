# Vision loading memory evidence

Status, 2026-09-06: full Vision-Exp inference is still unverified. The first
TP2 startup was stopped under its predeclared swap-out gate; the accepted
0731 text service was restored and verified. A subsequent non-disruptive
diagnosis made no service, cache or system-tuning changes.

## Follow-up DSpark5 decision and control recovery

A new Python-only candidate at `d3195176f9c9` was built once and loaded with
the same image ID on both ranks. Its separately compiled vision module is
byte-identical to the previously tested binary. Only a verified 1.5 MB delta
crossed the fabric; existing base layers were reused without rebuilding.

The new run measured five-second memory-pressure intervals. It was rejected
under its predeclared single-interval `full PSI >25%` loading gate. Both ranks
had ample available memory and no OOM. The head subsequently finished all
48 target shards in 137.42 s while failure evidence was being captured; the
worker finished its DSpark load (99 parameters) and reported 79.41 GiB model
allocation. Neither fact proves successful image inference or graph capture.

The mandatory **text control recovery also crossed this loading gate**:

| Observation window | Peak full PSI | Minimum available GiB | OOM kills |
|---|---:|---:|---:|
| Vision candidate, rank 0 | 35.19% | 31.54 | 0 |
| Vision candidate, rank 1 | 30.48% | 30.08 | 0 |
| Text recovery, rank 0 | 41.85% | 13.05 | 0 |
| Text recovery, rank 1 | 49.22% | 15.39 | 0 |

The recovery windows include warmup/serving and are longer than the interrupted
candidate windows: this is not a matched pressure A/B. Nevertheless, the same
gate would reject a successful known-good startup. Text target weights loaded
in 137.53 s, followed by successful health, streaming and tool checks. It is
incorrect to present the transient PSI rejection as a vision-specific OOM or
proof that the vision model cannot fit. The test remains rejected, not a pass.

This revealed a decision-protocol problem: an isolated startup-stall peak is
too conservative to be a capacity gate on this deployment. Before another
attempt, define loading viability from bounded progress and completion time,
hard memory/swap/disk floors, OOM/transport errors and sustained stalls that
prevent progress. Do not repeat the same known false-positive threshold or
relax serving stability. Actual image-content checks are still required.

[Recorded interval summary](../benchmarks/results/vision-loading-pressure-20260906.json)
contains observation windows and counters. Raw JSONL and both-rank logs are
retained in private experiment evidence. The worker's peer-closed traceback
occurred during the intentional head-first teardown, not before the abort
decision. Normal missing optional NCCL-plugin notices also occurred in the
control. A cold W4A16 JIT warning occurred during recovery's tool smoke, not
route-pack JIT; warning-mode recovery passed and no zero-JIT claim is made.

API interruption was 12m14s. Full recovery correctness was verified within
13m33s. The original text image/profiles remain active and unchanged; both
monitor pairs and the transfer listener stopped, and the cluster lock was
released. No image, checkpoint or failure log was deleted.

## Observations, not a fit verdict

Both ranks recognized the multimodal architecture and initialized. The abort
decision was made during checkpoint loading at 26/48 shards; progress reached
39/48 while failure evidence was being captured. Recorded swap-out samples
were 193168, 309420, 22352 and 0 KiB/s while available host memory was about
37 GiB. No OOM, CUDA or NCCL failure was found in the captured logs.

Swap also increased during the successful text rollback's checkpoint load.
The old vision run did not record interval memory PSI or cgroup OOM events,
so it cannot distinguish safe transient reclaim from unacceptable sustained
pressure. Its abort remains an abort, not a retrospective pass. This is not
evidence that Vision-Exp can never fit or that the vision wrapper leaks.

Pinned vLLM's safetensors loader uses per-shard lazy mmap and yields weights;
our wrapper retains that streaming iterator. No checkpoint-sized Python list
was introduced. Eager shard loading would add host allocation and is not an
accepted workaround. The SGLang recipe's drop-cache controls are not imported
into this vLLM deployment.

Linux swappiness describes relative reclaim costs, not a free-memory trigger.
Global cache dropping also has I/O/CPU costs; it is not an unexplained default
fix for this observation. See the [kernel VM documentation](https://www.kernel.org/doc/html/latest/admin-guide/sysctl/vm.html#swappiness).

## Read-only collector

`scripts/sample-node-memory.py` produces bounded JSONL with host availability,
swap, reclaim counters, memory PSI, optional process RSS/swap and optional
exact-container cgroup events. It uses the actual host page size. First,
missing or reset counters never become a fabricated zero rate. Interval
validity applies to derived counters; inspect snapshot errors and required
memory fields separately before making a decision.

For example, on a Linux node:

```bash
python3 scripts/sample-node-memory.py --samples 61 --interval 1 \
  --disk-path . --pid <verified-process-id> \
  --cgroup-dir <verified-container-cgroup> > <private-run-directory>/memory.jsonl
```

Use resolved process/cgroup identities from the specific running container.
Save the output in persistent, ignored evidence storage, not only a temporary
directory. The collector neither changes settings nor kills anything.

PSI `full` measures time when all non-idle tasks are simultaneously stalled.
The collector differences total microseconds over each actual sampling
interval, rather than treating a since-boot count or avg10 as an interval
measurement. See the [kernel PSI documentation](https://docs.kernel.org/accounting/psi.html).

Fourteen collector tests pass. Six passive samples per rank produced five
valid intervals each: zero swap-out, zero full-memory-stall time and zero
cgroup OOM/high/max event increments. Minimum available memory was 14417 MiB
on rank 0 and 16089 MiB on rank 1. These short idle observations are not a
model-load stress test and cannot reconstruct the earlier pressure event.

Before another full vision attempt, predeclare loading and serving pressure
gates, collect both ranks' interval evidence alongside shard progress, retain
absolute load/outage deadlines and preserve the operational text rollback.
Do not silently relax the earlier test's gate or run an unbounded diagnostic.
