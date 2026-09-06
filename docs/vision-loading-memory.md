# Vision loading memory evidence

Status, 2026-09-06: full Vision-Exp inference is still unverified. The first
TP2 startup was stopped under its predeclared swap-out gate; the accepted
0731 text service was restored and verified. A subsequent non-disruptive
diagnosis made no service, cache or system-tuning changes.

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
