# Dashboard setup

Run `dashboard/run-dashboard.sh` on the head node. The dashboard is shipped in
this repository but is deliberately separate from the vLLM files under
`overlay/`: it reads vLLM Prometheus metrics locally and optionally uses SSH
for worker telemetry.

Quick start:

```bash
cp dashboard/dashboard.env.example dashboard/dashboard.env
# Edit dashboard/dashboard.env if worker telemetry is required.
./dashboard/run-dashboard.sh
```

Required environment variables for LAN access:

```bash
export DASHBOARD_BIND=0.0.0.0
export DASHBOARD_PORT=11001
export VLLM_METRICS_URL=http://127.0.0.1:8888/metrics
export DASHBOARD_HEAD_LABEL=SPARK-head
export DASHBOARD_WORKER_LABEL=SPARK-worker
export DASHBOARD_WORKER_SSH=user@10.200.0.2
export DASHBOARD_WORKER_IDENTITY_FILE=$HOME/.ssh/dashboard_telemetry
./dashboard/run-dashboard.sh
```

To start it automatically at boot on a systemd-based Spark/GX10 host:

```bash
./scripts/install-dashboard-service.sh
# The first run creates dashboard/dashboard.env and asks you to edit it.
# Run the installer a second time to enable and start the service.
```

The worker SSH key should be restricted to telemetry access. Do not commit the
key. If `DASHBOARD_WORKER_SSH` is empty, the dashboard still displays local GPU
and vLLM metrics.

The optional load-state and NVMe-temperature features use non-interactive
`sudo docker logs` and `sudo nvme smart-log`. Configure narrowly scoped sudoers
rules if those cards are required; otherwise they degrade to unavailable.

## Weight-load diagnostics

The RAM weight-ingress panel reads structured startup records from both
container logs. It reports the selected path, the sum of the slower TP rank for
each target/drafter phase, and (for `roce_tp`) the exact application payload
sent to rank 1. A partial or mixed-mode result is displayed as a diagnostic but
is not admitted to the direct/RoCE comparison.

Use `DSPARK_WEIGHT_LOAD_FORMAT=direct_timed` for the direct half of a matched
A/B. It runs the unchanged default local loader and adds the same outer timer
and final CUDA-stream synchronization used by `roce_tp`. The normal `auto`
loader remains the production default and rollback path. Its existing vLLM
`Loading weights took` messages are displayed as a useful fallback, but their
timer boundary is different and they are never treated as an A/B sample.

The dashboard admits a sample only after the API is ready and both ranks have
reported a complete synchronized run. It keeps the newest ready direct and
RoCE samples in dashboard-process memory so the comparison appears after the
second run; restarting the dashboard clears that history. For a fair test, use
the same model, image digest, settings, node order, and cache policy, starting
the worker first for each run. These identities are operator-enforced: the
dashboard labels what it observed in this process but cannot prove that model,
image, settings, or page-cache state matched between separate launches.

`DASHBOARD_LOAD_LOG_TAIL` defaults to 160 lines. The value matches the narrow
legacy sudoers rule used by this deployment; changing it may require a matching
sudoers update.

Traffic counters have deliberately narrow meanings:

- `sourceBytes` is the logical byte size of checkpoint tensors consumed by the
  loader. It is not measured physical disk I/O.
- `trafficBytes` is the exact tensor payload passed to the rank-1 NCCL sends.
  It excludes NCCL protocol, RoCE, Ethernet, and link-layer overhead.
- The displayed GiB/s is effective payload divided by the whole synchronized
  model-load phase. It is not an instantaneous wire-rate measurement.
