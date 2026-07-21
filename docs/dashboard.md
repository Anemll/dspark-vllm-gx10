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

This foreground mode is useful for development. For a persistent monitor that
can also read startup progress from the active Compose container, use the
systemd installer instead:

```bash
./scripts/install-dashboard-service.sh
# The first run creates dashboard/dashboard.env and exits.
# Edit that file, then install and start the service:
./scripts/install-dashboard-service.sh
sudo systemctl --no-pager --full status dspark-live-dashboard.service
```

Required environment variables for LAN access:

```bash
export DASHBOARD_BIND=0.0.0.0
export DASHBOARD_PORT=11001
export VLLM_METRICS_URL=http://127.0.0.1:8888/metrics
export DASHBOARD_HEAD_LABEL=SPARK-head
export DASHBOARD_WORKER_LABEL=SPARK-worker
export DASHBOARD_WORKER_SSH=user@WORKER_FABRIC_IP
export DASHBOARD_WORKER_IDENTITY_FILE=$HOME/.ssh/dashboard_telemetry
./dashboard/run-dashboard.sh
```

The worker SSH key should be restricted to telemetry access. Do not commit the
key. If `DASHBOARD_WORKER_SSH` is empty, the dashboard still displays local GPU
and vLLM metrics.

The installer deploys a root-owned, read-only container-log helper and grants
the dashboard user permission to invoke exactly that helper with the default
160-line bound. The helper discovers the sole running
`com.docker.compose.service=vllm-dspark` container, so Compose project and
container renames do not break startup progress after a reboot. The optional
NVMe-temperature feature still requires a narrowly scoped non-interactive
`sudo nvme smart-log` rule; otherwise that card degrades to unavailable.

## Startup and prepared W4A4 load progress

Start the dashboard service before starting the two vLLM ranks. Start the
worker/rank 1 server first and the head/rank 0 server second. During startup the
vLLM `/metrics` endpoint does not exist yet, so the dashboard's top-level
scrape indicator may show `STALE`; that is not a load failure. The two model
load cards are sourced independently from the bounded head and worker
container logs.

For the prepared W4A4 path, the rank logs provide one
`NVFP4_PREPARED event=layer_load` record per target layer. A valid bulk-reader
run begins with `event=enabled ... io_mode=preadv` and completes with
`event=complete layers=43 reads=344 copies=344 ... io_mode=preadv`. The normal
draft-load, graph-capture, and API-readiness phases follow. Once `/metrics`
becomes available, the dashboard switches to live request, prefill, decode,
KV-cache, and DSpark-acceptance data.

If either model-load card says `unavailable`, check these separately from API
health:

```bash
# HEAD: the restricted helper must find exactly one running Compose service.
sudo -n /usr/local/libexec/dspark-dashboard-container-logs 160 >/dev/null

# HEAD: the dashboard service must be able to use the configured worker key.
ssh -o BatchMode=yes -i /path/to/dashboard_key user@WORKER_FABRIC_IP \
  sudo -n /usr/local/libexec/dspark-dashboard-container-logs 160 >/dev/null

sudo journalctl -u dspark-live-dashboard.service -n 100 --no-pager
```

The installer grants only the exact local helper invocation with the configured
160-line bound. Install the same helper/sudoers rule on both nodes, and ensure
the worker key is accepted non-interactively with strict host-key checking.
Do not point `DASHBOARD_CONTAINER_NAME` at an ephemeral container name when the
helper is enabled; label-based discovery deliberately survives Compose and
container renames.

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

`DASHBOARD_LOAD_LOG_TAIL` defaults to 160 lines. The installed sudoers rule
allows exactly that value; changing it requires rerunning the installer with a
matching reviewed bound.

Traffic counters have deliberately narrow meanings:

- `sourceBytes` is the logical byte size of checkpoint tensors consumed by the
  loader. It is not measured physical disk I/O.
- `trafficBytes` is the exact tensor payload passed to the rank-1 NCCL sends.
  It excludes NCCL protocol, RoCE, Ethernet, and link-layer overhead.
- The displayed GiB/s is effective payload divided by the whole synchronized
  model-load phase. It is not an instantaneous wire-rate measurement.
