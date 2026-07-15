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
