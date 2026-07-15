#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

dashboard_dir="$(cd "$(dirname "$0")" && pwd)"
env_file="${DASHBOARD_ENV_FILE:-$dashboard_dir/dashboard.env}"

if [[ -f "$env_file" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
fi

exec python3 "$dashboard_dir/server.py"
