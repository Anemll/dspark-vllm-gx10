#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
env_file="$root/dashboard/dashboard.env"
example="$root/dashboard/dashboard.env.example"
template="$root/dashboard/dspark-live-dashboard.service.in"
service_name=dspark-live-dashboard.service

if [[ ! -f "$env_file" ]]; then
  cp "$example" "$env_file"
  chmod 600 "$env_file"
  echo "Created $env_file"
  echo "Edit it, then rerun this installer."
  exit 2
fi

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
python3 - "$template" "$tmp" "$(id -un)" "$(id -gn)" "$root" <<'PY'
from pathlib import Path
import sys

source, target, user, group, root = sys.argv[1:]
text = Path(source).read_text()
text = text.replace("@USER@", user).replace("@GROUP@", group)
text = text.replace("@REPO_ROOT@", root)
Path(target).write_text(text)
PY

sudo install -m 0644 "$tmp" "/etc/systemd/system/$service_name"
sudo systemctl daemon-reload
sudo systemctl enable --now "$service_name"
sudo systemctl --no-pager --full status "$service_name"
