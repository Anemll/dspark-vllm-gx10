#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
env_file="$root/dashboard/dashboard.env"
example="$root/dashboard/dashboard.env.example"
template="$root/dashboard/dspark-live-dashboard.service.in"
log_helper_source="$root/dashboard/read-container-logs.sh"
log_helper=/usr/local/libexec/dspark-dashboard-container-logs
sudoers_file=/etc/sudoers.d/dspark-live-dashboard
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

sudo install -d -m 0755 /usr/local/libexec
sudo install -o root -g root -m 0755 "$log_helper_source" "$log_helper"

sudoers_tmp="$(mktemp)"
trap 'rm -f "$tmp" "$sudoers_tmp"' EXIT
printf '%s ALL=(root) NOPASSWD: %s 160\n' "$(id -un)" "$log_helper" >"$sudoers_tmp"
sudo chmod 0440 "$sudoers_tmp"
sudo /usr/sbin/visudo -cf "$sudoers_tmp"
sudo install -o root -g root -m 0440 "$sudoers_tmp" "$sudoers_file"

sudo install -m 0644 "$tmp" "/etc/systemd/system/$service_name"
sudo systemctl daemon-reload
sudo systemctl enable --now "$service_name"
sudo systemctl --no-pager --full status "$service_name"
