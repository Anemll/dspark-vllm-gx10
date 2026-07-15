#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

[[ $# -eq 2 ]] || { echo "Usage: $0 IMAGE_TAG ENV_FILE" >&2; exit 2; }
tag="$1"
env_file="$2"
[[ -f "$env_file" ]] || { echo "Missing $env_file" >&2; exit 1; }
[[ "$tag" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid image tag: $tag" >&2; exit 1; }
image="ghcr.io/anemll/dspark-vllm-gx10:$tag"

python3 - "$env_file" "$image" <<'PY'
from pathlib import Path
import shutil
import sys
import time

path = Path(sys.argv[1])
image = sys.argv[2]
backup = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
shutil.copy2(path, backup)
lines = path.read_text().splitlines()
updated = []
found = False
for line in lines:
    if line.startswith("DSPARK_VLLM_IMAGE="):
        updated.append(f"DSPARK_VLLM_IMAGE={image}")
        found = True
    else:
        updated.append(line)
if not found:
    updated.append(f"DSPARK_VLLM_IMAGE={image}")
path.write_text("\n".join(updated) + "\n")
print(f"Updated {path}; backup: {backup}")
PY

sudo docker pull "$image"
echo "Image staged. Restart worker first, then head, with scripts/start-node.sh."
