#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

patterns=(
  'sk-[A-Za-z0-9_-]{16,}'
  '-----BEGIN [A-Z ]*PRIVATE KEY-----'
  '/Users/[^ /]+'
  '/home/anemll'
  '192\.168\.[0-9]+\.[0-9]+'
  '(password|passwd|api[_-]?key)[[:space:]]*=[[:space:]]*[^$<{][^[:space:]]+'
)

failed=0
for pattern in "${patterns[@]}"; do
  if grep -RInE --exclude-dir=.git --exclude='.git' \
      --exclude='check-private-data.sh' \
      --exclude-dir=.build --exclude-dir=__pycache__ --exclude='*.pyc' \
      -e "$pattern" .; then
    failed=1
  fi
done

if find . -type d \( -name .git -o -name .build -o -name __pycache__ \) \
    -prune -o -type f \
    \( -name '.env' -o -name '*.pem' -o -name '*.key' -o \
       -name 'id_rsa*' -o -name 'id_ed25519*' \) -print | grep -q .; then
  echo "Private-looking files found." >&2
  failed=1
fi

(( failed == 0 )) || { echo "Private-data check failed." >&2; exit 1; }
echo "Private-data check passed."
