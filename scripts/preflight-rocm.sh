#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

command -v docker >/dev/null 2>&1 || fail "Docker is not installed."
docker info >/dev/null 2>&1 || fail "Docker daemon is unavailable or your user lacks permission."
test -c /dev/kfd || fail "/dev/kfd is unavailable; install a supported AMD ROCm host driver first."
test -d /dev/dri || fail "/dev/dri is unavailable."
docker compose --profile rocm config --quiet

printf 'ROCm host device preflight passed. The worker is not privileged and receives only /dev/kfd and /dev/dri.\n'
