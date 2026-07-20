#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

command -v docker >/dev/null 2>&1 || fail "Docker is not installed."
docker info >/dev/null 2>&1 || fail "Docker daemon is unavailable or your user lacks permission."
test -d /dev/dri || fail "/dev/dri is unavailable; install a supported Intel GPU host driver first."
docker compose --profile xpu config --quiet

printf 'Intel XPU host device preflight passed. Runtime capability still must pass the worker tensor smoke test.\n'
