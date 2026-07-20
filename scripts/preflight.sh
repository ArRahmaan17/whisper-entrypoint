#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

command -v docker >/dev/null 2>&1 || fail "Docker is not installed."
docker info >/dev/null 2>&1 || fail "Docker daemon is unavailable or your user lacks permission."
docker compose version >/dev/null 2>&1 || fail "Docker Compose is unavailable."
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is unavailable; install a supported NVIDIA host driver first."

COMPOSE_CUDA_IMAGE="$(docker compose config --environment | awk -F= '$1 == "CUDA_IMAGE" {print substr($0, index($0, "=") + 1); exit}')"
CUDA_IMAGE="${CUDA_IMAGE:-${COMPOSE_CUDA_IMAGE:-nvidia/cuda:13.3.0-runtime-ubuntu24.04}}"

printf 'Host GPU:\n'
nvidia-smi --query-gpu=name,driver_version,memory.total,temperature.gpu,power.limit --format=csv,noheader

printf '\nValidating GPU passthrough with %s...\n' "$CUDA_IMAGE"
docker run --rm --gpus all \
  --security-opt=no-new-privileges:true \
  --cap-drop=ALL \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  "$CUDA_IMAGE" \
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

printf '\nGPU preflight passed. No clocks, voltage, fan, power, persistence, or MIG settings were changed.\n'
