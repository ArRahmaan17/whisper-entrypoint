#!/usr/bin/env bash
set -uo pipefail

detected=0
nvidia_reported=0

printf 'Detected GPU devices:\n'

if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia_output="$(nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader 2>/dev/null)" && [[ -n "$nvidia_output" ]]; then
    while IFS= read -r gpu; do
      printf '  NVIDIA %s -> profile=cuda (supported)\n' "$gpu"
      detected=1
      nvidia_reported=1
    done <<< "$nvidia_output"
  fi
fi

if command -v lspci >/dev/null 2>&1; then
  while IFS= read -r device; do
    normalized="${device,,}"
    case "$normalized" in
      *nvidia*)
        if (( nvidia_reported == 0 )); then
          printf '  %s -> profile=cuda (run make preflight-cuda)\n' "$device"
          detected=1
        fi
        ;;
      *amd/ati*|*advanced\ micro\ devices*)
        printf '  %s -> profile=rocm (experimental; run make preflight-rocm)\n' "$device"
        detected=1
        ;;
      *intel*)
        printf '  %s -> profile=xpu (experimental; run make preflight-xpu)\n' "$device"
        detected=1
        ;;
    esac
  done < <(lspci -D 2>/dev/null | awk 'BEGIN {IGNORECASE=1} /VGA compatible controller|3D controller|Display controller/')
fi

if (( detected == 0 )); then
  printf '  No supported GPU was detected with nvidia-smi or lspci.\n'
fi

cat <<'EOF'

Project worker profiles:
  cuda  NVIDIA CUDA  supported
  rocm  AMD ROCm     experimental
  xpu   Intel XPU    experimental; Whisper jobs require hardware validation
  cpu   CPU fallback supported (not a GPU profile)

Detection identifies candidate hardware; the matching preflight is the compatibility check.
EOF
