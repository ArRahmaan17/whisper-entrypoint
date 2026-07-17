# Whisper Docker

NVIDIA GPU-first Whisper transcription stack for Docker.

This project is intended to provide a production-oriented speech-to-text application with:

- NVIDIA CUDA acceleration by default
- Docker Compose deployment
- FastAPI backend
- Dedicated GPU worker
- Custom web UI
- Multi-file job queue
- Persistent model and output storage
- CPU fallback only when explicitly allowed

The detailed implementation plan is maintained in [PLAN.md](./PLAN.md).

## Current Status

This repository is currently in the planning/scaffolding stage.

Planned services:

- `frontend` — Next.js web UI
- `backend` — FastAPI API server
- `worker` — CUDA-enabled Whisper worker
- `redis` — queue backend
- `postgres` — persistent metadata database

Only the worker should receive GPU access.

## Target Architecture

```text
Browser
  |
  v
Next.js frontend
  |
  | REST API / WebSocket
  v
FastAPI backend
  |
  | enqueue / status / metadata
  v
Redis queue + PostgreSQL
  |
  v
Whisper worker
  |
  | CUDA / PyTorch / FFmpeg
  v
NVIDIA GPU
  |
  v
Persistent outputs
```

## GPU and Docker Safety Policy

The worker must use official NVIDIA CUDA images and should not modify GPU hardware settings.

Supported Ubuntu amd64 targets:

| Ubuntu version | NVIDIA distro codename | Docker tag suffix |
| --- | --- | --- |
| Ubuntu 26.04 LTS | `ubuntu2604` | `ubuntu26.04` |
| Ubuntu 24.04 LTS | `ubuntu2404` | `ubuntu24.04` |
| Ubuntu 22.04 LTS | `ubuntu2204` | `ubuntu22.04` |

Preferred worker image:

```text
nvidia/cuda:13.3.0-runtime-ubuntu26.04
```

Fallback images:

```text
nvidia/cuda:13.3.0-runtime-ubuntu24.04
nvidia/cuda:13.3.0-runtime-ubuntu22.04
```

Worker safety rules:

- Do not run the worker as `privileged`.
- Use only `NVIDIA_DRIVER_CAPABILITIES=compute,utility`.
- Do not install NVIDIA host drivers inside the container.
- Do not change GPU clocks, fan speeds, voltage, persistence mode, MIG mode, or power limits from the app.
- Run one GPU job at a time by default.
- Monitor GPU temperature, utilization, and VRAM.
- Stop scheduling new GPU jobs when configured temperature or VRAM safety thresholds fail.

## Host Requirements

Expected host requirements:

- Ubuntu LTS amd64 host supported by NVIDIA
- NVIDIA GPU
- NVIDIA driver installed on the host
- NVIDIA Container Toolkit installed on the host
- Docker or Docker Compose

Before running GPU workloads, the host should pass:

```bash
nvidia-smi
```

And Docker should pass:

```bash
docker run --rm --gpus all nvidia/cuda:13.3.0-runtime-ubuntu26.04 nvidia-smi
```

If the preferred Ubuntu 26.04 CUDA image is not compatible with the installed Docker/NVIDIA/PyTorch environment, use the Ubuntu 24.04 or 22.04 fallback image from the supported list.

## Planned Features

### Transcription

- OpenAI Whisper engine for the first release
- Future engine abstraction for faster-whisper, CTranslate2, or TensorRT
- CUDA by default
- CPU fallback when configured
- Persistent model cache

### Web UI

- Upload audio/video files
- Configure Whisper parameters
- Track queue and job progress
- View transcripts and structured outputs
- Download generated files

### Output Formats

Planned output formats:

- `txt`
- `json`
- `srt`
- `vtt`
- `tsv`

### Supported Media

Planned audio support:

- `mp3`
- `wav`
- `flac`
- `ogg`
- `m4a`
- `aac`

Planned video support:

- `mp4`
- `mkv`
- `mov`
- `avi`
- `webm`

## Planned API

Core endpoints:

```http
GET /health
GET /system/gpu
GET /system/workers
GET /models
POST /uploads
POST /jobs
GET /jobs
GET /jobs/{job_id}
POST /jobs/{job_id}/cancel
DELETE /jobs/{job_id}
GET /jobs/{job_id}/outputs/{format}
GET /jobs/{job_id}/outputs.zip
WebSocket /ws/jobs/{job_id}
```

## Configuration

Planned environment variables:

```text
APP_ENV=development
DATABASE_URL=postgresql+psycopg://...
REDIS_URL=redis://redis:6379/0
UPLOAD_DIR=/app/uploads
OUTPUT_DIR=/app/outputs
MODEL_CACHE_DIR=/app/models
DEFAULT_DEVICE=cuda
ALLOW_CPU_FALLBACK=true
MAX_UPLOAD_MB=2048
MAX_CONCURRENT_GPU_JOBS=1
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility
GPU_TEMP_WARN_C=80
GPU_TEMP_STOP_C=87
GPU_MIN_FREE_VRAM_MB=1024
DEFAULT_MODEL=base
DEFAULT_OUTPUT_FORMATS=json,srt,txt
```

## Development Roadmap

1. Docker and GPU foundation
2. Backend API and persistent storage
3. Web UI MVP
4. Live progress, worker heartbeat, and operational hardening
5. Multi-GPU and scheduling improvements
6. Engine upgrades such as faster-whisper or TensorRT

See [PLAN.md](./PLAN.md) for acceptance criteria and implementation details.

## References

- NVIDIA Driver Installation Guide: https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/introduction.html
- NVIDIA CUDA Docker images: https://hub.docker.com/r/nvidia/cuda
- OpenAI Whisper: https://github.com/openai/whisper
