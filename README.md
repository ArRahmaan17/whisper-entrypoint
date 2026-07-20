# Whisper Docker

A Docker-first, local speech-to-text application powered by OpenAI Whisper. It includes a FastAPI API, a Next.js web interface, PostgreSQL metadata, Redis/RQ queues, persistent model storage, and capability-based worker routing.

The project is GPU-first, but it can run entirely on CPU. Python, Node.js, FFmpeg, PyTorch, and Whisper are installed inside Docker images—nothing from the application needs to be installed directly on the host.

## What Works Today

| Runtime | Status | How to start |
| --- | --- | --- |
| CPU | Ready and smoke-tested with `tiny.en` | `make up-cpu` |
| NVIDIA CUDA | Implemented; requires a working NVIDIA driver and Container Toolkit | `make up-cuda` |
| AMD ROCm | Experimental; requires compatible AMD hardware and host drivers | `make up-rocm` |
| Intel XPU | Experimental; engine compatibility remains disabled until hardware validation | `make up-xpu` |
| OpenVINO | Provider detection only; transcription engine not implemented | Not available yet |
| Apple MPS | Provider detection only; native macOS worker not packaged | Not available yet |

The application supports:

- Uploading up to 100 audio or video files per batch
- Automatic or manual accelerator selection
- Explicit and auditable device fallback
- Whisper model, language, task, precision, and decoding controls
- Queue progress and worker heartbeat monitoring
- `txt`, `json`, `srt`, `vtt`, and `tsv` outputs
- Persistent model cache, uploads, outputs, queue data, and database data

## Architecture

```text
Browser
  |
  v
Next.js frontend :3000
  |
  | REST API / WebSocket
  v
FastAPI backend :8000
  |
  +---- PostgreSQL (jobs and metadata)
  |
  +---- Redis/RQ (backend-specific queues)
             |
             v
       Whisper worker
             |
             v
      CUDA / ROCm / XPU / CPU
```

Only the selected worker receives access to its accelerator devices. Workers are not privileged.

## Prerequisites

Every deployment needs:

- A Linux container runtime on an amd64 host. Use Docker Engine on Linux or
  Docker Desktop with the WSL 2 backend on Windows.
- Docker Engine or Docker Desktop
- Docker Compose v2 (`docker compose`, not the legacy `docker-compose` command)
- Git
- Enough disk space for Docker images and downloaded Whisper models

Check Docker before continuing:

```bash
docker --version
docker compose version
docker info
```

CPU mode needs no GPU driver. CUDA, ROCm, and XPU mode each require the corresponding driver and device support on the host; those drivers are not installed inside the application container. On Windows, use the CPU or NVIDIA CUDA profile through Docker Desktop. The ROCm and XPU profiles require a native Linux host.

## Windows: Docker Compose Setup

This project does not run as a native Windows application. Docker Desktop runs
the existing Linux containers through WSL 2, while PowerShell controls the same
[`compose.yaml`](./compose.yaml) used on Linux. No Windows-specific Compose file
is required.

### 1. Install WSL 2 and Docker Desktop

In an Administrator PowerShell window, install WSL if it is not already
installed, then restart Windows when requested:

```powershell
wsl --install
```

Update an existing WSL installation with:

```powershell
wsl --update
wsl --version
```

Install [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/),
start it, and ensure **Settings → General → Use the WSL 2 based engine** is
enabled. Docker Desktop must be using **Linux containers**, not Windows
containers. Docker documents the WSL configuration in its
[WSL 2 backend guide](https://docs.docker.com/desktop/features/wsl/).

Open a normal PowerShell window and verify the Docker daemon and Compose:

```powershell
docker --version
docker compose version
docker info --format '{{.OSType}}'
```

The last command must print `linux`.

### 2. Clone and configure the project

```powershell
git clone https://github.com/ArRahmaan17/whisper-entrypoint.git
Set-Location whisper-entrypoint

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
}
notepad .env
```

In `.env`, replace `change-me` in both settings with the same strong,
URL-safe password (letters, numbers, `_`, and `-` are safe here):

```text
POSTGRES_PASSWORD=use-a-strong-password
DATABASE_URL=postgresql+psycopg://whisper:use-a-strong-password@postgres:5432/whisper
```

Validate the resolved Compose configuration:

```powershell
docker compose config --quiet
```

No output means the configuration is valid.

### 3. Start the CPU stack

Start with CPU mode to verify the complete application:

```powershell
docker compose --profile cpu up -d --build
docker compose --profile cpu ps
Invoke-RestMethod http://localhost:8000/health
```

Then open:

- Web UI: http://localhost:3000
- API documentation: http://localhost:8000/docs

Useful PowerShell operations are:

```powershell
# Follow the CPU worker log (Ctrl+C only stops following the log)
docker compose --profile cpu logs -f worker-cpu

# Stop the stack while preserving named volumes
docker compose down

# Start it again
docker compose --profile cpu up -d
```

The Linux `make` shortcuts are not required on Windows. If commands are run
from a WSL Linux terminal instead of PowerShell, the existing `make up-cpu` and
other Makefile targets remain available.

### 4. Optional NVIDIA CUDA profile

Docker Desktop GPU access on Windows requires its WSL 2 backend, a supported
NVIDIA GPU, current Windows NVIDIA drivers, and an up-to-date WSL kernel. See
Docker's [Windows GPU prerequisites](https://docs.docker.com/desktop/features/gpu/).
Do not install a Linux NVIDIA display driver inside WSL or this project.

After updating the Windows NVIDIA driver and running `wsl --update`, validate
GPU access from PowerShell using the same CUDA image configured by the project:

```powershell
nvidia-smi
docker run --rm --gpus all nvidia/cuda:13.3.0-runtime-ubuntu24.04 nvidia-smi
```

Only start the CUDA stack if that container test succeeds:

```powershell
docker compose down
docker compose --profile cuda up -d --build
docker compose --profile cuda ps
docker compose --profile cuda logs -f worker
```

The `make preflight` command is a Bash/Linux helper. The `docker run --gpus all`
check above performs the required GPU-passthrough check when using PowerShell.

## Quick Start: CPU

CPU mode is the simplest way to verify the complete application before configuring a GPU.

1. Clone the repository and enter it:

   ```bash
   git clone https://github.com/ArRahmaan17/whisper-entrypoint.git
   cd whisper-entrypoint
   ```

   If the repository is already cloned, run the remaining commands from its root directory.

2. Create the local configuration file:

   ```bash
   make setup
   ```

   This copies `.env.example` to `.env` only when `.env` does not already exist.

3. Open `.env` and replace `change-me` in both values below with the same password:

   ```text
   POSTGRES_PASSWORD=use-a-strong-password
   DATABASE_URL=postgresql+psycopg://whisper:use-a-strong-password@postgres:5432/whisper
   ```

4. Build and start the complete CPU stack:

   ```bash
   make up-cpu
   ```

   This starts the frontend, backend, PostgreSQL, Redis, and CPU worker. The first build downloads the application images and dependencies. The first transcription also downloads the selected Whisper model into the persistent `models` volume.

5. Confirm that the containers are healthy:

   ```bash
   docker compose --profile cpu ps
   curl http://localhost:8000/health
   ```

6. With the default `.env` ports, open the application:

   - Web UI: http://localhost:3000
   - API documentation: http://localhost:8000/docs

7. In the web UI:

   - Drop one or more supported audio/video files into the upload area.
   - Keep **Device policy** on **Auto select**, or choose **Manual → CPU**.
   - Choose a Whisper model. `tiny` or `tiny.en` is recommended for the first test.
   - Select the required output formats.
   - Click **Queue file** or **Queue files**.
   - Follow progress and download completed outputs from the activity panel.

## Quick Start: NVIDIA CUDA

CUDA is the preferred production path. The NVIDIA driver and NVIDIA Container Toolkit must be installed on the host. Do not install the host GPU driver in the Whisper container.

1. Complete the repository and `.env` setup from the CPU guide.

2. Confirm that the host driver can see the GPU:

   ```bash
   nvidia-smi
   ```

3. Run the project's read-only CUDA preflight:

   ```bash
   make preflight
   ```

   The preflight checks Docker, Compose, `nvidia-smi`, and GPU passthrough using the configured CUDA image. It does not change clocks, voltage, fan settings, power limits, persistence mode, or MIG configuration.

4. Start the complete CUDA stack:

   ```bash
   make up-cuda
   ```

5. Verify the API and worker:

   ```bash
   docker compose --profile cuda ps
   curl http://localhost:8000/health
    curl http://localhost:8000/system/accelerators
   ```

6. Open http://localhost:3000 and submit a small file with the `tiny` model first. The dashboard should show a fresh CUDA worker and the detected NVIDIA device.

If preflight fails, do not bypass it by making the container privileged. Fix the host driver or NVIDIA Container Toolkit configuration first.

## Selecting the NVIDIA CUDA Image

The default worker uses:

```text
nvidia/cuda:13.3.0-runtime-ubuntu24.04
```

The configured Ubuntu amd64 image options are:

| Ubuntu base | NVIDIA repository codename | `CUDA_IMAGE` value |
| --- | --- | --- |
| Ubuntu 26.04 LTS | `ubuntu2604` | `nvidia/cuda:13.3.0-runtime-ubuntu26.04` |
| Ubuntu 24.04 LTS | `ubuntu2404` | `nvidia/cuda:13.3.0-runtime-ubuntu24.04` |
| Ubuntu 22.04 LTS | `ubuntu2204` | `nvidia/cuda:13.3.0-runtime-ubuntu22.04` |

Ubuntu 24.04 is the project default. To use another listed base, update `.env` before building:

```text
CUDA_IMAGE=nvidia/cuda:13.3.0-runtime-ubuntu22.04
```

Then rebuild and start:

```bash
make preflight
make up-cuda
```

The host distribution and container distribution do not need to be identical. What matters is that the host NVIDIA driver supports the CUDA runtime selected by the container and that the preflight passes.

## Experimental AMD ROCm Runtime

ROCm mode expects compatible AMD hardware, a supported host driver, `/dev/kfd`, and `/dev/dri`.

```bash
make setup
make preflight-rocm
make up-rocm
docker compose --profile rocm ps
docker compose --profile rocm logs -f worker-rocm
```

The worker receives only `/dev/kfd` and `/dev/dri`; it is not privileged. This profile is experimental until a real transcription test passes on the target AMD hardware.

## Experimental Intel XPU Runtime

Intel XPU mode expects a supported Intel GPU driver and `/dev/dri` on the host.

```bash
make setup
make preflight-xpu
make up-xpu
docker compose --profile xpu ps
docker compose --profile xpu logs -f worker-xpu
```

The XPU profile currently validates packaging and device detection. OpenAI Whisper jobs are intentionally not advertised as XPU-compatible until real hardware validation succeeds.

## Changing Ports

The default host ports are:

- Frontend: `3000`
- Backend API: `8000`

If either port is occupied, update `.env` before building the frontend:

```text
FRONTEND_PORT=13000
BACKEND_PORT=18000
NEXT_PUBLIC_API_URL=http://localhost:18000
```

Compose automatically allows `http://localhost:${FRONTEND_PORT}` through CORS. Set
`CORS_ORIGINS` only when the browser uses a different origin.

Then start the selected profile. For example:

```bash
make up-cpu
```

Open `http://localhost:13000`; API documentation will be at `http://localhost:18000/docs`.

`NEXT_PUBLIC_API_URL` is compiled into the frontend image. Rebuild the frontend whenever this value changes.

## Daily Operations

### Check container status

Use the profile that is currently running:

```bash
docker compose --profile cpu ps
docker compose --profile cuda ps
```

### Follow logs

Application logs:

```bash
make logs
```

CPU worker logs:

```bash
docker compose --profile cpu logs -f worker-cpu
```

CUDA worker logs:

```bash
docker compose --profile cuda logs -f worker
```

Infrastructure logs:

```bash
docker compose logs -f postgres redis
```

Press `Ctrl+C` to stop following logs. This does not stop the containers.

### Stop the application

```bash
make down
```

This stops and removes project containers and networks but preserves all named volumes.

### Start it again

Run the command for the desired worker:

```bash
make up-cpu
```

or:

```bash
make up-cuda
```

### Switch worker backend

Stop the current stack before switching profiles:

```bash
make down
make up-cuda
```

Replace `up-cuda` with `up-cpu`, `up-rocm`, or `up-xpu` as needed.

### List compatible GPU profiles

```bash
make gpu-list
```

This maps GPUs detected through `nvidia-smi` or `lspci` to the CUDA, ROCm, or
XPU worker profile. Detection identifies candidate hardware; run the printed
preflight target to validate driver and container access.

### Rebuild after source or dependency changes

```bash
make rebuild PROFILE=cpu
```

Equivalent explicit targets are `rebuild-cuda`, `rebuild-cpu`, `rebuild-rocm`,
and `rebuild-xpu`. They rebuild and force-recreate the frontend, backend, and
selected worker without recreating PostgreSQL or Redis. To ignore Docker's
build cache, run:

```bash
make rebuild PROFILE=cpu REBUILD_FLAGS=--no-cache
```

Run `make help` to list all targets and options.

## Data and Persistence

Docker named volumes store:

| Volume | Contents |
| --- | --- |
| `uploads` | Uploaded source media |
| `outputs` | Generated transcript files |
| `models` | Downloaded Whisper model cache |
| `postgres-data` | Jobs, uploads, outputs, and worker metadata |
| `redis-data` | Persistent Redis queue state |

`make down` preserves these volumes. The following command deletes all project volumes and their data:

```bash
docker compose down -v
```

Only run it when you intentionally want to erase uploads, outputs, cached models, database records, and Redis data.

## Testing Without Local Python or Node.js

Run the backend test suite, frontend production build, TypeScript checks, and Compose validation:

```bash
make test
```

Verify that the CUDA worker image contains PyTorch, Whisper, and FFmpeg without giving the container GPU access:

```bash
make smoke-worker
```

Verify the CPU worker dependencies:

```bash
make smoke-cpu
```

Run a real one-second `tiny.en` CPU transcription inside an ephemeral container:

```bash
make smoke-cpu-transcription
```

The transcription smoke test downloads `tiny.en` into the temporary test container. It does not install Python packages or application dependencies on the host.

## API Workflow

The web UI is the recommended client. The same workflow can be performed through the API.

1. Upload a file:

   ```bash
   curl -X POST http://localhost:8000/uploads \
     -F "files=@/absolute/path/to/audio.mp3"
   ```

   Copy the `id` from the response.

2. Validate a CPU request:

   ```bash
   curl -X POST http://localhost:8000/jobs/validate \
     -H 'Content-Type: application/json' \
     -d '{
       "engine": "openai-whisper",
       "model": "tiny",
       "device": {
         "policy": "manual",
         "backend": "cpu",
         "device_index": null,
         "allow_fallback": false
       },
       "precision": "float32"
     }'
   ```

3. Create a job using the upload ID:

   ```bash
   curl -X POST http://localhost:8000/jobs \
     -H 'Content-Type: application/json' \
     -d '{
       "upload_ids": ["PASTE_UPLOAD_ID_HERE"],
       "engine": "openai-whisper",
       "model": "tiny",
       "device": {
         "policy": "manual",
         "backend": "cpu",
         "device_index": null,
         "allow_fallback": false
       },
       "precision": "float32",
       "parameters": {
         "language": null,
         "task": "transcribe",
         "output_formats": ["json", "txt", "srt"]
       }
     }'
   ```

4. Check the returned job ID:

   ```bash
   curl http://localhost:8000/jobs/JOB_ID
   ```

5. Download an output after the job completes:

   ```bash
   curl -OJ http://localhost:8000/jobs/JOB_ID/outputs/txt
   ```

Interactive endpoint documentation and schemas are available at `/docs`.

## API Endpoints

```http
GET       /health
GET       /system/gpu
GET       /system/resources
GET       /system/accelerators
GET       /system/compatibility
GET       /system/workers
GET       /system/workers/{worker_id}
GET       /models
POST      /uploads
POST      /jobs/validate
POST      /jobs
GET       /jobs
GET       /jobs/{job_id}
POST      /jobs/{job_id}/cancel
DELETE    /jobs/{job_id}
GET       /jobs/{job_id}/outputs/{format}
GET       /jobs/{job_id}/outputs.zip
WebSocket /ws/jobs/{job_id}
```

## Important Configuration

All settings are read from `.env` by Docker Compose.

| Variable | Default | Purpose |
| --- | --- | --- |
| `FRONTEND_PORT` | `3000` | Frontend host port |
| `BACKEND_PORT` | `8000` | API host port |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | API address used by the browser |
| `CORS_ORIGINS` | `http://localhost:${FRONTEND_PORT}` | Optional override for browser origins accepted by FastAPI |
| `DEFAULT_ENGINE` | `openai-whisper` | Default transcription engine |
| `DEFAULT_DEVICE` | `auto` | Default accelerator policy |
| `ACCELERATOR_PRIORITY` | `cuda,rocm,xpu,openvino,mps,cpu` | Auto-selection priority |
| `ALLOW_DEVICE_FALLBACK` | `true` | Allows fallback when requested by a job |
| `ALLOW_CPU_FALLBACK` | `false` | Global opt-in for CPU fallback |
| `MAX_CONCURRENT_JOBS` | `1` | Worker concurrency |
| `MAX_LOADED_MODELS` | `1` | Number of models retained by a worker |
| `CPU_THREADS` | `0` | CPU thread override; `0` uses runtime default |
| `MAX_UPLOAD_MB` | `2048` | Maximum size per uploaded file |
| `GPU_TEMP_WARN_C` | `80` | GPU warning threshold |
| `GPU_TEMP_STOP_C` | `87` | Stops admitting new jobs at this temperature |
| `GPU_MIN_FREE_VRAM_MB` | `1024` | Required free VRAM before admitting a job |
| `NVIDIA_VISIBLE_DEVICES` | `all` | NVIDIA devices exposed to the CUDA worker |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` | Minimal required NVIDIA capabilities |

See [.env.example](./.env.example) for the complete configuration.

## Supported Input and Output

Input audio:

- `mp3`, `wav`, `flac`, `ogg`, `m4a`, `aac`

Input video:

- `mp4`, `mkv`, `mov`, `avi`, `webm`

Output:

- `txt`, `json`, `srt`, `vtt`, `tsv`

FFmpeg runs inside the worker container to decode media.

## Troubleshooting

### The UI says `No worker`

Confirm that the selected profile is running and inspect its worker log:

```bash
docker compose --profile cpu ps
docker compose --profile cpu logs --tail=200 worker-cpu
curl http://localhost:8000/system/workers
```

Use `--profile cuda` and service `worker` for CUDA. A worker is considered unavailable when its heartbeat becomes stale.

### `make preflight` reports that `nvidia-smi` is unavailable

The NVIDIA host driver is missing, broken, or not visible in the current environment. Install or repair the supported host driver, confirm `nvidia-smi`, then install/configure NVIDIA Container Toolkit. Do not copy host driver packages into this project image.

### Docker cannot access the NVIDIA GPU

Run:

```bash
docker run --rm --gpus all \
  nvidia/cuda:13.3.0-runtime-ubuntu24.04 \
  nvidia-smi
```

If this fails, the problem is below the application layer—usually the host driver, Docker runtime, or NVIDIA Container Toolkit.

### A job remains queued

Check that a fresh worker is listening for the requested backend and that job validation succeeds:

```bash
curl http://localhost:8000/system/compatibility
curl http://localhost:8000/system/workers
```

Then inspect the backend and selected worker logs.

### The first job is slow to start

Whisper downloads the selected model on first use. Later jobs reuse the model from the persistent `models` volume. Larger models take more disk space, download time, VRAM, and RAM.

### CUDA runs out of memory

- Start with a smaller model.
- Keep `MAX_CONCURRENT_JOBS=1` and `MAX_LOADED_MODELS=1`.
- Leave `GPU_MIN_FREE_VRAM_MB` enabled.
- Stop other GPU-heavy applications.
- Do not raise GPU power limits or overclock the device for this application.

### Port `3000` or `8000` is already in use

Set `FRONTEND_PORT`, `BACKEND_PORT`, and `NEXT_PUBLIC_API_URL` as shown in [Changing Ports](#changing-ports), then rebuild the selected profile. Override `CORS_ORIGINS` only if the browser origin does not use `FRONTEND_PORT`.

### Reset only the containers

```bash
make down
make up-cpu
```

This preserves application data. Avoid `docker compose down -v` unless deleting all project data is intentional.

## Accelerator Safety

- No worker uses `privileged: true`.
- NVIDIA access is limited to `compute,utility` capabilities.
- ROCm receives only `/dev/kfd` and `/dev/dri`.
- XPU receives only `/dev/dri`.
- GPU jobs default to one at a time.
- The application does not modify clocks, voltage, fans, power limits, persistence mode, or MIG configuration.
- Temperature and free-VRAM admission checks prevent new CUDA jobs from starting outside configured limits.
- CPU fallback is disabled globally by default and remains visible in each job's requested-versus-actual audit fields.

These controls reduce application-level risk, but they do not replace correct cooling, a suitable power supply, supported host drivers, and normal hardware monitoring.

## Project Structure

```text
backend/                  FastAPI API, worker, providers, engines, scheduler, tests
frontend/                 Next.js web interface
scripts/                  Read-only accelerator preflight checks
compose.yaml              Services and accelerator profiles
.env.example              Runtime configuration template
Makefile                  Containerized lifecycle and test commands
PLAN.md                   Original CUDA-first product plan
Whisper-Multi-Accelerator-Implementation-Plan.md
                          Multi-accelerator implementation plan and status
```

## Current Limitations

- CUDA needs a real NVIDIA hardware test on the target host before production use.
- ROCm and XPU profiles are experimental.
- OpenVINO transcription and compiled-model caching are not implemented.
- Apple MPS requires a future native macOS worker; Docker Desktop cannot expose MPS to a Linux container.
- A dedicated scheduler service is not yet present; initial routing is handled through backend validation, worker capabilities, and backend-specific queues.
- Hardware CI, migration rollback rehearsal, and OOM recovery hardening remain roadmap work.

Detailed implementation status is tracked in [Whisper-Multi-Accelerator-Implementation-Plan.md](./Whisper-Multi-Accelerator-Implementation-Plan.md). The original acceptance criteria are in [PLAN.md](./PLAN.md).

## References

- [NVIDIA Driver Installation Guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/introduction.html)
- [NVIDIA CUDA container images](https://hub.docker.com/r/nvidia/cuda)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [AMD ROCm PyTorch images](https://hub.docker.com/r/rocm/pytorch)
- [PyTorch Intel XPU setup](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)
- [OpenVINO Docker installation](https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-docker-linux.html)
- [OpenAI Whisper](https://github.com/openai/whisper)
