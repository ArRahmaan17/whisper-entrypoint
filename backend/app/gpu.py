import csv
import io
import subprocess
from typing import Any


GPU_FIELDS = (
    "index,name,memory.total,memory.used,memory.free,temperature.gpu,"
    "utilization.gpu,driver_version,power.draw,power.limit"
)


def _number(value: str, *, integer: bool = False) -> int | float | None:
    value = value.strip()
    if not value or value.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        parsed = float(value)
        return int(parsed) if integer else parsed
    except ValueError:
        return None


def query_nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        f"--query-gpu={GPU_FIELDS}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc), "devices": []}

    devices: list[dict[str, Any]] = []
    reader = csv.reader(io.StringIO(result.stdout))
    for row in reader:
        if len(row) != 10:
            continue
        devices.append(
            {
                "index": _number(row[0], integer=True),
                "name": row[1].strip(),
                "memory_total_mb": _number(row[2], integer=True),
                "memory_used_mb": _number(row[3], integer=True),
                "memory_free_mb": _number(row[4], integer=True),
                "temperature_c": _number(row[5], integer=True),
                "utilization_percent": _number(row[6], integer=True),
                "driver_version": row[7].strip(),
                "power_draw_w": _number(row[8]),
                "power_limit_w": _number(row[9]),
            }
        )
    return {"available": bool(devices), "devices": devices, "error": None}


def torch_cuda_details() -> dict[str, Any]:
    try:
        import torch

        return {
            "torch_cuda": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    except ImportError:
        return {"torch_cuda": False, "torch_version": None, "cuda_version": None, "device_count": 0}


def gpu_status() -> dict[str, Any]:
    return {**query_nvidia_smi(), **torch_cuda_details()}


def validate_gpu_safety(status: dict[str, Any], stop_temp_c: int, min_free_vram_mb: int) -> tuple[bool, str | None]:
    if not status.get("available") or not status.get("torch_cuda"):
        return False, "CUDA is unavailable to the worker"
    for device in status.get("devices", []):
        temperature = device.get("temperature_c")
        free_vram = device.get("memory_free_mb")
        if temperature is not None and temperature >= stop_temp_c:
            return False, f"GPU {device.get('index')} temperature {temperature}C reached the {stop_temp_c}C stop threshold"
        if free_vram is not None and free_vram < min_free_vram_mb:
            return False, f"GPU {device.get('index')} has only {free_vram} MiB free VRAM; {min_free_vram_mb} MiB is required"
    return True, None
