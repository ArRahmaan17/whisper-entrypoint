import os
import platform

import psutil

from .base import JobExecutionContext
from .models import AcceleratorBackend, DeviceCapability, DeviceInfo, DeviceTelemetry


class CpuProvider:
    backend = AcceleratorBackend.cpu

    def is_runtime_available(self) -> bool:
        return True

    def list_devices(self) -> list[DeviceInfo]:
        memory = psutil.virtual_memory()
        return [
            DeviceInfo(
                backend=self.backend,
                index=None,
                id="cpu",
                name=platform.processor() or platform.machine() or "CPU",
                vendor="host",
                device_type="cpu",
                total_memory_bytes=memory.total,
                free_memory_bytes=memory.available,
                architecture=platform.machine(),
                telemetry=DeviceTelemetry(
                    utilization_percent=psutil.cpu_percent(interval=None),
                    memory_total_bytes=memory.total,
                    memory_used_bytes=memory.used,
                    memory_free_bytes=memory.available,
                ),
            )
        ]

    def get_capability(self, device_index: int | None) -> DeviceCapability:
        if device_index is not None:
            raise ValueError("CPU does not accept a device index")
        return DeviceCapability(
            backend=self.backend,
            precisions=["float32"],
            engines=["openai-whisper"],
            supports_memory_telemetry=True,
            supports_concurrent_jobs=True,
        )

    def resolve_runtime_device(self, device_index: int | None) -> str:
        self.get_capability(device_index)
        return "cpu"

    def health(self, device_index: int | None) -> tuple[bool, str | None]:
        self.get_capability(device_index)
        return True, None

    def before_job(self, context: JobExecutionContext) -> None:
        threads = context.parameters.get("threads")
        if not threads:
            threads = int(os.getenv("CPU_THREADS", "0")) or None
        if threads:
            try:
                import torch

                torch.set_num_threads(int(threads))
            except ImportError:
                return

    def after_job(self, context: JobExecutionContext) -> None:
        return None

    def clear_cache(self, device_index: int | None) -> None:
        self.get_capability(device_index)

