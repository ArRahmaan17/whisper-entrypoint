import platform
from types import ModuleType
from typing import Callable

import psutil

from .base import JobExecutionContext
from .models import AcceleratorBackend, DeviceCapability, DeviceInfo, DeviceTelemetry


class AppleMpsProvider:
    backend = AcceleratorBackend.mps

    def __init__(self, *, torch_loader: Callable[[], ModuleType] | None = None) -> None:
        self._torch_loader = torch_loader
        self._torch_module: ModuleType | None = None

    def _torch(self) -> ModuleType:
        if self._torch_module is None:
            if self._torch_loader is not None:
                self._torch_module = self._torch_loader()
            else:
                import torch

                self._torch_module = torch
        return self._torch_module

    def is_runtime_available(self) -> bool:
        if platform.system() != "Darwin":
            return False
        try:
            mps = self._torch().backends.mps
            return bool(mps.is_built() and mps.is_available())
        except (ImportError, AttributeError, RuntimeError):
            return False

    def list_devices(self) -> list[DeviceInfo]:
        if not self.is_runtime_available():
            return []
        memory = psutil.virtual_memory()
        return [
            DeviceInfo(
                backend=self.backend,
                index=None,
                id="mps",
                name="Apple Silicon GPU",
                vendor="Apple",
                device_type="gpu",
                total_memory_bytes=memory.total,
                free_memory_bytes=memory.available,
                architecture=platform.machine(),
                runtime_version=getattr(self._torch(), "__version__", None),
                experimental=True,
                telemetry=DeviceTelemetry(
                    memory_total_bytes=memory.total,
                    memory_used_bytes=memory.used,
                    memory_free_bytes=memory.available,
                ),
            )
        ]

    def get_capability(self, device_index: int | None) -> DeviceCapability:
        if device_index is not None or not self.is_runtime_available():
            raise ValueError("MPS is unavailable or received an invalid device index")
        return DeviceCapability(
            backend=self.backend,
            precisions=["float16", "float32"],
            engines=["openai-whisper"],
            supports_memory_telemetry=True,
            experimental=True,
        )

    def resolve_runtime_device(self, device_index: int | None) -> str:
        self.get_capability(device_index)
        return "mps"

    def health(self, device_index: int | None) -> tuple[bool, str | None]:
        try:
            self.get_capability(device_index)
            return True, None
        except ValueError as exc:
            return False, str(exc)

    def before_job(self, context: JobExecutionContext) -> None:
        healthy, reason = self.health(context.device_index)
        if not healthy:
            raise RuntimeError(reason or "MPS device is unhealthy")

    def after_job(self, context: JobExecutionContext) -> None:
        self.clear_cache(context.device_index)

    def clear_cache(self, device_index: int | None) -> None:
        if self.is_runtime_available() and hasattr(self._torch().mps, "empty_cache"):
            self._torch().mps.empty_cache()

