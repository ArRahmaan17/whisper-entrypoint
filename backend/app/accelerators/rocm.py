from types import ModuleType
from typing import Callable

from .base import JobExecutionContext, validate_device_index
from .models import AcceleratorBackend, DeviceCapability, DeviceInfo


class RocmProvider:
    backend = AcceleratorBackend.rocm

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
        try:
            torch = self._torch()
            return bool(torch.version.hip is not None and torch.cuda.is_available())
        except (ImportError, AttributeError, RuntimeError):
            return False

    def list_devices(self) -> list[DeviceInfo]:
        if not self.is_runtime_available():
            return []
        torch = self._torch()
        devices: list[DeviceInfo] = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            free_bytes = None
            try:
                free_bytes, _ = torch.cuda.mem_get_info(index)
            except (AttributeError, RuntimeError):
                pass
            devices.append(
                DeviceInfo(
                    backend=self.backend,
                    index=index,
                    id=f"rocm:{index}",
                    name=getattr(properties, "name", None) or torch.cuda.get_device_name(index),
                    vendor="AMD",
                    device_type="gpu",
                    total_memory_bytes=int(getattr(properties, "total_memory", 0)) or None,
                    free_memory_bytes=free_bytes,
                    architecture=getattr(properties, "gcnArchName", None),
                    runtime_version=str(torch.version.hip),
                    experimental=True,
                )
            )
        return devices

    def get_capability(self, device_index: int | None) -> DeviceCapability:
        validate_device_index(self.list_devices(), device_index)
        return DeviceCapability(
            backend=self.backend,
            precisions=["float16", "float32"],
            engines=["openai-whisper"],
            supports_memory_telemetry=True,
            experimental=True,
        )

    def resolve_runtime_device(self, device_index: int | None) -> str:
        device = validate_device_index(self.list_devices(), device_index)
        return f"cuda:{device.index}"

    def health(self, device_index: int | None) -> tuple[bool, str | None]:
        try:
            device = validate_device_index(self.list_devices(), device_index)
            return device.healthy, device.health_reason
        except (RuntimeError, ValueError) as exc:
            return False, str(exc)

    def before_job(self, context: JobExecutionContext) -> None:
        healthy, reason = self.health(context.device_index)
        if not healthy:
            raise RuntimeError(reason or "ROCm device is unhealthy")
        self._torch().cuda.set_device(context.device_index or 0)

    def after_job(self, context: JobExecutionContext) -> None:
        self.clear_cache(context.device_index)

    def clear_cache(self, device_index: int | None) -> None:
        if self.is_runtime_available():
            self._torch().cuda.empty_cache()

