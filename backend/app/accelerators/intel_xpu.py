from types import ModuleType
from typing import Callable

from .base import JobExecutionContext, validate_device_index
from .models import AcceleratorBackend, DeviceCapability, DeviceInfo


class IntelXpuProvider:
    backend = AcceleratorBackend.xpu

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
            xpu = self._torch().xpu
            return bool(xpu.is_available())
        except (ImportError, AttributeError, RuntimeError):
            return False

    def list_devices(self) -> list[DeviceInfo]:
        if not self.is_runtime_available():
            return []
        torch = self._torch()
        devices = []
        for index in range(torch.xpu.device_count()):
            properties = torch.xpu.get_device_properties(index)
            devices.append(
                DeviceInfo(
                    backend=self.backend,
                    index=index,
                    id=f"xpu:{index}",
                    name=getattr(properties, "name", None) or f"Intel XPU {index}",
                    vendor="Intel",
                    device_type="gpu",
                    total_memory_bytes=getattr(properties, "total_memory", None),
                    runtime_version=getattr(torch, "__version__", None),
                    experimental=True,
                )
            )
        return devices

    def get_capability(self, device_index: int | None) -> DeviceCapability:
        validate_device_index(self.list_devices(), device_index)
        return DeviceCapability(
            backend=self.backend,
            precisions=["float16", "bfloat16", "float32"],
            engines=[],
            experimental=True,
        )

    def resolve_runtime_device(self, device_index: int | None) -> str:
        device = validate_device_index(self.list_devices(), device_index)
        return f"xpu:{device.index}"

    def health(self, device_index: int | None) -> tuple[bool, str | None]:
        try:
            device = validate_device_index(self.list_devices(), device_index)
            return device.healthy, device.health_reason
        except (RuntimeError, ValueError) as exc:
            return False, str(exc)

    def before_job(self, context: JobExecutionContext) -> None:
        healthy, reason = self.health(context.device_index)
        if not healthy:
            raise RuntimeError(reason or "XPU device is unhealthy")
        torch = self._torch()
        device = self.resolve_runtime_device(context.device_index)
        (torch.ones(1, device=device) + 1).cpu()

    def after_job(self, context: JobExecutionContext) -> None:
        self.clear_cache(context.device_index)

    def clear_cache(self, device_index: int | None) -> None:
        if self.is_runtime_available() and hasattr(self._torch().xpu, "empty_cache"):
            self._torch().xpu.empty_cache()

