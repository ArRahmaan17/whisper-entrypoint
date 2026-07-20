from types import ModuleType
from typing import Any, Callable

from ..gpu import query_nvidia_smi
from .base import JobExecutionContext, validate_device_index
from .models import AcceleratorBackend, DeviceCapability, DeviceInfo, DeviceTelemetry


MIB = 1024 * 1024


class CudaProvider:
    backend = AcceleratorBackend.cuda

    def __init__(
        self,
        *,
        torch_loader: Callable[[], ModuleType] | None = None,
        stop_temp_c: int = 87,
        min_free_memory_bytes: int = 1024 * MIB,
    ) -> None:
        self._torch_loader = torch_loader
        self._torch_module: ModuleType | None = None
        self.stop_temp_c = stop_temp_c
        self.min_free_memory_bytes = min_free_memory_bytes

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
            return bool(torch.version.cuda is not None and torch.version.hip is None and torch.cuda.is_available())
        except (ImportError, AttributeError, RuntimeError):
            return False

    def list_devices(self) -> list[DeviceInfo]:
        if not self.is_runtime_available():
            return []
        torch = self._torch()
        smi_by_index = {item.get("index"): item for item in query_nvidia_smi().get("devices", [])}
        devices: list[DeviceInfo] = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            free_bytes: int | None = None
            try:
                free_bytes, _ = torch.cuda.mem_get_info(index)
            except (AttributeError, RuntimeError):
                pass
            smi = smi_by_index.get(index, {})
            if free_bytes is None and smi.get("memory_free_mb") is not None:
                free_bytes = int(smi["memory_free_mb"]) * MIB
            total_bytes = int(getattr(properties, "total_memory", 0)) or None
            telemetry = DeviceTelemetry(
                utilization_percent=smi.get("utilization_percent"),
                memory_total_bytes=total_bytes,
                memory_used_bytes=(int(smi["memory_used_mb"]) * MIB if smi.get("memory_used_mb") is not None else None),
                memory_free_bytes=free_bytes,
                temperature_c=smi.get("temperature_c"),
                power_watts=smi.get("power_draw_w"),
            )
            healthy, reason = self._telemetry_health(telemetry)
            devices.append(
                DeviceInfo(
                    backend=self.backend,
                    index=index,
                    id=f"cuda:{index}",
                    name=getattr(properties, "name", None) or torch.cuda.get_device_name(index),
                    vendor="NVIDIA",
                    device_type="gpu",
                    total_memory_bytes=total_bytes,
                    free_memory_bytes=free_bytes,
                    architecture=self._architecture(properties),
                    driver_version=smi.get("driver_version"),
                    runtime_version=str(torch.version.cuda),
                    healthy=healthy,
                    health_reason=reason,
                    telemetry=telemetry,
                )
            )
        return devices

    @staticmethod
    def _architecture(properties: Any) -> str | None:
        major = getattr(properties, "major", None)
        minor = getattr(properties, "minor", None)
        return f"sm_{major}{minor}" if major is not None and minor is not None else None

    def _telemetry_health(self, telemetry: DeviceTelemetry) -> tuple[bool, str | None]:
        if telemetry.temperature_c is not None and telemetry.temperature_c >= self.stop_temp_c:
            return False, f"temperature {telemetry.temperature_c:g}C reached stop threshold {self.stop_temp_c}C"
        if telemetry.memory_free_bytes is not None and telemetry.memory_free_bytes < self.min_free_memory_bytes:
            return False, "insufficient free device memory"
        return True, None

    def get_capability(self, device_index: int | None) -> DeviceCapability:
        validate_device_index(self.list_devices(), device_index)
        return DeviceCapability(
            backend=self.backend,
            precisions=["float16", "float32"],
            engines=["openai-whisper"],
            supports_temperature=True,
            supports_utilization=True,
            supports_memory_telemetry=True,
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
            raise RuntimeError(reason or "CUDA device is unhealthy")
        index = context.device_index if context.device_index is not None else 0
        self._torch().cuda.set_device(index)

    def after_job(self, context: JobExecutionContext) -> None:
        self.clear_cache(context.device_index)

    def clear_cache(self, device_index: int | None) -> None:
        if self.is_runtime_available():
            self._torch().cuda.empty_cache()

