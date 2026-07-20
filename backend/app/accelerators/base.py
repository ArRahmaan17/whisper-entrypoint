from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .models import AcceleratorBackend, DeviceCapability, DeviceInfo


@dataclass(slots=True)
class JobExecutionContext:
    job_id: str
    device_index: int | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AcceleratorProvider(Protocol):
    backend: AcceleratorBackend

    def is_runtime_available(self) -> bool: ...
    def list_devices(self) -> list[DeviceInfo]: ...
    def get_capability(self, device_index: int | None) -> DeviceCapability: ...
    def resolve_runtime_device(self, device_index: int | None) -> str: ...
    def health(self, device_index: int | None) -> tuple[bool, str | None]: ...
    def before_job(self, context: JobExecutionContext) -> None: ...
    def after_job(self, context: JobExecutionContext) -> None: ...
    def clear_cache(self, device_index: int | None) -> None: ...


def validate_device_index(devices: list[DeviceInfo], device_index: int | None) -> DeviceInfo:
    if not devices:
        raise RuntimeError("accelerator runtime is unavailable")
    selected = device_index if device_index is not None else devices[0].index
    for device in devices:
        if device.index == selected:
            return device
    raise ValueError(f"invalid device index: {device_index}")

