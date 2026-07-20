from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


class AcceleratorBackend(StrEnum):
    cuda = "cuda"
    rocm = "rocm"
    xpu = "xpu"
    openvino = "openvino"
    mps = "mps"
    cpu = "cpu"


class DeviceTelemetry(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    utilization_percent: float | None = None
    memory_total_bytes: int | None = None
    memory_used_bytes: int | None = None
    memory_free_bytes: int | None = None
    temperature_c: float | None = None
    power_watts: float | None = None


class DeviceInfo(BaseModel):
    backend: AcceleratorBackend
    index: int | None
    id: str
    name: str
    vendor: str
    device_type: str
    total_memory_bytes: int | None = None
    free_memory_bytes: int | None = None
    architecture: str | None = None
    driver_version: str | None = None
    runtime_version: str | None = None
    healthy: bool = True
    health_reason: str | None = None
    experimental: bool = False
    telemetry: DeviceTelemetry | None = None


class DeviceCapability(BaseModel):
    backend: AcceleratorBackend
    precisions: list[str]
    engines: list[str]
    supported_models: list[str] | None = None
    supports_temperature: bool = False
    supports_utilization: bool = False
    supports_memory_telemetry: bool = False
    supports_concurrent_jobs: bool = False
    experimental: bool = False


class ResolvedDevice(BaseModel):
    backend: AcceleratorBackend
    device_index: int | None
    runtime_device: str
    precision: str
    engine: str
    fallback_used: bool = False
    fallback_reason: str | None = None

