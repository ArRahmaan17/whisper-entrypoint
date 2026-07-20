from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ..accelerators.models import AcceleratorBackend, DeviceInfo, ResolvedDevice


ProgressCallback = Callable[[str, float], None]


class ModelLoadRequest(BaseModel):
    model: str
    resolved_device: ResolvedDevice
    cache_dir: Path


class TranscriptionRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_path: Path
    options: dict[str, Any] = Field(default_factory=dict)
    model: Any


@runtime_checkable
class TranscriptionEngine(Protocol):
    name: str

    def is_runtime_available(self) -> bool: ...
    def supported_models(self) -> list[str]: ...
    def supported_backends(self) -> set[AcceleratorBackend]: ...
    def supported_precisions(self, backend: AcceleratorBackend, device: DeviceInfo | None) -> set[str]: ...
    def estimate_memory_bytes(self, model: str, backend: AcceleratorBackend, precision: str) -> int | None: ...
    def load_model(self, request: ModelLoadRequest) -> Any: ...
    def transcribe(self, request: TranscriptionRequest, progress_callback: ProgressCallback | None = None) -> dict[str, Any]: ...
    def write_outputs(self, result: dict[str, Any], source_path: Path, output_dir: Path, formats: list[str]) -> list[Path]: ...
    def unload_model(self) -> None: ...
