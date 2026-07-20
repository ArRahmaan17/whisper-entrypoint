from pydantic import BaseModel, Field

from ..accelerators.base import AcceleratorProvider
from ..accelerators.models import AcceleratorBackend, DeviceInfo
from ..engines.base import TranscriptionEngine
from .errors import JobErrorCode


class CompatibilityIssue(BaseModel):
    code: JobErrorCode
    message: str


class CompatibilityResult(BaseModel):
    compatible: bool
    backend: AcceleratorBackend
    engine: str
    model: str
    requested_precision: str
    resolved_precision: str | None = None
    issues: list[CompatibilityIssue] = Field(default_factory=list)


def _resolve_precision(
    requested: str,
    provider_precisions: set[str],
    engine_precisions: set[str],
    backend: AcceleratorBackend,
) -> str | None:
    available = provider_precisions & engine_precisions
    if requested != "auto":
        return requested if requested in available else None
    preference = ["float16", "bfloat16", "float32"] if backend != AcceleratorBackend.cpu else ["float32"]
    return next((value for value in preference if value in available), None)


def validate_compatibility(
    provider: AcceleratorProvider,
    engine: TranscriptionEngine,
    *,
    model: str,
    precision: str,
    device_index: int | None,
) -> CompatibilityResult:
    issues: list[CompatibilityIssue] = []
    backend = provider.backend
    if not provider.is_runtime_available():
        issues.append(CompatibilityIssue(code=JobErrorCode.accelerator_unavailable, message=f"{backend.value} runtime is unavailable"))
        return CompatibilityResult(
            compatible=False,
            backend=backend,
            engine=engine.name,
            model=model,
            requested_precision=precision,
            issues=issues,
        )

    devices = provider.list_devices()
    device: DeviceInfo | None = None
    if devices:
        selected_index = device_index if device_index is not None else devices[0].index
        device = next((item for item in devices if item.index == selected_index), None)
    if device is None:
        issues.append(CompatibilityIssue(code=JobErrorCode.accelerator_unavailable, message="requested device was not found"))
    elif not device.healthy:
        issues.append(CompatibilityIssue(code=JobErrorCode.accelerator_unhealthy, message=device.health_reason or "device is unhealthy"))

    if backend not in engine.supported_backends():
        issues.append(CompatibilityIssue(code=JobErrorCode.engine_unsupported, message=f"{engine.name} does not support {backend.value}"))
    if model not in engine.supported_models():
        issues.append(CompatibilityIssue(code=JobErrorCode.model_unsupported, message=f"model {model} is not supported"))

    resolved_precision: str | None = None
    if device is not None:
        capability = provider.get_capability(device_index)
        resolved_precision = _resolve_precision(
            precision,
            set(capability.precisions),
            engine.supported_precisions(backend, device),
            backend,
        )
        if resolved_precision is None:
            issues.append(CompatibilityIssue(code=JobErrorCode.precision_unsupported, message=f"precision {precision} is unavailable"))
        elif device.free_memory_bytes is not None:
            estimate = engine.estimate_memory_bytes(model, backend, resolved_precision)
            if estimate is not None and estimate > device.free_memory_bytes:
                issues.append(
                    CompatibilityIssue(
                        code=JobErrorCode.insufficient_device_memory,
                        message=f"model requires approximately {estimate} bytes but only {device.free_memory_bytes} bytes are free",
                    )
                )

    return CompatibilityResult(
        compatible=not issues,
        backend=backend,
        engine=engine.name,
        model=model,
        requested_precision=precision,
        resolved_precision=resolved_precision,
        issues=issues,
    )

