from pydantic import BaseModel

from ..accelerators.models import AcceleratorBackend, ResolvedDevice
from ..accelerators.registry import ProviderRegistry
from ..engines.registry import EngineRegistry
from .compatibility import CompatibilityResult, validate_compatibility


class DeviceRequest(BaseModel):
    policy: str = "auto"
    backend: AcceleratorBackend | None = None
    device_index: int | None = None
    allow_fallback: bool = True


class SelectionError(RuntimeError):
    def __init__(self, results: list[CompatibilityResult]) -> None:
        self.results = results
        reasons = [issue.message for result in results for issue in result.issues]
        super().__init__("; ".join(reasons) or "no compatible accelerator was found")


def select_device(
    providers: ProviderRegistry,
    engines: EngineRegistry,
    *,
    request: DeviceRequest,
    priority: list[AcceleratorBackend],
    engine_name: str,
    model: str,
    precision: str,
) -> ResolvedDevice:
    engine = engines.get(engine_name)
    requested_backend = request.backend
    candidates = [requested_backend] if requested_backend else list(priority)
    if request.policy == "manual" and requested_backend is None:
        raise ValueError("manual device policy requires a backend")
    if requested_backend and request.allow_fallback:
        candidates.extend(item for item in priority if item != requested_backend)

    results: list[CompatibilityResult] = []
    for backend in candidates:
        if backend is None:
            continue
        provider = providers.get(backend)
        result = validate_compatibility(
            provider,
            engine,
            model=model,
            precision=precision,
            device_index=request.device_index if backend == requested_backend else None,
        )
        results.append(result)
        if result.compatible and result.resolved_precision:
            fallback_used = requested_backend is not None and backend != requested_backend
            reason = None
            if fallback_used and results:
                reason = "; ".join(issue.message for item in results[:-1] for issue in item.issues)
            return ResolvedDevice(
                backend=backend,
                device_index=request.device_index if backend == requested_backend else None,
                runtime_device=provider.resolve_runtime_device(request.device_index if backend == requested_backend else None),
                precision=result.resolved_precision,
                engine=engine_name,
                fallback_used=fallback_used,
                fallback_reason=reason,
            )
        if requested_backend and not request.allow_fallback:
            break
    raise SelectionError(results)

