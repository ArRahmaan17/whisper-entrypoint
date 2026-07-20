from app.accelerators.cpu import CpuProvider
from app.accelerators.models import AcceleratorBackend
from app.accelerators.registry import ProviderRegistry
from app.engines.openai_whisper import OpenAIWhisperEngine
from app.engines.registry import EngineRegistry
from app.scheduler.compatibility import validate_compatibility
from app.scheduler.selector import DeviceRequest, select_device


def registries() -> tuple[ProviderRegistry, EngineRegistry]:
    providers = ProviderRegistry({AcceleratorBackend.cpu: CpuProvider})
    engines = EngineRegistry({"openai-whisper": lambda: OpenAIWhisperEngine(whisper_loader=lambda: object())})
    return providers, engines


def test_cpu_precision_auto_resolves_to_float32() -> None:
    providers, engines = registries()
    result = validate_compatibility(
        providers.get("cpu"),
        engines.get("openai-whisper"),
        model="base",
        precision="auto",
        device_index=None,
    )
    assert result.compatible
    assert result.resolved_precision == "float32"


def test_cpu_rejects_float16() -> None:
    providers, engines = registries()
    result = validate_compatibility(
        providers.get("cpu"),
        engines.get("openai-whisper"),
        model="base",
        precision="float16",
        device_index=None,
    )
    assert not result.compatible
    assert result.issues[0].code == "PRECISION_UNSUPPORTED"


def test_auto_selection_falls_through_to_cpu() -> None:
    providers, engines = registries()
    resolved = select_device(
        providers,
        engines,
        request=DeviceRequest(policy="auto", allow_fallback=True),
        priority=[AcceleratorBackend.cpu],
        engine_name="openai-whisper",
        model="base",
        precision="auto",
    )
    assert resolved.backend == AcceleratorBackend.cpu
    assert resolved.runtime_device == "cpu"
    assert resolved.precision == "float32"

