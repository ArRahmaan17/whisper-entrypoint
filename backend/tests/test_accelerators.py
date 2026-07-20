from types import SimpleNamespace

import pytest

from app.accelerators.base import AcceleratorProvider
from app.accelerators.cpu import CpuProvider
from app.accelerators.cuda import CudaProvider
from app.accelerators.intel_xpu import IntelXpuProvider
from app.accelerators.models import AcceleratorBackend, DeviceTelemetry
from app.accelerators.openvino import OpenVinoProvider
from app.accelerators.registry import ProviderRegistry
from app.accelerators.rocm import RocmProvider


class FakeCudaApi:
    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.cache_clears = 0
        self.selected = None

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return 1

    def get_device_properties(self, index: int) -> SimpleNamespace:
        return SimpleNamespace(name="Test GPU", total_memory=8 * 1024**3, major=8, minor=9, gcnArchName="gfx-test")

    def get_device_name(self, index: int) -> str:
        return "Test GPU"

    def mem_get_info(self, index: int) -> tuple[int, int]:
        return 7 * 1024**3, 8 * 1024**3

    def set_device(self, index: int) -> None:
        self.selected = index

    def empty_cache(self) -> None:
        self.cache_clears += 1


def fake_torch(*, cuda_version: str | None, hip_version: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        __version__="test",
        version=SimpleNamespace(cuda=cuda_version, hip=hip_version),
        cuda=FakeCudaApi(),
    )


def test_cpu_provider_conforms_to_contract() -> None:
    provider = CpuProvider()
    assert isinstance(provider, AcceleratorProvider)
    assert provider.is_runtime_available()
    assert provider.resolve_runtime_device(None) == "cpu"
    assert provider.list_devices()[0].backend == AcceleratorBackend.cpu


def test_cuda_build_is_not_rocm(monkeypatch) -> None:
    monkeypatch.setattr("app.accelerators.cuda.query_nvidia_smi", lambda: {"devices": []})
    torch = fake_torch(cuda_version="13.2", hip_version=None)
    assert CudaProvider(torch_loader=lambda: torch).is_runtime_available()
    assert not RocmProvider(torch_loader=lambda: torch).is_runtime_available()
    device = CudaProvider(torch_loader=lambda: torch).list_devices()[0]
    assert device.backend == AcceleratorBackend.cuda
    assert device.vendor == "NVIDIA"
    assert device.free_memory_bytes == 7 * 1024**3


def test_rocm_build_is_not_reported_as_cuda(monkeypatch) -> None:
    monkeypatch.setattr("app.accelerators.cuda.query_nvidia_smi", lambda: {"devices": []})
    torch = fake_torch(cuda_version=None, hip_version="6.4")
    assert RocmProvider(torch_loader=lambda: torch).is_runtime_available()
    assert not CudaProvider(torch_loader=lambda: torch).is_runtime_available()
    assert RocmProvider(torch_loader=lambda: torch).list_devices()[0].backend == AcceleratorBackend.rocm


def test_missing_optional_runtime_is_clean() -> None:
    def missing():
        raise ImportError("not installed")

    assert CudaProvider(torch_loader=missing).list_devices() == []
    assert RocmProvider(torch_loader=missing).list_devices() == []
    assert IntelXpuProvider(torch_loader=missing).list_devices() == []
    assert OpenVinoProvider(core_factory=missing).list_devices() == []


def test_registry_initializes_lazily() -> None:
    loaded: list[str] = []

    def factory() -> CpuProvider:
        loaded.append("cpu")
        return CpuProvider()

    registry = ProviderRegistry({AcceleratorBackend.cpu: factory})
    assert loaded == []
    assert registry.get("cpu") is registry.get(AcceleratorBackend.cpu)
    assert loaded == ["cpu"]


def test_telemetry_does_not_fabricate_unsupported_values() -> None:
    payload = DeviceTelemetry().model_dump()
    assert payload["temperature_c"] is None
    assert payload["power_watts"] is None


def test_cpu_rejects_device_index() -> None:
    with pytest.raises(ValueError, match="device index"):
        CpuProvider().resolve_runtime_device(0)
