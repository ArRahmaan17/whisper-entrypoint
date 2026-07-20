import platform
from collections.abc import Callable, Iterable

from .base import AcceleratorProvider
from .models import AcceleratorBackend


ProviderFactory = Callable[[], AcceleratorProvider]


def load_cuda_provider() -> AcceleratorProvider:
    from .cuda import CudaProvider

    return CudaProvider()


def load_rocm_provider() -> AcceleratorProvider:
    from .rocm import RocmProvider

    return RocmProvider()


def load_xpu_provider() -> AcceleratorProvider:
    from .intel_xpu import IntelXpuProvider

    return IntelXpuProvider()


def load_openvino_provider() -> AcceleratorProvider:
    from .openvino import OpenVinoProvider

    return OpenVinoProvider()


def load_mps_provider() -> AcceleratorProvider:
    from .apple_mps import AppleMpsProvider

    return AppleMpsProvider()


def load_cpu_provider() -> AcceleratorProvider:
    from .cpu import CpuProvider

    return CpuProvider()


PROVIDER_FACTORIES: dict[AcceleratorBackend, ProviderFactory] = {
    AcceleratorBackend.cuda: load_cuda_provider,
    AcceleratorBackend.rocm: load_rocm_provider,
    AcceleratorBackend.xpu: load_xpu_provider,
    AcceleratorBackend.openvino: load_openvino_provider,
    AcceleratorBackend.mps: load_mps_provider,
    AcceleratorBackend.cpu: load_cpu_provider,
}


class ProviderRegistry:
    def __init__(self, factories: dict[AcceleratorBackend, ProviderFactory] | None = None) -> None:
        self._factories = factories or PROVIDER_FACTORIES
        self._providers: dict[AcceleratorBackend, AcceleratorProvider] = {}

    def get(self, backend: AcceleratorBackend | str) -> AcceleratorProvider:
        normalized = AcceleratorBackend(backend)
        if normalized not in self._factories:
            raise KeyError(f"provider is not registered: {normalized.value}")
        if normalized not in self._providers:
            self._providers[normalized] = self._factories[normalized]()
        return self._providers[normalized]

    def detect(self, priority: Iterable[AcceleratorBackend | str]) -> list[AcceleratorProvider]:
        available: list[AcceleratorProvider] = []
        for backend in priority:
            try:
                provider = self.get(backend)
                if provider.is_runtime_available():
                    available.append(provider)
            except (ImportError, RuntimeError, ValueError):
                continue
        return available


def default_priority() -> list[AcceleratorBackend]:
    if platform.system() == "Darwin":
        return [AcceleratorBackend.mps, AcceleratorBackend.cpu]
    return [
        AcceleratorBackend.cuda,
        AcceleratorBackend.rocm,
        AcceleratorBackend.xpu,
        AcceleratorBackend.openvino,
        AcceleratorBackend.cpu,
    ]


def create_provider_registry() -> ProviderRegistry:
    return ProviderRegistry()

