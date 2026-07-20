from typing import Any, Callable

from .base import JobExecutionContext, validate_device_index
from .models import AcceleratorBackend, DeviceCapability, DeviceInfo


class OpenVinoProvider:
    backend = AcceleratorBackend.openvino

    def __init__(self, *, core_factory: Callable[[], Any] | None = None) -> None:
        self._core_factory = core_factory
        self._core_instance: Any | None = None

    def _core(self) -> Any:
        if self._core_instance is None:
            if self._core_factory is not None:
                self._core_instance = self._core_factory()
            else:
                from openvino import Core

                self._core_instance = Core()
        return self._core_instance

    def is_runtime_available(self) -> bool:
        try:
            return bool(self._core().available_devices)
        except (ImportError, RuntimeError):
            return False

    def list_devices(self) -> list[DeviceInfo]:
        if not self.is_runtime_available():
            return []
        devices: list[DeviceInfo] = []
        for index, runtime_name in enumerate(self._core().available_devices):
            device_class = runtime_name.split(".", 1)[0].upper()
            devices.append(
                DeviceInfo(
                    backend=self.backend,
                    index=index,
                    id=f"openvino:{runtime_name}",
                    name=f"OpenVINO {runtime_name}",
                    vendor="Intel" if device_class in {"GPU", "NPU"} else "host",
                    device_type=device_class.lower(),
                    runtime_version=self._runtime_version(),
                    experimental=True,
                )
            )
        return devices

    @staticmethod
    def _runtime_version() -> str | None:
        try:
            import openvino

            return getattr(openvino, "__version__", None)
        except ImportError:
            return None

    def get_capability(self, device_index: int | None) -> DeviceCapability:
        validate_device_index(self.list_devices(), device_index)
        return DeviceCapability(
            backend=self.backend,
            precisions=["float16", "float32"],
            engines=[],
            experimental=True,
        )

    def resolve_runtime_device(self, device_index: int | None) -> str:
        device = validate_device_index(self.list_devices(), device_index)
        return device.id.removeprefix("openvino:")

    def health(self, device_index: int | None) -> tuple[bool, str | None]:
        try:
            validate_device_index(self.list_devices(), device_index)
            return True, None
        except (RuntimeError, ValueError) as exc:
            return False, str(exc)

    def before_job(self, context: JobExecutionContext) -> None:
        healthy, reason = self.health(context.device_index)
        if not healthy:
            raise RuntimeError(reason or "OpenVINO device is unhealthy")

    def after_job(self, context: JobExecutionContext) -> None:
        return None

    def clear_cache(self, device_index: int | None) -> None:
        return None

