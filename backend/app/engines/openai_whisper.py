from collections import OrderedDict
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from ..accelerators.models import AcceleratorBackend, DeviceInfo
from .base import ModelLoadRequest, ProgressCallback, TranscriptionRequest


MODELS = [
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large", "large-v2", "large-v3", "turbo",
]
MODEL_MEMORY_BYTES = {
    "tiny": 1 * 1024**3,
    "base": 1 * 1024**3,
    "small": 2 * 1024**3,
    "medium": 5 * 1024**3,
    "large": 10 * 1024**3,
    "large-v2": 10 * 1024**3,
    "large-v3": 10 * 1024**3,
    "turbo": 6 * 1024**3,
}


class OpenAIWhisperEngine:
    name = "openai-whisper"

    def __init__(
        self,
        *,
        whisper_loader: Callable[[], ModuleType] | None = None,
        max_loaded_models: int = 1,
    ) -> None:
        self._whisper_loader = whisper_loader
        self._whisper_module: ModuleType | None = None
        self._models: OrderedDict[tuple[str, str, str], Any] = OrderedDict()
        self.max_loaded_models = max(1, max_loaded_models)

    def _whisper(self) -> ModuleType:
        if self._whisper_module is None:
            if self._whisper_loader is not None:
                self._whisper_module = self._whisper_loader()
            else:
                import whisper

                self._whisper_module = whisper
        return self._whisper_module

    def is_runtime_available(self) -> bool:
        try:
            self._whisper()
            return True
        except ImportError:
            return False

    def supported_models(self) -> list[str]:
        return list(MODELS)

    def supported_backends(self) -> set[AcceleratorBackend]:
        return {
            AcceleratorBackend.cuda,
            AcceleratorBackend.rocm,
            AcceleratorBackend.mps,
            AcceleratorBackend.cpu,
        }

    def supported_precisions(self, backend: AcceleratorBackend, device: DeviceInfo | None) -> set[str]:
        if backend == AcceleratorBackend.cpu:
            return {"float32"}
        if backend in self.supported_backends():
            return {"float16", "float32"}
        return set()

    def estimate_memory_bytes(self, model: str, backend: AcceleratorBackend, precision: str) -> int | None:
        normalized = model.removesuffix(".en")
        estimate = MODEL_MEMORY_BYTES.get(normalized)
        if estimate is None:
            return None
        return estimate * 2 if precision == "float32" and backend != AcceleratorBackend.cpu else estimate

    def load_model(self, request: ModelLoadRequest) -> Any:
        key = (request.model, request.resolved_device.runtime_device, request.resolved_device.precision)
        if key in self._models:
            self._models.move_to_end(key)
            return self._models[key]
        model = self._whisper().load_model(
            request.model,
            device=request.resolved_device.runtime_device,
            download_root=str(request.cache_dir),
        )
        self._models[key] = model
        while len(self._models) > self.max_loaded_models:
            self._models.popitem(last=False)
        return model

    def transcribe(
        self,
        request: TranscriptionRequest,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        if progress_callback:
            progress_callback("transcribing", 0.35)
        return request.model.transcribe(str(request.source_path), **request.options)

    def unload_model(self) -> None:
        self._models.clear()

    def write_outputs(
        self,
        result: dict[str, Any],
        source_path: Path,
        output_dir: Path,
        formats: list[str],
    ) -> list[Path]:
        writer_options = {
            "highlight_words": False,
            "max_line_count": None,
            "max_line_width": None,
            "max_words_per_line": None,
        }
        paths: list[Path] = []
        for output_format in formats:
            writer = self._whisper().utils.get_writer(output_format, str(output_dir))
            writer(result, str(source_path), writer_options)
            path = output_dir / f"{source_path.stem}.{output_format}"
            if not path.is_file():
                raise RuntimeError(f"engine did not create the expected {output_format} output")
            paths.append(path)
        return paths

    @property
    def loaded_models(self) -> list[str]:
        return [key[0] for key in self._models]

