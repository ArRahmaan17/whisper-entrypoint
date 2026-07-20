from collections.abc import Callable

from .base import TranscriptionEngine


EngineFactory = Callable[[], TranscriptionEngine]


def load_openai_whisper_engine() -> TranscriptionEngine:
    from ..config import get_settings
    from .openai_whisper import OpenAIWhisperEngine

    return OpenAIWhisperEngine(max_loaded_models=get_settings().max_loaded_models)


ENGINE_FACTORIES: dict[str, EngineFactory] = {
    "openai-whisper": load_openai_whisper_engine,
}


class EngineRegistry:
    def __init__(self, factories: dict[str, EngineFactory] | None = None) -> None:
        self._factories = factories or ENGINE_FACTORIES
        self._engines: dict[str, TranscriptionEngine] = {}

    def get(self, name: str) -> TranscriptionEngine:
        if name not in self._factories:
            raise KeyError(f"engine is not registered: {name}")
        if name not in self._engines:
            self._engines[name] = self._factories[name]()
        return self._engines[name]

    def names(self) -> list[str]:
        return list(self._factories)


def create_engine_registry() -> EngineRegistry:
    return EngineRegistry()

