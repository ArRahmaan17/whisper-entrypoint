from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .accelerators.models import AcceleratorBackend


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

    app_env: str = "development"
    database_url: str = "sqlite:///./whisper.db"
    redis_url: str = "redis://localhost:6379/0"
    upload_dir: Path = Path("./uploads")
    output_dir: Path = Path("./outputs")
    model_cache_dir: Path = Path("./models")
    default_engine: str = "openai-whisper"
    default_device: str = "auto"
    accelerator_priority: str = "cuda,rocm,xpu,openvino,mps,cpu"
    allow_device_fallback: bool = True
    allow_cpu_fallback: bool = False
    worker_backend: AcceleratorBackend = AcceleratorBackend.cuda
    worker_device_index: int | None = None
    max_concurrent_jobs: int = 1
    worker_heartbeat_seconds: int = 10
    worker_stale_after_seconds: int = 30
    device_health_refresh_seconds: int = 30
    max_loaded_models: int = 1
    cpu_threads: int = 0
    max_upload_mb: int = 2048
    max_concurrent_gpu_jobs: int = 1
    gpu_temp_warn_c: int = 80
    gpu_temp_stop_c: int = 87
    gpu_min_free_vram_mb: int = 1024
    default_model: str = "base"
    default_output_formats: str = "json,srt,txt"
    job_retention_days: int = 30
    cors_origins: str = "http://localhost:3000"
    worker_name: str = "whisper-gpu-0"

    @field_validator("default_device")
    @classmethod
    def valid_device(cls, value: str) -> str:
        value = value.lower()
        if value not in {"auto", *(item.value for item in AcceleratorBackend)}:
            raise ValueError("DEFAULT_DEVICE must be auto or a known accelerator backend")
        return value

    @field_validator("max_concurrent_jobs", "worker_heartbeat_seconds", "worker_stale_after_seconds", "device_health_refresh_seconds", "max_loaded_models")
    @classmethod
    def positive_integer(cls, value: int) -> int:
        if value < 1:
            raise ValueError("value must be at least 1")
        return value

    @property
    def output_formats(self) -> list[str]:
        return [value.strip() for value in self.default_output_formats.split(",") if value.strip()]

    @property
    def origins(self) -> list[str]:
        return [value.strip() for value in self.cors_origins.split(",") if value.strip()]

    @property
    def priority(self) -> list[AcceleratorBackend]:
        values: list[AcceleratorBackend] = []
        for raw in self.accelerator_priority.split(","):
            value = raw.strip().lower()
            if value and AcceleratorBackend(value) not in values:
                values.append(AcceleratorBackend(value))
        return values

    def ensure_directories(self) -> None:
        for directory in (self.upload_dir, self.output_dir, self.model_cache_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
