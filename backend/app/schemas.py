from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .accelerators.models import AcceleratorBackend
from .models import JobStatus
from .scheduler.selector import DeviceRequest


OutputFormat = Literal["txt", "json", "srt", "vtt", "tsv"]


class UploadRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    original_name: str
    content_type: str | None
    size_bytes: int
    sha256: str
    created_at: datetime


class WhisperParameters(BaseModel):
    engine: str = "openai-whisper"
    model: str = "base"
    language: str | None = None
    task: Literal["transcribe", "translate"] = "transcribe"
    device: Literal["auto", "cuda", "rocm", "xpu", "openvino", "mps", "cpu"] = "auto"
    device_index: int | None = Field(default=None, ge=0)
    precision: Literal["auto", "float16", "bfloat16", "float32", "int8"] = "auto"
    allow_fallback: bool = True
    fp16: bool | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    beam_size: int | None = Field(default=None, ge=1, le=100)
    best_of: int | None = Field(default=None, ge=1, le=100)
    patience: float | None = Field(default=None, ge=0.0)
    length_penalty: float | None = None
    suppress_tokens: str | None = "-1"
    initial_prompt: str | None = Field(default=None, max_length=4000)
    condition_on_previous_text: bool = True
    word_timestamps: bool = False
    prepend_punctuations: str | None = None
    append_punctuations: str | None = None
    clip_timestamps: str | None = "0"
    hallucination_silence_threshold: float | None = Field(default=None, ge=0.0)
    threads: int | None = Field(default=None, ge=1, le=256)
    output_formats: list[OutputFormat] = Field(default_factory=lambda: ["json", "srt", "txt"], min_length=1)

    @model_validator(mode="after")
    def validate_combination(self) -> "WhisperParameters":
        if self.model == "turbo" and self.task == "translate":
            raise ValueError("The turbo model does not support reliable translation; select another model")
        if self.device == "cpu" and (self.fp16 is True or self.precision == "float16"):
            raise ValueError("fp16 cannot be enabled for CPU jobs")
        if self.temperature == 0 and self.best_of is not None:
            raise ValueError("best_of is only used with non-zero temperature")
        return self


class JobCreate(BaseModel):
    upload_ids: list[str] = Field(min_length=1, max_length=100)
    parameters: WhisperParameters = Field(default_factory=WhisperParameters)
    engine: str | None = None
    model: str | None = None
    device: DeviceRequest | None = None
    precision: Literal["auto", "float16", "bfloat16", "float32", "int8"] | None = None

    def execution_parameters(self) -> dict[str, Any]:
        values = self.parameters.model_dump()
        if self.engine is not None:
            values["engine"] = self.engine
        if self.model is not None:
            values["model"] = self.model
        if self.precision is not None:
            values["precision"] = self.precision
        if self.device is not None:
            values["device_request"] = self.device.model_dump(mode="json")
            values["device"] = self.device.backend.value if self.device.backend else "auto"
            values["device_index"] = self.device.device_index
            values["allow_fallback"] = self.device.allow_fallback
        else:
            backend = None if values["device"] == "auto" else AcceleratorBackend(values["device"])
            values["device_request"] = DeviceRequest(
                policy="manual" if backend else "auto",
                backend=backend,
                device_index=values.get("device_index"),
                allow_fallback=values.get("allow_fallback", True),
            ).model_dump(mode="json")
        return values


class JobValidationRequest(BaseModel):
    engine: str = "openai-whisper"
    model: str = "base"
    device: DeviceRequest = Field(default_factory=DeviceRequest)
    precision: Literal["auto", "float16", "bfloat16", "float32", "int8"] = "auto"


class JobValidationRead(BaseModel):
    valid: bool
    queue: str | None = None
    backend: AcceleratorBackend | None = None
    precision: str | None = None
    worker_id: str | None = None
    issues: list[dict[str, str]] = Field(default_factory=list)


class OutputRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    format: str
    size_bytes: int
    download_url: str


class JobRead(BaseModel):
    id: str
    upload_id: str
    filename: str
    status: JobStatus
    stage: str
    progress: float
    parameters: dict[str, Any]
    device_used: str | None
    gpu_name: str | None
    engine_requested: str
    engine_used: str | None
    device_requested: dict[str, Any]
    backend_used: str | None
    device_index_used: int | None
    runtime_device_used: str | None
    precision_requested: str
    precision_used: str | None
    fallback_used: bool
    fallback_reason: str | None
    worker_id: str | None
    hardware_snapshot: dict[str, Any] | None
    runtime_versions: dict[str, Any] | None
    error_code: str | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    runtime_seconds: float | None
    outputs: list[OutputRead]


class JobList(BaseModel):
    items: list[JobRead]
    total: int
    limit: int
    offset: int
