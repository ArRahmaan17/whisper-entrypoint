import pytest
from pydantic import ValidationError

from app.schemas import JobCreate, WhisperParameters


def test_auto_defaults_to_supported_outputs() -> None:
    parameters = WhisperParameters()
    assert parameters.device == "auto"
    assert parameters.output_formats == ["json", "srt", "txt"]


def test_cpu_rejects_fp16() -> None:
    with pytest.raises(ValidationError, match="fp16"):
        WhisperParameters(device="cpu", fp16=True)


def test_turbo_rejects_translation() -> None:
    with pytest.raises(ValidationError, match="turbo"):
        WhisperParameters(model="turbo", task="translate")


def test_best_of_requires_nonzero_temperature() -> None:
    with pytest.raises(ValidationError, match="best_of"):
        WhisperParameters(temperature=0, best_of=5)


def test_new_execution_request_normalizes_without_breaking_parameters() -> None:
    payload = JobCreate.model_validate(
        {
            "upload_ids": ["upload-1"],
            "engine": "openai-whisper",
            "model": "small",
            "device": {"policy": "manual", "backend": "cpu", "allow_fallback": False},
            "precision": "float32",
            "parameters": {"language": "en", "output_formats": ["txt"]},
        }
    )
    normalized = payload.execution_parameters()
    assert normalized["model"] == "small"
    assert normalized["device_request"]["backend"] == "cpu"
    assert normalized["precision"] == "float32"
    assert normalized["language"] == "en"
