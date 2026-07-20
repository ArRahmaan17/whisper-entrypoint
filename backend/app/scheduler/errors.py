from enum import StrEnum


class JobErrorCode(StrEnum):
    accelerator_unavailable = "ACCELERATOR_UNAVAILABLE"
    accelerator_unhealthy = "ACCELERATOR_UNHEALTHY"
    engine_unsupported = "ENGINE_UNSUPPORTED"
    model_unsupported = "MODEL_UNSUPPORTED"
    precision_unsupported = "PRECISION_UNSUPPORTED"
    insufficient_device_memory = "INSUFFICIENT_DEVICE_MEMORY"
    worker_unavailable = "WORKER_UNAVAILABLE"
    runtime_initialization_failed = "RUNTIME_INITIALIZATION_FAILED"
    model_load_failed = "MODEL_LOAD_FAILED"
    transcription_failed = "TRANSCRIPTION_FAILED"
    device_out_of_memory = "DEVICE_OUT_OF_MEMORY"


class ExecutionError(RuntimeError):
    def __init__(self, code: JobErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

