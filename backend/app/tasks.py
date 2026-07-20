import time
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select

from .accelerators.base import AcceleratorProvider, JobExecutionContext
from .accelerators.models import AcceleratorBackend, ResolvedDevice
from .accelerators.registry import create_provider_registry
from .config import get_settings
from .database import SessionLocal
from .engines.base import ModelLoadRequest, TranscriptionEngine, TranscriptionRequest
from .engines.registry import create_engine_registry
from .models import Job, JobOutput, JobStatus, WorkerHeartbeat, utcnow
from .scheduler.errors import ExecutionError, JobErrorCode
from .scheduler.selector import DeviceRequest, SelectionError, select_device


settings = get_settings()
provider_registry = create_provider_registry()
engine_registry = create_engine_registry()


def _set_job(job_id: str, **values: Any) -> None:
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None:
            return
        for key, value in values.items():
            setattr(job, key, value)
        db.commit()


def _cancelled(job_id: str) -> bool:
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        return job is None or job.status == JobStatus.cancelled


def _request_from_parameters(parameters: dict[str, Any]) -> DeviceRequest:
    if isinstance(parameters.get("device_request"), dict):
        return DeviceRequest.model_validate(parameters["device_request"])
    legacy = str(parameters.get("device", settings.default_device))
    backend = None if legacy == "auto" else AcceleratorBackend(legacy)
    return DeviceRequest(
        policy="manual" if backend else "auto",
        backend=backend,
        device_index=parameters.get("device_index"),
        allow_fallback=bool(parameters.get("allow_fallback", settings.allow_device_fallback)),
    )


def _resolve_device(parameters: dict[str, Any]) -> tuple[ResolvedDevice, AcceleratorProvider, TranscriptionEngine]:
    request = _request_from_parameters(parameters)
    worker_backend = settings.worker_backend
    assigned_backend = parameters.get("assigned_backend")
    if assigned_backend and AcceleratorBackend(assigned_backend) != worker_backend:
        raise ExecutionError(
            JobErrorCode.accelerator_unavailable,
            f"job was assigned to {assigned_backend} but worker provides {worker_backend.value}",
        )
    if not assigned_backend and request.backend is not None and request.backend != worker_backend:
        raise ExecutionError(
            JobErrorCode.accelerator_unavailable,
            f"job requested {request.backend.value} but worker provides {worker_backend.value}",
        )
    local_request = DeviceRequest(
        policy="manual",
        backend=worker_backend,
        device_index=request.device_index,
        allow_fallback=False,
    )
    engine_name = str(parameters.get("engine", settings.default_engine))
    try:
        resolved = select_device(
            provider_registry,
            engine_registry,
            request=local_request,
            priority=[worker_backend],
            engine_name=engine_name,
            model=str(parameters.get("model", settings.default_model)),
            precision=str(parameters.get("precision", "auto")),
        )
    except SelectionError as exc:
        issue = next((item for result in exc.results for item in result.issues), None)
        raise ExecutionError(issue.code if issue else JobErrorCode.accelerator_unavailable, str(exc)) from exc
    fallback_used = request.backend is not None and request.backend != worker_backend
    if fallback_used:
        resolved = resolved.model_copy(
            update={
                "fallback_used": True,
                "fallback_reason": parameters.get("assignment_fallback_reason") or f"fell back from {request.backend.value} to {worker_backend.value}",
            }
        )
    return resolved, provider_registry.get(worker_backend), engine_registry.get(engine_name)


def _transcribe_options(parameters: dict[str, Any], resolved: ResolvedDevice) -> dict[str, Any]:
    allowed = {
        "language",
        "task",
        "temperature",
        "beam_size",
        "best_of",
        "patience",
        "length_penalty",
        "suppress_tokens",
        "initial_prompt",
        "condition_on_previous_text",
        "word_timestamps",
        "prepend_punctuations",
        "append_punctuations",
        "clip_timestamps",
        "hallucination_silence_threshold",
    }
    options = {key: value for key, value in parameters.items() if key in allowed and value is not None}
    options["fp16"] = parameters.get("fp16") if parameters.get("fp16") is not None else resolved.precision == "float16"
    return options


def _write_outputs(
    engine: TranscriptionEngine,
    job: Job,
    result: dict[str, Any],
    formats: list[str],
) -> list[JobOutput]:
    source = Path(job.upload.path)
    output_dir = settings.output_dir / job.id
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = engine.write_outputs(result, source, output_dir, formats)
    return [
            JobOutput(
                job_id=job.id,
                format=path.suffix.removeprefix("."),
                path=str(path),
                size_bytes=path.stat().st_size,
            )
            for path in paths
    ]


def _set_worker_job(job_id: str | None) -> None:
    with SessionLocal() as db:
        worker = db.get(WorkerHeartbeat, settings.worker_name)
        if worker:
            worker.current_job_id = job_id
            worker.active_jobs = 1 if job_id else 0
            db.commit()


def _runtime_snapshot(provider: AcceleratorProvider, resolved: ResolvedDevice) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    devices = provider.list_devices()
    device = next((item for item in devices if item.index == resolved.device_index), devices[0] if devices else None)
    snapshot = device.model_dump(mode="json") if device else None
    versions = {
        "accelerator_runtime": device.runtime_version if device else None,
        "driver": device.driver_version if device else None,
    }
    return snapshot, versions


def transcribe_job(job_id: str) -> None:
    started = time.monotonic()
    provider: AcceleratorProvider | None = None
    context: JobExecutionContext | None = None
    try:
        with SessionLocal() as db:
            job = db.scalar(select(Job).where(Job.id == job_id))
            if job is None or job.status == JobStatus.cancelled:
                return
            job.status = JobStatus.running
            job.stage = "starting"
            job.progress = 0.05
            job.started_at = utcnow()
            db.commit()
            parameters = dict(job.parameters)

        _set_worker_job(job_id)
        resolved, provider, transcription_engine = _resolve_device(parameters)
        context = JobExecutionContext(job_id=job_id, device_index=resolved.device_index, parameters=parameters)
        provider.before_job(context)
        hardware_snapshot, runtime_versions = _runtime_snapshot(provider, resolved)
        device_name = hardware_snapshot.get("name") if hardware_snapshot else None
        _set_job(
            job_id,
            stage="model_loading",
            progress=0.15,
            device_used=resolved.backend.value,
            gpu_name=device_name if resolved.backend != AcceleratorBackend.cpu else None,
            engine_used=resolved.engine,
            backend_used=resolved.backend.value,
            device_index_used=resolved.device_index,
            runtime_device_used=resolved.runtime_device,
            precision_used=resolved.precision,
            fallback_used=resolved.fallback_used,
            fallback_reason=resolved.fallback_reason,
            worker_id=settings.worker_name,
            hardware_snapshot=hardware_snapshot,
            runtime_versions=runtime_versions,
        )

        try:
            model = transcription_engine.load_model(
                ModelLoadRequest(
                    model=str(parameters.get("model", settings.default_model)),
                    resolved_device=resolved,
                    cache_dir=settings.model_cache_dir,
                )
            )
        except Exception as exc:
            raise ExecutionError(JobErrorCode.model_load_failed, f"model load failed: {exc}") from exc
        if _cancelled(job_id):
            return

        _set_job(job_id, stage="transcribing", progress=0.35)
        with SessionLocal() as db:
            job = db.get(Job, job_id)
            if job is None:
                return
            result = transcription_engine.transcribe(
                TranscriptionRequest(
                    source_path=Path(job.upload.path),
                    options=_transcribe_options(parameters, resolved),
                    model=model,
                ),
                lambda stage, progress: _set_job(job_id, stage=stage, progress=progress),
            )

        if _cancelled(job_id):
            return
        _set_job(job_id, stage="writing_outputs", progress=0.9)

        formats = parameters.get("output_formats") or settings.output_formats
        with SessionLocal() as db:
            job = db.get(Job, job_id)
            if job is None:
                return
            db.execute(delete(JobOutput).where(JobOutput.job_id == job_id))
            for output in _write_outputs(transcription_engine, job, result, formats):
                db.add(output)
            job.status = JobStatus.completed
            job.stage = "finished"
            job.progress = 1.0
            job.completed_at = utcnow()
            job.runtime_seconds = round(time.monotonic() - started, 3)
            db.commit()
    except Exception as exc:
        if isinstance(exc, ExecutionError):
            error_code = exc.code.value
            message = exc.message
        else:
            text = str(exc)
            error_code = (
                JobErrorCode.device_out_of_memory.value
                if "out of memory" in text.lower()
                else JobErrorCode.transcription_failed.value
            )
            message = text or "transcription failed"
        _set_job(
            job_id,
            status=JobStatus.failed,
            stage="failed",
            error=message[:1000],
            error_code=error_code,
            completed_at=utcnow(),
            runtime_seconds=round(time.monotonic() - started, 3),
        )
        raise
    finally:
        if provider is not None and context is not None:
            try:
                provider.after_job(context)
            except Exception:
                pass
        _set_worker_job(None)
