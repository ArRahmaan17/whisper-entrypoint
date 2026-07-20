import asyncio
import hashlib
import json
import shutil
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psutil
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from redis.exceptions import RedisError
from rq.command import send_stop_job_command
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from ..accelerators.models import AcceleratorBackend
from ..config import Settings, get_settings
from ..database import SessionLocal, get_db
from ..models import Job, JobStatus, Upload, WorkerHeartbeat, utcnow
from ..queue import queue_for_backend, queue_name, redis_connection, transcription_queue
from ..schemas import JobCreate, JobList, JobRead, JobValidationRead, JobValidationRequest, OutputRead, UploadRead


router = APIRouter()
MEDIA_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac",
    ".mp4", ".mkv", ".mov", ".avi", ".webm",
}
MODELS = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large", "large-v2", "large-v3", "turbo"]


def _job_read(job: Job) -> JobRead:
    return JobRead(
        id=job.id,
        upload_id=job.upload_id,
        filename=job.upload.original_name,
        status=job.status,
        stage=job.stage,
        progress=job.progress,
        parameters=job.parameters,
        device_used=job.device_used,
        gpu_name=job.gpu_name,
        engine_requested=job.engine_requested,
        engine_used=job.engine_used,
        device_requested=job.device_requested or {},
        backend_used=job.backend_used,
        device_index_used=job.device_index_used,
        runtime_device_used=job.runtime_device_used,
        precision_requested=job.precision_requested,
        precision_used=job.precision_used,
        fallback_used=job.fallback_used,
        fallback_reason=job.fallback_reason,
        worker_id=job.worker_id,
        hardware_snapshot=job.hardware_snapshot,
        runtime_versions=job.runtime_versions,
        error_code=job.error_code,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        runtime_seconds=job.runtime_seconds,
        outputs=[
            OutputRead(
                format=output.format,
                size_bytes=output.size_bytes,
                download_url=f"/jobs/{job.id}/outputs/{output.format}",
            )
            for output in job.outputs
        ],
    )


def _get_job(db: Session, job_id: str) -> Job:
    job = db.scalar(
        select(Job)
        .where(Job.id == job_id)
        .options(selectinload(Job.upload), selectinload(Job.outputs))
    )
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _is_worker_fresh(worker: WorkerHeartbeat, stale_after_seconds: int) -> bool:
    updated_at = worker.updated_at
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    return updated_at >= datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds)


def _worker_read(worker: WorkerHeartbeat, settings: Settings) -> dict:
    fresh = _is_worker_fresh(worker, settings.worker_stale_after_seconds)
    return {
        "worker_id": worker.worker_name,
        "name": worker.worker_name,
        "hostname": worker.hostname,
        "backend": worker.backend,
        "status": worker.status if fresh else "offline",
        "healthy": fresh and worker.status == "online",
        "current_job_id": worker.current_job_id,
        "max_concurrency": worker.max_concurrency,
        "active_jobs": worker.active_jobs,
        "engines": worker.engines or [],
        "devices": worker.devices or [],
        "capabilities": worker.capabilities or [],
        "loaded_models": worker.loaded_models or [],
        "last_error": worker.last_error,
        "started_at": worker.started_at,
        "last_seen_at": worker.updated_at,
        "updated_at": worker.updated_at,
        "details": worker.details,
    }


def _fresh_workers(db: Session, settings: Settings) -> list[WorkerHeartbeat]:
    workers = db.scalars(select(WorkerHeartbeat).order_by(WorkerHeartbeat.worker_name)).all()
    return [item for item in workers if item.status == "online" and _is_worker_fresh(item, settings.worker_stale_after_seconds)]


def _candidate_workers(
    db: Session,
    settings: Settings,
    request: JobValidationRequest,
) -> list[WorkerHeartbeat]:
    workers = _fresh_workers(db, settings)
    by_backend: dict[str, list[WorkerHeartbeat]] = {}
    for worker in workers:
        if worker.backend:
            by_backend.setdefault(worker.backend, []).append(worker)
    priority = [item.value for item in settings.priority]
    if request.device.backend:
        requested = request.device.backend.value
        backend_order = [requested]
        if request.device.allow_fallback:
            backend_order.extend(item for item in priority if item != requested)
    else:
        backend_order = priority
    ordered: list[WorkerHeartbeat] = []
    for backend in backend_order:
        ordered.extend(sorted(by_backend.get(backend, []), key=lambda item: (item.active_jobs, item.worker_name)))
    return ordered


def _validate_against_workers(
    payload: JobValidationRequest,
    db: Session,
    settings: Settings,
) -> JobValidationRead:
    for worker in _candidate_workers(db, settings, payload):
        if payload.engine not in (worker.engines or []):
            continue
        device = next(
            (
                item for item in (worker.devices or [])
                if payload.device.device_index is None or item.get("index") == payload.device.device_index
            ),
            None,
        )
        if device is None or not device.get("healthy", True):
            continue
        capability = next((item for item in (worker.capabilities or []) if item.get("backend") == worker.backend), None)
        if capability is None:
            continue
        if payload.engine not in capability.get("engines", []):
            continue
        precisions = capability.get("precisions", [])
        resolved_precision = payload.precision
        if resolved_precision == "auto":
            preference = ["float32"] if worker.backend == "cpu" else ["float16", "bfloat16", "float32"]
            resolved_precision = next((item for item in preference if item in precisions), "")
        if resolved_precision not in precisions:
            continue
        return JobValidationRead(
            valid=True,
            queue=queue_name(worker.backend),
            backend=AcceleratorBackend(worker.backend),
            precision=resolved_precision,
            worker_id=worker.worker_name,
        )
    return JobValidationRead(
        valid=False,
        issues=[{"code": "WORKER_UNAVAILABLE", "message": "No fresh compatible worker is currently registered"}],
    )


@router.get("/health")
def health(db: Session = Depends(get_db)) -> dict:
    db.execute(select(1))
    try:
        redis_ok = bool(redis_connection.ping())
    except RedisError:
        redis_ok = False
    return {"status": "ok" if redis_ok else "degraded", "database": True, "redis": redis_ok}


@router.get("/system/gpu")
def system_gpu(db: Session = Depends(get_db)) -> dict:
    heartbeat = db.scalar(
        select(WorkerHeartbeat)
        .where(WorkerHeartbeat.backend == AcceleratorBackend.cuda.value)
        .order_by(WorkerHeartbeat.updated_at.desc())
        .limit(1)
    )
    if heartbeat is None:
        return {"available": False, "worker_status": "offline", "devices": [], "error": "No worker heartbeat received"}
    return {
        **heartbeat.details,
        "worker_name": heartbeat.worker_name,
        "worker_status": heartbeat.status,
        "updated_at": heartbeat.updated_at,
    }


@router.get("/system/workers")
def system_workers(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> list[dict]:
    workers = db.scalars(select(WorkerHeartbeat).order_by(WorkerHeartbeat.worker_name)).all()
    return [_worker_read(item, settings) for item in workers]


@router.get("/system/workers/{worker_id}")
def system_worker_detail(
    worker_id: str,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    worker = db.get(WorkerHeartbeat, worker_id)
    if worker is None:
        raise HTTPException(status_code=404, detail="Worker not found")
    return _worker_read(worker, settings)


@router.get("/system/accelerators")
def system_accelerators(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    workers = db.scalars(select(WorkerHeartbeat).order_by(WorkerHeartbeat.worker_name)).all()
    normalized = [_worker_read(item, settings) for item in workers]
    backends: dict[str, dict] = {}
    for worker in normalized:
        backend = worker.get("backend")
        if not backend:
            continue
        summary = backends.setdefault(
            backend,
            {"backend": backend, "available": False, "healthy_workers": 0, "devices": [], "experimental": backend not in {"cuda", "cpu"}},
        )
        summary["available"] = summary["available"] or worker["healthy"]
        summary["healthy_workers"] += int(worker["healthy"])
        summary["devices"].extend(worker["devices"])
    return {
        "selected_policy": settings.default_device,
        "priority": [item.value for item in settings.priority],
        "backends": list(backends.values()),
        "workers": normalized,
        "updated_at": datetime.now(timezone.utc),
    }


@router.get("/system/compatibility")
def system_compatibility(
    engine: str | None = None,
    model: str | None = None,
    backend: AcceleratorBackend | None = None,
    precision: str | None = None,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    inventory = system_accelerators(db, settings)
    combinations = []
    for item in inventory["backends"]:
        if backend and item["backend"] != backend.value:
            continue
        for worker in (entry for entry in inventory["workers"] if entry["backend"] == item["backend"]):
            for capability in worker["capabilities"]:
                for engine_name in worker["engines"]:
                    if engine and engine_name != engine:
                        continue
                    supported_precisions = capability.get("precisions", [])
                    if precision and precision != "auto" and precision not in supported_precisions:
                        continue
                    combinations.append(
                        {
                            "engine": engine_name,
                            "backend": item["backend"],
                            "worker_id": worker["worker_id"],
                            "devices": worker["devices"],
                            "precisions": supported_precisions,
                            "models": capability.get("supported_models"),
                            "experimental": capability.get("experimental", False),
                            "available": worker["healthy"],
                        }
                    )
    return {"model": model, "combinations": combinations, "updated_at": datetime.now(timezone.utc)}


@router.get("/system/resources")
def system_resources(settings: Settings = Depends(get_settings)) -> dict:
    disk = psutil.disk_usage(settings.output_dir)
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": disk.percent,
        "disk_free_bytes": disk.free,
    }


@router.get("/models")
def models() -> list[dict]:
    vram = {"tiny": 1024, "base": 1024, "small": 2048, "medium": 5120, "large": 10240, "large-v2": 10240, "large-v3": 10240, "turbo": 6144}
    return [{"id": name, "estimated_vram_mb": vram.get(name.replace(".en", ""))} for name in MODELS]


@router.post("/jobs/validate", response_model=JobValidationRead)
def validate_job(
    payload: JobValidationRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> JobValidationRead:
    if payload.model not in MODELS:
        return JobValidationRead(valid=False, issues=[{"code": "MODEL_UNSUPPORTED", "message": f"Unknown model: {payload.model}"}])
    if payload.engine != "openai-whisper":
        return JobValidationRead(valid=False, issues=[{"code": "ENGINE_UNSUPPORTED", "message": f"Unknown engine: {payload.engine}"}])
    return _validate_against_workers(payload, db, settings)


@router.post("/uploads", response_model=list[UploadRead], status_code=201)
async def upload_files(
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> list[Upload]:
    if not files or len(files) > 100:
        raise HTTPException(status_code=400, detail="Upload between 1 and 100 files")
    max_bytes = settings.max_upload_mb * 1024 * 1024
    created: list[Upload] = []
    for incoming in files:
        original_name = Path(incoming.filename or "upload").name
        suffix = Path(original_name).suffix.lower()
        if suffix not in MEDIA_EXTENSIONS:
            raise HTTPException(status_code=415, detail=f"Unsupported media extension: {suffix or '(none)'}")
        stored_name = f"{uuid.uuid4()}{suffix}"
        destination = settings.upload_dir / stored_name
        digest = hashlib.sha256()
        size = 0
        try:
            with destination.open("wb") as target:
                while chunk := await incoming.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        raise HTTPException(status_code=413, detail=f"{original_name} exceeds {settings.max_upload_mb} MiB")
                    digest.update(chunk)
                    target.write(chunk)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        record = Upload(
            original_name=original_name,
            stored_name=stored_name,
            content_type=incoming.content_type,
            size_bytes=size,
            sha256=digest.hexdigest(),
            path=str(destination),
        )
        db.add(record)
        created.append(record)
    db.commit()
    for record in created:
        db.refresh(record)
    return created


@router.post("/jobs", response_model=list[JobRead], status_code=201)
def create_jobs(
    payload: JobCreate,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> list[JobRead]:
    uploads = db.scalars(select(Upload).where(Upload.id.in_(payload.upload_ids))).all()
    by_id = {upload.id: upload for upload in uploads}
    missing = [upload_id for upload_id in payload.upload_ids if upload_id not in by_id]
    if missing:
        raise HTTPException(status_code=404, detail={"missing_upload_ids": missing})

    parameters = payload.execution_parameters()
    validation_payload = JobValidationRequest(
        engine=parameters["engine"],
        model=parameters["model"],
        device=parameters["device_request"],
        precision=parameters["precision"],
    )
    assignment = _validate_against_workers(validation_payload, db, settings)
    requested_backend = validation_payload.device.backend
    assigned_backend = assignment.backend or requested_backend
    parameters["assigned_backend"] = assigned_backend.value if assigned_backend else None
    if assignment.valid and requested_backend and assignment.backend != requested_backend:
        parameters["assignment_fallback_reason"] = f"No compatible {requested_backend.value} worker; assigned {assignment.backend.value}"

    jobs: list[Job] = []
    for upload_id in payload.upload_ids:
        job = Job(
            upload_id=upload_id,
            parameters=parameters,
            engine_requested=parameters["engine"],
            device_requested=parameters["device_request"],
            precision_requested=parameters["precision"],
        )
        db.add(job)
        jobs.append(job)
    db.commit()

    try:
        target_queue = queue_for_backend(assigned_backend)
        for job in jobs:
            queued = target_queue.enqueue(
                "app.tasks.transcribe_job",
                job.id,
                job_timeout=-1,
                result_ttl=86400,
                failure_ttl=604800,
            )
            job.queue_job_id = queued.id
        db.commit()
    except RedisError as exc:
        for job in jobs:
            job.status = JobStatus.failed
            job.stage = "queue_unavailable"
            job.error = str(exc)
            job.completed_at = utcnow()
        db.commit()
        raise HTTPException(status_code=503, detail="Queue is unavailable") from exc

    return [_job_read(_get_job(db, job.id)) for job in jobs]


@router.get("/jobs", response_model=JobList)
def list_jobs(
    status: JobStatus | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> JobList:
    condition = Job.status == status if status else True
    total = db.scalar(select(func.count(Job.id)).where(condition)) or 0
    items = db.scalars(
        select(Job)
        .where(condition)
        .options(selectinload(Job.upload), selectinload(Job.outputs))
        .order_by(Job.created_at.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return JobList(items=[_job_read(item) for item in items], total=total, limit=limit, offset=offset)


@router.get("/jobs/{job_id}", response_model=JobRead)
def job_detail(job_id: str, db: Session = Depends(get_db)) -> JobRead:
    return _job_read(_get_job(db, job_id))


@router.post("/jobs/{job_id}/cancel", response_model=JobRead)
def cancel_job(job_id: str, db: Session = Depends(get_db)) -> JobRead:
    job = _get_job(db, job_id)
    if job.status in {JobStatus.completed, JobStatus.failed, JobStatus.cancelled}:
        raise HTTPException(status_code=409, detail=f"Cannot cancel a {job.status.value} job")
    if job.queue_job_id:
        try:
            send_stop_job_command(redis_connection, job.queue_job_id)
        except Exception:
            try:
                queued = transcription_queue.fetch_job(job.queue_job_id)
                if queued:
                    queued.cancel()
            except Exception:
                pass
    job.status = JobStatus.cancelled
    job.stage = "cancelled"
    job.completed_at = utcnow()
    db.commit()
    return _job_read(_get_job(db, job_id))


@router.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str, db: Session = Depends(get_db)) -> None:
    job = _get_job(db, job_id)
    if job.status in {JobStatus.queued, JobStatus.running}:
        raise HTTPException(status_code=409, detail="Cancel the job before deleting it")
    output_dir = get_settings().output_dir / job.id
    db.delete(job)
    db.commit()
    shutil.rmtree(output_dir, ignore_errors=True)


@router.get("/jobs/{job_id}/outputs/{output_format}")
def download_output(job_id: str, output_format: str, db: Session = Depends(get_db)) -> FileResponse:
    job = _get_job(db, job_id)
    output = next((item for item in job.outputs if item.format == output_format), None)
    if output is None or not Path(output.path).is_file():
        raise HTTPException(status_code=404, detail="Output not found")
    filename = f"{Path(job.upload.original_name).stem}.{output.format}"
    return FileResponse(output.path, filename=filename)


@router.get("/jobs/{job_id}/outputs.zip")
def download_all_outputs(job_id: str, db: Session = Depends(get_db)) -> FileResponse:
    job = _get_job(db, job_id)
    if not job.outputs:
        raise HTTPException(status_code=404, detail="No outputs found")
    zip_path = get_settings().output_dir / job.id / "outputs.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for output in job.outputs:
            path = Path(output.path)
            if path.is_file():
                archive.write(path, arcname=f"{Path(job.upload.original_name).stem}.{output.format}")
    return FileResponse(zip_path, filename=f"{Path(job.upload.original_name).stem}-outputs.zip")


@router.websocket("/ws/jobs/{job_id}")
async def job_progress(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    try:
        while True:
            with SessionLocal() as db:
                try:
                    payload = _job_read(_get_job(db, job_id)).model_dump(mode="json")
                except HTTPException:
                    await websocket.send_json({"error": "Job not found"})
                    await websocket.close(code=4404)
                    return
            await websocket.send_json(payload)
            if payload["status"] in {"completed", "failed", "cancelled"}:
                await websocket.close()
                return
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        return
