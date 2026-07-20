import os
import socket
import multiprocessing
from datetime import datetime, timezone
from typing import Any

from redis import Redis
from rq import Queue, Worker

from .accelerators.registry import create_provider_registry
from .config import get_settings
from .database import SessionLocal, create_tables
from .models import WorkerHeartbeat
from .queue import queue_name


settings = get_settings()


def _heartbeat(stop: Any, started_at: datetime) -> None:
    provider = create_provider_registry().get(settings.worker_backend)
    while not stop.is_set():
        last_error = None
        try:
            available = provider.is_runtime_available()
            devices = provider.list_devices() if available else []
            capabilities = [provider.get_capability(device.index).model_dump(mode="json") for device in devices]
            engine_names = sorted({engine for capability in capabilities for engine in capability.get("engines", [])})
            healthy = available and bool(devices) and all(device.healthy for device in devices)
        except Exception as exc:
            available = False
            devices = []
            capabilities = []
            engine_names = []
            healthy = False
            last_error = str(exc)[:1000]
        serialized_devices = [device.model_dump(mode="json") for device in devices]
        details = {
            "available": available,
            "backend": settings.worker_backend.value,
            "devices": serialized_devices,
            "capabilities": capabilities,
            "engines": engine_names,
        }
        with SessionLocal() as db:
            heartbeat = db.get(WorkerHeartbeat, settings.worker_name)
            if heartbeat is None:
                heartbeat = WorkerHeartbeat(worker_name=settings.worker_name)
                db.add(heartbeat)
            heartbeat.hostname = socket.gethostname()
            heartbeat.backend = settings.worker_backend.value
            heartbeat.status = "online" if healthy else "degraded"
            heartbeat.max_concurrency = settings.max_concurrent_jobs
            heartbeat.engines = engine_names
            heartbeat.devices = serialized_devices
            heartbeat.capabilities = capabilities
            heartbeat.loaded_models = []
            heartbeat.last_error = last_error
            heartbeat.started_at = started_at
            heartbeat.details = details
            heartbeat.updated_at = datetime.now(timezone.utc)
            db.commit()
        stop.wait(settings.worker_heartbeat_seconds)


def main() -> None:
    create_tables()
    started_at = datetime.now(timezone.utc)
    context = multiprocessing.get_context("spawn")
    stop = context.Event()
    heartbeat = context.Process(target=_heartbeat, args=(stop, started_at), daemon=True, name="accelerator-heartbeat")
    heartbeat.start()
    connection = Redis.from_url(settings.redis_url)
    queues = [
        Queue(queue_name(settings.worker_backend), connection=connection, default_timeout=-1),
        Queue(queue_name(None), connection=connection, default_timeout=-1),
        Queue("transcription", connection=connection, default_timeout=-1),
    ]
    worker_name = os.getenv("WORKER_NAME", f"{socket.gethostname()}-{settings.worker_backend.value}")
    try:
        Worker(queues, connection=connection, name=worker_name).work(with_scheduler=True)
    finally:
        stop.set()
        heartbeat.join(timeout=3)


if __name__ == "__main__":
    main()
