from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.api import routes
from app.database import SessionLocal
from app.main import app
from app.models import WorkerHeartbeat


class FakeQueuedJob:
    id = "rq-test-job"


class FakeQueue:
    def enqueue(self, *args, **kwargs):
        return FakeQueuedJob()


def test_models_are_available() -> None:
    with TestClient(app) as client:
        response = client.get("/models")
    assert response.status_code == 200
    assert any(model["id"] == "turbo" for model in response.json())


def test_upload_and_create_cuda_job(monkeypatch) -> None:
    monkeypatch.setattr(routes, "queue_for_backend", lambda backend: FakeQueue())
    with TestClient(app) as client:
        upload = client.post("/uploads", files=[("files", ("meeting.wav", b"RIFF-test", "audio/wav"))])
        assert upload.status_code == 201
        upload_id = upload.json()[0]["id"]

        created = client.post(
            "/jobs",
            json={
                "upload_ids": [upload_id],
                "parameters": {"model": "base", "device": "cuda", "output_formats": ["json", "txt"]},
            },
        )

    assert created.status_code == 201
    job = created.json()[0]
    assert job["status"] == "queued"
    assert job["filename"] == "meeting.wav"
    assert job["parameters"]["device"] == "cuda"
    assert job["engine_requested"] == "openai-whisper"
    assert job["device_requested"]["backend"] == "cuda"


def test_upload_rejects_unknown_extension() -> None:
    with TestClient(app) as client:
        response = client.post("/uploads", files=[("files", ("notes.exe", b"nope", "application/octet-stream"))])
    assert response.status_code == 415


def test_accelerator_inventory_and_validation() -> None:
    with SessionLocal() as db:
        db.merge(
            WorkerHeartbeat(
                worker_name="cpu-test",
                hostname="test-host",
                backend="cpu",
                status="online",
                engines=["openai-whisper"],
                devices=[{"backend": "cpu", "index": None, "id": "cpu", "name": "Test CPU", "vendor": "host", "device_type": "cpu", "healthy": True}],
                capabilities=[{"backend": "cpu", "precisions": ["float32"], "engines": ["openai-whisper"]}],
                details={"available": True, "backend": "cpu"},
            )
        )
        db.commit()

    with TestClient(app) as client:
        inventory = client.get("/system/accelerators")
        validation = client.post(
            "/jobs/validate",
            json={"engine": "openai-whisper", "model": "base", "device": {"policy": "auto"}, "precision": "auto"},
        )

    assert inventory.status_code == 200
    assert any(item["backend"] == "cpu" for item in inventory.json()["backends"])
    assert validation.status_code == 200
    assert validation.json()["valid"] is True
    assert validation.json()["backend"] == "cpu"
    assert validation.json()["precision"] == "float32"


def test_stale_worker_is_reported_offline() -> None:
    worker = WorkerHeartbeat(
        worker_name="stale-test",
        status="online",
        updated_at=datetime.now(timezone.utc) - timedelta(minutes=5),
    )
    assert routes._is_worker_fresh(worker, stale_after_seconds=30) is False
