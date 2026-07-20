from redis import Redis
from rq import Queue

from .accelerators.models import AcceleratorBackend
from .config import get_settings


settings = get_settings()
redis_connection = Redis.from_url(settings.redis_url)


def queue_name(backend: AcceleratorBackend | str | None) -> str:
    value = backend.value if isinstance(backend, AcceleratorBackend) else backend
    return f"transcription:{value or 'auto'}"


def queue_for_backend(backend: AcceleratorBackend | str | None) -> Queue:
    return Queue(queue_name(backend), connection=redis_connection, default_timeout=-1)


# Kept for callers that still patch or inspect the legacy queue object.
transcription_queue = Queue("transcription", connection=redis_connection, default_timeout=-1)
