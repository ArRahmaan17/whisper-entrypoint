import os
import shutil
import tempfile
from pathlib import Path


TEST_ROOT = Path(tempfile.mkdtemp(prefix="whisper-docker-tests-"))
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_ROOT / 'test.db'}"
os.environ["UPLOAD_DIR"] = str(TEST_ROOT / "uploads")
os.environ["OUTPUT_DIR"] = str(TEST_ROOT / "outputs")
os.environ["MODEL_CACHE_DIR"] = str(TEST_ROOT / "models")


def pytest_sessionfinish(session, exitstatus) -> None:
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
