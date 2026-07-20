from app.gpu import validate_gpu_safety


def status(*, temperature: int = 45, free_vram: int = 8192) -> dict:
    return {
        "available": True,
        "torch_cuda": True,
        "devices": [{"index": 0, "temperature_c": temperature, "memory_free_mb": free_vram}],
    }


def test_accepts_healthy_gpu() -> None:
    safe, reason = validate_gpu_safety(status(), stop_temp_c=87, min_free_vram_mb=1024)
    assert safe is True
    assert reason is None


def test_rejects_hot_gpu() -> None:
    safe, reason = validate_gpu_safety(status(temperature=87), stop_temp_c=87, min_free_vram_mb=1024)
    assert safe is False
    assert "temperature" in str(reason)


def test_rejects_low_vram() -> None:
    safe, reason = validate_gpu_safety(status(free_vram=512), stop_temp_c=87, min_free_vram_mb=1024)
    assert safe is False
    assert "free VRAM" in str(reason)


def test_rejects_missing_cuda() -> None:
    safe, reason = validate_gpu_safety({"available": False, "torch_cuda": False}, 87, 1024)
    assert safe is False
    assert reason == "CUDA is unavailable to the worker"

