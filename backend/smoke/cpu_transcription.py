import json
import tempfile
import wave
from pathlib import Path

import torch
import whisper


def create_silence(path: Path, seconds: int = 1, sample_rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\x00\x00" * sample_rate * seconds)


def main() -> None:
    assert not torch.cuda.is_available(), "CPU smoke image unexpectedly exposes CUDA"
    with tempfile.TemporaryDirectory(prefix="whisper-cpu-smoke-") as temporary:
        root = Path(temporary)
        source = root / "silence.wav"
        cache = root / "models"
        create_silence(source)
        model = whisper.load_model("tiny.en", device="cpu", download_root=str(cache))
        result = model.transcribe(str(source), language="en", fp16=False, temperature=0)
        assert isinstance(result.get("text"), str)
        assert isinstance(result.get("segments"), list)
        print(json.dumps({"ok": True, "model": "tiny.en", "device": "cpu", "text_length": len(result["text"])}))


if __name__ == "__main__":
    main()
