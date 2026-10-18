from __future__ import annotations

import io
import wave


def pcm16_to_wav(pcm: bytes, sample_rate: int = 16_000) -> bytes:
    """Wrap little-endian mono PCM16 frames in a WAV container."""
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()


def duration_ms(pcm: bytes, sample_rate: int = 16_000) -> int:
    samples = len(pcm) // 2
    return round(samples / sample_rate * 1000)
