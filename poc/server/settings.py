from __future__ import annotations

import os
from dataclasses import dataclass


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_stt_model: str = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
    groq_chat_model: str = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
    groq_tts_model: str = os.getenv("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english")
    groq_tts_voice: str = os.getenv("GROQ_TTS_VOICE", "autumn")
    sample_rate: int = int(os.getenv("VOICE_SAMPLE_RATE", "16000"))
    endpoint_silence_ms: int = int(os.getenv("VOICE_ENDPOINT_SILENCE_MS", "450"))
    max_utterance_seconds: int = int(os.getenv("VOICE_MAX_UTTERANCE_SECONDS", "30"))
    tool_timeout_ms: int = int(os.getenv("VOICE_TOOL_TIMEOUT_MS", "2500"))
    provider_timeout_ms: int = int(os.getenv("VOICE_PROVIDER_TIMEOUT_MS", "10000"))
    store_transcripts: bool = _bool_env("VOICE_STORE_TRANSCRIPTS")
    log_level: str = os.getenv("VOICE_LOG_LEVEL", "INFO")

    @property
    def providers_configured(self) -> bool:
        return bool(self.groq_api_key)


settings = Settings()
