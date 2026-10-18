from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from ..settings import Settings
from .base import Message, ToolDefinition


class GroqProvider:
    """Groq adapter; credentials never cross the server boundary."""

    base_url = "https://api.groq.com/openai/v1"

    def __init__(self, config: Settings) -> None:
        if not config.groq_api_key:
            raise ValueError("GROQ_API_KEY is required for hosted providers")
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {config.groq_api_key}"},
            timeout=config.provider_timeout_ms / 1000,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def transcribe(self, wav_audio: bytes) -> str:
        response = await self._client.post(
            "/audio/transcriptions",
            data={"model": self.config.groq_stt_model, "language": "en"},
            files={"file": ("utterance.wav", wav_audio, "audio/wav")},
        )
        response.raise_for_status()
        return str(response.json()["text"]).strip()

    async def stream_reply(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> AsyncIterator[dict[str, Any]]:
        body: dict[str, Any] = {
            "model": self.config.groq_chat_model,
            "messages": list(messages),
            "stream": True,
        }
        if tools:
            body["tools"] = list(tools)

        async with self._client.stream("POST", "/chat/completions", json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    return
                yield json.loads(data)

    async def synthesize(self, text: str, voice: str, emotion: str) -> bytes:
        response = await self._client.post(
            "/audio/speech",
            json={
                "model": self.config.groq_tts_model,
                "voice": voice,
                "input": f"<{emotion}>{text}",
                "response_format": "wav",
            },
        )
        response.raise_for_status()
        return response.content
