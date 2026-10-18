from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol


Message = dict[str, Any]
ToolDefinition = dict[str, Any]


class SpeechToTextProvider(Protocol):
    async def transcribe(self, wav_audio: bytes) -> str: ...


class ChatProvider(Protocol):
    def stream_reply(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> AsyncIterator[dict[str, Any]]: ...


class TextToSpeechProvider(Protocol):
    async def synthesize(self, text: str, voice: str, emotion: str) -> bytes: ...
