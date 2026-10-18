from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from .audio import pcm16_to_wav
from .protocol import SessionState
from .providers.base import SpeechToTextProvider
from .sessions import VoiceSession


StateCallback = Callable[[SessionState, int], Awaitable[None]]


class TurnCoordinator:
    """Coordinates one interruptible turn; higher epochs invalidate old work."""

    def __init__(self, stt: SpeechToTextProvider) -> None:
        self.stt = stt
        self._task: asyncio.Task[None] | None = None

    def cancel(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    async def transcribe(
        self,
        session: VoiceSession,
        on_state: StateCallback,
    ) -> str:
        epoch = session.turn_id
        await on_state(SessionState.TRANSCRIBING, epoch)
        wav_audio = pcm16_to_wav(bytes(session.input_audio))
        transcript = await self.stt.transcribe(wav_audio)
        if session.turn_id != epoch:
            raise asyncio.CancelledError
        return transcript
