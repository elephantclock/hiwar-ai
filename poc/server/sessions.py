from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field

from .protocol import SessionState


@dataclass(slots=True)
class VoiceSession:
    id: str
    state: SessionState = SessionState.LISTENING
    turn_id: int = 0
    seq: int = 0
    created_at: float = field(default_factory=time.time)
    input_audio: bytearray = field(default_factory=bytearray, repr=False)

    def next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def begin_turn(self) -> int:
        self.turn_id += 1
        return self.turn_id

    def cancel_turn(self) -> int:
        self.turn_id += 1
        self.state = SessionState.LISTENING
        self.input_audio.clear()
        return self.turn_id


class SessionStore:
    def __init__(self) -> None:
        self._items: dict[str, VoiceSession] = {}
        self._lock = asyncio.Lock()

    async def create(self) -> VoiceSession:
        session = VoiceSession(id=uuid.uuid4().hex)
        async with self._lock:
            self._items[session.id] = session
        return session

    async def get(self, session_id: str) -> VoiceSession | None:
        async with self._lock:
            return self._items.get(session_id)

    async def delete(self, session_id: str) -> bool:
        async with self._lock:
            return self._items.pop(session_id, None) is not None
