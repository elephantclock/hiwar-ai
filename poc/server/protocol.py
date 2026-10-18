from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


PROTOCOL_VERSION = 1


class SessionState(StrEnum):
    LISTENING = "LISTENING"
    ENDPOINTING = "ENDPOINTING"
    TRANSCRIBING = "TRANSCRIBING"
    THINKING = "THINKING"
    TOOL_RUNNING = "TOOL_RUNNING"
    SYNTHESIZING = "SYNTHESIZING"
    SPEAKING = "SPEAKING"


class ClientEvent(StrEnum):
    SESSION_CONFIGURE = "session.configure"
    AUDIO_COMMIT = "input_audio.commit"
    RESPONSE_CANCEL = "response.cancel"
    PING = "ping"


class ServerEvent(StrEnum):
    SESSION_CREATED = "session.created"
    SESSION_STATE = "session.state"
    TRANSCRIPT_FINAL = "transcript.final"
    RESPONSE_TEXT_DELTA = "response.text.delta"
    RESPONSE_AUDIO_CHUNK = "response.audio.chunk"
    RESPONSE_DONE = "response.done"
    RESPONSE_CANCELLED = "response.cancelled"
    ERROR = "error"
    PONG = "pong"


@dataclass(slots=True)
class Event:
    type: str
    session_id: str
    turn_id: int
    seq: int
    payload: dict[str, Any] = field(default_factory=dict)
    v: int = PROTOCOL_VERSION
    ts: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


def parse_client_event(raw: str) -> dict[str, Any]:
    data = json.loads(raw)
    if data.get("v") != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version: {data.get('v')!r}")
    if data.get("type") not in {event.value for event in ClientEvent}:
        raise ValueError(f"unsupported client event: {data.get('type')!r}")
    return data
