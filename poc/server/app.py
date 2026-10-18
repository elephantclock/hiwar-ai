from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect, status
from fastapi.staticfiles import StaticFiles

from .audio import duration_ms
from .protocol import (
    ClientEvent,
    Event,
    ServerEvent,
    SessionState,
    parse_client_event,
)
from .sessions import SessionStore, VoiceSession
from .settings import settings


logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("hiwar.poc")

app = FastAPI(
    title="Hiwar Voice PoC",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url=None,
)
sessions = SessionStore()
web_root = Path(__file__).resolve().parents[1] / "web"


def make_event(
    session: VoiceSession,
    event_type: ServerEvent,
    payload: dict[str, Any] | None = None,
) -> Event:
    return Event(
        type=event_type.value,
        session_id=session.id,
        turn_id=session.turn_id,
        seq=session.next_seq(),
        payload=payload or {},
    )


async def send_event(
    socket: WebSocket,
    session: VoiceSession,
    event_type: ServerEvent,
    payload: dict[str, Any] | None = None,
) -> None:
    await socket.send_text(make_event(session, event_type, payload).to_json())


async def send_state(
    socket: WebSocket,
    session: VoiceSession,
    state: SessionState,
) -> None:
    session.state = state
    await send_event(socket, session, ServerEvent.SESSION_STATE, {"state": state.value})


@app.get("/healthz", tags=["operations"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz", tags=["operations"])
async def ready(response: Response) -> dict[str, Any]:
    if not settings.providers_configured:
        response.status_code = status.HTTP_200_OK
    return {
        "status": "ready" if settings.providers_configured else "scaffold",
        "providers_configured": settings.providers_configured,
    }


@app.post("/api/v1/sessions", status_code=status.HTTP_201_CREATED, tags=["sessions"])
async def create_session() -> dict[str, Any]:
    session = await sessions.create()
    return {
        "id": session.id,
        "protocol_version": 1,
        "stream_url": f"/api/v1/sessions/{session.id}/stream",
        "audio": {
            "encoding": "pcm_s16le",
            "sample_rate": settings.sample_rate,
            "channels": 1,
        },
    }


@app.delete(
    "/api/v1/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["sessions"],
)
async def delete_session(session_id: str) -> Response:
    if not await sessions.delete(session_id):
        raise HTTPException(status_code=404, detail="session not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.websocket("/api/v1/sessions/{session_id}/stream")
async def session_stream(socket: WebSocket, session_id: str) -> None:
    session = await sessions.get(session_id)
    if session is None:
        await socket.close(code=4404, reason="session not found")
        return

    await socket.accept()
    await send_event(
        socket,
        session,
        ServerEvent.SESSION_CREATED,
        {"state": session.state.value},
    )

    max_audio_bytes = settings.max_utterance_seconds * settings.sample_rate * 2
    try:
        while True:
            message = await socket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                audio = message["bytes"]
                if session.state != SessionState.LISTENING:
                    continue
                if len(session.input_audio) + len(audio) > max_audio_bytes:
                    await send_event(
                        socket,
                        session,
                        ServerEvent.ERROR,
                        {"code": "UTTERANCE_TOO_LONG", "recoverable": True},
                    )
                    session.input_audio.clear()
                    continue
                session.input_audio.extend(audio)
                continue

            raw = message.get("text")
            if raw is None:
                continue
            try:
                event = parse_client_event(raw)
            except (ValueError, TypeError) as error:
                await send_event(
                    socket,
                    session,
                    ServerEvent.ERROR,
                    {"code": "BAD_EVENT", "message": str(error), "recoverable": True},
                )
                continue

            event_type = event["type"]
            if event_type == ClientEvent.PING:
                await send_event(socket, session, ServerEvent.PONG)
            elif event_type == ClientEvent.SESSION_CONFIGURE:
                await send_state(socket, session, SessionState.LISTENING)
            elif event_type == ClientEvent.RESPONSE_CANCEL:
                cancelled_turn = session.turn_id
                session.cancel_turn()
                await send_event(
                    socket,
                    session,
                    ServerEvent.RESPONSE_CANCELLED,
                    {"cancelled_turn_id": cancelled_turn},
                )
                await send_state(socket, session, SessionState.LISTENING)
            elif event_type == ClientEvent.AUDIO_COMMIT:
                if not session.input_audio:
                    await send_event(
                        socket,
                        session,
                        ServerEvent.ERROR,
                        {"code": "EMPTY_AUDIO", "recoverable": True},
                    )
                    continue
                session.begin_turn()
                await send_state(socket, session, SessionState.ENDPOINTING)
                audio_ms = duration_ms(bytes(session.input_audio), settings.sample_rate)
                session.input_audio.clear()
                await send_event(
                    socket,
                    session,
                    ServerEvent.ERROR,
                    {
                        "code": "TURN_PIPELINE_NOT_WIRED",
                        "message": "Audio transport is working; connect TurnCoordinator next.",
                        "audio_ms": audio_ms,
                        "recoverable": True,
                    },
                )
                await send_state(socket, session, SessionState.LISTENING)
    except WebSocketDisconnect:
        logger.info("session disconnected", extra={"session_id": session.id})
    finally:
        await sessions.delete(session.id)


app.mount("/", StaticFiles(directory=web_root, html=True), name="web")
