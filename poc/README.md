# Hiwar English Voice PoC

This directory is the implementation scaffold for the
[browser voice-agent architecture](https://elephantclock.github.io/hiwar-ai/architecture.html).
It keeps the browser, orchestration, tools and hosted model adapters independently
testable.

## What works in this scaffold

- A responsive browser call UI served by FastAPI.
- Microphone capture with browser echo cancellation and noise suppression.
- AudioWorklet resampling to mono PCM16 at 16 kHz in 20 ms frames.
- Versioned JSON control events plus binary audio over one WebSocket.
- Session state, turn epochs, interruption and stale-playback cancellation.
- Typed STT, chat and TTS boundaries with a Groq adapter.
- A server-side tool registry backed by mock order data.
- Unit tests for audio framing, protocol validation and sentence assembly.

The WebSocket currently stops after validating a committed utterance and emits
`TURN_PIPELINE_NOT_WIRED`. This is intentional: connect `TurnCoordinator`,
`AgentRuntime` and sentence-level TTS in the next implementation slice. The
provider key is never sent to the browser.

## Run locally

Requirements: Python 3.11+ and a Chromium-, Firefox- or Safari-based browser.

```bash
cd poc
python3 -m venv .venv
python -m pip install --upgrade pip setuptools wheel
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install -e ".[dev]"
cp .env.example .env
uvicorn server.app:app --reload --port 8000
```

Open <http://localhost:8000>. Microphone access is permitted on `localhost`.
The OpenAPI explorer is at <http://localhost:8000/api/docs>.

Environment variables are read by the server process. Export them before
starting Uvicorn; `.env` is ignored by Git and is not loaded automatically.

## Verify

```bash
cd poc
python -m unittest discover -s tests/unit -v
python -m compileall -q server
```

## Repository map

```text
poc/
├── web/                    Browser UI, capture and playback
├── server/
│   ├── agent/              Prompt, model loop and server-owned tools
│   ├── providers/          Vendor-neutral interfaces and Groq adapter
│   ├── app.py              HTTP/WebSocket composition root
│   ├── audio.py            PCM/WAV boundary
│   ├── protocol.py         Versioned wire contract
│   ├── sessions.py         In-memory PoC session state
│   └── turns.py            Interruptible turn coordination
├── fixtures/               Synthetic, non-customer test records
└── tests/
    ├── unit/               Deterministic component tests
    ├── integration/        API/provider contract test home
    └── scenarios/          End-to-end call scenarios
```

## Engineering boundaries

- Treat browser audio as untrusted input; cap duration and buffer sizes.
- Do not log raw audio, provider keys, or customer records.
- Keep tool authorization on the server and confirm all write operations.
- Check `turn_id` immediately before emitting text or audio so cancelled work
  cannot reach the caller.
- Replace the in-memory session store before running more than one server
  process.
