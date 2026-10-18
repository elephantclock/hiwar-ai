from fastapi.testclient import TestClient

from server.app import app


def test_health_and_session_contract() -> None:
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        response = client.post("/api/v1/sessions")

    assert response.status_code == 201
    payload = response.json()
    assert payload["protocol_version"] == 1
    assert payload["audio"] == {
        "encoding": "pcm_s16le",
        "sample_rate": 16_000,
        "channels": 1,
    }
    assert payload["stream_url"].endswith(f"/{payload['id']}/stream")


def test_websocket_ping_round_trip() -> None:
    with TestClient(app) as client:
        session = client.post("/api/v1/sessions").json()
        with client.websocket_connect(session["stream_url"]) as socket:
            created = socket.receive_json()
            assert created["type"] == "session.created"

            socket.send_json({"v": 1, "type": "ping", "payload": {}})
            pong = socket.receive_json()
            assert pong["type"] == "pong"
            assert pong["session_id"] == session["id"]
