import json
import unittest

from server.protocol import Event, parse_client_event


class ProtocolTest(unittest.TestCase):
    def test_event_serializes_protocol_envelope(self) -> None:
        event = Event(
            type="session.state",
            session_id="session-1",
            turn_id=2,
            seq=3,
            payload={"state": "LISTENING"},
        )
        data = json.loads(event.to_json())
        self.assertEqual(data["v"], 1)
        self.assertEqual(data["turn_id"], 2)
        self.assertEqual(data["payload"]["state"], "LISTENING")

    def test_rejects_unknown_client_event(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported client event"):
            parse_client_event('{"v":1,"type":"unknown"}')

    def test_rejects_unknown_protocol_version(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported protocol version"):
            parse_client_event('{"v":9,"type":"ping"}')


if __name__ == "__main__":
    unittest.main()
