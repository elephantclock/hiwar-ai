export const PROTOCOL_VERSION = 1;

export const ClientEvent = Object.freeze({
  CONFIGURE: "session.configure",
  AUDIO_COMMIT: "input_audio.commit",
  CANCEL: "response.cancel",
  PING: "ping",
});

export function clientEvent(type, payload = {}) {
  return JSON.stringify({
    v: PROTOCOL_VERSION,
    type,
    ts: Date.now(),
    payload,
  });
}

export function parseServerEvent(raw) {
  const event = JSON.parse(raw);
  if (event.v !== PROTOCOL_VERSION) {
    throw new Error("Unsupported protocol version: " + event.v);
  }
  return event;
}
