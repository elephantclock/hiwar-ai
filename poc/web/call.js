import { PlaybackQueue } from "./playback-queue.js";
import { ClientEvent, clientEvent, parseServerEvent } from "./protocol.js";

const elements = {
  connect: document.querySelector("#connect"),
  record: document.querySelector("#record"),
  interrupt: document.querySelector("#interrupt"),
  status: document.querySelector("#status"),
  transcript: document.querySelector("#transcript"),
  events: document.querySelector("#events"),
};

let session;
let socket;
let stream;
let audioContext;
let captureNode;
let playback;
let recording = false;
let currentTurn = 0;

function setStatus(text, tone = "idle") {
  elements.status.textContent = text;
  elements.status.dataset.tone = tone;
}

function logEvent(event) {
  const item = document.createElement("li");
  item.textContent = (event.seq ?? "–") + " · " + event.type;
  elements.events.prepend(item);
  while (elements.events.children.length > 8) {
    elements.events.lastElementChild.remove();
  }
}

async function createSession() {
  const response = await fetch("/api/v1/sessions", { method: "POST" });
  if (!response.ok) throw new Error("Session creation failed (" + response.status + ")");
  return response.json();
}

function websocketUrl(path) {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  return scheme + "//" + location.host + path;
}

async function connect() {
  elements.connect.disabled = true;
  setStatus("Connecting…", "working");
  try {
    session = await createSession();
    socket = new WebSocket(websocketUrl(session.stream_url));
    socket.binaryType = "arraybuffer";
    socket.onmessage = onMessage;
    socket.onclose = () => {
      setStatus("Disconnected", "error");
      elements.record.disabled = true;
      elements.interrupt.disabled = true;
    };
    socket.onerror = () => setStatus("Connection error", "error");
    socket.onopen = async () => {
      socket.send(clientEvent(ClientEvent.CONFIGURE, { language: "en" }));
      await prepareAudio();
      setStatus("Ready to listen", "ready");
      elements.record.disabled = false;
      elements.interrupt.disabled = false;
    };
  } catch (error) {
    setStatus(error.message, "error");
    elements.connect.disabled = false;
  }
}

async function prepareAudio() {
  audioContext = new AudioContext();
  await audioContext.audioWorklet.addModule("./audio-worklet.js");
  stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  const source = audioContext.createMediaStreamSource(stream);
  captureNode = new AudioWorkletNode(audioContext, "pcm-capture");
  const silent = audioContext.createGain();
  silent.gain.value = 0;
  source.connect(captureNode).connect(silent).connect(audioContext.destination);
  captureNode.port.onmessage = ({ data }) => {
    if (recording && socket?.readyState === WebSocket.OPEN && data.type === "pcm") {
      socket.send(data.pcm);
    }
  };
  playback = new PlaybackQueue(audioContext);
}

function onMessage({ data }) {
  if (data instanceof ArrayBuffer) {
    playback?.enqueue(data, currentTurn);
    return;
  }

  const event = parseServerEvent(data);
  logEvent(event);
  currentTurn = Math.max(currentTurn, event.turn_id ?? 0);

  if (event.type === "session.state") {
    const label = event.payload.state.toLowerCase().replaceAll("_", " ");
    setStatus(label, event.payload.state === "LISTENING" ? "ready" : "working");
  } else if (event.type === "transcript.final") {
    elements.transcript.textContent = event.payload.text;
  } else if (event.type === "response.text.delta") {
    elements.transcript.textContent += event.payload.delta;
  } else if (event.type === "error") {
    const detail = event.payload.message || event.payload.code;
    elements.transcript.textContent = detail;
  }
}

function toggleRecording() {
  if (!recording) {
    recording = true;
    elements.record.textContent = "Finish speaking";
    elements.record.dataset.active = "true";
    elements.transcript.textContent = "Listening…";
    setStatus("Listening", "live");
    return;
  }

  recording = false;
  elements.record.textContent = "Start speaking";
  elements.record.dataset.active = "false";
  socket.send(clientEvent(ClientEvent.AUDIO_COMMIT));
  setStatus("Finishing turn…", "working");
}

function interrupt() {
  recording = false;
  elements.record.textContent = "Start speaking";
  elements.record.dataset.active = "false";
  currentTurn += 1;
  playback?.cancel(currentTurn);
  socket.send(clientEvent(ClientEvent.CANCEL));
  setStatus("Interrupted", "ready");
}

elements.connect.addEventListener("click", connect);
elements.record.addEventListener("click", toggleRecording);
elements.interrupt.addEventListener("click", interrupt);
