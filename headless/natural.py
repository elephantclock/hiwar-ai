"""Local voice companion for Apple Silicon: Pipecat 1.13 + MLX Whisper + Ollama + Kokoro/Orpheus.

Key behaviours solving the "feels like a bot" problem:

1. UserBackchannelFilter — Intercepts stray listening tokens AND laughter while bot
   speaks, converting them to UserBackchannelFrame.  Prevents false interruptions
   and stops "yeah" / "ha ha" from polluting the LLM context.

2. AttentiveBackchannelProcessor — Injects occasional BotBackchannelFrame ("mm-hmm",
   "right") during sustained user speech.  TextType.BACKCHANNEL means TTS speaks it
   without touching the LLM history or ending the user turn.

3. ThinkingFiller — On user-turn-stopped, immediately queues a brief filler
   ("hmm", "well", "let's see") via worker.queue_frames.  Proven to pass through
   the LLM service's catch-all push_frame, it reaches TTS in ~50 ms — filling the
   dead silence while Ollama generates its first token (200–500 ms).

4. MinWordsUserTurnStartStrategy(min_words=2) — Single-word backchannels cannot
   interrupt the bot.  Combined with UserBackchannelFilter, multi-word listening
   tokens ("oh yeah", "ha ha ha", "I see") are ALSO intercepted before they reach
   the turn controller.

5. Dual-TTS — Kokoro (ultra-fast, local) or Orpheus 3B Speech-LLM (emotional
   laughter, sighs, breath via vocal tags) behind any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import asyncio
import math
import random
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import ollama
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotBackchannelFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    OutputAudioRawFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserBackchannelFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnStoppedMessage,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.whisper.stt import MLXModel, WhisperSTTServiceMLX
from pipecat.transcriptions.language import Language
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)
from pipecat.turns.user_start.min_words_user_turn_start_strategy import MinWordsUserTurnStartStrategy
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

SYSTEM_PROMPT_PATH = Path(__file__).with_name("system_prompt.md")

# ---------------------------------------------------------------------------
# Backchannel & laughter vocabulary
# ---------------------------------------------------------------------------
# Every token Whisper might transcribe for a listening acknowledgment or laugh.
# The filter catches both exact matches AND composed laughter patterns.

BACKCHANNEL_TOKENS: set[str] = {
    # Affirmatives
    "yeah", "yes", "yep", "yup", "yea",
    # Listening acknowledgments
    "mhm", "mm-hmm", "mmhmm", "mm hmm", "mmm", "hmm",
    "uh-huh", "uhhuh", "uh huh",
    "right", "okay", "ok", "sure", "cool", "aha", "huh",
    # Multi-word acknowledgments (≥2 words — critical for MinWords guard)
    "oh yeah", "oh yes", "oh okay", "oh ok", "oh right", "oh sure",
    "oh really", "oh wow", "oh nice", "oh cool",
    "i see", "got it", "gotcha", "go on", "go ahead",
    "thank you", "thanks",
    # Fillers / hesitations Whisper may pick up
    "uh", "um", "mm", "ah", "oh",
    # Short reactions
    "wow", "nice", "great", "awesome", "true", "totally",
    "exactly", "absolutely", "definitely", "perfect", "interesting",
}

# Syllables that constitute laughter when repeated ("ha", "ha ha", "ha ha ha")
LAUGHTER_SYLLABLES: set[str] = {"ha", "haha", "heh", "hehe", "ho", "hoho", "aha"}


def _is_backchannel_or_laughter(text: str) -> bool:
    """Return True if *text* is a listening acknowledgment or laughter."""
    cleaned = text.strip().lower().rstrip(".,!?")
    if not cleaned:
        return False
    # Direct vocabulary match
    if cleaned in BACKCHANNEL_TOKENS:
        return True
    # Composed laughter: all words are laughter syllables ("ha ha ha")
    words = cleaned.split()
    if words and all(w in LAUGHTER_SYLLABLES for w in words):
        return True
    return False


@dataclass
class CompanionConfig:
    chat_model: str = "qwen2.5:14b"
    backend_model: str = "qwen2.5:32b"
    tts_engine: Literal["kokoro", "orpheus"] = "kokoro"
    tts_base_url: str = "http://localhost:8000/v1"
    tts_voice: str = "af_nicole"
    tts_speed: float = 1.0
    vad_stop_secs: float = 0.4      # Raised from 0.3 → 0.4 to tolerate natural breath pauses
    vad_confidence: float = 0.7
    min_interruption_words: int = 2
    input_device_index: int | None = None
    output_device_index: int | None = None
    play_test_tone: bool = True
    greeting: str | None = "Hi, I'm ready. Just start talking whenever you like."


# ---------------------------------------------------------------------------
# Pipeline processors
# ---------------------------------------------------------------------------

class UserBackchannelFilter(FrameProcessor):
    """Converts stray listening tokens and laughter into UserBackchannelFrame.

    Placed between STT and LLMUserAggregator.  While the bot is speaking, any
    transcription that matches a known acknowledgment or laughter pattern is
    rewritten as a UserBackchannelFrame so that:

    1. It never enters ``self._aggregation`` (no stray words in the LLM prompt).
    2. It never fires ``MinWordsUserTurnStartStrategy`` (no false interruptions).
    3. Pipecat's ``on_user_backchannel`` event fires cleanly for observers.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._bot_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        elif isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            if self._bot_speaking and _is_backchannel_or_laughter(frame.text):
                logger.debug(
                    f"UserBackchannelFilter: '{frame.text}' → UserBackchannelFrame "
                    "(bot speaking; bypassing turn aggregation)"
                )
                await self.push_frame(
                    UserBackchannelFrame(
                        text=frame.text,
                        user_id=getattr(frame, "user_id", "user"),
                        timestamp=getattr(frame, "timestamp", ""),
                    ),
                    direction,
                )
                return  # consume the original transcription frame

        await self.push_frame(frame, direction)


class AttentiveBackchannelProcessor(FrameProcessor):
    """Emits occasional BotBackchannelFrame while the user speaks at length.

    The spoken acknowledgment uses TextType.BACKCHANNEL:
    - TTS synthesises and plays it during the user's speech.
    - ``append_to_context=False`` keeps it out of the LLM history.
    - ``interruptible=False`` means ambient noise won't cancel it.
    """

    def __init__(
        self,
        min_words_for_backchannel: int = 14,
        min_speech_duration: float = 3.5,
        cooldown_secs: float = 9.0,
        acknowledgments: list[str] | None = None,
        probability: float = 0.60,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._min_words = min_words_for_backchannel
        self._min_duration = min_speech_duration
        self._cooldown = cooldown_secs
        self._probability = probability
        self._acknowledgments = acknowledgments or ["mm-hmm", "uh-huh", "right", "hmm"]
        self._speech_start_time: float | None = None
        self._last_backchannel_time: float = 0.0
        self._accumulated_words = 0
        self._bot_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        now = time.monotonic()
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._speech_start_time = None
            self._accumulated_words = 0
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            if not self._bot_speaking and self._speech_start_time is None:
                self._speech_start_time = now
        elif isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            if not self._bot_speaking:
                self._accumulated_words = max(self._accumulated_words, len(frame.text.split()))
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if not self._bot_speaking and self._speech_start_time is not None:
                duration = now - self._speech_start_time
                if (
                    duration >= self._min_duration
                    and self._accumulated_words >= self._min_words
                    and (now - self._last_backchannel_time) >= self._cooldown
                    and random.random() < self._probability
                ):
                    ack = random.choice(self._acknowledgments)
                    self._last_backchannel_time = now
                    logger.debug(f"Attentive backchannel → '{ack}'")
                    await self.push_frame(BotBackchannelFrame(text=ack), FrameDirection.DOWNSTREAM)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._speech_start_time = None
            self._accumulated_words = 0

        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Thinking filler phrases (queued via worker.queue_frames)
# ---------------------------------------------------------------------------

# Short fillers — spoken in ~200 ms, filling the dead silence while Ollama
# generates its first token.
SHORT_FILLERS: list[str] = ["hmm", "mm", "well"]

# Medium fillers — for longer, more complex user utterances.
MEDIUM_FILLERS: list[str] = ["hmm, let me think", "let's see", "well"]


def _pick_filler(user_word_count: int) -> str | None:
    """Decide whether and what filler to emit based on utterance length.

    Returns a filler string or None (meaning: respond immediately with no filler).
    """
    if user_word_count < 4:
        # Short question → quick response feels natural, no filler
        return None
    elif user_word_count < 12:
        # Medium question → 30% chance of a short filler
        return random.choice(SHORT_FILLERS) if random.random() < 0.30 else None
    else:
        # Long question → 55% chance of a medium filler
        return random.choice(MEDIUM_FILLERS) if random.random() < 0.55 else None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def list_audio_devices() -> list[dict[str, Any]]:
    """List all available audio devices with PyAudio indexes."""
    import pyaudio

    p = pyaudio.PyAudio()
    devices = []
    try:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            devices.append(
                {
                    "index": i,
                    "name": info["name"],
                    "inputs": info["maxInputChannels"],
                    "outputs": info["maxOutputChannels"],
                    "sample_rate": int(info["defaultSampleRate"]),
                }
            )
    finally:
        p.terminate()
    return devices


def play_test_tone(device_index: int | None) -> None:
    """Plays a brief 440 Hz test tone to confirm output device functionality."""
    import pyaudio

    rate = 24000
    total_frames = int(rate * 0.5)
    data = b"".join(
        struct.pack("<h", int(6000 * math.sin(2 * math.pi * 440 * n / rate)))
        for n in range(total_frames)
    )
    p = pyaudio.PyAudio()
    try:
        info = (
            p.get_device_info_by_index(device_index)
            if device_index is not None
            else p.get_default_output_device_info()
        )
        logger.info(f"Test tone on output: {info['name']} (index {info['index']})")
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, output=True, output_device_index=device_index)
        stream.write(data)
        stream.stop_stream()
        stream.close()
    except Exception as e:
        logger.warning(f"Test tone skipped: {e}")
    finally:
        p.terminate()


async def create_backend_handler(backend_model: str):
    """Returns a delegation handler bound to the specified backend model."""

    async def ask_backend(params: FunctionCallParams):
        task = params.arguments.get("task", "")
        logger.info(f"Delegating to backend ({backend_model}): {task[:80]}...")

        client = ollama.AsyncClient()
        resp = await client.chat(
            model=backend_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise backend reasoning agent. "
                        "Answer accurately and concisely. "
                        "If unsure, say so without hallucinating."
                    ),
                },
                {"role": "user", "content": task},
            ],
        )
        await params.result_callback({"answer": resp["message"]["content"]})

    return ask_backend


def create_tts_service(config: CompanionConfig) -> FrameProcessor:
    """Instantiates the selected TTS service."""
    if config.tts_engine == "orpheus":
        logger.info(f"TTS: Orpheus via {config.tts_base_url} (voice: {config.tts_voice})")
        return OpenAITTSService(
            base_url=config.tts_base_url,
            api_key="local",
            settings=OpenAITTSService.Settings(
                voice=config.tts_voice,
                model="orpheus-3b",
                speed=config.tts_speed,
            ),
        )
    logger.info(f"TTS: Kokoro (voice: {config.tts_voice}, speed: {config.tts_speed}x)")
    return KokoroTTSService(
        settings=KokoroTTSService.Settings(
            voice=config.tts_voice,
            speed=config.tts_speed,
        )
    )


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

async def build_companion_pipeline(config: CompanionConfig) -> tuple[PipelineWorker, WorkerRunner]:
    """Builds the complete speech companion pipeline."""
    system_prompt = (
        SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        if SYSTEM_PROMPT_PATH.exists()
        else "You are a warm, intelligent English conversational companion."
    )

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            input_device_index=config.input_device_index,
            output_device_index=config.output_device_index,
        )
    )

    stt = WhisperSTTServiceMLX(
        settings=WhisperSTTServiceMLX.Settings(
            model=MLXModel.LARGE_V3_TURBO_Q4,
            language=Language.EN,
        )
    )

    llm = OLLamaLLMService(
        settings=OLLamaLLMService.Settings(
            model=config.chat_model,
            system_instruction=system_prompt,
        )
    )

    tts = create_tts_service(config)

    backend_handler = await create_backend_handler(config.backend_model)
    tools = ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="ask_backend",
                description=(
                    "Delegate complex reasoning, calculations, deep research, "
                    "or precise factual lookups to the backend agent."
                ),
                properties={"task": {"type": "string", "description": "The precise task or question to solve."}},
                required=["task"],
                handler=backend_handler,
            )
        ]
    )

    context = LLMContext(tools=tools)

    smart_turn = LocalSmartTurnAnalyzerV3()
    user_turn_strategies = UserTurnStrategies(
        start=[
            MinWordsUserTurnStartStrategy(min_words=config.min_interruption_words),
        ],
        stop=[
            TurnAnalyzerUserTurnStopStrategy(turn_analyzer=smart_turn),
        ],
    )

    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=config.vad_stop_secs,
                    confidence=config.vad_confidence,
                )
            ),
            user_turn_strategies=user_turn_strategies,
        ),
    )

    user_backchannel_filter = UserBackchannelFilter()
    attentive_backchannel = AttentiveBackchannelProcessor()

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_backchannel_filter,
            attentive_backchannel,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    worker = PipelineWorker(pipeline, params=PipelineParams(), idle_timeout_secs=None)

    # -----------------------------------------------------------------------
    # Thinking filler: fill the dead silence while Ollama generates its first
    # token.  Injected via worker.queue_frames so the BotBackchannelFrame
    # traverses the pipeline and reaches TTS immediately (~50 ms) while the
    # LLM is still processing (~200–500 ms TTFT).
    # -----------------------------------------------------------------------
    user_agg = aggregators.user()

    @user_agg.event_handler("on_user_turn_stopped")
    async def _on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
        if not message.content:
            return  # empty turn — no filler needed
        word_count = len(message.content.split())
        filler_text = _pick_filler(word_count)
        if filler_text:
            logger.debug(f"Thinking filler → '{filler_text}' (user said {word_count} words)")
            await worker.queue_frames([BotBackchannelFrame(text=filler_text)])

    if config.greeting:
        await worker.queue_frames([TTSSpeakFrame(config.greeting)])

    if config.play_test_tone:
        await asyncio.to_thread(play_test_tone, config.output_device_index)

    runner = WorkerRunner()
    await runner.add_workers(worker)
    return worker, runner


async def run_companion(config: CompanionConfig | None = None) -> None:
    """Runs the voice companion until interrupted."""
    cfg = config or CompanionConfig()
    worker, runner = await build_companion_pipeline(cfg)
    logger.info("Voice companion ready.")
    await runner.run()


if __name__ == "__main__":
    asyncio.run(run_companion())