"""Local voice companion for Apple Silicon: Pipecat 1.13 + MLX Whisper + Ollama + Kokoro.

Setup (uv):
    brew install portaudio
    uv add "pipecat-ai[silero,whisper,mlx-whisper,ollama,kokoro,local]" ollama
    ollama pull qwen2.5:14b          # conversational model
    ollama pull qwen2.5:32b          # backend model (or llama3.3:70b)
    ollama run qwen2.5:14b ""        # preload so the first reply isn't slow
    uv run bot.py

Use headphones: the local transport has no echo cancellation.
"""

import asyncio
import math
import random
import re
import struct
import time
import wave
from collections.abc import AsyncGenerator
from pathlib import Path

import numpy as np
import ollama
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotBackchannelFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    Frame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    UserBackchannelFrame,
    UserStartedSpeakingFrame,
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
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.whisper.stt import MLXModel, WhisperSTTServiceMLX
from pipecat.transcriptions.language import Language
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)
from pipecat.turns.empty_user_turn import EmptyUserTurnConfig
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.types import assert_given, require_given
from pipecat.workers.runner import WorkerRunner

CHAT_MODEL = "qwen2.5:14b"
BACKEND_MODEL = "qwen2.5:32b"
KOKORO_VOICE = "af_nicole"  # any voice name that exists in the Kokoro voices file (see voice_audition.py)
KOKORO_SPEED = 1.0  # try 0.9 to 0.95 for a calmer, more conversational pace
VAD_STOP_SECS = 0.2  # recommended with Smart Turn v3, which decides when you're done talking

# Audio devices (PyAudio indexes). None = system default.
# List them with: python -c "import pyaudio; p=pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]"
INPUT_DEVICE_INDEX: int | None = None
OUTPUT_DEVICE_INDEX: int | None = 1

# Startup checks: a 1-second beep on the chosen output device (and its name in the log),
# then the bot says a greeting out loud. Set to False/None once audio works.
PLAY_TEST_TONE = True
GREETING: str | None = "Hi, I'm ready. Just start talking whenever you like."

# Conversation style at startup: "natural", "fun" or "formal". You can also switch by voice
# ("be more formal", "let's have some fun", "go back to normal").
CONVERSATION_STYLE = "natural"

# Optional real recordings for the sound tags. Put 16-bit WAV files in a "sounds" folder next to
# this script, named chuckle.wav, laugh.wav, sigh.wav, hum.wav (extra takes like laugh2.wav are
# picked at random). Without a file, the bot speaks an approximation instead ("Heh.", "Ha ha ha.").
SOUND_DIR = Path(__file__).with_name("sounds")
SOUND_GAIN = 0.8

# Silence between sentences. Each style sets a base length (see STYLES); this scales all of them,
# so 1.0 = as defined, 1.5 = more thoughtful gaps, 0.5 = brisker. The gap also depends on how the
# sentence ends: longer after "..." and "?", shorter after "!" and a comma.
PAUSE_SCALE = 1.0

# Interruption and backchannel behaviour.
# While the bot is talking, you must keep speaking this long before it counts as an interruption.
# Shorter sounds ("mm-hmm", "yeah", a laugh) are treated as listening noises and the bot carries on.
INTERRUPT_MIN_SECS = 0.8
# The bot may drop in a quiet "mm-hmm" when you pause mid-thought, but only after you have been
# talking this long, no more often than the cooldown, and only some of the time.
BACKCHANNELS_ENABLED = True
BACKCHANNEL_MIN_TURN_SECS = 4.0
BACKCHANNEL_COOLDOWN_SECS = 12.0
BACKCHANNEL_PROBABILITY = 0.6
BACKCHANNEL_PHRASES = ["Mm-hmm.", "Mhm.", "Uh-huh.", "Mm.", "Right.", "Mmm.", "I see."]
# Speed multipliers (on the current base pace) for short acknowledgments the bot speaks, such as
# "Mm-hmm" or "Right", and for laughter syllables such as "Ha ha" and "Heh". Both are spoken quickly,
# the way people actually say them.
BACKCHANNEL_SPEED = 1.25
LAUGHTER_SPEED = 1.45

# Spoken (and kept out of the conversation history) when the bot hands a task to the backend model.
HOLD_LINES = [
    "Let me check on that.",
    "Hmm, give me a second.",
    "One moment, let me look into that.",
]

SYSTEM_PROMPT = Path(__file__).with_name("system_prompt.md").read_text()


_BACKCHANNEL_VOCAB = {
    "mm", "mmm", "hmm", "hm", "mhm", "mmhmm", "uh", "huh", "um", "yeah", "yes", "yep", "yup",
    "right", "ok", "okay", "sure", "oh", "ah", "aha", "ha", "haha", "hah", "heh", "wow", "cool",
    "nice", "true", "exactly", "indeed", "i", "see", "got", "it", "interesting", "really", "gotcha",
    "hehe", "lol",
}  # fmt: skip
_LAUGHTER_VOCAB = {"ha", "haha", "hah", "heh", "hehe", "lol"}


def _words(text: str) -> list[str]:
    return re.sub(r"[^a-z ]", "", text.lower().replace("-", " ")).split()


def is_backchannel(text: str) -> bool:
    """True for a short listening noise like "mm-hmm", "yeah", "uh huh", "I see", "haha"."""
    words = _words(text)
    return 0 < len(words) <= 4 and all(w in _BACKCHANNEL_VOCAB for w in words)


def is_laughter(text: str) -> bool:
    """True for laughter syllables only, like "Ha ha ha!", "Heh.", "Hehe"."""
    words = _words(text)
    return 0 < len(words) <= 8 and all(w in _LAUGHTER_VOCAB for w in words)


def quick_speed(text: str, base: float) -> float | None:
    """Fixed fast pace for laughter and short acknowledgments; None for ordinary sentences."""
    if is_laughter(text):
        return min(1.6, base * LAUGHTER_SPEED)
    if is_backchannel(text):
        return min(1.5, base * BACKCHANNEL_SPEED)
    return None


class PatientInterruptionStrategy(BaseUserTurnStartStrategy):
    """Starts a user turn on speech, but while the bot talks, only once the speech lasts long enough.

    A quick "mm-hmm" or "yeah" ends before the timer does, so the bot is not interrupted.
    Real interruptions still stop it, after `min_interrupt_secs`.
    """

    def __init__(self, *, min_interrupt_secs: float = 0.8, **kwargs):
        super().__init__(**kwargs)
        self._min_interrupt_secs = min_interrupt_secs
        self._bot_speaking = False
        self._user_speaking = False
        self._timer: asyncio.Task | None = None

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            if self._user_speaking:  # the user kept talking past the end of the bot's speech
                self._cancel_timer()
                await self.trigger_user_turn_started()
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._user_speaking = True
            self._cancel_timer()
            if self._bot_speaking:
                self._timer = asyncio.create_task(self._interrupt_if_still_speaking())
            else:
                await self.trigger_user_turn_started()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_speaking = False
            self._cancel_timer()  # too short to be an interruption
        return ProcessFrameResult.CONTINUE

    async def _interrupt_if_still_speaking(self):
        await asyncio.sleep(self._min_interrupt_secs)
        self._timer = None
        if self._user_speaking:
            await self.trigger_user_turn_started()

    def _cancel_timer(self):
        if self._timer and not self._timer.done():
            self._timer.cancel()
        self._timer = None

    async def cleanup(self):
        self._cancel_timer()
        await super().cleanup()


class PaceState:
    """How fast the user talks, as a multiplier on the bot's pace (1.0 = typical).

    Written by UserPaceTracker, read by the TTS, so the bot gently matches your speaking speed.
    """

    def __init__(self) -> None:
        self.factor = 1.0


class UserPaceTracker(FrameProcessor):
    """Measures how fast you speak (words per second of speech) and nudges the bot's pace to match.

    A quick talker gets a slightly quicker bot, a slow reflective one a slower bot. Smoothed over
    several utterances and kept within about -12% / +15%.
    """

    TYPICAL_WPS = 2.5

    def __init__(self, pace: PaceState, **kwargs):
        super().__init__(**kwargs)
        self._pace = pace
        self._burst_started: float | None = None
        self._burst_secs = 0.0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._burst_started = time.monotonic()
        elif isinstance(frame, VADUserStoppedSpeakingFrame) and self._burst_started is not None:
            self._burst_secs = time.monotonic() - self._burst_started
            self._burst_started = None
        elif isinstance(frame, TranscriptionFrame):
            words = len(frame.text.split())
            if words >= 3 and self._burst_secs >= 0.8:
                wps = words / self._burst_secs
                target = 1.0 + (wps - self.TYPICAL_WPS) * 0.125
                target = max(0.88, min(1.15, target))
                self._pace.factor = 0.5 * self._pace.factor + 0.5 * target
                logger.info(
                    f"You spoke {wps:.1f} words/s, so the bot's pace factor is "
                    f"{self._pace.factor:.2f}"
                )
            self._burst_secs = 0.0
        await self.push_frame(frame, direction)


class BackchannelFilter(FrameProcessor):
    """Keeps the user's "mm-hmm" / "yeah" out of the conversation while the bot is talking.

    Such a transcript becomes a UserBackchannelFrame (reported, but never sent to the LLM), so it
    can't start a turn or get glued onto your next message.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._bot_speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False

        if (
            isinstance(frame, TranscriptionFrame)
            and self._bot_speaking
            and is_backchannel(frame.text)
        ):
            logger.info(f"Listening noise while the bot talks, ignored: {frame.text!r}")
            await self.push_frame(
                UserBackchannelFrame(
                    text=frame.text, user_id=frame.user_id, timestamp=frame.timestamp
                ),
                direction,
            )
            return
        await self.push_frame(frame, direction)


class BackchannelInjector(FrameProcessor):
    """Says a quiet "mm-hmm" when you pause mid-thought during a long turn.

    It only fires if you are still mid-turn shortly after the pause (so not when you've finished and
    the bot is about to answer), and never while you or the bot are speaking. Uses Pipecat's
    BotBackchannelFrame: spoken like normal, not saved in the history, and not cut by interruptions.
    """

    def __init__(
        self,
        *,
        min_turn_secs: float = 4.0,
        cooldown_secs: float = 12.0,
        probability: float = 0.6,
        delay_secs: float = 0.6,
        phrases: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._min_turn_secs = min_turn_secs
        self._cooldown_secs = cooldown_secs
        self._probability = probability
        self._delay_secs = delay_secs
        self._phrases = phrases or ["Mm-hmm."]
        self._in_turn = False
        self._turn_started = 0.0
        self._last_backchannel = 0.0
        self._user_speaking = False
        self._bot_speaking = False
        self._pending: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStartedSpeakingFrame):
            self._in_turn = True
            self._turn_started = time.monotonic()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._in_turn = False
            self._cancel_pending()
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._user_speaking = True
            self._cancel_pending()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_speaking = False
            if self._eligible():
                self._pending = asyncio.create_task(self._speak_after_delay())
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        await self.push_frame(frame, direction)

    def _eligible(self) -> bool:
        now = time.monotonic()
        return (
            self._in_turn
            and not self._bot_speaking
            and now - self._turn_started >= self._min_turn_secs
            and now - self._last_backchannel >= self._cooldown_secs
            and random.random() < self._probability
        )

    async def _speak_after_delay(self):
        await asyncio.sleep(self._delay_secs)
        self._pending = None
        if self._in_turn and not self._user_speaking and not self._bot_speaking:
            self._last_backchannel = time.monotonic()
            await self.push_frame(BotBackchannelFrame(text=random.choice(self._phrases)))

    def _cancel_pending(self):
        if self._pending and not self._pending.done():
            self._pending.cancel()
        self._pending = None

    async def cleanup(self):
        self._cancel_pending()
        await super().cleanup()


# ---------------------------------------------------------------------------------------------
# Conversation styles
# ---------------------------------------------------------------------------------------------
# speed: multiplier on KOKORO_SPEED. variation: how much the pace moves from sentence to sentence
# (1.0 = as defined, higher = more animated, lower = steadier). sentence_pause: base seconds of
# silence between spoken sentences. clause_pause: pause at commas inside a sentence.
# sounds: whether [chuckle]/[laugh]/[sigh]/[hum] are performed (formal drops them).
STYLES: dict[str, dict] = {
    "natural": {
        "speed": 1.0,
        "variation": 1.0,
        "sentence_pause": 0.40,
        "clause_pause": 0.12,
        "sounds": True,
        "prompt": (
            "Current style: natural. Warm, relaxed and conversational, like a thoughtful friend. "
            "Use a light [chuckle] or a soft hesitation only now and then, when it truly fits."
        ),
    },
    "fun": {
        "speed": 1.08,
        "variation": 1.3,
        "sentence_pause": 0.28,
        "clause_pause": 0.08,
        "sounds": True,
        "prompt": (
            "Current style: fun. Playful, upbeat and quick-witted, like a funny friend. "
            "Use light humor, vivid and cheeky phrasing, short punchy sentences and lively "
            "reactions. Laugh at funny moments with [laugh] or [chuckle], and use [hum] or "
            "'ooh' and 'ha' naturally. At most two sounds per reply. Stay kind, and never joke "
            "when the person seems sad, stressed or is asking something serious."
        ),
    },
    "formal": {
        "speed": 0.94,
        "variation": 0.5,
        "sentence_pause": 0.55,
        "clause_pause": 0.18,
        "sounds": False,
        "prompt": (
            "Current style: formal. Courteous, precise and professional. Use complete sentences, "
            "no slang, avoid contractions, no jokes, no filler words or hesitations, and never "
            "use sound tags. Address the person respectfully and stay concise."
        ),
    },
}


def build_instruction(style: str) -> str:
    return f"{SYSTEM_PROMPT}\n\n{STYLES[style]['prompt']}"


# ---------------------------------------------------------------------------------------------
# Expressive speech: sound tags, dynamic speed
# ---------------------------------------------------------------------------------------------
SOUND_FALLBACK_TEXT = {
    "chuckle": "Heh.",
    "laugh": "Ha ha ha!",
    "sigh": "Ahh...",
    "hum": "Hmm, mm-hmm.",
}
_SOUND_ALIASES = {
    "chuckle": "chuckle", "chuckles": "chuckle", "giggle": "chuckle", "giggles": "chuckle",
    "laugh": "laugh", "laughs": "laugh", "laughing": "laugh",
    "sigh": "sigh", "sighs": "sigh", "sighing": "sigh",
    "hum": "hum", "hums": "hum", "humming": "hum",
}  # fmt: skip
# Models sometimes write "*chuckles*", "(sighs)" or "<laugh>" instead of the exact tag.
_SOUND_ALIAS_RE = re.compile(
    r"[\[\(\*<]\s*(" + "|".join(_SOUND_ALIASES) + r")\s*[\]\)\*>]", re.IGNORECASE
)
_SOUND_SPLIT_RE = re.compile(r"(\[(?:chuckle|laugh|sigh|hum)\])")
_SOUND_TAG_RE = re.compile(r"\[(chuckle|laugh|sigh|hum)\]")


def normalize_sound_tags(text: str) -> str:
    """Canonical [tag] form, and drop stray stage directions like *smiles* or <pause>."""
    text = _SOUND_ALIAS_RE.sub(lambda m: f"[{_SOUND_ALIASES[m.group(1).lower()]}]", text)
    text = re.sub(r"\*[^*\n]{1,30}\*", "", text)
    return re.sub(r"<[^>\n]{1,20}>", "", text)


# The model can also steer pace itself by writing one of these at the start of a sentence.
PACE_TAGS = {"slow": 0.78, "calm": 0.88, "excited": 1.15, "fast": 1.25}
_PACE_RE = re.compile(r"\[(" + "|".join(PACE_TAGS) + r")\]", re.IGNORECASE)
_SLOW_WORDS = re.compile(
    r"\b(sorry|unfortunately|sadly|regret|difficult|hard|loss|careful|serious|important|please)\b",
    re.IGNORECASE,
)
_LIVELY_WORDS = re.compile(
    r"\b(wow|amazing|great|awesome|fantastic|love|fun|exciting|brilliant|congratulations)\b",
    re.IGNORECASE,
)
_HESITATION = re.compile(r"\b(hmm+|um+|uh+|well|let me think)\b", re.IGNORECASE)


def extract_pace(text: str) -> tuple[str, float]:
    """Pull a [slow]/[calm]/[excited]/[fast] tag out of a sentence; returns (clean text, factor)."""
    found = _PACE_RE.search(text)
    factor = PACE_TAGS[found.group(1).lower()] if found else 1.0
    return _PACE_RE.sub("", text), factor


def sentence_speed(text: str, base: float, pace: float = 1.0, variation: float = 1.0) -> float:
    """Speed for one sentence. Wide enough to hear: roughly 0.7x to 1.4x of the base.

    Slower for questions, trailing thoughts, hesitation, serious or apologetic wording, very short
    remarks and digits (clarity). Quicker for exclamations, lively wording and long runs of words.
    `pace` is the model's own tag; `variation` scales how far the rules may move the speed.
    """
    t = text.strip()
    words = len(t.split())
    factor = 1.0
    if t.endswith("?"):
        factor *= 0.90
    elif t.endswith("!"):
        factor *= 1.14
    if "..." in t or "…" in t or _HESITATION.search(t):
        factor *= 0.80
    if _SLOW_WORDS.search(t):
        factor *= 0.88
    if _LIVELY_WORDS.search(t):
        factor *= 1.12
    if words <= 3:
        factor *= 0.88
    elif words >= 18:
        factor *= 1.10
    if re.search(r"\d", t):
        factor *= 0.92
    # A lively style exaggerates the speed-ups only (slow-downs stay as they are); a steady style
    # (variation < 1) softens both.
    exponent = variation if (factor > 1 or variation < 1) else 1.0
    return max(0.6, min(1.45, base * (factor**exponent) * pace))


class ExpressiveKokoroTTSService(KokoroTTSService):
    """Kokoro with [chuckle]/[laugh]/[sigh]/[hum] sound tags and per-sentence pace.

    A tag plays a real clip from the sounds folder when there is one, otherwise a spoken
    approximation. `style` (one of STYLES) sets pace, pauses and whether sounds are performed,
    and can be swapped at any time.
    """

    def __init__(self, *, style: dict, pace: PaceState | None = None, **kwargs):
        super().__init__(**kwargs)
        self.style = style
        self.pace = pace or PaceState()
        self._clips: dict[str, list[bytes]] = {}
        # Silence owed before the next sentence of the same response (none after the last one).
        self._pending_gap: tuple[str, float] | None = None

    def _pause_after(self, text: str) -> float:
        """Seconds of silence after a sentence: longer after '...' and '?', shorter after '!'."""
        t = text.strip()
        if t.endswith(("...", "…")):
            factor = 1.6
        elif t.endswith("?"):
            factor = 1.1
        elif t.endswith("!"):
            factor = 0.9
        elif t.endswith((",", ";", ":", "-", "—")):
            factor = 0.5
        else:
            factor = 1.0
        # A little jitter so the rhythm isn't mechanical.
        return self.style["sentence_pause"] * PAUSE_SCALE * factor * random.uniform(0.85, 1.2)

    def _silence(self, seconds: float, context_id: str) -> TTSAudioRawFrame:
        samples = int(self.sample_rate * seconds)
        return TTSAudioRawFrame(
            audio=b"\x00\x00" * samples,
            sample_rate=self.sample_rate,
            num_channels=1,
            context_id=context_id,
        )

    def _clips_for(self, tag: str) -> list[bytes]:
        if tag not in self._clips:
            clips: list[bytes] = []
            for path in sorted(SOUND_DIR.glob(f"{tag}*.wav")):
                try:
                    clips.append(self._load_clip(path))
                except Exception as e:
                    logger.warning(f"Could not use {path.name} (need 16-bit PCM WAV): {e}")
            self._clips[tag] = clips
        return self._clips[tag]

    def _load_clip(self, path: Path) -> bytes:
        with wave.open(str(path), "rb") as w:
            if w.getsampwidth() != 2:
                raise ValueError("not 16-bit")
            channels, rate = w.getnchannels(), w.getframerate()
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
        if channels > 1:
            audio = audio[: len(audio) // channels * channels].reshape(-1, channels).mean(axis=1)
        if rate != self.sample_rate and len(audio) > 1:
            n = int(len(audio) * self.sample_rate / rate)
            audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
        return np.clip(audio * SOUND_GAIN, -32768, 32767).astype(np.int16).tobytes()

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        text, pace = extract_pace(normalize_sound_tags(text))
        for part in _SOUND_SPLIT_RE.split(text):
            if not part.strip():
                continue
            tag = _SOUND_TAG_RE.fullmatch(part.strip())
            if tag and not self.style["sounds"]:
                continue

            # Silence between this and the previous sentence of the same response.
            gap, self._pending_gap = self._pending_gap, None
            if gap and gap[0] == context_id:
                yield self._silence(gap[1], context_id)

            quick_ok = True
            if tag:
                clips = self._clips_for(tag.group(1))
                if clips:
                    yield TTSAudioRawFrame(
                        audio=random.choice(clips),
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    )
                    self._pending_gap = (context_id, 0.15)
                    continue
                part = SOUND_FALLBACK_TEXT[tag.group(1)]
                quick_ok = tag.group(1) in ("laugh", "chuckle")  # a sigh or hum stays unhurried
            async for frame in self._speak(part, context_id, pace, quick_ok):
                yield frame
            self._pending_gap = (context_id, self._pause_after(part))

    async def _speak(
        self, text: str, context_id: str, pace: float = 1.0, quick_ok: bool = True
    ) -> AsyncGenerator[Frame, None]:
        try:
            await self.start_tts_usage_metrics(text)
            voice = require_given(self._settings.voice, "Kokoro TTS voice")
            lang = require_given(self._settings.language, "Kokoro TTS language")
            base = assert_given(self._settings.speed) * self.style["speed"] * self.pace.factor
            # Laughter and short acknowledgments are spoken quickly, unless the model explicitly
            # tagged the pace.
            quick = quick_speed(text, base) if (quick_ok and pace == 1.0) else None
            speed = quick or sentence_speed(text, base, pace, self.style.get("variation", 1.0))
            logger.info(
                f"Speaking at {speed:.2f}x (base {base:.2f}, pace tag {pace:.2f}): "
                f"{text.strip()[:60]!r}"
            )
            stream = self._kokoro.create_stream(
                text,
                voice=voice,
                lang=lang,
                speed=speed,
                sentence_pause=self.style["sentence_pause"],
                clause_pause=self.style["clause_pause"],
            )
            async for samples, sample_rate in stream:
                await self.stop_ttfb_metrics()
                audio = await self._resampler.resample(
                    (samples * 32767).astype(np.int16).tobytes(), sample_rate, self.sample_rate
                )
                yield TTSAudioRawFrame(
                    audio=audio, sample_rate=self.sample_rate, num_channels=1, context_id=context_id
                )
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            await self.stop_ttfb_metrics()


def play_test_tone(device_index: int | None) -> None:
    """Beep on the output device so you can hear that the device itself works."""
    import pyaudio

    rate = 24000
    data = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * n / rate)))
        for n in range(rate)
    )
    p = pyaudio.PyAudio()
    try:
        info = (
            p.get_device_info_by_index(device_index)
            if device_index is not None
            else p.get_default_output_device_info()
        )
        logger.info(f"Test tone on output device: {info['name']} (index {info['index']})")
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=rate,
            output=True,
            output_device_index=device_index,
        )
        stream.write(data)
        stream.stop_stream()
        stream.close()
    except Exception as e:
        logger.error(f"Test tone failed: {e}")
    finally:
        p.terminate()


async def ask_backend(params: FunctionCallParams):
    """Delegation tool: hand hard tasks to the bigger model."""
    task = params.arguments["task"]
    client = ollama.AsyncClient()
    resp = await client.chat(
        model=BACKEND_MODEL,
        messages=[
            {"role": "system", "content": "Answer accurately and concisely. If unsure, say so."},
            {"role": "user", "content": task},
        ],
    )
    await params.result_callback({"answer": resp["message"]["content"]})


async def main():
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            input_device_index=INPUT_DEVICE_INDEX,
            output_device_index=OUTPUT_DEVICE_INDEX,
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
            model=CHAT_MODEL,
            system_instruction=build_instruction(CONVERSATION_STYLE),
        )
    )
    pace = PaceState()
    tts = ExpressiveKokoroTTSService(
        style=STYLES[CONVERSATION_STYLE],
        pace=pace,
        settings=KokoroTTSService.Settings(voice=KOKORO_VOICE, speed=KOKORO_SPEED),
    )

    async def set_conversation_style(params: FunctionCallParams):
        name = str(params.arguments.get("style", "")).strip().lower()
        if name not in STYLES:
            await params.result_callback({"error": f"Unknown style. Choose one of: {', '.join(STYLES)}."})
            return
        tts.style = STYLES[name]
        await llm._update_settings(
            OLLamaLLMService.Settings(system_instruction=build_instruction(name))
        )
        logger.info(f"Conversation style is now: {name}")
        await params.result_callback(
            {"style": name, "note": "Style changed. Briefly acknowledge it in the new style."}
        )

    # The handler travels with the schema, so no register_function call is needed.
    tools = ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="ask_backend",
                description="Delegate research, reasoning, calculation, or lookups to the backend agent.",
                properties={"task": {"type": "string", "description": "What to work out."}},
                required=["task"],
                handler=ask_backend,
            ),
            FunctionSchema(
                name="set_conversation_style",
                description=(
                    "Change how you talk. Call it only when the person asks to be more formal, "
                    "more fun or playful, or to go back to normal or natural."
                ),
                properties={
                    "style": {
                        "type": "string",
                        "enum": list(STYLES),
                        "description": "formal, fun, or natural.",
                    }
                },
                required=["style"],
                handler=set_conversation_style,
            ),
        ]
    )

    context = LLMContext(tools=tools)  # system prompt lives on the LLM service
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS)),
            # Start strategy: a short "mm-hmm" doesn't interrupt the bot, real speech does.
            # The stop strategy stays the default (Smart Turn), which waits through mid-thought pauses.
            user_turn_strategies=UserTurnStrategies(
                start=[PatientInterruptionStrategy(min_interrupt_secs=INTERRUPT_MIN_SECS)]
            ),
            # If an interruption yields no recognizable words, stay quiet instead of asking to repeat.
            empty_user_turn=EmptyUserTurnConfig(interrupted_prompt=None),
        ),
    )

    processors = [transport.input(), stt, UserPaceTracker(pace), BackchannelFilter(), aggregators.user()]
    if BACKCHANNELS_ENABLED:
        processors.append(
            BackchannelInjector(
                min_turn_secs=BACKCHANNEL_MIN_TURN_SECS,
                cooldown_secs=BACKCHANNEL_COOLDOWN_SECS,
                probability=BACKCHANNEL_PROBABILITY,
                phrases=BACKCHANNEL_PHRASES,
            )
        )
    processors += [llm, tts, transport.output(), aggregators.assistant()]
    pipeline = Pipeline(processors)

    # idle_timeout_secs=None: never shut down just because nobody is talking.
    worker = PipelineWorker(pipeline, params=PipelineParams(), idle_timeout_secs=None)

    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        # Deterministic holding line, so the bot never goes silent (or invents an answer) while
        # the bigger model works. Not for quick tools like the style switch.
        if any(fc.function_name == "ask_backend" for fc in function_calls):
            await worker.queue_frames(
                [TTSSpeakFrame(random.choice(HOLD_LINES), append_to_context=False)]
            )

    if GREETING:
        await worker.queue_frames([TTSSpeakFrame(GREETING)])

    if PLAY_TEST_TONE:
        await asyncio.to_thread(play_test_tone, OUTPUT_DEVICE_INDEX)

    runner = WorkerRunner()
    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())