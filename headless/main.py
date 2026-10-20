"""Main CLI entry point for the Local Voice Conversational Companion.

Usage:
    uv run main.py                                    # Kokoro TTS, system defaults
    uv run main.py --list-devices                     # List microphones & speakers
    uv run main.py --input-device 0 --output-device 1
    uv run main.py --tts-engine orpheus --tts-voice tara
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import ollama
from loguru import logger

from natural import CompanionConfig, list_audio_devices, run_companion


def print_banner(config: CompanionConfig) -> None:
    """Prints a terminal banner with active configuration."""
    print("=" * 66)
    print(" 🎙️  LOCAL CONVERSATIONAL VOICE COMPANION")
    print("=" * 66)
    print(f" • Conversational LLM : {config.chat_model}")
    print(f" • Backend Reasoner   : {config.backend_model}")
    print(f" • Speech-to-Text     : MLX Whisper (large-v3-turbo-q4)")
    if config.tts_engine == "orpheus":
        print(f" • Text-to-Speech     : Orpheus 3B ({config.tts_voice}) at {config.tts_base_url}")
    else:
        print(f" • Text-to-Speech     : Kokoro ({config.tts_voice}, speed: {config.tts_speed}x)")
    print(f" • Turn Analyzer      : Smart Turn v3 (ONNX, local)")
    print(f" • Interruption Guard : Min {config.min_interruption_words} words + laughter filter")
    print(f" • Backchannel Filter : Stray token → UserBackchannelFrame")
    print(f" • Thinking Filler    : BotBackchannelFrame during LLM TTFT")
    print(f" • VAD Stop Threshold : {config.vad_stop_secs}s")
    print(f" • Input Device       : {config.input_device_index if config.input_device_index is not None else 'Default'}")
    print(f" • Output Device      : {config.output_device_index if config.output_device_index is not None else 'Default'}")
    print("-" * 66)
    print(" 💡 Use headphones to prevent microphone echo/feedback.")
    print(" 💡 Press Ctrl+C to exit.")
    print("=" * 66 + "\n")


def check_ollama_status(chat_model: str, backend_model: str) -> bool:
    """Verifies Ollama is reachable and models are available."""
    client = ollama.Client()
    try:
        response = client.list()
        available = [m.model for m in response.models]
    except Exception as e:
        logger.error(f"Cannot connect to Ollama: {e}\nRun: ollama serve")
        return False

    missing = [m for m in (chat_model, backend_model)
               if not any(m in a or a in m for a in available)]
    if missing:
        logger.warning(
            f"Missing model(s): {missing}\n"
            f"Run: {' && '.join(f'ollama pull {m}' for m in missing)}"
        )
        return False
    return True


def display_devices() -> None:
    """Prints available audio devices."""
    devices = list_audio_devices()
    print("\nAvailable Audio Devices:")
    print(f"{'Index':<6} {'Name':<36} {'In':<5} {'Out':<5} {'Rate'}")
    print("-" * 62)
    for d in devices:
        print(f"[{d['index']}]   {d['name'][:34]:<36} {d['inputs']:<5} {d['outputs']:<5} {d['sample_rate']} Hz")
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Local voice companion: Pipecat + MLX Whisper + Ollama + Kokoro/Orpheus."
    )
    p.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    p.add_argument("--input-device", type=int, default=None, help="PyAudio input device index.")
    p.add_argument("--output-device", type=int, default=None, help="PyAudio output device index.")
    p.add_argument("--chat-model", default="qwen2.5:14b", help="Ollama conversational model.")
    p.add_argument("--backend-model", default="qwen2.5:32b", help="Ollama backend reasoning model.")
    p.add_argument("--tts-engine", choices=["kokoro", "orpheus"], default="kokoro",
                   help="'kokoro' (local, fast) or 'orpheus' (emotional 3B Speech-LLM).")
    p.add_argument("--tts-url", default="http://localhost:8000/v1",
                   help="Endpoint for Orpheus TTS.")
    p.add_argument("--tts-voice", default=None,
                   help="TTS voice (default: af_heart for Kokoro, tara for Orpheus).")
    p.add_argument("--speed", type=float, default=1.0, help="TTS speed multiplier.")
    p.add_argument("--min-words", type=int, default=2,
                   help="Min words to interrupt the bot while speaking.")
    p.add_argument("--no-tone", action="store_true", help="Skip startup beep.")
    p.add_argument("--no-greeting", action="store_true", help="Skip spoken greeting.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_devices:
        display_devices()
        sys.exit(0)

    voice = args.tts_voice or ("tara" if args.tts_engine == "orpheus" else "af_heart")

    config = CompanionConfig(
        chat_model=args.chat_model,
        backend_model=args.backend_model,
        tts_engine=args.tts_engine,
        tts_base_url=args.tts_url,
        tts_voice=voice,
        tts_speed=args.speed,
        min_interruption_words=args.min_words,
        input_device_index=args.input_device,
        output_device_index=args.output_device,
        play_test_tone=not args.no_tone,
        greeting=None if args.no_greeting else "Hi, I'm ready. Just start talking whenever you like.",
    )

    print_banner(config)

    logger.info("Pre-flight checks...")
    if not check_ollama_status(config.chat_model, config.backend_model):
        sys.exit(1)

    try:
        asyncio.run(run_companion(config))
    except KeyboardInterrupt:
        print("\n\n👋 Companion shut down. See you next time!")
    except Exception as e:
        logger.error(f"Fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
