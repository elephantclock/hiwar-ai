"""Hear every Kokoro English voice reading the same lines, then pick one for bot.py.

Run it in the same venv as the bot (it reuses the model files Pipecat already downloaded):

    python voice_audition.py                 # all English voices, default speed, default output
    python voice_audition.py 1               # play on PyAudio output device index 1
    python voice_audition.py 1 0.93          # ...at speed 0.93 (slightly slower often sounds calmer)
    python voice_audition.py 1 0.93 af_nicole am_michael   # only these voices

Press Ctrl+C to stop. Note the voice name you like and put it in KOKORO_VOICE in bot.py.
"""

import sys
from pathlib import Path

import numpy as np
import pyaudio
from kokoro_onnx import Kokoro

CACHE = Path.home() / ".cache" / "pipecat" / "kokoro-onnx"
LINES = (
    "Hmm, that's a good question... let me think for a second. "
    "Honestly? I'd probably start small, and see how it feels."
)


def main() -> None:
    out_index = int(sys.argv[1]) if len(sys.argv) > 1 else None
    speed = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    only = sys.argv[3:]

    model, voices = CACHE / "kokoro-v1.0.onnx", CACHE / "voices-v1.0.bin"
    if not (model.exists() and voices.exists()):
        sys.exit(f"Kokoro files not found in {CACHE}. Run bot.py once so Pipecat downloads them.")

    kokoro = Kokoro(str(model), str(voices))
    names = sorted(v for v in kokoro.get_voices() if v[:2] in ("af", "am", "bf", "bm"))
    if only:
        names = [n for n in names if n in only]
    if not names:
        sys.exit("No matching voices.")

    p = pyaudio.PyAudio()
    stream = None
    try:
        for name in names:
            lang = "en-gb" if name.startswith("b") else "en-us"
            samples, rate = kokoro.create(LINES, voice=name, speed=speed, lang=lang)
            if stream is None:
                stream = p.open(
                    format=pyaudio.paFloat32,
                    channels=1,
                    rate=rate,
                    output=True,
                    output_device_index=out_index,
                )
            print(f"Playing {name} (speed {speed})")
            stream.write(np.asarray(samples, dtype=np.float32).tobytes())
    except KeyboardInterrupt:
        pass
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        p.terminate()


if __name__ == "__main__":
    main()
