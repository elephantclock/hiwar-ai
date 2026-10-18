import io
import unittest
import wave

from server.audio import duration_ms, pcm16_to_wav


class AudioTest(unittest.TestCase):
    def test_duration_for_one_twentieth_second_frame(self) -> None:
        pcm = bytes(320 * 2)
        self.assertEqual(duration_ms(pcm), 20)

    def test_wav_wrapper_preserves_contract(self) -> None:
        pcm = bytes(range(64)) * 10
        with wave.open(io.BytesIO(pcm16_to_wav(pcm)), "rb") as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getframerate(), 16_000)
            self.assertEqual(wav.readframes(wav.getnframes()), pcm)


if __name__ == "__main__":
    unittest.main()
