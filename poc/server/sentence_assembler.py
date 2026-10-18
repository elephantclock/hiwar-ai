from __future__ import annotations

import re


_BOUNDARY = re.compile(r"[.!?](?:\s+|$)")


class SentenceAssembler:
    """Collect text deltas into TTS-sized, sentence-aware chunks."""

    def __init__(self, max_chars: int = 200) -> None:
        if max_chars < 20:
            raise ValueError("max_chars must be at least 20")
        self.max_chars = max_chars
        self._buffer = ""

    def push(self, delta: str) -> list[str]:
        self._buffer += delta
        ready: list[str] = []

        while match := _BOUNDARY.search(self._buffer):
            sentence_end = match.start() + 1
            sentence = self._buffer[:sentence_end].strip()
            if sentence:
                ready.append(sentence)
            self._buffer = self._buffer[match.end():]

        while len(self._buffer) > self.max_chars:
            split_at = self._buffer.rfind(" ", 0, self.max_chars + 1)
            split_at = split_at if split_at > 0 else self.max_chars
            ready.append(self._buffer[:split_at].strip())
            self._buffer = self._buffer[split_at:].lstrip()
        return ready

    def flush(self) -> list[str]:
        final = self._buffer.strip()
        self._buffer = ""
        return [final] if final else []
