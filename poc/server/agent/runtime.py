from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ..providers.base import ChatProvider
from .prompt import SYSTEM_PROMPT
from .tools import ToolRegistry


class AgentRuntime:
    """Owns conversation context and exposes normalized model stream events."""

    def __init__(self, provider: ChatProvider, tools: ToolRegistry) -> None:
        self.provider = provider
        self.tools = tools
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def respond(self, transcript: str) -> AsyncIterator[dict[str, Any]]:
        self.messages.append({"role": "user", "content": transcript})
        async for chunk in self.provider.stream_reply(self.messages, self.tools.schemas):
            yield chunk

        # The turn coordinator parses accumulated content/tool calls, executes
        # approved tools, and appends the canonical assistant message here.
