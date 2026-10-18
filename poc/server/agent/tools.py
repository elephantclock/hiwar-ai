from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any


ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


LOOKUP_ORDER_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "lookup_order",
        "description": "Look up an order after the caller provides its order number.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "pattern": "^ORD-[0-9]{4}$",
                    "description": "Order identifier such as ORD-1001.",
                }
            },
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
}


class ToolRegistry:
    def __init__(self, fixture_path: Path | None = None) -> None:
        self.fixture_path = fixture_path or (
            Path(__file__).resolve().parents[2] / "fixtures" / "orders.json"
        )
        self.schemas = [LOOKUP_ORDER_SCHEMA]
        self._handlers: dict[str, ToolHandler] = {"lookup_order": self.lookup_order}

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handler = self._handlers.get(name)
        if handler is None:
            raise ValueError(f"unknown tool: {name}")
        return await handler(arguments)

    async def lookup_order(self, arguments: dict[str, Any]) -> dict[str, Any]:
        order_id = str(arguments.get("order_id", "")).upper()
        records = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        order = next((item for item in records if item["order_id"] == order_id), None)
        if order is None:
            return {"found": False, "order_id": order_id}
        return {"found": True, "order": order}
