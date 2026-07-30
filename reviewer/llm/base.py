"""Provider-agnostic contract for the review loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class ProviderResponse:
    tool_calls: tuple[ToolCall, ...]
    text: str | None


class LLMProvider(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> ProviderResponse:
        """Send the conversation and return either tool calls or plain text."""
