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
    def complete(
        self, messages: list[dict], tools: list[dict], required_tool: str | None = None
    ) -> ProviderResponse:
        """Send the conversation and return either tool calls or plain text.

        required_tool names a tool the answer must call. A provider forces it
        where its API can; where it cannot, the request goes out as usual and the
        caller's own reminder in the conversation has to do the work. The tool
        list itself is never narrowed for it: on Claude a change of tool
        definitions invalidates the whole prompt cache.
        """
