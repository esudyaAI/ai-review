"""OpenAI and OpenAI-compatible provider."""

from __future__ import annotations

import json

from openai import OpenAI

from reviewer.llm.base import ProviderResponse, ToolCall


class OpenAIProvider:
    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 120.0) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=3)
        self._model = model

    def complete(self, messages: list[dict], tools: list[dict]) -> ProviderResponse:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        message = response.choices[0].message
        calls = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            calls.append(ToolCall(id=call.id, name=call.function.name, arguments=arguments))
        return ProviderResponse(tool_calls=tuple(calls), text=message.content)
