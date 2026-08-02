"""OpenAI and OpenAI-compatible provider."""

from __future__ import annotations

import json
import sys

from openai import BadRequestError, OpenAI

from reviewer.llm.base import ProviderResponse, ToolCall


class OpenAIProvider:
    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 120.0) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=3)
        self._model = model

    def complete(self, messages: list[dict], tools: list[dict]) -> ProviderResponse:
        # A plain-text answer has no meaning in our protocol: the model must always
        # either call a read tool or submit_findings. tool_choice="required" enforces
        # that at the API level instead of relying on the model to follow the prompt.
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools,
                tool_choice="required",
            )
        except BadRequestError as exc:
            # Not every OpenAI-compatible backend supports tool_choice="required" -
            # fall back to "auto" rather than failing the whole review over it.
            print(
                f"AI review: провайдер отклонил tool_choice='required' ({exc}), "
                "повтор с 'auto'",
                file=sys.stderr,
            )
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
