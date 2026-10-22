"""Continue a run on another provider when the first one is unavailable.

Written for the prepaid Claude budget: when it runs out, or the key expires or
is revoked, the reviewer must not turn every merge request in the group red -
it is a blocking gate. The run moves to the repository's own OpenAI
configuration, says so once in the job log, and stays there.

The conversation itself needs no translation: the loop keeps it in OpenAI's
format, which the fallback speaks natively.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

from reviewer.llm.base import LLMProvider, ProviderResponse


@dataclass
class FallbackSwitch:
    """Shared by every provider of one run: review halves and adjudication.

    Once the primary is known to be unavailable, nobody asks it again - each
    attempt would cost a round trip and, after the SDK's own retries, seconds.
    """

    tripped: bool = False
    reason: str = ""


class FallbackProvider:
    def __init__(
        self,
        primary: LLMProvider,
        make_fallback: Callable[[], LLMProvider | None] | None,
        switch: FallbackSwitch,
        errors: tuple[type[BaseException], ...],
        fallback_label: str,
    ) -> None:
        self.primary = primary
        self._make_fallback = make_fallback
        self._switch = switch
        # Only these trip the switch. Anything else - a TypeError of ours in
        # message conversion - must surface, not be reviewed by someone else.
        self._errors = errors
        self._fallback_label = fallback_label
        self._fallback: LLMProvider | None = None

    def build_fallback(self) -> LLMProvider | None:
        if self._fallback is None and self._make_fallback is not None:
            self._fallback = self._make_fallback()
        return self._fallback

    def complete(self, messages: list[dict], tools: list[dict]) -> ProviderResponse:
        if self._switch.tripped:
            fallback = self.build_fallback()
            if fallback is not None:
                return fallback.complete(messages, tools)
        try:
            return self.primary.complete(messages, tools)
        except self._errors as exc:
            fallback = self.build_fallback()
            if fallback is None:
                raise
            if not self._switch.tripped:
                self._switch.tripped = True
                self._switch.reason = f"{type(exc).__name__}: {exc}"
                print(
                    f"AI review: Claude недоступен ({self._switch.reason}), "
                    f"ревью продолжено на OpenAI ({self._fallback_label})",
                    file=sys.stderr,
                )
            return fallback.complete(messages, tools)
