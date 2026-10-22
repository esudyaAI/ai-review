"""Tell Better Stack how Claude fared in a run switched to it.

The OpenAI fallback keeps merge requests moving, and for the same reason nobody
would notice it: the pipeline stays green. An exhausted prepaid budget or an
expired key is exactly what needs a person, so a switched run reports to a
Better Stack heartbeat - the channel the project already uses for host alerts,
which Better Stack forwards to Slack:

* fell back to OpenAI  -> POST <url>/fail, reason in a text/plain body;
* Claude answered      -> GET <url>, which resolves an open incident;
* Claude never asked   -> nothing (a docs-only merge request has no news).

Both request forms were checked live on 2026-08-26 (Регламент, «Живость
алертов»), including the trap that made the disk alert silent for weeks: a body
with "%" sent as form data is a 400, as text/plain it is accepted.

Never raises and never prints the URL: it carries the heartbeat's token.
"""

from __future__ import annotations

import sys
from typing import Callable, Mapping

import httpx

from reviewer.llm.fallback import FallbackSwitch

MAX_REASON = 800
TIMEOUT = 10.0


def _failure_text(switch: FallbackSwitch, env: Mapping[str, str]) -> str:
    reason = switch.reason
    if len(reason) > MAX_REASON:
        reason = reason[:MAX_REASON] + "…"
    project = env.get("CI_PROJECT_PATH", "?")
    mr = env.get("CI_MERGE_REQUEST_IID", "?")
    job = env.get("CI_JOB_URL", "")
    return (
        f"ai-review: Claude недоступен, ревью продолжено на OpenAI.\n"
        f"Причина: {reason}\n"
        f"{project} !{mr} {job}\n"
        f"Кончились кредиты или истёк ключ — очистить AI_REVIEW_PROVIDER на группе "
        f"или заменить ANTHROPIC_API_KEY."
    )


def report_provider_health(
    url: str,
    switch: FallbackSwitch,
    env: Mapping[str, str],
    post: Callable[..., httpx.Response] = httpx.post,
    get: Callable[..., httpx.Response] = httpx.get,
) -> None:
    if not url:
        return
    try:
        if switch.tripped:
            response = post(
                url.rstrip("/") + "/fail",
                content=_failure_text(switch, env).encode("utf-8"),
                headers={"Content-Type": "text/plain; charset=utf-8"},
                timeout=TIMEOUT,
            )
            what = "сбой"
        elif switch.primary_answered:
            response = get(url, timeout=TIMEOUT)
            what = "норма"
        else:
            return
    except Exception as exc:  # noqa: BLE001 - an alert must never fail the review
        print(f"AI review: сигнал в Better Stack не отправлен ({type(exc).__name__})", file=sys.stderr)
        return
    if response.status_code >= 300:
        print(
            f"AI review: сигнал в Better Stack ({what}) отклонён, HTTP {response.status_code}",
            file=sys.stderr,
        )
