"""Alert when a run switched to Claude had to fall back to OpenAI.

The fallback keeps merge requests moving, which is also why nobody would
notice it: the pipeline stays green. Running out of the prepaid Claude budget
or an expired key is exactly what needs a human, so the run tells Better Stack,
through the heartbeat channel the project already uses (POST <url>/fail with a
text/plain body; a plain GET resolves the incident - both checked live on
2026-08-26, Регламент, «Живость алертов»).
"""

import httpx

from reviewer.alert import report_provider_health
from reviewer.llm.fallback import FallbackSwitch

URL = "https://uptime.betterstack.com/api/v1/heartbeat/TOKEN"
ENV = {
    "CI_PROJECT_PATH": "jobbytime.com-group/jobbytime-go",
    "CI_MERGE_REQUEST_IID": "512",
    "CI_JOB_URL": "https://gitlab.com/jobbytime.com-group/jobbytime-go/-/jobs/1",
}


class Recorder:
    def __init__(self, response=None, error=None):
        self.calls = []
        self.response = response if response is not None else httpx.Response(200)
        self.error = error

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.error is not None:
            raise self.error
        return self.response


def tripped(reason="AuthenticationError: invalid x-api-key"):
    switch = FallbackSwitch()
    switch.tripped = True
    switch.reason = reason
    return switch


def answered():
    switch = FallbackSwitch()
    switch.primary_answered = True
    return switch


def test_a_fallback_reports_a_failure_with_the_reason_and_the_merge_request():
    post, get = Recorder(), Recorder()
    report_provider_health(URL, tripped(), ENV, post=post, get=get)

    assert get.calls == []
    url, kwargs = post.calls[0]
    assert url == URL + "/fail"
    # Without text/plain a "%" in the body broke the disk alert for weeks.
    assert kwargs["headers"]["Content-Type"].startswith("text/plain")
    body = kwargs["content"].decode("utf-8")
    assert "invalid x-api-key" in body
    assert "jobbytime-go" in body and "!512" in body
    assert ENV["CI_JOB_URL"] in body
    assert "AI_REVIEW_PROVIDER" in body


def test_a_run_where_claude_answered_resolves_the_incident():
    post, get = Recorder(), Recorder()
    report_provider_health(URL, answered(), ENV, post=post, get=get)
    assert post.calls == []
    assert get.calls[0][0] == URL


def test_a_run_that_never_asked_claude_says_nothing():
    # A docs-only merge request makes no model call: no news about Claude.
    post, get = Recorder(), Recorder()
    report_provider_health(URL, FallbackSwitch(), ENV, post=post, get=get)
    assert post.calls == [] and get.calls == []


def test_without_a_url_nothing_is_sent():
    post, get = Recorder(), Recorder()
    report_provider_health("", tripped(), ENV, post=post, get=get)
    assert post.calls == [] and get.calls == []


def test_a_failed_alert_never_fails_the_run_and_never_prints_the_url(capsys):
    post = Recorder(error=httpx.ConnectError("boom"))
    report_provider_health(URL, tripped(), ENV, post=post, get=Recorder())
    err = capsys.readouterr().err
    assert "ConnectError" in err
    assert "TOKEN" not in err  # the heartbeat URL carries its token


def test_a_rejected_alert_is_named_in_the_log(capsys):
    post = Recorder(response=httpx.Response(400))
    report_provider_health(URL, tripped(), ENV, post=post, get=Recorder())
    err = capsys.readouterr().err
    assert "400" in err
    assert "TOKEN" not in err


def test_a_long_reason_is_cut():
    post = Recorder()
    report_provider_health(URL, tripped("x" * 10_000), ENV, post=post, get=Recorder())
    assert len(post.calls[0][1]["content"]) < 2_000
