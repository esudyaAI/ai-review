"""Structural checks on the template consumers include.

The template is an interface, like code: jobbytime-go includes
`templates/ai-review.yml` by `ref: v1`, so a typo in it breaks somebody else's
pipeline rather than ours. What is checked here is exactly what has to hold for
the reviewer's exit codes to mean anything in a consuming pipeline.
"""

from pathlib import Path

import pytest
import yaml

from reviewer.__main__ import EXIT_FINDINGS, EXIT_TECHNICAL, EXIT_TOO_LARGE

TEMPLATE = Path(__file__).resolve().parents[2] / "templates" / "ai-review.yml"


@pytest.fixture(scope="module")
def job() -> dict:
    # The template holds two YAML documents: the spec with inputs, and the jobs.
    # The second one is the one we want.
    documents = [d for d in yaml.safe_load_all(TEMPLATE.read_text(encoding="utf-8")) if d]
    body = documents[-1]
    assert "ai-review" in body, f"the ai-review job vanished from the template: {sorted(body)}"
    return body["ai-review"]


def test_only_a_technical_failure_is_retried(job):
    """A retry is for exactly one class of failure: the review did not happen for a
    technical reason - network, iteration cap, a failed start. One retry, because a
    second failure is not a blip but something to fix by hand."""
    retry = job.get("retry")
    assert retry, "ai-review has no retry: a technical failure needs a manual restart"
    assert retry["max"] == 1, retry
    assert retry["exit_codes"] == [EXIT_TECHNICAL], retry


def test_findings_are_never_retried(job):
    """Re-running a review that found something costs a second review and changes
    nothing: findings are a function of the diff, not of the attempt."""
    assert EXIT_FINDINGS not in job["retry"]["exit_codes"]


def test_an_oversized_diff_is_never_retried(job):
    """A diff over the limit is still over the limit on the second run. That is fixed
    by splitting the merge request, not by restarting the job."""
    assert EXIT_TOO_LARGE not in job["retry"]["exit_codes"]


def test_the_job_still_blocks_by_default_choice_of_the_consumer(job):
    """allow_failure stays an input of the template rather than a baked-in value:
    the including repository decides whether the job blocks, and that decision is
    visible in its own .gitlab-ci.yml instead of hidden here."""
    assert job["allow_failure"] == "$[[ inputs.allow_failure ]]"
