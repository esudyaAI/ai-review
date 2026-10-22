import pytest

from reviewer.llm.usage import ModelUsage, UsageTracker, cost_for, report_lines

PRICING = {
    "gpt-5.4-mini-2026-03-17": {"input": 0.75, "output": 4.50},
    "gpt-5.5-2026-04-23": {"input": 5.00, "output": 30.00, "cached_input": 0.50},
}


def test_tracker_accumulates_per_model_across_multiple_records():
    tracker = UsageTracker()
    tracker.record("gpt-5.4-mini-2026-03-17", prompt_tokens=1000, completion_tokens=200)
    tracker.record("gpt-5.4-mini-2026-03-17", prompt_tokens=500, completion_tokens=100)
    tracker.record("gpt-5.5-2026-04-23", prompt_tokens=2000, completion_tokens=300, cached_tokens=1500)

    by_model = tracker.by_model()
    assert by_model["gpt-5.4-mini-2026-03-17"] == ModelUsage(prompt_tokens=1500, completion_tokens=300)
    assert by_model["gpt-5.5-2026-04-23"] == ModelUsage(
        prompt_tokens=2000, completion_tokens=300, cached_tokens=1500
    )


def test_cost_for_computes_input_output_and_cached_prices():
    usage = ModelUsage(prompt_tokens=2000, completion_tokens=300, cached_tokens=1500)
    # 500 non-cached input @ 5.00/M + 1500 cached @ 0.50/M + 300 output @ 30.00/M
    expected = (500 * 5.00 + 1500 * 0.50 + 300 * 30.00) / 1_000_000
    assert cost_for("gpt-5.5-2026-04-23", usage, PRICING) == expected


def test_cost_for_falls_back_to_input_price_when_no_cached_price_given():
    usage = ModelUsage(prompt_tokens=1000, completion_tokens=200, cached_tokens=400)
    # No "cached_input" entry for this model - cached tokens priced at the input rate.
    expected = (1000 * 0.75 + 200 * 4.50) / 1_000_000
    assert cost_for("gpt-5.4-mini-2026-03-17", usage, PRICING) == expected


def test_cost_for_returns_none_when_model_has_no_price_entry():
    usage = ModelUsage(prompt_tokens=1000, completion_tokens=200)
    assert cost_for("some-unpriced-model", usage, PRICING) is None


def test_cost_for_returns_none_when_price_entry_is_incomplete():
    usage = ModelUsage(prompt_tokens=1000, completion_tokens=200)
    assert cost_for("m", usage, {"m": {"input": 1.0}}) is None


def test_report_lines_includes_cost_for_priced_model():
    by_model = {"gpt-5.4-mini-2026-03-17": ModelUsage(prompt_tokens=1000, completion_tokens=200)}
    stdout_lines, stderr_warnings = report_lines(by_model, PRICING)
    assert len(stdout_lines) == 1
    assert "gpt-5.4-mini-2026-03-17" in stdout_lines[0]
    assert "1000" in stdout_lines[0]
    assert "200" in stdout_lines[0]
    assert "$" in stdout_lines[0]
    assert stderr_warnings == []


def test_report_lines_warns_and_omits_cost_for_unpriced_model():
    by_model = {"some-unpriced-model": ModelUsage(prompt_tokens=1000, completion_tokens=200)}
    stdout_lines, stderr_warnings = report_lines(by_model, PRICING)
    assert len(stdout_lines) == 1
    assert "some-unpriced-model" in stdout_lines[0]
    assert "1000" in stdout_lines[0]
    assert "$" not in stdout_lines[0]
    assert len(stderr_warnings) == 1
    assert "some-unpriced-model" in stderr_warnings[0]


# --- Usage breakdown by phase ------------------------------------------------
#
# Over four rounds on one merge request the input grew from 23,157 to 159,131
# tokens (measured 2026-08-16, jobbytime-go !27). The cause was never
# established: three candidates - a growing diff, tool results accumulating in
# the conversation, and thread text during adjudication - and a single number
# per run cannot tell them apart.
#
# These tests pin down the breakdown that can.


def test_tracker_splits_usage_by_phase():
    tracker = UsageTracker()
    tracker.record("m", prompt_tokens=1000, completion_tokens=100, phase="review")
    tracker.record("m", prompt_tokens=4000, completion_tokens=200, phase="review")
    tracker.record("m", prompt_tokens=500, completion_tokens=50, phase="adjudication")

    by_phase = tracker.by_phase()
    assert by_phase["review"] == ModelUsage(prompt_tokens=5000, completion_tokens=300)
    assert by_phase["adjudication"] == ModelUsage(prompt_tokens=500, completion_tokens=50)


def test_by_model_still_totals_across_phases():
    # by_model is what the cost line is built from; the phase split must not change it.
    tracker = UsageTracker()
    tracker.record("m", prompt_tokens=1000, completion_tokens=100, phase="review")
    tracker.record("m", prompt_tokens=500, completion_tokens=50, phase="adjudication")

    assert tracker.by_model()["m"] == ModelUsage(prompt_tokens=1500, completion_tokens=150)


def test_phase_defaults_to_review_for_callers_that_do_not_pass_one():
    tracker = UsageTracker()
    tracker.record("m", prompt_tokens=10, completion_tokens=1)

    assert set(tracker.by_phase()) == {"review"}


def test_report_lines_include_the_phase_breakdown():
    tracker = UsageTracker()
    tracker.record("m", prompt_tokens=9000, completion_tokens=100, phase="review")
    tracker.record("m", prompt_tokens=1000, completion_tokens=50, phase="adjudication")

    stdout_lines, _ = report_lines(
        tracker.by_model(), {"m": {"input": 1.0, "output": 1.0}}, by_phase=tracker.by_phase()
    )

    breakdown = [line for line in stdout_lines if "разбивка" in line]
    assert len(breakdown) == 1
    # Both phases, with the share that answers "where did the tokens go".
    assert "ревью 9000" in breakdown[0]
    assert "90%" in breakdown[0]
    assert "адъюдикация 1000" in breakdown[0]


def test_breakdown_is_omitted_when_there_is_nothing_to_split():
    tracker = UsageTracker()
    tracker.record("m", prompt_tokens=100, completion_tokens=10, phase="review")

    stdout_lines, _ = report_lines(
        tracker.by_model(), {"m": {"input": 1.0, "output": 1.0}}, by_phase=tracker.by_phase()
    )

    assert not [line for line in stdout_lines if "разбивка" in line]


class FakeLoopStats:
    iterations = 7
    tool_calls = 12
    tool_result_bytes = 48000


def test_loop_counters_are_reported_even_without_a_phase_split():
    """A run with no disputed threads has a single phase, and that is the common
    case - yet it is exactly the run whose iteration count and bytes read explain
    where the tokens went. Tying the counters to the phase breakdown hid them from
    every ordinary review. Found by review on !8."""
    tracker = UsageTracker()
    tracker.record("m", prompt_tokens=9000, completion_tokens=100, phase="review")

    stdout_lines, _ = report_lines(
        tracker.by_model(),
        {"m": {"input": 1.0, "output": 1.0}},
        by_phase=tracker.by_phase(),
        loop_stats=FakeLoopStats(),
    )

    loop_line = [line for line in stdout_lines if "итераций" in line]
    assert len(loop_line) == 1
    assert "итераций ревью 7" in loop_line[0]
    assert "вызовов инструментов 12" in loop_line[0]
    assert "48000" in loop_line[0]


def test_phase_breakdown_and_loop_counters_are_separate_lines():
    tracker = UsageTracker()
    tracker.record("m", prompt_tokens=9000, completion_tokens=100, phase="review")
    tracker.record("m", prompt_tokens=1000, completion_tokens=50, phase="adjudication")

    stdout_lines, _ = report_lines(
        tracker.by_model(),
        {"m": {"input": 1.0, "output": 1.0}},
        by_phase=tracker.by_phase(),
        loop_stats=FakeLoopStats(),
    )

    assert len([line for line in stdout_lines if "разбивка" in line]) == 1
    assert len([line for line in stdout_lines if "итераций" in line]) == 1


def test_loop_stats_are_summed_across_passes():
    from reviewer.llm.loop import LoopStats
    from reviewer.llm.usage import merge_loop_stats

    merged = merge_loop_stats([
        LoopStats(iterations=6, tool_calls=18, tool_result_bytes=1000),
        LoopStats(iterations=9, tool_calls=23, tool_result_bytes=2500),
    ])
    assert (merged.iterations, merged.tool_calls, merged.tool_result_bytes) == (15, 41, 3500)


def test_loop_stats_of_nothing_is_none():
    from reviewer.llm.usage import merge_loop_stats

    assert merge_loop_stats([]) is None


# --- Cache writes (Anthropic) ------------------------------------------------
#
# Anthropic bills a cache write at 1.25x the input price and reports it apart
# from cache reads. Both are part of the prompt; only reads are discounted.

CLAUDE_PRICING = {
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cached_input": 0.10, "cache_write": 2.50},
}


def test_tracker_accumulates_cache_writes():
    tracker = UsageTracker()
    tracker.record("claude-sonnet-5-5", 1000, 10, cached_tokens=600, cache_write_tokens=300)
    tracker.record("claude-sonnet-5-5", 1000, 10, cached_tokens=900, cache_write_tokens=50)
    usage = tracker.by_model()["claude-sonnet-5-5"]
    assert usage.cache_write_tokens == 350
    assert usage.cached_tokens == 1500


def test_cost_for_prices_cache_writes_at_their_own_rate():
    usage = ModelUsage(prompt_tokens=1000, completion_tokens=100, cached_tokens=600, cache_write_tokens=300)
    # 100 plain input @ 2.00 + 600 read @ 0.10 + 300 written @ 2.50 + 100 out @ 10.00
    expected = (100 * 2.00 + 600 * 0.10 + 300 * 2.50 + 100 * 10.00) / 1_000_000
    assert cost_for("claude-sonnet-5-5", usage, CLAUDE_PRICING) == pytest.approx(expected)


def test_cost_for_prices_cache_writes_at_input_rate_without_a_write_price():
    usage = ModelUsage(prompt_tokens=1000, completion_tokens=0, cached_tokens=0, cache_write_tokens=400)
    assert cost_for("m", usage, {"m": {"input": 1.0, "output": 1.0}}) == pytest.approx(1000 / 1_000_000)


def test_report_line_names_cache_writes_only_when_there_are_some():
    with_writes = {"claude-sonnet-5-5": ModelUsage(1000, 10, cached_tokens=600, cache_write_tokens=300)}
    without = {"gpt-5.4-mini-2026-03-17": ModelUsage(prompt_tokens=1000, completion_tokens=200)}

    claude_line = report_lines(with_writes, CLAUDE_PRICING)[0][0]
    openai_line = report_lines(without, PRICING)[0][0]

    assert "запись в кеш 300" in claude_line
    # The OpenAI line is read by people and by the cost measurement scripts;
    # it does not change shape because another provider exists.
    assert "запись в кеш" not in openai_line
